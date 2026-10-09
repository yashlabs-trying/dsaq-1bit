"""E16c: exact segmented-CE search over selective joint residual codes."""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, codec_state, decode_weights, unpack_uint
from e14b_neuron_joint_ce import ce_gradient, exact_ce_segments


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "packed/E14c_joint_ce_consistent_005.packed"
BASE = Path(os.environ.get("E16C_BASE", ROOT / "packed/E16b_selective_joint_zero_base.packed"))
CACHE = ROOT / "cache/e9_layer_cache.pt"
LABEL = os.environ.get("E16C_LABEL", "E16c_functional_joint_proof")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
GROUPS = int(os.environ.get("E16C_GROUPS", "2"))
GROUP_OFFSET = int(os.environ.get("E16C_GROUP_OFFSET", "0"))
CHUNKS = int(os.environ.get("E16C_CHUNKS", "2"))
CANDIDATES = int(os.environ.get("E16C_CANDIDATES", "8"))
THREADS = int(os.environ.get("E16C_THREADS", "4"))
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_base(path):
    blob = path.read_bytes()
    ml = struct.unpack_from("<I",blob,0)[0]
    meta = json.loads(blob[4:4+ml])
    pos = 4+ml+meta["compressed_parent_bytes"]
    count, bits = meta["critical_count"], meta["critical_index_bits"]
    index_bytes = (count*bits+7)//8
    indices = unpack_uint(blob[pos:pos+index_bytes],count,bits)
    pos += index_bytes
    ids_offset = pos
    ids = torch.from_numpy(np.frombuffer(blob,dtype=np.uint8,count=count,offset=pos).copy()).long()
    pos += count
    k = meta["correction_k"]
    q = torch.from_numpy(np.frombuffer(blob,dtype=np.int8,count=k*48,offset=pos).copy()).view(k,48).float()
    pos += k*48
    scales = torch.from_numpy(np.frombuffer(blob,dtype="<f2",count=k,offset=pos).copy()).float()
    return bytearray(blob),meta,indices,ids,ids_offset,q*scales[:,None]


def main():
    torch.set_num_threads(THREADS); torch.set_num_interop_threads(1)
    started=time.time()
    blob,meta,critical_indices,correction_ids,ids_offset,codebook=parse_base(BASE)
    position={int(p):i for i,p in enumerate(critical_indices.tolist())}
    cache=torch.load(CACHE,map_location="cpu",weights_only=True)
    weights,_=decode_weights(BASE)
    parent_state=codec_state(PARENT)
    original=cache["original_weights"]
    x_a,x_b=cache["x_a"],cache["x_b"]
    gate_a,gate_b=x_a@weights["gate_proj"].T,x_b@weights["gate_proj"].T
    up_a,up_b=x_a@weights["up_proj"].T,x_b@weights["up_proj"].T
    z_a,z_b=F.silu(gate_a)*up_a,F.silu(gate_b)*up_b
    h2_a=cache["h1_a"]+z_a@weights["down_proj"].T
    h2_b=cache["h1_b"]+z_b@weights["down_proj"].T
    print(json.dumps({"event":"load_model"}),flush=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float32,
        low_cpu_mem_usage=True,attn_implementation="eager").eval()
    grad_a,ce_a=ce_gradient(model,h2_a,cache["token_ids_a"])
    grad_b,ce_b=ce_gradient(model,h2_b,cache["token_ids_b"])
    seg_a=exact_ce_segments(model,h2_a,cache["token_ids_a"])
    seg_b=exact_ce_segments(model,h2_b,cache["token_ids_b"])
    initial={"CE_A":ce_a,"CE_B":ce_b,"segments_A":seg_a,"segments_B":seg_b}

    zref_a=F.silu(x_a@original["gate_proj"].T)*(x_a@original["up_proj"].T)
    zref_b=F.silu(x_b@original["gate_proj"].T)*(x_b@original["up_proj"].T)
    damage=((grad_a@weights["down_proj"]*(z_a-zref_a)).abs().mean(0)
           +(grad_b@weights["down_proj"]*(z_b-zref_b)).abs().mean(0)).view(-1,4).mean(1)
    top_groups=torch.topk(damage,GROUP_OFFSET+GROUPS).indices.tolist()[GROUP_OFFSET:]
    hidden_chunks=meta["hidden_chunks"]; neuron_groups=meta["neuron_groups"]
    slices=parent_state["slices"]
    g_start=slices["gate_proj"][0]; u_start=slices["up_proj"][0]; d_start=slices["down_proj"][0]
    orig_blocks={n:blockify(original[n]) for n in ("gate_proj","up_proj","down_proj")}
    parent_blocks={n:blockify(weights[n]) for n in ("gate_proj","up_proj","down_proj")}
    accepted=exact_evals=0; changed=[]

    for gi,group in enumerate(top_groups,1):
        # Rank chunks by the norm of the exact normalized BF16 correction target.
        scores=[]
        for chunk in range(hidden_chunks):
            p=group*hidden_chunks+chunk; di=chunk*neuron_groups+group
            scores.append(float((orig_blocks["gate_proj"][p]-parent_blocks["gate_proj"][p]).square().sum()
                         +(orig_blocks["up_proj"][p]-parent_blocks["up_proj"][p]).square().sum()
                         +(orig_blocks["down_proj"][di]-parent_blocks["down_proj"][di]).square().sum()))
        selected=torch.topk(torch.tensor(scores),CHUNKS).indices.tolist()
        rows=slice(group*4,group*4+4)
        for chunk in selected:
            p=group*hidden_chunks+chunk
            if p not in position: continue
            posi=position[p]; current=int(correction_ids[posi]); cols=slice(chunk*4,chunk*4+4)
            outrows=cols; di=chunk*neuron_groups+group
            gidx=g_start+p; uidx=u_start+p; didx=d_start+di
            target=torch.cat(((orig_blocks["gate_proj"][p]-parent_blocks["gate_proj"][p])/parent_state["scales"][gidx],
                              (orig_blocks["up_proj"][p]-parent_blocks["up_proj"][p])/parent_state["scales"][uidx],
                              (orig_blocks["down_proj"][di]-parent_blocks["down_proj"][di])/parent_state["scales"][didx]))
            choices=torch.topk((codebook-target).square().sum(1),CANDIDATES,largest=False).indices.tolist()
            if 0 not in choices: choices.append(0)
            old_columns=weights["down_proj"][:,rows].clone()
            best=None; best_a=ce_a
            for cid in choices:
                if cid==current: continue
                exact_evals+=1
                delta=codebook[cid]-codebook[current]
                dg=delta[:16].view(4,4)*parent_state["scales"][gidx]
                du=delta[16:32].view(4,4)*parent_state["scales"][uidx]
                dd=delta[32:].view(4,4)*parent_state["scales"][didx]
                nga=gate_a[:,rows]+x_a[:,cols]@dg.T; ngb=gate_b[:,rows]+x_b[:,cols]@dg.T
                nua=up_a[:,rows]+x_a[:,cols]@du.T; nub=up_b[:,rows]+x_b[:,cols]@du.T
                nza=F.silu(nga)*nua; nzb=F.silu(ngb)*nub
                dza,dzb=nza-z_a[:,rows],nzb-z_b[:,rows]
                cross_a=nza@dd.T; cross_b=nzb@dd.T
                cha=h2_a+dza@old_columns.T; chb=h2_b+dzb@old_columns.T
                cha[:,outrows]+=cross_a; chb[:,outrows]+=cross_b
                tsa=exact_ce_segments(model,cha,cache["token_ids_a"]); ta=sum(tsa)/len(tsa)
                if ta>=best_a-1e-10 or any(n>o for n,o in zip(tsa,seg_a)): continue
                tsb=exact_ce_segments(model,chb,cache["token_ids_b"]); tb=sum(tsb)/len(tsb)
                if tb<=ce_b+1e-7 and not any(n>o for n,o in zip(tsb,seg_b)):
                    best_a=ta; best=(cid,dg,du,dd,nga,ngb,nua,nub,nza,nzb,cha,chb,tsa,tsb,tb)
            if best:
                cid,dg,du,dd,nga,ngb,nua,nub,nza,nzb,h2_a,h2_b,seg_a,seg_b,ce_b=best
                ce_a=best_a; gate_a[:,rows]=nga; gate_b[:,rows]=ngb; up_a[:,rows]=nua; up_b[:,rows]=nub
                z_a[:,rows]=nza; z_b[:,rows]=nzb
                weights["gate_proj"][rows,cols]+=dg; weights["up_proj"][rows,cols]+=du
                weights["down_proj"][outrows,rows]+=dd
                correction_ids[posi]=cid; accepted+=1; changed.append(p)
        print(json.dumps({"event":"progress","group":gi,"groups":GROUPS,"accepted":accepted,
                          "CE_A":ce_a,"CE_B":ce_b,"segments_A":seg_a,"segments_B":seg_b}),flush=True)

    blob[ids_offset:ids_offset+len(correction_ids)]=correction_ids.numpy().astype(np.uint8).tobytes()
    OUTPUT.write_bytes(blob)
    result={"experiment":LABEL,"base":str(BASE),"base_sha256":sha256(BASE),
            "groups":GROUPS,"group_offset":GROUP_OFFSET,
            "chunks_per_group":CHUNKS,"candidates":CANDIDATES,
            "accepted":accepted,"changed_packages":changed,"exact_evaluations":exact_evals,
            "initial":initial,"final":{"CE_A":ce_a,"CE_B":ce_b,"segments_A":seg_a,"segments_B":seg_b},
            "bytes":OUTPUT.stat().st_size,"bpw":OUTPUT.stat().st_size*8/meta["weights"],
            "output_sha256":sha256(OUTPUT),"elapsed_seconds":time.time()-started,"strict_ppl":None}
    RESULT.write_text(json.dumps(result,indent=2)); print(json.dumps({"event":"complete",**result},indent=2),flush=True)


if __name__=="__main__": main()
