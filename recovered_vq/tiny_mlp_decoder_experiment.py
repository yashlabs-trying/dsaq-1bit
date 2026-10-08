"""Layer-0 asymmetric codec: frozen cheap attention + tiny learned MLP decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM

import waverel_checkpoint_patterns as base

MLP = ("gate_proj", "up_proj", "down_proj")
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
BITS = (6, 8, 10)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-codec", type=Path, default=Path("/workspace/transform_patterns/none.bin"))
    p.add_argument("--tokens", type=Path, default=Path("/workspace/waverel_patterns/fixed_token_ids.pt"))
    p.add_argument("--output", type=Path, default=Path("/workspace/tiny_mlp_decoder"))
    p.add_argument("--sequence-length", type=int, default=256)
    return p.parse_args()


def pack_uint(x, bits):
    a = x.detach().cpu().numpy().astype(np.uint16).reshape(-1)
    bm = ((a[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(bm.reshape(-1), bitorder="little").tobytes()


def unpack_uint(blob, count, bits):
    raw = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[:count * bits]
    bm = raw.reshape(count, bits).astype(np.uint16)
    return torch.from_numpy((bm * (1 << np.arange(bits, dtype=np.uint16))).sum(1).astype(np.int32))


class SharedDecoder(nn.Module):
    def __init__(self, k, conditioned=False):
        super().__init__()
        self.conditioned = conditioned
        self.code = nn.Embedding(k, 8)
        self.proj = nn.Embedding(3, 4) if conditioned else None
        self.fc1 = nn.Linear(12 if conditioned else 8, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, ids, typ):
        x = self.code(ids)
        if self.conditioned:
            x = torch.cat((x, self.proj(typ)), 1)
        return self.fc2(torch.nn.functional.silu(self.fc1(x)))


class SeparateDecoder(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.parts = nn.ModuleList([SharedDecoder(k, False) for _ in range(3)])

    def forward(self, ids, typ):
        out = torch.empty(ids.shape[0], 16, device=ids.device)
        for t, dec in enumerate(self.parts):
            mask = typ == t
            if mask.any():
                out[mask] = dec(ids[mask], typ[mask])
        return out


def make_decoder(variant, k):
    if variant == "shared": return SharedDecoder(k, False)
    if variant == "conditioned": return SharedDecoder(k, True)
    if variant == "separate": return SeparateDecoder(k)
    raise ValueError(variant)


def quantized_state(model):
    state = OrderedDict()
    for name, value in model.state_dict().items():
        scale = max(float(value.detach().abs().max()) / 127, np.finfo(np.float16).tiny)
        q = torch.round(value.detach().cpu() / scale).clamp(-127, 127).to(torch.int8)
        state[name] = (q, np.float16(scale))
    return state


def load_quantized_state(model, state):
    decoded = {n: q.float() * float(s) for n, (q, s) in state.items()}
    model.load_state_dict(decoded)


def decoder_tables(model, k, variant):
    model.eval()
    with torch.no_grad():
        ids = torch.arange(k, device="cuda")
        return [model(ids, torch.full_like(ids, t)).cpu() for t in range(3)]


def source_attention_payload(source, temp, originals):
    meta, scales, cp, codes, pids, sids, hard, rids, rq, rs = base.read_codec(source)
    end = meta["slices"]["o_proj"][1]
    slices = {n: (meta["slices"][n][0], meta["slices"][n][1],
                  (meta["slices"][n][2], meta["slices"][n][3])) for n in ATTN}
    global_rid = torch.full((meta["nblocks"],), -1, dtype=torch.int16)
    global_rid[hard] = rids.short()
    ahard = hard[:end]
    base.write_codec(temp, "Frozen attention", slices, scales[:4], sids[:end], cp, codes,
                     pids[:end], ahard, global_rid[:end][ahard].byte(), rq, rs,
                     meta_extra={"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    recon, _ = base.decode(temp)
    return temp.read_bytes(), recon


def mlp_data(originals, fisher):
    vec, imp, typ, slices = [], [], [], {}
    cursor = 0
    for t, name in enumerate(MLP):
        vb, ib = base.blocks(originals[name].float()), base.blocks(fisher[name].float())
        slices[name] = (cursor, cursor + len(vb), tuple(originals[name].shape))
        cursor += len(vb); vec.append(vb); imp.append(ib)
        typ.append(torch.full((len(vb),), t, dtype=torch.long))
    vec, imp, typ = torch.cat(vec), torch.cat(imp), torch.cat(typ)
    matrix_index = typ.clone()
    raw = vec.abs().amax(1).clamp_min(1e-12)
    scale_tables, scale_ids, decoded_scale = base.quantize_scales(raw, matrix_index, 3)
    return vec, imp, typ, slices, scale_tables, scale_ids, decoded_scale, vec / decoded_scale[:, None]


def initial_centers(source, typ, target, k, per_projection):
    meta, scales, cp, codes, pids, sids, hard, rids, rq, rs = base.read_codec(source)
    if cp.shape[0] == 1: patterns = cp[0][codes.long()]
    else:
        fam = (torch.arange(codes.shape[0]) // base.PATTERNS_PER_FAMILY).long()
        patterns = cp[fam[:, None], codes.long()]
    mlp_start = meta["slices"]["gate_proj"][0]
    old_ids = pids[mlp_start:].long()
    tables = []
    for t in range(3 if per_projection else 1):
        mask = typ == t if per_projection else torch.ones_like(typ, dtype=torch.bool)
        use = torch.bincount(old_ids[mask], minlength=base.PATTERNS)
        chosen = torch.topk(use, k).indices
        tables.append(patterns[chosen].float())
    return tables


def assign_from_tables(target, imp, typ, tables):
    out = torch.empty(target.shape[0], dtype=torch.int32)
    if len(tables) == 1:
        return base.fisher_assign(target, imp, tables[0], batch=4096, topk=64)
    for t in range(3):
        mask = typ == t
        out[mask] = base.fisher_assign(target[mask], imp[mask], tables[t], batch=4096, topk=64)
    return out


def pretrain_tables(model, tables, steps=400):
    """Make the neural decoder reproduce its VQ initialization before block loss."""
    k = tables[0].shape[0]
    ids = torch.arange(k, device="cuda").repeat(len(tables))
    typ = torch.cat([torch.full((k,), t, device="cuda", dtype=torch.long)
                     for t in range(len(tables))])
    goal = torch.cat(tables).cuda()
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-6)
    for _ in range(steps):
        pred = model(ids, typ)
        loss = (pred-goal).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()


def fit_weight(model, ids, typ, target, imp, steps, lr):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = len(ids)
    for _ in range(steps):
        ix = torch.randint(n, (16384,))
        pred = model(ids[ix].cuda().long(), typ[ix].cuda()).float()
        h = imp[ix].cuda(); y = target[ix].cuda()
        loss = ((pred - y).square() * h).sum() / h.sum().clamp_min(1e-20)
        opt.zero_grad(); loss.backward(); opt.step()


def reconstruct(model, ids, typ, decoded_scale, slices):
    chunks = []
    model.eval()
    for s in range(0, len(ids), 65536):
        chunks.append(model(ids[s:s+65536].cuda().long(), typ[s:s+65536].cuda()))
    b = torch.cat(chunks) * decoded_scale.cuda()[:, None]
    return {n: base.unblock(b[s:e], shape) for n, (s, e, shape) in slices.items()}


def functional_finetune(model, ids, typ, decoded_scale, slices, originals, acts,
                        target, imp, steps=20):
    model.train(); opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = acts["gate_proj"][:32].cuda()
    xd = acts["down_proj"][:32].cuda()
    og, ou, od = (originals[n].cuda() for n in MLP)
    with torch.no_grad():
        yg, yu = x @ og.T, x @ ou.T
        ym = torch.nn.functional.silu(yg) * yu
        yd = xd @ od.T
        yfull = ym @ od.T
    for _ in range(steps):
        w = reconstruct(model, ids, typ, decoded_scale, slices)
        g, u, d = (w[n] for n in MLP)
        pg, pu = x @ g.T, x @ u.T
        pm = torch.nn.functional.silu(pg) * pu
        pd = xd @ d.T
        pfull = pm @ d.T
        pre = (pg-yg).square().sum()/yg.square().sum() + (pu-yu).square().sum()/yu.square().sum()
        joint = (pm-ym).square().sum()/ym.square().sum().clamp_min(1e-20)
        down = (pd-yd).square().sum()/yd.square().sum().clamp_min(1e-20)
        full = (pfull-yfull).square().sum()/yfull.square().sum().clamp_min(1e-20)
        ix=torch.randint(len(ids),(8192,)); wb=model(ids[ix].cuda().long(),typ[ix].cuda())
        wh=imp[ix].cuda(); wy=target[ix].cuda()
        weight=((wb-wy).square()*wh).sum()/wh.sum().clamp_min(1e-20)
        loss = .10*weight + .25*pre + 2*joint + 2*down + 4*full
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()


def write_combined(path, variant, bits, att_blob, scale_tables, scale_ids, ids,
                   state, slices, total_weights):
    params = []
    for name, (q, scale) in state.items():
        params.append({"name": name, "shape": list(q.shape), "count": q.numel()})
    meta = {"version": 1, "method": variant, "latent_bits": bits, "latent_k": 1 << bits,
            "weights": total_weights, "nblocks": len(ids), "attention_bytes": len(att_blob),
            "scale_shape": list(scale_tables.shape), "scale_id_count": len(scale_ids),
            "slices": {n: [s,e,*shape] for n,(s,e,shape) in slices.items()}, "params": params,
            "decoder": {"embedding": 8, "hidden": 32, "projection_embedding": 4 if variant=="conditioned" else 0,
                        "parameter_format": "symmetric_int8_per_tensor_fp16_scale"}}
    mh = json.dumps(meta, separators=(",", ":"), sort_keys=True).encode()
    parts = [struct.pack("<I", len(mh)), mh, att_blob, scale_tables.numpy().astype("<f2").tobytes(),
             base.pack_nibbles(scale_ids), pack_uint(ids, bits)]
    for q, scale in state.values():
        parts += [q.numpy().tobytes(), np.asarray(scale, dtype="<f2").tobytes()]
    path.write_bytes(b"".join(parts))


def decode_combined(path):
    blob = path.read_bytes(); pos = 0
    ml = struct.unpack_from("<I", blob, pos)[0]; pos += 4
    meta = json.loads(blob[pos:pos+ml]); pos += ml
    att_path = path.with_suffix(".attention.tmp")
    att_path.write_bytes(blob[pos:pos+meta["attention_bytes"]]); pos += meta["attention_bytes"]
    att, _ = base.decode(att_path); att_path.unlink()
    scount = math.prod(meta["scale_shape"])
    scales = torch.from_numpy(np.frombuffer(blob, dtype="<f2", count=scount, offset=pos).copy()).view(meta["scale_shape"]); pos += scount*2
    n = meta["scale_id_count"]; nb=(n+1)//2
    sids = base.unpack_nibbles(blob[pos:pos+nb], n); pos += nb
    ib=(n*meta["latent_bits"]+7)//8
    ids=unpack_uint(blob[pos:pos+ib],n,meta["latent_bits"]); pos += ib
    state=OrderedDict()
    for p in meta["params"]:
        q=torch.from_numpy(np.frombuffer(blob,dtype=np.int8,count=p["count"],offset=pos).copy()).view(p["shape"]); pos += p["count"]
        scale=np.frombuffer(blob,dtype="<f2",count=1,offset=pos)[0]; pos += 2
        state[p["name"]]=(q,scale)
    assert pos == len(blob), (pos,len(blob))
    model=make_decoder(meta["method"],meta["latent_k"]).cuda(); load_quantized_state(model,state)
    typ=torch.empty(n,dtype=torch.long); slices={}
    for t,name in enumerate(MLP):
        s,e,r,c=meta["slices"][name]; typ[s:e]=t; slices[name]=(s,e,(r,c))
    decoded_scale=scales[typ,sids.long()].float()
    mlp={n:w.cpu() for n,w in reconstruct(model,ids,typ,decoded_scale,slices).items()}
    return {**att,**mlp},meta


@torch.inference_mode()
def interaction_metrics(originals, recon, acts):
    x=acts["gate_proj"].cuda(); og,ou,od=(originals[n].cuda() for n in MLP); qg,qu,qd=(recon[n].cuda() for n in MLP)
    om=torch.nn.functional.silu(x@og.T)*(x@ou.T); qm=torch.nn.functional.silu(x@qg.T)*(x@qu.T)
    interaction=math.sqrt(float((qm-om).square().sum()/om.square().sum().clamp_min(1e-20)))
    oo=om@od.T; qo=qm@qd.T
    residual=math.sqrt(float((qo-oo).square().sum()/oo.square().sum().clamp_min(1e-20)))
    return interaction,residual


def evaluate(path, model, originals, acts, eval_ids, ref, seq, attention_bytes):
    recon,meta=decode_combined(path); base.apply(model,recon); ppl=base.perplexity(model,eval_ids,seq)
    act,per=base.activation_metrics({n:originals[n] for n in MLP},{n:recon[n] for n in MLP},{n:acts[n] for n in MLP})
    inter,resid=interaction_metrics(originals,recon,acts); total=sum(w.numel() for w in originals.values())
    pcount=sum(p["count"] for p in meta["params"]); decoder_bytes=sum(p["count"]+2 for p in meta["params"])
    att_bpw=attention_bytes*8/total; total_bpw=path.stat().st_size*8/total
    rows={x["family"]:x["activation_error"] for x in per}
    return {"method":meta["method"],"latent_bits":meta["latent_bits"],"bpw":total_bpw,
            "ppl":ppl["ppl"],"delta_ppl_pct":100*(ppl["ppl"]/ref["ppl"]-1),
            "gate_error":rows["gate_proj"],"up_error":rows["up_proj"],"down_error":rows["down_proj"],
            "mlp_activation_error":act,"gate_up_interaction_error":inter,"down_residual_stream_error":resid,
            "decoder_parameter_count":pcount,"decoder_storage_bpw":decoder_bytes*8/total,
            "latent_id_bpw":meta["latent_bits"]*meta["nblocks"]/total,"attention_bpw":att_bpw,
            "mlp_bpw":total_bpw-att_bpw,"file_bytes":path.stat().st_size,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    a=args();a.output.mkdir(parents=True,exist_ok=True);base.seed_all();started=time.time()
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={"":"cuda"}).eval()
    originals={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()}; total=sum(w.numel() for w in originals.values())
    tokens=torch.load(a.tokens,map_location="cpu",weights_only=True); held,evaluation=tokens["heldout"],tokens["evaluation"]
    base.restore(model,originals); ref=base.perplexity(model,evaluation,a.sequence_length); acts=base.capture_inputs(model,held,a.sequence_length)
    names={f"L0.{'SA' if n in ATTN else 'MLP'}.{n}":w for n,w in originals.items()}
    f0=base.legacy.compute_fisher(model,names);model.to("cuda")
    fisher={n:f0[f"L0.{'SA' if n in ATTN else 'MLP'}.{n}"].cpu() for n in originals}
    att_blob,att_recon=source_attention_payload(a.source_codec,a.output/"attention.bin",originals)
    vec,imp,typ,slices,scale_tables,scale_ids,decoded_scale,target=mlp_data(originals,fisher)
    results=[]
    init_cache={}
    for bits in BITS:
        k=1<<bits
        init_cache[(bits,"shared")]=assign_from_tables(target,imp,typ,initial_centers(a.source_codec,typ,target,k,False))
        init_cache[(bits,"per")]=assign_from_tables(target,imp,typ,initial_centers(a.source_codec,typ,target,k,True))
        for variant in ("shared","conditioned","separate"):
            print(f"===== {variant} {bits}-bit =====",flush=True);base.seed_all()
            ids=init_cache[(bits,"shared" if variant=="shared" else "per")].clone()
            init_tables=initial_centers(a.source_codec,typ,target,k,variant!="shared")
            dec=make_decoder(variant,k).cuda();pretrain_tables(dec,init_tables)
            fit_weight(dec,ids,typ,target,imp,200,1e-3)
            tables=decoder_tables(dec,k,variant); ids=assign_from_tables(target,imp,typ,tables if variant!="shared" else tables[:1])
            fit_weight(dec,ids,typ,target,imp,100,5e-4)
            functional_finetune(dec,ids,typ,decoded_scale,slices,originals,acts,target,imp,20)
            state=quantized_state(dec);load_quantized_state(dec,state)
            tables=decoder_tables(dec,k,variant);ids=assign_from_tables(target,imp,typ,tables if variant!="shared" else tables[:1])
            path=a.output/f"{variant}_{bits}bit.bin";write_combined(path,variant,bits,att_blob,scale_tables,scale_ids,ids,state,slices,total)
            if path.stat().st_size*8/total>1.20: raise RuntimeError(f"budget exceeded {path}")
            row=evaluate(path,model,originals,acts,evaluation,ref,a.sequence_length,len(att_blob));results.append(row);print(json.dumps(row,indent=2),flush=True)
    source_recon,_=base.decode(a.source_codec);base.apply(model,source_recon);sp=base.perplexity(model,evaluation,a.sequence_length);sa,sper=base.activation_metrics({n:originals[n] for n in MLP},{n:source_recon[n] for n in MLP},{n:acts[n] for n in MLP});si,sr=interaction_metrics(originals,source_recon,acts)
    baseline={"method":"Current VQ","bpw":a.source_codec.stat().st_size*8/total,"ppl":sp["ppl"],"delta_ppl_pct":100*(sp["ppl"]/ref["ppl"]-1),"mlp_activation_error":sa,"gate_up_interaction_error":si,"down_residual_stream_error":sr,**{x["family"].replace("_proj","")+"_error":x["activation_error"] for x in sper}}
    qualifying=[r for r in results if r["mlp_activation_error"]<baseline["mlp_activation_error"] and r["ppl"]<baseline["ppl"]]
    report={"model":base.MODEL,"layer":0,"reference":ref,"baseline":baseline,"results":results,"best":min(qualifying,key=lambda r:r["ppl"])["method"]+f" {min(qualifying,key=lambda r:r['ppl'])['latent_bits']}-bit" if qualifying else None,"material_improvement":bool(qualifying),"elapsed_seconds":time.time()-started}
    base.dump(a.output/"results.json",report);print("FINAL\n"+json.dumps(report,indent=2),flush=True)


if __name__=="__main__":main()
