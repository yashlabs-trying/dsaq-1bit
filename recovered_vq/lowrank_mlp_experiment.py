"""Layer-0 global low-rank MLP codec with packed 1-bit residuals."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base

MLP=("gate_proj","up_proj","down_proj"); ATTN=("q_proj","k_proj","v_proj","o_proj")

def pack_bits(x): return np.packbits(x.detach().cpu().numpy().astype(np.uint8).reshape(-1),bitorder='little').tobytes()
def unpack_bits(blob,n): return torch.from_numpy(np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder='little')[:n].copy())

def qrows(x):
    scale=(x.abs().amax(1)/127).clamp_min(torch.finfo(torch.float16).tiny)
    q=torch.round(x/scale[:,None]).clamp(-127,127).to(torch.int8)
    return q,scale.half()

def fit_factors(w, h, rank):
    d=h.clamp_min(1e-12).sqrt(); z=w*d[None,:]
    # Small rank randomized SVD is sufficient and runs on the A4000.
    u,s,v=torch.svd_lowrank(z.cuda(),q=rank,niter=4)
    a=(u[:,:rank]*s[:rank]).cpu(); b=(v[:,:rank]/d.cuda()[:,None]).cpu()
    aq,ascl=qrows(a); bq,bscl=qrows(b)
    adeq=aq.float()*ascl.float()[:,None]; bdeq=bq.float()*bscl.float()[:,None]
    return aq,ascl,bq,bscl,adeq@bdeq.T

def write_codec(path, rank, att_blob, factors, residuals, total):
    meta={"version":1,"method":"Activation-Hessian low-rank + 1-bit residual","rank":rank,
          "weights":total,"attention_bytes":len(att_blob),"names":list(MLP),"residual_bits":1,
          "factor_format":"int8 per-row fp16 scale","residual_format":"packed signs + fp16 row scales",
          "shapes":{n:list(factors[n][0].shape[:1])+[rank] for n in MLP},
          "matrix_shapes":{n:list(residuals[n].shape) for n in MLP}}
    mh=json.dumps(meta,separators=(',',':'),sort_keys=True).encode(); parts=[struct.pack('<I',len(mh)),mh,att_blob]
    for n in MLP:
        aq,ascl,bq,bscl=factors[n]; r=residuals[n]
        signs=(r>=0); rs=r.abs().mean(1).clamp_min(torch.finfo(torch.float16).tiny).half()
        parts += [aq.numpy().tobytes(),ascl.numpy().astype('<f2').tobytes(),bq.numpy().tobytes(),bscl.numpy().astype('<f2').tobytes(),pack_bits(signs),rs.numpy().astype('<f2').tobytes()]
    path.write_bytes(b''.join(parts))

def decode_codec(path):
    blob=path.read_bytes();pos=0;ml=struct.unpack_from('<I',blob,pos)[0];pos+=4;meta=json.loads(blob[pos:pos+ml]);pos+=ml
    att=path.with_suffix('.att.tmp');att.write_bytes(blob[pos:pos+meta['attention_bytes']]);pos+=meta['attention_bytes'];aw,_=base.decode(att);att.unlink()
    out={};
    for n in MLP:
        r,c=meta['matrix_shapes'][n]; rank=meta['rank']; na=r*rank;nb=c*rank
        aq=torch.from_numpy(np.frombuffer(blob,dtype=np.int8,count=na,offset=pos).copy()).view(r,rank);pos+=na
        asc=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=r,offset=pos).copy());pos+=2*r
        bq=torch.from_numpy(np.frombuffer(blob,dtype=np.int8,count=nb,offset=pos).copy()).view(c,rank);pos+=nb
        bsc=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=c,offset=pos).copy());pos+=2*c
        signs=unpack_bits(blob[pos:pos+(r*c+7)//8],r*c).view(r,c);pos+=(r*c+7)//8
        rs=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=r,offset=pos).copy());pos+=2*r
        a=aq.float()*asc.float()[:,None];b=bq.float()*bsc.float()[:,None]
        out[n]=a@b.T + (signs.float()*2-1)*rs.float()[:,None]
    assert pos==len(blob),(pos,len(blob));return {**aw,**out},meta

@torch.inference_mode()
def interaction(originals,recon,acts):
    x=acts['gate_proj'].cuda();og,ou,od=(originals[n].cuda() for n in MLP);qg,qu,qd=(recon[n].cuda() for n in MLP)
    om=torch.nn.functional.silu(x@og.T)*(x@ou.T);qm=torch.nn.functional.silu(x@qg.T)*(x@qu.T)
    oo=om@od.T;qo=qm@qd.T
    return math.sqrt(float((qm-om).square().sum()/om.square().sum())),math.sqrt(float((qo-oo).square().sum()/oo.square().sum()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--attention',type=Path,default=Path('/workspace/tiny_mlp_decoder/attention.bin'));p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt'));p.add_argument('--output',type=Path,default=Path('/workspace/lowrank_mlp'));p.add_argument('--sequence-length',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);base.seed_all();start=time.time()
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval();originals={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()};total=sum(w.numel() for w in originals.values())
    tok=torch.load(a.tokens,map_location='cpu',weights_only=True);held,evaluation=tok['heldout'],tok['evaluation'];base.restore(model,originals);ref=base.perplexity(model,evaluation,a.sequence_length);acts=base.capture_inputs(model,held,a.sequence_length)
    att_blob=a.attention.read_bytes();results=[]
    for rank in (4,8,16):
        print('RANK',rank,flush=True); factors={};residuals={};lowrank_only={}
        for n in MLP:
            h=acts[n].square().mean(0);aq,asc,bq,bsc,lq=fit_factors(originals[n].float(),h,rank);factors[n]=(aq,asc,bq,bsc);residuals[n]=originals[n].float()-lq;lowrank_only[n]=lq
        path=a.output/f'lowrank_r{rank}.bin';write_codec(path,rank,att_blob,factors,residuals,total);recon,meta=decode_codec(path);base.apply(model,recon);ppl=base.perplexity(model,evaluation,a.sequence_length)
        act,per=base.activation_metrics({n:originals[n] for n in MLP},{n:recon[n] for n in MLP},{n:acts[n] for n in MLP});inter,resid=interaction(originals,recon,acts);rows={x['family']:x['activation_error'] for x in per};bpw=path.stat().st_size*8/total
        # Measure the fitted low-rank component before residual signs as a diagnostic.
        loact,lop=base.activation_metrics({n:originals[n] for n in MLP},lowrank_only,{n:acts[n] for n in MLP})
        row={'rank':rank,'bpw':bpw,'ppl':ppl['ppl'],'delta_ppl_pct':100*(ppl['ppl']/ref['ppl']-1),'gate_error':rows['gate_proj'],'up_error':rows['up_proj'],'down_error':rows['down_proj'],'mlp_activation_error':act,'gate_up_interaction_error':inter,'down_residual_stream_error':resid,'lowrank_only_mlp_activation_error':loact,'lowrank_only_gate_up_interaction_error':interaction(originals,lowrank_only,acts)[0],'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'lowrank_parameter_count':sum((f[0].numel()+f[2].numel()) for f in factors.values()),'lowrank_storage_bpw':sum((f[0].numel()+f[2].numel()+f[1].numel()+f[3].numel())*8 for f in factors.values())/total,'attention_bpw':att_blob.__len__()*8/total,'residual_bpw':sum(r.numel() for r in residuals.values())/total,'residual_scale_bpw':sum(r.shape[0]*16 for r in residuals.values())/total};results.append(row);print(json.dumps(row,indent=2),flush=True)
    # Reference current VQ on the same run.
    aw,_=base.decode(a.attention);# use original MLP with frozen attention only as an oracle diagnostic
    base.apply(model,{**aw,**{n:originals[n] for n in MLP}});mp=base.perplexity(model,evaluation,a.sequence_length)
    report={'model':base.MODEL,'layer':0,'reference':ref,'baseline_attention_plus_original_mlp':mp,'results':results,'constraints':{'max_bpw':1.2,'residual':'1-bit packed sign per weight + row fp16 scale','attention_codec':str(a.attention)},'elapsed_seconds':time.time()-start}
    base.dump(a.output/'results.json',report);print('FINAL\n'+json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
