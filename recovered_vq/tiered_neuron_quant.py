"""Importance-tiered joint neuron quantization for Layer 0 MLP."""
from __future__ import annotations
import argparse, hashlib, json, math, struct, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base

MLP=("gate_proj","up_proj","down_proj"); N=4864

def pack_ids(x,bits):
    if bits==0:return b''
    a=x.detach().cpu().numpy().astype(np.uint16).reshape(-1);bm=((a[:,None]>>np.arange(bits,dtype=np.uint16))&1).astype(np.uint8)
    return np.packbits(bm.reshape(-1),bitorder='little').tobytes()
def unpack_ids(blob,count,bits):
    if bits==0:return torch.zeros(count,dtype=torch.int32)
    raw=np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder='little')[:count*bits].reshape(count,bits).astype(np.uint16)
    return torch.from_numpy((raw*(1<<np.arange(bits,dtype=np.uint16))).sum(1).astype(np.int32))
def quant(x,bits):
    if bits==0:return None,None
    levels=1 if bits==1 else (1<<(bits-1))-1;scale=x.abs().amax(1).clamp_min(torch.finfo(torch.float16).tiny)/levels
    q=torch.round(x/scale[:,None]).clamp(-levels,levels).to(torch.int32);return q,scale.half()

def write(path,bits_tiers,order,att,qs,scales,total):
    counts=[int((torch.arange(N)>=0).sum())]
    n0=round(.10*N);n1=round(.20*N);counts=[n0,n1,N-n0-n1]
    meta={'version':1,'method':'Importance-tiered joint neuron quantization','bits_tiers':list(bits_tiers),'counts':counts,'weights':total,'attention_bytes':len(att),'order_count':N,'components':list(MLP)}
    mh=json.dumps(meta,separators=(',',':'),sort_keys=True).encode();parts=[struct.pack('<I',len(mh)),mh,att,order.numpy().astype('<u2').tobytes()]
    for c in range(3):
        for t,bits in enumerate(bits_tiers):
            m0=sum(counts[:t]);m1=m0+counts[t]
            if bits:
                code=(qs[c][m0:m1]>0).to(torch.int32) if bits==1 else qs[c][m0:m1]+(1<<(bits-1))
                parts += [scales[c][m0:m1].numpy().astype('<f2').tobytes(),pack_ids(code,bits)]
    path.write_bytes(b''.join(parts))

def decode(path):
    blob=path.read_bytes();pos=0;ml=struct.unpack_from('<I',blob,pos)[0];pos+=4;meta=json.loads(blob[pos:pos+ml]);pos+=ml
    ap=path.with_suffix('.att.tmp');ap.write_bytes(blob[pos:pos+meta['attention_bytes']]);pos+=meta['attention_bytes'];att,_=base.decode(ap);ap.unlink()
    order=torch.from_numpy(np.frombuffer(blob,dtype='<u2',count=meta['order_count'],offset=pos).copy()).long();pos+=2*meta['order_count'];counts=meta['counts'];out=[torch.zeros(N,896) for _ in range(3)]
    for c in range(3):
        for t,bits in enumerate(meta['bits_tiers']):
            m0=sum(counts[:t]);m1=m0+counts[t]
            if bits:
                sc=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=counts[t],offset=pos).copy()).float();pos+=2*counts[t];nb=(counts[t]*896*bits+7)//8; # 896 values per neuron
                q=unpack_ids(blob[pos:pos+nb],counts[t]*896,bits).view(counts[t],896).float();q=(q*2-1) if bits==1 else q-(1<<(bits-1));pos+=nb;out[c][m0:m1]=q*sc[:,None]
    assert pos==len(blob),(pos,len(blob));g,u,d=out;weights={'gate_proj':g[order],'up_proj':u[order],'down_proj':d[order].T};return {**att,**weights},meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--attention',type=Path,default=Path('/workspace/tiny_mlp_decoder/attention.bin'));p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt'));p.add_argument('--output',type=Path,default=Path('/workspace/tiered_neuron'));p.add_argument('--sequence-length',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);base.seed_all();start=time.time()
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval();orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()};total=sum(w.numel() for w in orig.values());tok=torch.load(a.tokens,map_location='cpu',weights_only=True);held,evaluation=tok['heldout'],tok['evaluation'];base.restore(model,orig);ref=base.perplexity(model,evaluation,a.sequence_length);acts=base.capture_inputs(model,held,a.sequence_length);att=a.attention.read_bytes()
    x=acts['gate_proj'];g,u,d=orig['gate_proj'],orig['up_proj'],orig['down_proj'];gx=x@g.T;ux=x@u.T;contrib=torch.nn.functional.silu(gx)*ux;dn=d.norm(dim=0);score=(contrib.abs().mean(0)*dn); # functional importance
    order=torch.argsort(score,descending=True);rank=torch.empty_like(order);rank[order]=torch.arange(N);n0=round(.10*N);n1=round(.20*N);tier=torch.where(rank<n0,torch.zeros(N,dtype=torch.long),torch.where(rank<n0+n1,torch.ones(N,dtype=torch.long),torch.full((N,),2,dtype=torch.long)))
    packages=[g,u,d.T];results=[]
    for bits_tiers in ((2,1,1),(2,1,0),(3,1,0),(4,2,0)):
        print('TIERS',bits_tiers,flush=True);qs=[];scs=[]
        for w in packages:
            by=torch.empty(N,896,dtype=torch.int32);bs=torch.ones(N,dtype=torch.float16)
            for t,bits in enumerate(bits_tiers):
                m=tier==t;q,s=quant(w[m].float(),bits)
                if bits:by[m]=q;bs[m]=s
                else:by[m]=0;bs[m]=1
            qs.append(by);scs.append(bs)
        path=a.output/('tier_'+'_'.join(map(str,bits_tiers))+'.bin');write(path,bits_tiers,order,att,qs,scs,total);recon,meta=decode(path);base.apply(model,recon);ppl=base.perplexity(model,evaluation,a.sequence_length)
        xx=x.cuda();og,ou,od=(orig[n].cuda() for n in MLP);qg,qu,qd=(recon[n].cuda() for n in MLP);yg,yu=xx@og.T,xx@ou.T;ygq,yuq=xx@qg.T,xx@qu.T;om=torch.nn.functional.silu(yg)*yu;qm=torch.nn.functional.silu(ygq)*yuq;oo=om@od.T;qo=qm@qd.T
        rows={};
        for n,w,q in zip(MLP,(og,ou,od.T),(qg,qu,qd.T)):rows[n]=float(torch.sqrt(((xx@(w-q).T).square().sum()/(xx@w.T).square().sum()).clamp_min(0)))
        row={'tier_bits':bits_tiers,'bpw':path.stat().st_size*8/total,'ppl':ppl['ppl'],'delta_ppl_pct':100*(ppl['ppl']/ref['ppl']-1),'gate_error':rows['gate_proj'],'up_error':rows['up_proj'],'down_error':rows['down_proj'],'gate_up_interaction_error':math.sqrt(float((qm-om).square().sum()/om.square().sum())),'down_residual_stream_error':math.sqrt(float((qo-oo).square().sum()/oo.square().sum())),'importance_score_quantiles':[float(v) for v in torch.quantile(score,torch.tensor([.1,.5,.9]))],'tier_counts':[n0,n1,N-n0-n1],'attention_bpw':len(att)*8/total,'tier_payload_bpw':sum(bits_tiers[t]*[n0,n1,N-n0-n1][t]*896 for t in range(3))/total,'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};results.append(row);print(json.dumps(row,indent=2),flush=True)
    aw,_=base.decode(a.attention);base.apply(model,{**aw,**{n:orig[n] for n in MLP}});oracle=base.perplexity(model,evaluation,a.sequence_length);report={'model':base.MODEL,'layer':0,'reference':ref,'attention_plus_original_mlp':oracle,'tier_definition':'top 10% critical, next 20% medium, remaining 70% weak','results':results,'max_bpw':1.2,'elapsed_seconds':time.time()-start};base.dump(a.output/'results.json',report);print('FINAL\n'+json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
