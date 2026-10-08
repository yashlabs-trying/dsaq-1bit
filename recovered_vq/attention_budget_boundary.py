"""Attention-only budget boundary with original Layer-0 MLP restored."""
from __future__ import annotations
import argparse, hashlib, json, struct
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base

ATTN=("q_proj","k_proj","v_proj","o_proj")

def subset_codec(source,names,out):
    meta,sc,cp,codes,pids,sids,hard,rids,rq,rs=base.read_codec(source);old=list(meta['names']);gmask=torch.zeros(meta['nblocks'],dtype=torch.bool);grid=torch.full((meta['nblocks'],),-1,dtype=torch.int16);grid[hard]=rids.short();parts=[];slices={};cur=0;tables=[];ids=[]
    for n in names:
        st,en,r,c=meta['slices'][n];nb=en-st;slices[n]=(cur,cur+nb,(r,c));cur+=nb;parts.append(pids[st:en]);ids.append(sids[st:en]);tables.append(sc[old.index(n)]);gmask_seg=hard[st:en];gmask_seg=gmask_seg.clone();
    pp=torch.cat(parts);si=torch.cat(ids);tables=torch.stack(tables);mask=torch.cat([hard[meta['slices'][n][0]:meta['slices'][n][1]] for n in names]);rids2=torch.cat([grid[meta['slices'][n][0]:meta['slices'][n][1]][mask[sum(meta['slices'][x][1]-meta['slices'][x][0] for x in names[:i]):sum(meta['slices'][x][1]-meta['slices'][x][0] for x in names[:i+1])]] for i,n in enumerate(names)]).byte() if mask.any() else None
    base.write_codec(out,'Attention subset',slices,tables,si,cp,codes,pp,mask,rids2,rq,rs)
    return out.read_bytes()

def write_sparse(path,source,selected,total):
    if selected:
        sub=path.with_suffix('.subset.tmp');blob=subset_codec(source,selected,sub);sub.unlink()
    else: blob=b''
    meta={'version':1,'method':'Sparse attention boundary','selected':list(selected),'attention_bytes':len(blob),'weights':total};mh=json.dumps(meta,separators=(',',':'),sort_keys=True).encode();path.write_bytes(struct.pack('<I',len(mh))+mh+blob)

def decode_sparse(path):
    b=path.read_bytes();ml=struct.unpack_from('<I',b,0)[0];meta=json.loads(b[4:4+ml]);pos=4+ml
    if not meta['attention_bytes']: return {},meta
    sub=path.with_suffix('.subset.tmp');sub.write_bytes(b[pos:pos+meta['attention_bytes']]);w,_=base.decode(sub);sub.unlink();return w,meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/workspace/transform_patterns/none.bin'));p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt'));p.add_argument('--output',type=Path,default=Path('/workspace/attention_boundary'));p.add_argument('--sequence-length',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval();orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()};total=sum(w.numel() for w in orig.values());tok=torch.load(a.tokens,map_location='cpu',weights_only=True);ev=tok['evaluation'];base.restore(model,orig);ref=base.perplexity(model,ev,a.sequence_length)
    cases=[('all_current',ATTN),('qk_removed',('v_proj','o_proj')),('vo_removed',('q_proj','k_proj')),('all_removed',())];rows=[]
    for name,sel in cases:
        f=a.output/(name+'.bin');write_sparse(f,a.source,sel,total);aw,meta=decode_sparse(f);recon={n:(aw[n] if n in aw else torch.zeros_like(orig[n])) for n in ATTN};base.apply(model,{**recon,**{n:orig[n] for n in ('gate_proj','up_proj','down_proj')}});m=base.perplexity(model,ev,a.sequence_length);rows.append({'method':name,'selected':list(sel),'bpw':f.stat().st_size*8/total,'ppl':m['ppl'],'delta_ppl_pct':100*(m['ppl']/ref['ppl']-1),'saved_vs_current_global_bpw':None if name=='all_current' else rows[0]['bpw']-f.stat().st_size*8/total,'file_bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()});print(json.dumps(rows[-1],indent=2),flush=True)
    report={'model':base.MODEL,'layer':0,'reference':ref,'results':rows,'attention_fraction':sum(orig[n].numel() for n in ATTN)/total,'max_mlp_gain_if_attention_zero':rows[0]['bpw']/((sum(orig[n].numel() for n in ('gate_proj','up_proj','down_proj')))/total)};base.dump(a.output/'results.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
