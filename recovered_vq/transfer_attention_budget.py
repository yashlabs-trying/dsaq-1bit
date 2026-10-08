"""Transfer Q/K attention savings into wider fixed-MLP residual coverage."""
from __future__ import annotations
import argparse, hashlib, json, struct, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base
import attention_budget_boundary as ab

MLP=("gate_proj","up_proj","down_proj")

def write_combo(path,att,mlp,total,att_names):
    meta={'version':1,'method':'QK removed + expanded MLP residual','attention_bytes':len(att),'mlp_bytes':len(mlp),'weights':total,'attention_selected':list(att_names)};h=json.dumps(meta,separators=(',',':'),sort_keys=True).encode();path.write_bytes(struct.pack('<I',len(h))+h+att+mlp)

def decode_combo(path):
    b=path.read_bytes();ml=struct.unpack_from('<I',b,0)[0];meta=json.loads(b[4:4+ml]);p=4+ml;ap=path.with_suffix('.a.tmp');mp=path.with_suffix('.m.tmp');ap.write_bytes(b[p:p+meta['attention_bytes']]);p+=meta['attention_bytes'];mp.write_bytes(b[p:p+meta['mlp_bytes']]);aw,_=base.decode(ap);mw,_=base.decode(mp);ap.unlink();mp.unlink();return {**aw,**mw},meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/workspace/transform_patterns/none.bin'));p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt'));p.add_argument('--output',type=Path,default=Path('/workspace/transfer_budget'));p.add_argument('--sequence-length',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);base.seed_all();start=time.time()
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval();orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()};total=sum(w.numel() for w in orig.values());tok=torch.load(a.tokens,map_location='cpu',weights_only=True);ev=tok['evaluation'];base.restore(model,orig);ref=base.perplexity(model,ev,a.sequence_length)
    # exact attention payload with Q/K removed
    atmp=a.output/'att.tmp';att=ab.subset_codec(a.source,('v_proj','o_proj'),atmp);atmp.unlink()
    names={f"L0.{'SA' if n not in MLP else 'MLP'}.{n}":w for n,w in orig.items()};f0=base.legacy.compute_fisher(model,names);model.to('cuda');fisher={n:f0[f"L0.{'SA' if n not in MLP else 'MLP'}.{n}"].cpu() for n in orig}
    meta,sc,cp,codes,pids,sids,hard,rids,rq,rs=base.read_codec(a.source);old=list(meta['names']);g_rid=torch.full((meta['nblocks'],),-1,dtype=torch.int16);g_rid[hard]=rids.short();vec=[];imp=[];slices={};cur=0;st_m=meta['slices']['gate_proj'][0]
    for n in MLP:
        st,en,r,c=meta['slices'][n];vb,ib=base.blocks(orig[n].float()),base.blocks(fisher[n].float());vec.append(vb);imp.append(ib);slices[n]=(cur,cur+len(vb),(r,c));cur+=len(vb)
    vec=torch.cat(vec);imp=torch.cat(imp);# decode current shared pattern table and scale IDs for MLP
    fam=(torch.arange(codes.shape[0])//base.PATTERNS_PER_FAMILY).long();patterns=cp[fam[:,None],codes.long()];mids=torch.cat([sids[meta['slices'][n][0]:meta['slices'][n][1]] for n in MLP]);mnames=[n for n in MLP];msc=torch.stack([sc[old.index(n)] for n in MLP]);mi=torch.cat([torch.full((slices[n][1]-slices[n][0],),i,dtype=torch.long) for i,n in enumerate(MLP)]);scale=msc[mi,mids.long()].float();base_blocks=patterns[pids[st_m:].long()]*scale[:,None];residual=vec-base_blocks
    rows=[]
    for fraction in (0.20,0.30,0.40,0.50):
        count=round(fraction*len(vec));err=(imp*residual.square()).sum(1)/imp.sum(1).clamp_min(1e-20);chosen=torch.topk(err,count).indices;mask=torch.zeros(len(vec),dtype=torch.bool);mask[chosen]=True;cent,_=base.legacy.fast_kmeans(residual[mask],base.K_RESIDUAL,n_iter=20,batch=4096);rs2=(cent.abs().amax(1)/127).clamp_min(torch.finfo(torch.float16).tiny).half();rq2=torch.round(cent/rs2.float()[:,None]).clamp(-127,127).to(torch.int8);rid=base.legacy.weighted_assign(residual[mask],imp[mask],rq2.float()*rs2.float()[:,None],verbose=False).byte();mtmp=a.output/'mlp.tmp';mslices=slices;mpids=pids[st_m:].clone();msids=mids.clone();base.write_codec(mtmp,'MLP fixed VQ expanded residual',mslices,msc,msids,cp,codes,mpids,mask,rid,rq2,rs2);mlp=mtmp.read_bytes();mtmp.unlink();path=a.output/f'qk_removed_mlp_{int(fraction*100)}pct.bin';write_combo(path,att,mlp,total,('v_proj','o_proj'));recon,_=decode_combo(path);recon.update({'q_proj':torch.zeros_like(orig['q_proj']),'k_proj':torch.zeros_like(orig['k_proj'])});base.apply(model,recon);m=base.perplexity(model,ev,a.sequence_length);bpw=path.stat().st_size*8/total;row={'residual_fraction':fraction,'bpw':bpw,'ppl':m['ppl'],'delta_ppl_pct':100*(m['ppl']/ref['ppl']-1),'budget_ok':bpw<=1.2,'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};rows.append(row);print(json.dumps(row,indent=2),flush=True)
    report={'reference':ref,'attention_selected':['v_proj','o_proj'],'attention_bpw':len(att)*8/total,'results':rows,'max_bpw':1.2,'elapsed_seconds':time.time()-start};base.dump(a.output/'results.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
