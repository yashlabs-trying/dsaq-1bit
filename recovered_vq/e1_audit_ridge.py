"""Audit E1 incremental losses and compute original/ridge Down diagnostics."""
from pathlib import Path
import hashlib,json,math,time
import torch
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base

ROOT=Path('/workspace/experiments/functional_vq_l0'); E1=ROOT/'packed/E1_functional_5pct.packed'; TOK=ROOT/'calibration_cache/fixed_token_ids.pt'; OUT=ROOT/'results/E1_audit_ridge.json'

@torch.inference_mode()
def capture_x(model,ids):
 vals=[]; h=model.model.layers[0].mlp.gate_proj.register_forward_pre_hook(lambda _m,i:vals.append(i[0].detach().float().cpu()))
 for s in range(0,ids.numel()-255,256): model(ids[s:s+256].unsqueeze(0).cuda(),use_cache=False)
 h.remove(); return torch.cat([v.reshape(-1,v.shape[-1]) for v in vals]).cuda()

def f(w,x):
 g=x@w['gate_proj'].T; u=x@w['up_proj'].T; z=torch.nn.functional.silu(g)*u; return g,u,z,z@w['down_proj'].T

def rel(a,b): return float((a-b).norm()/b.norm())
def cos(a,b): return float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))

def main():
 t0=time.time(); base.seed_all(); model=AutoModelForCausalLM.from_pretrained(base.MODEL,dtype=torch.float32,device_map={'':'cuda'}).eval(); orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()}; ids=torch.load(TOK,map_location='cpu',weights_only=True)['heldout']; base.restore(model,orig); xA=capture_x(model,ids[:512]); xB=capture_x(model,ids[512:1024]); ow={k:orig[k].cuda() for k in ('gate_proj','up_proj','down_proj')}; tA=f(ow,xA); tB=f(ow,xB)
 w,meta=base.decode(E1); w={k:v.cuda() for k,v in w.items()}; m0,st,cp,codes,pids,sids,hard,rids,rq,rs=base.read_codec(E1); fam=(torch.arange(codes.shape[0])//base.PATTERNS_PER_FAMILY).long(); pats=cp[fam[:,None],codes.long()].float(); mi=torch.empty(m0['nblocks'],dtype=torch.long); slices={n:tuple(m0['slices'][n]) for n in m0['names']}
 for j,n in enumerate(m0['names']): mi[slices[n][0]:slices[n][1]]=j
 scales=st[mi,sids.long()].float(); residual=torch.zeros(m0['nblocks'],16)
 if hard is not None: residual[hard]=rq.float()[rids.long()]*rs.float()[rids.long(),None]
 sA=f(w,xA); denZ=tA[2].square().sum(); denY=tA[3].square().sum(); gen=torch.Generator().manual_seed(777); records=[]; mlp_ranges=[]
 for n in ('gate_proj','up_proj','down_proj'):
  s,e,r,c=slices[n]; mlp_ranges.extend([(n,q) for q in torch.randint(s,e,(34,),generator=gen).tolist()])
 for n,q in mlp_ranges[:100]:
  s,e,r,c=slices[n]; local=q-s; nbc=c//4; br=local//nbc; bc=local%nbc; rows=slice(br*4,br*4+4); cols=slice(bc*4,bc*4+4); old=w[n][rows,cols].clone(); cid=(int(pids[q])+1+(q%31))%pats.shape[0]; nb=(pats[cid]*float(scales[q])+residual[q]).cuda().view(4,4)
  if n=='gate_proj':
   dg=xA[:,cols]@(nb-old).T; nz=torch.nn.functional.silu(sA[0][:,rows]+dg)*sA[1][:,rows]; inc=float((sA[2]-tA[2]).square().sum()- (sA[2][:,rows]-tA[2][:,rows]).square().sum()+(nz-tA[2][:,rows]).square().sum())/float(denZ)
  elif n=='up_proj':
   du=xA[:,cols]@(nb-old).T; nz=torch.nn.functional.silu(sA[0][:,rows])*(sA[1][:,rows]+du); inc=float((sA[2]-tA[2]).square().sum()- (sA[2][:,rows]-tA[2][:,rows]).square().sum()+(nz-tA[2][:,rows]).square().sum())/float(denZ)
  else:
   dy=sA[2][:,cols]@(nb-old).T; inc=float((sA[3]-tA[3]).square().sum()-(sA[3][:,rows]-tA[3][:,rows]).square().sum()+(sA[3][:,rows]+dy-tA[3][:,rows]).square().sum())/float(denY)
  tw={k:v for k,v in w.items()}; tw[n]=w[n].clone(); tw[n][rows,cols]=nb; full=f(tw,xA); fullv=float(((full[2]-tA[2]).square().sum()/denZ) if n!='down_proj' else ((full[3]-tA[3]).square().sum()/denY)); records.append({'projection':n,'abs_diff':abs(inc-fullv),'incremental':inc,'full':fullv})
 # Down diagnostics after E1 G/U.
 zA=sA[2]; zB=f(w,xB)[2]; ytA=tA[3]; ytB=tB[3]; origA=zA@ow['down_proj'].T; origB=zB@ow['down_proj'].T
 scale=float((zA@zA.T).diagonal().mean()); ridge=[]
 for mult in (1e-8,1e-6,1e-4,1e-2):
  lam=scale*mult; alpha=torch.linalg.solve(zA@zA.T+lam*torch.eye(zA.shape[0],device='cuda'),ytA); wd=(zA.T@alpha).T; ya=zA@wd.T; yb=zB@wd.T; ridge.append({'multiplier':mult,'lambda':lam,'Ey_A':rel(ya,ytA),'Ey_B':rel(yb,ytB),'cos_A':cos(ya,ytA),'cos_B':cos(yb,ytB)})
 best=min(ridge,key=lambda r:r['Ey_B']); result={'audit_count':len(records),'audit_max_abs_diff':max(r['abs_diff'] for r in records),'audit_mean_abs_diff':sum(r['abs_diff'] for r in records)/len(records),'audit_pass':max(r['abs_diff'] for r in records)<1e-5,'E1_compressed_down':{'Ey_A':rel(sA[3],ytA),'Ey_B':rel(f(w,xB)[3],ytB)},'original_bf16_down':{'Ey_A':rel(origA,ytA),'Ey_B':rel(origB,ytB),'cos_A':cos(origA,ytA),'cos_B':cos(origB,ytB)},'ridge':ridge,'best_ridge':best,'elapsed_seconds':time.time()-t0}; OUT.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
