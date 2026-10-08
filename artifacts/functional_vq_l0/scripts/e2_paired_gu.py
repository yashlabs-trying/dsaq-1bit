"""E2: paired Gate/Up functional search ranked by exact final MLP output loss."""
from pathlib import Path
import hashlib,json,time
import torch
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base
from strict_functional_vq_smoke import capture_x, mlp, patch_ids, candidates

ROOT=Path('/workspace/experiments/functional_vq_l0'); INPUT=ROOT/'packed/E1_functional_5pct.packed'; OUTPUT=ROOT/'packed/E2_paired_gu.packed'; TOK=ROOT/'calibration_cache/fixed_token_ids.pt'; RES=ROOT/'results/E2_paired_gu.json'

def metric(z,y,zt,yt): return {'Ez':float((z-zt).norm()/zt.norm()),'Ey':float((y-yt).norm()/yt.norm()),'cos_y':float(torch.nn.functional.cosine_similarity(y.flatten(),yt.flatten(),dim=0))}

def main():
 t0=time.time(); base.seed_all(); model=AutoModelForCausalLM.from_pretrained(base.MODEL,dtype=torch.float32,device_map={'':'cuda'}).eval(); orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()}; tok=torch.load(TOK,map_location='cpu',weights_only=True); held=tok['heldout']; base.restore(model,orig); xA=capture_x(model,held[:512]).cuda(); xB=capture_x(model,held[512:1024]).cuda(); ow={k:orig[k].cuda() for k in ('gate_proj','up_proj','down_proj')}; tA=mlp(ow,xA); tB=mlp(ow,xB)
 dec,meta=base.decode(INPUT); w={k:v.cuda().clone() for k,v in dec.items()}; m0,st,cp,codes,pids,sids,hard,rids,rq,rs=base.read_codec(INPUT); ids=pids.clone(); fam=(torch.arange(codes.shape[0])//base.PATTERNS_PER_FAMILY).long(); pats=cp[fam[:,None],codes.long()].float(); slices={n:tuple(m0['slices'][n]) for n in m0['names']}; mi=torch.empty(m0['nblocks'],dtype=torch.long)
 for j,n in enumerate(m0['names']): mi[slices[n][0]:slices[n][1]]=j
 scales=st[mi,sids.long()].float(); residual=torch.zeros(m0['nblocks'],16)
 if hard is not None: residual[hard]=rq.float()[rids.long()]*rs.float()[rids.long(),None]
 gA,uA,zA,yA=mlp(w,xA); gB,uB,zB,yB=mlp(w,xB); beforeA=metric(zA,yA,tA[2],tA[3]); beforeB=metric(zB,yB,tB[2],tB[3]); ydenA=tA[3].square().sum(); ydenB=tB[3].square().sum(); D=w['down_proj']; changed=[]; accepted=0; tested=0
 score=((zA-tA[2]).square().mean(0)*D.norm(dim=0).square()).view(-1,4).mean(1); row_groups=torch.topk(score,max(1,int(.01*score.numel()))).indices.tolist(); sg,eg,rg,cg=slices['gate_proj']; su,eu,ru,cu=slices['up_proj']; nbc=cg//4; x2=xA.square().mean(0)
 for br in row_groups:
  rows=slice(br*4,br*4+4); dj=D[:,rows]; gram=dj.T@dj
  for bc in range(nbc):
   cols=slice(bc*4,bc*4+4); qg=sg+br*nbc+bc; qu=su+br*nbc+bc; oldg=w['gate_proj'][rows,cols].clone(); oldu=w['up_proj'][rows,cols].clone(); scg=float(scales[qg]); scu=float(scales[qu]); aw=x2[cols].repeat(4).cpu(); cgids=candidates(pats,(base.blocks(orig['gate_proj'])[qg-sg]/scg).cpu()[None,:],aw[None,:],int(ids[qg]),qg); cuids=candidates(pats,(base.blocks(orig['up_proj'])[qu-su]/scu).cpu()[None,:],aw[None,:],int(ids[qu]),qu)
   RA=yA-tA[3]; RB=yB-tB[3]; projA=RA@dj; projB=RB@dj; baseLA=float(RA.square().sum()/ydenA); baseLB=float(RB.square().sum()/ydenB); best=(baseLA,baseLB,int(ids[qg]),int(ids[qu]),None)
   for ig in cgids:
    ng=(pats[ig]*scg+residual[qg]).cuda().view(4,4); dgA=xA[:,cols]@(ng-oldg).T; dgB=xB[:,cols]@(ng-oldg).T
    for iu in cuids:
     tested+=1; nu=(pats[iu]*scu+residual[qu]).cuda().view(4,4); duA=xA[:,cols]@(nu-oldu).T; duB=xB[:,cols]@(nu-oldu).T; nzA=torch.nn.functional.silu(gA[:,rows]+dgA)*(uA[:,rows]+duA); nzB=torch.nn.functional.silu(gB[:,rows]+dgB)*(uB[:,rows]+duB); dzA=nzA-zA[:,rows]; dzB=nzB-zB[:,rows]; deltaA=2*(dzA*projA).sum()+torch.einsum('ni,ij,nj->',dzA,gram,dzA); deltaB=2*(dzB*projB).sum()+torch.einsum('ni,ij,nj->',dzB,gram,dzB); la=baseLA+float(deltaA/ydenA); lb=baseLB+float(deltaB/ydenB)
     if la<best[0] and lb<=best[1]+1e-12: best=(la,lb,ig,iu,(ng,nu,dgA,dgB,duA,duB,nzA,nzB,dzA,dzB))
   if best[4] is not None:
    _,_,ig,iu,p=best; ng,nu,dgA,dgB,duA,duB,nzA,nzB,dzA,dzB=p; w['gate_proj'][rows,cols]=ng; w['up_proj'][rows,cols]=nu; ids[qg]=ig; ids[qu]=iu; gA[:,rows]+=dgA; gB[:,rows]+=dgB; uA[:,rows]+=duA; uB[:,rows]+=duB; zA[:,rows]=nzA; zB[:,rows]=nzB; yA+=dzA@dj.T; yB+=dzB@dj.T; changed.extend((qg,qu)); accepted+=1
 afterA=metric(zA,yA,tA[2],tA[3]); afterB=metric(zB,yB,tB[2],tB[3]); bfA=zA@ow['down_proj'].T; bfB=zB@ow['down_proj'].T; oracle={'Ey_A':float((bfA-tA[3]).norm()/tA[3].norm()),'Ey_B':float((bfB-tB[3]).norm()/tB[3].norm())}; patch_ids(INPUT,OUTPUT,ids); authoritative,_=base.decode(OUTPUT); hashes={n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in authoritative.items()}; base.restore(model,authoritative); ppl=base.perplexity(model,tok['evaluation'],256)
 out={'experiment':'E2_paired_gu','input':'E1_functional_5pct','bytes':OUTPUT.stat().st_size,'bpw':OUTPUT.stat().st_size*8/m0['weights'],'paired_row_groups':len(row_groups),'pairs_tested':tested,'block_pairs_accepted':accepted,'ids_changed':len(set(changed)),'before_A':beforeA,'before_B':beforeB,'after_A':afterA,'after_B':afterB,'bf16_down_oracle_after_pair':oracle,'ppl':ppl,'delta_ppl_pct':100*(ppl['ppl']/22.188272129960975-1),'hashes':hashes,'elapsed_seconds':time.time()-t0}; RES.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
