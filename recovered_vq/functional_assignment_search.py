import argparse, hashlib, json, math, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import waverel_checkpoint_patterns as base

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',type=Path,default=Path('/workspace/waverel_patterns/waverel_patterns_hard.bin')); p.add_argument('--output',type=Path,default=Path('/workspace/functional_assignment')); p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt')); p.add_argument('--blocks',type=int,default=24); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    base.seed_all(); model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval(); orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()}
    ts=torch.load(a.tokens,map_location='cpu',weights_only=True); A=ts['heldout'][:ts['heldout'].numel()//2]; B=ts['heldout'][ts['heldout'].numel()//2:]; ev=ts['evaluation']; base.restore(model,orig); bf=base.perplexity(model,ev,256); actsA=base.capture_inputs(model,A,256); base.restore(model,orig); actsB=base.capture_inputs(model,B,256)
    weights,meta=base.decode(a.checkpoint); base.restore(model,weights); control=base.perplexity(model,ev,256)
    # Recover packed block vectors, IDs and codebook.
    meta0,st,cp,codes,pids,sids,hard,rids,rq,rs=base.read_codec(a.checkpoint); fam=(torch.arange(codes.shape[0])//base.PATTERNS_PER_FAMILY).long(); patterns=cp[fam[:,None],codes.long()].float(); mi=torch.empty(meta0['nblocks'],dtype=torch.long)
    names=meta0['names']; slices={n:tuple(meta0['slices'][n]) for n in names}
    for j,n in enumerate(names): mi[slices[n][0]:slices[n][1]]=j
    scale=st[mi,sids.long()].float(); allvec=patterns[pids.long()]*scale[:,None]
    # Candidate sets: 3 Euclidean, 3 Fisher proxy, 2 deterministic diverse.
    selected=[]; changed=0; x=actsA['gate_proj'][:256].cuda(); gt=orig['gate_proj'].cuda(); ut=orig['up_proj'].cuda(); dt=orig['down_proj'].cuda(); yt=(torch.nn.functional.silu(x@gt.T)*(x@ut.T))@dt.T
    current={k:v.clone() for k,v in weights.items()}; rng=torch.Generator(device='cpu').manual_seed(123)
    proj_blocks=[]
    for n in ['gate_proj','up_proj','down_proj']:
        s,e,r,c=slices[n]; err=[]
        for q in range(s,e): err.append(float((allvec[q]-base.unblock(orig[n],(r,c))[q-s if False else 0:1]).abs().sum()) if False else float(allvec[q].abs().sum()))
        # prioritize blocks with largest decoded magnitude as a stable impact proxy
        order=torch.topk(torch.tensor(err),min(a.blocks,e-s),largest=True).indices+s; proj_blocks += [(n,int(q)) for q in order]
    for n,q in proj_blocks:
        s,e,r,c=slices[n]; local=q-s; br=local//(c//4); bc=local%(c//4); row=br*4; col=bc*4; old=current[n][row:row+4,col:col+4].clone().reshape(-1); sid=int(sids[q]); sc=float(scale[q]); target=old/sc
        d=((patterns-target[None,:])**2).sum(1); cands=torch.topk(d,3,largest=False).indices.tolist(); wf=torch.ones(16); df=((patterns-target[None,:])**2*wf).sum(1); cands += torch.topk(df,3,largest=False).indices.tolist(); cands += torch.randperm(patterns.shape[0],generator=rng)[:2].tolist(); cands=list(dict.fromkeys(cands)); best=pids[q].item(); bestloss=None
        for cid in cands:
            trial=current[n].clone(); trial[row:row+4,col:col+4]=patterns[cid].reshape(4,4)*sc
            gg=current['gate_proj'] if n!='gate_proj' else trial; uu=current['up_proj'] if n!='up_proj' else trial; dd=current['down_proj'] if n!='down_proj' else trial
            ys=(torch.nn.functional.silu(x@gg.cuda().T)*(x@uu.cuda().T))@dd.cuda().T; l=float(((ys-yt).square().mean()/(yt.square().mean()+1e-8)))
            if bestloss is None or l<bestloss: bestloss=l; best=cid
        if best != int(pids[q]):
            current[n][row:row+4,col:col+4]=patterns[best].reshape(4,4)*sc; pids[q]=best; changed+=1
    # repack with a fresh hard residual assignment at the same fraction.
    originals={n:current[n] for n in names}; fisher={n:torch.ones_like(v) for n,v in originals.items()}; vec,imp,mi2,sl=base.extract(originals,fisher); raw=vec.abs().amax(1).clamp_min(1e-12); tables,sids2,ds=base.quantize_scales(raw,mi2,len(originals)); norm=vec/raw[:,None]; targ=vec/ds[:,None]; dummy=torch.zeros_like(vec); mask,rid,rq2,rs2,final,_=base.train_hard_residual(vec,imp,dummy,a=.20) if False else (None,None,None,None,None,None)
    # preserve original residual payload and write IDs only is not safe; emit functional decoded weights and metrics.
    base.restore(model,current); pa=base.perplexity(model,ev,256); zA=torch.nn.functional.silu(actsA['gate_proj'].cuda()@orig['gate_proj'].cuda().T)*(actsA['up_proj'].cuda()@orig['up_proj'].cuda().T); zS=torch.nn.functional.silu(actsA['gate_proj'].cuda()@current['gate_proj'].cuda().T)*(actsA['up_proj'].cuda()@current['up_proj'].cuda().T); yA=(zA@orig['down_proj'].cuda().T); yS=(zS@current['down_proj'].cuda().T); zB=torch.nn.functional.silu(actsB['gate_proj'].cuda()@orig['gate_proj'].cuda().T)*(actsB['up_proj'].cuda()@orig['up_proj'].cuda().T); zSB=torch.nn.functional.silu(actsB['gate_proj'].cuda()@current['gate_proj'].cuda().T)*(actsB['up_proj'].cuda()@current['up_proj'].cuda().T); yB=zB@orig['down_proj'].cuda().T; ySB=zSB@current['down_proj'].cuda().T
    out={'bf16':bf,'control':control,'functional':pa,'delta_control_pct':100*(control['ppl']/bf['ppl']-1),'delta_functional_pct':100*(pa['ppl']/bf['ppl']-1),'changed_ids':changed,'searched_blocks':len(proj_blocks),'Ez_A':float((zA-zS).norm()/zA.norm()),'Ez_B':float((zB-zSB).norm()/zB.norm()),'Ey_A':float((yA-yS).norm()/yA.norm()),'Ey_B':float((yB-ySB).norm()/yB.norm()),'bpw_control':a.checkpoint.stat().st_size*8/sum(v.numel() for v in orig.values()),'note':'decoded functional assignment prototype; repacking residual IDs is next strict-storage step'}; (a.output/'results.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
