"""E1: strict functional-VQ smoke test and 5% hard-block search."""
from __future__ import annotations
import hashlib, json, math, os, struct, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import waverel_checkpoint_patterns as base

ROOT=Path('/workspace/experiments/functional_vq_l0')
CONTROL=Path(os.environ.get('FVQ_INPUT',ROOT/'packed/E0_control.packed')); TOKENS=ROOT/'calibration_cache/fixed_token_ids.pt'
EXPERIMENT=os.environ.get('FVQ_EXPERIMENT','E1_functional_5pct'); COVERAGE=float(os.environ.get('FVQ_COVERAGE','0.05'))
OUT=Path(os.environ.get('FVQ_OUTPUT',ROOT/f'packed/{EXPERIMENT}.packed')); RESULTS=ROOT/f'results/{EXPERIMENT}.json'

@torch.inference_mode()
def capture_x(model, ids):
    vals=[]; mod=model.model.layers[0].mlp.gate_proj
    h=mod.register_forward_pre_hook(lambda _m,inp: vals.append(inp[0].detach().float().cpu()))
    for s in range(0,ids.numel()-255,256): model(ids[s:s+256].unsqueeze(0).cuda(),use_cache=False)
    h.remove(); return torch.cat([v.reshape(-1,v.shape[-1]) for v in vals])

def mlp(w,x):
    g=x@w['gate_proj'].T; u=x@w['up_proj'].T; z=torch.nn.functional.silu(g)*u; return g,u,z,z@w['down_proj'].T

def patch_ids(src,dst,new_ids):
    blob=bytearray(src.read_bytes()); ml=struct.unpack_from('<I',blob,0)[0]; pos=4+ml
    rec=np.frombuffer(blob,dtype='<u2',count=new_ids.numel(),offset=pos).copy()
    rec=(rec&0xf000)|(new_ids.numpy().astype(np.uint16)&0x0fff); blob[pos:pos+2*len(rec)]=rec.astype('<u2').tobytes(); dst.write_bytes(blob)

def candidates(patterns,target,aw,current,seed):
    d=((patterns-target)**2).sum(1); top=torch.topk(d,64,largest=False).indices
    dw=(((patterns-target)**2)*aw).sum(1); a=torch.topk(dw,3,largest=False).indices.tolist(); b=torch.topk(d,3,largest=False).indices.tolist()
    chosen=list(dict.fromkeys([current]+b+a)); pool=patterns[top]; pn=torch.nn.functional.normalize(pool,dim=1); chosen_vec=torch.nn.functional.normalize(patterns[torch.tensor(chosen)],dim=1)
    for _ in range(2):
        diversity=1-(pn@chosen_vec.T).abs().max(1).values; order=torch.argsort(diversity,descending=True)
        pick=next(int(top[i]) for i in order if int(top[i]) not in chosen); chosen.append(pick); chosen_vec=torch.nn.functional.normalize(patterns[torch.tensor(chosen)],dim=1)
    return chosen[:8]

def main():
    started=time.time(); base.seed_all(); model=AutoModelForCausalLM.from_pretrained(base.MODEL,dtype=torch.float32,device_map={'':'cuda'}).eval()
    orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()}; ts=torch.load(TOKENS,map_location='cpu',weights_only=True); held=ts['heldout']; Aids=held[:512]; Bids=held[512:1024]; ev=ts['evaluation']
    base.restore(model,orig); xA=capture_x(model,Aids).cuda(); xB=capture_x(model,Bids).cuda(); teacherA=mlp({k:orig[k].cuda() for k in ('gate_proj','up_proj','down_proj')},xA); teacherB=mlp({k:orig[k].cuda() for k in ('gate_proj','up_proj','down_proj')},xB)
    control,meta=base.decode(CONTROL); m0,st,cp,codes,pids,sids,hard,rids,rq,rs=base.read_codec(CONTROL); fam=(torch.arange(codes.shape[0])//base.PATTERNS_PER_FAMILY).long(); patterns=cp[fam[:,None],codes.long()].float(); mi=torch.empty(m0['nblocks'],dtype=torch.long)
    slices={n:tuple(m0['slices'][n]) for n in m0['names']}
    for j,n in enumerate(m0['names']): mi[slices[n][0]:slices[n][1]]=j
    scales=st[mi,sids.long()].float(); residual=torch.zeros(m0['nblocks'],16)
    if hard is not None: residual[hard]=rq.float()[rids.long()]*rs.float()[rids.long(),None]
    current={n:control[n].cuda().clone() for n in control}; ids=pids.clone(); changed=[]; smoke=[]
    # Cache current student activations.
    gA,uA,zA,yA=mlp(current,xA); gB,uB,zB,yB=mlp(current,xB); zdenA=teacherA[2].square().sum(); zdenB=teacherB[2].square().sum(); ydenA=teacherA[3].square().sum(); ydenB=teacherB[3].square().sum()
    def errs(): return {'Ez_A':float((zA-teacherA[2]).norm()/teacherA[2].norm()),'Ez_B':float((zB-teacherB[2]).norm()/teacherB[2].norm()),'Ey_A':float((yA-teacherA[3]).norm()/teacherA[3].norm()),'Ey_B':float((yB-teacherB[3]).norm()/teacherB[3].norm())}
    before=errs(); gen=torch.Generator().manual_seed(123)
    # Search 5% highest-error row groups for Gate then Up. Exact local Lz updates.
    neuron=((zA-teacherA[2]).square().mean(0)*orig['down_proj'].norm(dim=0).cuda().square()).view(-1,4).mean(1); top_rows=torch.topk(neuron,max(1,int(COVERAGE*neuron.numel()))).indices.tolist()
    for proj in ('gate_proj','up_proj'):
      s,e,r,c=slices[proj]; nbc=c//4; x2=xA.square().mean(0)
      for br in top_rows:
       rows=slice(br*4,br*4+4)
       for bc in range(nbc):
        q=s+br*nbc+bc; cols=slice(bc*4,bc*4+4); old=current[proj][rows,cols].clone(); sc=float(scales[q]); target=base.blocks(orig[proj])[q-s]/sc; aw=x2[cols].repeat(4).cpu(); cand=candidates(patterns,target.cpu()[None,:],aw[None,:],int(ids[q]),q)
        best=int(ids[q]); bestA=float(((zA[:,rows]-teacherA[2][:,rows]).square().sum())/zdenA); bestB=float(((zB[:,rows]-teacherB[2][:,rows]).square().sum())/zdenB); bestpack=None
        for cid in cand:
          nb=(patterns[cid]*sc+residual[q]).cuda().view(4,4)
          dA=xA[:,cols]@(nb-old).T; dB=xB[:,cols]@(nb-old).T
          if proj=='gate_proj': nzA=torch.nn.functional.silu(gA[:,rows]+dA)*uA[:,rows]; nzB=torch.nn.functional.silu(gB[:,rows]+dB)*uB[:,rows]
          else: nzA=torch.nn.functional.silu(gA[:,rows])*(uA[:,rows]+dA); nzB=torch.nn.functional.silu(gB[:,rows])*(uB[:,rows]+dB)
          la=float(((nzA-teacherA[2][:,rows]).square().sum())/zdenA); lb=float(((nzB-teacherB[2][:,rows]).square().sum())/zdenB)
          if la<bestA and lb<=bestB+1e-12: best,bestA,bestB,bestpack=cid,la,lb,(nb,dA,dB,nzA,nzB)
        if bestpack:
          nb,dA,dB,nzA,nzB=bestpack; current[proj][rows,cols]=nb; ids[q]=best; changed.append(q)
          if proj=='gate_proj': gA[:,rows]+=dA; gB[:,rows]+=dB
          else: uA[:,rows]+=dA; uB[:,rows]+=dB
          zA[:,rows]=nzA; zB[:,rows]=nzB
      # refresh Down outputs after each projection sweep
      yA=zA@current['down_proj'].T; yB=zB@current['down_proj'].T
    after_gu=errs()
    # Search 5% worst Down output-row groups, exact local Ly.
    outscore=(yA-teacherA[3]).square().mean(0).view(-1,4).mean(1); top_out=torch.topk(outscore,max(1,int(COVERAGE*outscore.numel()))).indices.tolist(); s,e,r,c=slices['down_proj']; nbc=c//4
    for br in top_out:
     rows=slice(br*4,br*4+4)
     for bc in range(nbc):
      q=s+br*nbc+bc; cols=slice(bc*4,bc*4+4); old=current['down_proj'][rows,cols].clone(); sc=float(scales[q]); target=base.blocks(orig['down_proj'])[q-s]/sc; aw=zA[:,cols].square().mean(0).repeat(4).cpu(); cand=candidates(patterns,target.cpu()[None,:],aw[None,:],int(ids[q]),q)
      best=int(ids[q]); bestA=float(((yA[:,rows]-teacherA[3][:,rows]).square().sum())/ydenA); bestB=float(((yB[:,rows]-teacherB[3][:,rows]).square().sum())/ydenB); bestpack=None
      for cid in cand:
       nb=(patterns[cid]*sc+residual[q]).cuda().view(4,4); dyA=zA[:,cols]@(nb-old).T; dyB=zB[:,cols]@(nb-old).T; la=float(((yA[:,rows]+dyA-teacherA[3][:,rows]).square().sum())/ydenA); lb=float(((yB[:,rows]+dyB-teacherB[3][:,rows]).square().sum())/ydenB)
       if la<bestA and lb<=bestB+1e-12: best,bestA,bestB,bestpack=cid,la,lb,(nb,dyA,dyB)
      if bestpack:
       nb,dyA,dyB=bestpack; current['down_proj'][rows,cols]=nb; ids[q]=best; changed.append(q); yA[:,rows]+=dyA; yB[:,rows]+=dyB
    after_down=errs(); patch_ids(CONTROL,OUT,ids); decoded,_=base.decode(OUT)
    # Packed decode is authoritative; evaluate only it.
    hashes={n:hashlib.sha256(decoded[n].numpy().tobytes()).hexdigest() for n in decoded}; base.restore(model,decoded); ppl=base.perplexity(model,ev,256)
    result={'experiment':EXPERIMENT,'input':str(CONTROL),'coverage':COVERAGE,'control_bytes':CONTROL.stat().st_size,'functional_bytes':OUT.stat().st_size,'bpw':OUT.stat().st_size*8/m0['weights'],'searched_gate_up_row_groups':len(top_rows),'searched_down_row_groups':len(top_out),'changed_ids':len(set(changed)),'changed_pct_all_blocks':100*len(set(changed))/m0['nblocks'],'before':before,'after_gate_up':after_gu,'after_down':after_down,'ppl':ppl,'delta_ppl_pct':100*(ppl['ppl']/22.188272129960975-1),'decoded_hashes':hashes,'elapsed_seconds':time.time()-started}
    RESULTS.write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
