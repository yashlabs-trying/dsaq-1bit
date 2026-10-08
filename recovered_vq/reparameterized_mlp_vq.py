"""Function-preserving Up/Down reparameterization followed by the frozen Layer-0 VQ codec."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import waverel_checkpoint_patterns as base

def args():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,default=Path('/workspace/reparameterized_vq'))
    p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt'))
    p.add_argument('--sequence-length',type=int,default=256); p.add_argument('--hard-fraction',type=float,default=.20)
    return p.parse_args()

def make_reparam(orig, imp):
    # Old-index scales; permutation is applied after scaling metadata is computed.
    ug=orig['up_proj']; dc=orig['down_proj'].T
    a=(dc.norm(dim=1).clamp_min(1e-8)/ug.norm(dim=1).clamp_min(1e-8)).sqrt()
    order=torch.argsort(imp,descending=True)
    cases=[]
    for name,perm in [('identity',torch.arange(ug.shape[0])),('norm_equalized',torch.arange(ug.shape[0])),('norm_equalized_importance_sorted',order)]:
        scales=torch.ones_like(a) if name=='identity' else a
        G=orig['gate_proj'][perm].clone(); U=orig['up_proj'][perm]*scales[perm,None]; D=orig['down_proj'][:,perm]/scales[perm][None,:]
        vals={k:v.clone() for k,v in orig.items()}; vals.update({'gate_proj':G,'up_proj':U,'down_proj':D})
        cases.append((name,vals,perm,scales))
    return cases

def fisher_reparam(f,perm,scales):
    out={k:v.clone() for k,v in f.items()}
    out['gate_proj']=f['gate_proj'][perm]
    out['up_proj']=f['up_proj'][perm]/scales[perm,None].square()
    out['down_proj']=f['down_proj'][:,perm]*scales[perm][None,:].square()
    return out

@torch.inference_mode()
def mlp_out(vals,x):
    return (torch.nn.functional.silu(x@vals['gate_proj'].T)*(x@vals['up_proj'].T))@vals['down_proj'].T

def train_one(name,vals,perm,scales,orig,fisher,model,bf16,eval_ids,acts,seq,outdir,hard_fraction,premax):
    started=time.time(); tf=fisher_reparam(fisher,perm,scales)
    vecs,imp,mi,slices=base.extract(vals,tf); raw=vecs.abs().amax(1).clamp_min(1e-12)
    st,si,ds=base.quantize_scales(raw,mi,len(vals)); norm=vecs/raw[:,None]; target=vecs/ds[:,None]
    base.seed_all(); cp,codes,pats,ids,_=base.train_waverel_patterns(norm,target,imp)
    pb=pats[ids]*ds[:,None]; mask,rids,rq,rs,final,_=base.train_hard_residual(vecs,imp,pb,hard_fraction)
    path=outdir/(name+'.bin'); base.write_codec(path,name,slices,st,si,cp,codes,ids,mask,rids,rq,rs,meta_extra={'codec_experiment':'function_preserving_reparam_v1','reparam':name})
    disk,_=base.decode(path)
    for proj,(s,e,shape) in slices.items(): assert torch.equal(disk[proj],base.unblock(final[s:e],shape)),proj
    base.apply(model,disk); ppl=base.perplexity(model,eval_ids,seq); act,pm=base.activation_metrics(vals,disk,acts)
    qout=mlp_out(disk,acts['gate_proj'][:0]) if False else None
    bpw=path.stat().st_size*8/sum(w.numel() for w in orig.values())
    return {'method':name,'bpw':bpw,'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'ppl':ppl['ppl'],'delta_ppl_pct':100*(ppl['ppl']/bf16['ppl']-1),'activation_error':act,'projection_metrics':pm,'prequant_function_max_abs':premax,'elapsed_seconds':time.time()-started}

def main():
    a=args(); a.output.mkdir(parents=True,exist_ok=True); base.seed_all()
    tok=AutoTokenizer.from_pretrained(base.MODEL); model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval()
    orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()}
    ts=torch.load(a.tokens,map_location='cpu',weights_only=True); fit,held,evaluation=ts['fit'],ts['heldout'],ts['evaluation']
    hashes={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in ts.items()}; base.restore(model,orig); bf16=base.perplexity(model,evaluation,a.sequence_length); acts=base.capture_inputs(model,held,a.sequence_length)
    legacy={f"L0.{'SA' if n in base.PROJS[:4] else 'MLP'}.{n}":w for n,w in orig.items()}; fl=base.legacy.compute_fisher(model,legacy); model.to('cuda'); fisher={n:fl[f"L0.{'SA' if n in base.PROJS[:4] else 'MLP'}.{n}"].cpu() for n in orig}
    x=acts['gate_proj'][:512]; imp=(torch.nn.functional.silu(x@orig['gate_proj'].T).abs()*(x@orig['up_proj'].T).abs()).mean(0)*orig['down_proj'].norm(dim=0); cases=make_reparam(orig,imp)
    orig_m=mlp_out(orig,x)
    results=[]
    for name,vals,perm,scales in cases:
        pre=float((mlp_out(vals,x)-orig_m).abs().max()); print('\n===',name,'premax',pre,flush=True); base.restore(model,orig); r=train_one(name,vals,perm,scales,orig,fisher,model,bf16,evaluation,acts,a.sequence_length,a.output,a.hard_fraction,pre); results.append(r); print(json.dumps(r,indent=2),flush=True)
    report={'model':base.MODEL,'layer':0,'bf16':bf16,'fixed_token_hashes':hashes,'results':results,'decision_rule':'function-preserving reparameterization must improve held-out PPL at same packed BPW'}; base.dump(a.output/'results.json',report); print('FINAL\n'+json.dumps(report,indent=2),flush=True)

if __name__=='__main__': main()
