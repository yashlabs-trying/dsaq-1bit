"""Function-preserving MLP reparameterization followed by the existing VQ codec."""
from __future__ import annotations
import argparse, hashlib, json, math, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base

MLP=("gate_proj","up_proj","down_proj"); ATTN=("q_proj","k_proj","v_proj","o_proj")

def reparam(orig, perm, scales):
    g,u,d=orig['gate_proj'],orig['up_proj'],orig['down_proj'];p=perm.long();a=scales[p]
    return {**{n:orig[n] for n in ATTN},'gate_proj':g[p],'up_proj':u[p]*a[:,None],'down_proj':d[:,p]/a[None,:]}

def transform_fisher(f,perm,scales):
    p=perm.long();a=scales[p]
    return {**{n:f[n] for n in ATTN},'gate_proj':f['gate_proj'][p], 'up_proj':f['up_proj'][p]/a[:,None].square(), 'down_proj':f['down_proj'][:,p]*a[None,:].square()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt'));p.add_argument('--output',type=Path,default=Path('/workspace/reparameterized_vq'));p.add_argument('--sequence-length',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);base.seed_all();start=time.time()
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval();orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()};total=sum(w.numel() for w in orig.values());tok=torch.load(a.tokens,map_location='cpu',weights_only=True);held,evaluation=tok['heldout'],tok['evaluation'];base.restore(model,orig);ref=base.perplexity(model,evaluation,a.sequence_length);acts=base.capture_inputs(model,held,a.sequence_length)
    names={f"L0.{'SA' if n in ATTN else 'MLP'}.{n}":w for n,w in orig.items()};f0=base.legacy.compute_fisher(model,names);model.to('cuda');fisher={n:f0[f"L0.{'SA' if n in ATTN else 'MLP'}.{n}"].cpu() for n in orig}
    x=acts['gate_proj'];g,u,d=(orig[n].cuda() for n in MLP);imp=(torch.nn.functional.silu(x.cuda()@g.T)*(x.cuda()@u.T)).abs().mean(0).cpu()*d.norm(dim=0).cpu();normscale=(d.norm(dim=0).clamp_min(1e-12)/u.norm(dim=1).clamp_min(1e-12)).sqrt();identity=torch.arange(len(normscale));cases=[('identity',identity,torch.ones_like(normscale)),('norm_equalized',identity,normscale),('norm_equalized_importance_sorted',torch.argsort(imp,descending=True),normscale)]
    rows=[]
    for name,perm,scales in cases:
        transformed=reparam(orig,perm,scales);tf=transform_fisher(fisher,perm,scales);base.restore(model,orig);# exact pre-quantization function check
        with torch.no_grad():
            xo=x.cuda();o=torch.nn.functional.silu(xo@g.T)*(xo@u.T)@d.T;gg,uu,dd=(transformed[n].cuda() for n in MLP);q=torch.nn.functional.silu(xo@gg.T)*(xo@uu.T)@dd.T;exact=float((o-q).abs().max())
        vec,im,mi,slices=base.extract(transformed,tf);raw=vec.abs().amax(1).clamp_min(1e-12);st,si,ds=base.quantize_scales(raw,mi,len(transformed));z=vec/raw[:,None];target=vec/ds[:,None];base.seed_all();cp,codes,pats,ids,pf=base.train_waverel_patterns(z,target,im);pb=pats[ids.long()]*ds[:,None];mask,rids,rq,rs,final,_=base.train_hard_residual(vec,im,pb,.20);path=a.output/(name+'.bin');base.write_codec(path,name,slices,st,si,cp,codes,ids,mask,rids,rq,rs,meta_extra={'reparameterization':name,'scale_min':float(scales.min()),'scale_max':float(scales.max())});recon,_=base.decode(path);base.apply(model,recon);m=base.perplexity(model,evaluation,a.sequence_length);act,per=base.activation_metrics(transformed,recon,acts);# compare decoded outputs to original function
        qg,qu,qd=(recon[n].cuda() for n in MLP);qo=torch.nn.functional.silu(xo@qg.T)*(xo@qu.T)@qd.T;func_err=math.sqrt(float((qo-o).square().sum()/o.square().sum()));row={'method':name,'bpw':path.stat().st_size*8/total,'ppl':m['ppl'],'delta_ppl_pct':100*(m['ppl']/ref['ppl']-1),'prequant_exact_max_abs':exact,'postquant_mlp_function_error':func_err,'projection_activation_error':per,'scale_min':float(scales.min()),'scale_max':float(scales.max()),'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};rows.append(row);print(json.dumps(row,indent=2),flush=True)
    report={'model':base.MODEL,'layer':0,'reference':ref,'results':rows,'constraints':{'max_bpw':1.2,'block':'4x4','patterns':base.PATTERNS,'hard_fraction':.20},'elapsed_seconds':time.time()-start};base.dump(a.output/'results.json',report);print('FINAL\n'+json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
