import sys,json,math,time,traceback,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from wavefly.core import basis,select
from wavefly.permutations import *
from wavefly.clustering import *

torch.manual_seed(20260922);np.random.seed(20260922);torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/clustering'
for d in ['csv','json','heatmaps','plots']:(OUT/d).mkdir(parents=True,exist_ok=True)
KS=[4,8,16,24,32,48,64,96,128,192,256];DENS=[0,.001,.0025,.005,.01,.02,.05]
ROWS=[];METHODS=[];FAIL=[]

def tiles(w,b):
    nr,nc=w.shape[0]//b,w.shape[1]//b;z=w[:nr*b,:nc*b]
    return z.reshape(nr,b,nc,b).permute(0,2,1,3).reshape(-1,b,b),nr,nc

def untile(t,nr,nc,b):return t.reshape(nr,nc,b,b).permute(0,2,1,3).reshape(nr*b,nc*b)
def tv(w):return float((w[:,1:]-w[:,:-1]).abs().sum()/w.numel()+(w[1:]-w[:-1]).abs().sum()/w.numel())
def adj(v):
    z=v.float()/v.float().norm(dim=1,keepdim=True).clamp_min(1e-12)
    return float((z[:-1]*z[1:]).sum(1).mean()) if len(z)>1 else 1.

def evaluate(w,x,name,method,pr=None,pc=None,b=16,scope='validation',macro=None,save_heat=False):
    ok,undo=verify(w,pr,pc);assert ok and undo==0,(name,method,undo)
    wp=apply_permutation(w,pr,pc);t,nr,nc=tiles(wp,b);d=basis(b);c=d@t.float()@d.T
    denom=c.square().sum().double(); base=dict(layer=name,method=method,scope=scope,tile=b,rows=w.shape[0],cols=w.shape[1],tiles=len(t),tv=tv(wp),row_corr=adj(wp),col_corr=adj(wp.T),
        permutation_bytes=permutation_bytes(pr,pc,macro),permutation_bpw=8*permutation_bytes(pr,pc,macro)/w.numel(),permutation_roundtrip_max=undo)
    if save_heat:
        safe=(name+'__'+method+f'_b{b}').replace('.','_').replace('/','_').replace('\\','_').replace('@','_')
        np.save(OUT/'heatmaps'/(safe+'.npy'),c.abs().mean(0).cpu().numpy())
    y=x@w.T
    for selection in ['fixed','topk']:
      for k in [q for q in KS if q<=b*b]:
        ix=select(c,k,selection);vals=c.flatten(1).gather(1,ix);energy=float(vals.square().sum().double()/denom)
        cc=torch.zeros(len(t),b*b,device=w.device).scatter_(1,ix,vals);hp=untile(d.T@cc.reshape_as(t)@d,nr,nc,b)
        h=undo_permutation(hp,pr,pc);err=w-h;yh=x@h.T
        common=dict(base,selection=selection,k=k,residual_requested=0,residual_actual=0,residual_energy=0,
          energy=energy,weight_error=float(err.norm()/w.norm()),max_error=float(err.abs().max()),cosine=float(torch.nn.functional.cosine_similarity(w.flatten(),h.flatten(),dim=0)),
          output_error=float((y-yh).norm()/y.norm()),output_cosine=float(torch.nn.functional.cosine_similarity(y.flatten(),yh.flatten(),dim=0)))
        ROWS.append(common)
        if selection=='topk' and k==min(32,b*b):
          flat=err.flatten();order=flat.abs().argsort(descending=True);total=flat.square().sum().clamp_min(1e-30)
          for density in DENS[1:]:
            count=round(density*w.numel());ids=order[:count];hh=h.clone();hh.flatten()[ids]+=flat[ids]
            yy=x@hh.T
            ROWS.append(dict(common,residual_requested=density,residual_actual=count/w.numel(),residual_energy=float(flat[ids].square().sum()/total),
              weight_error=float((w-hh).norm()/w.norm()),max_error=float((w-hh).abs().max()),cosine=float(torch.nn.functional.cosine_similarity(w.flatten(),hh.flatten(),dim=0)),
              output_error=float((y-yy).norm()/y.norm()),output_cosine=float(torch.nn.functional.cosine_similarity(y.flatten(),yy.flatten(),dim=0))))
    pd.DataFrame(ROWS).to_csv(OUT/'csv/measurements.csv',index=False)

def make_perm(w,kind,k=16,norm=True,within='pca',ain=None,aout=None,alpha=1.,macro=None):
    pr=pc=None;seed=20260922
    if kind in ['row','both','activation','magnitude','correlation']:
        ex=None
        if kind=='activation' and aout is not None:ex=alpha*(aout-aout.mean()).div(aout.std().clamp_min(1e-8))[:,None]
        if kind=='magnitude':ex=magnitude_features(w,alpha)
        pr=(local_permutation(w,k,macro,norm,within,ex,seed) if macro else ordered_permutation(w,k,norm,within,ex,seed))
    wr=w if pr is None else w[pr]
    if kind in ['col','both','activation','magnitude','correlation']:
        ex=None
        if kind=='activation' and ain is not None:ex=alpha*(ain-ain.mean()).div(ain.std().clamp_min(1e-8))[:,None]
        if kind=='magnitude':ex=magnitude_features(wr.T,alpha)
        pc=(local_permutation(wr.T,k,macro,norm,within,ex,seed+1) if macro else ordered_permutation(wr.T,k,norm,within,ex,seed+1))
    return pr,pc

def discovery(w,x,ain,aout,name):
    candidates=[dict(label='raw',kind='raw')]
    for kind in ['row','col']:
      for norm in [False,True]:
       for k in [4,8,16,32,64]:
        for within in ['centroid','norm','pca']:candidates.append(dict(label=f'{kind}_k{k}_{"norm" if norm else "raw"}_{within}',kind=kind,k=k,norm=norm,within=within))
    for k in [4,8,16,32,64]:
      for within in ['centroid','norm','pca']:candidates.append(dict(label=f'both_k{k}_norm_{within}',kind='both',k=k,norm=True,within=within))
    for k in [8,16,32,64]:candidates.append(dict(label=f'correlation_k{k}',kind='correlation',k=k,norm=True,within='pca'))
    for alpha in [.1,1.,3.]:candidates.append(dict(label=f'activation_a{alpha}',kind='activation',k=16,norm=True,within='pca',alpha=alpha))
    for alpha in [.25,1.]:candidates.append(dict(label=f'magnitude_a{alpha}',kind='magnitude',k=16,norm=True,within='pca',alpha=alpha))
    for macro in [128,256]:candidates.append(dict(label=f'local{macro}_both',kind='both',k=16,norm=True,within='pca',macro=macro))
    for i,cfg in enumerate(candidates):
        st=time.time();pr,pc=(None,None) if cfg['kind']=='raw' else make_perm(w,ain=ain,aout=aout,**{k:v for k,v in cfg.items() if k!='label'})
        METHODS.append(dict(layer=name,seconds=time.time()-st,**cfg,perm_bytes=permutation_bytes(pr,pc,cfg.get('macro'))))
        evaluate(w,x,name,cfg['label'],pr,pc,16,'discovery',cfg.get('macro'),save_heat=True)
        print('DISCOVERY',i+1,len(candidates),cfg['label'],flush=True)
    pd.DataFrame(METHODS).to_csv(OUT/'csv/methods.csv',index=False)
    z=pd.DataFrame(ROWS);z=z[(z.scope=='discovery')&(z.selection=='topk')&(z.k==32)&(z.residual_actual==0)]
    winner=z.sort_values('output_error').iloc[0].method
    best=next(c for c in candidates if c['label']==winner)
    pr,pc=make_perm(w,ain=ain,aout=aout,**{k:v for k,v in best.items() if k!='label'})
    wp=apply_permutation(w,pr,pc)
    # Display arrays are capped only for plotting; all metrics use the complete matrix.
    np.save(OUT/'heatmaps/discovery_raw_weight.npy',w[:512,:512].cpu().numpy())
    np.save(OUT/'heatmaps/discovery_clustered_weight.npy',wp[:512,:512].cpu().numpy())
    np.savez(OUT/'json/discovery_permutations.npz',rows=pr.cpu().numpy() if pr is not None else np.array([],dtype=np.int64),cols=pc.cpu().numpy() if pc is not None else np.array([],dtype=np.int64))
    return best

TEXTS=['The numerical experiment compares weight compression and functional error across transformer layers.',
'Plants convert sunlight into chemical energy while ecosystems exchange matter through many connected processes.',
'Write a Python function that sums a list, handles an empty input, and returns a numeric result.',
'A shop sold twelve items on Monday and eighteen on Tuesday for a total of thirty items.',
'The traveler watched rain cross the fields while reading a history of mathematics and modern science.']

def get_layers(model,tok,mid):
    count=model.config.num_hidden_layers;stages={0:'early',count//2:'middle',count-1:'late'};acts={};outs={};mods={};hooks=[]
    # Qwen receives full seven-type coverage at all positions; other families validate representative attention/MLP types.
    for name,m in model.named_modules():
      li=next((int(p) for p in name.split('.') if p.isdigit()),None)
      if not isinstance(m,torch.nn.Linear) or li not in stages:continue
      keep=(mid.startswith('Qwen') or li==count//2)
      if keep:
        mods[name]=(m,stages[li]);acts[name]=[];outs[name]=[]
        def hook(mod,args,out,n=name):
            acts[n].append(args[0].detach().reshape(-1,args[0].shape[-1])[:64].cpu());outs[n].append(out.detach().reshape(-1,out.shape[-1])[:64].cpu())
        hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
      for text in TEXTS:model(**tok(text,return_tensors='pt',truncation=True,max_length=128).to('cuda'))
    for h in hooks:h.remove()
    result=[]
    for name,(m,stage) in mods.items():
      x=torch.cat(acts[name])[:256].cuda();o=torch.cat(outs[name])[:256].cuda();w=m.weight.detach().float()
      if name.endswith('query_key_value'):
        heads=model.config.num_attention_heads;hd=w.shape[1]//heads;v=w.reshape(heads,3,hd,-1)
        for j,label in enumerate(['q_proj','k_proj','v_proj']):result.append((name+'.'+label,v[:,j].reshape(-1,v.shape[-1]),x,stage,x.abs().mean(0),o.reshape(len(o),heads,3,hd)[:,:,j].reshape(len(o),-1).abs().mean(0)))
      else:result.append((name,w,x,stage,x.abs().mean(0),o.abs().mean(0)))
    return result

def main():
 from transformers import AutoModelForCausalLM,AutoTokenizer
 specs=['Qwen/Qwen2.5-0.5B','facebook/opt-125m','EleutherAI/pythia-70m'];winner=None;models=[]
 for mid in specs:
  print('LOAD',mid,flush=True)
  try:
   tok=AutoTokenizer.from_pretrained(mid);model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
   layers=get_layers(model,tok,mid);models.append(dict(model=mid,revision=getattr(model.config,'_commit_hash',None),layers=len(layers)))
   for ix,(lname,w,x,stage,ain,aout) in enumerate(layers):
    full=mid+'/'+lname
    if winner is None and 'q_proj' in lname:
        winner=discovery(w,x,ain,aout,full);(OUT/'json/winner.json').write_text(json.dumps(winner,indent=2))
    # Raw and discovered winner on every validation layer.
    evaluate(w,x,full+'@'+stage,'raw',scope='validation',save_heat=(ix<2))
    pr,pc=make_perm(w,ain=ain,aout=aout,**{k:v for k,v in winner.items() if k!='label'})
    evaluate(w,x,full+'@'+stage,'winner_'+winner['label'],pr,pc,16,'validation',winner.get('macro'),save_heat=(ix<2))
    # Strongest method tile-size sweep on Qwen middle q projection.
    if mid.startswith('Qwen') and stage=='middle' and lname.endswith('q_proj'):
      for b in [8,32,64]:evaluate(w,x,full+'@'+stage,'winner_'+winner['label'],pr,pc,b,'tile_sweep',winner.get('macro'),save_heat=True)
    print('VALIDATE',mid,ix+1,len(layers),lname,flush=True)
   del model,layers,w,x,ain,aout;torch.cuda.empty_cache()
  except Exception:
   FAIL.append(dict(model=mid,traceback=traceback.format_exc()));print(FAIL[-1],flush=True)
 (OUT/'json/models.json').write_text(json.dumps(models,indent=2));(OUT/'json/failures.json').write_text(json.dumps(FAIL,indent=2));pd.DataFrame(ROWS).to_csv(OUT/'csv/measurements.csv',index=False)
 env=dict(seed=20260922,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,python=sys.version,tf32=False)
 (OUT/'json/environment.json').write_text(json.dumps(env,indent=2));print('CLUSTERING DONE',len(ROWS),winner,flush=True)

if __name__=='__main__':main()
