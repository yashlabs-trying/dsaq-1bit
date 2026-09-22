import sys,json,math,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from wavefly.wavelet import haar_basis,band_index
from wavefly.clustering import ordered_permutation
from wavefly.permutations import permutation_bytes

torch.manual_seed(20260922);np.random.seed(20260922);torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/v4'
for d in ['csv','json','plots','heatmaps']:(OUT/d).mkdir(parents=True,exist_ok=True)
FRACTIONS=[0,.005,.01,.02,.025,.05];PRECISIONS=[4,8,16];CORE_GROUPS=[32,64];BITS=[1.,1.5,2.]
ROWS=[];SAL=[];STABILITY=[];DIAG=[];BAND=[];RES=[];FAIL=[];ORIENT=[];SALIENT_SETS=[];PREP_CACHE={}

def outputs(w,h,xtr,xte,bias):
 def one(x):
  y=x@w.T;yh=x@h.T
  if bias is not None:y=y+bias;yh=yh+bias
  e=y-yh;return float(e.norm()/y.norm().clamp_min(1e-30)),float(torch.nn.functional.cosine_similarity(y.flatten(),yh.flatten(),dim=0)),float(e.abs().max()),float(e.square().mean())
 return (*one(xtr),*one(xte))

def saliency_scores(w,x):
 wn=w.double().square().sum(0).sqrt();ar=x.double().square().mean(0).sqrt();functional=wn.square()*x.double().square().sum(0)
 return {'weight_norm':wn.float(),'activation_rms':ar.float(),'weight_x_activation':(wn*ar).float(),'functional':functional.float()}
def top_columns(score,fraction,block=1):
 n=len(score);k=round(n*fraction)
 if k==0:return torch.empty(0,dtype=torch.long,device=score.device)
 if block==1:return score.topk(k).indices.sort().values
 ng=math.ceil(n/block);pad=ng*block-n;z=torch.cat([score,torch.zeros(pad,device=score.device)]).reshape(ng,block).sum(1);take=math.ceil(k/block)
 ids=z.topk(take).indices[:,None]*block+torch.arange(block,device=score.device)[None];return ids.flatten()[ids.flatten()<n].sort().values
def rank_corr(a,b):
 ra=torch.empty_like(a);rb=torch.empty_like(b);ra[a.argsort()]=torch.arange(len(a),device=a.device,dtype=a.dtype);rb[b.argsort()]=torch.arange(len(b),device=b.device,dtype=b.dtype)
 return float(torch.corrcoef(torch.stack([ra,rb]))[0,1])

def quant_salient(w,cols,bits):
 if len(cols)==0:return torch.empty(w.shape[0],0,device=w.device),0,0
 z=w[:,cols]
 if bits==16:return z.half().float(),z.numel()*16,0
 qmax=(2**bits-1)/2;sc=(z.abs().amax(0,keepdim=True)/qmax).clamp_min(6e-8).half().float();q=(torch.round(z/sc+qmax).clamp(0,2**bits-1)-qmax)*sc
 return q,z.numel()*bits,len(cols)*16

def cluster_bulk(wb):
 pr=ordered_permutation(wb,16,True,'centroid',seed=20260922);pc=ordered_permutation(wb[pr].T,16,True,'centroid',seed=20260923)
 return pr,pc
def prepare_bulk(w,xtr,bulk,transform,scheme,pr,pc,importance_order=False):
 wb=w[:,bulk];xp=xtr[:,bulk];wp=wb[pr][:,pc];xpp=xp[:,pc];m,n=wp.shape
 if transform=='haar':
  pad=(-n)%16;z=torch.nn.functional.pad(wp,(0,pad));ntc=z.shape[1]//16;nr=m//16;t=z.reshape(nr,16,ntc,16).permute(0,2,1,3).reshape(-1,16,16);h=haar_basis(16,3);c=h@t@h.T
  xb=torch.nn.functional.pad(xpp,(0,pad)).reshape(len(xpp),ntc,16);proj=torch.einsum('tcb,vb->tcv',xb,h);sens=proj.square().sum(0)[None].expand(nr,-1,-1).reshape(len(t),1,16);fw=c.square()*sens
  bandmap,names=band_index(16,3);flatidx=torch.arange(c.numel(),device='cuda').reshape(len(c),256);segments=[];pattern_bits=0
  if scheme=='contiguous':segments=[torch.arange(c.numel(),device='cuda')]
  else:
   for bi in range(len(names)):
    pos=(bandmap.flatten()==bi).nonzero().flatten()
    if scheme=='subband_importance':
     pos=pos[fw.reshape(len(c),256)[:,pos].mean(0).argsort(descending=True)];pattern_bits+=len(pos)*8
    segments.append(flatidx[:,pos].reshape(-1))
  return dict(values=c.flatten(),weights=fw.flatten()/c.square().clamp_min(1e-30).flatten(),segments=segments,shape=(m,n,pad,nr,ntc),h=h,pr=pr,pc=pc,bulk=bulk,pattern_bits=pattern_bits,transform=transform)
 # Same clustering for the no-wavelet control isolates the transform's contribution.
 fw=wp.square()*xpp.square().sum(0)[None];return dict(values=wp.flatten(),weights=(fw/wp.square().clamp_min(1e-30)).flatten(),segments=[torch.arange(wp.numel(),device='cuda')],shape=(m,n,0),pr=pr,pc=pc,bulk=bulk,pattern_bits=0,transform=transform)

def arrange(prep,group):
 perm=torch.cat(prep['segments']);v=prep['values'][perm];weights=prep['weights'][perm];inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm),device=perm.device)
 ng=math.ceil(len(v)/group);pad=ng*group-len(v);vv=torch.nn.functional.pad(v,(0,pad)).reshape(ng,group);ww=torch.nn.functional.pad(weights,(0,pad)).reshape(ng,group);valid=torch.ones_like(vv,dtype=torch.bool);valid.flatten()[len(v):]=False
 return vv,ww,valid,inverse,len(v)

def scale_cast(s,precision):
 return s.half().float() if precision=='fp16' else s.to(torch.float8_e4m3fn).float()
def quant_groups(v,w,valid,avg_bits,share,scale_precision,scale_opt,form='standard'):
 ng,g=v.shape;importance=(v.square()*w*valid).sum(1);gbits=torch.ones(ng,dtype=torch.long,device=v.device)
 if avg_bits==0:return torch.zeros_like(v),0,0,0,torch.zeros(ng,dtype=torch.long,device=v.device)
 if avg_bits>1:
  count=round((avg_bits-1)*ng);gbits[importance.topk(count).indices]=2
 out=torch.zeros_like(v);scale_count=0
 for bit in [1,2]:
  ids=(gbits==bit).nonzero().flatten()
  if len(ids)==0:continue
  pad=(-len(ids))%share;safe=torch.cat([ids,torch.zeros(pad,dtype=torch.long,device=v.device)]);nb=len(safe)//share
  z=v[safe].reshape(nb,share,g);wt=w[safe].reshape(nb,share,g);mask=valid[safe].reshape(nb,share,g)
  if pad:mask[-1,-pad:]=False
  scale_count+=nb;dims=(1,2)
  if bit==1:
   if scale_opt=='max':sc=torch.where(mask,z.abs(),torch.zeros_like(z)).amax(dims,keepdim=True)
   elif scale_opt=='functional':sc=(wt*z.abs()*mask).sum(dims,keepdim=True)/(wt*mask).sum(dims,keepdim=True).clamp_min(1e-30)
   else:sc=(z.abs()*mask).sum(dims,keepdim=True)/mask.sum(dims,keepdim=True).clamp_min(1)
   sc=scale_cast(sc.clamp_min(6e-8),scale_precision);q=torch.where(z>=0,sc,-sc)*mask
  elif form=='ternary':
   threshold=.7*(z.abs()*mask).sum(dims,keepdim=True)/mask.sum(dims,keepdim=True).clamp_min(1);code=torch.where((z.abs()>=threshold)&mask,torch.sign(z),torch.zeros_like(z));
   wt2=wt if scale_opt=='functional' else torch.ones_like(wt);sc=(wt2*z*code*mask).sum(dims,keepdim=True)/(wt2*code.square()*mask).sum(dims,keepdim=True).clamp_min(1e-30);sc=scale_cast(sc.clamp_min(6e-8),scale_precision);q=code*sc*mask
  else:
   qmax=1.5;sc=(torch.where(mask,z.abs(),torch.zeros_like(z)).amax(dims,keepdim=True)/qmax).clamp_min(6e-8)
   if scale_opt!='max':
    wt2=wt if scale_opt=='functional' else torch.ones_like(wt)
    for _ in range(4):
     lev=(torch.round(z/sc+qmax).clamp(0,3)-qmax)*mask;sc=(wt2*z*lev).sum(dims,keepdim=True)/(wt2*lev.square()).sum(dims,keepdim=True).clamp_min(1e-30)
   sc=scale_cast(sc.clamp_min(6e-8),scale_precision);q=(torch.round(z/sc+qmax).clamp(0,3)-qmax)*sc*mask
  out[ids]=q.reshape(-1,g)[:len(ids)]
 code_bits=int((gbits[:,None]*valid).sum());bit_metadata=ng if avg_bits not in [1,2] else 0;scale_bits=scale_count*(16 if scale_precision=='fp16' else 8)
 return out,code_bits,bit_metadata,scale_bits,gbits

def restore_bulk(q,prep,inverse,length):
 arranged=q.flatten()[:length];original=arranged[inverse]
 if prep['transform']=='haar':
  m,n,pad,nr,ntc=prep['shape'];c=original.reshape(nr*ntc,16,16);h=prep['h'];t=h.T@c@h;wp=t.reshape(nr,ntc,16,16).permute(0,2,1,3).reshape(m,n+pad)[:,:n]
 else:m,n,_=prep['shape'];wp=original.reshape(m,n)
 invc=torch.empty_like(prep['pc']);invc[prep['pc']]=torch.arange(len(prep['pc']),device='cuda');invr=torch.empty_like(prep['pr']);invr[prep['pr']]=torch.arange(len(prep['pr']),device='cuda')
 return wp[invr][:,invc]

def run_config(w,xtr,xte,bias,name,scores,metric,fraction,sal_bits,transform,group,bulk_bits,scheme='subband',scale_precision='fp8',share=4,scale_opt='mse',block=1,form='standard',scope='core'):
 cols=top_columns(scores[metric],fraction,block);mask=torch.ones(w.shape[1],dtype=torch.bool,device='cuda');mask[cols]=False;bulk=mask.nonzero().flatten()
 # Transform preparation is invariant to precision/group settings and dominates setup cost.
 cache_key=(tuple(cols.cpu().tolist()),transform,scheme if transform=='haar' else 'contiguous')
 if cache_key not in PREP_CACHE:
  pr,pc=cluster_bulk(w[:,bulk]);PREP_CACHE[cache_key]=prepare_bulk(w,xtr,bulk,transform,scheme if transform=='haar' else 'contiguous',pr,pc)
 prep=PREP_CACHE[cache_key];pr,pc=prep['pr'],prep['pc'];v,fw,valid,inverse,length=arrange(prep,group)
 q,code_bits,bitmeta,scale_bits,gbits=quant_groups(v,fw,valid,bulk_bits,share,scale_precision,scale_opt,form);hb=restore_bulk(q,prep,inverse,length);hs,salvalue,salscale=quant_salient(w,cols,sal_bits)
 h=torch.zeros_like(w);h[:,bulk]=hb;h[:,cols]=hs
 index_bits=min(w.shape[1],len(cols)*math.ceil(math.log2(max(2,w.shape[1]))))+1
 perm_bits=permutation_bytes(pr,pc)*8;header=256+64;total=code_bits+bitmeta+scale_bits+salvalue+salscale+index_bits+perm_bits+prep['pattern_bits']+header
 tr,trc,trmax,trmse,te,tec,temax,temse=outputs(w,h,xtr,xte,bias);err=w-h
 row=dict(layer=name,scope=scope,saliency_metric=metric,salient_fraction=fraction,actual_salient_fraction=len(cols)/w.shape[1],salient_precision=sal_bits,salient_block=block,bulk_transform=transform,grouping_scheme=scheme if transform=='haar' else 'contiguous',group_size=group,bulk_bits=bulk_bits,quant_form=form,scale_precision=scale_precision,scale_share=share,scale_opt=scale_opt,
  bpw=total/w.numel(),bulk_code_bpw=code_bits/w.numel(),bulk_bit_metadata_bpw=bitmeta/w.numel(),scale_bpw=scale_bits/w.numel(),salient_value_bpw=salvalue/w.numel(),salient_scale_bpw=salscale/w.numel(),salient_index_bpw=index_bits/w.numel(),permutation_bpw=perm_bits/w.numel(),wave_metadata_bpw=(prep['pattern_bits']+header)/w.numel(),
  weight_error=float(err.norm()/w.norm()),train_output_error=tr,train_output_cosine=trc,train_max_output_error=trmax,train_output_mse=trmse,test_output_error=te,test_output_cosine=tec,test_max_output_error=temax,test_output_mse=temse)
 ROWS.append(row);return row,h,prep

TEXTS=['The numerical experiment compares weight compression and functional error across transformer layers.',
'Plants convert sunlight into chemical energy while ecosystems exchange matter through connected processes.',
'Write a Python function that sums a list, handles an empty input, and returns a numeric result.',
'A shop sold twelve items on Monday and eighteen on Tuesday for a total of thirty items.',
'The traveler watched rain cross the fields while reading a history of mathematics and modern science.']
def get_layers(model,tok,mid):
 count=model.config.num_hidden_layers;stages={0:'early',count//2:'middle',count-1:'late'};mods={};store={};hooks=[];phase={'i':0}
 for name,m in model.named_modules():
  li=next((int(q) for q in name.split('.') if q.isdigit()),None)
  if isinstance(m,torch.nn.Linear) and li in stages and (mid.startswith('Qwen') or li==count//2):
   mods[name]=(m,stages[li]);store[name]=[[],[],[]]
   def hook(mod,args,out,n=name):store[n][phase['i']].append((args[0].detach().reshape(-1,args[0].shape[-1]).cpu(),out.detach().reshape(-1,out.shape[-1]).cpu()))
   hooks.append(m.register_forward_hook(hook))
 with torch.no_grad():
  for i,text in enumerate(TEXTS):phase['i']=0 if i<2 else (1 if i==2 else 2);model(**tok(text,return_tensors='pt',truncation=True,max_length=128).to('cuda'))
 for h in hooks:h.remove()
 result=[]
 for name,(mod,stage) in mods.items():
  data=[]
  for phase_data in store[name]:data.append((torch.cat([q[0] for q in phase_data])[:256].cuda(),torch.cat([q[1] for q in phase_data])[:256].cuda()))
  xa,oa=data[0];xb,ob=data[1];xte,ote=data[2];xtr=torch.cat([xa,xb]);w=mod.weight.detach().float();bias=mod.bias.detach().float() if mod.bias is not None else None
  calc=torch.cat([xa,xb,xte])@w.T+(bias if bias is not None else 0);ORIENT.append(dict(layer=mid+'/'+name,max_abs=float((torch.cat([oa,ob,ote])-calc).abs().max())))
  if name.endswith('query_key_value'):
   heads=model.config.num_attention_heads;hd=w.shape[1]//heads;wv=w.reshape(heads,3,hd,-1);bb=bias.reshape(heads,3,hd) if bias is not None else None
   for j,label in enumerate(['q_proj','k_proj','v_proj']):result.append((name+'.'+label,wv[:,j].reshape(-1,wv.shape[-1]),xa,xb,xtr,xte,stage,bb[:,j].reshape(-1) if bb is not None else None))
  else:result.append((name,w,xa,xb,xtr,xte,stage,bias))
 return result

def diagnostics(w,xtr,name,scores):
 for a in scores:
  for b in scores:
   if a<b:SAL.append(dict(layer=name,kind='rank_correlation',a=a,b=b,value=rank_corr(scores[a],scores[b])))
 total=scores['functional'].sum()
 for frac in FRACTIONS[1:]:
  ids=top_columns(scores['functional'],frac);SAL.append(dict(layer=name,kind='concentration',fraction=frac,value=float(scores['functional'][ids].sum()/total)));SALIENT_SETS.append(dict(layer=name,fraction=frac,width=w.shape[1],indices=ids.cpu().tolist()))
def stability(name,w,xa,xb,xte):
 ss=[saliency_scores(w,x)['functional'] for x in [xa,xb,xte]]
 for frac in FRACTIONS[1:]:
  sets=[set(top_columns(q,frac).cpu().tolist()) for q in ss]
  for (i,a),(j,b) in [((0,sets[0]),(1,sets[1])),((0,sets[0]),(2,sets[2])),((1,sets[1]),(2,sets[2]))]:STABILITY.append(dict(layer=name,fraction=frac,pair=['A-B','A-C','B-C'][[(0,1),(0,2),(1,2)].index((i,j))],jaccard=len(a&b)/max(1,len(a|b)),overlap=len(a&b)/max(1,min(len(a),len(b)))))

def distribution(prep,name,fraction):
 z=prep['values'];mean=z.abs().mean();DIAG.append(dict(layer=name,salient_fraction=fraction,transform=prep['transform'],variance=float(z.var()),kurtosis=float(((z-z.mean())/z.std().clamp_min(1e-30)).pow(4).mean()),max_mean=float(z.abs().max()/mean.clamp_min(1e-30)),outlier_fraction=float((z.abs()>3*z.std()).float().mean())))
 if prep['transform']=='haar':
  c=z.reshape(-1,16,16);weight=prep['weights'].reshape_as(c);bandmap,names=band_index(16,3);energy=c.square();functional=energy*weight
  for bi,band in enumerate(names):
   mask=bandmap==bi;BAND.append(dict(layer=name,salient_fraction=fraction,band=band,coefficient_fraction=float(mask.float().mean()),energy_fraction=float(energy[:,mask].sum()/energy.sum()),functional_importance_fraction=float(functional[:,mask].sum()/functional.sum())))

def main():
 from transformers import AutoModelForCausalLM,AutoTokenizer
 specs=['Qwen/Qwen2.5-0.5B','facebook/opt-125m','EleutherAI/pythia-70m'];models=[];first=True
 for mid in specs:
  print('LOAD',mid,flush=True)
  try:
   tok=AutoTokenizer.from_pretrained(mid);model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval();layers=get_layers(model,tok,mid);models.append(dict(model=mid,revision=getattr(model.config,'_commit_hash',None),layers=len(layers)))
   for li,(lname,w,xa,xb,xtr,xte,stage,bias) in enumerate(layers):
    PREP_CACHE.clear();name=mid+'/'+lname+'@'+stage;scores=saliency_scores(w,xtr);diagnostics(w,xtr,name,scores);stability(name,w,xa,xb,xte)
    # Broad grid across all 33 matrices. Functional and weight×activation rankings are identical analytically; functional is executed once.
    for metric in ['weight_norm','activation_rms','functional']:
     for frac in FRACTIONS:
      for sp in ([8] if frac==0 else PRECISIONS):
       for transform in ['none','haar']:
        for group in CORE_GROUPS:
         for bits in BITS:
          row,_,_=run_config(w,xtr,xte,bias,name,scores,metric,frac,sp,transform,group,bits,'subband','fp8',4,'mse',scope='core')
          if metric=='functional':
           duplicate=dict(row);duplicate['saliency_metric']='weight_x_activation';duplicate['scope']='core_analytic_duplicate';ROWS.append(duplicate)
    # Explicit architectural controls requested for the critical comparison.
    run_config(w,xtr,xte,bias,name,scores,'activation_rms',.05,8,'none',32,0,'contiguous','fp8',4,'mse',scope='saliency_only')
    run_config(w,xtr,xte,bias,name,scores,'activation_rms',.05,8,'haar',32,1.5,'contiguous','fp8',4,'mse',scope='saliency_haar_contiguous')
    run_config(w,xtr,xte,bias,name,scores,'activation_rms',.05,8,'haar',32,1.5,'subband','fp8',4,'mse',scope='saliency_haar_subband')
    # Expensive design ablations on the fixed discovery layer only.
    if first:
     base=dict(metric='functional',fraction=.02,sal_bits=8,transform='haar')
     for group in [16,32,64,128]:
      for bits in [1,1.25,1.5,1.75,2]:
       for scheme in ['contiguous','subband','subband_importance']:
        for scalep in ['fp16','fp8']:
         for share in [1,2,4,8]:
          for opt in ['max','mse','functional']:run_config(w,xtr,xte,bias,name,scores,group=group,bulk_bits=bits,scheme=scheme,scale_precision=scalep,share=share,scale_opt=opt,scope='discovery_ablation',**base)
     for block in [4,8,16]:run_config(w,xtr,xte,bias,name,scores,'functional',.02,8,'haar',32,1.5,'subband','fp8',4,'mse',block=block,scope='structured_saliency')
     run_config(w,xtr,xte,bias,name,scores,'functional',.02,8,'haar',32,2,'subband','fp8',4,'mse',form='ternary',scope='ternary')
    # Fixed residual diagnostic candidate; the report does not add a residual codec.
    _,candidate,_=run_config(w,xtr,xte,bias,name,scores,'activation_rms',.05,8,'none',32,1.5,'subband','fp8',4,'mse',scope='residual_candidate')
    rr=(w-candidate).flatten();order=rr.abs().argsort(descending=True);den=rr.square().sum().clamp_min(1e-30)
    for density in [.001,.0025,.005,.01,.02,.05]:
     count=round(density*w.numel());RES.append(dict(layer=name,density=density,sparse_energy=float(rr[order[:count]].square().sum()/den)))
    if first:
     sv=torch.linalg.svdvals((w-candidate).double());e=sv.square();detail=dict(layer=name,density=np.nan,kurtosis=float((rr/rr.std()).pow(4).mean()))
     for rank in [1,4,8,16,32]:detail[f'rank{rank}_energy']=float(e[:rank].sum()/e.sum())
     RES.append(detail)
    # Distribution before/after protection at the same transform.
    for frac in [0,.02,.05]:
     cols=top_columns(scores['functional'],frac);mask=torch.ones(w.shape[1],dtype=torch.bool,device='cuda');mask[cols]=False;bulk=mask.nonzero().flatten();pr,pc=cluster_bulk(w[:,bulk]);prep=prepare_bulk(w,xtr,bulk,'haar','subband',pr,pc);distribution(prep,name,frac)
    first=False
    pd.DataFrame(ROWS).to_csv(OUT/'csv/configurations.csv',index=False);pd.DataFrame(SAL).to_csv(OUT/'csv/saliency.csv',index=False);pd.DataFrame(STABILITY).to_csv(OUT/'csv/stability.csv',index=False);pd.DataFrame(DIAG).to_csv(OUT/'csv/distribution.csv',index=False);pd.DataFrame(BAND).to_csv(OUT/'csv/bulk_subbands.csv',index=False);pd.DataFrame(ORIENT).to_csv(OUT/'csv/orientation.csv',index=False);pd.DataFrame(RES).to_csv(OUT/'csv/residual.csv',index=False)
    print('V4',mid,li+1,len(layers),lname,len(ROWS),flush=True)
   del model,layers,w,xtr,xte;torch.cuda.empty_cache()
  except Exception:
   FAIL.append(dict(model=mid,traceback=traceback.format_exc()));print(FAIL[-1],flush=True)
 overlap=[]
 for i,a in enumerate(SALIENT_SETS):
  ma='/'.join(a['layer'].split('/')[:2])
  for b in SALIENT_SETS[i+1:]:
   if a['fraction']!=b['fraction'] or a['width']!=b['width'] or ma!='/'.join(b['layer'].split('/')[:2]):continue
   aa,bb=set(a['indices']),set(b['indices']);overlap.append(dict(layer_a=a['layer'],layer_b=b['layer'],fraction=a['fraction'],jaccard=len(aa&bb)/max(1,len(aa|bb)),overlap=len(aa&bb)/max(1,len(aa))))
 pd.DataFrame(overlap).to_csv(OUT/'csv/cross_layer_overlap.csv',index=False)
 (OUT/'json/models.json').write_text(json.dumps(models,indent=2));(OUT/'json/failures.json').write_text(json.dumps(FAIL,indent=2));(OUT/'json/environment.json').write_text(json.dumps(dict(seed=20260922,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,tf32=False),indent=2))
 print('V4 DONE',len(ROWS),flush=True)
if __name__=='__main__':main()
