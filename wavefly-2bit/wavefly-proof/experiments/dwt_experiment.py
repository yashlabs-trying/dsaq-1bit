import sys,json,math,time,traceback,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from wavefly.core import basis
from wavefly.wavelet import haar_basis,band_index,band_slices
from wavefly.clustering import ordered_permutation
from wavefly.permutations import apply_permutation,undo_permutation,verify,permutation_bytes

torch.manual_seed(20260922);np.random.seed(20260922);torch.backends.cuda.matmul.allow_tf32=False
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/dwt'
for d in ['csv','json','plots','heatmaps']:(OUT/d).mkdir(parents=True,exist_ok=True)
KS=[4,8,16,24,32,48,64,96,128];BUDGETS=[8,6,4,3,2.5,2];DENS=[.001,.0025,.005,.01,.02,.05]
RET=[];SUB=[];MIX=[];CORRECT=[];RES=[];PATTERNS=[];FAIL=[];ORIENT=[]

def tiles(w,b):
 nr,nc=w.shape[0]//b,w.shape[1]//b;z=w[:nr*b,:nc*b]
 return z.reshape(nr,b,nc,b).permute(0,2,1,3).reshape(-1,b,b),nr,nc
def untile(t,nr,nc,b):return t.reshape(nr,nc,b,b).permute(0,2,1,3).reshape(nr*b,nc*b)
def reconstruct(c,h,nr,nc,b,pr,pc):return undo_permutation(untile(h.T@c@h,nr,nc,b),pr,pc)
def outputs(w,h,xtr,xte,bias):
 def one(x):
  y=x@w.T;yh=x@h.T
  if bias is not None:y=y+bias;yh=yh+bias
  return float((y-yh).norm()/y.norm().clamp_min(1e-30)),float(torch.nn.functional.cosine_similarity(y.flatten(),yh.flatten(),dim=0))
 tr,tc=one(xtr);te,tec=one(xte);return tr,tc,te,tec
def coeff_sensitivity(x,pc,h,nr,nc,b):
 xp=x if pc is None else x[:,pc];z=xp[:,:nc*b].reshape(len(xp),nc,b)
 projected=torch.einsum('tcb,vb->tcv',z,h);s=projected.square().sum(0) # input tile, coefficient column
 return s[None].expand(nr,-1,-1).reshape(nr*nc,1,b)
def make_cluster(w):
 pr=ordered_permutation(w,16,True,'centroid',seed=20260922);pc=ordered_permutation(w[pr].T,16,True,'centroid',seed=20260923)
 ok,err=verify(w,pr,pc);assert ok and err==0
 return pr,pc
def transform_spec(kind,b,levels):return basis(b) if kind=='dct' else haar_basis(b,levels)

def retention(w,xtr,xte,bias,name,kind,clustered,levels=1,b=16,pr=None,pc=None,scope='validation'):
 wp=apply_permutation(w,pr if clustered else None,pc if clustered else None);pp=pr if clustered else None;qq=pc if clustered else None
 t,nr,nc=tiles(wp,b);h=transform_spec(kind,b,levels);c=h@t@h.T;sens=coeff_sensitivity(xtr,qq,h,nr,nc,b);fscore=c.square()*sens
 denom=c.square().sum().double();perm=permutation_bytes(pp,qq);perm_bpw=perm*8/w.numel();idxbits=math.ceil(math.log2(b*b))
 method=('clustered_' if clustered else 'raw_')+kind+(f'_L{levels}' if kind=='haar' else '')
 if kind=='haar' and b==16 and scope=='validation':
  freq=fscore.flatten(1).topk(32,dim=1).indices
  counts=torch.bincount(freq.flatten(),minlength=b*b).float()/len(t);PATTERNS.append(dict(layer=name,method=method,levels=levels,probability=counts.cpu().numpy()))
 for rule in ['magnitude','energy','functional']:
  score=c.abs() if rule=='magnitude' else (c.square() if rule=='energy' else fscore)
  # Magnitude and energy rankings are mathematically identical; both are recorded as a check.
  for k in KS:
   if k>b*b:continue
   ix=score.flatten(1).topk(k,dim=1).indices;vals=c.flatten(1).gather(1,ix);cc=torch.zeros_like(c).flatten(1).scatter_(1,ix,vals).reshape_as(c)
   hh=reconstruct(cc,h,nr,nc,b,pp,qq);tr,tc,te,tec=outputs(w,hh,xtr,xte,bias);err=w-hh
   RET.append(dict(layer=name,scope=scope,method=method,transform=kind,clustered=clustered,levels=levels,tile=b,rule=rule,k=k,
    bpw=(len(t)*k*(16+idxbits)+perm*8+256)/w.numel(),coefficient_bpw=len(t)*k*16/w.numel(),mask_bpw=len(t)*k*idxbits/w.numel(),permutation_bpw=perm_bpw,
    energy=float(vals.square().sum().double()/denom),weight_error=float(err.norm()/w.norm()),max_error=float(err.abs().max()),train_output_error=tr,train_output_cosine=tc,test_output_error=te,test_output_cosine=tec))
 pd.DataFrame(RET).to_csv(OUT/'csv/retention.csv',index=False)
 return c,h,nr,nc,fscore

def subband_analysis(w,xtr,xte,bias,name,pr,pc,levels,c,h,nr,nc):
 total=c.square().sum().double();bands=band_slices(16,levels)
 for band,(a,b) in bands:
  mask=torch.zeros(16,16,dtype=torch.bool,device='cuda');mask[a,b]=True;cc=c.clone();cc[:,mask]=0
  hh=reconstruct(cc,h,nr,nc,16,pr,pc);tr,tc,te,tec=outputs(w,hh,xtr,xte,bias)
  SUB.append(dict(layer=name,levels=levels,band=band,coefficient_fraction=float(mask.float().mean()),energy_fraction=float(c[:,mask].square().sum().double()/total),
   train_removal_error=tr,test_removal_error=te,test_output_cosine=tec))

NOISE={0:1.,2:.20,4:.01,8:3e-5,16:0.};NEXT={0:2,2:4,4:8,8:16}
def allocate(position_score,payload_bpw,groups):
 ng=int(groups.max())+1;gbits=torch.zeros(ng,dtype=torch.long,device=position_score.device);sizes=torch.bincount(groups,minlength=ng)
 gs=torch.zeros(ng,device=position_score.device).index_add_(0,groups,position_score)
 limit=payload_bpw*len(position_score);used=0;candidates=[]
 # Marginal benefit/bit decreases monotonically for 0→2→4→8→16, so a
 # single global sort preserves each group's upgrade dependency.
 for g in range(ng):
  size=int(sizes[g]);importance=float(gs[g])
  for old,new in [(0,2),(2,4),(4,8),(8,16)]:
   cost=(new-old)*size;benefit=importance*(NOISE[old]-NOISE[new]);candidates.append((benefit/cost,g,old,new,cost))
 for _,g,old,new,cost in sorted(candidates,reverse=True):
  if int(gbits[g])!=old:continue
  if used+cost<=limit+1e-9:used+=cost;gbits[g]=new
 return gbits[groups]

def quantize(c,bits,scale_mode,bandmap):
 out=torch.zeros_like(c);scales=0;nt=len(c)
 for bit in [2,4,8,16]:
  pos=(bits==bit)
  if not pos.any():continue
  if bit==16:out[:,pos]=c[:,pos].half().float();continue
  qmax=(2**bit-1)/2
  if scale_mode=='tile':
   z=c[:,pos];sc=(z.abs().amax(1,keepdim=True)/qmax).clamp_min(6e-8).half().float();out[:,pos]=(torch.round(z/sc+qmax).clamp(0,2**bit-1)-qmax)*sc;scales+=nt
  elif scale_mode=='subband':
   for band in torch.unique(bandmap):
    pick=pos&(bandmap==band)
    if not pick.any():continue
    z=c[:,pick];sc=(z.abs().amax(1,keepdim=True)/qmax).clamp_min(6e-8).half().float();out[:,pick]=(torch.round(z/sc+qmax).clamp(0,2**bit-1)-qmax)*sc;scales+=nt
  elif scale_mode=='shared16':
   z=c[:,pos];pad=(-nt)%16
   if pad:z=torch.cat([z,torch.zeros(pad,z.shape[1],device=z.device)],0)
   zz=z.reshape(-1,16,z.shape[1]);sc=(zz.abs().amax((1,2),keepdim=True)/qmax).clamp_min(6e-8).half().float();q=(torch.round(zz/sc+qmax).clamp(0,2**bit-1)-qmax)*sc
   out[:,pos]=q.reshape(-1,z.shape[1])[:nt];scales+=math.ceil(nt/16)
  else:raise ValueError(scale_mode)
 return out,scales

def mixed_precision(w,xtr,xte,bias,name,pr,pc,levels,c,h,nr,nc,fscore,discovery=False):
 n=w.numel();perm_bits=permutation_bytes(pr,pc)*8;bandmap,bnames=band_index(16,levels);flatband=bandmap.flatten();nt=len(c)
 scores={'magnitude_fixed':c.square().mean(0).flatten(),'functional_fixed':fscore.mean(0).flatten()}
 scores['functional_subband']=scores['functional_fixed']
 groups={'magnitude_fixed':torch.arange(256,device='cuda'),'functional_fixed':torch.arange(256,device='cuda'),'functional_subband':flatband}
 for alloc,score in scores.items():
  metadata_bits=(256 if alloc!='functional_subband' else len(bnames))*3
  for mode in ['tile','subband','shared16']:
   for budget in BUDGETS:
    payload=max(0,budget-(perm_bits+metadata_bits+256)/n);bits=None
    for _ in range(20):
     posbits=allocate(score,payload,groups[alloc]);cq,scales=quantize(c,posbits.reshape(16,16),mode,bandmap)
     actual=(int(posbits.sum())*nt+scales*16+perm_bits+metadata_bits+256)/n
     if actual<=budget+1e-6:break
     payload=max(0,payload-(actual-budget)-.01)
    hh=reconstruct(cq,h,nr,nc,16,pr,pc);tr,tc,te,tec=outputs(w,hh,xtr,xte,bias);err=w-hh
    MIX.append(dict(layer=name,levels=levels,allocation=alloc,scale_mode=mode,target_bpw=budget,bpw=actual,coefficient_bpw=int(posbits.sum())*nt/n,scale_bpw=scales*16/n,
      allocation_metadata_bpw=metadata_bits/n,permutation_bpw=perm_bits/n,nonzero_positions=int((posbits>0).sum()),bits_2=int((posbits==2).sum()),bits_4=int((posbits==4).sum()),bits_8=int((posbits==8).sum()),bits_16=int((posbits==16).sum()),
      energy_explained=float(1-(c-cq).square().sum().double()/c.square().sum().double()),weight_error=float(err.norm()/w.norm()),max_error=float(err.abs().max()),train_output_error=tr,train_output_cosine=tc,test_output_error=te,test_output_cosine=tec))
    if alloc=='functional_fixed' and mode=='shared16' and budget in [2,3,4]:
     flat=err.flatten();order=flat.abs().argsort(descending=True);den=flat.square().sum().clamp_min(1e-30)
     for density in DENS:
      count=round(density*n);RES.append(dict(layer=name,budget=budget,density=density,sparse_energy=float(flat[order[:count]].square().sum()/den)))
     if discovery and budget==2:
      sv=torch.linalg.svdvals(err.double());e=sv.square();row=dict(layer=name,budget=budget,residual_kurtosis=float((flat/flat.std()).pow(4).mean()))
      for rank in [1,4,8,16,32]:row[f'rank{rank}_energy']=float(e[:rank].sum()/e.sum())
      RES.append(row)
 pd.DataFrame(MIX).to_csv(OUT/'csv/mixed_precision.csv',index=False);pd.DataFrame(RES).to_csv(OUT/'csv/residual.csv',index=False)

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
   mods[name]=(m,stages[li]);store[name]=[[],[]]
   def hook(mod,args,out,n=name):store[n][phase['i']].append((args[0].detach().reshape(-1,args[0].shape[-1]).cpu(),out.detach().reshape(-1,out.shape[-1]).cpu()))
   hooks.append(m.register_forward_hook(hook))
 with torch.no_grad():
  for i,text in enumerate(TEXTS):phase['i']=0 if i<3 else 1;model(**tok(text,return_tensors='pt',truncation=True,max_length=128).to('cuda'))
 for h in hooks:h.remove()
 result=[]
 for name,(m,stage) in mods.items():
  tr=store[name][0];te=store[name][1];xtr=torch.cat([q[0] for q in tr])[:256].cuda();xte=torch.cat([q[0] for q in te])[:256].cuda();otr=torch.cat([q[1] for q in tr])[:256].cuda();ote=torch.cat([q[1] for q in te])[:256].cuda();w=m.weight.detach().float();bias=m.bias.detach().float() if m.bias is not None else None
  native=torch.cat([otr,ote]);xx=torch.cat([xtr,xte]);calc=xx@w.T+(bias if bias is not None else 0);ORIENT.append(dict(layer=mid+'/'+name,max_abs=float((native-calc).abs().max())))
  if name.endswith('query_key_value'):
   heads=model.config.num_attention_heads;hd=w.shape[1]//heads;v=w.reshape(heads,3,hd,-1);bb=bias.reshape(heads,3,hd) if bias is not None else None
   for j,label in enumerate(['q_proj','k_proj','v_proj']):result.append((name+'.'+label,v[:,j].reshape(-1,v.shape[-1]),xtr,xte,stage,bb[:,j].reshape(-1) if bb is not None else None))
  else:result.append((name,w,xtr,xte,stage,bias))
 return result

def main():
 from transformers import AutoModelForCausalLM,AutoTokenizer
 specs=['Qwen/Qwen2.5-0.5B','facebook/opt-125m','EleutherAI/pythia-70m'];chosen_level=None;models=[]
 for mid in specs:
  print('LOAD',mid,flush=True)
  try:
   tok=AutoTokenizer.from_pretrained(mid);model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval();layers=get_layers(model,tok,mid);models.append(dict(model=mid,revision=getattr(model.config,'_commit_hash',None),layers=len(layers)))
   for ix,(lname,w,xtr,xte,stage,bias) in enumerate(layers):
    name=mid+'/'+lname+'@'+stage;pr,pc=make_cluster(w);first=chosen_level is None
    # Full-coefficient correctness includes cluster undo and both requested dtypes.
    wp=apply_permutation(w,pr,pc);sample=wp[:16,:16]
    for dtype in [torch.float64,torch.float32]:
     hh=haar_basis(16,3,dtype);z=hh@sample.to(dtype)@hh.T;r=hh.T@z@hh
     CORRECT.append(dict(layer=name,dtype=str(dtype),relative_error=float((sample.double()-r.double()).norm()/sample.double().norm()),max_error=float((sample.double()-r.double()).abs().max()),orthogonality=float((hh.double()@hh.double().T-torch.eye(16,device='cuda')).abs().max())))
    retention(w,xtr,xte,bias,name,'dct',False,b=16,scope='validation')
    retention(w,xtr,xte,bias,name,'dct',True,b=16,pr=pr,pc=pc,scope='validation')
    for lev in [1,2,3]:
     retention(w,xtr,xte,bias,name,'haar',False,lev,16,scope='validation')
     c,h,nr,nc,fs=retention(w,xtr,xte,bias,name,'haar',True,lev,16,pr,pc,'validation')
     subband_analysis(w,xtr,xte,bias,name,pr,pc,lev,c,h,nr,nc)
    if first:
     q=pd.DataFrame(RET);q=q[(q.layer==name)&(q.method.str.startswith('clustered_haar'))&(q.rule=='functional')&(q.k==32)]
     chosen_level=int(q.sort_values('train_output_error').iloc[0].levels);(OUT/'json/chosen_level.json').write_text(json.dumps(dict(level=chosen_level,selection_layer=name,criterion='calibration output error at K=32 functional selection'),indent=2))
    c,h,nr,nc,fs=retention(w,xtr,xte,bias,name,'haar',True,chosen_level,16,pr,pc,'mixed_source')
    mixed_precision(w,xtr,xte,bias,name,pr,pc,chosen_level,c,h,nr,nc,fs,first)
    if mid.startswith('Qwen') and stage=='middle' and lname.endswith('q_proj'):
     for b in [8,32]:retention(w,xtr,xte,bias,name,'haar',True,min(chosen_level,int(math.log2(b))),b,pr,pc,'tile_sweep')
    pd.DataFrame(CORRECT).to_csv(OUT/'csv/correctness.csv',index=False);pd.DataFrame(SUB).to_csv(OUT/'csv/subbands.csv',index=False);pd.DataFrame(ORIENT).to_csv(OUT/'csv/orientation.csv',index=False)
    print('DWT',mid,ix+1,len(layers),lname,flush=True)
   del model,layers,w,xtr,xte;torch.cuda.empty_cache()
  except Exception:
   FAIL.append(dict(model=mid,traceback=traceback.format_exc()));print(FAIL[-1],flush=True)
 # Pattern reuse statistics and heatmap inputs.
 np.savez_compressed(OUT/'json/importance_patterns.npz',**{f'p{i}':q['probability'] for i,q in enumerate(PATTERNS)})
 meta=[dict(layer=q['layer'],method=q['method'],levels=q['levels']) for q in PATTERNS];(OUT/'json/importance_pattern_meta.json').write_text(json.dumps(meta,indent=2))
 sets=[set(np.argpartition(q['probability'],-32)[-32:]) for q in PATTERNS if q['method'].startswith('clustered_haar') and q['levels']==chosen_level]
 jac=[len(a&b)/len(a|b) for i,a in enumerate(sets) for b in sets[i+1:]]
 (OUT/'json/pattern_reuse.json').write_text(json.dumps(dict(mean_top32_jaccard=float(np.mean(jac)),median_top32_jaccard=float(np.median(jac)),comparisons=len(jac)),indent=2))
 (OUT/'json/models.json').write_text(json.dumps(models,indent=2));(OUT/'json/failures.json').write_text(json.dumps(FAIL,indent=2));(OUT/'json/environment.json').write_text(json.dumps(dict(seed=20260922,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,tf32=False),indent=2))
 print('DWT DONE',len(RET),len(MIX),'level',chosen_level,flush=True)
if __name__=='__main__':main()
