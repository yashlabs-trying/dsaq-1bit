"""V5: calibration-only selection, stored-precision reconstruction, bounded banks.

BPW is byte-aligned representation accounting, not a measured model container.
Global allocations cover the sampled matrices, not a complete deployed model.
"""
import os, sys, json, math, time, traceback, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import v4_experiment as v4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v5'
for d in ['csv', 'json', 'plots']: (OUT/d).mkdir(parents=True, exist_ok=True)
torch.manual_seed(20260922)
np.random.seed(20260922)
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_num_threads(4)
ROWS, DIAG, ROUTES, FAIL, MODELS = [], [], [], [], []
EXTRA = [
 'def binary_search(a, key):\n    lo, hi = 0, len(a)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if a[mid] < key: lo = mid + 1\n        else: hi = mid\n    return lo',
 'A geometric series has first term three and common ratio two. Compute the sum of its first eight terms using the finite geometric series formula.',
 'In a database transaction, atomicity means that either all operations commit or none do. Isolation prevents concurrent transactions from observing intermediate updates.',
 'Suppose a triangle has sides five, twelve, and thirteen. The Pythagorean theorem establishes that it is right angled. Its area is thirty square units.',
 'The museum conservator examined pigment samples under a microscope before documenting the restoration of a nineteenth century landscape painting.',
 'An exception handler should release acquired resources even when parsing fails. A context manager provides deterministic cleanup when leaving a block.'
]

def aligned(bits): return int(math.ceil(float(bits)/8)*8)

def capture(model, tok, mid):
    count=model.config.num_hidden_layers
    stages={0:'early',count//2:'middle',count-1:'late'}
    mods={}; store={}; hooks=[]; phase=[0]
    for name,m in model.named_modules():
        li=next((int(q) for q in name.split('.') if q.isdigit()),None)
        if isinstance(m,torch.nn.Linear) and li in stages and (mid.startswith('Qwen') or li==count//2):
            mods[name]=(m,stages[li]); store[name]=[[],[],[]]
            def hook(mod,args,out,n=name):
                store[n][phase[0]].append(args[0].detach().reshape(-1,args[0].shape[-1]).cpu())
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        for p,texts in enumerate([v4.TEXTS[:3],v4.TEXTS[3:],EXTRA]):
            phase[0]=p
            for text in texts: model(**tok(text,return_tensors='pt',truncation=True,max_length=128).to('cuda'))
    for hook in hooks: hook.remove()
    result=[]
    for name,(mod,stage) in mods.items():
        xs=[torch.cat(s)[:256].cuda() for s in store[name]]
        w=mod.weight.detach().float(); bias=mod.bias.detach().float() if mod.bias is not None else None
        if name.endswith('query_key_value'):
            heads=model.config.num_attention_heads; hd=w.shape[1]//heads
            for j,label in enumerate(['q_proj','k_proj','v_proj']):
                ww=w.reshape(heads,3,hd,-1)[:,j].reshape(-1,w.shape[1])
                bb=bias.reshape(heads,3,hd)[:,j].reshape(-1) if bias is not None else None
                result.append((mid+'/'+name+'.'+label+'@'+stage,ww,xs,bb))
        else: result.append((mid+'/'+name+'@'+stage,w,xs,bias))
    return result

def errors(w,h,xs):
    # Bias excluded by the V5 definition. No selection uses B or C.
    return [float((x@(w-h).T).norm()/(x@w.T).norm().clamp_min(1e-30)) for x in xs]

def record(name,w,h,xs,family,config,storage,**extra):
    storage={k:aligned(v) for k,v in storage.items()}
    ea,eb,ec=errors(w,h,xs)
    row=dict(layer=name,n=w.numel(),out=w.shape[0],inp=w.shape[1],family=family,config=config,
             bits=sum(storage.values()),bpw=sum(storage.values())/w.numel(),error_a=ea,error_b=eb,error_c=ec,**storage,**extra)
    ROWS.append(row)
    return row

def nearest(z,cb):
    return torch.cat([(-2*t@cb.T+cb.square().sum(1)[None]).argmin(1) for t in z.split(8192)])

def lloyd(z,k,iters=12):
    # Deterministic sampled training, all vectors assigned using rounded codebook.
    sample=z[torch.randperm(len(z),device=z.device)[:min(16384,len(z))]]
    cb=sample[torch.linspace(0,len(sample)-1,k,device=z.device).long()].clone()
    for _ in range(iters):
        idx=nearest(sample,cb); counts=torch.bincount(idx,minlength=k)
        sums=torch.zeros_like(cb).index_add_(0,idx,sample)
        cb=torch.where(counts[:,None]>0,sums/counts[:,None].clamp_min(1),cb)
    cb=cb.half().float()
    return cb,nearest(z,cb)

def quant_factor(z):
    scale=(z.abs().amax()/127).clamp_min(6e-8).half().float()
    return (z/scale).round().clamp(-127,127)*scale, aligned(z.numel()*8)+16

def scalar_options(v,weights,learned=False):
    result=[]
    # one shared FP8 scale per four G32 groups; stored rounded before assignment
    pad=(-len(v))%4
    blocks=torch.nn.functional.pad(v,(0,0,0,pad)).reshape(-1,128)
    scale=blocks.square().mean(1,keepdim=True).sqrt().clamp_min(2**-9).to(torch.float8_e4m3fn).float()
    norm=(blocks/scale).flatten()[:v.numel()].reshape_as(v)
    for bits in [1,2,3,4]:
        k=2**bits
        if learned:
            cb,_=lloyd(norm.flatten()[:,None],k)
            lev=cb.flatten().sort().values
        else: lev=torch.linspace(-2.5,2.5,k,device=v.device)
        idx=torch.bucketize(norm,(lev[:-1]+lev[1:])/2)
        q=lev[idx]*scale.repeat_interleave(4,0)[:len(v)]
        loss=((v-q).square()*weights).sum(1)
        result.append((q,loss,k*16 if learned else 0))
    return result,scale.numel()*8

def adaptive(options,v,avg):
    # Lagrangian allocation using diagonal functional perturbation energy.
    loss=torch.stack([t[1] for t in options],1)
    costs=torch.tensor([1,2,3,4],device=v.device,dtype=v.dtype)
    lo,hi=0.,float(loss.max().clamp_min(1e-12))*2
    target=avg*len(v)
    for _ in range(32):
        lam=(lo+hi)/2
        idx=(loss+lam*costs).argmin(1)
        if float(costs[idx].sum())>target: lo=lam
        else: hi=lam
    idx=(loss+hi*costs).argmin(1)
    q=torch.zeros_like(v)
    for j,opt in enumerate(options): q[idx==j]=opt[0][idx==j]
    return q,int(costs[idx].sum())*32,len(v)*2

def lowrank(r,rank=32):
    u,s,v=torch.svd_lowrank(r,q=min(rank+8,min(r.shape)),niter=4)
    return u[:,:rank],s[:rank],v[:,:rank]

def oracle(x,w,base,atoms,k):
    target=x@(w-base).T
    ao=torch.stack([x@a.T for a in atoms],1)
    rem=target.clone(); used=torch.zeros(len(x),len(atoms),device=x.device,dtype=torch.bool)
    for _ in range(min(k,len(atoms))):
        gain=2*(ao*rem[:,None]).sum(2)-ao.square().sum(2)
        gain[used]=-float('inf')
        best=gain.argmax(1); take=gain.gather(1,best[:,None])[:,0]>0
        ids=torch.arange(len(x),device=x.device)[take]
        rem[ids]-=ao[ids,best[take]]; used[ids,best[take]]=True
    return float(rem.norm()/(x@w.T).norm().clamp_min(1e-30)),used

def router_features(x):
    return torch.nn.functional.pad(x,(0,(-x.shape[1])%32)).reshape(len(x),-1,32).square().mean(2).sqrt()

def learned_router(xs,w,base,atoms,k):
    z=router_features(xs[0]); norm=z.mean(0).clamp_min(1e-6).half().float(); z=z/norm
    lin=torch.nn.Linear(z.shape[1],len(atoms),device=z.device)
    opt=torch.optim.Adam(lin.parameters(),lr=.02)
    target=xs[0]@(w-base).T
    ao=torch.stack([xs[0]@a.T for a in atoms],1)
    with torch.enable_grad():
        for _ in range(250):
            opt.zero_grad(); prob=lin(z).sigmoid(); hard=torch.zeros_like(prob).scatter(1,prob.topk(k,1).indices,1)
            gate=hard+prob-prob.detach()
            loss=(target-(ao*gate[:,:,None]).sum(1)).square().mean()+1e-5*prob.mean()
            loss.backward(); opt.step()
    wr=lin.weight.detach().half().float(); br=lin.bias.detach().half().float()
    es=[]; freqs=[]
    for x in xs:
        score=(router_features(x)/norm)@wr.T+br
        mask=torch.zeros_like(score).scatter(1,score.topk(k,1).indices,1)
        pred=sum((x@a.T)*mask[:,j:j+1] for j,a in enumerate(atoms))
        es.append(float((x@(w-base).T-pred).norm()/(x@w.T).norm().clamp_min(1e-30)))
        freqs.append(mask.mean(0).tolist())
    return es,aligned((wr.numel()+br.numel()+norm.numel())*16),freqs

def corrections(name,w,xs,base,row):
    r=w-base; denom=float(r.square().sum()); u,s,v=lowrank(r)
    flat=r.flatten(); ch=r.square().sum(0)
    diag=dict(layer=name,kurtosis=float(((flat-flat.mean())/flat.std().clamp_min(1e-30)).pow(4).mean()),
              zero_fraction=float((flat==0).float().mean()),top1pct_energy=float(flat.square().topk(max(1,len(flat)//100)).values.sum()/denom),
              channel_top5pct_energy=float(ch.topk(max(1,len(ch)//20)).values.sum()/denom),
              activation_rms=float(xs[0].square().mean().sqrt()),base_bpw=row['bpw'],base_error_b=row['error_b'])
    grouped=torch.nn.functional.pad(flat.square(),(0,(-len(flat))%32)).reshape(-1,32).sum(1)
    diag['group_top5pct_energy']=float(grouped.topk(max(1,len(grouped)//20)).values.sum()/denom)
    out=xs[0]@r.T; normalized=out/out.norm(dim=1,keepdim=True).clamp_min(1e-30)
    similarity=normalized@normalized.T
    offdiag=~torch.eye(len(out),device=w.device,dtype=torch.bool)
    diag['sample_residual_abs_cosine']=float(similarity[offdiag].abs().mean())
    diag['activation_max_abs']=float(xs[0].abs().max())
    sal=v4.saliency_scores(w,xs[0])['functional']; cols=v4.top_columns(v4.saliency_scores(w,xs[0])['activation_rms'],.05)
    diag['protected_functional_fraction']=float(sal[cols].sum()/sal.sum())
    for j,value in enumerate(s): diag[f'sigma_{j+1}']=float(value)
    for rank in [1,2,4,8,16,32]: diag[f'rank{rank}_energy']=float(s[:rank].square().sum()/denom)
    DIAG.append(diag)
    storage={k:row[k] for k in ['codes','scales','salient','codebooks','metadata','correction','router']}
    atoms=[]; atom_bits=0
    for rank in [1,2,4,8,16]:
        uq,bu=quant_factor(u[:,:rank]*s[:rank]); vq,bv=quant_factor(v[:,:rank])
        corr=uq@vq.T
        record(name,w,base+corr,xs,'static',row['config']+f'+rank{rank}',dict(storage,correction=bu+bv),base_family=row['family'])
    # Store individually quantized rank-one paths, not dense FP32 atoms.
    for j in range(8):
        uq,bu=quant_factor(u[:,j:j+1]*s[j]); vq,bv=quant_factor(v[:,j:j+1])
        atoms.append(uq@vq.T); atom_bits+=bu+bv
    full=record(name,w,base+sum(atoms),xs,'bank_static',row['config']+'+bank8',dict(storage,correction=atom_bits))
    for k in [1,2,4,8]:
        es=[]; frequencies=[]
        for x in xs:
            err,mask=oracle(x,w,base,atoms,k); es.append(err); frequencies.append(mask.float().mean(0).tolist())
        rr=dict(full,family='oracle',config=row['config']+f'+oracle{k}',error_a=es[0],error_b=es[1],error_c=es[2],topk=k)
        ROWS.append(rr)
        ROUTES.append(dict(layer=name,kind='oracle',topk=k,error_a=es[0],error_b=es[1],error_c=es[2],frequencies=frequencies,bpw=rr['bpw']))
        # Gate based exclusively on calibration, fixed before any validation results.
        if k in [1,2,4] and es[0]<.10 and es[0]<row['error_a']*.6:
            learned,rb,freq=learned_router(xs,w,base,atoms,k)
            ROWS.append(dict(rr,family='learned_router',config=row['config']+f'+router{k}',router=rb,bits=rr['bits']+rb,bpw=(rr['bits']+rb)/w.numel(),error_a=learned[0],error_b=learned[1],error_c=learned[2]))
            ROUTES.append(dict(layer=name,kind='learned',topk=k,error_a=learned[0],error_b=learned[1],error_c=learned[2],frequencies=freq,bpw=(rr['bits']+rb)/w.numel()))
    rem=r.clone(); corr=torch.zeros_like(r)
    for plane in [1,2,3]:
        sign=torch.where(rem>=0,1.,-1.); alpha=rem.abs().mean().half().float()
        corr+=alpha*sign; rem=r-corr
        record(name,w,base+corr,xs,'static',row['config']+f'+binary{plane}',dict(storage,correction=plane*(aligned(r.numel())+16)))
    # Sparse matching pursuit over residual vectors. INT8 atoms, FP16 amplitudes.
    dim=32; pad=(-r.numel())%dim; z=torch.nn.functional.pad(r.flatten(),(0,pad)).reshape(-1,dim)
    for k in [8,16,32,64]:
        cb,_=lloyd(z,k)
        cb,cbits=quant_factor(cb)
        remaining=z.clone(); recon=torch.zeros_like(z)
        for count in [1,2,3,4]:
            norm=cb.square().sum(1).clamp_min(1e-30)
            dots=remaining@cb.T; ix=(dots.square()/norm).argmax(1)
            coeff=(dots.gather(1,ix[:,None])[:,0]/norm[ix]).half().float()
            recon+=coeff[:,None]*cb[ix]; remaining=z-recon
            if count in [1,2,4]:
                bits=cbits+aligned(len(z)*count*(int(math.log2(k))+16))
                record(name,w,base+recon.flatten()[:r.numel()].reshape_as(r),xs,'static',row['config']+f'+dict{k}x{count}',dict(storage,correction=bits))

def run_layer(name,w,xs,bias):
    v4.PREP_CACHE.clear(); scores=v4.saliency_scores(w,xs[0]); local=[]; best={}
    def keep(row,h):
        local.append(row)
        # Two base budgets ensure corrections can displace base bits.
        for budget in [1.5,2.0]:
            if row['bpw']<=budget and (budget not in best or row['error_a']<best[budget][0]['error_a']): best[budget]=(row,h.clone())
    for b in [0,1,1.25,1.5,1.75,2]:
        rr,h,prep=v4.run_config(w,xs[0],xs[1],None,name,scores,'activation_rms',.05,8,'none',32,b,'contiguous','fp8',4,'mse')
        storage=dict(codes=rr['bulk_code_bpw']*w.numel(),scales=rr['scale_bpw']*w.numel(),salient=(rr['salient_value_bpw']+rr['salient_scale_bpw'])*w.numel(),codebooks=0,correction=0,router=0,metadata=(rr['bulk_bit_metadata_bpw']+rr['salient_index_bpw']+rr['permutation_bpw']+rr['wave_metadata_bpw'])*w.numel())
        keep(record(name,w,h,xs,'v4',f'v4_{b}',storage),h)
    cols=v4.top_columns(scores['activation_rms'],.05)
    hs,sb,ss=v4.quant_salient(w,cols,8)
    v,fw,valid,inverse,length=v4.arrange(prep,32)
    common=dict(salient=sb+ss,metadata=storage['metadata'],correction=0,router=0)
    def restore(q):
        h=torch.zeros_like(w); h[:,prep['bulk']]=v4.restore_bulk(q,prep,inverse,length); h[:,cols]=hs; return h
    for learned in [False,True]:
        opts,scbits=scalar_options(v,fw,learned)
        for avg in [.5,.75,1,1.25,1.5,1.75,2,2.25,2.5,3]:
            if avg<1: continue
            q,cbits,mbits=adaptive(opts,v,avg)
            st=dict(common,codes=cbits,scales=scbits,codebooks=sum(t[2] for t in opts),metadata=common['metadata']+mbits)
            h=restore(q)
            keep(record(name,w,h,xs,'learned_scalar' if learned else 'group_adaptive',f'{"learned" if learned else "group"}_{avg}',st),h)
    # Vectors share an FP16 RMS scale per G32 group; codebooks are layer-wide.
    scale=v.square().mean(1,keepdim=True).sqrt().clamp_min(6e-8).half().float()
    for dim in [2,4,8]:
        z=(v/scale).reshape(-1,dim)
        # A bounded initial VQ run: <=256 entries. Large 8D codebooks are explicitly not tested.
        for logk in ([2,3,4,5,6] if dim==2 else ([4,5,6,7,8] if dim==4 else [6,7,8])):
            cb,ix=lloyd(z,2**logk)
            q=cb[ix].reshape_as(v)*scale
            st=dict(common,codes=len(z)*logk,scales=scale.numel()*16,codebooks=cb.numel()*16)
            h=restore(q)
            keep(record(name,w,h,xs,'vector',f'vq{dim}x{2**logk}',st),h)
            # Shared diagonal functional metric by vector position. Decoded
            # codebook is stored directly, so no runtime metric is required.
            metric=fw.reshape(-1,dim).mean(0).sqrt().clamp_min(1e-12)
            metric=metric/metric.mean()
            weighted_cb,_=lloyd(z*metric,2**logk)
            decoded_cb=(weighted_cb/metric).half().float()
            ix=nearest(z*metric,decoded_cb*metric)
            h=restore(decoded_cb[ix].reshape_as(v)*scale)
            keep(record(name,w,h,xs,'vector',f'vqfunctional{dim}x{2**logk}',st),h)
    for budget,(row,h) in best.items():
        print('CORRECTIONS',name,budget,row['config'],row['bpw'],flush=True)
        corrections(name,w,xs,h,row)
    print('LAYER_DONE',name,'rows',len(ROWS),flush=True)

def save():
    pd.DataFrame(ROWS).to_csv(OUT/'csv/configurations.csv',index=False)
    pd.DataFrame(DIAG).to_csv(OUT/'csv/residual.csv',index=False)
    (OUT/'json/routing.json').write_text(json.dumps(ROUTES,indent=2))
    (OUT/'json/failures.json').write_text(json.dumps(FAIL,indent=2))
    (OUT/'json/models.json').write_text(json.dumps(MODELS,indent=2))

def main():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    smoke='--smoke' in sys.argv
    for mid in ['Qwen/Qwen2.5-0.5B','facebook/opt-125m','EleutherAI/pythia-70m']:
        try:
            print('LOAD',mid,flush=True)
            tok=AutoTokenizer.from_pretrained(mid)
            model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
            layers=capture(model,tok,mid)
            MODELS.append(dict(model=mid,revision=getattr(model.config,'_commit_hash',None),layers=len(layers)))
            for name,w,xs,bias in layers:
                print('LAYER_START',name,list(w.shape),[len(x) for x in xs],flush=True)
                with torch.no_grad(): run_layer(name,w,xs,bias)
                save()
                if smoke: break
            del model,layers,w,xs; torch.cuda.empty_cache()
            if smoke: break
        except Exception:
            FAIL.append(dict(model=mid,traceback=traceback.format_exc())); print(FAIL[-1],flush=True); save()
            raise
    (OUT/'json/environment.json').write_text(json.dumps(dict(seed=20260922,gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,smoke=smoke),indent=2))
    print('V5_DONE',len(ROWS),flush=True)

if __name__=='__main__': main()
