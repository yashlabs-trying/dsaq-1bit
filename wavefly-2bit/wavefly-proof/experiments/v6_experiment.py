"""Static V6 ablations, then calibration-selected joint candidates.

All costs are byte-aligned representation sizes. Reference matrices are used
offline only. No runtime routing or deployed GPU kernel is implemented.
"""
import sys, json, math, traceback
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import v5_experiment as v5
import v4_experiment as v4

ROOT=v5.ROOT; OUT=ROOT/'results/v6'; CACHE=ROOT/'.v6-cache'
for d in [OUT/'csv',OUT/'json',OUT/'plots',CACHE]: d.mkdir(parents=True,exist_ok=True)
torch.manual_seed(20260923); np.random.seed(20260923)
ROWS=[]; CHECKS=[]; MODELS=[]; SHARED={}
FRACTIONS=[.0025,.005,.01,.02,.03,.05]
AVGS=[1,1.25,1.5,1.75,2,2.25,2.5,3,4]
FIELDS=['bulk_codes','macro_scales','micro_scales','zero_points','codebooks','residual8','residual4','residual2','residual16','residual_scales','indices','permutation','metadata','legacy_salient','legacy_correction']

def storage(**kw): return {k:v5.aligned(kw.get(k,0)) for k in FIELDS}

def prepare_cache():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    paths=[]; ratios=[]
    for mid in ['Qwen/Qwen2.5-0.5B','facebook/opt-125m','EleutherAI/pythia-70m']:
        print('CAPTURE',mid,flush=True)
        tok=AutoTokenizer.from_pretrained(mid)
        model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
        layers=v5.capture(model,tok,mid)
        MODELS.append(dict(model=mid,revision=getattr(model.config,'_commit_hash',None),layers=len(layers)))
        for name,w,xs,bias in layers:
            path=CACHE/f'{len(paths):02d}.pt'
            torch.save(dict(name=name,w=w.cpu(),xs=[x.cpu() for x in xs]),path); paths.append(path)
            z=w.reshape(-1,32); gr=z.square().mean(1).sqrt(); macro=gr.reshape(-1,4).square().mean(1).sqrt().repeat_interleave(4)
            ratio=gr/macro.clamp_min(1e-12)
            ratios.append(ratio[::max(1,len(ratio)//2048)].cpu())
        del model,layers,w,xs; torch.cuda.empty_cache()
    # One FP16 table shared globally; charge it conservatively in every layer.
    z=torch.cat(ratios).cuda()[:,None]; cb,_=v5.lloyd(z,4)
    table=cb.flatten().sort().values.half().float()
    (OUT/'json/setup.json').write_text(json.dumps(dict(models=MODELS,micro_table=table.tolist(),paths=[str(p) for p in paths]),indent=2))
    return paths,table

def pack_groups(w,x,perm):
    wp=w[:,perm]; m,n=wp.shape; ng=math.ceil(n/32)
    z=torch.nn.functional.pad(wp,(0,ng*32-n)).reshape(-1,32)
    xp=torch.nn.functional.pad(x[:,perm],(0,ng*32-n)).reshape(len(x),ng,32)
    gram=torch.einsum('tgi,tgj->gij',xp,xp)/len(x)
    weights=xp.square().mean(0).repeat(m,1)
    mask=torch.ones(m,ng*32,device=w.device,dtype=torch.bool); mask[:,n:]=False
    return dict(z=z,gram=gram,weights=weights,mask=mask.reshape(-1,32),m=m,n=n,ng=ng,perm=perm)

def functional_loss(e,p):
    # Full covariance within each channel group; no cross-group interaction.
    e=e.reshape(p['m'],p['ng'],32)
    return torch.einsum('rgi,gij,rgj->rg',e,p['gram'],e).clamp_min(0).flatten()

def options(p,table,zero_bits=0,objective='functional'):
    z=p['z']; padded=torch.nn.functional.pad(z,(0,0,0,(-len(z))%4))
    macro=padded.reshape(-1,128).square().mean(1,keepdim=True).sqrt().clamp_min(2**-9).to(torch.float8_e4m3fn).float()
    s=macro.repeat_interleave(4,0)[:len(z)]
    ratios=torch.tensor([1.],device=z.device) if table is None else table
    offsets=([0.] if zero_bits==0 else ([-.25,.25] if zero_bits==1 else [-.5,-.16666667,.16666667,.5]))
    opts=[]; cbits=0
    for bit in [1,2,3,4]:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(20260923+bit)
            cb,_=v5.lloyd((z/s).flatten()[:,None],2**bit)
        lev=cb.flatten().sort().values; cbits+=lev.numel()*16
        best=torch.full((len(z),),float('inf'),device=z.device); qbest=torch.zeros_like(z)
        for ratio in ratios:
            eff=s*ratio
            for offset in offsets:
                centers=lev-offset
                idx=torch.bucketize(z/eff,(centers[:-1]+centers[1:])/2)
                q=centers[idx]*eff*p['mask']
                err=z-q
                loss=functional_loss(err,p) if objective=='functional' else err.square().sum(1)
                take=loss<best; best[take]=loss[take]; qbest[take]=q[take]
        opts.append((qbest,functional_loss(z-qbest,p),0))
    st=storage(macro_scales=macro.numel()*8,micro_scales=len(z)*2 if table is not None else 0,
               zero_points=len(z)*zero_bits,codebooks=cbits+(64 if table is not None else 0),metadata=384)
    return opts,st

def allocate_groups(opts,p,avg,mixed):
    if mixed: return v5.adaptive(opts,p['z'],avg)
    lo=int(math.floor(avg)); hi=min(4,lo+1); count=round((avg-lo)*len(p['z']))
    # Evenly spaced higher precision groups define the fractional uniform control.
    q=opts[lo-1][0].clone()
    if count:
        ids=torch.linspace(0,len(q)-1,count,device=q.device).long(); q[ids]=opts[hi-1][0][ids]
    return q,(lo*len(q)+count)*32,(len(q) if count else 0)

def restore(q,p):
    wp=q.reshape(p['m'],-1)[:,:p['n']]
    out=torch.empty_like(wp); out[:,p['perm']]=wp
    return out

def perm_cost(perm,mode):
    if mode=='original': return 0
    return len(perm)*(7 if mode=='block_local' else math.ceil(math.log2(len(perm))))

def record(name,w,xs,h,scope,config,st,**extra):
    es=v5.errors(w,h,xs)
    row=dict(layer=name,n=w.numel(),out=w.shape[0],inp=w.shape[1],scope=scope,config=config,
             error_a=es[0],error_b=es[1],error_c=es[2],bits=sum(st.values()),bpw=sum(st.values())/w.numel(),**st,**extra)
    ROWS.append(row); return row

def residual_sweep(name,w,xs,h,base,scope,metrics=('functional',),reference=False):
    r=w-h; rn=r.square().sum(0).sqrt(); ar=xs[0].square().mean(0).sqrt()
    scores={'magnitude':rn,'functional':rn*ar,'activation_rms':ar,'original_functional':w.norm(dim=0)*ar}
    residual_out=[x@r.T for x in xs]; den=[(x@w.T).norm().clamp_min(1e-30) for x in xs]
    base_st={k:base[k] for k in FIELDS}
    def add(metric,fraction,tiers,label):
        order=scores[metric].argsort(descending=True); count=round(w.shape[1]*fraction); selected=order[:count]
        corr=torch.zeros(w.shape[0],count,device=w.device); st=dict(base_st)
        start=0
        for stop,bits in tiers:
            end=round(stop*count); cols=selected[start:end]
            if len(cols):
                q,val,sc=v4.quant_salient(r,cols,bits); corr[:,start:end]=q
                st[f'residual{bits}']+=v5.aligned(val); st['residual_scales']+=v5.aligned(sc)
            start=end
        index=min(w.shape[1],count*math.ceil(math.log2(w.shape[1])))+1
        if len(tiers)>1: index+=count*2
        st['indices']+=v5.aligned(index); st['metadata']+=32
        es=[float((e-x[:,selected]@corr.T).norm()/dn) for x,e,dn in zip(xs,residual_out,den)]
        row=dict(base,scope=scope,config=base['config']+f'+res_{metric}_{fraction}_{label}',**st)
        row.update(bits=sum(st.values()),bpw=sum(st.values())/w.numel(),error_a=es[0],error_b=es[1],error_c=es[2],
                   residual_metric=metric,residual_fraction=fraction,residual_precision=label,
                   base_error_a=base['error_a'],base_error_b=base['error_b'],base_error_c=base['error_c'],
                   residual_bpw=(sum(st.values())-base['bits'])/w.numel(),damage_recovered_b=1-(es[1]/max(base['error_b'],1e-30))**2)
        ROWS.append(row)
    for metric in metrics:
        for frac in FRACTIONS:
            for bits in ([2,4,8,16] if reference else [2,4,8]): add(metric,frac,[(1.,bits)],str(bits))
    # AR×residual and squared direct functional score have identical rankings.
    for frac in FRACTIONS:
        for proportions in [(0.2,0.5),(0.1,0.3),(0.5,0.8)]:
            add('functional',frac,[(proportions[0],8),(proportions[1],4),(1.,2)],f'cascade{proportions}')

def run_layer(name,w,xs,global_table):
    n=w.shape[1]; identity=torch.arange(n,device=w.device)
    p0=pack_groups(w,xs[0],identity)
    opts0,st0=options(p0,None,0,'mse')
    h0=restore(allocate_groups(opts0,p0,1.5,False)[0],p0)
    rn=(w-h0).norm(dim=0); ar=xs[0].square().mean(0).sqrt(); wr=w.square().mean(0).sqrt()
    score=rn*ar
    feat=torch.stack([wr,ar,rn,score],1)
    feat=(feat-feat.mean(0))/feat.std(0).clamp_min(1e-12)
    feature_perm=v4.ordered_permutation(feat,8,True,'centroid',seed=20260923)
    local=torch.cat([part[score[part].argsort()] for part in identity.split(128)])
    # Share one calibration-derived ordering where model and input width agree.
    key=('/'.join(name.split('/')[:2]),n)
    if key not in SHARED: SHARED[key]=ar.argsort()
    orderings={'original':identity,'activation':ar.argsort(),'residual':rn.argsort(),
               'activation_residual':score.argsort(),'features':feature_perm,'block_local':local,'shared':SHARED[key]}
    micro_schemes=[('off',None,0,'mse')]
    for kind,table in [('fixed',torch.tensor([.5,.75,1.,1.5],device=w.device)),('learned',global_table)]:
        for zero in [0,1,2]:
            for obj in ['mse','functional']: micro_schemes.append((kind,table,zero,obj))
    order_candidates=[]; scale_candidates=[]; full_bases=[]
    def evaluate(order,scheme,mixed,scope,averages=AVGS):
        kind,table,zero,obj=scheme
        p=pack_groups(w,xs[0],orderings[order])
        # Restore check ensures every permutation is a true bijection.
        assert torch.equal(restore(p['z'],p),w)
        if order=='original' and kind=='off': opts,st=opts0,st0
        else: opts,st=options(p,table,zero,obj)
        rows=[]
        for avg in averages:
            q,code,meta=allocate_groups(opts,p,avg,mixed)
            ss=dict(st,bulk_codes=v5.aligned(code),metadata=st['metadata']+v5.aligned(meta),permutation=v5.aligned(perm_cost(orderings[order],order)))
            h=restore(q,p)
            cfg=f'{order}_{kind}_z{zero}_{obj}_{"mixed" if mixed else "uniform"}_{avg}'
            row=record(name,w,xs,h,scope,cfg,ss,grouping=order,micro=kind,zero_bits=zero,scale_objective=obj,mixed=mixed,bulk_avg=avg)
            rows.append((row,h))
        return rows
    control=evaluate('original',micro_schemes[0],False,'control')
    for row,h in control:
        if row['bulk_avg'] in [1,1.5,2]:
            residual_sweep(name,w,xs,h,row,'residual',('magnitude','functional','activation_rms','original_functional'),True)
    # Independently change only ordering, scaling or precision allocation.
    for order in orderings:
        candidates=evaluate(order,micro_schemes[0],False,'grouping')
        score=min((r['error_a'] for r,h in candidates if r['bpw']<=2.05),default=float('inf'))
        order_candidates.append((score,order))
    for scheme in micro_schemes[1:]:
        candidates=evaluate('original',scheme,False,'scaling')
        score=min((r['error_a'] for r,h in candidates if r['bpw']<=2.05),default=float('inf'))
        scale_candidates.append((score,scheme))
    evaluate('original',micro_schemes[0],True,'mixed_bulk')
    control_score=min(r['error_a'] for r,h in control if r['bpw']<=2.05)
    winning_orders=[order for score,order in sorted(order_candidates) if score<control_score][:2]
    winning_scales=[scheme for score,scheme in sorted(scale_candidates,key=lambda x:x[0]) if score<control_score][:2]
    selected_orders=list(dict.fromkeys(['original']+winning_orders))
    selected_schemes=[micro_schemes[0]]+winning_scales
    for order in selected_orders:
        for scheme in selected_schemes:
            full_bases+=evaluate(order,scheme,True,'joint_base')
    # Joint refinement chooses bases on A, then re-trades bulk and residual bits.
    selected={}
    for budget in [1.25,1.5,1.75,2.0]:
        feasible=[t for t in full_bases if t[0]['bpw']<=budget]
        if feasible:
            row,h=min(feasible,key=lambda t:t[0]['error_a']); selected[row['config']]=(row,h)
    for row,h in selected.values(): residual_sweep(name,w,xs,h,row,'joint',('functional',),False)
    CHECKS.append(dict(layer=name,permutation_roundtrip=True,selected_orders=selected_orders,
                       selected_micro=[(s[0],s[2],s[3]) for s in selected_schemes],calibration_tokens=len(xs[0]),validation_tokens=len(xs[1]),ood_tokens=len(xs[2])))
    print('LAYER_DONE',name,len(ROWS),flush=True)

def main():
    setup=OUT/'json/setup.json'
    if setup.exists():
        data=json.loads(setup.read_text()); paths=[Path(p) for p in data['paths']]; table=torch.tensor(data['micro_table'],device='cuda')
    else: paths,table=prepare_cache()
    smoke='--smoke' in sys.argv
    for path in paths[:1] if smoke else paths:
        try:
            data=torch.load(path,weights_only=False); w=data['w'].cuda(); xs=[x.cuda() for x in data['xs']]
            print('LAYER_START',data['name'],flush=True)
            with torch.no_grad(): run_layer(data['name'],w,xs,table)
            pd.DataFrame(ROWS).to_csv(OUT/'csv/configurations.csv',index=False)
            (OUT/'json/checks.json').write_text(json.dumps(CHECKS,indent=2))
            del data,w,xs; torch.cuda.empty_cache()
        except Exception:
            (OUT/'json/failure.json').write_text(traceback.format_exc()); raise
    (OUT/'json/environment.json').write_text(json.dumps(dict(gpu=torch.cuda.get_device_name(),torch=torch.__version__,seed=20260923,smoke=smoke),indent=2))
    print('V6_DONE',len(ROWS),flush=True)

if __name__=='__main__': main()
