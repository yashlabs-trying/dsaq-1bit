"""Calibration-only mean/worst allocation and the complete V6 report."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import lil_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/v6'
new=pd.read_csv(OUT/'csv/configurations.csv').fillna(0)
old=pd.read_csv(ROOT/'results/v5/csv/configurations.csv').fillna(0)
repro=pd.read_csv(OUT/'baseline_reproduction/csv/configurations.csv').fillna(0)
oldsel=pd.read_csv(ROOT/'results/v5/csv/allocations.csv')
oldsel=oldsel[(oldsel.method=='+ static correction')&(oldsel.target==2)]
audit=oldsel[['layer','config','bits','error_a','error_b','error_c']].merge(repro.drop_duplicates(['layer','config'])[['layer','config','bits','error_a','error_b','error_c']],on=['layer','config'],suffixes=('_old','_reproduced'))
assert len(audit)==33
for c in ['bits','error_a','error_b','error_c']: audit[c+'_delta']=audit[c+'_reproduced']-audit[c+'_old']
audit.to_csv(OUT/'csv/baseline_audit.csv',index=False)
assert audit.bits_delta.abs().max()==0
assert audit.error_b_delta.abs().max()<1e-3,'V5 baseline did not reproduce within 0.1 percentage point per layer'
old=old[~old.family.isin(['oracle','learned_router'])].copy()
fields=['bulk_codes','macro_scales','micro_scales','zero_points','codebooks','residual8','residual4','residual2','residual16','residual_scales','indices','permutation','metadata','legacy_salient','legacy_correction']
for k in fields:
    if k not in old: old[k]=0
old['bulk_codes']=old.codes; old['macro_scales']=old.scales; old['legacy_salient']=old.salient; old['legacy_correction']=old.correction
old['scope']='legacy'; old['grouping']='V5'; old['micro']='off'; old['zero_bits']=0
eligible_new=new[new.residual_precision.astype(str)!='16']
data=pd.concat([old,eligible_new],ignore_index=True).fillna(0)
layers=list(new.layer.unique()); assert len(layers)==33
N=data.groupby('layer').n.first().sum()
TARGETS=[1.75,1.9,2.,2.1,2.25,2.5]
LAMBDA=.25
FAMILIES={'Bit stealing':['legacy'], 'Bridge control':['legacy','control'],
 'Single residual':['legacy','control','residual'],
 'Residual saliency':['legacy','control','residual'], 'Activation grouping':['legacy','control','grouping'],
 'Micro-scaling':['legacy','control','scaling'], 'Mixed bulk bits':['legacy','control','mixed_bulk'],
 'Full V6':['legacy','control','residual','grouping','scaling','mixed_bulk','joint_base','joint']}
SUMMARY=[]; ALLOCATIONS=[]; SOLVERS=[]

def pareto(frame):
    parts=[]
    for _,g in frame.groupby('layer',sort=False):
        best=np.inf; ids=[]
        for i,r in g.sort_values(['bits','error_a']).iterrows():
            if r.error_a<best-1e-10: best=r.error_a; ids.append(i)
        parts.append(g.loc[ids])
    return pd.concat(parts).reset_index(drop=True)

def allocate(frame,target,lam,algorithm='constrained'):
    p=pareto(frame)
    if algorithm=='greedy':
        groups=[g for _,g in p.groupby('layer',sort=False)]; ids=[g.index[0] for g in groups]
        while True:
            chosen=p.loc[ids]; room=target*N-chosen.bits.sum()
            current=chosen.error_a.mean()+lam*chosen.error_a.max(); upgrades=[]
            for j,g in enumerate(groups):
                oldr=p.loc[ids[j]]
                othermax=p.loc[ids[:j]+ids[j+1:]].error_a.max()
                for i,r in g.iterrows():
                    cost=r.bits-oldr.bits
                    obj=chosen.error_a.mean()+(r.error_a-oldr.error_a)/len(layers)+lam*max(othermax,r.error_a)
                    gain=current-obj
                    if 0<cost<=room and gain>0: upgrades.append((gain/cost,j,i))
            if not upgrades: break
            _,j,i=max(upgrades); ids[j]=i
        return p.loc[ids]
    n=len(p); L=len(layers); mat=lil_matrix((2*L+1,n+1))
    for j,layer in enumerate(layers):
        ids=np.flatnonzero(p.layer.to_numpy()==layer)
        mat[j,ids]=1; mat[L+1+j,ids]=p.iloc[ids].error_a.to_numpy(); mat[L+1+j,-1]=-1
    mat[L,:n]=p.bits.to_numpy()/N
    low=np.r_[np.ones(L),-np.inf,np.full(L,-np.inf)]; high=np.r_[np.ones(L),target,np.zeros(L)]
    c=np.r_[p.error_a.to_numpy()/L,lam]
    result=milp(c,integrality=np.r_[np.ones(n),0],bounds=Bounds(np.zeros(n+1),np.r_[np.ones(n),1000]),
                constraints=LinearConstraint(mat.tocsr(),low,high),options={'time_limit':15,'mip_rel_gap':1e-4})
    SOLVERS.append(dict(target=target,lam=lam,status=result.status,message=result.message,gap=getattr(result,'mip_gap',None),choices=n))
    if result.x is None: return allocate(frame,target,lam,'greedy')
    chosen=p.iloc[np.flatnonzero(result.x[:n]>.5)]
    assert len(chosen)==L and chosen.bits.sum()/N<=target+1e-8
    return chosen

def summarize(chosen,method,target,lam,algorithm):
    row=dict(method=method,target=target,lam=lam,algorithm=algorithm,bpw=chosen.bits.sum()/N,
             mean=chosen.error_b.mean(),median=chosen.error_b.median(),p90=chosen.error_b.quantile(.9),worst=chosen.error_b.max(),std=chosen.error_b.std(ddof=0),
             calibration=chosen.error_a.mean(),cal_worst=chosen.error_a.max(),ood=chosen.error_c.mean())
    SUMMARY.append(row); chosen=chosen.copy(); chosen['method']=method; chosen['target']=target; chosen['lam']=lam; chosen['algorithm']=algorithm; ALLOCATIONS.append(chosen)

if '--cached' in sys.argv:
    summary=pd.read_csv(OUT/'csv/summary.csv'); allocations=pd.read_csv(OUT/'csv/allocations.csv')
else:
    for target in TARGETS:
        prev=pd.read_csv(ROOT/'results/v5/csv/allocations.csv')
        chosen=prev[(prev.method=='+ static correction')&(prev.target==target)]
        if len(chosen)==33: summarize(chosen,'V5',target,0,'historical constrained')
        for method,scopes in FAMILIES.items():
            pool=data[data.scope.isin(scopes)]
            if method=='Single residual': pool=pool[~pool.residual_precision.astype(str).str.startswith('cascade')]
            summarize(allocate(pool,target,LAMBDA),method,target,LAMBDA,'constrained')
        summarize(allocate(data,target,LAMBDA,'greedy'),'Full V6 greedy',target,LAMBDA,'greedy')
        print('ALLOCATED',target,flush=True)
    for lam in [0.,1.]: summarize(allocate(data,2.,lam),'Full V6 lambda sweep',2.,lam,'constrained')
    # A true uniform per-layer budget reference, distinct from adaptive V5.
    choices=[]
    for _,g in data[data.scope=='legacy'].groupby('layer'):
        feasible=g[g.bpw<=2.]
        choices.append(feasible.sort_values('error_a').iloc[[0]])
    summarize(pd.concat(choices),'Uniform per-layer cap',2.,0.,'independent')
    summary=pd.DataFrame(SUMMARY); allocations=pd.concat(ALLOCATIONS)
    summary.to_csv(OUT/'csv/summary.csv',index=False); allocations.to_csv(OUT/'csv/allocations.csv',index=False)
    (OUT/'json/solver.json').write_text(json.dumps(SOLVERS,indent=2))

best=summary[(summary.method=='Full V6')&(summary.target==2)].iloc[0]
chosen=allocations[(allocations.method=='Full V6')&(allocations.target==2)].copy()
baseline=summary[(summary.method=='V5')&(summary.target==2)].iloc[0]
hard=oldsel[oldsel.error_b>.25][['layer','bpw','error_b']].merge(chosen[['layer','bpw','error_b','config','base_error_b']],on='layer',suffixes=('_v5','_v6'))
hard['bulk_error']=np.where(hard.base_error_b>0,hard.base_error_b,hard.error_b_v6)
hard['residual_gain']=hard.bulk_error-hard.error_b_v6
hard.to_csv(OUT/'csv/hard_layers.csv',index=False)
comparison=oldsel[['layer','bpw','error_b']].merge(chosen[['layer','bpw','error_b']],on='layer',suffixes=('_v5','_v6'))
comparison.to_csv(OUT/'csv/layer_comparison.csv',index=False)

# Calibration distortion curves and marginal gains at requested layer caps.
curve=[]
for layer,g in data.groupby('layer'):
    prev=None
    for cap in [1,1.25,1.5,1.75,2,2.25,2.5,3,4]:
        f=g[g.bpw<=cap]
        if not len(f): continue
        r=f.sort_values('error_a').iloc[0]
        gain=(prev.error_a-r.error_a)/(r.bpw-prev.bpw) if prev is not None and r.bpw>prev.bpw else 0
        curve.append(dict(layer=layer,cap=cap,bpw=r.bpw,error_a=r.error_a,error_b=r.error_b,gain_per_bpw=gain,config=r.config)); prev=r
pd.DataFrame(curve).to_csv(OUT/'csv/distortion_curves.csv',index=False)

def table(d,cols):
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(f'{r[c]:.5f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |' for _,r in d.iterrows())

def savefig(name):
    plt.tight_layout(); plt.savefig(OUT/'plots'/name); plt.close()
for metric in ['mean','worst']:
    plt.figure(figsize=(10,6))
    for method,g in summary[~summary.method.isin(['Full V6 lambda sweep','Uniform per-layer cap'])].groupby('method'):
        plt.plot(g.bpw,g[metric]*100,'o-',label=method)
    v4=pd.read_csv(ROOT/'results/v5/csv/summary.csv'); v4=v4[v4.method=='V4 uniform']
    plt.plot(v4.bpw,v4[metric]*100,'--',label='V4 bias-excluded')
    plt.xlim(1.7,2.55); plt.xlabel('Weight-count-weighted true BPW'); plt.ylabel(metric+' held-out layer error (%)'); plt.legend(fontsize=8); plt.grid(alpha=.2); savefig(metric+'_vs_bpw.png')
plt.figure(figsize=(12,5)); ix=np.argsort(comparison.error_b_v5.to_numpy()); plt.plot(comparison.error_b_v5.to_numpy()[ix]*100,'o-',label='V5'); plt.plot(comparison.error_b_v6.to_numpy()[ix]*100,'o-',label='V6'); plt.xlabel('Layer sorted by V5 error'); plt.ylabel('Held-out error (%)'); plt.legend(); savefig('sorted_errors.png')
plt.figure(figsize=(12,5)); plt.plot(comparison.bpw_v5.to_numpy()[ix],'o-',label='V5'); plt.plot(comparison.bpw_v6.to_numpy()[ix],'o-',label='V6'); plt.xlabel('Layer sorted by V5 error'); plt.ylabel('Allocated BPW'); plt.legend(); savefig('layer_bpw.png')
res=new[new.residual_fraction>0]
plt.figure(figsize=(9,5))
for precision in ['2','4','8','16']:
    g=res[(res.scope=='residual')&(res.residual_metric=='functional')&(res.residual_precision.astype(str)==precision)]
    if len(g):
        avg=g.groupby('residual_fraction').damage_recovered_b.mean(); plt.plot(avg.index*100,avg.values*100,'o-',label='INT'+precision if precision!='16' else 'FP16')
plt.xlabel('Residual columns retained (%)'); plt.ylabel('Mean squared output damage recovered (%)'); plt.legend(); savefig('residual_fraction.png')
plt.figure(figsize=(9,5)); plt.scatter(res.residual_bpw,res.damage_recovered_b*100,s=3,alpha=.08); plt.xlabel('Additional residual BPW'); plt.ylabel('Squared output damage recovered (%)'); savefig('residual_roi.png')
cost=chosen[fields].sum()/N
plt.figure(figsize=(12,5)); cost.plot.bar(); plt.ylabel('Global BPW'); savefig('storage.png')
main=summary[(summary.target==2)&(~summary.method.isin(['Full V6 lambda sweep']))].copy()
for c in ['mean','median','p90','worst','std','calibration','cal_worst','ood']: main[c]*=100

bridge=summary[(summary.method=='Bridge control')&(summary.target==2)].iloc[0]
contrib=[]
for method in ['Bit stealing','Residual saliency','Activation grouping','Micro-scaling','Mixed bulk bits']:
    r=summary[(summary.method==method)&(summary.target==2)].iloc[0]
    before=baseline if method=='Bit stealing' else bridge
    contrib.append(dict(component=method,before=100*before['mean'],after=100*r['mean'],gain=100*(before['mean']-r['mean']),bpw_difference=r.bpw-before.bpw))
contrib=pd.DataFrame(contrib); contrib.to_csv(OUT/'csv/component_contributions.csv',index=False)
single=summary[(summary.method=='Single residual')&(summary.target==2)].iloc[0]
mixedres=summary[(summary.method=='Residual saliency')&(summary.target==2)].iloc[0]
contrib=pd.concat([contrib,pd.DataFrame([dict(component='Mixed residual',before=100*single['mean'],after=100*mixedres['mean'],gain=100*(single['mean']-mixedres['mean']),bpw_difference=mixedres.bpw-single.bpw)])],ignore_index=True)
contrib.to_csv(OUT/'csv/component_contributions.csv',index=False)
diagnostic=res[(res.scope=='residual')&(res.bulk_avg==1.5)&(res.residual_fraction==.01)&(res.residual_precision.astype(str)=='8')]
metric_summary=diagnostic.groupby('residual_metric')[['error_b','damage_recovered_b','bpw']].mean().reset_index()
metric_summary.to_csv(OUT/'csv/saliency_comparison.csv',index=False)
winner=contrib.sort_values('gain',ascending=False).iloc[0]
minimum=best['mean']<.15 and best.worst<.35
strongpartial=best['mean']<.10 and best.worst<.20
strong=best['mean']<.05 and best.worst<.10
verdict='STRONG PASS' if strong else ('PARTIAL' if minimum else 'FAIL')
selected_new=chosen[chosen.scope!='legacy']
counts=chosen.scope.value_counts().to_dict()
hardest=chosen.sort_values('error_b',ascending=False).iloc[0]; easiest=chosen.sort_values('error_b').iloc[0]
scale_counts=chosen.micro.value_counts().to_dict(); group_counts=chosen.grouping.value_counts().to_dict()
rescounts=chosen[chosen.residual_fraction>0].residual_precision.value_counts().to_dict()
report=f'''# WaveFly V6 experiment

## 1. V5 baseline reproduction

V5 was independently rerun on the same RTX 2000 Ada, same checkpoint revisions, same 33 matrices, and unchanged A/B/C texts and bias-excluded metric. Selected V5 representations reproduce with maximum absolute B-error difference **{audit.error_b_delta.abs().max():.8f}** and identical counted bits. The rerun mean is {100*audit.error_b_reproduced.mean():.4f}% versus historical {100*baseline['mean']:.4f}%; the small run-to-run variation is retained in the audit. The control is {baseline.bpw:.6f} BPW, {100*baseline['mean']:.2f}% mean, {100*baseline.worst:.2f}% worst, and {100*baseline.ood:.2f}% new-domain error.

Global BPW refers to the weight-count-weighted union of the sampled matrices across three models, not a deployed complete model. Errors are relative output Frobenius norms, not token error rates or perplexity. B was used in earlier development; C contains six additional passages. Neither B nor C chooses V6 representations or allocation.

## 2. Cross-layer bit allocation

V5 already used cross-layer allocation. V6 adds an explicit worst-layer penalty and expands candidate representations; it does not claim bit stealing was absent from V5. The objective is mean calibration error + λ × maximum calibration error (ω_l=1/33). Primary λ=0.25 was fixed before evaluation; λ=0 and 1 are diagnostic tradeoffs at 2 BPW. Greedy gain-per-stored-bit and a discrete constrained optimizer are both reported. The solver uses a 15-second limit and retains feasible solutions; statuses/gaps are in `json/solver.json`.

The uniform-cap reference restricts each layer to ≤2 BPW and can leave global capacity unused; its actual BPW is shown. It is distinct from the already adaptive V5 control. All other comparisons enforce the global budget, allowing small hard layers to exceed 2 BPW. Per-layer distortion curves cover caps 1/1.25/1.5/1.75/2/2.25/2.5/3/4, including actual attainable rates and marginal gains.

{table(summary[(summary.target==2)&(summary.method.isin(['V5','Bit stealing','Full V6 greedy','Uniform per-layer cap','Full V6 lambda sweep']))],['method','lam','bpw','mean','worst','calibration','cal_worst'])}

Errors in diagnostic tables are fractions unless labelled percentages.

## 3. Residual-driven saliency

After bulk quantization, scores compare residual-column norm, activation RMS × residual norm, activation RMS alone, and original weight norm × activation RMS. Squared direct functional damage is exactly the square of activation RMS × residual norm times the token count, so it has the same ranking; this equivalence avoids a redundant numerical sweep.

Residual fractions are 0.25/0.5/1/2/3/5%, with INT2/4/8 and FP16 references. Corrections store the actual residual, including under the original-score controls. This isolates selection criteria; it is not an exact implementation of V5's remove-original-columns path, which remains available as a separate legacy candidate.

![Residual concentration](results/v6/plots/residual_fraction.png)

Matched diagnostic: 1.5-bit preliminary bulk, 1% residual columns, INT8 correction (errors and recovered damage are fractions):

{table(metric_summary,['residual_metric','error_b','damage_recovered_b','bpw'])}

## 4. Activation-aware grouping

Orderings are original, activation-RMS sort, residual-RMS sort, activation×residual sort, lightweight normalized four-feature clustering, block-local (128 channels), and a shared ordering for matching model/input width. Residual features come from a fixed 1.5-bit preliminary quantizer. No expensive permutation search is used. Every ordering has an exact reconstruction/bijection assertion.

Full permutations count ceil(log2 input_width) bits per input channel. Block-local counts seven bits per channel. Shared orderings are conservatively charged in every layer; no unimplemented storage-sharing discount is taken. Activation-dependent sorted order cannot be reconstructed implicitly from codes at deployment, so it is not treated as free. Original order needs no permutation. Row permutations are not added in V6.

## 5. Nested micro-scaling

One FP8 E4M3 macro scale covers four G32 groups. Optional two-bit micro codes select fixed [0.5,0.75,1,1.5] or one globally learned four-entry FP16 multiplier table. The learned table uses pooled weight-derived group/macro RMS ratios across all sampled matrices, without B/C activations, and is conservatively charged in every layer. No floating-point micro value is stored per group.

Zero-point choices use 0/1/2 bits per group. Fixed codebook shifts are [0], [-0.25,0.25], or [-0.5,-1/6,1/6,0.5]. Local candidates are chosen by either weight MSE or full calibration covariance within G32. This includes within-group correlations but omits cross-group terms during local selection; final layer-output evaluation is exact. Macro RMS is rounded to FP8 before fitting/assignment. Learned scalar levels and multiplier tables are rounded to FP16 before use. Identical codebook seeds hold the base fit fixed across scale ablations.

## 6. Mixed bulk precision

Bulk options are 1/2/3/4-bit learned scalar codes. Fractional uniform controls distribute the next precision evenly across groups. Mixed allocation instead minimizes block-functional distortion with a Lagrange multiplier and counts a two-bit precision tag per group. Actual cost includes all codes and metadata; requested average bulk bits are not substituted for total BPW.

## 7. Mixed residual precision

Single-precision corrections compete against cascades assigning the highest residual-score columns INT8, the next INT4, and the rest INT2, using three predefined tier proportions. Each corrected column stores its value stream and FP16 scale (FP16 reference values need no quantization scale), plus packed indices or the cheaper bitmap. Cascade precision tags add two bits per retained column. Full V6 excludes FP16 residual-reference candidates.

## 8. Full V6 joint optimization

Independent ablations run first within each layer. Orderings and micro-scale choices are shortlisted using calibration error at matched attainable rates near 2 BPW; unchanged controls remain available. Joint candidates combine winning order/scale options with mixed group precision. Bases selected under 1.25/1.5/1.75/2 BPW are refined with residual sweeps, allowing global allocation to trade bulk against correction storage. The final candidate union includes all V5 static candidates and V6 candidates; calibration Pareto pruning discards dominated choices without deleting their raw measurements. Dynamic routing is absent.

The selected scope counts are {counts}. Selected scaling: {scale_counts}. Selected grouping: {group_counts}. Selected residual precisions: {rescounts}. Full V6 can retain V5 representations when new components fail to earn their cost.

## 9. Exact BPW accounting

{table(pd.DataFrame({'component':cost.index,'bpw':cost.values,'bytes':cost.values*N/8}),['component','bpw','bytes'])}

Each component is byte aligned. V5 candidates retain their original complete codebook/salient/static-correction costs; their combined metadata remains classified as legacy metadata because the old CSV did not split every index stream. This is an explicit representation size model, not a physically serialized V6 container. Offline dense reconstructed tensors and original reference weights are not deployment payload. No claim of V6 pack/unpack validation is made.

![Storage](results/v6/plots/storage.png)

## 10. Held-out accuracy

At {best.bpw:.6f} BPW: mean **{100*best['mean']:.2f}%**, median {100*best['median']:.2f}%, P90 {100*best.p90:.2f}%, worst **{100*best.worst:.2f}%**, standard deviation {100*best['std']:.2f}%.

## 11. OOD/generalization

Calibration mean {100*best.calibration:.2f}%; held-out B {100*best['mean']:.2f}%; different-domain C {100*best.ood:.2f}%. These short fixed passages cannot establish model-level generalization or language-model accuracy.

## 12. Worst-layer rescue

All V5 layers above 25% B error appear below. For a retained legacy static correction, an isolated bulk measurement may not exist in this table; such rows repeat final error as bulk and zero gain rather than invent a decomposition.

{table(hard,['layer','error_b_v5','bpw_v6','bulk_error','residual_gain','error_b_v6'])}

![Sorted errors](results/v6/plots/sorted_errors.png)
![Layer allocation](results/v6/plots/layer_bpw.png)

## 13. Ablation results

Percentages below. Each independent ablation adds only its candidate family to V5 plus the common V6 bridge control. Bit-stealing uses unchanged V5 candidates. λ is held at 0.25 across new families. Thus gains are matched-budget system gains after calibration allocation, not a claim that each selected layer used the added component. Failed/fallback configurations remain in the raw CSV.

{table(main,['method','bpw','mean','median','p90','worst','std','ood'])}

{table(contrib,['component','before','after','gain','bpw_difference'])}

## 14. Pareto frontier

![Mean versus BPW](results/v6/plots/mean_vs_bpw.png)
![Worst versus BPW](results/v6/plots/worst_vs_bpw.png)
![Residual return](results/v6/plots/residual_roi.png)

## 15. Kernel feasibility

The selected representation accounts for {chosen.bits.sum()/8:,.0f} total bytes across sampled matrices. Bulk lookup/unpack is roughly one code operation per weight; nested scaling requires one macro×micro product per G32, then a dequantization multiply per weight. Zero points add a subtraction. Matrix arithmetic is approximately two FLOPs per weight per token, plus correction work. Residual column corrections use contiguous stored columns with O(retained_columns) index reads and approximately 2×stored_residual_values multiply/add operations per token. They require no per-weight index table. Full permutations do require an input gather; block-local/original ordering reduce its scope.

The ideal weight-only intensity is approximately 16/BPW FLOPs per payload byte for single-token dense matvec, before activation/output traffic and correction overhead. No runtime advantage follows from that estimate. Research evaluation materializes dense reconstructions; a proposed fused deployment must decode in registers/shared memory and avoid global-memory reconstruction. No final CUDA/Triton kernel or runtime benchmark was produced.

## 16. Final verdict and scientific questions

**{verdict}**. Minimum target (mean <15%, worst <35%): {minimum}. Strong-progress target (mean <10%, worst <20%): {strongpartial}. Strong-pass target (mean <5%, worst <10%): {strong}. The final requested labels omit a separate minimum-PASS label, so a minimum-only pass is labelled PARTIAL.

1. Bit stealing versus V5: see fixed-candidate λ table; worst-layer behavior is explicitly optimized on A and measured on B.
2. Easy-layer breakpoints: attainable distortion curves at caps 1–4 BPW are saved in `distortion_curves.csv`; selected minimum layer rate is {chosen.bpw.min():.4f} BPW. No universal safe cutoff is asserted.
3. Layers deserving >2 BPW are identified in `allocations.csv` and the hard-layer table; allocation is based on calibration, not B labels.
4. Residual-based versus original-score saliency: the matched 1%-INT8 diagnostic is won by {metric_summary.sort_values('error_b').iloc[0].residual_metric}; its complete matched numbers appear in section 3. Global selections can choose a different fraction/precision.
5. At 1% INT8 columns on the 1.5-bit base, functional residual ranking recovers {100*metric_summary[metric_summary.residual_metric=='functional'].iloc[0].damage_recovered_b:.2f}% of squared held-out output damage on average. Cross-column cancellation makes score concentration differ from actual recovery.
6. Grouping: the independent activation-grouping ablation reports whether it earns its permutation cost relative to the common bridge.
7. Micro-scales: the scaling ablation charges exactly two bits per G32 plus the table and reports matched-budget gain.
8. Asymmetry: all 0/1/2-bit choices are retained; selected zero-point counts are {chosen.zero_bits.value_counts().to_dict()}.
9. Mixed residual versus single precision: adding cascade choices changes matched-global-budget mean error from {100*single['mean']:.2f}% to {100*mixedres['mean']:.2f}%. Chosen full-system precision mix is {rescounts}.
10. Largest independent mean-error gain: {winner.component}, {winner.gain:.2f} percentage points relative to its documented control at matched budget. Negative gain means regression.
11. Mean below 10% at 2 BPW: {best['mean']<.10}.
12. Worst below 20%: {best.worst<.20}.
13. New-domain mean is {100*best.ood:.2f}%; representative benchmarks remain necessary.
14. Full-model gate (mean <10% and worst <20%): {strongpartial}. {'A full-model experiment is required next.' if strongpartial else 'Perplexity and downstream benchmarks are not run because the local gate was not met.'}
15. Kernel regularity: G32 codes, macro/micro scales and column corrections are structured; arbitrary permutations and retained V5 paths still need implementation and timing before an efficiency claim.

WAVEFLY V6 FINAL RESULT

V5 baseline:
Global BPW: {baseline.bpw:.6f}
Mean held-out error: {100*baseline['mean']:.2f}%
Worst-layer error: {100*baseline.worst:.2f}%

Full V6:
Global BPW: {best.bpw:.6f}
Mean held-out error: {100*best['mean']:.2f}%
Median error: {100*best['median']:.2f}%
P90 error: {100*best.p90:.2f}%
Worst-layer error: {100*best.worst:.2f}%
New-domain error: {100*best.ood:.2f}%

Best layer BPW: {easiest.bpw:.6f}
Worst/hardest layer BPW: {hardest.bpw:.6f}

Bulk representation: calibration-selected V5/V6 static candidates
Group size: 32
Macro scale: FP8 for new V6 scalar candidates; legacy formats retained where selected
Micro-scale: {scale_counts}
Residual strategy: functional column correction or retained V5 static paths
Residual precision mix: {rescounts}
Permutation/grouping strategy: {group_counts}

Ablation winner: {winner.component}
Largest single improvement: {winner.gain:.2f} percentage points

Verdict:
{verdict}
'''
(ROOT/'WAVEFLY_V6_REPORT.md').write_text(report)
print(main.to_string(index=False)); print('REPORT_DONE',verdict)
