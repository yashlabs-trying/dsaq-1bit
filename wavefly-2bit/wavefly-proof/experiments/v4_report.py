from pathlib import Path
import json,math
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results/v4';P=O/'plots';P.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(O/'csv/configurations.csv',low_memory=False);sal=pd.read_csv(O/'csv/saliency.csv');stab=pd.read_csv(O/'csv/stability.csv');dist=pd.read_csv(O/'csv/distribution.csv');band=pd.read_csv(O/'csv/bulk_subbands.csv');cross=pd.read_csv(O/'csv/cross_layer_overlap.csv');res=pd.read_csv(O/'csv/residual.csv');orient=pd.read_csv(O/'csv/orientation.csv')
fails=json.loads((O/'json/failures.json').read_text());core=p[p.scope=='core'].copy()
keys=['saliency_metric','salient_fraction','salient_precision','bulk_transform','group_size','bulk_bits','grouping_scheme','scale_precision','scale_share','scale_opt','quant_form']
agg=core.groupby(keys).agg(bpw=('bpw','mean'),error=('test_output_error','mean'),median=('test_output_error','median'),p90=('test_output_error',lambda x:x.quantile(.9)),worst=('test_output_error','max'),train_error=('train_output_error','mean'),weight_error=('weight_error','mean'),scale_bpw=('scale_bpw','mean'),salient_index_bpw=('salient_index_bpw','mean'),permutation_bpw=('permutation_bpw','mean'),salient_value_bpw=('salient_value_bpw','mean'),bulk_code_bpw=('bulk_code_bpw','mean')).reset_index()
budgets=[1.5,1.75,2,2.25,2.5,3,4];best=[]
for budget in budgets:
 q=agg[agg.bpw<=budget].sort_values('error')
 if len(q):z=q.iloc[0].to_dict();z['budget']=budget;best.append(z)
best=pd.DataFrame(best);best.to_csv(O/'csv/best_by_budget.csv',index=False)
z=agg.sort_values(['bpw','error']);front=[];record=float('inf')
for _,row in z.iterrows():
 if row.error<record-1e-9:front.append(row);record=row.error
front=pd.DataFrame(front);front.to_csv(O/'csv/pareto_frontier.csv',index=False)

def filt(df,row):
 q=df
 for k in keys:q=q[q[k]==row[k]]
 return q
winner=best[best.budget==2].iloc[0];per=filt(core,winner);worst_rows=per.nlargest(8,'test_output_error')[['layer','bpw','test_output_error','test_output_cosine','weight_error']]
def ltype(s):
 for q in ['q_proj','k_proj','v_proj','o_proj','out_proj','gate_proj','up_proj','down_proj','fc1','fc2','dense_h_to_4h','dense_4h_to_h']:
  if q in s:return q
 return 'attention_output' if '.attention.dense' in s else s
per=per.copy();per['type']=per.layer.map(ltype);per['stage']=per.layer.str.rsplit('@',n=1).str[-1];bytype=per.groupby(['type','stage']).test_output_error.agg(['mean','median','max']).reset_index();bytype.to_csv(O/'csv/best_by_type_stage.csv',index=False)

# Damage prevented relative to the same unprotected 1.5-bit direct bulk codec.
curve=core[(core.saliency_metric=='activation_rms')&(core.salient_precision==8)&(core.bulk_transform=='none')&(core.group_size==32)&(core.bulk_bits==1.5)].copy()
base=curve[curve.salient_fraction==0][['layer','test_output_error']].rename(columns={'test_output_error':'base_error'});curve=curve.merge(base,on='layer');curve['damage_prevented']=1-(curve.test_output_error/curve.base_error).pow(2)
frac=curve.groupby('salient_fraction').agg(error=('test_output_error','mean'),bpw=('bpw','mean'),damage_prevented=('damage_prevented','mean')).reset_index();frac.to_csv(O/'csv/salient_fraction_curve.csv',index=False)
conc=sal[sal.kind=='concentration'].groupby('fraction').value.agg(['mean','median','min','max']).reset_index();corr=sal[sal.kind=='rank_correlation'].groupby(['a','b']).value.mean().reset_index()
metric=agg[(agg.bpw<=2.5)].sort_values('error').groupby('saliency_metric').head(1)[['saliency_metric','salient_fraction','salient_precision','bulk_transform','group_size','bulk_bits','bpw','error']]
precision=agg[(agg.saliency_metric=='activation_rms')&(agg.salient_fraction==.05)&(agg.bulk_transform=='none')&(agg.group_size==32)&(agg.bulk_bits==1.5)].sort_values('error').groupby('salient_precision').head(1)[['salient_precision','bpw','error','worst']].sort_values('salient_precision')
group=agg[(agg.saliency_metric=='activation_rms')&(agg.salient_fraction==.05)&(agg.salient_precision==8)&(agg.bulk_transform=='none')&(agg.bulk_bits==1.5)][['group_size','bpw','error','scale_bpw']].sort_values('group_size')
bulk=agg[(agg.saliency_metric=='activation_rms')&(agg.salient_fraction==.05)&(agg.salient_precision==8)&(agg.bulk_transform=='none')&(agg.group_size==32)][['bulk_bits','bpw','error']].sort_values('bulk_bits')

controls=p[p.scope.isin(['saliency_only','saliency_haar_contiguous','saliency_haar_subband'])].groupby('scope').agg(bpw=('bpw','mean'),error=('test_output_error','mean'),worst=('test_output_error','max')).reset_index()
direct=agg[(agg.saliency_metric=='activation_rms')&(agg.salient_fraction==.05)&(agg.salient_precision==8)&(agg.bulk_transform=='none')&(agg.group_size==32)&(agg.bulk_bits==1.5)].sort_values('error').iloc[0]
simple2=agg[(agg.salient_fraction==0)&(agg.bulk_transform=='none')&(agg.group_size==32)&(agg.bulk_bits==2)].sort_values('error').iloc[0]
previous=pd.read_csv(R/'results/dwt/csv/best_rate_distortion.csv')
critical=[]
for b in [2,2.5,3,4]:
 q=previous[previous.target_bpw==b].iloc[0];critical.append(dict(Method='Previous Haar',BPW=q.bpw,Held_out_output_error=q.test_output_error))
critical += [dict(Method='Simple groupwise scalar 2-bit',BPW=simple2.bpw,Held_out_output_error=simple2.error),dict(Method='Saliency only',BPW=float(controls[controls.scope=='saliency_only'].bpw.iloc[0]),Held_out_output_error=float(controls[controls.scope=='saliency_only'].error.iloc[0])),dict(Method='Saliency + group quant',BPW=direct.bpw,Held_out_output_error=direct.error),dict(Method='Saliency + Haar',BPW=float(controls[controls.scope=='saliency_haar_contiguous'].bpw.iloc[0]),Held_out_output_error=float(controls[controls.scope=='saliency_haar_contiguous'].error.iloc[0])),dict(Method='Saliency + Haar + grouping',BPW=float(controls[controls.scope=='saliency_haar_subband'].bpw.iloc[0]),Held_out_output_error=float(controls[controls.scope=='saliency_haar_subband'].error.iloc[0]))]
critical=pd.DataFrame(critical);critical.to_csv(O/'csv/critical_comparison.csv',index=False)

# Discovery-only design ablations.
disc=p[p.scope=='discovery_ablation'];discagg=disc.groupby(['group_size','bulk_bits','grouping_scheme','scale_precision','scale_share','scale_opt']).agg(bpw=('bpw','mean'),error=('test_output_error','mean'),scale_bpw=('scale_bpw','mean')).reset_index()
discgroup=discagg.sort_values('error').groupby('group_size').head(1).sort_values('group_size');discscale=discagg.sort_values('error').groupby(['scale_precision','scale_share']).head(1).sort_values(['scale_precision','scale_share'])
structured=p[p.scope=='structured_saliency'][['salient_block','bpw','test_output_error']];ternary=p[p.scope=='ternary'][['bpw','test_output_error']]
stability=stab.groupby(['fraction','pair'])[['jaccard','overlap']].mean().reset_index();crosssum=cross.groupby('fraction')[['jaccard','overlap']].mean().reset_index()
distribution=dist.groupby('salient_fraction')[['variance','kurtosis','max_mean','outlier_fraction']].mean().reset_index();subbands=band.groupby(['salient_fraction','band'])[['energy_fraction','functional_importance_fraction']].mean().reset_index()
resmean=res[res.density.notna()].groupby('density').sparse_energy.mean().reset_index();resdetail=res[res.density.isna()]

def md(df):
 x=df.copy()
 for col in x.select_dtypes('number'):x[col]=x[col].map(lambda q:f'{q:.6g}')
 return '| '+' | '.join(x.columns)+' |\n| '+' | '.join(['---']*len(x.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in x.itertuples(index=False,name=None))

# Required plots.
fig,ax=plt.subplots(figsize=(9,5));ax.scatter(agg.bpw,agg.error,s=8,alpha=.12,label='All V4 configurations');ax.plot(front.bpw,front.error,'o-',label='V4 Pareto frontier');ax.plot(previous.bpw,previous.test_output_error,'s--',label='Previous Haar');ax.set(xlabel='True effective BPW',ylabel='Mean held-out output error',xlim=(0,4.2));ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/'error_vs_bpw.png',dpi=180);plt.close(fig)
for y,ylabel,file in [('error','Output error','fraction_vs_error.png'),('bpw','True BPW','fraction_vs_bpw.png'),('damage_prevented','Quantization damage prevented','fraction_vs_damage.png')]:
 fig,ax=plt.subplots(figsize=(7,4));ax.plot(frac.salient_fraction*100,frac[y],marker='o');ax.set(xlabel='Protected input columns (%)',ylabel=ylabel);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(P/file,dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(conc.fraction*100,conc['mean']*100,marker='o');ax.fill_between(conc.fraction*100,conc['min']*100,conc['max']*100,alpha=.2);ax.set(xlabel='Columns selected (%)',ylabel='Functional saliency captured (%)');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(P/'saliency_concentration.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4));axs[0].plot(group.group_size,group.error,marker='o');axs[0].set(xlabel='Group size',ylabel='Output error');axs[1].plot(group.group_size,group.scale_bpw,marker='o');axs[1].set(xlabel='Group size',ylabel='Scale metadata BPW');[a.grid(alpha=.25) for a in axs];fig.tight_layout();fig.savefig(P/'group_size_tradeoff.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));ax.plot(bulk.bulk_bits,bulk.error,marker='o');ax.set(xlabel='Average bulk code bits',ylabel='Output error');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(P/'bulk_bits_vs_error.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));ax.bar(critical.Method,critical.Held_out_output_error);ax.tick_params(axis='x',rotation=35);ax.set_ylabel('Held-out output error');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(P/'raw_vs_haar.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));ax.hist(per.test_output_error,bins=12);ax.set(xlabel='Per-layer output error: best ≤2 BPW configuration',ylabel='Layer count');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(P/'per_layer_histogram.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].plot(distribution.salient_fraction*100,distribution['kurtosis'],marker='o');axs[0].set(xlabel='Protected columns (%)',ylabel='Bulk Haar coefficient kurtosis');axs[1].plot(distribution.salient_fraction*100,distribution.max_mean,marker='o');axs[1].set(xlabel='Protected columns (%)',ylabel='Coefficient max / mean absolute');[a.grid(alpha=.25) for a in axs];fig.tight_layout();fig.savefig(P/'distribution_before_after.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));
for pair,q in stability.groupby('pair'):ax.plot(q.fraction*100,q.jaccard,marker='o',label=pair)
ax.set(xlabel='Salient fraction (%)',ylabel='Mean set Jaccard');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/'calibration_overlap.png',dpi=180);plt.close(fig)

top5_damage=float(frac[frac.salient_fraction==.05].damage_prevented.iloc[0]);top5_sal=float(conc[conc.fraction==.05]['mean'].iloc[0]);worst=per.loc[per.test_output_error.idxmax()]
report=f'''# WaveFly V4: saliency-protected columns and grouped bulk compression

## A. Previous baseline

The unchanged clustered-Haar mixed-precision baseline produced 50.87% held-out layer-output error near 2 BPW, 45.13% near 2.5 BPW, 34.50% near 3 BPW, and 13.39% near 4 BPW. V4 reuses the same 33 matrices, models, layer selection, token texts, train/held-out split, `nn.Linear` orientation, and metrics. Native `XW^T+b` verification had worst absolute difference {orient.max_abs.max():.6g}. Runtime failures: {len(fails)}.

## B. Functional saliency analysis

For column j, direct removal damage is exactly `sum(X_j^2) * ||W[:,j]||_2^2` for this linear-output metric. Therefore direct functional damage and weight×activation produce identical rankings; this is a mathematical identity, not a missing experiment. Activation RMS, weight norm, and their product were measured separately.

Mean rank correlations:

{md(corr)}

Best configuration below 2.5 BPW for each independently ranked metric:

{md(metric)}

Activation RMS wins this grid. The functional product overweights high-norm columns relative to the best observed compression behavior, while weight norm alone is worst.

## C. Saliency concentration

{md(conc)}

![Saliency concentration](results/v4/plots/saliency_concentration.png)

The top 5% of columns contain {top5_sal:.2%} of direct functional saliency. Protecting the activation-RMS top 5% prevents {top5_damage:.2%} of squared output damage from the matched 1.5-bit bulk codec. Saliency helps materially, but damage is not concentrated into only 0.5–1% of columns.

Cross-input salient-set stability:

{md(stability)}

![Calibration overlap](results/v4/plots/calibration_overlap.png)

Cross-layer reuse, restricted to the same model and input width:

{md(crosssum)}

## D. Mixed-precision salient path

Protected values use per-column FP16 scales for INT4/INT8 and no scale for FP16. Column locations use the smaller of an explicit index list and a bitmask, plus one representation flag bit.

{md(precision)}

INT8 is the minimum sensible protected precision in this grid: FP16 adds storage with negligible benefit, while INT4 loses more function. The best observed fraction is 5%, the upper boundary tested, so this experiment does not claim a global optimum beyond 5%.

Structured groups of salient columns were tested on the discovery layer:

{md(structured)}

Arbitrary columns remained better enough to retain as the main path; structured protection trades index regularity for extra or less-targeted protected values.

## E. Haar bulk path

The critical architectural controls use the same activation-RMS 5% salient set, INT8 salient values, 1.5-bit bulk target, G=32, FP8 scales shared across four groups, and MSE scale optimization:

{md(critical)}

![Raw group quantization versus Haar](results/v4/plots/raw_vs_haar.png)

Direct raw-weight group quantization beats both Haar controls. After saliency protection, Haar does not contribute enough and should be removed from the leading design.

## F. Frequency-aware grouping

The validation grid compares G=32 and G=64; discovery additionally tests G=16 and G=128, contiguous groups, subband groups, and shared importance ordering.

Validation trade-off for the winning raw bulk path:

{md(group)}

Discovery best by group size:

{md(discgroup)}

![Group-size trade-off](results/v4/plots/group_size_tradeoff.png)

For Haar bulk, subband grouping and fixed within-band importance ordering were counted explicitly. They did not beat the raw G=32 path across validation layers.

Wavelet subband energy and functional importance before/after saliency removal:

{md(subbands)}

Protecting salient columns changes coefficient statistics only modestly rather than turning the remaining wavelet details into disposable noise.

## G. Scale/metadata cost

Each group or shared group bundle stores one scale. FP16 costs 16 bits; simulated E4M3 FP8 costs 8 bits. Non-integer average bulk precision stores one bitwidth flag per base group. Permutations, saliency masks/indices, scale values, wave ordering, and headers are included.

Discovery scale-sharing ablation:

{md(discscale)}

Ternary discovery result:

{md(ternary)}

Distribution diagnostics:

{md(distribution)}

![Coefficient distribution](results/v4/plots/distribution_before_after.png)

## H. True BPW calculation

Best measured configurations at or below each requested budget:

{md(best[['budget','bpw','saliency_metric','salient_fraction','salient_precision','bulk_transform','group_size','bulk_bits','scale_precision','scale_share','error','worst']])}

For the ≤2-BPW winner, mean storage components are:

{md(pd.DataFrame([winner])[['bulk_code_bpw','scale_bpw','salient_value_bpw','salient_index_bpw','permutation_bpw','bpw']])}

![Salient fraction versus BPW](results/v4/plots/fraction_vs_bpw.png)

## I. Held-out functional accuracy

![V4 rate-distortion frontier](results/v4/plots/error_vs_bpw.png)

The best ≤2-BPW configuration improves mean held-out error from the previous 50.87% to {winner.error:.2%}. Median is {winner['median']:.2%}, p90 is {winner.p90:.2%}, and worst-layer error is {winner.worst:.2%}. This is a major relative improvement, but it remains far from single-digit error. Calibration error is {winner.train_error:.2%}, so held-out degradation is present but does not explain the main failure.

![Salient fraction versus error](results/v4/plots/fraction_vs_error.png)

![Damage prevented](results/v4/plots/fraction_vs_damage.png)

Bulk precision ablation:

{md(bulk)}

![Bulk precision](results/v4/plots/bulk_bits_vs_error.png)

Bulk weights can be encoded at 1–1.5 average bits, but functional error remains too large for useful inference.

## J. Per-layer failures

Best ≤2-BPW error by layer type and position:

{md(bytype)}

Worst individual layers:

{md(worst_rows)}

![Per-layer errors](results/v4/plots/per_layer_histogram.png)

The remaining failure is not confined to one projection or position; the p90 and worst values show catastrophic layers remain.

## K. Pareto frontier

The Pareto CSV contains every nondominated aggregate point; raw measurements preserve all {len(p):,} configurations.

{md(front[['bpw','error','saliency_metric','salient_fraction','salient_precision','bulk_transform','group_size','bulk_bits']])}

The V4 frontier moves substantially below the previous Haar curve, but plateaus well above acceptable error at 2–3 BPW.

## L. Residual structure

For the winning architecture family (activation-RMS 5%, INT8, direct 1.5-bit G=32 bulk), fraction of residual weight energy captured by largest entries:

{md(resmean)}

Discovery-layer residual rank and distribution:

{md(resdetail)}

The residual remains distributed: neither a tiny sparse correction nor a small low-rank correction captures enough energy to solve the remaining error cheaply.

## M. Kernel feasibility estimate

The winner no longer uses Haar. Its conceptual decode path is: load a 5% irregular INT8 salient-column stream and indices, dequantize direct 1.5-bit grouped bulk codes with FP8 scales shared across four groups, then feed both paths into the linear accumulation. Stored traffic is {winner.bpw/8:.4f} bytes per original weight plus runtime alignment/container effects already excluded from BPW. There is no inverse-Haar arithmetic; bulk dequantization requires unpacking, a scale multiply per weight, and group bitwidth dispatch. The irregular salient path and mixed 1/2-bit bulk remain kernel challenges, but kernel work is not justified while mean error is {winner.error:.2%}.

## N. Final verdict

V4 produces a major improvement over fixed Haar near 2 BPW, from 50.87% to {winner.error:.2%}, so it earns PARTIAL rather than FAIL. It does not approach the required single-digit functional error, its worst layer remains {winner.worst:.2%}, and Haar is absent from the winning configuration. The next representation should retain activation-aware protected columns but replace the bulk path with additive codebooks, binary factorization, or a learned dictionary/transform.

WaveFly V4:
PARTIAL

Best configuration:
Saliency method: activation RMS
Salient fraction: 5%
Salient precision: INT8
Bulk transform: none
Bulk precision: 1.5-bit average
Group size: 32
Scale precision: FP8 E4M3, shared across 4 groups
True BPW: {winner.bpw:.3f}
Held-out output error: {winner.error:.2%}
Worst-layer output error: {winner.worst:.2%}
'''
(R/'WAVEFLY_V4_REPORT.md').write_text(report)
print('V4 report complete',len(p),len(front),winner.bpw,winner.error,winner.worst)
