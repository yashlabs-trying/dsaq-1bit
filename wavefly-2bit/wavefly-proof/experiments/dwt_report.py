from pathlib import Path
import json,math
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results/dwt';P=O/'plots';P.mkdir(parents=True,exist_ok=True)
r=pd.read_csv(O/'csv/retention.csv');m=pd.read_csv(O/'csv/mixed_precision.csv');s=pd.read_csv(O/'csv/subbands.csv');c=pd.read_csv(O/'csv/correctness.csv');o=pd.read_csv(O/'csv/orientation.csv');res=pd.read_csv(O/'csv/residual.csv')
chosen=json.loads((O/'json/chosen_level.json').read_text())['level'];reuse=json.loads((O/'json/pattern_reuse.json').read_text());fails=json.loads((O/'json/failures.json').read_text())
v=r[(r.scope=='validation')];k32=v[v.k==32]

def avg(method,rule='magnitude',k=32):return v[(v.method==method)&(v.rule==rule)&(v.k==k)].mean(numeric_only=True)
raw_dct=avg('raw_dct');cl_dct=avg('clustered_dct');raw_haar=avg(f'raw_haar_L{chosen}');cl_haar=avg(f'clustered_haar_L{chosen}')
cl_func=avg(f'clustered_haar_L{chosen}','functional');raw_func=avg(f'raw_haar_L{chosen}','functional');dct_func=avg('clustered_dct','functional')
mg=m.groupby(['target_bpw','allocation','scale_mode'],as_index=False).mean(numeric_only=True)
best=mg.loc[mg.groupby('target_bpw').test_output_error.idxmin()].sort_values('target_bpw',ascending=False);best.to_csv(O/'csv/best_rate_distortion.csv',index=False)
mag2=mg[(mg.target_bpw==2)&(mg.allocation=='magnitude_fixed')].sort_values('test_output_error').iloc[0]
fun2=mg[(mg.target_bpw==2)&(mg.allocation=='functional_fixed')].sort_values('test_output_error').iloc[0]
def row(label,q,energy='energy'):
 return dict(Method=label,BPW=q.bpw,Energy_retained=q[energy],Weight_error=q.weight_error,Output_error=q.test_output_error)
primary=pd.DataFrame([row('Raw DCT',raw_dct),row('Clustered DCT',cl_dct),row('Raw Haar',raw_haar),row('Clustered Haar',cl_haar),
 row('Clustered Haar + magnitude allocation',mag2,'energy_explained'),row('Clustered Haar + functional allocation',fun2,'energy_explained')]);primary.to_csv(O/'csv/primary_table.csv',index=False)

def md(df):
 x=df.copy()
 for col in x.select_dtypes('number'):x[col]=x[col].map(lambda z:f'{z:.6g}')
 return '| '+' | '.join(x.columns)+' |\n| '+' | '.join(['---']*len(x.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,z))+' |' for z in x.itertuples(index=False,name=None))

# Common Top-K plots use magnitude, matching the prior DCT baseline.
methods=['raw_dct','clustered_dct',f'raw_haar_L{chosen}',f'clustered_haar_L{chosen}'];labels=['Raw DCT','Clustered DCT','Raw Haar','Clustered Haar']
for col,ylabel,file in [('energy','Retained coefficient energy','energy_vs_k.png'),('test_output_error','Held-out output relative error','output_vs_k.png')]:
 fig,ax=plt.subplots(figsize=(8,5))
 for method,label in zip(methods,labels):
  q=v[(v.method==method)&(v.rule=='magnitude')].groupby('k')[[col]].mean();ax.plot(q.index,q[col],marker='o',label=label)
 ax.set(xlabel='Coefficients retained per tile (K)',ylabel=ylabel);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/file,dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
for method,label in zip(methods,labels):
 q=v[(v.method==method)&(v.rule=='magnitude')].groupby('k')[['bpw','test_output_error']].mean().sort_values('bpw');ax.plot(q.bpw,q.test_output_error,marker='o',label=label)
q=best.sort_values('bpw');ax.plot(q.bpw,q.test_output_error,marker='s',linewidth=2.5,label='Importance-aware clustered Haar')
ax.set(xlabel='Effective bits per weight',ylabel='Held-out output relative error');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/'output_vs_bpw.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
for alloc,label in [('magnitude_fixed','Magnitude'),('functional_fixed','Functional'),('functional_subband','Subband functional')]:
 q=mg[mg.allocation==alloc].loc[lambda z:z.groupby('target_bpw').test_output_error.transform('min')==z.test_output_error].sort_values('bpw');ax.plot(q.bpw,q.test_output_error,marker='o',label=label)
ax.set(xlabel='Effective bits per weight',ylabel='Held-out output relative error');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/'allocation_rate_distortion.png',dpi=180);plt.close(fig)

sub=s[s.levels==chosen].groupby('band',as_index=False)[['coefficient_fraction','energy_fraction','train_removal_error','test_removal_error']].mean()
order=[f'LL{chosen}']+[f'{band}{level}' for level in range(1,chosen+1) for band in ['LH','HL','HH']];sub['order']=sub.band.map({x:i for i,x in enumerate(order)});sub=sub.sort_values('order').drop(columns='order');sub.to_csv(O/'csv/subband_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(sub));ax.bar(x-.2,sub.energy_fraction,.4,label='Energy fraction');ax.bar(x+.2,sub.test_removal_error,.4,label='Error when removed');ax.set_xticks(x,sub.band);ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/'subband_sensitivity.png',dpi=180);plt.close(fig)

z=np.load(O/'json/importance_patterns.npz');meta=json.loads((O/'json/importance_pattern_meta.json').read_text());arr=np.stack([z[f'p{i}'] for i,q in enumerate(meta) if q['method']==f'clustered_haar_L{chosen}' and q['levels']==chosen])
meanpat=arr.mean(0).reshape(16,16);np.save(O/'heatmaps/mean_functional_top32_probability.npy',meanpat)
fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(meanpat,cmap='viridis');ax.set_title('Probability coefficient is functional Top-32');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(P/'importance_probability.png',dpi=180);plt.close(fig)

gaps=mg.copy();gaps['generalization_gap']=gaps.test_output_error-gaps.train_output_error
held=best[['target_bpw','bpw','allocation','scale_mode','train_output_error','test_output_error']].copy();held['gap']=held.test_output_error-held.train_output_error
metadata=best[['target_bpw','bpw','coefficient_bpw','scale_bpw','allocation_metadata_bpw','permutation_bpw','allocation','scale_mode','nonzero_positions']]
correct=c.groupby('dtype',as_index=False)[['relative_error','max_error','orthogonality']].max()
resmean=res[res.density.notna()].groupby(['budget','density'],as_index=False).sparse_energy.mean();resdetail=res[res.density.isna()]
tile=r[(r.scope=='tile_sweep')&(r.rule=='functional')&(r.k==r.tile*r.tile/8)].groupby('tile',as_index=False)[['bpw','energy','weight_error','test_output_error']].mean()

haar_add=2*sum((16//(2**i))**2 for i in range(chosen));haar_mul=haar_add;dct_mul=2*16**3;dct_add=2*16*16*(16-1)
report=f'''# WaveFly normalized clustering + Haar DWT experiment

## A. Correctness

The implementation uses a separable orthonormal packed Haar transform and recursively transforms only the LL region. Complete coefficients reconstruct the permuted tile, and inverse row/column permutations restore the original matrix. Exact permutation identity was asserted on every layer. Worst observed transform errors:

{md(correct)}

The actual framework orientation was also checked: `nn.Linear` output was reproduced as `XW^T+b`; the worst absolute difference across captured native outputs was {o.max_abs.max():.6g} (ordinary GPU accumulation-order rounding).

## B. Raw Haar results

At B=16, K=32, and magnitude selection, level-{chosen} raw Haar retained {raw_haar.energy:.2%} energy with {raw_haar.weight_error:.2%} weight error and {raw_haar.test_output_error:.2%} held-out output error. Functional selection changed held-out error to {raw_func.test_output_error:.2%}. Results for Haar levels 1, 2, and 3 are retained separately.

## C. Clustered Haar results

The exact prior winner was held fixed: normalized row and column K-means, 16 clusters, centroid-distance ordering. At K=32 magnitude selection, clustering changed Haar held-out error from {raw_haar.test_output_error:.2%} to {cl_haar.test_output_error:.2%}. With functional selection it changed {raw_func.test_output_error:.2%} to {cl_func.test_output_error:.2%}. Therefore clustering helps magnitude-selected Haar modestly but does not improve activation-aware Haar on average.

Tile-size results at one-eighth coefficient retention on the Qwen middle Q projection:

{md(tile)}

## D. DCT vs DWT comparison

All figures below average the same 33 matrices and use unseen activation samples for output error. Top-K storage includes FP16 coefficient values, arbitrary within-tile indices, permutation tables, and a 32-byte layer header.

{md(primary)}

At K=32, clustered Haar functional selection reaches {cl_func.test_output_error:.2%} error versus {dct_func.test_output_error:.2%} for clustered DCT functional selection. Haar is better, but both remain far outside useful inference accuracy.

![Energy versus K](results/dwt/plots/energy_vs_k.png)

![Output error versus K](results/dwt/plots/output_vs_k.png)

## E. Subband sensitivity

No subband was assumed to be noise. Each row reports its coefficient and energy fractions and the complete held-out layer-output error caused by removing only that band:

{md(sub)}

![Subband sensitivity](results/dwt/plots/subband_sensitivity.png)

LL{chosen} occupies only {float(sub[sub.band==f'LL{chosen}'].coefficient_fraction.iloc[0]):.2%} of coefficients but its removal causes {float(sub[sub.band==f'LL{chosen}'].test_removal_error.iloc[0]):.2%} mean output error. High-frequency bands are also functionally important: removing HH1 alone causes {float(sub[sub.band=='HH1'].test_removal_error.iloc[0]):.2%}. Important information is distributed across LL, LH, HL, and HH.

## F. Functional coefficient importance

For coefficient `c[u,v]`, exact isolated linear-output importance is `c[u,v]^2 ||X_tile H[v]^T||^2`; row-basis norm is one. Rankings use calibration activations only. Magnitude and energy selection have identical rankings by definition; their recorded reconstructions agree. Functional ranking improves level-{chosen} clustered Haar K=32 held-out error from {cl_haar.test_output_error:.2%} to {cl_func.test_output_error:.2%}. Thus magnitude is informative but incomplete.

![Repeated importance pattern](results/dwt/plots/importance_probability.png)

Top-32 fixed-pattern overlap across layers has mean Jaccard {reuse['mean_top32_jaccard']:.4f} and median {reuse['median_top32_jaccard']:.4f} over {reuse['comparisons']} pairs. Some locations repeat, but this overlap is insufficient for one universal lossless importance layout.

## G. BPW vs output-error curves

Mixed precision uses {{0,2,4,8,16}} bits. It learns a layer-wide 3-bit precision code per wavelet position from calibration importance, so all tiles share the mask and bit layout. It compares per-tile, per-tile/subband, and one-scale-per-16-tiles grouping. Every FP16 scale, allocation code, permutation index, and header is included.

{md(best[['target_bpw','bpw','allocation','scale_mode','energy_explained','weight_error','train_output_error','test_output_error']])}

![Output error versus BPW](results/dwt/plots/output_vs_bpw.png)

![Allocation rate distortion](results/dwt/plots/allocation_rate_distortion.png)

At or below 2 BPW, the best mean held-out error is {float(best[best.target_bpw==2].test_output_error.iloc[0]):.2%}. At 2.5, 3, 4, 6, and 8 target BPW the best measured errors are {', '.join(f'{b:g}: {float(best[best.target_bpw==b].test_output_error.iloc[0]):.2%}' for b in [2.5,3,4,6,8])}. The 2–3 BPW region remains unusable.

## H. Generalization to held-out activations

The first three calibration passages select functional importance; the final two passages are held out. No held-out activations enter importance ranking or bit allocation.

{md(held)}

The small average gaps show that the failure at low BPW is not primarily calibration overfitting.

## I. Metadata cost

{md(metadata)}

Arbitrary Top-K stores an index for every retained coefficient. Mixed precision avoids per-tile masks by sharing one position-to-precision pattern across the whole layer. Global row/column permutation overhead is measured per matrix rather than reusing the earlier 0.034-BPW estimate. The mean in each selected rate point appears above. Runtime objects, container filesystem blocks, and entropy coding are excluded.

For the best 2/3/4-BPW functional reconstruction, sparse residual energy capture is:

{md(resmean)}

Discovery-layer residual structure details:

{md(resdetail)}

Sparse FP16 residual storage is not added to the claimed rate; prior experiments showed its indices quickly overwhelm the budget.

## J. Estimated inverse-transform complexity

For B=16 and L={chosen}, a dense separable inverse DCT requires approximately {dct_mul} constant multiplications and {dct_add} additions per tile. A direct orthonormal inverse Haar requires approximately {haar_add} additions/subtractions and {haar_mul} scale multiplications; powers of `1/sqrt(2)` may be combined or implemented cheaply. Both require 256 coefficient loads when dense and roughly 256-value working storage before scheduling optimizations. Sparse/mixed coefficient decoding adds loads and dequantization according to the retained representation.

Haar is mathematically cheap enough to remain hardware-interesting if accuracy succeeds. Accuracy at 2–3 BPW did not succeed, so no CUDA/Triton kernel was built.

## K. Final verdict

Normalized clustering + DWT for WaveFly:

FAIL

Measured reasons: clustered Haar improves the K=32 functional error over clustered DCT from {dct_func.test_output_error:.2%} to {cl_func.test_output_error:.2%}, but clustering itself does not improve activation-aware Haar over raw Haar, and the best complete 2-BPW mixed-precision representation retains {float(best[best.target_bpw==2].test_output_error.iloc[0]):.2%} held-out output error. At 3 BPW it still has {float(best[best.target_bpw==3].test_output_error.iloc[0]):.2%} error. High-frequency subbands demonstrably carry model function and cannot be discarded wholesale.

The measured rate-distortion curve rejects fixed Haar wavelets as the core WaveFly representation for these pretrained weights. The next experiment should move to a learned orthogonal basis/PCA, dictionary learning, or vector/codebook representation, with held-out activation-aware allocation retained as an evaluation tool.

The run covered 33 matrices from Qwen2.5-0.5B, OPT-125M, and Pythia-70M; model/runtime failures: {len(fails)}. End-to-end perplexity was not measured because the representation failed the layer-accuracy gate.
'''
(R/'DWT_WAVEFLY_REPORT.md').write_text(report)
print('DWT report complete',len(r),len(m),chosen)
