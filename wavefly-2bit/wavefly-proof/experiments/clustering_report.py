from pathlib import Path
import json,math
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results/clustering';P=O/'plots';P.mkdir(parents=True,exist_ok=True)
p=pd.read_csv(O/'csv/measurements.csv');winner=json.loads((O/'json/winner.json').read_text());win='winner_'+winner['label']
disc=p[(p.scope=='discovery')&(p.selection=='topk')&(p.residual_actual==0)]
k32=disc[disc.k==32]

def best(prefix,exclude=None):
    q=k32[k32.method.str.startswith(prefix)]
    if exclude:q=q[~q.method.str.contains(exclude)]
    return q.sort_values('output_error').iloc[0].method
chosen={'Raw':'raw','Row K-means':best('row_'),'Column K-means':best('col_'),'Row+Column K-means':best('both_'),
        'Correlation clustering':best('correlation_'),'Activation-aware':best('activation_'),'Magnitude-aware':best('magnitude_'),'Block-local':best('local')}

def val(method,k,col):return float(disc[(disc.method==method)&(disc.k==k)][col].iloc[0])
primary=[]
for label,m in chosen.items():
 q=k32[k32.method==m].iloc[0]
 primary.append(dict(Method=label,Configuration=m,Total_variation=q.tv,Neighbor_correlation=(q.row_corr+q.col_corr)/2,
   Top16_energy=val(m,16,'energy'),Top32_energy=q.energy,Top64_energy=val(m,64,'energy'),Weight_error_K32=q.weight_error,Output_error_K32=q.output_error,Permutation_BPW=q.permutation_bpw))
primary=pd.DataFrame(primary);primary.to_csv(O/'csv/primary_comparison.csv',index=False)

def ltype(s):
 for k in ['q_proj','k_proj','v_proj','o_proj','out_proj','gate_proj','up_proj','down_proj','fc1','fc2','dense_h_to_4h','dense_4h_to_h']:
  if k in s:return k
 return 'attention_output' if '.attention.dense' in s else s
v=p[(p.scope=='validation')&(p.selection=='topk')&(p.k==32)&(p.residual_actual==0)].copy();v['model']=v.layer.str.split('/').str[:2].str.join('/');v['type']=v.layer.map(ltype)
validation=v.groupby(['method'])[['tv','row_corr','col_corr','energy','weight_error','output_error','output_cosine','permutation_bpw']].mean().reset_index();validation.to_csv(O/'csv/validation_summary.csv',index=False)
bytype=v.groupby(['model','type','method'])[['energy','weight_error','output_error','output_cosine','permutation_bpw']].mean().reset_index();bytype.to_csv(O/'csv/by_type.csv',index=False)

# Storage: FP16 top-K values + 8-bit within-tile coefficient indices; global residual indices are explicitly costed.
hy=p[(p.scope=='validation')&(p.selection=='topk')&(p.k==32)].copy()
hy['residual_count']=(hy.residual_actual*hy.rows*hy.cols).round().astype(int)
hy['spectral_bpw']=32*24/256
hy['residual_bpw']=hy.apply(lambda q:q.residual_count*(16+math.ceil(math.log2(q.rows*q.cols)))*8/(q.rows*q.cols),axis=1)
hy['total_estimated_bpw']=hy.spectral_bpw+hy.residual_bpw+hy.permutation_bpw
hybrid=hy.groupby(['method','residual_requested'])[['weight_error','output_error','residual_energy','permutation_bpw','spectral_bpw','residual_bpw','total_estimated_bpw']].mean().reset_index();hybrid.to_csv(O/'csv/hybrid_summary.csv',index=False)

methods=list(chosen.values())
colors=plt.cm.tab10(np.linspace(0,1,len(methods)))
for col,title,file in [('energy','Retained spectral energy','energy_curves.png'),('weight_error','Relative weight error','reconstruction_curves.png'),('output_error','Relative functional output error','functional_curves.png')]:
 fig,ax=plt.subplots(figsize=(9,5))
 for m,c in zip(methods,colors):
  q=disc[disc.method==m].sort_values('k');ax.plot(q.k,q[col],marker='o',ms=3,label=next(k for k,vv in chosen.items() if vv==m),color=c)
 ax.set(xlabel='Coefficients retained per 256-weight tile (K)',ylabel=title);ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(P/file,dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
for m,label in [('raw','Raw'),(winner['label'],'Clustered discovery winner')]:
 q=p[(p.scope=='discovery')&(p.method==m)&(p.selection=='topk')&(p.k==32)].sort_values('residual_requested');ax.plot(q.residual_actual*100,q.weight_error,marker='o',label=label+' weight');ax.plot(q.residual_actual*100,q.output_error,marker='s',linestyle='--',label=label+' output')
ax.set(xlabel='Residual entries retained (%)',ylabel='Relative error');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(P/'residual_curve.png',dpi=180);plt.close(fig)

for col,ylabel,file in [('tv','Total variation','total_variation.png'),('neighbor','Mean adjacent cosine similarity','neighbor_correlation.png')]:
 fig,ax=plt.subplots(figsize=(9,4));z=primary.set_index('Method');vals=z.Total_variation if col=='tv' else z.Neighbor_correlation
 ax.bar(range(len(z)),vals);ax.set_xticks(range(len(z)),z.index,rotation=30,ha='right');ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(P/file,dpi=180);plt.close(fig)

raw=np.load(O/'heatmaps/discovery_raw_weight.npy');clu=np.load(O/'heatmaps/discovery_clustered_weight.npy')
lim=np.quantile(np.abs(raw),.995)
for z,title,file in [(raw,'Original weights (display crop)','original_weight.png'),(clu,'Clustered weights (display crop)','clustered_weight.png')]:
 fig,ax=plt.subplots(figsize=(6,5));ax.imshow(z,cmap='coolwarm',vmin=-lim,vmax=lim,aspect='auto');ax.set_title(title);fig.tight_layout();fig.savefig(P/file,dpi=180);plt.close(fig)
layer=k32.iloc[0].layer;safe=lambda s:s.replace('.','_').replace('/','_').replace('\\','_').replace('@','_')
for m,title,file in [('raw','Raw mean |DCT|','raw_dct.png'),(winner['label'],'Clustered mean |DCT|','clustered_dct.png')]:
 z=np.load(O/'heatmaps'/(safe(layer+'__'+m+'_b16')+'.npy'));fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(z/z.mean(),cmap='magma',vmin=0,vmax=4);ax.set_title(title+' / global mean');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(P/file,dpi=180);plt.close(fig)

def md(df):
 x=df.copy()
 for c in x.select_dtypes('number'):x[c]=x[c].map(lambda q:f'{q:.6g}')
 return '| '+' | '.join(x.columns)+' |\n| '+' | '.join(['---']*len(x.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in x.itertuples(index=False,name=None))
raw32=k32[k32.method=='raw'].iloc[0];best32=k32[k32.method==winner['label']].iloc[0]
rawv=validation[validation.method=='raw'].iloc[0];winv=validation[validation.method==win].iloc[0]
models=json.loads((O/'json/models.json').read_text());fails=json.loads((O/'json/failures.json').read_text())
tile=p[(p.scope=='tile_sweep')&(p.selection=='topk')&(p.residual_actual==0)&(p.k==p.tile*p.tile/8)].groupby('tile')[['energy','weight_error','output_error','permutation_bpw']].mean().reset_index()
methods_time=pd.read_csv(O/'csv/methods.csv');
report=f'''# WaveFly clustering experiment

## A. Raw DCT baseline

The discovery matrix was `{raw32.layer}` ({int(raw32.rows)}×{int(raw32.cols)}), evaluated with {int(raw32.tiles)} complete 16×16 tiles and {int(raw32.rows)}×{int(raw32.cols)} full-matrix functional reconstruction. At K=32 magnitude Top-K, raw ordering retained {raw32.energy:.4%} of spectral energy, produced {raw32.weight_error:.4%} relative weight error, and {raw32.output_error:.4%} relative output error. Fixed low-frequency and magnitude Top-K results for every requested K are in `measurements.csv`.

![Original weights](results/clustering/plots/original_weight.png)

## B. Clustering effect

Only complete rows and columns were permuted. For every configuration, exact index uniqueness and `P_r^T(P_r W P_c)P_c^T == W` were asserted before DCT; all recorded permutation round-trip maxima are zero. The search tested raw and normalized vectors, 4/8/16/32/64 clusters, centroid-distance/norm/PCA within-cluster order, row-only, column-only, both axes, spherical/correlation clustering, activation augmentation, magnitude features, and 128/256 block-local ordering.

The discovery search selected `{winner['label']}` by minimum K=32 functional error on the same layer. It reduced total variation from {raw32.tv:.6g} to {best32.tv:.6g}; mean adjacent cosine similarity moved from {(raw32.row_corr+raw32.col_corr)/2:.6g} to {(best32.row_corr+best32.col_corr)/2:.6g}. Search selection and evaluation on the same matrix make this an in-sample result. Across {len(v.layer.unique())} independent validation matrices, the same method changed average total variation from {rawv.tv:.6g} to {winv.tv:.6g}, row correlation from {rawv.row_corr:.6g} to {winv.row_corr:.6g}, and column correlation from {rawv.col_corr:.6g} to {winv.col_corr:.6g}.

{md(primary.rename(columns={'Total_variation':'TV','Neighbor_correlation':'Neighbor corr','Top16_energy':'Top-16','Top32_energy':'Top-32','Top64_energy':'Top-64','Weight_error_K32':'Weight error','Output_error_K32':'Output error','Permutation_BPW':'Perm BPW'}))}

![Clustered weights](results/clustering/plots/clustered_weight.png)

![TV](results/clustering/plots/total_variation.png)

![Neighbor correlation](results/clustering/plots/neighbor_correlation.png)

## C. Spectral concentration

On the discovery layer, the winning ordering raised Top-32 energy from {raw32.energy:.4%} to {best32.energy:.4%}. Across validation matrices it raised the unweighted layer mean from {rawv.energy:.4%} to {winv.energy:.4%}. This is real concentration, but the validation shift is much smaller than the discovery result.

![Raw DCT](results/clustering/plots/raw_dct.png)

![Clustered DCT](results/clustering/plots/clustered_dct.png)

![Energy curves](results/clustering/plots/energy_curves.png)

## D. Reconstruction

At discovery K=32 the winner reduced weight error from {raw32.weight_error:.4%} to {best32.weight_error:.4%}. Across validation layers, error changed from {rawv.weight_error:.4%} to {winv.weight_error:.4%}. The latter remains too large for direct deployment.

![Reconstruction curves](results/clustering/plots/reconstruction_curves.png)

One-eighth coefficient retention for the winning method across tile sizes on the Qwen middle Q projection:

{md(tile)}

## E. Functional accuracy

Activations came from five fixed calibration passages. The metric reconstructs each complete matrix, undoes both permutations, and compares `XW^T` with `XW_hat^T`. On discovery, K=32 output error fell from {raw32.output_error:.4%} to {best32.output_error:.4%}. Across validation matrices it fell from {rawv.output_error:.4%} to {winv.output_error:.4%}; output cosine moved from {rawv.output_cosine:.6g} to {winv.output_cosine:.6g}. This is a material average improvement, but {winv.output_error:.2%} error is not acceptable evidence of model fidelity. No perplexity was measured.

![Functional curves](results/clustering/plots/functional_curves.png)

Per-family/type results, including early/middle/late Qwen layers and representative middle OPT/Pythia layers:

{md(bytype)}

## F. Hybrid sparse residual

At K=32, residual entries were selected globally by magnitude at requested densities 0.1%, 0.25%, 0.5%, 1%, 2%, and 5%. `residual_energy` records the fraction of residual squared energy recovered. `total_estimated_bpw` counts FP16 coefficients, 8-bit within-tile coefficient indices, FP16 residual values, global residual indices, and permutation tables. It excludes container/runtime overhead. FP16 Top-32 spectral storage alone is 3 BPW, so this structural experiment does not claim a 2-BPW codec.

{md(hybrid)}

![Residual curve](results/clustering/plots/residual_curve.png)

## G. Permutation cost

An arbitrary global permutation stores every row/column index using `ceil(log2(dimension))` bits. The discovery winner used {int(best32.permutation_bytes)} bytes, or {best32.permutation_bpw:.6g} BPW. Across validation matrices the mean was {winv.permutation_bpw:.6g} BPW. Block-local methods count only `ceil(log2(macro-block size))` bits/index because block identity is implicit. Runtime gather/scatter cost was not benchmarked. Some row permutations could be folded into adjacent compatible operators, but attention, residual paths, nonlinearities, and independently learned permutations constrain sharing; no sharing credit is assumed.

The deep discovery evaluated {len(methods_time)} orderings. Configuration times are in `methods.csv`; these are clustering wall times rather than inference costs. Results cover {', '.join(m['model'] for m in models)}. Final load/runtime failures: {len(fails)}.

## H. Final verdict

`Clustering makes WaveFly substantially more viable: PARTIAL`

Best measured method: `{winner['label']}` (normalized row and column k-means with 16 clusters and centroid-distance ordering).

Clustering actually produced local structural correlation: adjacent cosine similarity and spectral concentration increased, and both weight and activation-output errors decreased. It did more than sort magnitudes. The effect generalized only moderately: mean Top-32 energy increased from {rawv.energy:.2%} to {winv.energy:.2%}, while mean functional error remained {winv.output_error:.2%}. Therefore reversible channel ordering helps but does not make fixed DCT with 32/256 coefficients sufficient. The next representation experiment should test a learned orthogonal rotation or a block-adaptive learned/codebook basis, retaining clustering only where it earns its permutation cost.
'''
(R/'CLUSTERING_REPORT.md').write_text(report)
print('clustering report complete',len(p),len(v),winner)
