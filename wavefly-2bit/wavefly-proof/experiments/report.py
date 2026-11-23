from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'results';P=O/'plots'
a=pd.read_csv(O/'csv/measurements.csv',low_memory=False);rt=pd.read_csv(O/'csv/roundtrip.csv');f=pd.read_csv(O/'csv/full_layers.csv')
real=a[(a.source=='real')&(a.tile==16)];pure=real[(real.residual_requested==0)&(real.bits==16)&(real.method!='scalar')]
def table(df):
    df=df.copy()
    for c in df.select_dtypes('number'):df[c]=df[c].map(lambda x:f'{x:.6g}')
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in df.itertuples(index=False,name=None))
def typ(s):
    for k in ['q_proj','k_proj','v_proj','o_proj','out_proj','gate_proj','up_proj','down_proj','fc1','fc2','dense_h_to_4h','dense_4h_to_h','query_key_value','lm_head','embedding']:
        if k in s:return k
    return 'attention_output' if '.attention.dense' in s else s
a['type']=a.layer.map(typ);real=a[(a.source=='real')&(a.tile==16)]
fig,axs=plt.subplots(1,3,figsize=(15,4))
for (m,method),g in pure.groupby(['model','method']):
    q=g.groupby('k')[['energy','weight_error','output_error']].mean()
    for ax,col in zip(axs,q.columns):ax.plot(q.index,q[col],label=m.split('/')[-1]+' '+method,linestyle='--' if method=='fixed' else '-');ax.set(xlabel='Coefficients K',ylabel=col);ax.grid(alpha=.2)
axs[0].legend(fontsize=6);fig.tight_layout();fig.savefig(P/'real_compression.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,5))
syn=a[(a.source=='synthetic')&(a.tile==16)&(a.method=='topk')&(a.bits==16)&(a.residual_requested==0)]
for name in ['gaussian','smooth','rank1','sparse0.95','outlier0.01','sinusoidal','mixed']:
    q=syn[syn.layer==name].sort_values('k');ax.plot(q.k,q.weight_error,label=name)
ax.set(xlabel='Coefficients K',ylabel='Relative weight error');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(P/'synthetic.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(15,4))
for ax,(model,g) in zip(axs,f.groupby('model')):
    for method,gg in g.groupby('method'):
        z=gg.groupby(['k','bits','residual_count'])[['bpw','output_error']].mean();ax.scatter(z.bpw,z.output_error,label=method)
    ax.set(title=model.split('/')[-1],xlabel='Actual BPW',ylabel='Full-layer output relative error');ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(P/'full_layer_storage_quality.png',dpi=180);plt.close(fig)
heats=list((O/'json').glob('*.npy'));fig,axs=plt.subplots(int(np.ceil(len(heats)/6)),6,figsize=(18,3*int(np.ceil(len(heats)/6))))
for ax,path in zip(axs.flat,heats):
    z=np.load(path);im=ax.imshow(z/z.mean(),vmin=0,vmax=3,cmap='magma');ax.set_title(path.stem,fontsize=5);ax.set_xticks([]);ax.set_yticks([])
for ax in list(axs.flat)[len(heats):]:ax.axis('off')
fig.suptitle('Mean absolute DCT coefficients / tile-wide mean (0–3 color range)',fontsize=12);fig.tight_layout(rect=(0,0,1,.98));fig.savefig(P/'all_layer_heatmaps.png',dpi=150);plt.close(fig)
summary=real[(real.method=='topk')&(real.k==32)&(real.bits==16)&(real.residual_requested==0)].groupby(['model','type','stage'])[['energy','bpw','weight_error','output_error']].mean().reset_index();summary.to_csv(O/'csv/real_layer_summary.csv',index=False)
full=f.groupby(['model','method','k','bits','residual_count'])[['bpw','weight_error','output_error']].mean().reset_index();full.to_csv(O/'csv/full_layer_summary.csv',index=False)
tile=a[(a.source=='real')&(a.method=='topk')&(a.bits==16)&(a.residual_requested==0)&(a.k==a.tile*a.tile/8)].groupby(['model','tile'])[['bpw','energy','weight_error','output_error','idct_flops_per_tile']].mean().reset_index()
synsum=syn[syn.k==32][['layer','energy','bpw','weight_error','output_error']]
env=json.loads((O/'json/environment.json').read_text());fail=json.loads((O/'json/failures.json').read_text());checks=json.loads((O/'json/serialization.json').read_text())
req=pd.DataFrame(json.loads((O/'json/real_requantization.json').read_text()))
low=real[(real.method!='scalar')&(real.bpw<=2.1)].sort_values('output_error').groupby(['model','layer','stage'],as_index=False).first()
low.to_csv(O/'csv/best_under_2_1_bpw.csv',index=False)
report=f'''# WaveFly experiments 1–2: numerical proof report

## Verdict

**Storage and reconstruction work; useful general-purpose low-bit DCT compression is not supported by these measurements.** Full-coefficient DCT is numerically reversible. Quantized coefficients and residuals can be physically packed, read, reconstructed, and requantized. These mechanics do not establish acceptable model quality.

- **Gate 1 — Representation: PASS** for the full-coefficient numerical round trip; compact reconstruction is strongly data-dependent.
- **Gate 2 — Universality: FAIL** for the tested fixed-DCT hypothesis across these three small pretrained transformer families. This is a conclusion about this basis, tile layout, and tested codec, not a proof that every wave-based or learned representation fails.
- **Gate 3 — Hardware: NOT TESTED.** The user requested the first two experiments. Assigning PASS/PARTIAL/FAIL without a hardware experiment would misstate the evidence.

## Scope and reproducibility

GPU: {env['gpu']}, {env['gpu_bytes']/2**30:.2f} GiB; PyTorch {env['torch']}; CUDA runtime {env['cuda']}; Triton {env['triton']}. Seed {env['seed']}; TF32 disabled. CPU details and checkpoint revision hashes are in results/json. The run produced **{len(a):,} configuration measurements**, **{len(rt):,} round-trip measurements**, and **{len(f):,} full-layer checks**.

Three pretrained models: Qwen/Qwen2.5-0.5B, facebook/opt-125m, EleutherAI/pythia-70m. Early/middle/late layers, attention projections, MLP projections, and embeddings were sampled. Architectures without a gated MLP have no gate projection. GPT-NeoX fused QKV was separated using its head-interleaved layout. Qwen and OPT LM heads were included; Pythia's separately named output head was not sampled. Each matrix/size uses 128 random aligned tiles sampled with replacement. Synthetic families each use 128 tiles per size. Sizes: 8,16,32,64.

Calibration uses five fixed short English passages covering prose, science, arithmetic, and code descriptions. These are a small illustrative calibration set, not a representative benchmark corpus. Tile output metrics measure local partial contributions using actual module input activations; **full-layer measurements additionally reconstruct the whole matrix offline and measure XW^T error**, excluding bias. Embedding tile output uses identity input and is a reconstruction metric, not a language-model activation metric. No perplexity, downstream accuracy, or acceptable-quality threshold was established.

## 1. Numerical representation

Explicit orthonormal DCT-II: C = D W D^T; reconstruction W_hat = D^T C D. Bases are formed in float64 and cast to the tested dtype. Error metrics accumulate in float64. Round trips cover every synthetic family plus real tiles at B=8,16,32. Worst observed errors:

{table(rt.groupby('dtype')[['orthogonality','max_abs','weight_error','output_error']].max().reset_index())}

This is numerical equivalence within the observed errors, not bitwise equality to the original weights. Float16 coefficients and arithmetic introduce additional rounding. Basis construction in float64 avoids conflating low-precision cosine construction with transform multiplication error.

## 2. Compression curves

The B=16 sweep uses K=256,192,128,96,64,48,32,24,16,8,4. Fixed selection is an exact-area low-frequency rectangle, choosing the most square factorization and anchoring at DC. Top-K selects the largest absolute coefficients and pays index costs. Energy is averaged per tile; weight/output relative errors use pooled squared norms.

![Real curves](results/plots/real_compression.png)

Synthetic results at K=32, FP16 coefficient storage, top-K, no residual:

{table(synsum)}

![Synthetic curves](results/plots/synthetic.png)

Smooth and deliberately DCT-aligned signals compress well. Low rank by itself does not guarantee a sparse DCT spectrum. Random and spatially sparse matrices are important negative controls.

## 3. Full-layer output and storage comparison

The following summaries average layer-relative errors within each model. Full per-layer rows are in results/csv/full_layers.csv, avoiding reliance on one aggregate. Integer scalar baselines use symmetric mid-rise levels with one FP16 scale per 256-weight tile (four levels for 2-bit). They are simple baselines, not GPTQ/AWQ or optimized learned quantizers. FP16 baseline casts weights to FP16 then evaluates in FP32. All coefficient tests use FP32 transform math unless the round-trip dtype test says otherwise.

{table(full)}

![Storage versus error](results/plots/full_layer_storage_quality.png)

Best **tile-level** configurations under 2.1 BPW are saved per model/layer in best_under_2_1_bpw.csv. This is an in-sample sweep selection, not held-out validation. Model-level range of those layer errors:

{table(low.groupby('model').output_error.agg(['min','median','max']).reset_index())}

The hybrid K=32 INT4 codec with three residuals does beat this simple 2-bit scalar baseline on mean full-layer output error in all three models, while using about 1.844 rather than 2.063 BPW. This is a positive relative result, but its roughly 51–58% output error is still large; no end-to-end quality claim follows. INT4 scalar uses more storage (about 4.063 BPW) and has much lower mean output error (about 8.5–13%).

## 4. Residuals, quantization, and actual serialization

B=16 tests FP16/INT8/INT4/2-bit spectral values, magnitude and activation-weighted residual selection on real matrices, and requested densities 0,0.1,0.25,0.5,1,2,5%. Residual slots use round(density*B^2): at B=16 these are **0,0,1,1,3,5,13 slots**, so requested sub-tile fractions are not all distinct. Actual density is recorded. Residual values are FP16. Importance is |R[out,in]| times mean absolute input activation for that input coordinate. This is a heuristic, not an optimal correction.

Every serialized tensor has a 32-byte header. Each tile stores a byte-aligned packed coefficient stream, top-K indices if needed (ceil(log2(B^2)) bits/index), FP16 residual values, packed residual indices, and a two-byte scale for integer coefficients. Fixed K and residual count are tensor-wide and recorded in the header. Fixed masks are derived from geometry and consume no per-tile index stream. The constant DCT basis is shared implementation data rather than weight payload; cached basis RAM and inference traffic are not measured here. No entropy coding, file-system blocks, runtime tensor-object overhead, or model-container overhead is included. Reported BPW is this concrete tile-tensor codec's payload, not end-to-end model file size.

At B=16, K=32, INT4, top-K and 3 residuals: coefficients=16 bytes/tile, indices=32, residuals=6, residual indices=3, scale=2 => 59 bytes/tile, or **1.84375 BPW plus the tensor header**. Fixed K=32 INT4 with no residual is 18 bytes/tile (0.5625 BPW plus header), but dropping more information produces high error. FP16 K=32 top-K costs 3 BPW before header.

All {len(checks)} sampled serialization tests reproduced the corresponding simulated reconstructions exactly; a separate test passed 32 combinations across four tile sizes, two selection modes, and four coefficient precisions, including packed residual indices. This verifies the implementation and byte accounting, not model accuracy.

Real pretrained tiles were also written, read, and successively requantized through FP16 → INT8 → INT4 → 2-bit. These are single-tile files, so their 32-byte header adds 1 BPW; numbers should not be compared directly to large-tensor amortized BPW. Requantization does not restore information already removed by top-K selection:

{table(req[['model','bits','bytes','bpw','weight_error','output_error']])}

## 5. Tile-size sweep and coefficient structure

Other tile sizes sweep retained fractions 1,3/4,1/2,1/4,1/8,1/16, FP16 coefficients, and residual requests 0,1,5%. The full coefficient-precision/residual Cartesian sweep is B=16 only. At one-eighth retention, top-K, FP16 and zero residual:

{table(tile)}

Dense separable IDCT costs approximately 4B^3 floating-point operations per tile (two multiplies, counting multiply-add as two). Direct sparse synthesis has different arithmetic and was not timed. These operation counts are not hardware performance estimates.

![All layer heatmaps](results/plots/all_layer_heatmaps.png)

Each heatmap is mean absolute coefficient magnitude over the sampled tiles, normalized by its own mean. Colors share a 0–3 range. The raw heatmaps are saved as NPY arrays. Predictable low-frequency concentration is insufficient here to make the tested compact fixed masks competitive across the tested layers.

## 6. Per-type and layer-position breakdown

K=32, top-K, FP16 spectral storage, no residual. Full raw grids and per-layer baseline/hybrid comparisons are in measurements.csv and full_layers.csv.

{table(summary)}

## Limitations and next decision

No experiment 3 kernel, on-chip reconstruction, timing, throughput, or Tensor Core claim is made. Offline reconstruction intentionally materializes weights for numerical comparisons; it is not the proposed inference implementation. Three small models do not establish behavior of all larger LLMs. Original checkpoint precision limits the source information; loading in FP32 does not increase that information. Tile sampling, short calibration, basic scalar quantizers, and fixed per-tile residual budgets limit generality. Real outlier-heavy layers were not separately classified; synthetic outlier densities were covered.

The initial model-download attempt failed because the pod enabled hf_transfer without installing it. Rerunning with HF_HUB_ENABLE_HF_TRANSFER=0 resolved it. Final model-load failures: {len(fail)}. Logs are retained.

**Decision:** do not proceed on the assumption that this fixed-DCT codec gives generally useful approximately 2-BPW LLM weights. Investigate adaptive or learned bases, channel permutations, or a fallback scalar representation. Define a held-out model-quality target before spending effort on a fused kernel. Successful storage/requantization is proven as an implementation operation; quality-preserving aggressive compression is not.
'''
(R/'FINAL_REPORT.md').write_text(report)
print('Report and plots complete',len(a),len(f),flush=True)
