# Refreshed Hessians x activation-aware node fitting

Status: complete. Both changes together modestly improved the measured scores;
neither change helped alone. **Under 30 was not reached.**

## Results

Lower perplexity is better. All rows use the same RTX 2000 Ada pod, original
model revision, calibration tokens, transforms, scales, code topology,
metadata allocation, final search and compensation/refinement settings.

| Arm | Original 4,096 tokens | Next 4,096 tokens | Actual serialized compressed-weight bpw |
| --- | ---: | ---: | ---: |
| BF16 reference | 22.1937 | 27.5206 | 16 |
| Current trellis | 42.5888 | 60.1728 | **2.00** |
| Refreshed Hessian only | 49.2619 | 64.7817 | **2.00** |
| Layer-output node fitting only | 43.3756 | 62.9393 | **2.00** |
| Both | **41.5643** | **59.2522** | **2.00** |

The combined arm lowered perplexity by about 2.41% on the first slice and
1.53% on the next slice versus the same-pod current baseline. This is a modest
single-configuration result, not broad quality proof or an under-30 success.

All four binary checkpoints reproduce both perplexity scores exactly when
decoded into fresh models. Each checkpoint's measured transformer-linear
payload is 89,456,640 bytes. Learned tables, scale/seed/shape headers and path
streams are included. BF16 embeddings/tied head make whole-model storage
5.85984 bits per parameter. The experiment still reconstructs dense weights
for evaluation; it does not demonstrate compressed inference speed.

The baseline run took 143.86 seconds including initial loading/dataset setup.
Refreshing input Hessians took 21.12 seconds. The three modified arm runs took
72.98, 77.88 and 78.45 seconds respectively. These are development durations,
not inference benchmarks. Quantization arms ran sequentially on one GPU.

Complete per-projection reports, solver diagnostics and fresh-checkpoint
evaluations are in [trellis_2x2_20260915.json](trellis_2x2_20260915.json).

## What the test establishes

Output-aware node fitting reduced its frozen-path objective in each projection,
but that reduction did not guarantee better final model perplexity. In the
original-Hessian fitting arm, median fixed-path error reduction was only about
0.185%. After path reselection/compensation, 147 of 168 projections had a better
local Hessian proxy, with a median improvement of about 0.223%, yet perplexity
worsened on both slices. This local proxy is not the language-model loss.

This bounded fixed-path solver found only a small extra gain; the experiment
does not establish that node fitting is the main remaining quality bottleneck.
It also does not prove convergence to the exact fixed-path optimum for every
projection. Refreshed sensitivity and fitted node values
worked better together in this test, but their interaction needs validation
across larger evaluation sets and quantization runs.

In the combined arm, median frozen-path fitting error reduction was about
0.438%, versus 0.185% with original-input Hessians. These are per-projection
surrogate improvements, not reductions in model loss or a universal guarantee.

The next focused hypothesis is joint path/node optimization with acceptance
against larger block-output behavior. Richer correlated-error path search may
also help. Neither hypothesis proves that under 30 is feasible at this format's
two-bit budget, and this test does not prove a universal quantization floor.

Twenty-three tests passed, including a comparison with an independent dense
least-squares oracle, signed uint64 path mapping, neutral-node preservation,
frozen original weights, correlated-error reduction and exact serialization.
Syntax checks passed. Code and raw reports are committed to GitHub; binary
checkpoint files remain under `/workspace/ablation_*` on the pod.

## Controlled design

| Arm | Input Hessian | Node-value learning |
| --- | --- | --- |
| Current | Original BF16-model inputs | Existing Euclidean path-table fitting |
| Hessian only | Inputs from the fixed current quantized model | Same current stored node values |
| Layer-output only | Original BF16-model inputs | Output-aware fitting of current stored nodes |
| Both | Inputs from the fixed current quantized model | Output-aware fitting of current stored nodes |

All arms use Qwen2.5-0.5B base at revision
`060db6499f32faf8b98477b0a26969ef7d8b9987`, the same 8,192 WikiText-2 training
calibration tokens in 256-token windows, two-sided DCT/Hadamard signs and
stored FP16 matrix scales, 256 states, 32-weight resets, eight endpoint
candidates, Block-LDLQ, one discrete path-refinement sweep, and global MLP
metadata payment. Total serialized transformer-linear storage must equal
2.00 bits per weight, including learned tables and headers. Embeddings remain
BF16. No original model parameters are finetuned.

The provided pod has a **16 GB NVIDIA RTX 2000 Ada**, not a 3090. The current
trellis is rerun there to establish a same-pod reference. Historical results
are 43.9234/59.2349 for current trellis and 39.3299/55.9547 for E8P on the two
4,096-token evaluation slices. Hardware-dependent numerical differences are
not assumed to be zero.

## What each change means

Refreshed Hessians are collected once from the complete fixed baseline
quantized model. Both relevant arms use that exact same input second-moment
estimate, avoiding a different calibration model as a third factor. This
estimates new input sensitivity after quantization; it is not a Hessian of
trellis coordinates or an exact Hessian of final language-model loss.

Layer-output fitting minimizes
`tr((W - decoded(paths, nodes)) H (W - decoded(paths, nodes))^T)` for each
linear projection. This is its squared output error on the same calibration
inputs, with the usual relative Hessian damping. It does not directly fit
whole-transformer-block outputs or model logits.

The fit uses the same baseline paths in both fitting arms. A matrix-free
preconditioned conjugate-gradient solver performs at most 12 iterations to
fit the shared 256x2 node table under full input correlations. Node zero stays
fixed to protect the implicit-zero representation. FP16-rounded candidates
are accepted only if they do not worsen fixed-path output loss. All arms then
use the same final search and compensation procedure; the fitting arms add
table fitting, not a different path-search budget.

The target is under 30 perplexity, but this experiment can regress. The
original evaluation slice has been repeatedly used for method selection.
The next slice provides an additional comparison, not broad quality proof.
No compressed inference-speed result is claimed.

## Reproduce

Use the pinned dependencies in `requirements-lattice.txt` with CUDA PyTorch.
No external QuIP# checkout is needed for these full trellis runs.

```bash
python -m unittest test_trellis_node_fit test_trellis_quant test_additive_vq test_lattice_experiment test_calibration
FLASHRUN_TRELLIS_WIDTH=32 FLASHRUN_TRELLIS_BUDGET=global_mlp \
FLASHRUN_TRELLIS_FULL=1 FLASHRUN_TRELLIS_VARIANT=path_trained_rerank \
FLASHRUN_OUTPUT=/workspace/ablation_current python -u trellis_experiment.py
python -u trellis_ablation.py
for arm in current hessian_only layer_output_only both; do
  FLASHRUN_CHECKPOINT=/workspace/ablation_$arm python -u trellis_eval.py
done
```

`trellis_node_fit.py` implements the table solver. `trellis_ablation.py` runs
the three modified arms from the baseline. Per-arm reports and layer files
are written under `/workspace/ablation_*`; the aggregate report is
`/workspace/ablation_report.json`. The original BF16 weights remain frozen in
CPU memory throughout all three arms.
