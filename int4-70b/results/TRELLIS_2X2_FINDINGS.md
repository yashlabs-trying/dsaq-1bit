# Refreshed Hessians x activation-aware node fitting

Status: GPU experiment in progress. No improvement or under-30 result claimed.

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
