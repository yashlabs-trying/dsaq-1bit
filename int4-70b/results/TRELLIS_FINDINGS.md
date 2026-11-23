# Controlled learned-trellis experiment

Status: completed. The proposed pipeline works, but its best tested variant
does not beat the current E8P adaptation.

## Full-model comparison

Lower perplexity is better. Both slices score 4,080 predictions in independent
256-token windows. The original slice has been used repeatedly for method
selection; neither short slice establishes general model quality.

| Method | Original 4,096 tokens | Next 4,096 tokens | Actual compressed-weight bpw |
| --- | ---: | ---: | ---: |
| BF16 | 22.1719 | 27.5071 | 16 |
| Current E8P, same-pod reproduction | **39.3299** | **55.9547** | **2.00000000** |
| Basic K-means + 8-weight trellis + weighted search + Block-LDLQ | 193.0078 | 434.0582 | 2.00000000 |
| Path fitting + neutral zero + 32-weight paths + full-metric reranking | 61.1652 | 83.5933 | 1.99997746 |
| Same 32-weight codec + global MLP metadata payment | **43.9234** | **59.2349** | **2.00000000** |

All three trellis model runs and the E8P reproduction were independently
reconstructed from their binary files on fresh models. Every reported score
reproduced exactly. The final trellis checkpoint stores 89,456,640 bytes.
Original model weights are never trained; only quantizer tables and indices
are fitted. Embeddings/tied LM head remain BF16, yielding 5.85984 bpw over the
whole model. Dense reconstruction is still used for inference evaluation;
there is no fused compressed runtime or inference-speed result.

The basic learned-node, sensitivity-weighted trellis **regressed** to 193.0078
perplexity on the original slice and 434.0582 on the next slice. Our current
E8P adaptation scored 39.3299 and 55.9547 respectively in the preceding run.
BF16 reproduced exactly at 22.1719 and 27.5071 on this new pod. The basic
trellis's serialized transformer-linear payload is exactly 2.00 bpw.

This independent prototype tests the proposed combination of two-sided
DCT/Hadamard mixing, activation Hessians, K-means patterns, a bit-shift trellis,
and Block-LDLQ compensation. It is not an exact QTIP reproduction or a claim
that these established ingredients are novel.

## Fixed controls

- Qwen2.5-0.5B base, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- RTX 3090, one GPU; no original model finetuning.
- 8,192 WikiText-2 training calibration tokens, 256-token windows.
- Same two-sided transforms and 0.01 relative Hessian damping as the current E8P adaptation.
- One discrete refinement sweep after Block-LDLQ.
- Fixed and learned trellises use identical serialized storage: exactly 2 bits per compressed weight.

## Prototype and limits

The trellis has 256 states. Each four-bit symbol advances the state and selects
a two-dimensional node value. Four symbols represent an eight-weight block in
a uint16 path. The start state is fixed at zero for every block. This reset
limits search cost and long-range modeling; it is not QTIP's long-chain codec.

Each matrix stores 256 two-dimensional FP16 node values (1,024 bytes) and an
18-byte shape/seed/scale header. The final 521 predetermined blocks have
implicit path zero, paying exactly for the 1,042 bytes of metadata. These paths
are constrained during optimization. Fixed tables are stored too, so fixed and
learned node comparisons have identical byte and position budgets.

K-means fits transformed, normalized weight pairs. Node placement is fixed;
initial clustering does not optimize the trellis topology. The optional
sensitivity-aware search weights additive pair errors by the diagonal of the
conditional Block-LDLQ metric. Full correlations enter error compensation and
the acceptance test for refinement; Viterbi is not an exact full-Hessian solver.

The first gate compares attention and MLP matrices with fixed nodes, learned
nodes, learned nodes with weighted search, and the current E8P method. Its
Hessian error is a calibration proxy, not model perplexity or a speed benchmark.

## Initial measurements

| Matrix | E8P relative Hessian error | Fixed trellis | K-means trellis | K-means + weighted search |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 attention Q | 0.001848 | 0.004584 | 0.004200 | 0.004213 |
| Layer 0 MLP down | 0.010349 | 0.016009 | 0.016216 | 0.016237 |
| Layer 12 attention Q | 0.019060 | 0.032584 | 0.032500 | 0.032466 |

The first gate took 39.81 seconds; the basic full-model run took 125.65 seconds.
These are development encoding/evaluation durations, not compressed inference
speed results. See [trellis_gate_initial.json](trellis_gate_initial.json).

Path-aware node fitting alternates discrete Viterbi assignment with centroid
least-squares updates. It improves all three matrix proxies relative to the
basic learned trellis. This is quantizer fitting, not model finetuning. A
second ablation pins node zero to zero, so implicit paths reconstruct a neutral
vector. The fitting plus neutral-path combination reduces matrix Hessian error
by about 10–29% relative to basic learned nodes, depending on the matrix.

The stronger 32-weight variant adds full-metric reranking of eight endpoint
winners from the diagonal-cost Viterbi search. These are not necessarily the
eight best paths overall; full-Hessian search remains approximate. It reached
**61.1652** on the original slice and **83.5933** on the next slice at
**1.99997746 serialized bpw**. Fresh-model binary reconstruction reproduced
both scores exactly. The same-pod E8P reproduction remains **39.3299 / 55.9547**.

Raw reports for every completed gate and model run, including fresh evaluations,
are in [trellis_20260915.json](trellis_20260915.json).

A 16-weight reset variant is also supported through
`FLASHRUN_TRELLIS_WIDTH=16`. It stores uint32 paths; 261 omitted paths pay for
1,042 metadata bytes, leaving two additional bytes of savings per matrix.
Its total payload is slightly below 2.00 bpw. The reset width is an algorithm
format setting required when decoding these files; it is recorded in reports.

The 32-weight format uses uint64 paths and 131 omitted paths per matrix. This
overpays metadata by six bytes per matrix. Its forced-zero fraction is about
3.65% in small 128x896 K/V matrices, versus about 0.19% over the entire compressed
weight set. This disproportionate payment is a concrete weakness of allocating
metadata independently per matrix.

`FLASHRUN_TRELLIS_BUDGET=global_mlp` instead pays the full model's learned-table
metadata using only predetermined MLP paths. Attention has no omitted paths;
individual matrix rates can exceed two bits, while the actual total payload
remains capped at two bits per compressed weight. The omitted count is inferred
from each file's length and header, so no additional allocation table is needed.

This controlled allocation change reduced original-slice perplexity from
61.1652 to 43.9234 and next-slice perplexity from 83.5933 to 59.2349. It changes
only the predetermined omitted positions/counts; node fitting, path search,
calibration and refinement settings stay the same. The 168 node tables and
headers total 175,056 bytes. Omitting 21,882 uint64 MLP paths pays that cost
exactly: 66 MLP matrices omit 304 paths each, and six omit 303. Attention omits
none. This is a global two-bit budget, not a two-bit guarantee per matrix.

The final global run took 97.53 seconds including loading/calibration-cache
read, encoding and evaluation. The stronger per-matrix run took 96.95 seconds;
the E8P reproduction took 114.19 seconds. These development durations are not
inference throughput comparisons. No experiment shared the GPU with another
quantization run; a brief CPU correctness suite ran during the global test.

## What to try next

1. **Fit node values under an activation-aware objective.** Current K-means and
   path-table fitting minimize Euclidean weight error; the Hessian enters search,
   compensation and refinement. Analytic shared-node updates under conditional
   Hessian metrics could directly target important errors without model finetuning.
2. **Preserve more paths before full-metric selection.** Eight endpoint winners
   discard alternative paths ending at the same state. A bounded beam per state
   could retain paths with better correlated-error costs, at greater encoding cost.
3. **Reduce table overhead without zeroing weights.** Sharing tables across
   compatible matrix groups or implementing an actual entropy-coded path stream
   could pay metadata without implicit-zero constraints. Both require measured
   bytes and a new quality comparison; an ideal entropy estimate is insufficient.

Longer continuous trellises are another option, but reset-state representation
and Block-LDLQ integration need a codec redesign beyond the current uint64 block.
None of these proposals guarantees lower perplexity or research novelty.

Checkpoint files remain on the pod under `/workspace/trellis_full_global`;
GitHub contains code and complete raw reports, not the binary model payloads.

## Reproduce

```bash
python -m unittest test_trellis_quant test_additive_vq test_lattice_experiment test_calibration
FLASHRUN_OUTPUT=/workspace/trellis_gate python -u trellis_experiment.py
FLASHRUN_TRELLIS_FULL=1 FLASHRUN_TRELLIS_VARIANT=learned_weighted \
FLASHRUN_OUTPUT=/workspace/trellis_full python -u trellis_experiment.py
FLASHRUN_TRELLIS_WIDTH=32 FLASHRUN_TRELLIS_BUDGET=global_mlp \
FLASHRUN_TRELLIS_FULL=1 FLASHRUN_TRELLIS_VARIANT=path_trained_rerank \
FLASHRUN_OUTPUT=/workspace/trellis_full_global python -u trellis_experiment.py
FLASHRUN_CHECKPOINT=/workspace/trellis_full_global python -u trellis_eval.py
```

Use the dependencies and external pinned QuIP# checkout documented in
[3090_LATTICE_FINDINGS.md](3090_LATTICE_FINDINGS.md). The E8P source is external,
GPL-3.0 licensed, and used only for the comparison gate. The trellis code is
an independent prototype of established bit-shift trellis/Viterbi ideas:
[QTIP](https://arxiv.org/abs/2406.11235),
[QuIP#](https://arxiv.org/abs/2402.04396).

Twenty correctness tests passed on the pod, including exhaustive comparison
of Viterbi with all 65,536 paths and exact two-bit serialization round trips.
