# Topology-aware residual specialist experiment

Status: complete. Specialist nodes learned from errors left by the bulk trellis
contain useful local alternatives, but the global checkpoint regresses and the
targeted hybrid improves combined held-out NLL by only 0.00390. The under-30
test target remains unmet.

## Method

The encoder first fits the ordinary 256-node trellis and obtains its paths. It
then measures the diagonal-Hessian energy of the residual at every emitted
weight pair. Trellis topology matters: a predecessor state's low nibble can
reach only one 16-node destination bucket. The experiment therefore:

1. Groups high-residual target pairs by their reachable destination bucket.
2. Clusters the highest-impact residual targets separately inside each bucket.
3. Replaces the least-used nodes in that same bucket with specialist values.
4. Recomputes all sampled paths and accepts a bucket replacement only when the
   sampled Hessian-weighted residual decreases.
5. Runs the normal BlockLDLQ encoding and full-metric candidate reranking.

Reserves of 16, 32, and 64 nodes were gated. Sixteen nodes, one specialist per
destination bucket, was selected for the full run because it sacrificed the
least bulk capacity. The representation and files are unchanged: 256 FP16
two-value nodes plus 64-bit paths for 32 weights, with metadata paid by omitted
paths under the global budget. There is no side residual or model finetuning.

## Gate and proxy result

The gate was weak. Sixteen specialist nodes slightly improved layer-12
attention q_proj relative Hessian error from 0.021950 to 0.021924, but worsened
layer-0 attention q_proj from 0.003594 to 0.003600 and layer-0 MLP down_proj
from 0.010319 to 0.010359.

Across all 168 full-model matrices, the specialist search accepted 425 bucket
updates. Yet the median sampled objective improvement per matrix was only
0.108%, and the maximum was 0.598%. The algorithm works as specified, but the
new representational capacity is too small and its diagonal proxy remains
misaligned with final BlockLDLQ and model loss.

## Full-model results

All results used the same RTX 4000 Ada, pinned Qwen2.5-0.5B revision, and
WikiText-2 test slices.

| Checkpoint | Selection window PPL | Test 0:4096 | Test 4096:8192 | Combined test NLL | Compressed-linear bpw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous joint best | 33.6840 | **41.0235** | 56.2289 | 3.87179 | **2.000** |
| Residual specialists globally | — | 42.4341 | 59.6228 | 3.917996 | **2.000** |
| Fresh-window targeted hybrid | **32.2634** | 41.4916 | **55.1620** | **3.86788** | **2.000** |

The targeted selector used previously unused WikiText-2 validation tokens
4096:8192, evaluated 343 gate/up/down combinations per layer, and accepted ten
layers. Residual-specialist gate matrices were chosen only in layers 1 and 21.
The earlier outlier-base matrices were chosen for layer-2 up/down. The rest
came from previous arms.

The hybrid worsens the first test slice by 0.4682 perplexity and improves the
second by 1.0669. Across all 8,160 scored test tokens, mean NLL improves from
3.87179 to 3.86788, equivalent to a 0.39% reduction in combined perplexity.
This is evidence of limited generalization, not a major quality gain.

Both the global and hybrid serialized checkpoints contain 89,456,640 bytes at
exactly 2.0 compressed bpw and 5.85984 whole-model bpw. Fresh reconstruction
from the binary files exactly reproduced every test score. Fourteen relevant
unit tests passed on the pod.

## Decision

Replacing low-use absolute nodes cannot provide a large jump because each
specialist still substitutes for a bulk value; it does not add a correction to
the chosen path. The next architecture must create genuinely additive capacity
inside the same global budget. The most defensible option is a variable-rate
two-stage codec: most blocks use a cheaper bulk path, and the saved bits fund a
small residual path only for blocks selected by nonlinear rollout error. Exact
global bit accounting, a clean selection corpus, and held-out acceptance remain
mandatory.

Raw reports are in `residual_specialist_gate_20260921.json`,
`residual_specialist_full_20260921.json`, and
`residual_specialist_joint_20260921.json`.
