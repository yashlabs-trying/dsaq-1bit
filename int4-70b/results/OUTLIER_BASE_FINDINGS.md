# Importance-aware outlier base experiment

Status: complete. Reserving part of the existing trellis base table for
Hessian-important transformed weight pairs helps selected matrices, but a
global application regresses and a validation-selected hybrid gives only a
small, non-generalizing improvement. The under-30 test target remains unmet.

## Architecture change

The serialized format is unchanged. Each matrix still stores a 256-by-2 FP16
node table and one 64-bit path per 32 weights. The table metadata is paid for
by omitted paths under the existing global budget.

The new encoder:

1. Applies the existing randomized two-sided transform and normalization.
2. Scores sampled weight pairs by diagonal-Hessian energy.
3. Learns most nodes from the bulk distribution and reserves 32 nodes for the
   highest-impact 5% tail.
4. Fits paths and nodes with per-position diagonal-Hessian weighted least
   squares, then uses BlockLDLQ and full-metric path reranking.
5. Stores the resulting nodes and paths in the same exact-two-bit files.

There is no sparse side channel, BF16 outlier list, model-weight update, or
unreported payload. Original BF16 weights remain frozen.

## Controlled gate

The gate compared 32, 64, and 96 reserved nodes with the current path-trained
trellis on three matrices. Thirty-two nodes gave the best tradeoff. It reduced
the local relative Hessian error from 0.003594 to 0.003565 on layer-0 attention
q_proj and from 0.010318 to 0.010172 on layer-0 MLP down_proj. It worsened
layer-12 attention q_proj from 0.021935 to 0.022178. Larger reserves did not
resolve that conflict and sometimes increased ordinary MSE.

## Full model and targeted hybrid

All values below were measured on the same RTX 4000 Ada with the same pinned
Qwen2.5-0.5B revision and WikiText-2 slices.

| Checkpoint | Validation PPL | Test 0:4096 | Test 4096:8192 | Compressed-linear bpw |
| --- | ---: | ---: | ---: | ---: |
| Current trellis | 32.8625 | 41.5400 | 58.0628 | **2.000** |
| Previous joint best | 30.1376 | 41.0235 | **56.2289** | **2.000** |
| Outlier base on every matrix | — | 42.6724 | 59.1566 | **2.000** |
| Joint selection with outlier candidate | **30.0032** | **40.6668** | 56.3341 | **2.000** |

The targeted search started from the previous joint checkpoint and evaluated
216 gate/up/down combinations per MLP layer. It accepted three layers. Only
layer 2 used the new representation, choosing `(both, outlier, outlier)` for
gate/up/down. Layers 7 and 10 selected combinations of older arms.

The targeted hybrid improves the first test slice by 0.3566 perplexity but
regresses the second by 0.1052. The 89,456,640-byte payload, 2.0 compressed
bpw, and 5.85984 whole-model bpw are unchanged. Fresh reconstruction from only
the serialized files exactly reproduced both the full outlier and hybrid
scores. Thirteen relevant unit tests passed on the pod.

## Finding

The idea is valid at matrix level, but the present importance definition is
too local. A large transformed weight times a large Hessian diagonal is not
necessarily the residual that matters after the bulk trellis path, and the
same 256 nodes must satisfy mutually constrained trellis transitions. Global
tail reservation therefore steals common-pattern capacity from matrices that
do not benefit.

The next representation-level experiment should learn specialist nodes from
the **post-trellis residual**, weighted by measured nonlinear MLP-output error
on quantized-rollout inputs. This would target errors the base path actually
fails to represent, instead of predicting outliers from weight energy before
encoding. Selection must use a new validation corpus because the current
WikiText validation slice has now controlled several sequential decisions.

Raw gate, full-model, and joint-selection reports are in
`outlier_gate_20260921.json`, `outlier_full_20260921.json`, and
`outlier_joint_20260921.json`.
