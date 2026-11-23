# Joint nonlinear MLP checkpoint selection

Status: complete. The controlled search improves both held-out slices at the
same exact two-bit compressed-linear payload, but remains far above 30.

## Motivation

The BF16 restoration map showed that MLP damage dominates and that attention
and MLP errors sometimes compensate. This experiment tests whether selecting
`gate`, `up`, and `down` representations jointly by the true post-SiLU MLP
output is better than accepting an entire independently optimized checkpoint.

## Method

For each of Qwen2.5-0.5B's 24 transformer layers:

1. Run the partially selected quantized model on the first 4,096 WikiText-2
   validation tokens and sample 256 actual inputs to that layer's MLP.
2. Decode gate/up/down matrices from four existing exact-two-bit checkpoints:
   current trellis, refreshed-Hessian, output-node-fit, and both.
3. Evaluate all 64 gate/up/down combinations against the BF16 teacher's true
   `down(SiLU(gate(x)) * up(x))` output for the same quantized-rollout inputs.
4. Propose the lowest-MSE triple, then run the complete model on validation.
   Accept only if complete validation NLL decreases.
5. Serialize the chosen source file for each matrix without changing its bytes.

Pass two adds the pass-one hybrid as a fifth source (125 triples per layer).
Pass three similarly starts from pass two. All original model weights remain
frozen; no gradient update or model finetuning occurs.

## Results

| Checkpoint | Accepted layer changes | Validation PPL | Test 0:4096 | Test 4096:8192 | Serialized compressed-linear bpw |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current trellis | — | 32.8625 | 41.5400 | 58.0628 | **2.00** |
| Joint pass 1 | 13 | 30.1540 | 41.2192 | 56.3744 | **2.00** |
| Joint pass 2 | 1 | **30.1376** | **41.0235** | **56.2289** | **2.00** |
| Joint pass 3 | 0 | 30.1376 | 41.0235 | 56.2289 | **2.00** |

Relative to the same-pod source, pass two improves perplexity by 1.24% on the
first test slice and 3.16% on the next. It lowers NLL by 0.01251 and 0.03209.
The payload remains 89,456,640 bytes including paths, node tables and headers;
BF16 embeddings/tied head keep whole-model storage at 5.85984 bpw.

Every pass reproduced exactly when decoded from its serialized files into a
fresh original model. Passes took 29.80, 30.33 and 32.22 seconds respectively,
excluding checkpoint generation and fresh-decoder verification.

## What was selected

Pass one accepted layers 2, 4, 5, 6, 8, 9, 15, 16, 17, 18, 19, 22 and 23.
Most accepted triples mix Hessian and combined-arm matrices rather than taking
all three projections from one checkpoint. Only two selected matrices use the
standalone node-fit arm. Pass two changes only layer 2, from
`(hessian, both, node)` to `(both, both, node)`. Pass three accepts nothing,
showing saturation within this finite candidate pool.

## Proxy mismatch remains

In pass one, the median post-SiLU teacher-MSE reduction is 6.13% for accepted
layers and 6.61% for rejected layers. A lower local nonlinear MLP error still
does not reliably imply lower full-model NLL. The full validation acceptance
gate is essential. Selection-chain invariants require every layer's starting
NLL to equal the preceding accepted state and prohibit accepted NLL increases.

The held-out improvement shows that joint MLP selection is useful. Its size is
small relative to the under-30 gap, and the method only recombines codes already
created by known quantizers. It is evidence for the direction, not yet a novel
complete quantizer or a record result.

## Decision

The next mechanism must generate new coordinated gate/up/down codes rather than
choose among four existing matrices. Candidate generation should optimize the
nonlinear MLP residual under quantized-rollout inputs, while complete-model NLL
continues to control acceptance. A separate final dataset is required before
tuning further on this validation slice.

Raw per-layer proposals, MSE values, NLL transitions, selected source files,
storage accounting and fresh evaluations are in
[`joint_mlp_selection_20260921.json`](joint_mlp_selection_20260921.json).
