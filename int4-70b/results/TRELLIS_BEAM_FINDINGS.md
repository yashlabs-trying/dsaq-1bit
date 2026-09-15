# Correlated per-state beam experiment

Status: full-model evaluation running. No new perplexity result is claimed yet.

The reference is the same-pod combined Hessian/node-fitting checkpoint:
41.5643 perplexity on the original 4,096 tokens and 59.2522 on the next slice.

## Change

Keep two, four or eight candidate paths per trellis state. Score the full
prefix Hessian distortion, including cross-dimensional terms, before pruning.
The old diagonal Viterbi plus eight-endpoint reranking remains a fallback for
each candidate vector. This improves search; it does not change the decoder,
node-table size, transforms, scales, state topology or reset width of 32.

The search remains approximate: future correlations can favor a discarded
prefix. The fallback guarantees non-worsening distortion against the old
search for the same input vector and metric, not non-worsening final model
perplexity. Block-LDLQ changes later inputs when earlier decisions change.

## Small gate

Twelve probes, each with 16 rows, from Q and down projections in layers 0 and
12. Inputs are realistic coordinate-refinement candidates from the reference
checkpoint. These are 192 sampled vectors, not a whole-model evaluation.

| Paths per state | Mean reduction in weighted candidate error |
| --- | ---: |
| 2 | 11.83% |
| 4 | 12.19% |
| 8 | 12.42% |

Two paths capture about 95% of the eight-path gain in this gate and were
selected for the full-model experiment. These percentages are not predicted
perplexity gains. Small-batch timings have warm-up/order effects and are not
inference speed benchmarks. Raw probes are in
[trellis_beam_gate_20260915.json](trellis_beam_gate_20260915.json).

Prefix Hessian projections are cached and incrementally updated to avoid
repeated decoding of every surviving prefix. A 128-vector correlated test
matched the uncached implementation's paths exactly. A separate 2,048-vector
batch took 0.245 seconds versus 0.322 seconds uncached on the pod; this is a
single development measurement, not a general speed claim.

## Controls and reproduction

Qwen2.5-0.5B base revision `060db6499f32faf8b98477b0a26969ef7d8b9987`;
RTX 2000 Ada 16 GB; calibration 8,192 WikiText-2 train tokens with 256-token
windows. Original model parameters stay frozen. Only quantized representations
are selected or fitted. Test slices and the existing global MLP metadata
allocation are unchanged.

`trellis_beam_experiment.py` requires the combined reference checkpoint and
refreshed Hessian cache from `trellis_ablation.py` on the pod. Set
`FLASHRUN_BEAM=2`, `FLASHRUN_NODE_ROUNDS=0` and `FLASHRUN_OUTPUT` to the output
directory. Then run `trellis_eval.py` with `FLASHRUN_CHECKPOINT` pointing there
to decode into a fresh model. `FLASHRUN_RESUME=1` can reuse completed matrices
after a compatible interrupted run.

Optional `FLASHRUN_NODE_ROUNDS` alternates activation-aware node fitting with
the stronger path search, accepting a round only when its whole-projection
Hessian loss does not increase. This option is not yet a measured model result.

Twenty-five relevant unit tests passed, including diagonal-search optimality,
correlated-search fallback and packed decoding. The evaluator reconstructs
dense BF16 weights. There is no fused compressed inference engine yet.
