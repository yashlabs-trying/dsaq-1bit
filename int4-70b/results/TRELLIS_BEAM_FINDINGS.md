# Correlated per-state beam experiment

Status: both full-model trials completed. Neither beat the previous combined
checkpoint. The requested 12-point improvement and under-30 target were not
achieved.

The reference is the same-pod combined Hessian/node-fitting checkpoint:
41.5643 perplexity on the original 4,096 tokens and 59.2522 on the next slice.

## Full-model result

| Arm | Original 4,096 tokens | Next 4,096 tokens | Compressed-weight bpw |
| --- | ---: | ---: | ---: |
| BF16 | 22.1937 | 27.5206 | 16 |
| Previous combined checkpoint | **41.5643** | **59.2522** | **2.00** |
| Correlated two-path beam | 42.3777 | 60.0480 | **2.00** |
| Beam + fixed-path node refit | 42.2641 | 59.8358 | **2.00** |

The stronger search worsened perplexity by about 1.96% and 1.34%. Its local
Hessian proxy improved in 167/168 projections with a median reduction of 2.60%.
This is evidence of a mismatch between the local calibration objective and
model quality, not evidence that larger search can never help. A wider beam
was tested only in the small gate, not in full models.

Fresh decoding into an original model reproduced both scores exactly.
The measured compressed payload is 89,456,640 bytes including tables/headers.
Embeddings/tied head remain BF16, giving whole-model storage of 5.85984 bpw.
Completed matrix encoding took 2,629.09 seconds (43.82 minutes); interrupted
work and test/optimization overhead are additional. This expensive eager
Python search is a quality experiment, not an inference implementation.

The additional node-refit arm kept every packed path fixed and fitted only
the shared node values against the same cached Hessians. It took 18.51 seconds
including loading/evaluation (10.62 seconds of matrix processing). It modestly
helped versus the beam checkpoint, but still lost to the previous combined
checkpoint on both slices. Both new checkpoints reproduce exactly in fresh
models and contain the same measured compressed payload. Full per-projection
reports, fresh-decoder checks and the test log are in
[trellis_beam_20260916.json](trellis_beam_20260916.json).

The previous combined checkpoint remains the best same-GPU trellis result.
The evidence favors testing objectives that include nonlinear block behavior
and accumulated input-distribution changes, rather than assuming lower local
Hessian error must improve perplexity. Such a test has not been implemented
or measured here; no 12-point gain is promised.

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

Discarding already-used projected dimensions further reduced a subsequent
batch measurement from 0.266 to 0.217 seconds and matched all 128 reference
paths in another comparison. Completed matrices were reused across compatible
resumes. These are implementation optimizations of the same search objective.

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

For the measured fixed-path refit arm, set `FLASHRUN_REFERENCE` to the finished
beam checkpoint, `FLASHRUN_REFIT_ONLY=1`, `FLASHRUN_NODE_ROUNDS=1` and a new
`FLASHRUN_OUTPUT` directory. Original model weights are still frozen. Run the
same fresh evaluator afterward. Binary checkpoints remain on the pod under
`/workspace/beam2_rounds0` and `/workspace/beam2_fixed_node_fit`; they are not
included in GitHub. The reference checkpoint remains `/workspace/ablation_both`.

## Node-fit diagnostic

After stronger search, fitting node values against the fixed paths reduced
full-projection Hessian distortion by 0.213% (layer-0 Q), 0.767% (K), 0.191%
(O) and 0.175% (down). This bounded 12-iteration diagnostic suggests a small
remaining gain from node fitting in these projections, not a large perplexity
jump or proof of a global optimum. The main checkpoint was unchanged. Raw
results are in [trellis_beam_node_gate_20260915.json](trellis_beam_node_gate_20260915.json).

Optional `FLASHRUN_NODE_ROUNDS` alternates activation-aware node fitting with
the stronger path search, accepting a round only when its whole-projection
Hessian loss does not increase. Alternating re-search rounds are not a measured
model result here; the measured additional arm is fixed-path node fitting.

Twenty-five relevant unit tests passed, including diagonal-search optimality,
correlated-search fallback and packed decoding. The evaluator reconstructs
dense BF16 weights. There is no fused compressed inference engine yet.
