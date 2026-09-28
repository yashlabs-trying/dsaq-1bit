# Experiment ledger

This ledger separates the scientific claim from the implementation status. A negative result is evidence about the tested hypothesis, not evidence that all possible 1-bit methods fail.

## Evidence status

| Status | Meaning |
|---|---|
| Reproducible here | Code, original specification, tables, and generated report are committed |
| Code here / result pending import | Implementation is committed; the completed result is recorded but its result directory is not in this checkout |
| Research record only | Result was logged during the program; implementation and raw artifacts must be imported before independent reproduction |
| Proposed | No result is claimed |

## Experiment 1 — Domain-Structured Approximation (DSA)

**Intent.** Test whether a cheap structure score identifies blocks where a compact magnitude surface improves over one sign bit plus one scale.

**Innovation.** Fit row, column, plane, and quadratic magnitude surfaces; train a held-out ridge router using only pre-reconstruction statistics; test with and without a fixed Hadamard transform.

**Outcome.** Hadamard + DSA reduced weighted error from 0.441844 to 0.440346 relative to Hadamard naive, at 1.00604 versus 1.00098 true BPW. The router obtained Pearson r=0.9422, Spearman r=0.2529, and balanced accuracy 0.9719 on 1,404 held-out blocks. It failed the preregistered rule because Spearman did not exceed 0.30.

**Learning.** A few high-gain blocks drove the linear correlation; broader rank ordering was unreliable. Hadamard helped, while the tested DSA surfaces added little.

**Status.** Reproducible here: [driver](../run_experiments.py), [report](../results/REPORT.md).

## Experiment 2 — Seeded residual search

**Intent.** Rescue the exact held-out irregular blocks where Experiment 1's structured fit was weak, while storing only a candidate index.

**Innovation.** Generate deterministic streams from a global seed, matrix, block, and candidate ID. Compare K={2,4,8,16} across uniform, Gaussian, robust, and global distributions; charge candidate indexes and non-shared parameters.

**Outcome.** Uniform K=16 with weighted selection was best on aggregate weighted error at 1.002261 BPW, but beat DSA on only 0.08% of held-out irregular blocks. In the representative-layer intervention, it produced PPL 57,009.6, far from BF16 17.8982.

**Learning.** Search cannot compensate for a candidate family that does not contain good reconstructions.

**Status.** Reproducible here: [driver](../run_experiments.py), [report](../results/REPORT.md).

## Experiment 3 — Low-rank residual reconstruction

**Intent.** Determine whether R=W-Q is low-rank and whether Hessian-weighted directions are more valuable than ordinary SVD directions.

**Innovation.** Add FP16 low-rank factors in one or two stages, match storage across ordinary and diagonal-Hessian corrections, and report recovered perplexity per metadata bit.

**Outcome.** Rank 8 captured 14.46% of mean residual energy. Rank-8 plus Hessian rank-4 reached PPL 1,072.61 at 1.296122 BPW, recovering 83.85% of the Hadamard baseline's excess perplexity. Hessian stage two beat ordinary stage two in every matched pair.

**Learning.** Low-rank correction and Hessian direction selection are useful, but the residual is not low-rank enough for BF16-like quality near one BPW.

**Status.** Reproducible here: [driver](../run_experiment3.py), [report](../results/experiment3/REPORT.md).

## Experiment 4 — Sensitivity-directed binary, low-rank, and sparse rescue

**Intent.** Test whether second-order weighting and explicit preservation of a small salient subset can concentrate representation power where it matters.

**Innovation.** Combine a Hessian-aware binary base, damage-routed low-rank correction, sparse BF16 residual restoration, and exact accounting for indexes, values, factors, and binary metadata.

**Outcome.** The best compressed configuration reached PPL 489.231 at 3.100977 BPW. The top 1% held 49.97% of diagonal-proxy damage, but restoring Hessian-selected weights did not translate proportionally to perplexity. Magnitude restoration sometimes preserved function better.

**Learning.** Concentration in a local error proxy does not imply concentration of end-to-end function.

**Status.** Reproducible here: [driver](../run_experiment4.py), [report](../results/experiment4/REPORT.md).

## Experiment 5 — Sequential Functional Binary Search

**Intent.** Select candidates from activations actually produced by preceding quantized layers instead of optimizing weights independently.

**Innovation.** Quantize all 168 transformer projections sequentially; compare clean-teacher and current-input objectives; ablate K, look-ahead horizon, and beam width with negligible layer-level candidate metadata.

**Outcome.** The best configuration reached PPL 96,086.5 at 1.00097676 BPW versus 644,467 for Hadamard naive. Independent K=8 reached 746,760; sequential current-target K=8 reached 286,008. Beam search did not help.

**Learning.** Functional selection and sequential state matter, but the binary candidate family remains the limiting factor.

**Status.** Reproducible here: [driver](../run_experiment5.py), [report](../results/experiment5/REPORT.md).

## Experiment 6 — Exact-weight importance oracle

**Intent.** Ignore encoding cost temporarily and ask whether a small subset of original BF16 values repairs the fixed destructive 1-bit base.

**Innovation.** Compare magnitude, activation-aware, gradient, and functional-Fisher selectors under global and layer-balanced budgets. Exact restoration isolates selection quality from decoder capacity.

**Outcome.** Functional Fisher was strongest: PPL 649.493 at 10% protected versus a Hadamard baseline near 637,559. No method reached PPL <100 at or below 10%. The set was enriched in V/K/O projections, but selector overlaps were low.

**Learning.** A small independent scalar subset was not found. Non-monotonic recovery suggested interacting and canceling errors.

**Status.** Reproducible here: [driver](../run_experiment6.py), [report](../results/experiment6/REPORT.md).

## Experiment 7 — Conditional quantization-error importance

**Intent.** Score damage in the already-quantized model, recompute after each repair, and test whether local causal gains compose.

**Innovation.** Compare quantization-error Fisher, blockwise Q-OBS, static ICQR, recomputed ICQR, and collapse-prior search. Direct interventions validate gains; pair tests measure non-additivity.

**Outcome.** Predicted and measured local gains correlated at r=0.723; 96.4% of tested interventions were locally positive. Pair interactions ranged from -41.01 to +33.76 with median -4.286. Iterative ICQR reached PPL 25,740 at 0.250043% protected, but later batches failed joint KL/NLL acceptance; no method reached PPL <100.

**Learning.** Local repair direction is predictable, yet beneficial corrections do not reliably compose into global recovery. Scalar importance is the wrong atomic unit for this base.

**Status.** Reproducible here: [driver](../run_experiment7.py), [report](../results/experiment7/REPORT.md).

## Experiment 8 — Joint Protection Elimination

**Intent.** Begin with exact reconstruction and remove protected residual information until crossing PPL <=19.688.

**Innovation.** Convert importance into constrained elimination, avoiding the unstable task of building from a catastrophically broken model.

**Outcome.** The best logged mask still protected 94.227% of the residual at PPL about 19.687; only about 5.77% could be removed safely.

**Learning.** Information required by the tested 1-bit base is dense in scalar weight coordinates. Sparse exact restoration cannot plausibly meet a near-1-bit target.

**Status.** Research record only. Code, prompt, and raw outputs are absent from this checkout; treat the number as logged until those artifacts are imported.

## Experiment 9 — Hierarchical Adaptive Group Rescue

**Intent.** Test whether Experiment 8's apparent density was caused by scalar search granularity.

**Innovation.** Build from 5%, 10%, and 15% Fisher cores; search layers, projections, heads/channels, and blocks; recompute rankings; backtrack; and refine accepted 128x128 regions down to 16x16 near the boundary.

**Outcome.** The completed run reached PPL 19.683539 while protecting 93.444921% of the residual. This improved Experiment 8 by about 0.78 percentage points but remained far from the sparse regime.

**Learning.** Hierarchical grouping marginally improves the mask but does not reveal a compact structured subset. The exact-residual rescue route is effectively closed for this base.

**Status.** Code and specification are here: [driver](../run_experiment9.py), [block refiner](../refine_experiment9_blocks.py), [prompt](../research_prompts/experiment9.md). The completed pod result bundle was not imported before transfer stopped.

## Experiment 10 — Activation-nullspace binary folding

**Intent.** Change the objective from W≈Q to XW≈XQ, allowing error in directions unused by calibration activations.

**Innovation.** Encode 32x32 blocks as signs plus low-cost row/column magnitude metadata and optimize in activation space, with a hard early-layer gate.

**Outcome.** The best logged Layer-0 configuration produced PPL about 1,071. No configuration passed Layer 0, so the run stopped rather than spending compute on deeper layers.

**Learning.** Function-space optimization is promising, but a separable row/column magnitude decoder is too close to a rank-1 magnitude model.

**Status.** Research record only. Code and raw outputs are absent from this checkout.

## Experiment 11 — Functional sigma-delta binary folding

**Intent.** Let neighboring stored bits jointly represent richer effective magnitudes through shared filters, functional error feedback, activation-correlated traversal, importance-aware routing, and beam search.

**Innovation.** Break the assumption that one stored bit independently reconstructs one weight while retaining amortized storage near one BPW.

**Status.** Proposed. No result is claimed. The first decision gate should cover only Layers 0–4 and terminate if capacity remains inadequate.

## Cumulative conclusions

1. Hadamard preprocessing is useful, but local reconstruction improvements do not close the functional gap.
2. True storage accounting matters; methods attractive by MSE can become unattractive after factor, index, or exact-value costs.
3. Local functional importance exists but is conditional and non-additive.
4. More than 93% exact residual protection was required at the strict quality boundary in the best logged search.
5. The remaining problem is decoder expressiveness: one physical bit must participate in a richer shared representation of the layer function.
