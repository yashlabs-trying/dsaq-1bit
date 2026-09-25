You are helping build and experimentally validate a new ultra-low-bit LLM quantization framework.

The main research target is an approximately 1-bit weight representation while preserving as much model quality as possible. The objective is NOT simply to minimize raw weight MSE. The real objective is to preserve the important transformations performed by the original model while keeping total effective bits-per-weight, including metadata, extremely small.

The central hypothesis is:

Different weight regions have different structures and therefore should not all be reconstructed using the same technique.

Some weight blocks contain predictable structure.
Some blocks appear irregular or random.
Some weights are much more sensitive to reconstruction error than others.
Some rare salient weights may require special protection.

The framework therefore combines several complementary techniques.

## 1. Hadamard transformation

Purpose:

Reduce outlier concentration and make weight distributions more uniform before ultra-low-bit representation.

We want to test whether applying a Hadamard transform makes later DSA or residual reconstruction easier.

Do not assume Hadamard always helps. Measure before and after.

Track:

- weight distribution
- kurtosis
- max/mean magnitude
- outlier ratio
- reconstruction error
- downstream perplexity

## 2. Hessian / sensitivity guidance

Raw weight error is not enough.

Two weights can have identical reconstruction error while having very different effects on model output.

We therefore estimate weight/block sensitivity using Hessian-based or practical approximate sensitivity metrics.

Conceptually score reconstruction as:

E = sum_i importance_i * (W_i - W_hat_i)^2

where importance_i approximates how damaging an error at weight i is.

Use sensitivity information primarily for:

- evaluating candidate reconstructions
- detecting dangerous blocks
- protecting highly important weights
- deciding where additional metadata is justified

Do not automatically preserve every high-sensitivity weight in full precision because metadata/storage can explode.

## 3. DSA structured reconstruction

DSA is our structured-weight reconstruction method.

The concept:

Instead of storing every weight magnitude independently, identify a compact mathematical pattern that explains a block.

The 1-bit representation already provides the sign/state information.

DSA attempts to reconstruct magnitudes using a small shared set of parameters.

Conceptually:

W_hat_i = sign_i * f(i; theta)

where theta is tiny metadata shared by the entire block.

Examples of possible patterns:

- approximately constant magnitude
- linear trend
- piecewise trend
- row-based structure
- column-based structure
- low-order polynomial-like relationship
- shared anchor + offsets
- other compact structured mappings

DSA should be used only when the block actually contains enough predictable structure.

We must measure:

- raw MSE
- sensitivity-weighted error
- metadata bits
- effective bits per weight
- perplexity
- percentage of blocks that DSA can reconstruct successfully

## 4. Seeded Residual Search for irregular/random blocks

DSA may fail on blocks whose magnitudes are irregular and difficult to describe with a compact deterministic pattern.

For those blocks, use seeded residual search.

Important principle:

DO NOT store all generated random values.

Instead use deterministic pseudo-random generation so encoder and decoder reproduce the same candidate reconstruction from:

- a global seed
- block ID
- candidate ID
- optional distribution class/parameters

Example:

Candidate_k = Generator(global_seed, block_id, k, distribution)

Generate K possible magnitude reconstructions.

Apply the known sign map.

Evaluate every candidate.

Choose the candidate with minimum sensitivity-weighted reconstruction error:

k* = argmin_k sum_i importance_i * (W_i - Candidate_k_i)^2

Store only the small candidate index and any unavoidable shared distribution metadata.

This effectively creates a generated codebook without explicitly storing a full codebook.

Test different candidate counts such as:

K = 2
K = 4
K = 8
K = 16

Remember:

candidate index cost = log2(K) bits PER BLOCK, not per weight.

The method should preferably generate candidates from the observed magnitude distribution rather than uniform random values.

Candidate distributions can include:

- mean + variance
- robust median + scale
- Laplace-like
- Gaussian-like
- quantile-derived
- predefined global distribution families

Keep metadata extremely small.

## 5. Salient-weight rescue

This is a last-resort accuracy mechanism.

After reconstruction, detect a very small subset of weights/blocks whose errors are highly damaging according to sensitivity.

Protect only those that cannot be safely reconstructed by DSA or seeded residual search.

Possible approaches:

- higher-bit residual for rare weights
- sparse correction values
- small outlier table
- higher precision for very small selected groups

This mechanism must be strictly limited because metadata could destroy the effective 1-bit target.

Always report its storage cost separately.

---

# Overall proposed architecture

Original weights

-> optional Hadamard transform

-> divide into blocks

-> compute block statistics

Statistics should include at minimum:

1. DSA reconstruction error
2. structure/correlation score
3. sensitivity/Hessian score
4. outlier score
5. magnitude distribution statistics

Then route each block:

Structured block
-> DSA

Irregular block
-> Seeded Residual Search

Then:

Sensitivity-weighted reconstruction check

If acceptable:
-> keep reconstruction

If unacceptable:
-> Salient-weight rescue

Finally undo any required transform and evaluate the model.

---

# CRITICAL METRICS

Every experiment must report BOTH compression and quality.

Do not report only MSE.

For every configuration record:

1. Raw weight MSE
2. Relative reconstruction error
3. Sensitivity-weighted reconstruction error
4. Cosine similarity if useful
5. Model perplexity
6. Perplexity degradation versus baseline
7. Effective bits per weight INCLUDING metadata
8. Metadata bits separately
9. Percentage of blocks handled by each method
10. Inference memory usage if practical
11. Reconstruction/dequantization overhead if practical

Always compare against the original BF16/FP16 model.

If possible also include simple low-bit baselines.

---

# EXPERIMENT 1 — Does DSA actually capture structured blocks?

Goal:

Isolate DSA and determine whether it gives meaningful reconstruction gains on naturally structured weight blocks.

Do NOT involve seeded residual search or salient rescue yet.

Run:

A. naive 1-bit sign + one scale
B. Hadamard + naive 1-bit
C. DSA without Hadamard
D. Hadamard + DSA

For every block calculate:

- DSA reconstruction error
- structure score
- sensitivity-weighted error
- metadata cost

Then study whether highly structured blocks consistently benefit from DSA.

Important question:

Does a measurable statistic predict when DSA will work?

Try correlations between DSA gain and:

- row correlation
- column correlation
- low-rank tendency
- local smoothness
- variance
- entropy
- spectral concentration
- simple regression fit score

Expected useful result:

There should be a subgroup of blocks where DSA produces substantially lower reconstruction error than naive 1-bit representation while requiring only tiny shared metadata.

If DSA works equally poorly everywhere, the hypothesis fails and its pattern family must be redesigned.

Main output:

Find a practical "DSA suitability score."

This experiment should tell us:

"When should a block be sent to DSA?"

---

# EXPERIMENT 2 — Can seeded residual search handle DSA failures?

Goal:

Take blocks where DSA performs poorly and determine whether deterministic seeded candidate search reconstructs them better for very little metadata.

Use ONLY the DSA-failure/irregular blocks identified in Experiment 1.

Compare:

A. naive 1-bit + scale
B. DSA result
C. seeded residual search with K=2
D. K=4
E. K=8
F. K=16

Test at least several candidate distributions:

1. uniform
2. Gaussian using block mean/std
3. robust distribution using median/scale
4. one global/shared distribution if possible

Evaluate candidates using BOTH:

raw MSE

and

sensitivity-weighted MSE.

Important comparison:

candidate chosen by raw MSE
versus
candidate chosen by sensitivity-weighted error.

Record:

- gain as K increases
- candidate index metadata cost
- distribution metadata cost
- perplexity impact
- whether returns diminish after K=4 or K=8

Expected useful result:

Seeded residual search should outperform DSA specifically on blocks that DSA classifies as irregular.

The desired outcome is NOT perfect reconstruction.

The desired outcome is:

large improvement in important reconstruction error for only a few bits of metadata per block.

Main output:

Determine:

- best candidate count K
- best distribution family
- whether Hessian-guided candidate selection matters
- threshold where DSA should hand the block to seeded search

---

# EXPERIMENT 3 — How much does Hessian/sensitivity guidance actually save?

Goal:

Measure whether protecting important errors matters more than minimizing total numerical error.

Build identical quantization configurations and change only the decision criterion.

Compare:

A. reconstruction chosen using raw MSE
B. reconstruction chosen using magnitude-weighted error
C. reconstruction chosen using Hessian/sensitivity-weighted error

For each method compare:

- raw MSE
- weighted error
- perplexity

Key question:

Can configuration B or C have WORSE raw MSE but BETTER perplexity?

If yes, that strongly validates sensitivity-guided reconstruction.

Then test salient rescue.

Start with extremely small fractions such as:

top 0.01%
top 0.05%
top 0.1%
top 0.5%

of the most damaging residuals.

Measure accuracy regained per metadata bit.

Main output:

Find the smallest salient subset that recovers the largest amount of model quality.

---

# EXPERIMENT 4 — Full adaptive hybrid router

Goal:

Combine all mechanisms and determine the best routing policy.

Pipeline:

Hadamard
-> block statistics
-> DSA or Seeded Residual Search
-> Hessian-weighted error check
-> optional salient rescue

Compare against ablations:

1. naive 1-bit
2. Hadamard only
3. DSA only
4. seeded residual only
5. DSA + seeded residual
6. DSA + seeded + Hessian
7. full system + salient rescue

For every configuration report:

- perplexity
- raw reconstruction error
- weighted reconstruction error
- effective bits/weight
- metadata
- method usage percentages

Example routing logic to test:

if DSA_score > threshold:
    use DSA
else:
    use seeded residual search

then:

if sensitivity_weighted_error > threshold:
    activate salient rescue

Do NOT hand-design the final thresholds based on one layer.

Sweep thresholds on a calibration subset and evaluate on held-out layers or calibration data to prevent overfitting.

Main output:

A routing table explaining:

- when DSA wins
- when seeded search wins
- when neither is safe
- when salient rescue is worth its metadata
- whether Hadamard helps each category

---

# EXPERIMENTAL DISCIPLINE

Change one variable at a time.

Do not modify the model architecture.

Keep calibration data fixed across comparisons.

Use identical evaluation data.

Use fixed random seeds.

Report results layer-by-layer AND globally.

Log every configuration automatically.

Do not conclude based on one layer.

Particularly inspect:

- attention projections
- Q/K/V
- attention output
- MLP up projection
- MLP gate
- MLP down projection

Different matrix types may require different routing.

---

# WHAT WE WANT TO DISCOVER

We are NOT trying to prove every proposed technique is useful.

We want experiments to kill weak ideas quickly.

At the end we should be able to produce a table like:

Method | Best block type | Accuracy gain | Metadata cost | Keep?
Hadamard | ... | ... | ... | yes/no
DSA | structured | ... | ... | yes/no
Seeded search | irregular | ... | ... | yes/no
Hessian | sensitive | ... | ... | yes/no
Salient rescue | extreme cases | ... | ... | yes/no

The final quantizer should contain ONLY techniques whose improvement justifies their actual storage and compute cost.

The core optimization target is approximately:

minimize:
    model quality degradation

subject to:
    extremely low effective bits/weight

where effective bits/weight includes EVERY scale, index, seed, flag, routing bit, residual and salient value.

Do not hide metadata.

---

# START NOW

Run Experiment 1 first.

Then run Experiment 2 using the exact block classifications and outputs produced by Experiment 1.

Do not run Experiments 3 or 4 yet.

For Experiments 1 and 2, produce:

1. implementation
2. raw results
3. per-layer table
4. aggregate table
5. effective bits-per-weight calculation
6. plots of reconstruction error distributions
7. relationship between structure score and DSA improvement
8. DSA-versus-seeded-search comparison on irregular blocks
9. short conclusion supported only by measured results

At the end answer these two questions clearly:

Q1: Can we reliably detect blocks where DSA is worth using?

Q2: On DSA-failure blocks, does seeded residual search recover significantly more useful information for sufficiently little metadata to justify keeping it?