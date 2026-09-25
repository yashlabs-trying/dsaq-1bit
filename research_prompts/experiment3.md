You are continuing the 1-bit quantization research project.

Experiments 1 and 2 already showed:

- Hadamard transformation clearly helps.
- Current DSA gives only a very small improvement after Hadamard.
- Seeded residual search does not justify itself and is removed from the current pipeline.
- The main unresolved problem is that even when average reconstruction error improves, perplexity is still catastrophic.

The new hypothesis is:

The quantization residual may contain low-rank structure that can be captured much more efficiently than using local pattern fitting.

Then, after removing that low-rank structure, Hessian/sensitivity information may reveal which remaining residual directions matter most.

We therefore want to test:

Hadamard
-> 1-bit base
-> first low-rank residual reconstruction
-> Hessian/sensitivity weighting
-> second low-rank residual reconstruction

The goal is to discover whether the missing information can be represented using only a few low-rank directions.

---

# Experiment 3 — Low-rank residual reconstruction with Hessian-guided second stage

## Main hypothesis

Let the original transformed weight matrix be:

W

After Hadamard + 1-bit quantization:

W_1bit

Residual:

R1 = W - W_1bit

We test whether:

R1 ≈ U1 * S1 * V1^T

using a very small rank.

Then subtract this reconstruction:

R2 = R1 - R1_hat

Now apply Hessian/sensitivity guidance to identify which remaining residual components are important.

Construct a sensitivity-weighted residual or equivalent importance-aware representation.

Then approximate the important residual using another low-rank reconstruction:

R2_important ≈ U2 * S2 * V2^T

Final reconstruction:

W_hat = W_1bit + R1_hat + R2_hat

The important question is:

Can a tiny number of low-rank directions recover the information that the 1-bit base loses?

---

# Scope

Use the same experimental setup as Experiments 1 and 2 unless there is a technical reason not to.

Use:

- Qwen2.5-0.5B
- layers 0, 12, 23
- Q/K/V/O projections
- MLP gate/up/down projections
- WikiText-2 calibration/evaluation
- same random seed
- same evaluation pipeline
- same perplexity calculation
- same block size initially

Do not change multiple unrelated variables at once.

---

# Baselines

Always include:

1. BF16
2. naive 1-bit
3. Hadamard + naive 1-bit

Optional useful baseline:

4. Hadamard + DSA from Experiment 1

The main new methods are compared against Hadamard + 1-bit.

---

# Stage 1 — First low-rank residual reconstruction

Start from:

R1 = W - W_1bit

Compute truncated SVD:

R1 ≈ U_r * S_r * V_r^T

Test ranks:

r1 = 1
r1 = 2
r1 = 4
r1 = 8

Do not start with larger ranks.

For every rank record:

- raw residual MSE
- relative residual error
- weighted error
- cosine similarity
- explained residual energy
- metadata bits
- effective BPW
- perplexity

Most importantly calculate:

residual energy captured by top-r singular directions.

Example:

rank 1 captures X%
rank 2 captures Y%
rank 4 captures Z%
rank 8 captures ...

We need to know whether the residual itself is naturally low-rank.

---

# Stage 2 — Hessian / sensitivity analysis

After first low-rank correction:

R2 = W - (W_1bit + R1_hat)

Now estimate importance/sensitivity.

Use the same practical Hessian or sensitivity approximation already available in the project.

Do NOT simply rank residuals by magnitude.

We want to determine which remaining errors are most harmful.

For each element or direction calculate something conceptually similar to:

damage_i = sensitivity_i * R2_i^2

Also calculate aggregate damage per:

- row
- column
- block
- singular direction if practical

We want to understand whether sensitive residual error has structure.

---

# Stage 3 — Second low-rank reconstruction

Now fit another low-rank approximation specifically targeted at the important residual.

Test at least two approaches.

## Method A — Direct second SVD

Apply SVD directly to R2.

R2 ≈ U2 * S2 * V2^T

This tests whether a second ordinary low-rank correction is enough.

## Method B — Hessian-guided low-rank residual

Weight or transform R2 using sensitivity before fitting.

Conceptually:

R2_weighted = sensitivity_transform(R2)

Then calculate a low-rank approximation that prioritizes directions causing large functional error.

The exact implementation can vary, but the objective must correspond to minimizing something closer to:

sum_i sensitivity_i * (R2_i - R2_hat_i)^2

rather than plain MSE.

Test second-stage ranks:

r2 = 1
r2 = 2
r2 = 4

Do not use rank 8 initially unless the smaller ranks clearly justify expansion.

---

# Rank combinations

At minimum test:

r1=1, r2=1
r1=2, r2=1
r1=2, r2=2
r1=4, r2=1
r1=4, r2=2
r1=4, r2=4
r1=8, r2=2
r1=8, r2=4

Also test first-stage-only versions so we can measure whether the second stage actually helps.

For example:

Hadamard + 1bit + rank2
Hadamard + 1bit + rank4
Hadamard + 1bit + rank8

versus:

Hadamard + 1bit + rank2 + Hessian + rank1
Hadamard + 1bit + rank4 + Hessian + rank2
etc.

---

# Metadata accounting

This is CRITICAL.

For every SVD component include the storage cost of:

U
S
V

Do not report only rank.

For a matrix M x N and rank r, storage is approximately:

r*M + r*N + r

values

before quantization/compression of the factors.

Calculate their exact effective BPW.

Test factor storage using:

- FP16 initially for scientific measurement
- optionally INT8 later

Do not hide low-rank factor metadata.

We are not yet trying to optimize the factors perfectly.

First discover whether the low-rank information is valuable enough.

---

# Main metrics

For every configuration report:

1. perplexity
2. perplexity delta from BF16
3. raw weight MSE
4. relative error
5. sensitivity-weighted error
6. cosine similarity
7. rank-1/2/4/8 explained residual energy
8. first-stage metadata
9. second-stage metadata
10. total metadata
11. true effective BPW
12. improvement in perplexity per added metadata bit

The final metric is very important:

accuracy recovery / metadata cost

We want to know whether a low-rank correction is storage-efficient.

---

# Critical plots

Generate these plots.

## Plot 1

Residual singular-value spectrum.

This tells us whether the 1-bit residual is actually low-rank.

## Plot 2

Cumulative residual energy captured versus rank.

Rank:
1, 2, 4, 8, 16 if needed for analysis only.

## Plot 3

Perplexity versus first-stage rank.

## Plot 4

Perplexity versus effective BPW.

## Plot 5

Hessian-weighted residual error before and after second SVD.

## Plot 6

Perplexity comparison:

Hadamard 1-bit
vs
Hadamard + first LRR
vs
Hadamard + first LRR + second ordinary LRR
vs
Hadamard + first LRR + Hessian-guided LRR

---

# Questions Experiment 3 must answer

Q1:

Is the residual after Hadamard + 1-bit naturally low-rank?

A useful result would be that very small ranks capture a significant fraction of residual energy.

Q2:

Does first-stage low-rank reconstruction meaningfully improve perplexity?

Do not accept only better MSE.

Q3:

After first-stage LRR, are the remaining harmful errors concentrated in a small number of Hessian-sensitive directions?

Q4:

Does Hessian-guided second LRR outperform ordinary second LRR at the same rank and same metadata budget?

Q5:

What rank combination gives the best:

perplexity recovery / effective BPW

Q6:

Does low-rank residual reconstruction outperform the previous DSA approach?

---

# Important failure cases

Be ready for these outcomes.

## Failure A

Residual spectrum is flat.

Meaning:

No strong low-rank structure exists.

Then low-rank reconstruction is not suitable.

## Failure B

SVD greatly improves MSE but barely improves perplexity.

Meaning:

The dominant residual energy is not aligned with sensitive model directions.

Then Hessian guidance becomes the main focus.

## Failure C

Hessian-guided second SVD dramatically improves perplexity.

Meaning:

Important model damage is concentrated in a few sensitive directions.

This would strongly support the new architecture.

## Failure D

Even rank-8 + Hessian-guided residual gives catastrophic perplexity.

Then low-rank structure is insufficient and we should pivot toward sparse/salient correction rather than increasing rank indefinitely.

Do not keep increasing rank just to force a win.

---

# Decision rule

Keep LRR only if it produces a meaningful quality gain relative to its true metadata cost.

Do not keep it just because reconstruction MSE decreases.

The strongest positive result would be:

very small r1 + very small r2
-> major perplexity recovery
-> modest BPW increase

The weak result would be:

large rank needed
-> small perplexity improvement
-> large metadata cost

---

# Final output

Produce:

1. complete implementation
2. aggregate results table
3. per-layer results
4. per-projection results
5. singular-value plots
6. perplexity versus BPW plot
7. first-stage rank ablation
8. second-stage rank ablation
9. ordinary versus Hessian-guided second LRR comparison
10. short conclusion

At the end answer clearly:

Q1: Is the 1-bit residual low-rank enough to exploit?

Q2: Does first LRR materially recover model quality?

Q3: Does Hessian-guided second LRR recover more quality per metadata bit than ordinary LRR?

Q4: Should LRR replace DSA as the main reconstruction mechanism?

Run Experiment 3 now.

Do not run later experiments until these questions are answered.