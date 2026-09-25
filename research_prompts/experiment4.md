You are continuing the 1-bit quantization research project.

Previous experiments established:

- Hadamard transformation is strongly useful.
- DSA provides only a small gain after Hadamard.
- Seeded residual search failed and is removed.
- Plain residual SVD helps, but the residual itself is only weakly low-rank.
- Hessian-guided low-rank correction is much better than ordinary low-rank correction at identical storage.
- In Experiment 3, Hessian-guided rank-1 reduced weighted error and perplexity much more than ordinary rank-1.
- Damage is highly concentrated: a small fraction of columns contains a large fraction of sensitivity-weighted damage.

The new goal is much harder:

Target only 5–10% perplexity degradation from BF16.

If BF16 perplexity is about 17.9, the desired range is approximately:

18.8–19.7.

Do not optimize only for reconstruction MSE.

The experiment must focus on preserving model function.

---

# Core hypothesis

The current pipeline makes a weak 1-bit approximation first and repairs it afterward.

Experiment 4 should improve the binary decision itself and then spend extra bits only on the small subset of directions that still cause large model damage.

The proposed decomposition is:

W_hat = Q_H + L_H + S_H

where:

Q_H = Hessian-aware 1-bit base
L_H = Hessian-guided low-rank correction
S_H = sparse salient correction

The architecture is:

Original weights
-> randomized Hadamard / current Hadamard
-> Hessian-aware binary quantization
-> calculate residual damage
-> targeted Hessian-guided LRR
-> sparse salient rescue
-> evaluate perplexity and true BPW

---

# PART 1 — Hessian-aware binary base

Current naive 1-bit quantization uses a sign-like representation plus scale.

Instead, choose the binary reconstruction to minimize sensitivity-weighted error.

Conceptually optimize:

Q* = argmin_Q (W-Q) H (W-Q)^T

Use the available practical Hessian / activation-importance approximation if the full Hessian is too expensive.

The purpose is:

Do not first create a bad binary base and then repair it.

Make the 1-bit assignment itself sensitivity aware.

Compare at minimum:

A. Hadamard + naive binary
B. Hadamard + magnitude-aware binary
C. Hadamard + Hessian-aware binary

The storage budget for A/B/C should be matched as closely as possible.

Possible binary form:

Q_ij = alpha_g * sign(W_ij)

where alpha may be:

- per matrix
- per row
- per column
- per block

BUT metadata must be counted exactly.

Try only a few scale granularities:

1. per 128x128 block
2. per 64x64 block
3. per output row if feasible

Do not explode metadata.

For Hessian-aware rounding, optimize sign/scale assignments according to weighted reconstruction loss rather than raw MSE.

Report:

- raw MSE
- weighted error
- cosine similarity
- perplexity
- true BPW

Main question:

Does a better binary base dramatically reduce the amount of repair needed?

---

# PART 2 — Damage map after the binary base

After obtaining Q_H:

R = W - Q_H

Calculate:

D_ij = importance_j * R_ij^2

or the most accurate sensitivity-weighted damage metric available.

Measure damage concentration by:

- top 0.1% weights
- top 0.5%
- top 1%
- top 2%
- top 5%
- top 10%

Also calculate concentration over:

- columns
- rows
- blocks
- projection types
- layers

Generate cumulative curves:

fraction of weights/directions
vs
fraction of total weighted damage.

We need to discover whether most harmful error is sparse.

---

# PART 3 — Targeted Hessian-guided low-rank correction

Do NOT apply the same rank everywhere.

Experiment 3 showed that blanket LRR wastes metadata.

Instead route rank according to weighted damage.

Test a simple adaptive policy.

Example:

low-damage matrices/columns:
rank 0

medium-damage:
rank 1

high-damage:
rank 2

very-high-damage:
rank 4

Use Hessian-guided SVD, not ordinary SVD, for the adaptive branch.

Test budgeted versions:

Budget A:
average added BPW <= 0.025

Budget B:
average added BPW <= 0.05

Budget C:
average added BPW <= 0.10

The router must maximize weighted-error reduction per added bit.

A useful greedy criterion is:

gain_per_bit =
(weighted_error_before - weighted_error_after)
/ added_bits

Allocate rank only where this value is high.

Do not allocate rank just because raw residual energy is high.

---

# PART 4 — Sparse salient rescue

After Hessian-aware binary + adaptive LRR, compute the remaining residual:

R2 = W - (Q_H + L_H)

Calculate remaining damage:

D2_ij = importance_j * R2_ij^2

Then restore or encode only the most damaging individual residuals.

Initially use exact BF16 residual restoration for scientific diagnosis.

Test:

0%
0.01%
0.05%
0.1%
0.25%
0.5%
1%
2%
5%

of highest-damage residual entries.

The initial goal is NOT to optimize sparse encoding.

The initial question is:

How much perplexity can be recovered by fixing a tiny number of catastrophic errors?

For every percentage calculate exact storage cost including:

- value storage
- index storage
- any row/column metadata

Do not hide sparse indices.

---

# PART 5 — Compare selection strategies

For salient rescue compare:

A. largest |W|
B. largest |residual|
C. largest residual squared
D. Hessian-weighted damage

D = importance * residual^2

All methods must restore exactly the same number of entries.

This determines whether Hessian selection really identifies the most valuable weights.

---

# PART 6 — Structured sparse rescue

If individual sparse indices are too expensive, test whether damage clusters.

Check whether highly damaging entries cluster into:

- columns
- rows
- small contiguous groups
- 8x8 blocks
- 16x16 blocks

If yes, compare:

individual sparse weights
vs
whole sensitive columns
vs
small sensitive blocks

This is important because structured rescue can dramatically reduce index metadata.

Example:

Instead of storing 1000 independent coordinates,
store:

column 117
column 904
column 1102

plus residual vectors.

Measure:

accuracy recovered per metadata bit.

---

# REQUIRED CONFIGURATIONS

At minimum evaluate:

1. BF16
2. Hadamard + naive 1-bit
3. Hadamard + Hessian-aware 1-bit
4. Hessian-aware 1-bit + global rank-1 LRR
5. Hessian-aware 1-bit + adaptive LRR
6. Hessian-aware 1-bit + sparse rescue only
7. Hessian-aware 1-bit + adaptive LRR + sparse rescue

Optional reference:

8. previous Experiment 3 best configuration

---

# STORAGE TARGETS

Report results at approximately these true BPW budgets:

1.00–1.05
1.05–1.10
1.10–1.20
1.20–1.30
1.30–1.50

Do not compare configurations with very different storage without clearly showing the tradeoff.

Our desired region is:

as close as possible to 1 bit/weight

while trying to move perplexity toward:

<= 19.7

---

# CRITICAL METRICS

For every configuration report:

1. perplexity
2. perplexity degradation %
3. raw MSE
4. weighted error
5. cosine similarity
6. total BPW
7. binary metadata BPW
8. LRR metadata BPW
9. salient rescue BPW
10. exact index overhead
11. fraction of weights rescued
12. fraction of total weighted damage removed
13. perplexity recovery per added bit

Most important metric:

perplexity vs true BPW

---

# IMPORTANT PLOTS

Generate:

## Plot 1
Perplexity vs true BPW.

## Plot 2
Weighted error vs BPW.

## Plot 3
Cumulative weighted damage:
top X% weights vs damage captured.

## Plot 4
Perplexity vs fraction of salient weights restored.

## Plot 5
Magnitude selection vs residual selection vs Hessian selection.

## Plot 6
Adaptive rank allocation by projection type.

## Plot 7
Damage concentration by layer and projection.

---

# KEY RESEARCH QUESTIONS

Q1:

Does Hessian-aware 1-bit rounding materially outperform naive 1-bit at the same storage?

Q2:

Does adaptive rank allocation outperform applying the same low rank everywhere?

Q3:

How sparse is the remaining catastrophic quantization damage?

Q4:

Can restoring <1% of weights collapse perplexity dramatically?

Q5:

Does Hessian-based salient selection outperform magnitude/outlier selection?

Q6:

Is structured rescue by columns/blocks more storage-efficient than individual sparse rescue?

Q7:

What is the lowest true BPW configuration that reaches:

- PPL < 100
- PPL < 50
- PPL < 30
- PPL < 25
- PPL <= 19.7

Report the lowest BPW achieved for each threshold.

---

# IMPORTANT FAILURE RULES

Do not force the method to succeed.

If 1% sparse rescue barely improves perplexity, report it.

If 5% BF16 restoration is required to approach baseline quality, calculate the true BPW honestly.

If Hessian-aware binary rounding does not improve perplexity, reject it.

If adaptive LRR costs more metadata than it saves in model quality, reject it.

The purpose is to find the information bottleneck, not defend the hypothesis.

---

# MAIN SCIENTIFIC IDEA

We are testing whether extreme 1-bit quantization fails because:

A. all weights need much more precision

OR

B. most weights tolerate 1-bit, while a small number of sensitive directions contain most of the missing intelligence.

Experiment 3 provides evidence for B because damage was highly concentrated.

Experiment 4 must test B directly.

---

# FINAL OUTPUT

Produce:

1. implementation
2. aggregate table
3. per-layer table
4. per-projection table
5. exact BPW accounting
6. damage concentration analysis
7. Hessian-aware binary ablation
8. adaptive-LRR ablation
9. sparse-rescue ablation
10. selection-strategy comparison
11. all requested plots
12. short scientific conclusion

At the end answer:

Q1: Is Hessian-aware binary quantization better than naive binary?

Q2: Is most model damage concentrated enough for targeted correction?

Q3: What percentage of weights/directions causes most of the damage?

Q4: How much can sparse salient rescue lower perplexity?

Q5: What is the best perplexity achieved below 1.10, 1.20, 1.30, and 1.50 BPW?

Q6: Did any tested configuration reach within 5–10% of BF16 perplexity?

Q7: If not, what is now the dominant remaining source of error?

Run Experiment 4 now.

Do not introduce any new algorithm until Experiment 4 identifies whether the remaining failure is caused by sparse catastrophic errors or broadly distributed lost information.