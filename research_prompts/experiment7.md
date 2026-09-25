# Experiment 7 — Solve IMPORTANCE

We are building a 1-bit LLM quantization framework around three problems:

1. IMPORTANCE
2. COMPACTNESS
3. ENCODING

We are still working ONLY on:

# PART 1 — IMPORTANCE

Do not work on compactness or encoding.

The purpose of this experiment is to find a much better way to identify which quantization errors must be corrected in order to prevent model collapse.

---

# Previous result

Model:

Qwen2.5-0.5B

BF16 PPL:

~17.90

Hadamard + 1-bit baseline:

~637,559 PPL

Best selector from Experiment 6:

Functional Fisher, global selection

Results:

5% restored:
PPL ~1,831

10% restored:
PPL ~649.5

Therefore:

- Fisher is much stronger than activation saliency.
- But even restoring 10% exactly does not reach PPL <100.
- The previous small-critical-subset hypothesis was NOT validated yet.
- Importance is clearly not captured well enough by independent weight scores.

Important additional observations:

- Fisher selection was strongly non-monotonic.
- Restoring more individually high-ranked weights sometimes made perplexity WORSE.
- This strongly suggests quantization-error interactions and error cancellation.
- V, K and O projections were strongly enriched in the best Fisher selection.
- Largest collapse occurred around layer 21.
- `o_proj` was identified by projection rescue.

Therefore Experiment 7 must explicitly model:

1. actual quantization error
2. interactions between errors
3. the current damaged 1-bit model state
4. conditional importance after previous corrections

---

# PRIMARY SUCCESS TARGET

The experiment is considered successful only if one method achieves:

PPL < 100

Preferably while restoring:

<=10% of weights

Stretch target:

PPL <50

Ultimate oracle target:

PPL <= ~19.7

Do not move to COMPACTNESS unless we first obtain a convincing importance mechanism.

---

# FIXED BASELINE

Keep Hadamard preprocessing fixed.

Construct exactly the same 1-bit baseline as Experiment 6.

Let:

W = original BF16 weight matrix

Q = reconstructed Hadamard 1-bit weight matrix

Define actual quantization error:

ΔW = W - Q

This is the quantity importance selection should reason about.

Do NOT score a weight simply because W itself is large.

We care about the damage created by:

W -> Q

---

# ALGORITHM A — Quantization-Aware SparseGPT / OBS

Call this:

# Q-OBS

Quantization-aware Optimal Brain Surgeon.

SparseGPT uses second-order reconstruction and inverse-Hessian information to account for interactions among parameters when weights are pruned.

We adapt this idea to quantization.

SparseGPT normally asks:

"What happens if this weight becomes zero?"

Our problem is:

"What happens if this BF16 weight becomes its 1-bit approximation?"

Therefore the perturbation is not:

δw = -w

It is:

δw = Q - W

or equivalently correction:

Δw = W - Q

---

# A1 — Build layer Hessian / activation covariance

For every linear projection collect calibration inputs:

X

Estimate:

H = XX^T + λI

Use damping for numerical stability.

Compute or approximate:

H^-1

Use the same type of numerically stable factorization strategy used by SparseGPT where possible.

Do this independently per projection.

---

# A2 — Quantization-aware OBS damage

For each candidate quantization error Δw_i, estimate the second-order functional damage.

The naive diagonal approximation is approximately:

Damage_i ≈ Δw_i² / [H^-1]_ii

But DO NOT stop there.

The reason we are using OBS/SparseGPT is to model interactions.

Use the inverse Hessian to calculate how correcting or keeping one quantization error changes the optimal compensation pattern across the remaining weights.

Conceptually:

correct weight i

then propagate the compensation through correlated dimensions using H^-1.

---

# A3 — Do NOT use independent ranking only

This is crucial.

Experiment 6 showed that a fixed independent ranking is insufficient.

Instead use iterative OBS-style selection.

At each step:

1. evaluate candidate correction damage
2. select the most useful correction
3. update the effective residual/error state
4. update remaining scores
5. continue

The importance ordering must be conditional.

Do NOT calculate scores once and sort forever.

---

# A4 — Selection objective

We want a protection mask M that minimizes functional quantization damage:

min_M

(ΔW_unprotected)^T H (ΔW_unprotected)

subject to:

number of protected weights <= budget

where:

ΔW_unprotected =
(W - Q) for weights not protected

and:

0 for protected weights

This is the core difference from ordinary Fisher.

---

# A5 — SparseGPT-style compensation experiment

Run TWO versions.

## Q-OBS Select Only

Use OBS only to determine which weights are important.

Then restore those selected weights exactly to BF16.

Do NOT permanently modify unselected weights.

This is the clean oracle selector test.

## Q-OBS + Temporary Compensation

For diagnostic purposes only:

after selecting a correction, allow SparseGPT-style temporary compensation of remaining quantized weights.

Measure how much extra functional recovery comes from correlated compensation.

Do not count this as the final selector result.

We want to know whether the value comes from:

- better importance selection
or
- reconstruction updates.

---

# ALGORITHM B — Iterative Causal Quantization Rescue

Call this:

# ICQR

This algorithm is built directly from Experiment 6's failures.

Experiment 6 taught us:

- BF16 Fisher is not enough.
- Importance changes after some weights are restored.
- Errors interact.
- The broken quantized model is the state we actually need to repair.
- PPL behavior is non-monotonic.

Therefore importance should be defined as:

"How much does repairing this quantization error improve the CURRENT broken model?"

Not:

"How important was this weight in the original BF16 model?"

---

# B1 — Teacher and broken student

Teacher:

BF16 model

Student:

Hadamard 1-bit model

Use calibration data.

Teacher outputs:

- hidden states
- logits
- probability distribution

Student outputs:

same quantities.

---

# B2 — Repair loss

Use a functional repair objective.

Primary:

L_repair =
KL(
P_teacher
||
P_quantized
)

Also track:

- language-model NLL
- hidden-state NMSE
- hidden-state cosine similarity
- logit KL

A candidate is valuable if restoring it reduces the teacher-student mismatch.

---

# B3 — Quantized-model gradient score

Compute gradients on the CURRENT 1-bit model, not the BF16 model.

For a weight:

ΔW_i = W_i - Q_i

Gradient:

g_i =
∂L_repair / ∂Q_i

First-order estimated rescue:

R_i =
|ΔW_i * g_i|

Also test signed estimated improvement:

R_i_signed =
-ΔW_i * g_i

The sign matters because some restorations may actually hurt due to error cancellation.

This addresses the non-monotonic result observed in Experiment 6.

---

# B4 — Hierarchical causal search

Directly testing hundreds of millions of individual weights is impossible.

Use a hierarchical search.

Start with large groups.

Level 1:

128x128 blocks

Measure predicted rescue.

Keep only the strongest candidate blocks.

Split them.

Level 2:

64x64

Level 3:

32x32

Level 4:

16x16

If worthwhile:

8x8

Only descend into regions showing large functional rescue potential.

This should drastically reduce direct evaluations.

---

# B5 — Actual causal rescue measurement

For promising group G:

temporarily restore:

Q_G -> W_G

Run calibration forward pass.

Measure:

CausalGain(G) =
L_before - L_after

If:

CausalGain > 0

the group genuinely repairs the binary model.

If:

CausalGain < 0

restoring the group makes the model worse.

This negative case is extremely important.

It is direct evidence of quantization-error cancellation.

Do NOT select groups only because their local score is high.

---

# B6 — Iterative conditional rescue

This is the main algorithm.

Start:

Q^(0) = full Hadamard 1-bit model

Then:

Iteration 1:
find most useful repair group G1

restore G1

obtain Q^(1)

RECOMPUTE:

- activations
- gradients
- candidate rescue scores

Iteration 2:
find best group G2 conditioned on G1 already being restored

restore G2

obtain Q^(2)

Repeat.

Therefore:

Importance(G2)

is measured after G1 is already corrected.

This explicitly captures interactions.

---

# B7 — Do not optimize weight MSE

Selection must be based primarily on functional recovery.

Do not choose candidates because they minimize:

||W-Q||²

A weight/group may have large reconstruction error and little functional impact.

Another may have small reconstruction error and catastrophic model impact.

Our primary objective is:

teacher-student functional recovery.

---

# B8 — Collapse-aware priority

Use our previous collapse diagnostics as a search prior.

We know:

- layer 21 is highly problematic
- o_proj caused major rescue at the collapse
- V/K/O are disproportionately selected by Fisher

Therefore test two ICQR search modes:

## Global ICQR

Search all layers/projections normally.

## Collapse-Prior ICQR

Initially allocate more causal evaluations to:

- V
- K
- O
- late transformer layers
- especially layer 21 and neighboring layers

This is a SEARCH PRIOR only.

Do not force the final protected set to remain there.

If causal evidence points elsewhere, follow it.

---

# BUDGETS

For both Q-OBS and ICQR evaluate protected percentages:

0.1%
0.25%
0.5%
1%
2%
3%
5%
7.5%
10%

If PPL <100 is still not reached by 10%, continue DIAGNOSTIC runs:

15%
20%

The 15–20% runs are not considered a successful small-critical-subset result.

They are only used to estimate where the recovery threshold actually lies.

---

# IMPORTANT: ADAPTIVE BUDGET ALLOCATION

Do not protect the same percentage in every layer.

Let the algorithm allocate protection globally.

A layer may receive:

0.1%

while another receives:

15%

if that is what maximizes global functional recovery under the same total budget.

Report the resulting layerwise allocation.

---

# BASELINES

Compare against:

1. BF16
2. Hadamard 1-bit
3. Experiment 6 Fisher global
4. Quantization-error Fisher:

S_i =
(W_i - Q_i)² * E[g_i²]

5. Q-OBS
6. ICQR gradient ranking without recomputation
7. full iterative ICQR
8. collapse-prior ICQR

This separates where gains come from.

---

# CRITICAL ABLATION: STATIC VS CONDITIONAL

For ICQR compare:

## Static

Compute rescue ranking once.

Restore top p%.

## Conditional

Restore a fraction.

Recompute model.

Recompute importance.

Select next fraction.

Repeat.

This is one of the most important comparisons in the experiment.

If conditional dramatically wins, Experiment 6's non-monotonicity was correctly diagnosed.

---

# ITERATION GRANULARITY

Do not restore all 10% at once.

Test iterative increments such as:

0.25%

or:

0.5%

At each increment:

1. restore
2. re-evaluate
3. recompute scores

For expensive causal search, use larger group-level increments initially and finer increments near sensitive regions.

---

# STOPPING / BACKTRACKING

If an iteration increases NLL or worsens teacher KL:

do not blindly keep it.

Test reverting the most recent group.

Maintain:

accepted correction set

and

rejected correction set

This creates a simple causal greedy algorithm with backtracking.

---

# VERY IMPORTANT ERROR-CANCELLATION ANALYSIS

We need direct evidence of interactions.

Find pairs/groups where:

repair A alone improves model

repair B alone improves model

but:

repair A+B is worse than expected

and cases where:

A alone is weak

B alone is weak

but:

A+B strongly improves model.

Measure an interaction score:

I(A,B) =
Gain(A+B)
-
Gain(A)
-
Gain(B)

Positive or negative interaction should be recorded.

This tells us how non-additive the quantization damage really is.

---

# METRICS

For every protected budget report:

- PPL
- mean NLL
- teacher logit KL
- final hidden-state NMSE
- hidden cosine
- weight NMSE
- protected %
- protected weight count
- layer distribution
- projection distribution
- runtime
- calibration evaluations

---

# REQUIRED CURVES

Plot:

## Plot 1

Protected % vs PPL

Curves:

Fisher
Quantization Fisher
Q-OBS
Static ICQR
Iterative ICQR

Use log-scale PPL.

Draw lines at:

100
50
30
19.69

---

## Plot 2

Protected % vs logit KL.

## Plot 3

Hidden-state drift across layers.

Compare:

Hadamard baseline
Fisher 10%
Q-OBS best
ICQR best

## Plot 4

Protection allocation per layer.

## Plot 5

Protection allocation by projection.

## Plot 6

Causal gain versus predicted gain.

This determines whether the cheap ICQR gradient score predicts actual repair.

---

# SUCCESS GATE

The main target is:

PPL < 100

with:

<=10% exact oracle restoration.

If achieved, report the SMALLEST percentage reaching it.

Then continue toward:

PPL <50
PPL <30
PPL <=19.69

If PPL <100 is achieved at <=5%, this is an exceptionally strong Importance result.

If achieved around 5–10%, Importance is promising but Compactness will need to be very strong.

---

# FAILURE RULE

If:

Q-OBS
and
full conditional ICQR

both fail to reach:

PPL <100

even with 10% exact restoration,

do NOT invent another simple scalar selector.

Conclude that either:

1. more than 10% of the weight information is genuinely necessary,
or
2. importance exists at a larger subspace/group level rather than individual weights,
or
3. the Hadamard 1-bit bulk representation is too destructive for sparse restoration to repair.

Use the 15% and 20% oracle runs to distinguish these possibilities.

---

# REQUIRED FINAL ANSWERS

Answer clearly:

Q1. Does Q-OBS beat Experiment 6 Fisher?

Q2. Does quantization-error-aware scoring beat W²Fisher?

Q3. Does ICQR beat Q-OBS?

Q4. Does iterative recomputation beat static ranking?

Q5. Does causal group rescue predict actual perplexity recovery?

Q6. What is the smallest protected percentage reaching PPL <100?

Q7. What percentage reaches <50?

Q8. What percentage reaches <30?

Q9. What percentage reaches <=19.69?

Q10. Are the most important corrections still concentrated in V/K/O?

Q11. Is layer 21/o_proj still the major collapse point after iterative repair?

Q12. How strong are error interactions according to I(A,B)?

Q13. Is the small-critical-subset hypothesis now supported or rejected?

---

# CORE SCIENTIFIC DISTINCTION

Experiment 6 asked:

"Which BF16 weights look important?"

Experiment 7 must ask:

# "Which quantization errors must be repaired to restore the function of the already-broken 1-bit model?"

Algorithm A answers this using:

SECOND-ORDER INTERACTION-AWARE QUANTIZATION DAMAGE.

Algorithm B answers this using:

DIRECT ITERATIVE CAUSAL FUNCTIONAL RECOVERY.

Run both.

Do not work on Compactness or Encoding yet.

The hard success gate is:

PPL <100.