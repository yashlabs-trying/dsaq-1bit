# Experiment 9 — WaveFly Hierarchical Adaptive Group Rescue

## Research objective

We are building **WaveFly-1 Dense**, a near-1-bit LLM quantization framework.

The framework has three stages:

1. IMPORTANCE
2. COMPACTNESS
3. ENCODING

We are still solving:

# STAGE 1 — IMPORTANCE ONLY

Do NOT work on compactness.

Do NOT work on encoding.

Do NOT prune/delete layers, heads, or channels from the architecture.

Do NOT use LoRA or fine-tuning.

We are trying to determine the **minimum original residual information that must remain protected** so the quantized model stays within:

PPL <= 1.10 × BF16 PPL.

For Qwen2.5-0.5B:

BF16 PPL ≈ 17.898

Hard ceiling:

PPL <= 19.688

---

# Previous results

Hadamard + 1-bit:

PPL ≈ 637,559

Best Experiment 6 Fisher restoration:

10% protected
PPL ≈ 649.5

Experiment 7 showed:

- local causal importance exists
- but local corrections interact strongly
- independent ranking does not compose well

Experiment 8 showed:

- starting from BF16 and removing residual information can maintain the 10% PPL bound
- but the best found mask still protected ~94.23% of the residual
- scalar/mask elimination therefore did not find a sufficiently small important subset

We now want a different search structure.

Instead of examining millions of weights individually:

# SEARCH IMPORTANCE HIERARCHICALLY IN LARGE GROUPS.

---

# Central hypothesis

A large part of the remaining model information may be necessary only because we are searching at the wrong granularity.

Functional importance may be concentrated in:

- specific transformer layers
- specific projections
- attention heads
- MLP channels
- contiguous blocks

Therefore search coarse-to-fine.

Start with a known strong core.

Then add increasingly important groups from the remaining 90% until the full model returns to the <=10% PPL boundary.

---

# STEP 1 — Build fixed 1-bit base

Use exactly the previous Hadamard + 1-bit implementation.

For every weight:

Q = Hadamard 1-bit reconstruction

Residual:

R = W - Q

No changes to the baseline quantizer.

---

# STEP 2 — Create the initial CORE

Use the strongest known Experiment 6 selector:

functional Fisher global.

Protect the top:

10%

of weights exactly.

Everything else remains at Q.

This should approximately reproduce:

PPL ~649.5

Verify this before continuing.

Call this:

CORE10

The core is temporarily fixed during the first search.

---

# IMPORTANT

The purpose of CORE10 is not to claim that these 10% are the final important set.

It gives us a substantially less destroyed starting model.

We then ask:

> What minimum ADDITIONAL information from the remaining 90% is needed to bring PPL from ~649 to <=19.688?

---

# Main representation

Model weights are:

W_hat =
Q
+
M_core ⊙ R
+
M_extra ⊙ R

where:

M_core = fixed top 10% Fisher mask

M_extra = groups selected from remaining 90%

Initially:

M_extra = 0

Then progressively activate groups.

---

# Main optimization objective

Find:

minimum |M_extra|

such that:

PPL(W_hat) <= 19.688

Do NOT target a fixed additional percentage.

The stopping condition is QUALITY.

---

# STEP 3 — Hierarchical groups

Organize the remaining 90% into a hierarchy.

Use:

LEVEL 1:
Transformer layer

LEVEL 2:
Projection

Examples:

q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj

LEVEL 3:
Functional structures

For attention:

attention head / head group

For MLP:

intermediate channels / channel groups

LEVEL 4:
Weight blocks

128×128

LEVEL 5:

64×64

LEVEL 6:

32×32

LEVEL 7:

16×16

Do not immediately search 16×16 across the entire model.

Start coarse.

Only refine groups that appear necessary.

---

# STEP 4 — Group importance score

For every currently unprotected group G calculate multiple cheap importance signals.

At minimum include:

## A. Fisher group score

Sum Fisher importance over G.

## B. Quantization-error Fisher

Use:

sum_G
(W-Q)^2 × Fisher

## C. Functional residual score

Use calibration activations.

Estimate:

||X(W-Q)_G||²

## D. Teacher-trajectory score

Temporarily activate group G and measure improvement in:

- teacher KL
- hidden-state NMSE
- NLL

Use cheap calibration samples.

Do not run full perplexity for every candidate.

---

# STEP 5 — Rank groups

At the current model state:

rank all candidate groups:

most useful
→
least useful

But the ranking must NOT remain fixed forever.

After accepted rescue batches:

RECOMPUTE the ranking.

Experiment 7 showed importance changes with model state.

---

# STEP 6 — Aggressive macro rescue

Start with large groups.

Example:

CORE10

PPL ~649

Then test adding the most important:

- projection
- layer-projection pair
- large channel group

Do not add one tiny block at a time.

The early objective is to move rapidly:

649
→ 200
→ 80
→ 30
→ ~20

with as few evaluations as possible.

---

# STEP 7 — Batch addition

At each iteration propose a batch of highest-ranked groups.

Test different rescue batch sizes:

large:
5% additional residual

medium:
2%

small:
1%

very small:
0.25–0.5%

Start large.

As PPL approaches the target, reduce batch size.

---

# Acceptance rule

For proposed batch B:

activate:

M_extra[B] = 1

Then evaluate.

If PPL improves strongly:

ACCEPT.

If PPL worsens:

REJECT.

If calibration surrogate improves but full PPL worsens:

REJECT.

Actual PPL is the final authority.

---

# STEP 8 — Backtracking

Because interactions are non-additive:

never assume every member of a successful batch is necessary.

Suppose:

adding 5% drops:

PPL 120 → 25

Great.

Now split that 5%.

Test whether half can be removed again.

Example:

5%
→ 2.5% A + 2.5% B

Try:

CORE + A

CORE + B

CORE + A+B

Keep only what is truly necessary.

Continue recursively.

---

# STEP 9 — Fine refinement near the boundary

Once:

PPL < 30

stop adding large groups.

Now refine the most recently accepted groups:

projection
→ head/channel group
→ 128×128
→ 64×64
→ 32×32
→ 16×16

Find the smallest subgroups actually needed.

The goal is:

PPL <=19.688

with the smallest protected percentage.

---

# STEP 10 — Revisit the original CORE10

This is extremely important.

Once the model has reached:

PPL <=19.688

do NOT assume all Fisher CORE10 weights are still necessary.

Now treat the original 10% core as removable too.

Start hierarchical elimination from CORE10.

Try removing:

- entire Fisher-core groups
- rows
- columns
- blocks

while maintaining:

PPL <=19.688

This means the final mask is not forced to contain the original 10%.

CORE10 is only a search starting point.

---

# STEP 11 — Swap search

Suppose current mask protects:

Group A

but another unprotected Group B could replace its function.

Test:

remove A
add B

If:

PPL stays <=19.688

and B costs less information:

accept the swap.

This helps escape the bias of the initial Fisher core.

---

# STEP 12 — Maintain best safe state

Always store:

BEST_SAFE_MODEL

with:

- PPL
- protected %
- protected count
- protected groups
- KL
- hidden-state metrics

If a new candidate fails:

restore BEST_SAFE_MODEL.

Never lose the best certified mask.

---

# THREE SEARCH MODES

Run three variants.

## Mode A — Pure Fisher hierarchy

Rank groups only by Fisher.

This is the baseline.

## Mode B — Functional hierarchy

Rank using:

teacher KL reduction
+
hidden-state recovery
+
quantization residual damage

## Mode C — Adaptive hybrid hierarchy

Start using cheap Fisher/residual scoring.

Near the quality boundary switch to direct functional group testing.

This is expected to be strongest.

---

# CRITICAL ABLATION

Also test starting core sizes:

5% Fisher

10% Fisher

15% Fisher

Do NOT assume 10% is optimal.

Measure:

starting PPL
additional protection required
final total protected %

The winning starting core is whichever leads to the smallest FINAL protection set.

---

# Why this experiment is different

Experiment 8:

started at 100% protected
and tried to remove everything.

Experiment 9:

starts with only a strong core
and searches for the minimum extra functional structures necessary to restore the model.

This is:

# BUILD UP + REFINE

instead of:

# REMOVE DOWN FROM 100%

---

# Most important measurement

For every accepted rescue step record:

Total protected %

PPL

Example:

10% → 649
15% → ?
20% → ?
25% → ?
30% → ?
...

We need to discover the actual recovery curve.

---

# Hard success target

PPL:

<=19.688

Then minimize:

total protected residual %

Do not stop at:

PPL 100

PPL 50

PPL 30

Those are milestones only.

The experiment succeeds only when the 10% degradation boundary is reached.

---

# Secondary objective

After reaching <=19.688:

remove as much protection again as possible.

Example:

25% protected → PPL 19.4

then recursively test:

24%
23%
22%
...

until no further reduction remains feasible.

---

# Required outputs

Produce:

1. PPL vs protected percentage curve
2. rescue history
3. accepted/rejected group log
4. final protected percentage
5. distribution by layer
6. distribution by projection
7. distribution by heads/channels
8. final hidden-state drift
9. initial CORE overlap with final mask
10. amount of CORE10 eventually removed
11. amount of non-core information eventually added
12. runtime and number of PPL evaluations

---

# Questions to answer

Q1. Can hierarchical group rescue reach PPL <=19.688?

Q2. What minimum total protected percentage was found?

Q3. How much additional protection beyond CORE10 was required?

Q4. Which hierarchy level produced the biggest recovery?

Q5. Are layers or projections more useful than individual blocks?

Q6. Which projection families dominate the final protection set?

Q7. Does adaptive recomputation outperform fixed ranking?

Q8. Does functional group scoring outperform Fisher group scoring?

Q9. Did backtracking significantly reduce protection?

Q10. How much of the original top-10% Fisher core could eventually be removed?

Q11. Is the final important set strongly structured?

Q12. Does starting with 5%, 10%, or 15% Fisher core give the smallest final mask?

Q13. What exact protected percentage is required to stay under 10% PPL degradation?

---

# Failure interpretation

If the model still requires:

50%+
80%+
90%+

of the residual to remain exact,

do not proceed to compactness.

Conclude that the current 1-bit base requires too much exact residual information.

If instead hierarchical rescue reaches <=19.688 with a much smaller structured set:

then Stage 1 has finally made meaningful progress.

---

# FINAL RULE

Do not delete model architecture.

Do not prune layers permanently.

Do not use LoRA.

Do not fine-tune.

The hierarchy is ONLY a search mechanism for locating the information that must survive 1-bit quantization.

The central question is:

# Starting from the strongest small core, what is the minimum additional STRUCTURED information required to restore BF16-like function?

Run Experiment 9 now.