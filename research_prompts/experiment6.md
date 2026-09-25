# Research Project: 1-Bit LLM Quantization

We are developing a new extreme quantization framework.

The final objective is approximately:

- ~1 bit per weight + very small metadata
- <10% reconstruction / quantization error where meaningful
- only 5–10% perplexity degradation versus BF16

But we are NOT trying to solve the entire quantization problem at once.

The research is divided into exactly three problems:

# PART 1 — IMPORTANCE

Find the smallest subset of weights / weight directions that must be preserved accurately for the LLM to retain its function.

# PART 2 — COMPACTNESS

Once those critical weights are known, reorganize or represent them as compact structured matrices/regions.

# PART 3 — ENCODING

Encode those compact important regions together with the remaining bulk weights using approximately 1 bit/weight + minimal metadata.

---

# CURRENT RULE

For this experiment work ONLY on:

# PART 1 — IMPORTANCE

Do NOT design:

- compact matrices
- permutations
- VQ
- codebooks
- LRR
- DSA
- residual compression
- metadata encoding
- new 1-bit representations

We will not touch Parts 2 or 3 until we know how to identify the genuinely important information in the weights.

---

# CENTRAL QUESTION

We want to answer:

> What is the smallest fraction of LLM weights that must remain highly accurate to prevent catastrophic model collapse?

And:

> What importance metric identifies those weights most accurately?

For example, could preserving:

0.1%
0.25%
0.5%
1%
2%
5%

of the correct weights while aggressively binarizing everything else keep the model close to BF16?

We do not know.

This experiment must determine it.

---

# MODEL AND BASELINE

Use the same model and evaluation setup as previous experiments:

- Qwen2.5-0.5B
- WikiText-2
- same calibration split
- same evaluation split
- fixed random seed
- exact BF16 baseline perplexity recorded

Hadamard preprocessing is considered part of our established base.

Use:

Hadamard → aggressive 1-bit bulk quantization

as the destructive baseline.

Then selectively restore important weights to their original BF16 values.

IMPORTANT:

Keeping selected weights in BF16 is ONLY an oracle experiment.

We are NOT proposing to store them in BF16 in the final method.

The purpose is to determine which weights matter.

---

# CORE ORACLE TEST

For every importance method:

1. rank all candidate weights by importance
2. select the top p%
3. preserve those selected weights exactly at BF16
4. quantize every remaining weight using the SAME fixed Hadamard + 1-bit baseline
5. evaluate full-model perplexity

Test:

p =

0%
0.1%
0.25%
0.5%
1%
2%
3%
5%
10%

Do NOT change the bulk quantizer between methods.

The only variable should be:

WHICH weights are protected.

This allows a fair comparison of importance metrics.

---

# IMPORTANCE METHOD 1 — MAGNITUDE

The simplest baseline.

For each weight:

S_mag(i,j) = |W_ij|

Rank weights by absolute magnitude.

Protect the largest weights.

This tests whether large-magnitude weights alone account for catastrophic 1-bit failure.

We already have evidence from previous experiments that magnitude sometimes preserved perplexity better than our diagonal Hessian proxy, so this baseline is essential.

---

# IMPORTANCE METHOD 2 — ACTIVATION-AWARE SALIENCY

A weight should matter more when:

- its magnitude is large
AND
- its input feature is heavily used

Use real calibration activations.

For linear layer:

Y = XW

A simple score is:

S_act(i,j) =
|W_ij| × RMS(X_j)

or:

S_act(i,j) =
W_ij² × E[X_j²]

Test both if inexpensive.

The intuition:

A large weight connected to an almost inactive input may matter little.

A moderate weight connected to a heavily active direction may matter enormously.

Rank weights globally or layerwise using this score.

Evaluate using the same protected percentages.

---

# IMPORTANCE METHOD 3 — FUNCTIONAL SENSITIVITY

This is the most important method.

Do not decide importance only from weight statistics.

Measure how much damaging a weight/direction affects the actual model function.

We want an approximation to:

S_func(i,j)
≈
change in model loss / hidden states / logits
if W_ij is quantized badly.

Possible implementation:

Use calibration data and compute the gradient of the teacher/model loss with respect to weights.

Candidate first-order score:

S_grad(i,j) =
|W_ij × grad(W_ij)|

Also test second-order / Fisher-style approximation if feasible:

S_fisher(i,j) =
W_ij² × E[grad(W_ij)²]

If a reliable Hessian approximation is already available, it may be included here as an additional variant, but DO NOT assume diagonal Hessian is correct.

Our previous experiments showed that the diagonal Hessian/activation proxy can disagree strongly with actual perplexity.

Therefore functional impact must ultimately be validated by the oracle restoration test.

---

# OPTIONAL STRONGER FUNCTIONAL TEST

For the most promising layers, directly perturb groups of weights.

For candidate group G:

1. replace those weights with their 1-bit values
2. keep everything else BF16
3. run calibration samples
4. measure:

- layer output deviation
- hidden-state deviation
- logit KL
- loss increase

Define:

S_direct(G)
=
ΔLoss

or:

S_direct(G)
=
KL(logits_FP || logits_perturbed)

This is expensive, so do it only for validation of the ranking methods.

The purpose is to determine whether our importance score is actually correlated with functional destruction.

---

# IMPORTANT: WE NEED BOTH GLOBAL AND LAYERWISE ANALYSIS

A global top 1% may accidentally concentrate almost entirely in several layers.

Therefore evaluate:

## Global ranking

Take top p% across the entire model.

## Layer-balanced ranking

Allocate p% within every layer.

Compare them.

This tells us whether importance is:

- globally concentrated in a few layers
or
- distributed across the whole network.

---

# ALSO ANALYZE PROJECTION TYPES

Report importance separately for:

- Q projection
- K projection
- V projection
- O projection
- MLP gate
- MLP up
- MLP down

We need to know whether catastrophic failure is concentrated in particular projection families.

---

# MOST IMPORTANT EXPERIMENT

For each importance method create this curve:

X axis:

percentage of weights preserved exactly

Y axis:

perplexity

Example:

0% protected → 100,000 PPL
0.1% → ?
0.25% → ?
0.5% → ?
1% → ?
2% → ?
5% → ?
10% → ?

Plot all three methods on the SAME graph.

This graph is the main result of Experiment 6 Part 1.

---

# TARGET

BF16 baseline is approximately:

PPL ≈ 17.9

The final research target is only 5–10% degradation:

approximately:

PPL <= 18.8–19.7

For this oracle experiment we want to discover:

What smallest p gives:

PPL < 100

then:

PPL < 50

then:

PPL < 30

then:

PPL <= ~19.7

These thresholds tell us how difficult Parts 2 and 3 will be.

---

# VERY IMPORTANT QUESTION

Suppose:

Magnitude top 1% → PPL 500

Activation saliency top 1% → PPL 80

Functional sensitivity top 1% → PPL 25

Then we have discovered something extremely valuable:

Functional sensitivity is identifying the genuinely load-bearing information.

Part 2 would then only need to compact approximately that 1%.

---

# OVERLAP ANALYSIS

Compare selected sets.

For top 1% weights compute overlap between:

Magnitude ∩ Activation

Magnitude ∩ Functional

Activation ∩ Functional

and intersection of all three.

Use Jaccard similarity.

This tells us whether all three methods find the same critical weights or completely different populations.

---

# UNION EXPERIMENT

This could be extremely important.

Suppose different methods detect different types of important weights.

Create hybrid selections.

Examples:

top 0.5% magnitude
+
top 0.5% functional

or:

top 0.33% magnitude
+
top 0.33% activation
+
top 0.33% functional

while maintaining approximately the same total protected percentage.

Compare against a single-method top 1%.

This tests whether importance has multiple distinct causes.

---

# FUNCTIONAL COLLAPSE ANALYSIS

Track hidden-state error after every transformer layer.

For each layer l:

E_l =
||H_l_BF16 - H_l_quant||²
/
||H_l_BF16||²

Also calculate cosine similarity.

For each protection method determine:

- where hidden-state divergence starts
- where it suddenly explodes
- which projection causes the jump
- whether protecting the selected weights prevents that collapse

This is crucial.

We don't merely want a lower final PPL.

We want to understand WHY the selected weights prevent model collapse.

---

# IMPORTANCE QUALITY METRIC

A good importance algorithm should achieve the largest functional recovery per protected weight.

Define approximately:

Recovery(p)
=
(PPL_1bit - PPL_protected(p))
/
p

But report raw PPL too.

Do not hide results behind one summary metric.

---

# NO METADATA CONSTRAINT YET

For this experiment ignore final storage cost of the protected BF16 values.

Again:

This is an ORACLE experiment.

Part 1 asks only:

WHICH information must survive?

After we know that, Part 2 asks:

HOW do we compact it?

Then Part 3 asks:

HOW do we encode it cheaply?

Do not mix these questions.

---

# REQUIRED RESULTS TABLE

For every selector and every p report:

Method
Protected %
Number of protected weights
Final perplexity
PPL degradation vs BF16
Hidden-state error
Logit KL if measured
Layer of first major collapse
Projection responsible for largest collapse

---

# REQUIRED SUMMARY

At the end answer exactly:

Q1. Which of the three importance methods is strongest?

Q2. What is the minimum percentage of weights needed for PPL <100?

Q3. What is needed for PPL <50?

Q4. What is needed for PPL <30?

Q5. What is needed for <=10% BF16 perplexity degradation?

Q6. Are important weights concentrated in certain layers?

Q7. Are they concentrated in certain projection types?

Q8. How much overlap exists between magnitude, activation saliency and functional sensitivity?

Q9. Does combining selectors outperform each selector individually?

Q10. At what layer does catastrophic model collapse happen when the important weights are NOT preserved?

---

# DECISION RULE

Do NOT proceed to Part 2 yet.

We stay on IMPORTANCE until we have a selector that is extremely convincing.

Ideal outcome:

<=1–3% of perfectly preserved weights can return the model close to BF16.

If that happens, Part 2 becomes:

“How do we turn that 1–3% scattered critical information into a compact structured representation?”

If even protecting 10% of supposedly important weights leaves perplexity catastrophic, then our importance algorithm is insufficient or the hypothesis that a small critical subset controls collapse is wrong.

In that case improve importance detection before touching compactness.

# FINAL RESEARCH PHILOSOPHY

The three-stage project is:

IMPORTANCE
→ find the minimum information the model cannot afford to lose

COMPACTNESS
→ reorganize that information into a small structured region

ENCODING
→ store that region + bulk weights near 1 BPW

For now:

SOLVE IMPORTANCE ONLY.

Run the three selector families:

1. magnitude
2. activation-aware saliency
3. functional sensitivity

and determine which one recovers the largest amount of model function using the smallest percentage of protected weights.