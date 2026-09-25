You are continuing the 1-bit quantization research project.

Previous experiments established:

- Hadamard preprocessing consistently helps.
- DSA is too weak after Hadamard.
- Seeded residual search failed.
- Low-rank residual reconstruction helps, but the residual is only weakly low-rank.
- Hessian-guided LRR beats ordinary LRR at the same storage.
- Sparse rescue helps, but even high BPW still gives very poor perplexity.
- Experiment 4 showed that repairing a bad 1-bit base afterward is not sufficient.
- The diagonal Hessian/activation-importance proxy is also not reliable enough to predict actual perplexity.
- Therefore the main bottleneck is now believed to be the binary base itself.

The new hypothesis is:

We should not quantize each weight matrix independently according to weight-space reconstruction error.

Instead, every layer should choose its binary representation based on the ACTUAL quantized activations produced by all previous layers.

The quantizer should preserve the transformation performed by the model:

XW

rather than merely preserving:

W

The new method is called:

Sequential Functional Binary Search — SFBS

---

# Core idea

For layer l:

The original FP16/BF16 model receives:

X_l_FP

But during real 1-bit inference, layer l receives:

X_l_Q

where X_l_Q already contains quantization errors created by layers:

0 ... l-1

Therefore candidate binary weights must be evaluated using:

X_l_Q

not ideal FP16 activations.

We want:

X_l_Q * W_l_hat

to produce a useful transformation given the already distorted input.

Conceptually:

W_l_hat* =
argmin over binary candidates
FunctionalLoss(W_l_hat ; X_l_Q)

After choosing the layer's binary base, propagate its output forward.

Then optimize the next layer using the NEW quantized activations.

This allows later layers to partially compensate for errors created earlier.

---

# Main experimental goal

We are NOT trying to reach final 5–10% perplexity degradation in this experiment yet.

The immediate milestone is:

Can a much stronger FUNCTION-AWARE 1-bit base move perplexity from the current ~5,000–6,000 range toward:

PPL < 100

while remaining close to:

1.00–1.05 BPW

No LRR or BF16 salient rescue should be used initially.

We first need to determine how strong the binary base itself can become.

---

# Scope

Start with the same setup as previous experiments:

- Qwen2.5-0.5B
- WikiText-2
- same calibration split
- same evaluation split
- same seed
- same evaluation implementation

Unlike earlier representative-layer experiments, this experiment should preferably quantize ALL transformer layers sequentially because the central hypothesis concerns accumulated error.

If full-model search is too expensive, first implement and debug on a smaller subset, then run the full model.

Always report clearly which version was used.

---

# BASELINE CONFIGURATIONS

At minimum compare:

1. BF16
2. standard naive 1-bit
3. Hadamard + naive 1-bit
4. independently calibrated functional 1-bit
5. sequential functional 1-bit
6. sequential functional 1-bit + beam search

Do not add LRR, sparse rescue, or high-precision corrections yet.

---

# PART 1 — Hadamard preprocessing

Use the strongest Hadamard setup already validated by previous experiments.

Do not introduce unnecessary new transforms in the first run.

The transformed matrix enters binary candidate generation.

Metadata must be counted exactly.

---

# PART 2 — Binary candidate generation

For each quantized matrix/layer generate K possible binary representations.

Start with:

K = 4
K = 8
K = 16

Candidates should differ meaningfully.

Do NOT simply generate random scales around one solution.

Candidate families can include:

## Candidate family A — scale search

Binary reconstruction:

W_hat = alpha * sign(W)

Search several alpha values around:

mean(|W|)
median(|W|)
weighted mean magnitude
least-squares optimal alpha

For fixed sign map, the raw-MSE optimal scale is:

alpha =
sum |W_i| / N

but functional calibration may prefer a different value.

Generate values such as:

0.7 alpha
0.85 alpha
1.0 alpha
1.15 alpha
1.3 alpha

Do not store candidate parameters independently if they can be reproduced from a small candidate index.

---

## Candidate family B — row/column compensated binary

Allow extremely cheap shared row or column scaling if metadata stays near budget.

Examples:

W_hat = diag(a) * sign(W)

or

W_hat = sign(W) * diag(b)

But charge every scale exactly.

Only keep this family if its quality improvement justifies BPW.

---

## Candidate family C — binary assignment adjustment

Instead of forcing:

sign(W)

for every coordinate, allow limited sign flips if functional loss improves.

Do NOT allow arbitrary independent flips that require another bit per weight.

Use structured or reproducible changes such as:

- row-level flip
- column-level flip
- block-level flip
- deterministic candidate masks generated from a seed

The selected candidate index should be very cheap.

The goal is to create multiple plausible binary bases without storing a second full bit-map.

---

# PART 3 — Functional loss

Do not rank candidates primarily by weight MSE.

For each candidate, evaluate its effect on real calibration activations.

At minimum compute:

## Loss A — layer output matching

For current quantized input:

X_Q

Original layer output target:

Y_target = Layer_FP(X_Q)

Candidate output:

Y_candidate = Layer_Q_candidate(X_Q)

Then:

L_layer =
mean squared error(Y_target, Y_candidate)

Important:

Use the SAME incoming quantized X_Q for both target and candidate.

This isolates how well the candidate preserves the desired transformation given the actual distorted state.

---

## Loss B — hidden-state horizon matching

A candidate that looks best immediately may damage later layers.

Therefore also evaluate a short downstream horizon.

Test:

horizon = 1 transformer block
horizon = 2 blocks
horizon = 4 blocks

For candidate C:

Run the candidate plus the next h blocks.

Compare:

H_target_(l+h)

versus

H_candidate_(l+h)

using:

L_hidden =
normalized MSE
or
1 - cosine similarity

Use FP/BF16 weights in the temporary downstream horizon initially, except for layers already permanently quantized before l.

This isolates the future effect of the current candidate.

---

## Loss C — optional logit/KL loss

For selected experiments, test whether final or partial-model logits give a better signal.

Possible metric:

KL(
softmax(logits_FP / T)
||
softmax(logits_Q / T)
)

or cross-entropy against teacher logits.

This is more expensive.

Do not use it for every candidate in the first implementation unless runtime permits.

First establish whether layer output + short-horizon loss works.

---

# PART 4 — Sequential calibration

This is the central experiment.

Start:

X_0_Q = model input activations

For layer 0:

generate candidates
score each using functional loss
select candidate
replace layer 0 weights
propagate calibration activations
save resulting X_1_Q

For layer 1:

generate candidates using X_1_Q
score them
select candidate
propagate

Continue through every layer.

Therefore:

Layer l is calibrated on the exact activations produced by already quantized layers 0 ... l-1.

DO NOT recalculate it using clean FP16 activations.

That would defeat the hypothesis.

---

# PART 5 — Compare independent vs sequential calibration

This comparison is critical.

## Independent functional calibration

For every layer use clean/full-precision incoming activations.

Choose the best binary candidate locally.

Then quantize the full model using those independently selected candidates.

## Sequential functional calibration

Use accumulated quantized activations.

Compare final perplexity.

This directly answers whether accumulated-error awareness matters.

---

# PART 6 — Greedy versus beam search

Greedy selection may make a locally optimal but globally bad decision.

Therefore test beam search.

Start with:

beam width B = 2
B = 4

Do not start larger.

At layer l:

Each current beam path has:

- selected candidate history
- current quantized activation state
- accumulated functional loss

Expand each path using K candidates.

Total temporary paths:

B * K

Score them.

Keep only the best B.

Proceed to the next layer.

Example:

B=4
K=8

32 candidate paths evaluated at each step
keep best 4

Do not enumerate K^L combinations.

---

# BEAM SCORE

Do not use accumulated raw weight MSE.

Use:

Score =
lambda1 * current_layer_output_loss
+
lambda2 * downstream_hidden_loss
+
lambda3 * accumulated_drift

Initially test simple choices:

Configuration A:
lambda1 = 1
lambda2 = 0

Configuration B:
lambda1 = 1
lambda2 = 1

Configuration C:
lambda1 = 0.5
lambda2 = 1

Normalize losses before combining so scale differences do not dominate.

---

# PART 7 — Teacher trajectory

Record the BF16 teacher hidden states on the calibration data:

H_0_FP
H_1_FP
...
H_L_FP

But be careful:

There are TWO useful targets.

Target A:

Original teacher trajectory:

H_l_FP

Target B:

What the FP16 current layer would output if fed the ACTUAL quantized input:

Layer_FP(X_l_Q)

Test both.

This distinction is important.

Target A asks:

"Return the model to the clean BF16 trajectory."

Target B asks:

"Given the distorted state we currently have, preserve what this layer should do."

Compare them experimentally.

Do not assume one is superior.

---

# PART 8 — Functional candidate selection versus MSE

For every selected candidate record:

- weight MSE rank
- layer-output-loss rank
- hidden-horizon-loss rank
- actual resulting perplexity contribution if measurable

We need examples where:

Candidate A:
lower weight MSE

Candidate B:
higher weight MSE
but better functional loss

and B produces better downstream behavior.

That would validate the central thesis.

---

# METADATA ACCOUNTING

This experiment targets an approximately 1-bit base.

For every candidate system count:

- sign bits
- block scales
- candidate IDs
- row/column scales
- routing metadata
- seed metadata
- any additional mask/index cost

Report:

true BPW

not theoretical binary BPW.

Candidate index cost:

ceil(log2(K))

bits per unit where the candidate is selected.

Clearly state whether candidate selection is per:

- matrix
- row
- column
- block
- layer

Prefer coarse selection because metadata must stay tiny.

---

# STORAGE TARGET

Primary target:

<= 1.05 BPW

Secondary diagnostic limit:

<= 1.10 BPW

Do not add expensive mechanisms simply to improve perplexity.

We need to know how powerful the functional base can be under a nearly pure 1-bit budget.

---

# REQUIRED ABLATIONS

Run these in order.

## Ablation A

Hadamard naive 1-bit.

## Ablation B

Hadamard + independent functional scale search.

## Ablation C

Hadamard + sequential functional scale search.

## Ablation D

Sequential functional search with K:

4
8
16

## Ablation E

Greedy versus beam:

B=1
B=2
B=4

## Ablation F

Loss horizon:

local only
1 block
2 blocks
4 blocks

## Ablation G

Target:

clean teacher hidden state
versus
FP current-layer output on quantized input

---

# IMPORTANT METRICS

For every configuration report:

1. perplexity
2. perplexity degradation from BF16
3. true BPW
4. layer-output functional MSE
5. hidden-state cosine similarity
6. hidden-state normalized MSE
7. logit KL if tested
8. weight MSE
9. number of candidate evaluations
10. calibration runtime
11. candidate metadata
12. selected candidate distribution across layers

---

# IMPORTANT PLOTS

Generate:

## Plot 1
Perplexity vs BPW.

## Plot 2
Hidden-state drift vs layer index.

X axis:
layer number

Y axis:
distance from BF16 teacher trajectory.

Compare:

naive binary
independent functional
sequential functional
beam search

This is one of the most important plots.

## Plot 3
Functional layer error vs layer index.

## Plot 4
Perplexity vs candidate count K.

## Plot 5
Perplexity vs beam width.

## Plot 6
Perplexity vs downstream horizon.

## Plot 7
Weight MSE versus functional loss for candidates.

We want to see whether these objectives disagree.

---

# CRITICAL QUESTIONS

Q1:

Does functional candidate selection outperform raw weight-MSE selection?

Q2:

Does sequential calibration outperform independent layer calibration?

Q3:

Does error accumulation become smaller across depth?

Q4:

Does beam search meaningfully outperform greedy search?

Q5:

What horizon gives the best quality/compute tradeoff?

Q6:

Can the base itself reach PPL < 100 under <=1.05 BPW?

Q7:

If not, how far does it move compared with Hadamard naive binary?

Q8:

Which layers cause the largest sudden hidden-state drift?

Q9:

Do later layers naturally compensate for earlier quantization error when calibrated sequentially?

---

# VERY IMPORTANT DIAGNOSTIC

Track the hidden-state error after EVERY transformer layer.

For each layer calculate:

E_l =
||H_l_FP - H_l_Q||^2
/
||H_l_FP||^2

Also cosine similarity.

We need to identify exactly where the model "falls off the cliff."

Example:

Layer 0: small error
Layer 1: small
Layer 2: small
...
Layer 9: moderate
Layer 10: huge jump
Layer 11+: destroyed

If this happens, report which projection inside that layer causes the jump.

This may identify a much smaller set of matrices requiring special treatment later.

---

# FAILURE CONDITIONS

Do not force SFBS to succeed.

Possible outcomes:

## Failure A

Sequential functional calibration barely beats independent calibration.

Conclusion:

Error accumulation awareness is not the main bottleneck.

## Failure B

Local functional loss improves dramatically but perplexity remains huge.

Conclusion:

Local layer matching does not capture long-range model function.

Increase horizon or use teacher-logit information.

## Failure C

Beam search gives little improvement over greedy.

Conclusion:

Candidate interactions across neighboring layers are weak or candidate family is too limited.

## Failure D

Even strong functional calibration stays above PPL 1000.

Conclusion:

The binary representation family itself is too restrictive.

Do not then add endless repair mechanisms.

The next research problem would be designing a richer binary representation.

---

# SUCCESS CRITERIA

Strong success:

PPL < 100
at <=1.05 BPW.

Major partial success:

PPL < 500
at <=1.05 BPW.

Useful directional success:

At least 5–10x perplexity reduction versus Hadamard naive binary at approximately equal BPW.

Failure:

Only small improvement such as:

6500 -> 5000

This would not justify the complexity.

---

# FINAL OUTPUT

Produce:

1. full implementation
2. aggregate results
3. per-layer drift table
4. per-projection diagnostics
5. independent vs sequential comparison
6. greedy vs beam comparison
7. K ablation
8. horizon ablation
9. exact BPW accounting
10. all requested plots
11. short scientific conclusion

At the end answer clearly:

Q1: Did functional optimization beat weight-space optimization?

Q2: Did sequential calibration reduce accumulated distortion?

Q3: Did beam search help?

Q4: What was the best PPL at <=1.05 BPW?

Q5: Did we reach PPL <100?

Q6: If not, exactly where in the transformer does functional information begin collapsing?

Q7: Is the next bottleneck the search algorithm, the candidate family, or the fundamental 1-bit representation?

Run Experiment 5 now.

Do not add LRR, DSA, sparse BF16 rescue, or seeded residual search during the primary experiment.

The purpose of Experiment 5 is to determine the maximum quality achievable from a nearly pure 1-bit base when the binary weights are chosen according to actual model function rather than static weight reconstruction.