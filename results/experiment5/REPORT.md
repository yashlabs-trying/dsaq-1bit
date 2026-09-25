# Experiment 5: Sequential Functional Binary Search

## Scope

All 24 Qwen2.5-0.5B transformer blocks and all 168 Q/K/V/O/gate/up/down matrices were quantized. Embeddings, normalization, and LM head remain BF16. Candidates use one reproducible scale-compensation pattern per transformer layer; storage includes one sign per target weight, FP16 block scales, and `ceil(log2(K))` candidate-ID bits per layer. No LRR, DSA, sparse rescue, or high-precision correction is present.

## Results

| configuration             |      perplexity |   mean_nll |   true_bpw |   k |   beam |   horizon | target   |   candidate_evaluations |   runtime_seconds |
|:--------------------------|----------------:|-----------:|-----------:|----:|-------:|----------:|:---------|------------------------:|------------------:|
| bf16                      |     17.9038     |    2.88501 |   16       | nan |    nan |       nan | none     |                       0 |       0           |
| sequential_k8_h1_clean    |  96086.5        |   11.473   |    1.00098 |   8 |      1 |         1 | clean    |                     192 |      18.7012      |
| sequential_k8_h2_current  | 158737          |   11.975   |    1.00098 |   8 |      1 |         2 | current  |                     192 |      22.698       |
| sequential_k8_h4_current  | 269723          |   12.5052  |    1.00098 |   8 |      1 |         4 | current  |                     192 |      27.5049      |
| sequential_k8_h0_current  | 286008          |   12.5638  |    1.00098 |   8 |      1 |         0 | current  |                     192 |      15.8051      |
| beam2_k8_h1_l05_1         | 302743          |   12.6206  |    1.00098 |   8 |      2 |         1 | current  |                     376 |      37.6163      |
| sequential_k16_h0_current | 308269          |   12.6387  |    1.00098 |  16 |      1 |         0 | current  |                     384 |      28.4926      |
| beam4_k8_h1_l1_1          | 321564          |   12.681   |    1.00098 |   8 |      4 |         1 | current  |                     744 |      76.1891      |
| beam2_k8_h1_l1_1          | 364268          |   12.8056  |    1.00098 |   8 |      2 |         1 | current  |                     376 |      38.2025      |
| sequential_k8_h1_current  | 509349          |   13.1409  |    1.00098 |   8 |      1 |         1 | current  |                     192 |      19.7031      |
| hadamard_naive            | 644467          |   13.3762  |    1.00098 |   1 |      1 |         0 | weight   |                       0 |       9.53674e-07 |
| sequential_k4_h0_current  | 681592          |   13.4322  |    1.00098 |   4 |      1 |         0 | current  |                      96 |       7.29683     |
| beam2_k8_h1_l1_0          | 731436          |   13.5028  |    1.00098 |   8 |      2 |         1 | current  |                     376 |      38.5101      |
| independent_k8            | 746760          |   13.5235  |    1.00098 |   8 |      1 |         0 | clean    |                     192 |      15.2945      |
| standard_naive            |      1.9921e+06 |   14.5047  |    1.00098 |   1 |      1 |         0 | weight   |                       0 |       2.38419e-07 |

Best result at <=1.05 BPW: **sequential_k8_h1_clean**, PPL **9.609e+04**, true BPW **1.00097676**. Hadamard naive PPL: **6.445e+05**.

## Answers

**Q1: Did functional optimization beat weight-space optimization? YES.** Best functional PPL 9.609e+04 versus Hadamard naive 6.445e+05.

**Q2: Did sequential calibration reduce accumulated distortion? YES.** Independent K=8 PPL 7.468e+05; sequential K=8 local/current-target PPL 2.86e+05.

**Q3: Did beam search help? NO.** Best beam result: beam2_k8_h1_l05_1, PPL 3.027e+05.

**Q4: What was the best PPL at <=1.05 BPW? 9.609e+04.**

**Q5: Did we reach PPL <100? NO.**

**Q6: Where does functional information begin collapsing?** The largest positive teacher-drift jump for the best configuration occurs at transformer layer **0**: NMSE changes by 0.735 to 0.735. Projection rescue diagnostics for that layer are in `per_projection.csv`.

**Q7: Next bottleneck:** the candidate family / fundamental one-bit representation, if horizon and beam ablations do not materially close the gap; the tables distinguish this from search failure. No repair mechanism is introduced.

The experiment uses calibration hidden states to select candidates and the unchanged WikiText-2 test split for final perplexity. Candidate patterns are coarse layer-level choices, which keeps metadata negligible but bounds the claim about richer per-matrix binary families.
