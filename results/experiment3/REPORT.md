# Experiment 3: low-rank residual reconstruction

## Scope

Qwen2.5-0.5B layers 0, 12, and 23; all Q/K/V/O and MLP gate/up/down projections; WikiText-2; seed 20260925; 128x128 Hadamard blocks. Every low-rank factor is rounded to FP16 and U, S, and V storage is charged exactly.

## Residual energy

|   rank |   mean_cumulative_energy |
|-------:|-------------------------:|
|      1 |                0.0461158 |
|      2 |                0.0694944 |
|      4 |                0.100565  |
|      8 |                0.144567  |
|     16 |                0.207102  |

Mean rank-8 captured energy is **14.46%**. The mean top-10%-column share of sensitivity-weighted residual damage is **62.67%**.

## Aggregate reconstruction results

| configuration        |   r1 |   r2 | stage2_method   |   true_bpw |   relative_error |   weighted_error |   cosine |
|:---------------------|-----:|-----:|:----------------|-----------:|-----------------:|-----------------:|---------:|
| hadamard_naive       |    0 |    0 | none            |    1.00098 |         0.390532 |         0.449183 | 0.780319 |
| naive                |    0 |    0 | none            |    1.00098 |         0.409787 |         0.525555 | 0.767793 |
| rank1                |    1 |    0 | none            |    1.02557 |         0.380927 |         0.413576 | 0.786744 |
| rank1_hessian_rank1  |    1 |    1 | hessian         |    1.05017 |         0.37924  |         0.372139 | 0.787849 |
| rank1_ordinary_rank1 |    1 |    1 | ordinary        |    1.05017 |         0.377553 |         0.408653 | 0.788942 |
| rank2                |    2 |    0 | none            |    1.05017 |         0.377553 |         0.408653 | 0.788942 |
| rank2_hessian_rank1  |    2 |    1 | hessian         |    1.07476 |         0.376438 |         0.368694 | 0.789659 |
| rank2_hessian_rank2  |    2 |    2 | hessian         |    1.09936 |         0.375256 |         0.350932 | 0.790415 |
| rank2_ordinary_rank1 |    2 |    1 | ordinary        |    1.07476 |         0.375066 |         0.405337 | 0.790546 |
| rank2_ordinary_rank2 |    2 |    2 | ordinary        |    1.09936 |         0.372783 |         0.402991 | 0.792011 |
| rank4                |    4 |    0 | none            |    1.09936 |         0.372783 |         0.402991 | 0.792011 |
| rank4_hessian_rank1  |    4 |    1 | hessian         |    1.12395 |         0.371929 |         0.363856 | 0.792552 |
| rank4_hessian_rank2  |    4 |    2 | hessian         |    1.14855 |         0.37103  |         0.34679  | 0.793123 |
| rank4_hessian_rank4  |    4 |    4 | hessian         |    1.19774 |         0.369199 |         0.327637 | 0.794283 |
| rank4_ordinary_rank1 |    4 |    1 | ordinary        |    1.12395 |         0.370792 |         0.400861 | 0.793286 |
| rank4_ordinary_rank2 |    4 |    2 | ordinary        |    1.14855 |         0.368884 |         0.398895 | 0.794508 |
| rank4_ordinary_rank4 |    4 |    4 | ordinary        |    1.19774 |         0.365342 |         0.395214 | 0.796768 |
| rank8                |    8 |    0 | none            |    1.19774 |         0.365342 |         0.395214 | 0.796768 |
| rank8_hessian_rank2  |    8 |    2 | hessian         |    1.24693 |         0.363791 |         0.34012  | 0.797741 |
| rank8_hessian_rank4  |    8 |    4 | hessian         |    1.29612 |         0.362233 |         0.321579 | 0.798722 |
| rank8_ordinary_rank2 |    8 |    2 | ordinary        |    1.24693 |         0.362052 |         0.391795 | 0.798862 |
| rank8_ordinary_rank4 |    8 |    4 | ordinary        |    1.29612 |         0.358979 |         0.388043 | 0.800812 |

## Perplexity and storage

| configuration        |   r1 |   r2 | stage2_method   |   perplexity |   delta_bf16 |     raw_mse |   relative_error |   weighted_error |   cosine |   true_bpw |   added_metadata_bits |   recovery_from_hadamard |   recovery_fraction |   recovery_per_added_metadata_bit |
|:---------------------|-----:|-----:|:----------------|-------------:|-------------:|------------:|-----------------:|-----------------:|---------:|-----------:|----------------------:|-------------------------:|--------------------:|----------------------------------:|
| bf16                 |    0 |    0 | none            |      17.8982 |         0    | 0           |         0        |         0        | 1        |   16       |                     0 |                  6530.48 |            1        |                     nan           |
| naive                |    0 |    0 | none            |   57151.2    |     57133.3  | 0.000217342 |         0.409787 |         0.525555 | 0.767793 |    1.00098 |                     0 |                -50602.8  |           -7.74871  |                     nan           |
| hadamard_naive       |    0 |    0 | none            |    6548.38   |      6530.48 | 0.000207951 |         0.390532 |         0.449183 | 0.780319 |    1.00098 |                     0 |                     0    |            0        |                     nan           |
| rank1                |    1 |    0 | none            |    3946.37   |      3928.47 | 0.00018688  |         0.380927 |         0.413576 | 0.786744 |    1.02557 |               1100112 |                  2602.01 |            0.398441 |                       0.00236523  |
| rank2                |    2 |    0 | none            |    4234.68   |      4216.78 | 0.00018103  |         0.377553 |         0.408653 | 0.788942 |    1.05017 |               2200224 |                  2313.71 |            0.354293 |                       0.00105158  |
| rank4                |    4 |    0 | none            |    4330.97   |      4313.07 | 0.00017435  |         0.372783 |         0.402991 | 0.792011 |    1.09936 |               4400448 |                  2217.41 |            0.339548 |                       0.000503906 |
| rank8                |    8 |    0 | none            |    3485.92   |      3468.02 | 0.000167266 |         0.365342 |         0.395214 | 0.796768 |    1.19774 |               8800896 |                  3062.46 |            0.468949 |                       0.000347972 |
| rank1_ordinary_rank1 |    1 |    1 | ordinary        |    4257.36   |      4239.46 | 0.00018103  |         0.377553 |         0.408653 | 0.788942 |    1.05017 |               2200224 |                  2291.02 |            0.35082  |                       0.00104127  |
| rank1_hessian_rank1  |    1 |    1 | hessian         |    2469.77   |      2451.87 | 0.000182818 |         0.37924  |         0.372139 | 0.787849 |    1.05017 |               2200224 |                  4078.61 |            0.62455  |                       0.00185373  |
| rank2_ordinary_rank1 |    2 |    1 | ordinary        |    4144.74   |      4126.84 | 0.000177417 |         0.375066 |         0.405337 | 0.790546 |    1.07476 |               3300336 |                  2403.64 |            0.368065 |                       0.000728302 |
| rank2_hessian_rank1  |    2 |    1 | hessian         |    1743.65   |      1725.75 | 0.000179159 |         0.376438 |         0.368694 | 0.789659 |    1.07476 |               3300336 |                  4804.73 |            0.735739 |                       0.00145583  |
| rank2_ordinary_rank2 |    2 |    2 | ordinary        |    4360.22   |      4342.32 | 0.00017435  |         0.372783 |         0.402991 | 0.792011 |    1.09936 |               4400448 |                  2188.16 |            0.335068 |                       0.000497258 |
| rank2_hessian_rank2  |    2 |    2 | hessian         |    1586.41   |      1568.51 | 0.000177037 |         0.375256 |         0.350932 | 0.790415 |    1.09936 |               4400448 |                  4961.97 |            0.759817 |                       0.00112761  |
| rank4_ordinary_rank1 |    4 |    1 | ordinary        |    3552.17   |      3534.27 | 0.000172251 |         0.370792 |         0.400861 | 0.793286 |    1.12395 |               5500560 |                  2996.21 |            0.458804 |                       0.00054471  |
| rank4_hessian_rank1  |    4 |    1 | hessian         |    1674.87   |      1656.97 | 0.000173448 |         0.371929 |         0.363856 | 0.792552 |    1.12395 |               5500560 |                  4873.51 |            0.746271 |                       0.000886003 |
| rank4_ordinary_rank2 |    4 |    2 | ordinary        |    3721.29   |      3703.39 | 0.000170371 |         0.368884 |         0.398895 | 0.794508 |    1.14855 |               6600672 |                  2827.09 |            0.432907 |                       0.000428304 |
| rank4_hessian_rank2  |    4 |    2 | hessian         |    1387.21   |      1369.31 | 0.000172544 |         0.37103  |         0.34679  | 0.793123 |    1.14855 |               6600672 |                  5161.17 |            0.79032  |                       0.000781916 |
| rank4_ordinary_rank4 |    4 |    4 | ordinary        |    3495.18   |      3477.28 | 0.000167266 |         0.365342 |         0.395214 | 0.796768 |    1.19774 |               8800896 |                  3053.2  |            0.467531 |                       0.000346919 |
| rank4_hessian_rank4  |    4 |    4 | hessian         |    1249.43   |      1231.53 | 0.000170575 |         0.369199 |         0.327637 | 0.794283 |    1.19774 |               8800896 |                  5298.95 |            0.811418 |                       0.000602092 |
| rank8_ordinary_rank2 |    8 |    2 | ordinary        |    3533.6    |      3515.7  | 0.000164693 |         0.362052 |         0.391795 | 0.798862 |    1.24693 |              11001120 |                  3014.78 |            0.461647 |                       0.000274043 |
| rank8_hessian_rank2  |    8 |    2 | hessian         |    1321.7    |      1303.8  | 0.00016599  |         0.363791 |         0.34012  | 0.797741 |    1.24693 |              11001120 |                  5226.68 |            0.800351 |                       0.000475104 |
| rank8_ordinary_rank4 |    8 |    4 | ordinary        |    3471.15   |      3453.25 | 0.000162474 |         0.358979 |         0.388043 | 0.800812 |    1.29612 |              13201344 |                  3077.23 |            0.47121  |                       0.0002331   |
| rank8_hessian_rank4  |    8 |    4 | hessian         |    1072.61   |      1054.71 | 0.000164698 |         0.362233 |         0.321579 | 0.798722 |    1.29612 |              13201344 |                  5475.77 |            0.838494 |                       0.000414789 |

Best quality: **rank8_hessian_rank4**, perplexity **1072.6081**, true BPW **1.296122**, recovery of the Hadamard-1-bit excess perplexity **83.85%**.

Best measured perplexity recovery per added metadata bit: **rank1** (0.00236523 perplexity points/bit).

Hessian-guided stage 2 beats ordinary stage 2 at equal ranks in **100.0%** of tested pairs.

## Answers

**Q1: Is the 1-bit residual low-rank enough to exploit? NO** under the preregistered practical rule that rank 8 captures at least 20% of residual energy.

**Q2: Does first LRR materially recover model quality? YES** under the rule that it recovers at least 25% of the Hadamard-1-bit excess perplexity.

**Q3: Does Hessian-guided second LRR recover more quality per metadata bit than ordinary LRR? YES** under the rule that it wins at equal storage in at least five of eight rank pairs with a positive median perplexity advantage.

**Q4: Should LRR replace DSA as the main reconstruction mechanism? YES** The best configuration at or below 1.05 BPW is **rank1** with perplexity **3946.3683**, compared with Experiment 1 Hadamard+DSA at **6482.0991** and 1.006040 BPW.

These conclusions apply to this representative-layer intervention and the tested FP16 factorization ranks. They are not whole-model compression claims.
