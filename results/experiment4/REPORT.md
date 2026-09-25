# Experiment 4: sensitivity-directed binary, low-rank, and sparse rescue

## Result

BF16 perplexity is **17.8982**. The best compressed configuration is **sparse_magnitude_p5** at perplexity **489.2310** and **3.100977 BPW**. The 5–10% target (perplexity <= 19.6880) was **not reached**.

The top 1% of individual residual weights contains **49.97%** of weighted damage after the selected Hessian-aware base.

## Principal configurations

| configuration                 | family         |   perplexity |   perplexity_degradation_pct |   weighted_error |   true_bpw |   binary_metadata_bpw |   lrr_metadata_bpw |   salient_rescue_bpw |   fraction_weights_rescued |   damage_removed_fraction |
|:------------------------------|:---------------|-------------:|-----------------------------:|-----------------:|-----------:|----------------------:|-------------------:|---------------------:|---------------------------:|--------------------------:|
| hadamard_hessian_b128         | hessian_base   |    5754.58   |                     32051.8  |         0.429903 |    1.00098 |           0.000976562 |          0         |           0          |                0           |                 0         |
| sparse_hessian_damage_p0      | sparse         |    5754.58   |                     32051.8  |         0.429903 |    1.00098 |           0.000976562 |          0         |           0          |                0           |                 0         |
| sparse_magnitude_p0           | sparse         |    5754.58   |                     32051.8  |         0.429903 |    1.00098 |           0.000976562 |          0         |           0          |                0           |                 0         |
| sparse_residual_p0            | sparse         |    5754.58   |                     32051.8  |         0.429903 |    1.00098 |           0.000976562 |          0         |           0          |                0           |                 0         |
| sparse_residual_squared_p0    | sparse         |    5754.58   |                     32051.8  |         0.429903 |    1.00098 |           0.000976562 |          0         |           0          |                0           |                 0         |
| hadamard_naive_b128           | naive_base     |    6469.95   |                     36048.6  |         0.448959 |    1.00098 |           0.000976562 |          0         |           0          |                0           |                 0         |
| hadamard_magnitude_b128       | magnitude_base |    6477.97   |                     36093.5  |         0.44896  |    1.00098 |           0.000976562 |          0         |           0          |                0           |                 0         |
| hadamard_hessian_b64          | hessian_base   |    6049.74   |                     33700.9  |         0.424562 |    1.00391 |           0.00390625  |          0         |           0          |                0           |                 0         |
| hadamard_magnitude_b64        | magnitude_base |    8506.71   |                     47428.4  |         0.448483 |    1.00391 |           0.00390625  |          0         |           0          |                0           |                 0         |
| sparse_hessian_damage_p0.01   | sparse         |    4608.25   |                     25647    |         0.380513 |    1.00518 |           0.000976562 |          0         |           0.00420087 |                0.000100004 |                 0.114885  |
| sparse_residual_p0.01         | sparse         |    4847.52   |                     26983.9  |         0.426227 |    1.00518 |           0.000976562 |          0         |           0.00420087 |                0.000100004 |                 0.008549  |
| sparse_residual_squared_p0.01 | sparse         |    4847.52   |                     26983.9  |         0.426227 |    1.00518 |           0.000976562 |          0         |           0.00420087 |                0.000100004 |                 0.008549  |
| sparse_magnitude_p0.01        | sparse         |    6236.6    |                     34744.9  |         0.424812 |    1.00518 |           0.000976562 |          0         |           0.00420087 |                0.000100004 |                 0.0118408 |
| hadamard_magnitude_row        | magnitude_base |    3885.43   |                     21608.5  |         0.416152 |    1.0136  |           0.0135989   |          0         |           0          |                0           |                 0         |
| hadamard_hessian_row          | hessian_base   |    7264.8    |                     40489.6  |         0.377668 |    1.0136  |           0.0135989   |          0         |           0          |                0           |                 0         |
| structured_block16_p0.1       | structured     |    5304.82   |                     29538.9  |         0.406086 |    1.01708 |           0.000976562 |          0         |           0.0161006  |                0.0010016   |                 0.0553994 |
| structured_row_p0.1           | structured     |    4316.75   |                     24018.4  |         0.415553 |    1.01728 |           0.000976562 |          0         |           0.0163048  |                0.00101877  |                 0.0333777 |
| structured_block8_p0.1        | structured     |    5196.1    |                     28931.5  |         0.395394 |    1.01731 |           0.000976562 |          0         |           0.0163345  |                0.00100017  |                 0.080271  |
| structured_column_p0.1        | structured     |    5084.11   |                     28305.7  |         0.373377 |    1.01792 |           0.000976562 |          0         |           0.0169467  |                0.00105884  |                 0.131485  |
| sparse_hessian_damage_p0.05   | sparse         |    4658.5    |                     25927.8  |         0.341661 |    1.02198 |           0.000976562 |          0         |           0.0210015  |                0.000500019 |                 0.205258  |
| sparse_residual_p0.05         | sparse         |    5695.34   |                     31720.8  |         0.417989 |    1.02198 |           0.000976562 |          0         |           0.0210015  |                0.000500019 |                 0.027713  |
| sparse_residual_squared_p0.05 | sparse         |    5695.34   |                     31720.8  |         0.417989 |    1.02198 |           0.000976562 |          0         |           0.0210015  |                0.000500019 |                 0.027713  |
| sparse_magnitude_p0.05        | sparse         |    6436.62   |                     35862.4  |         0.418188 |    1.02198 |           0.000976562 |          0         |           0.0210015  |                0.000500019 |                 0.02725   |
| global_hessian_rank1          | global_lrr     |    5359.81   |                     29846.1  |         0.378658 |    1.02557 |           0.000976562 |          0.0245954 |           0          |                0           |                 0         |
| adaptive_lrr_b0.025           | adaptive_lrr   |    7063.63   |                     39365.7  |         0.363514 |    1.0258  |           0.000976562 |          0.0248229 |           0          |                0           |                 0         |
| sparse_hessian_damage_p0.1    | sparse         |    5097.76   |                     28382    |         0.317911 |    1.04298 |           0.000976562 |          0         |           0.0420014  |                0.00100002  |                 0.260504  |
| sparse_magnitude_p0.1         | sparse         |    5506.84   |                     30667.6  |         0.414365 |    1.04298 |           0.000976562 |          0         |           0.0420014  |                0.00100002  |                 0.0361425 |
| sparse_residual_p0.1          | sparse         |    6432.92   |                     35841.8  |         0.414079 |    1.04298 |           0.000976562 |          0         |           0.0420014  |                0.00100002  |                 0.0368068 |
| sparse_residual_squared_p0.1  | sparse         |    6432.92   |                     35841.8  |         0.414079 |    1.04298 |           0.000976562 |          0         |           0.0420014  |                0.00100002  |                 0.0368068 |
| adaptive_lrr_b0.050           | adaptive_lrr   |    5040.26   |                     28060.8  |         0.340549 |    1.05085 |           0.000976562 |          0.0498769 |           0          |                0           |                 0         |
| combined_b0.05_p0             | combined       |    5040.26   |                     28060.8  |         0.340549 |    1.05085 |           0.000976562 |          0.0498769 |           0          |                0           |                 0         |
| combined_b0.05_p0.01          | combined       |    4703.71   |                     26180.4  |         0.322699 |    1.05505 |           0.000976562 |          0.0498769 |           0.00420087 |                0.000100004 |                 0.052415  |
| combined_b0.05_p0.05          | combined       |    5866.75   |                     32678.5  |         0.302102 |    1.07186 |           0.000976562 |          0.0498769 |           0.0210015  |                0.000500019 |                 0.112898  |
| structured_block16_p0.5       | structured     |    4857.44   |                     27039.3  |         0.380156 |    1.08138 |           0.000976562 |          0         |           0.0804072  |                0.00500229  |                 0.115715  |
| structured_column_p0.5        | structured     |    4797.28   |                     26703.2  |         0.319062 |    1.0825  |           0.000976562 |          0         |           0.0815217  |                0.00509386  |                 0.257826  |
| structured_block8_p0.5        | structured     |    4426.28   |                     24630.4  |         0.359803 |    1.08263 |           0.000976562 |          0         |           0.0816574  |                0.00500086  |                 0.16306   |
| structured_row_p0.5           | structured     |    6645.24   |                     37028    |         0.401472 |    1.08265 |           0.000976562 |          0         |           0.0816692  |                0.00510245  |                 0.0661315 |
| combined_b0.05_p0.1           | combined       |    5961.21   |                     33206.2  |         0.287419 |    1.09285 |           0.000976562 |          0.0498769 |           0.0420014  |                0.00100002  |                 0.156013  |
| adaptive_lrr_b0.100           | adaptive_lrr   |    3057.13   |                     16980.7  |         0.33474  |    1.09936 |           0.000976562 |          0.0983817 |           0          |                0           |                 0         |
| sparse_hessian_damage_p0.25   | sparse         |    4102.29   |                     22820.2  |         0.280356 |    1.10598 |           0.000976562 |          0         |           0.105001   |                0.0025      |                 0.347862  |
| sparse_magnitude_p0.25        | sparse         |    4272.83   |                     23773    |         0.404362 |    1.10598 |           0.000976562 |          0         |           0.105001   |                0.0025      |                 0.0594101 |
| sparse_residual_p0.25         | sparse         |    4747.11   |                     26422.9  |         0.40808  |    1.10598 |           0.000976562 |          0         |           0.105001   |                0.0025      |                 0.0507625 |
| sparse_residual_squared_p0.25 | sparse         |    4747.11   |                     26422.9  |         0.40808  |    1.10598 |           0.000976562 |          0         |           0.105001   |                0.0025      |                 0.0507625 |
| combined_b0.05_p0.25          | combined       |    4111.97   |                     22874.2  |         0.261433 |    1.15585 |           0.000976562 |          0.0498769 |           0.105001   |                0.0025      |                 0.232319  |
| structured_row_p1             | structured     |   11650.3    |                     64992    |         0.390686 |    1.16112 |           0.000976562 |          0         |           0.16014    |                0.0100046   |                 0.091221  |
| structured_block16_p1         | structured     |    5801.12   |                     32311.8  |         0.359223 |    1.16179 |           0.000976562 |          0         |           0.160814   |                0.0100046   |                 0.164409  |
| structured_column_p1          | structured     |    4257.19   |                     23685.6  |         0.278598 |    1.16206 |           0.000976562 |          0         |           0.161079   |                0.0100647   |                 0.35195   |
| structured_block8_p1          | structured     |    4530.86   |                     25214.6  |         0.334797 |    1.16427 |           0.000976562 |          0         |           0.163289   |                0.0100003   |                 0.221225  |
| sparse_magnitude_p0.5         | sparse         |    2717.46   |                     15082.9  |         0.39209  |    1.21098 |           0.000976562 |          0         |           0.210001   |                0.00500001  |                 0.0879553 |
| sparse_hessian_damage_p0.5    | sparse         |    3766.68   |                     20945.1  |         0.248815 |    1.21098 |           0.000976562 |          0         |           0.210001   |                0.00500001  |                 0.42123   |
| sparse_residual_p0.5          | sparse         |    4249.65   |                     23643.5  |         0.398267 |    1.21098 |           0.000976562 |          0         |           0.210001   |                0.00500001  |                 0.0735882 |
| sparse_residual_squared_p0.5  | sparse         |    4249.65   |                     23643.5  |         0.398267 |    1.21098 |           0.000976562 |          0         |           0.210001   |                0.00500001  |                 0.0735882 |
| combined_b0.05_p0.5           | combined       |    4139.16   |                     23026.1  |         0.236878 |    1.26085 |           0.000976562 |          0.0498769 |           0.210001   |                0.00500001  |                 0.304423  |
| sparse_magnitude_p1           | sparse         |    1635.44   |                      9037.46 |         0.376806 |    1.42098 |           0.000976562 |          0         |           0.420001   |                0.01        |                 0.123508  |
| sparse_residual_p1            | sparse         |    2234.31   |                     12383.4  |         0.378973 |    1.42098 |           0.000976562 |          0         |           0.420001   |                0.01        |                 0.118468  |
| sparse_residual_squared_p1    | sparse         |    2234.31   |                     12383.4  |         0.378973 |    1.42098 |           0.000976562 |          0         |           0.420001   |                0.01        |                 0.118468  |
| sparse_hessian_damage_p1      | sparse         |    3490.22   |                     19400.5  |         0.215099 |    1.42098 |           0.000976562 |          0         |           0.420001   |                0.01        |                 0.499656  |
| combined_b0.05_p1             | combined       |    2922.62   |                     16229.2  |         0.20787  |    1.47085 |           0.000976562 |          0.0498769 |           0.420001   |                0.01        |                 0.389602  |
| sparse_magnitude_p2           | sparse         |    1222.08   |                      6727.95 |         0.357892 |    1.84098 |           0.000976562 |          0         |           0.840001   |                0.02        |                 0.167504  |
| sparse_residual_p2            | sparse         |    1642.06   |                      9074.47 |         0.349582 |    1.84098 |           0.000976562 |          0         |           0.840001   |                0.02        |                 0.186834  |
| sparse_residual_squared_p2    | sparse         |    1642.06   |                      9074.47 |         0.349582 |    1.84098 |           0.000976562 |          0         |           0.840001   |                0.02        |                 0.186834  |
| sparse_hessian_damage_p2      | sparse         |    2994.78   |                     16632.3  |         0.177687 |    1.84098 |           0.000976562 |          0         |           0.840001   |                0.02        |                 0.586681  |
| combined_b0.05_p2             | combined       |    2661.68   |                     14771.2  |         0.173405 |    1.89085 |           0.000976562 |          0.0498769 |           0.840001   |                0.02        |                 0.490809  |
| sparse_magnitude_p5           | sparse         |     489.231  |                      2633.41 |         0.32473  |    3.10098 |           0.000976562 |          0         |           2.1        |                0.05        |                 0.244642  |
| sparse_residual_p5            | sparse         |     624.345  |                      3388.32 |         0.290396 |    3.10098 |           0.000976562 |          0         |           2.1        |                0.05        |                 0.324508  |
| sparse_residual_squared_p5    | sparse         |     624.345  |                      3388.32 |         0.290396 |    3.10098 |           0.000976562 |          0         |           2.1        |                0.05        |                 0.324508  |
| sparse_hessian_damage_p5      | sparse         |    1809.57   |                     10010.3  |         0.121157 |    3.10098 |           0.000976562 |          0         |           2.1        |                0.05        |                 0.718177  |
| combined_b0.05_p5             | combined       |    1727.34   |                      9550.95 |         0.119252 |    3.15085 |           0.000976562 |          0.0498769 |           2.1        |                0.05        |                 0.649824  |
| bf16                          | baseline       |      17.8982 |                         0    |         0        |   16       |          15           |          0         |           0          |                1           |                 1         |

## Lowest BPW reaching perplexity thresholds

|   ppl_threshold |   lowest_true_bpw | configuration   |
|----------------:|------------------:|:----------------|
|           100   |               nan | not reached     |
|            50   |               nan | not reached     |
|            30   |               nan | not reached     |
|            25   |               nan | not reached     |
|            19.7 |               nan | not reached     |

## Best result under each storage cap

|   bpw_cap |   best_perplexity | configuration         |
|----------:|------------------:|:----------------------|
|       1.1 |           3057.13 | adaptive_lrr_b0.100   |
|       1.2 |           3057.13 | adaptive_lrr_b0.100   |
|       1.3 |           2717.46 | sparse_magnitude_p0.5 |
|       1.5 |           1635.44 | sparse_magnitude_p1   |

## Answers

**Q1: Is Hessian-aware binary quantization better than naive binary? YES.** Best Hessian base: hadamard_hessian_b128, PPL 5754.5775, weighted error 0.429903; naive PPL 6469.9486, weighted error 0.448959.

**Q2: Is most model damage concentrated enough for targeted correction? NO.** The diagonal proxy is concentrated (top 1% captures 49.97%), but restoring that 1% recovers only 39.47% of the excess perplexity.

**Q3: What percentage causes most damage?** See `damage_concentration.csv`; top 1% weights capture 49.97%, with row, column, and block curves reported separately.

**Q4: How much can sparse salient rescue lower perplexity?** Hessian-selected 1% restoration changes PPL from 5754.5775 to 3490.2249 at 1.420978 BPW. The best sparse result is sparse_magnitude_p5, PPL 489.2310 at 3.100977 BPW. At about 1% restored, the best individual method (sparse_magnitude_p1, PPL 1635.4387) beats the best structured method (structured_column_p1, PPL 4257.1903).

**Q5: Best perplexity below 1.10/1.20/1.30/1.50 BPW:** shown in the storage-cap table above.

**Q6: Did any tested configuration reach within 5–10% of BF16? NO.**

**Adaptive-LRR ablation:** FAIL at the matched ~0.025 added-BPW point. Global rank-1 gives PPL 5359.8113 at 1.025572 BPW; the smallest adaptive budget gives PPL 7063.6345 at 1.025799 BPW.

**Q7: Dominant remaining error:** the diagonal activation-importance proxy does not identify the weights that control perplexity, and the remaining functional loss is not isolated by the tested sparse or rank-4 repairs. The strongest evidence is that Hessian selection removes much more proxy damage while magnitude selection produces much lower perplexity. No new algorithm is proposed here.

Magnitude and residual-squared strategies use identical cardinalities. `residual` and `residual_squared` must produce identical orderings and are retained as an explicit control. All sparse values are BF16 residuals; individual indices use a global flat index, while structured methods charge matrix and local-unit identifiers. Every low-rank factor stores FP16 U, S, and V.
