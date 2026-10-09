# Experimental Summary: Layer-0 4x4 VQ Codec Optimization

## Setup

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen2.5-0.5B |
| Revision | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Device | Laptop CPU (Windows, 4 threads) |
| Packed size | 2,223,107 bytes |
| BPW | 1.19285875 |
| Evaluation | 4,096 fixed WikiText tokens |
| Reference PPL | 22.188 |
| Calibration | 512 tokens A (search) + 512 tokens B (gate) |
| Framework | `functional_vq_cpu/scripts/` |

---

## PPL Progression (best-to-worst)

| Stage | PPL | Δ vs BF16 | NLL | ΔNLL removed | Key change |
|---|---|---|---|---|---|
| **BF16** | **22.19** | — | 3.100 | — | Unquantized reference |
| **E14c** | **57.65** | +159.8% | 4.054 | **0.954** | Robust neuron-aligned joint G/U/Down assignment |
| E14b4 | 58.48 | +163.6% | 4.069 | 0.969 | Consistency-fixed two-group joint proof |
| E10a | 59.18 | +166.7% | 4.081 | 0.981 | Gate+Up downstream L1+L2 polish |
| E9c | 66.61 | +200.2% | 4.199 | 1.099 | Down L1+L2 polish |
| E9b | 73.15 | +229.7% | 4.293 | 1.193 | Down L1 sensitivity |
| E8g | 76.17 | +243.3% | 4.333 | 1.233 | Down MSE polish (champion) |
| E9a | 76.79 | +246.1% | 4.341 | 1.241 | Layer transition MSE (rejected) |

---

## Experiment Log

### E7 Phase: Initial Gate/Up reassignment + Down polish
*Focus: MSE-based functional optimization of the original E7a codebook*

| Experiment | PPL | Ly(B) | Coverage | Changed | Finding |
|---|---|---|---|---|---|
| E7a (codebook) | ~89.6 | 0.1423 | — | — | Starting codebook |
| E7b 10% | — | 0.1296 | 10% | 17,457 IDs | Gate/Up reassignment reduces Ly |
| E7b 20% | — | 0.1159 | 20% | 26,000 IDs | More coverage helps |
| E7c paired G/U 5% | — | 0.1151 | 5% pairs | 4,116 IDs | Paired search weak after E7b |
| E7d Down 5% | — | 0.1099 | 5% | 6,273 IDs | Down polish efficient |

### E8 Phase: Codebook EM + repeated Down polish
*Focus: prototype adaptation and iterative Down refinement*

| Experiment | PPL | Ly(B) | Changed | Finding |
|---|---|---|---|---|
| E8a codebook EM | — | 0.1097 | 937 coordinates | Prototype adaptation marginal |
| E8b reassign 10% | — | 0.1084 | 4,085 IDs | Re-assignment after EM |
| E8c Down r1 | — | 0.1034 | 6,255 IDs | Down polish still productive |
| E8d Down r2 | — | 0.0989 | 5,985 IDs | |
| E8e Down r3 | — | 0.0962 | 4,173 IDs | |
| E8f Down r4 | — | 0.0942 | 3,318 IDs | |
| **E8g Down r5** | **76.17** | **0.0925** | 2,845 IDs | Stop rule (gain < 2%) |

### E9 Phase: Downstream sensitivity optimization
*Focus: Replace MSE proxy with actual downstream loss (L1+L2)*

| Experiment | PPL | Loss | η | Finding |
|---|---|---|---|---|
| E9a transition MSE | 76.79 | Lh2=0.139 | — | Whole-layer MSE is wrong proxy |
| **E9b L1 sensitivity** | **73.15** | Lsense=0.351 | — | Sensitivity works; first PPL gain over E8g |
| **E9c L1+L2 sensitivity** | **66.61** | Lsense+2=0.102 | — | +2 layers better than +1 |

### E10 Phase: Multi-projection polish + oracle
*Focus: Apply L1+L2 sensitivity to Gate+Up, measure projection importance*

| Experiment | PPL | Loss | Finding |
|---|---|---|---|
| **E10a Gate+Up x4 rounds** | **59.18** | Lsense=0.0002 | Best shared codec ever |
| E10c CE gradient search | 59.18 (same) | 0/40K accepted | Block-level reassignment is SATURATED |
| Oracle: restore Down to BF16 | 37.74 | — | 0.45 NLL recoverable, but non-local |
| Oracle: restore Gate | 42.42 | — | |
| Oracle: restore Up | 40.28 | — | |
| Oracle: restore O | 46.38 | — | |
| Oracle: restore V | 50.77 | — | |
| Oracle: restore K | 58.53 | — | Q/K nearly solved |
| Oracle: restore Q | 59.16 | — | Q/K nearly solved |

### E11 Phase: Codebook prototype retraining
*Focus: Direct CE gradient on prototype values*

| Experiment | PPL | Finding |
|---|---|---|
| E11 global CE retrain | 276 | Naive global prototype gradient destructive |
| E11 staged EM (32 proto) | rolled back | Targeted replacement fails B-validation |

### E12 Phase: Projection-specialized codebooks
*Focus: Separate codebooks per projection at same 4096 total*

| Experiment | Loss(B) | BPW | Finding |
|---|---|---|---|
| E12 specialized | 1.740 | 0.257 | Shared vocabulary is strictly better |

### FR1 Phase: Functional residual corrections
*Focus: Sparse additive corrections from CE gradient*

| Experiment | PPL | η | Finding |
|---|---|---|---|
| FR1 Down (η=0.3, 10%) | 61.69 | 0.3 | Down CE gradient ≈ 0 at E10a |
| FR1 Gate+Up (η=0.3) | 59.19 (same) | 0.3 | Gate/Up CE gradient also ≈ 0 |
| FR1 Gate+Up (η=0.01, 2%) | 59.19 (same) | 0.01 | No first-order signal in any MLP proj |

### E13 Phase: Loss-specialized dual vocabulary
*Focus: 3072 general + 1024 critical patterns, functionally weighted*

| Experiment | PPL | BPW | Finding |
|---|---|---|---|
| E13 dual vocab | 42,552 | 0.257 | Implementation issues (underfit, wrong BPW) |

E13 is an infrastructure/implementation failure, not a valid test of the
dual-vocabulary hypothesis: it did not preserve the required 1.19-BPW codec.

### E14 Phase: Neuron-aligned joint Gate/Up/Down assignment
*Focus: atomically search coupled Gate, Up, and Down 4x4 IDs for the same MLP neuron group*

| Experiment | Strict PPL | IDs changed | BPW | Finding |
|---|---:|---:|---:|---|
| E14 frozen L1+L2 proxy, 5% | 61.77 | 250 | 1.19285875 | Proxy improves while PPL worsens; rejected |
| E14 micro frozen-proxy proof | 59.92 | 6 | 1.19285875 | Frozen sensitivity still misaligned |
| E14b malformed 512-token CE | 59.75 | 6 | 1.19285875 | Exposed missing final RMSNorm and sequence-boundary bug |
| E14b2 segmented CE | 59.30 | 6 | 1.19285875 | Correct 256-token segmentation nearly removes overfit |
| E14b3 per-segment gate | 59.05 | 10 | 1.19285875 | Positive, but in-memory multi-location tracking was inconsistent |
| **E14b4 consistent proof** | **58.48** | **10** | **1.19285875** | First fully reproducible neuron-aligned win |
| **E14c robust 0.5%** | **57.65** | **15 vs E14b4** | **1.19285875** | New champion; exact same packed bytes |
| E14d ranks 7-12 | 57.84 | 12 vs E14c | 1.19285875 | Calibration improved but held-out regressed; rejected |

E14b4 and E14c require every accepted move to improve calibration-A mean CE,
not worsen calibration-B mean CE, and not worsen any of the four independent
256-token calibration sequences. Candidate triples are shortlisted by the
signed first-order CE change and then evaluated with exact downstream CE.

### E15 Phase: Stored joint neuron-package vocabulary
*Focus: replace three independent MLP IDs with one aligned package representation*

| Experiment | Strict PPL / gate | BPW | Finding |
|---|---:|---:|---|
| E15, 8,140 FP16 joint prototypes, BF16-MSE target | 23,361.74 | 1.19285875 | Flat joint ID destroys functional structure |
| E15a, 8,140 FP16 prototypes, E14c-imitation target | 31,399.19 | 1.19285875 | Preserving parent targets does not overcome package entropy collapse |
| E15b, 4,096 joint bases + three 4-bit residual subcodes | functional gate failed | 1.19285875 | 82.7–83.1% parent MLP-output error; strict PPL skipped |

E15 validated the exact storage idea: all variants decode in a fresh process,
retain attention exactly, preserve the residual stream, and occupy exactly
2,223,107 bytes. The negative result is representational, not infrastructure.
One global package ID collapses the combinatorial entropy of three independently
selected patterns. Even three small component residual subcodes leave about 50%
relative weight error and 83% MLP-output error versus E14c.

---

## Tail Token Analysis (from E10a, calibration A)

| Token percentile | Fraction of excess NLL |
|---|---|
| Top 1% | 7.3% |
| Top 5% | 24.9% |
| Top 10% | 38.1% |
| Top 20% | 56.3% |
| Top 50% | 88.2% |

Excess NLL is highly concentrated: 10% of tokens cause 38% of damage.

---

## Key Scientific Findings

1. **Downstream sensitivity > MSE**. Changing from MSE → L1 sensitivity → L1+L2 sensitivity improved PPL from 76.2 → 73.2 → 66.6 (+2 layers better).

2. **Gate/Up are the highest-leverage projections after Down**. E10a added Gate+Up optimization to the sensitivity framework, improving PPL from 66.6 → 59.2.

3. **Independent single-block reassignment is saturated, but coupled moves are not**. E14 found valid Gate/Up/Down triples after E10c found no individual move.

4. **Earlier E10c CE evidence was not definitive**. Its downstream path omitted final RMSNorm, flattened independent 256-token sequences into one 512-token sequence, and squared the projection instead of minimizing signed CE change. E14 corrected all three issues.

5. **Zero gradient ≠ zero recoverable error**. The BF16 oracle proved restoring Down alone drops PPL to 37.74, but this requires non-local moves that gradient steps cannot find.

6. **Global prototype gradient descent is destructive**. E11 proved that moving all 4096 prototypes simultaneously (even with CE gradient) produces PPL 276.

7. **Neuron-aligned coupling is real but the old vocabulary remains limiting**. Correct joint moves improved 59.18 → 57.65 at identical storage, but ranks 7-12 failed to generalize. Assignment-only coupling is not on a credible path to <30.

8. **Global flat neuron packages are too restrictive**. Saving ID bits by forcing all 272,384 triples through one 8K vocabulary removes essential Gate/Up/Down combinatorial information. Future neuron alignment must be selective or use a factorized/learned decoder with substantially more latent entropy.

---

## Where experiments were run

All experiments were executed on a **Windows laptop** (no GPU) using PowerShell, with 4 CPU threads via `torch.set_num_threads(4)`. The working directory is:

```
C:\Users\yashs\OneDrive\Documents\ChatGPT\1 bit\
```

Key paths:
- Scripts: `functional_vq_cpu/scripts/`
- Results: `functional_vq_cpu/results/`
- Packed codecs: `functional_vq_cpu/packed/`
- Checkpoints: `functional_vq_cpu/checkpoints/`
- Caches: `functional_vq_cpu/cache/`
- Artifacts (tokens): `artifacts/functional_vq_l0/calibration_cache/`

---

## File inventory (packed codecs)

All 2,223,107 bytes except where noted:

| File | Bytes | SHA-256 |
|---|---|---|
| E7b_reassign_10pct.packed | 2,223,107 | — |
| E7b_reassign_20pct.packed | 2,223,107 | 97843dea... |
| E7c_paired_gu_5pct.packed | 2,223,107 | 224ab2f7... |
| E7d_down_polish_5pct.packed | 2,223,107 | de3ae63c... |
| E8a_codebook_30.packed | 2,223,107 | 1b3ed341... |
| E8b_reassign_10pct.packed | 2,223,107 | e0ebd628... |
| E8c_down_polish_5pct.packed | 2,223,107 | f38d7487... |
| E8d_down_polish_round2.packed | 2,223,107 | b83d6aac... |
| E8e_down_polish_round3.packed | 2,223,107 | c27fe1e7... |
| E8f_down_polish_round4.packed | 2,223,107 | df21ce82... |
| **E8g_down_polish_round5.packed** | 2,223,107 | d6dabae1... |
| E9a_down_transition_5pct.packed | 2,223,107 | a9e87545... |
| E9b_downstream_l1_5pct.packed | 2,223,107 | e1e85eb1... |
| **E9c_downstream_l12_5pct.packed** | 2,223,107 | 4dee94f4... |
| **E10a_gate_up_downstream.packed** | 2,223,107 | 83478887... |
| **E14b4_joint_ce_consistent_proof.packed** | 2,223,107 | f93e4320... |
| **E14c_joint_ce_consistent_005.packed** | 2,223,107 | b76cd223... |
| E14d_joint_ce_consistent_rank7_12.packed | 2,223,107 | 99f08954... |
| E15_joint_package_8140.packed | 2,223,107 | 878e153b... |
| E15a_joint_parent_imitation_8140.packed | 2,223,107 | 96354e38... |
| E15b_joint_base4096_res4.packed | 2,223,107 | c5918e77... |
| E11_retrained.packed | 2,223,107 | f02a57d5... |
| E12_specialized.packed | 1,913,747 | 49b27495... |
| E13_dual_vocab.packed | 1,913,747 | a86884ec... |
| FR1_down_residual.packed | n/a | — |
| FR1_gate_up_residual.packed | n/a | — |

---

## Best stored checkpoints (results JSON)

| File | Notable metrics |
|---|---|
| E8g_strict_cpu_ppl.json | PPL=76.169, NLL=4.333 |
| E9a_strict_cpu_ppl.json | PPL=76.792 (rejected) |
| E9b_strict_cpu_ppl.json | PPL=73.155, NLL=4.293 |
| E9c_strict_cpu_ppl.json | PPL=66.606, NLL=4.199 |
| E10a_strict_cpu_ppl.json | PPL=59.176, NLL=4.081 |
| E14b4_joint_ce_consistent_proof_strict_ppl.json | PPL=58.484, NLL=4.069 |
| **E14c_joint_ce_consistent_005_strict_ppl.json** | **PPL=57.650, NLL=4.054 (best)** |
| E14d_joint_ce_consistent_rank7_12_strict_ppl.json | PPL=57.838, NLL=4.058 (rejected) |
| E15_joint_package_8140_strict_ppl.json | PPL=23,361.743, NLL=10.059 (rejected) |
| E15a_joint_parent_imitation_8140_strict_ppl.json | PPL=31,399.192, NLL=10.355 (rejected) |
| E10d_projection_oracle.json | Oracle table of per-projection BF16 restoration |
| E11_strict_cpu_ppl.json | PPL=276.198 (codebook retrain failed) |
| E13_dual_vocab.json | PPL=42551 (implementation issues) |
| **E16c_functional_joint_proof_strict_ppl.json** | **PPL=57.548, NLL=4.053 (1 accepted selective joint correction)** |
| **E16d_functional_joint_rank3_8_strict_ppl.json** | **PPL=57.621, NLL=4.054 (5 accepted corrections; rejected vs E16c)** |

---

## Conclusion

The shared 4×4 VQ codebook with 4096 patterns at 1.19 BPW on Layer 0 now reaches **PPL 57.65** after robust neuron-aligned joint assignment. The minimal proof is positive: coupled Gate/Up/Down changes can escape the independent-block local optimum. The scaling curve is not strong enough to approach PPL 30 with the old vocabulary, however.

E15 shows that a flat global joint vocabulary is not that representation: the
ID savings come from deleting too much entropy. The next credible path is a
**selective hybrid**—retain independent coding for most blocks and apply richer
joint neuron packages only to the functionally critical minority—or a compact
factorized learned decoder whose latent code preserves component-specific
freedom. E14c remains the frozen champion.

### E16 selective joint residual results

The entropy audit found that the existing 12-bit MLP IDs are near-maximal
entropy (Gate 11.931, Up 11.958, Down 11.927 bits/ID), so ordinary lossless
coding cannot recover enough budget to replace independent choices. E16 instead
keeps the frozen E14c codec and adds a selective joint Gate/Up/Down correction
for critical neuron groups, while preserving the exact packed size of
2,223,107 bytes (1.1928587526 BPW).

E16c used exact segmented calibration-CE gating and accepted one correction.
Strict held-out WikiText-2 PPL improved from 57.6497 to **57.5483** (NLL
4.052625), SHA-256 `c5ac7150860dc4b95d915f90542bfef8114f3790035e4f4f26bba8e883517d42`.

E16d expanded the search to six groups and accepted five corrections. Although
calibration-B CE improved slightly, strict held-out PPL was **57.6207** (NLL
4.053882), worse than E16c. E16c is therefore the current E16 champion. This
is a small positive signal with diminishing returns, not yet a breakthrough;
the next useful step is stronger candidate generation or a learned
low-dimensional joint decoder rather than simply searching more groups.
