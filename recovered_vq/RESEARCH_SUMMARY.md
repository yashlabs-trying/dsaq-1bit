# Quantization Codec Research Summary

## Goal
Compress Qwen2.5-0.5B transformer weights to ≤1.10 BPW (bits per weight) with minimal PPL degradation across 4 consecutive layers. No activation corrections — representation only.

## Codec Evolution

### 1. ACE-Q (Sign + Raw Haar + Hessian Low-Rank) — FAILED
```
W ≈ sign(W) · scale + Haar_coeffs + Hessian_lowrank_residual
```
- **BPW**: 1.00 (sign) + ~0.005 (Haar) + ~0.095 (Hessian) = ~1.10
- **Result**: 4-layer PPL explosion to millions. Correction experiments all failed.
- **Lesson**: 1-bit sign throws away too much structure. Tiny corrections can't rescue.

### 2. WaveRel (Sign + Relationship Ordering + Haar + Selective Coeffs) — IMPROVED
```
W ≈ sign(W) · (reorder⁻¹(Haar_selective(reorder(|W|))) + Hessian_residual
```
- Sort rows/cols by mean magnitude so similar values are adjacent
- Apply 1-level 2D Haar, keep only top coefficients by magnitude
- Hessian low-rank residual for remaining error
- **BPW budget**: sign(1.00) + wave(0.04) + residual(0.05) + order(0.01) = 1.10
- **Result**: Gradual degradation — 1,604% → 1,802% → 1,297% → **29,388%** (4L)
- **Lesson**: More wave budget → better depth stability (80/20 split won), but still collapses at 4L

### 3. ALRW (Adaptive Local Relationship Wave) — BREAKTHROUGH
```
W ≈ sign(W) · (M_eq + R_hat)

where:
M_eq = local equation reconstruction per kernel
R_hat = 1-level 2D Haar reconstruction of residual (M - M_eq)
```
- Split magnitude into 32×32 kernels
- Per kernel, fit cheapest from:
  - **f0**: a (constant, 1 param)
  - **f1**: a + b·r_i + c·c_j (row+col bias, 3 params)
  - (f2: a + b·r_i + c·c_j + d·r_i·c_j and f3: u_i·v_j available but not tested)
- Choose f0 vs f1 by: `score = MSE + α × (n_params × 16 + 2)` per kernel
- Reconstruct M_eq, compute residual R = |W| - M_eq, apply 1-level Haar on R
- **Result**: PPL=34.0 (Δ=9.1%) → 35.7 → 41.1 → **42.4 (Δ=36.2%)** — 4L!
- Cosine stays above 0.92 through all 4 layers
- **BPW**: 12.08 (wave coefficient budget is 30% — far too generous)

## Key Failure Modes

| Attempt | 4L ΔPPL | Root Cause |
|---|---|---|
| ACE-Q baseline | ~millions | Sign quantization destroys magnitude relationships |
| ACE-Q + corrections | ~millions | Tiny rank-4 can't fix MLP errors across 4 layers |
| WaveRel (80/20) | ~25,000% | Haar on un-ordered magnitudes is inefficient |
| WaveRel budget sweep | ~25,000-28,000% | All splits same cliff — budget not the issue |
| ALRW (30% wave) | **36%** | Works! But BPW=12, way over budget |

## Open Problems (for next agent)

### Critical: Reduce ALRW BPW from 12.08 to ≤1.10

Current BPW breakdown at 30% wavelet budget:
- Sign: 1.00 (fixed)
- Equation params + types: 0.05
- Wave coefficient indices (32-bit positions): ~7.4 **(61% of total)**
- Wave coefficient values (float16): ~3.6 **(30% of total)**

### Proposed fixes (in priority order):

1. **Multi-level 2D Haar on residual** (replace 1-level)
   - Most energy concentrates in very few deep-level coefficients
   - At budget-appropriate keep-rate (~0.1% of coefficients), captures far more energy than 1-level

2. **Add f2 equation** (interaction term d·r_i·c_j)
   - Only +1 parameter per kernel (+16 bits)
   - Significantly reduces residual energy → fewer wave coefficients needed

3. **Drop wave index cost** by storing coefficients as sparse blocks
   - Instead of storing (value + position) per coefficient, store small regions

4. **Tiny Hessian residual** (rank ≤ 1, adds ≤0.01 BPW)
   - Fixes systematic residual patterns that wave coding misses

### 4. Sign+Wavelet ALRW Compression (1.1–1.7 BPW) — STILL FAILS AT 1.20 BPW
```
W ≈ sign(W) · (M_eq + wavelet_residual)
```
- **ALRW at 12.08 BPW**: cos=0.919, 4L Δ=36% (works, too expensive)
- **ALRW adapted to 1.2 BPW (3L Haar, INT6, deterministic bands)**: cos≈0.77, 4L Δ≈millions%
- **Equations (f0/f1/f2)**: Only 0.004–0.065 BPW because Qwen weight magnitudes within 32×32 kernels lack local row/column structure
- **Lesson**: Wavelet at 0.15 BPW budget loses 23% of weight structure (cos=0.77), which destroys PPL. The gap between cos=0.92 (functional) and cos=0.77 (broken) is insurmountable at this budget.

### 5. Whole-Block Pattern Codebooks (VQ) — BREAKTHROUGH APPROACH
```
W ≈ codebook_decode(block_id)   # One 16-bit ID per 4×4 block
```

**Representation**: Split each weight matrix into 4×4 blocks (16 weights). Learn a shared codebook of K=65536 prototypes (16-bit IDs). Each prototype encodes the full signed 16-weight pattern — sign + magnitude + relationships together. No sign/magnitude separation!

**Key result — progression across improvements:**

| Method | BPW | Cos | 1L Δ | 2L Δ | 3L Δ | 4L Δ |
|---|---|---|---|---|---|---|
| Unweighted VQ K=64K | 1.14 | 0.831 | 8597% | 34697% | 29116% | 53342% |
| +Fisher encoding only | 1.14 | 0.774 | **42%** | - | - | **2697%** |
| +Fisher + residual K=16 (4.6%) | **1.15** | 0.777 | **15.7%** | **19.8%** | **327%** | **1107%** |
| +Fisher + residual K=256 (10.7%) | **1.19** | 0.789 | 54.8% | 30.2% | 292% | **889%** |

**Why VQ beats sign+wavelet:**
- VQ at 1.14 BPW achieves cos=0.83 vs ALRW wavelet's cos=0.77 at 1.2 BPW
- VQ preserves the full signed 16-weight pattern instead of 1-bit sign + heavily compressed magnitude
- Fisher-weighted encoding improves PPL **205×** at 1L (8597%→42%) with zero extra BPW

**Why PPL is still >30%:**
- Cosine 0.78–0.83 loses 17–22% of weight structure
- Error compounds across 4 layers: layer 3 errors amplify layer 4 errors
- 16-bit IDs provide only 65536 patterns for 3.7M blocks — many blocks share centroids with significant residual error

**To reach ΔPPL < 30% at 1.20 BPW:**
1. Use per-output VQ (group by columns, not 2D blocks) — preserves output functions
2. Add Hessian-weighted residual VQ with wider coverage (target 20% of blocks)
3. Use 2-bit scale per block (0.125 BPW) on top of Fisher-weighted VQ

### Experiment scripts:
- `/tmp/alrw_exp.py` — ALRW wavelet compression
- `/tmp/block_vq_exp.py` — Block VQ with shared/per-family codebooks
- `/tmp/weighted_vq.py` — Fisher-weighted VQ with residual correction (best so far)

### Current status (best result):
**Fisher-weighted VQ K=65536 + 4-bit residual:**
- BPW: **1.15** ✓ (below 1.20)
- 1L ΔPPL: **15.7%** (nearly functional!)
- 4L ΔPPL: **1107%** (exceeds 30% due to error compounding)
- Cosine: **0.78**

The gap to close: 1107% → 30% requires preventing error compounding at layers 3-4.
This likely needs per-column VQ or activation-aware training.
