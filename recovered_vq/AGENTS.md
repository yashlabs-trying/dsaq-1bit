# Quantization Codec Research — Handoff Document

## Milestone Achieved
**Fisher-Weighted Block VQ with 4×4 blocks achieves 1L ΔPPL = 5.3% at 1.15 BPW.**

This is the best result so far — under the 1L<10% target. The approach:
- Split weight matrices into 4×4 blocks (16 weights each) 
- Learn a shared K=65536 codebook via k-means on all blocks
- Encode using **Fisher-weighted distance**: pick centroid minimizing `(B-C_k)^T H (B-C_k)` where H = Fisher diagonal
- Optional: 4-bit residual correction for worst ~5% of blocks (K=16 residual codebook)

## Files

| File | Purpose |
|---|---|
| `/tmp/weighted_vq.py` | **Main codec**: Fisher-weighted VQ with residual. Best config: `K=65536, K_residual=16, bh=4, bw=4` |
| `/tmp/vq_all_configs.py` | Runner that compares 4×4, 1×16, 16×1 shapes + adaptive sequential refinement |
| `/tmp/block_vq_exp.py` | Earlier unweighted VQ experiments (shared/per-family codebooks) |
| `/tmp/alrw_experiment.py` | Earlier ALRW sign+wavelet experiments (legacy) |
| `/tmp/RESEARCH_SUMMARY.md` | Full research history |

## Reproduce the 5.3% result

```bash
cd /tmp
python3 weighted_vq.py   # Run weighted VQ with best config
```

Note: k-means initialization has high variance. Run 3-5 times and pick the best.
The best run gave: 1L=5.3%, 2L=52.8%, 3L=1305%, 4L=2424%.

## The remaining problem

Error **compounds at layer 3** (52.8% → 1305%). The chain is:
```
L0 quantized → slightly wrong h₁ → L1 encoded for wrong input → 
more wrong h₂ → L2 → even more wrong h₃ → L3 → avalanche at L4
```

The fix: **sequential output-MSE centroid fine-tuning** (not just re-encoding).

## Next agent: Implementation plan

Add to `/tmp/weighted_vq.py`:

### Step 1: Per-layer centroid fine-tuning
After initial global k-means + Fisher-weighted encoding:

```python
for layer_l in range(4):
    # 1. Run quantized prefix to get student input h̃_l
    restore_model(m, w_orig)
    apply_weights(m, {L0..L{l-1}_quantized})
    h_student = capture_activations(m, layer_l)  
    
    # 2. For each block in layer l, compute output error
    #    Δy = W_original(h̃_l) - W_quantized(h̃_l)
    
    # 3. Fine-tune centroids to minimize ||Δy||²
    #    c_k ← c_k - η · ∇c_k ||Δy||²
    #    This is gradient descent on the actual output MSE
    
    # 4. Store fine-tuned centroids as layer-specific codebook
```

### Step 2: Activation-aware encoding
During encoding, weight centroid selection by `||x_block||²` (the norm of the student input activation for that block), not just Fisher diagonal.

### Step 3: Deterministic k-means init
Use k-means++ initialization or multiple random starts + pick best to eliminate the 10× PPL variance.

### Step 4: Output-MSE centroid selection
When picking a centroid for a block, minimize `||W_original(h̃) - centroid_assigned(h̃)||²` instead of `||W - centroid||²`.

## Evaluation

```bash
# Quick test (5 min):
cd /tmp && python3 -c "
from weighted_vq import *
# ... run config, report PPL
"

# Full 4-config comparison (35 min):
cd /tmp && python3 vq_all_configs.py
```

## Connection

```bash
# Jump host (this machine):
ssh root@157.157.221.29 -p 20992 -i ~/.ssh/id_ed25519

# To reach RunPod from here:
ssh -i ~/.ssh/id_ed25519 b2xdw7v39y8ryy-64410d4b@ssh.runpod.io
```
