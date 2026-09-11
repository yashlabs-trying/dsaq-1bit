# First real-model result: representation failed the quality gate

Run on 2026-09-11 using an NVIDIA L4, PyTorch 2.8.0+cu128 and Transformers 4.51.3. Model: Qwen/Qwen2.5-0.5B, revision 060db6499f32faf8b98477b0a26969ef7d8bb9987. The pod is not the target RTX 3090.

## What ran

168 transformer linear weight matrices were quantized, encoded, saved, reconstructed, and evaluated. All candidates used the first 4096 WikiText-2 test tokens in disjoint 256-token windows (4080 predictions). No evaluation text was used for PCA fitting. Embeddings, tied output head, norms and biases were unchanged.

| Representation | Perplexity (lower is better) | Tensor payload, MB decimal |
| --- | ---: | ---: |
| Original BF16 | 22.20 | Not reported in initial run |
| Basic 4-bit | 26.72 | 473.69 |
| Basic 2-bit | 112423.50 | 384.23 |
| 2-bit + PCA | 106073.41 | 406.66 |
| 2-bit + PCA + sparse | 88951.77 | 428.13 |
| 2-bit + sparse, matched correction budget | 70026.43 | 428.13 |

These are naive affine min-max quantizers, not state-of-the-art low-bit baselines. The enormous 2-bit perplexity indicates catastrophic degradation. Relative improvement from that broken baseline does not establish useful quality. Sparse-only beats the hybrid at the same correction storage budget on this test.

## What succeeded

- Bit packing/unpacking and constant-group unit tests passed.
- PCA and sparse corrections decreased real weight reconstruction error.
- Weighted MSE decreased from 9.3121e-5 (2-bit) to 8.9339e-5 (PCA), 8.1500e-5 (hybrid), and 8.0328e-5 (matched sparse-only). The 4-bit control achieved 3.6825e-6.
- A reference compressed GPU runtime was implemented. Its buffers remain GPU-resident and it reconstructs one linear weight matrix temporarily per forward call.
- All runtime layer reconstructions matched the independently assembled BF16 hybrid tensors exactly. Full evaluation perplexity also matched exactly.
- Greedy generation completed, but the continuation was incoherent. It is a functional test, not a speed achievement.

## What failed or remains unproven

- Rank-2 PCA removes only about 4.1% of residual squared error in this setup. PCA plus sparse removes about 12.5%; this is nowhere near sufficient quality recovery.
- The correction payload costs approximately 0.98 additional bits per quantized linear weight, before base scales/offsets. This configuration does not meet the 70B memory target.
- Whole-model effective bits are much higher because the small model's large embedding table remains BF16. Do not extrapolate those figures directly to 70B.
- Selecting corrections by magnitude is not the same as selecting them by task sensitivity.
- No AE, VQ, activation-aware fitting, or strong quantization baseline has been tested.
- No 70B checkpoint was downloaded or evaluated. No BF16-equivalent quality, 24GB 70B fit, or RTX 3090 performance has been established.

## Runtime measurements and limits

The reference runtime reported 450,218,496 bytes resident PyTorch allocation, 839,669,760 bytes peak allocation, and 1,126,170,624 bytes peak reservation during this small-model test. These are PyTorch allocator measurements, not total process/device memory accounting. The evaluation used no KV cache; a separate 32-token generation smoke test used the normal generation path. They do not establish the final 4K-context memory budget.

Full-matrix temporary reconstruction, initial BF16 model loading, and experimental multi-variant archives make this a research runtime, not a deployable 70B engine. Speed work should not proceed on this failed representation.

## Next decision

Improve the quality experiment before reducing bits or writing custom kernels: establish a strong low-bit base, inspect residual structure across block shapes, and compare sensitivity-aware correction/codebook alternatives at matched bytes. This result rejects this particular naive rank-2 configuration; it does not reject all shared latent representations.

Raw metrics: [real_model_l4.json](real_model_l4.json).
