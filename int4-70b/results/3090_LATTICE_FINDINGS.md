# RTX 3090: 39.33 perplexity at a serialized 2.00 bpw

On 2026-09-15, the E8P lattice adaptation reached **39.3299 perplexity** on the original 4,096-token WikiText-2 test slice, versus **22.1719 BF16**. A fresh model reconstructed only from the binary checkpoint reproduced exactly the same perplexity. No original model parameters, layer norms, or language-model head were finetuned. The fixed lattice indices were optimized with discrete Hessian-guided updates.

This reaches the 30–40 target on that slice. It does not establish that range across evaluation data: the next unused 4,096 tokens scored **55.9547**, versus **27.5071 BF16**. Each slice scored 4,080 next-token predictions in independent 256-token windows.

## Complete experiment table

| Method | Original-slice perplexity | Compressed-weight bpw |
| --- | ---: | ---: |
| BF16 | 22.1719 | 16 |
| Previous Hessian VQ + row affine, same-pod reproduction | 66.3076 | 2.03489 |
| Hessian-aware initialization, two restarts | 71.3984 | 2.03489 |
| Metric initialization + sequential block calibration | 73.5720 | 2.03489 |
| Original initialization + sequential block calibration | 80.3222 | 2.03489 |
| Two-sided mixing + E8P + BlockLDLQ | 51.4102 | 2.00009, estimated format |
| Strict format + one discrete coordinate sweep | 49.4608 | **2.00000, serialized** |
| Strict format + one sweep + 8,192 calibration tokens, 256-token windows | **39.3299** | **2.00000, serialized** |

Larger calibration and longer windows changed together in the final run, so their individual contributions have not been isolated. The final method uses BF16-input Hessians; sequential calibration was rejected for the existing learned-codebook method.

## Representation and accounting

The method combines full two-sided orthogonal mixing, the official QuIP# E8P codebook, and buffered BlockLDLQ. For Qwen's unsupported odd dimension factors, the transform uses a DCT times a Walsh-Hadamard transform rather than the official Hadamard utility. **This is an adaptation, not an exact QuIP# reproduction or an entirely new quantizer.** The official codebook source is external and GPL-3.0 licensed.

Every eight transformed weights use one 16-bit lattice index. A shared 1,024-byte packed absolute grid is stored once. Each matrix stores an 18-byte header containing dimensions, a 64-bit sign seed, and an FP16 scalar. To pay for those 4,048 bytes of metadata, 2,024 predetermined lattice vectors use implicit code zero rather than stored indices. These positions are constrained during quantization and refinement, not dropped afterward. This affects 16,192 of 357,826,560 compressed weights (about 0.0045%).

Measured binary payload: **89,456,640 bytes**, exactly **2.00 bits per compressed transformer-linear weight**. All code files round-tripped, and fresh reconstruction reproduced model perplexity. Sign regeneration currently depends on the pinned PyTorch CUDA RNG implementation.

Embeddings and the tied language-model head remain BF16. The complete model representation is therefore **5.85984 bits per model parameter**, not 2 bits for the whole model. No 70B/24 GB deployment or fused compressed inference kernel is demonstrated by this result. Evaluation reconstructs dense weights.

## Reproduce

Use an RTX 3090 with PyTorch 2.8.0+cu128. The model revision recorded in the report is `060db6499f32faf8b98477b0a26969ef7d8b9987`.

```bash
pip install -r requirements-lattice.txt
export HF_HUB_ENABLE_HF_TRANSFER=0
git clone https://github.com/Cornell-RelaxML/quip-sharp.git /workspace/quip-sharp
git -C /workspace/quip-sharp checkout 1d8f873e9a2a8b86b12bb1064c312c5689b77d98
python -m unittest test_additive_vq test_lattice_experiment test_calibration
FLASHRUN_OUTPUT=/workspace/lattice_strict_8k \
FLASHRUN_QUIP_ROOT=/workspace/quip-sharp \
FLASHRUN_LATTICE_STRICT=1 FLASHRUN_LATTICE_TUNE_ITERS=1 \
FLASHRUN_CAL_TOKENS=8192 FLASHRUN_CAL_SEQUENCE_LENGTH=256 \
FLASHRUN_GPU_HESSIAN=1 python lattice_experiment.py
FLASHRUN_CHECKPOINT=/workspace/lattice_strict_8k python lattice_eval.py
```

The final encoding run took 171.34 seconds on this pod; another quality experiment shared the GPU, so this is an observed development duration, not an isolated performance benchmark. Twelve tests passed. Raw reports, including the independent serialized-checkpoint evaluation, are in [3090_20260915.json](3090_20260915.json).

The next gate is evaluation over larger, separate datasets and task scores, followed by compression of the currently BF16 components and a fused runtime. The original test slice has been used repeatedly for method selection.
