# A40 strict 2-bit experiments

These experiments used Qwen2.5-0.5B base and the same held-out evaluation as the earlier runs: the first 4,096 WikiText-2 test tokens, producing 4,080 scored next-token predictions. The BF16 reference was 22.1929 perplexity. Lower is better. No model weights were finetuned.

The best result entering this run was 66.0807 perplexity. It used one learned 256-entry four-weight codebook per matrix, full-Hessian codebook updates, and an FP16 scale and offset per output row. Its complete transformer-linear payload was 2.03489 bits per compressed weight, so it was close to but above a strict 2-bit limit.

## Results

| Method | Perplexity | Bits/compressed weight | Decision |
| --- | ---: | ---: | --- |
| BF16 reference | 22.1929 | 16 | Reference |
| Hessian VQ + row affine, previous best | 66.0807 | 2.03489 | Retained quality baseline; over strict budget |
| BlockLDLQ + Hessian VQ + row affine | 66.5589 | 2.03489 | Reject: slightly worse |
| 128-wide randomized Hadamard + Hessian VQ | 221.9839 | 2.03492 | Reject: much worse |
| Measured global mixed allocation | 215.1708 | 1.999943 | Reject: meets budget, poor quality |
| 1.75-bit VQ + FP8 scale per 64 weights | 411.9151 | 1.87885 | Reject: poor quality |
| 2-bit VQ + FP8 scale per 64 + entropy coding | 86.9825 | 2.10283 | Reject: worse and over budget |

The global allocator profiled 1.75, 2.5, and 3-bit index formats using local Hessian distortion. It assigned 127 matrices to G4/K128, 10 to G2/K32, and 31 to G2/K64, including FP16 codebooks, scales, and offsets under a 1.999943 bpw total. This demonstrates that local `E H E^T` ranking alone does not predict the accumulated end-to-end language-model loss.

## What the run established

- Sequential BlockLDLQ feedback is not enough with the current learned 4D codebook.
- A block Hadamard transform is incompatible with the structure this codebook learned. QuIP-style transforms need a codebook designed for the transformed distribution.
- Spending bits using isolated linear-layer Hessian distortion can make end-to-end perplexity worse, even while meeting the storage budget.
- Local scales cannot recover the quality lost by shrinking the codebook to 128 entries.
- The 256-entry grouped-scale representation does not have enough index entropy slack to pay for its scale metadata.

The next serious implementation should use a lattice/codebook designed jointly with incoherence processing, or optimize quantization against held-out transformer-block outputs while keeping the original model weights frozen. More heuristics around the present per-matrix codebook are unlikely to reach perplexity below 40.
