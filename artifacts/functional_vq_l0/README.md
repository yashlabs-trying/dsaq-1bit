# Qwen2.5-0.5B Layer-0 functional VQ artifacts

Frozen snapshot of the reproducible Layer-0 experiment completed on 2026-10-08.

## Fixed conditions

- Model: Qwen2.5-0.5B
- Scope: transformer Layer 0
- Codec: packed/decodable 4x4 VQ with 4096 patterns, 12-bit IDs, scales, and fixed residual payload
- Stored size at E0-E6: 2,223,107 bytes
- Exact BPW: 1.1928587525755494
- BF16 held-out WikiText-2 PPL: 22.1882704768
- Final evaluation was performed only after decoding the stored checkpoint
- Calibration A drove search; calibration B gated acceptance

## Results

| Stage | Change | PPL | Delta PPL | Ez(B) | Ey(B) | IDs changed | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| E0 | Frozen control | 1229.9896 | 5443.42% | - | - | - | 310.75 s |
| E1 | 5% functional assignment | 408.7116 | 1742.02% | 0.6599 | 0.6681 | 22,971 | 977.01 s |
| E2 | Paired Gate/Up search | 345.7966 | 1458.47% | 0.6601 | 0.6602 | 4,622 | 247.87 s |
| E3 | 10% functional assignment | 216.2042 | 874.41% | 0.6193 | 0.6145 | 41,699 | 1915.09 s |
| E4 | 20% functional assignment | 164.4477 | 641.15% | 0.5673 | 0.5406 | 80,577 | 3891.30 s |
| E5 | 30% functional assignment | 134.6497 | 506.85% | 0.4953 | 0.4623 | 112,726 | 5815.91 s |
| E6 | 40% functional assignment | 97.7278 | 340.45% | 0.4114 | 0.3809 | 136,940 | 7734.77 s |
| E7a-short | 24 accepted projected prototype steps | 89.5863 | 303.76% | 0.4158 | 0.3773 | 2,822 pattern coordinates | 7.27 s |
| E7a-100 | 25 accepted projected prototype steps; saturated | 89.5881 | 303.76% | 0.4158 | 0.3773 | 2,822 pattern coordinates | 7.44 s |

E6 reduced PPL by 92.1% relative to E0 without changing packed size. E7a then proved same-byte functional prototype adaptation has leverage: the short run improved PPL by 8.3% to 89.5863. The stricter 100-step attempt accepted only one additional micro-update before calibration-B rejection and did not materially improve PPL, marking the local frontier for this prototype subset.

## Packed checkpoint SHA-256

```text
E0 44408f1bdf85168c9b785414ec4a02bb98b31da2a37589ef3f579774aa77d060
E1 ee696f9e3cd27d91b2ea80bc5b315fe380a7c1d5a57dfcc38d0eb59643958555
E2 4b7a47e353ac3105082a529a701182d9a609a863011bb5873bd13197503b63f2
E3 f87d9f5c0a520b06165363c41e26f117b6572be17ebbeefc87530cf68fb53ced
E4 310c784f94da4b9277f3bfb7a51e367e4f3f9537d0978c91e923db97b6f55eee
E5 baa60f589a6b841353c6e2c8e9735e2558c46627d7514eefb68ce66cc101ad9e
E6 518938c0070f40f7ca2ce3c5206fa639bc86ecfb9d00f5ba77d6b6637bbadf71
E7a-short 2f6813041478110e67bd54dc8476c40e64a776a9ee73ed3c3277c9bd39b7ae60
E7a-100 5e15e9d8921d1de2c0fd9ab3756cc292413b39bd7ee562e9535a3b182375a768
```

## Directory contents

- `packed/`: complete E0-E6 decodable checkpoints
- `results/`: raw JSON results and hashes
- `scripts/`: exact scripts used for construction and functional search
- `calibration_cache/`: fixed calibration/evaluation token IDs
- `logs/`: original run logs
- `environment/`: Python and GPU information
