# BF16 restoration sensitivity map

Status: complete. This is an oracle diagnostic, not a deployable mixed-precision
method and not an under-30 result at two bits.

## Question

Can a small number of unusually sensitive transformer blocks explain the gap
between the current exact-two-bit trellis and BF16? If so, a global bit allocator
could concentrate capacity there. If damage is widespread, changing allocation
alone is unlikely to reach the target.

## Controls

- Model: Qwen2.5-0.5B base, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- GPU: NVIDIA RTX 4000 Ada Generation, 20 GB; PyTorch 2.8.0+cu128.
- Ranking: first 4,096 tokens of WikiText-2 validation.
- Reporting: the original and next 4,096-token WikiText-2 test slices.
- Starting checkpoint: freshly reconstructed `path_trained_rerank`, exactly
  2.00 serialized bpw across compressed transformer linear weights.
- Every BF16-restored parameter is charged 14 extra bits over its two-bit
  representation. Tables and headers remain included in the base payload.
- Original model weights are never trained. Restoration is deliberately an
  unattainable oracle used to locate damage.

| Model | Validation PPL | Test 0:4096 | Test 4096:8192 |
| --- | ---: | ---: | ---: |
| BF16 | 14.825 | 22.185 | 27.569 |
| Serialized two-bit baseline | 32.866 | 41.540 | 58.063 |

The baseline scores measured by the sensitivity script exactly match the source
checkpoint report after serialized reconstruction.

## Whole-layer oracle

Layers are ranked only by validation NLL gain. Lower perplexity is better.

| BF16 layers restored | Linear-equivalent bpw | Whole-model bpw | Test 0:4096 | Test 4096:8192 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.583 | 6.282 | 38.426 | 52.730 |
| 2 | 3.167 | 6.705 | 36.669 | 49.618 |
| 4 | 4.333 | 7.550 | 33.670 | 44.767 |
| 8 | 6.667 | 9.240 | 30.947 | 39.524 |
| 12 | 9.000 | 10.930 | **27.289** | 34.162 |
| 16 | 11.333 | 12.620 | 25.237 | 31.307 |
| 20 | 13.667 | 14.310 | 23.233 | **28.940** |
| 24 | 16.000 | 16.000 | 22.185 | 27.569 |

The top whole layers are 0, 2, 23, 3, 4, 5, 22 and 1. Restoring one
layer already costs 2.583 equivalent bpw across compressed linears. The first
slice crosses 30 only after 12 layers; both slices cross only after 20/24.

## Attention versus MLP

| BF16 groups restored | Attention test 0 | Attention test 1 | Attention linear bpw | MLP test 0 | MLP test 1 | MLP linear bpw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 41.298 | 57.291 | 2.072 | 38.496 | 53.096 | 2.512 |
| 4 | 40.095 | 55.944 | 2.287 | 34.876 | 46.331 | 4.046 |
| 8 | 38.817 | 54.304 | 2.574 | 32.240 | 41.670 | 6.092 |
| 12 | 38.007 | 52.773 | 2.862 | **28.687** | 35.937 | 8.138 |
| 16 | 37.188 | 51.756 | 3.149 | 26.989 | 33.695 | 10.185 |
| 24 | 36.609 | 51.104 | 3.723 | 24.722 | 30.577 | 14.277 |

MLP restoration helps much more, although MLP matrices also contain far more
parameters. The most sensitive MLPs by validation NLL are in layers 0, 2, 23,
4, 3 and 6. Restoring every attention group cannot reach 30 on either test
slice. Restoring half the MLP groups reaches 28.69 only on the first slice and
costs 8.138 linear-equivalent bpw.

## Cost-normalized attention/MLP ranking

Each attention or MLP group is ranked by validation-NLL gain per extra BF16 bit.

| Groups restored | Linear-equivalent bpw | Whole-model bpw | Test 0:4096 | Test 4096:8192 |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 2.727 | 6.386 | 37.491 | 51.497 |
| 8 | 3.014 | 6.594 | 36.386 | 49.761 |
| 16 | 4.028 | 7.329 | 33.443 | 45.326 |
| 24 | 6.801 | 9.337 | **29.546** | 38.349 |
| 32 | 9.135 | 11.027 | 26.304 | 32.981 |
| 36 | 11.181 | 12.509 | 24.574 | 30.582 |
| 40 | 12.787 | 13.673 | 23.649 | **29.589** |
| 44 | 14.833 | 15.155 | 22.547 | 28.025 |
| 48 | 16.000 | 16.000 | 22.185 | 27.569 |

Even an oracle that can restore exact BF16 groups needs 24/48 groups and 6.801
linear-equivalent bpw to cross 30 on the original slice. It needs 40/48 groups
and 12.787 bpw to cross 30 on both slices. This rejects the simple hypothesis
that a few bad blocks can be protected within the current global two-bit budget.

## Interaction finding

For 17/24 layers, the validation-NLL gain from restoring the entire layer is
less than the sum of restoring its attention and MLP groups separately. The
median interaction term is -0.00093 NLL. In layer 0, MLP-only restoration gains
0.09255 NLL, attention-only gains 0.00791, but whole-layer restoration gains
only 0.09150. Restoring layer-12 attention alone worsens validation NLL.

Quantization errors therefore sometimes cancel. Independent matrix or group
optimization can remove a compensating error and hurt the model. This supports
joint nonlinear block optimization and explicit cross-group error coordination.

## Reproducibility warning

The prior combined Hessian/node-fit arm was not stable across the two Ada GPUs.
When regenerated here it measured 42.618 / 60.312, worse than the plain trellis
at 41.540 / 58.063. Its files reproduced exactly after fresh decoding. The
sensitivity map therefore starts from the better plain trellis on this pod.
Results from different devices must not be silently combined.

## Decision

Do not build a simple layer router that merely moves bits toward a handful of
layers. The next quality experiment should concentrate on the MLP mechanism:
jointly select `gate`, `up`, and `down` codes using the true post-SiLU MLP output
under quantized-rollout inputs. It must permit coordinated error, including
beneficial cancellation, and demonstrate held-out NLL improvement before any
full-model run. A rate-response curve at 1.5/2/2.5/3 bpw is still needed before
claiming that global reallocation can help at exactly two bits.

Complete unit rankings, selected sets, storage accounting and all scores are in
[`block_sensitivity_20260921.json`](block_sensitivity_20260921.json). The map
took 78.05 seconds in the extended run. Binary checkpoints remain on the pod.
