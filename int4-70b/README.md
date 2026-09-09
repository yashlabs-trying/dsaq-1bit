# INT-4

An INT4 inference engine research project focused on faster execution with minimal model-quality loss.

## Target

Build our own INT4 inference engine that outperforms Marlin on relevant workloads while keeping model quality close to the original BF16 model. Demonstrate the result on two models from different families and two different GPU architectures.

These are research targets, not achieved results.

## Initial scope

- **Precision:** W4A16 — INT4 weights, BF16 activations, and higher-precision accumulation.
- **Quantization:** Develop and evaluate weight representations, scales, grouping, clipping, and rounding to minimize quality loss.
- **GPU kernels:** Optimize work distribution across thread blocks, warps, and SMs; packed weight layouts; register and shared-memory use; and overlap between data movement, unpacking, scaling, and computation.
- **Runtime:** Load quantized models, manage memory, and select kernels for the hardware and matrix shapes.
- **Portability:** Use a shared design with configurations or implementations adapted to each architecture and workload.

## How we will evaluate success

| Goal | Evaluation |
| --- | --- |
| Faster kernels | Compare latency against a pinned Marlin version on the same GPU across a predefined suite of actual model layer shapes. |
| Faster inference | Measure full-model throughput and latency for decoding, multi-token verification, and prefill separately. |
| Minimal quality loss | Compare perplexity and relevant task scores against the original BF16 model under identical evaluation settings. |
| Correct execution | Compare kernel outputs against a trusted reference using justified numerical tolerances. |
| Consistent results | Report all four model–GPU combinations, including regressions and memory use. |

Use identical quantized weights, scales, group sizes, and activation dtypes for kernel comparisons wherever compatible. Evaluate our quantization method separately against established INT4 methods and BF16. If representations differ, disclose the differences and compare their speed–quality tradeoffs.

Include the strongest applicable architecture-specific alternatives alongside Marlin. Beating Marlin alone does not establish a world-record result.

## Benchmark commitments

- Select the two models, two GPU architectures, workloads, and numerical quality-loss limits before the final evaluation.
- Pin software versions and record hardware, timing methodology, cache conditions, and tuning settings.
- Include all conversion and other costs incurred during inference. Allow one-time weight preparation for every implementation and report that cost separately.
- Publish per-case results, aggregate speedups, correctness checks, and reproducible benchmark commands.
- Distinguish kernel speedups from full-model improvements and account for measurement variability.

## Current status

Project target defined. Model and GPU selection, quantitative acceptance thresholds, implementation, and benchmark results are pending.
