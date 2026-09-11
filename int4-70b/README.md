# INT-4

Research into extreme model compression and efficient local inference. **Quality first, speed second.**

## Mission

Run **Llama 3.3 70B Instruct entirely within one RTX 3090's 24 GB VRAM**, preserving quality close to the original BF16 model, then optimize the engine for interactive generation.

This is an ambitious research target. Feasibility, quality retention, and performance have not been demonstrated.

## Final stretch target

| Dimension | Target |
| --- | --- |
| Model | Llama 3.3 70B Instruct |
| GPU | One NVIDIA RTX 3090 with 24 GB VRAM |
| Residency | All weights, quantization metadata, KV cache, and execution workspace fit within available VRAM. |
| Offloading | No weight or KV-cache offloading to CPU RAM or disk during inference; no remote inference or additional GPU. |
| Workload | Batch size 1; 4,096 input tokens followed by 512 generated tokens on a predefined mixed prompt suite. |
| Generation | At least 20 accepted output tokens/second median and at least 10 tokens/second at the 10th percentile across the suite. |
| First-token latency | Under 10 seconds on the specified workload with the model loaded and no reusable prompt cache. Report the distribution and failures. |
| Quality | No more than 1 percentage point absolute loss on each selected task score versus BF16, with statistical uncertainty reported. |
| Reproducibility | Public scripts, pinned configurations, memory measurements, and complete results. |

Ordinary host RAM for the application, tokenization, and loading is permitted. It must not serve as model execution storage to evade the VRAM limit. Record the host specification and actual device memory available.

## Why compression comes first

For approximately 70 billion parameters, raw weight storage alone is about:

| Precision | Raw storage, decimal GB |
| --- | ---: |
| 4 bits per weight | 35 |
| 3 bits per weight | 26.25 |
| 2 bits per weight | 17.5 |

These estimates exclude scales, zero points, codebooks, higher-precision exceptions, caches, and workspace. Exact checkpoint parameter counts and measured bytes determine feasibility. Conventional 4-bit weights cannot satisfy this residency target. Investigate representations around 2 bits per weight, or other compression schemes, while measuring their complete storage and quality costs.

The repository name remains INT-4, but research is no longer restricted to INT4 or W4A16. Representation and activation/accumulation precision will be selected from evidence.

## Phase 1: Prove quality and memory feasibility

1. Pin model/tokenizer revisions and establish a reproducible BF16 quality reference.
2. Select task evaluations, perplexity data, prompt formats, seeds, and statistical methods before optimizing against them. Keep calibration data separate from held-out evaluation data.
3. Compare established low-bit compression methods before developing our own. Investigate calibration, clipping, rounding, grouping, mixed precision, or quantization-aware training where justified.
4. Measure effective bits per weight, including metadata and exceptions. Build a full runtime memory budget for the 4K-input/512-output workload.
5. Evaluate perplexity, task scores, and individual failures. Fix a perplexity tolerance before acceptance testing; it is currently pending. Do not conceal per-task losses behind an average.
6. Advance to speed optimization only when a candidate meets the quality criteria and has a credible full-memory fit. Validate actual peak device memory during integration.

BF16 reference evaluation, calibration, and training may require separate hardware. The single-3090 restriction applies to final deployed inference. Report development resources separately.

## Phase 2: Optimize speed

- Implement and validate kernels for the accepted compressed representation.
- Optimize packed layouts, tiling, warp/SM work distribution, registers, shared memory, and overlap of data movement with computation.
- Investigate adaptive execution or speculative verification only when measurements justify them. Include draft weights, caches, workspace, rejected work, and controller overhead in the same budgets.
- Preserve numerical correctness within justified tolerances and rerun model-quality checks after implementation changes.
- Measure full-model generation and first-token latency; report kernel timing separately.

Scheduling cannot recover information lost through compression or bypass physical bandwidth limits. No new algorithm or speedup is assumed in advance.

## Fair comparisons

- Compare against tuned existing single-GPU implementations under the same memory and quality constraints.
- Include Marlin where representation and workload are compatible; beating it is no longer the sole success criterion.
- Separate quantization-quality comparisons from execution comparisons. Use identical quantized weights where compatible and disclose differences otherwise.
- Include all inference-time conversion costs. Report one-time loading and packing separately.
- Pin hardware/software settings, timing boundaries, warm-up, cache conditions, and sampling configuration. Count accepted output tokens, not speculative proposals or aggregate batch throughput.
- Publish per-prompt results, regressions, quality uncertainty, and peak total device memory. Longer-context claims require separate measurements.
- Make no world-first or record-performance claim without supporting comparisons.

## Scope change

This supersedes the earlier primary goal of beating Marlin with W4A16 across two models and two GPU architectures. The first milestone is now **70B quality under a strict 24 GB deployment budget**. Speed follows; broader model and architecture coverage comes after validation.

## First compression experiment

The FlashRun prototype explores a shared PCA representation of quantization residual blocks plus sparse index-value corrections for remaining errors. See [the experiment design and commands](EXPERIMENT.md).

## Current status

A real-model Python quality pipeline and reference compressed GPU runtime have been tested on Qwen2.5-0.5B using an NVIDIA L4. **The first PCA-plus-sparse configuration failed the quality gate.** Reconstruction correctness passed, but the quantized model's quality degraded severely. See [findings](results/REAL_MODEL_FINDINGS.md) and [reproduction details](REAL_MODEL.md).

No 70B memory fit, BF16-quality retention, or RTX 3090 performance has been demonstrated. Final evaluation datasets and perplexity acceptance tolerance remain to be selected.

## Work completed so far

The first prototype was implemented in Python with NumPy and PyTorch. It includes:

- A synthetic PCA-plus-sparse residual codec with save/load round-trip checks.
- A real-model pipeline that downloads a model, creates a simple groupwise 2-bit and 4-bit baseline, computes BF16 residuals, fits shared rank-2 PCA blocks, selects sparse corrections, reconstructs candidates, and evaluates perplexity.
- Bit-packing unit tests for 2-bit and 4-bit groups.
- A reference GPU-resident runtime that reconstructs compressed linear weights and validates its output against the dense reconstruction.
- Reproducibility notes, requirements, per-layer archives, and JSON reports.

### First real-model result

On 2026-09-11, the pipeline ran on **Qwen/Qwen2.5-0.5B**, revision `060db6499f32faf8b98477b0a26969ef7d8bb9987`, with an **NVIDIA L4**, PyTorch 2.8.0+cu128, and Transformers 4.51.3. The evaluation used 4,080 next-token predictions from the first 4,096 WikiText-2 test tokens. Lower perplexity is better.

| Representation | Perplexity |
| --- | ---: |
| BF16 reference | 22.20 |
| Naive groupwise 4-bit | 26.72 |
| Naive groupwise 2-bit | 112,423 |
| 2-bit + rank-2 PCA residual | 106,073 |
| 2-bit + PCA + 1% sparse residual | 88,952 |
| 2-bit + sparse residual at matched correction budget | 70,026 |

This result is a failed quality gate for the initial configuration. PCA and sparse corrections reduce numerical weight error, but the model quality remains unusable. Sparse-only corrections performed better than the PCA hybrid at the same correction budget. The correction payload was approximately one additional bit per quantized weight before base metadata, so it also misses the 24 GB 70B budget.

### What succeeded

The codec, serialization, bit packing, GPU reconstruction, and evaluation pipeline all function. The GPU runtime matched the dense hybrid reconstruction exactly in the perplexity test, and a 32-token generation smoke test completed.

### What remains unsolved

The naive quantizer is not a strong baseline, rank-2 PCA captures too little useful residual structure, magnitude-based sparse selection does not identify model-sensitive errors, and the runtime reconstructs full matrices temporarily rather than using a fused compressed kernel. The 70B model, RTX 3090 deployment, BF16-quality retention, 24 GB fit, and speed targets remain unproven.

See [REAL_MODEL_FINDINGS.md](results/REAL_MODEL_FINDINGS.md) for raw measurements and limitations. The next gate is a matched-byte comparison against stronger low-bit methods with sensitivity-aware corrections before any C++/CUDA implementation.
