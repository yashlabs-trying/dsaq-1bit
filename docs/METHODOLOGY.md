# Methodology and validity

## Research question

The project studies post-training quantization near one bit per transformer weight. Each experiment tests a specific mechanism and uses a decision gate to prevent unconstrained iteration on a failed idea.

## Common setup

- **Model:** `Qwen/Qwen2.5-0.5B`.
- **Data:** WikiText-2 `raw-v1`, with fixed calibration and evaluation slices.
- **Reference quality:** BF16 perplexity around 17.90; small differences across reports come from explicitly recorded evaluation settings.
- **Binary base:** one sign bit per weight with block-scale metadata, usually after a fixed Hadamard transform along the input dimension.
- **Representative-layer studies:** Experiments 1–4 use layers 0, 12, and 23 across Q/K/V/O and MLP gate/up/down projections.
- **Full-transformer studies:** Experiments 5–9 intervene on all 24 blocks and 168 target projections while leaving embeddings, normalization, and the LM head in BF16 unless a report says otherwise.

## Metrics and storage

Perplexity is the final functional metric. Reconstruction studies also report raw MSE, relative error, cosine similarity, and activation-sensitivity-weighted error. Full-model importance studies add teacher-logit KL, hidden-state NMSE/cosine, collapse location, protected fraction, runtime, and calibration evaluations.

`true_bpw` includes binary signs, FP16 block scales and low-rank factors, DSA family and routing identifiers, candidate IDs and non-shared parameters, sparse indexes and values, and global seeds when they are not free by construction.

Exact BF16 restoration in Experiments 6–9 is an **oracle diagnostic**. It is not presented as a deployable encoding, and protected percentage is not equivalent to true BPW.

## Decision protocol

1. Define the hypothesis and success gate before interpreting the run.
2. Fit or rank with calibration data.
3. Measure final perplexity on a fixed held-out evaluation slice.
4. Keep perplexity, weighted error, and storage together when storage is part of the hypothesis.
5. Preserve negative results and use them to constrain the next experiment.
6. Distinguish representative-layer evidence from full-model evidence.
7. Report the tested mechanism narrowly; do not generalize a failed family to all 1-bit representations.

## Reproducibility

Committed result directories contain a generated `REPORT.md`, machine-readable tables, plots, and an environment snapshot. Original specifications are preserved in `research_prompts/`. Candidate streams derive from a global seed plus stable matrix, block, and candidate identifiers.

```bash
python -m pytest -q
python -m py_compile run_experiments.py run_experiment3.py run_experiment4.py \
  run_experiment5.py run_experiment6.py run_experiment7.py run_experiment9.py
python run_experiments.py --synthetic --output results/smoke
```

Large experiments require CUDA plus access to the Hugging Face model and dataset. Exact CLI options and environment metadata should be retained with every result.

## Threats to validity

- The program studies one small model family and one language-modeling dataset.
- Calibration and evaluation slices are smaller than a production benchmark suite.
- Diagonal Fisher/Hessian statistics and tractable blockwise Q-OBS are approximations.
- Greedy, beam, and hierarchical searches cannot certify global optimality.
- PPL near a boundary is sensitive to evaluation details; later studies use the fixed 19.688 ceiling for comparability.
- Experiments 8 and 10 need code and raw outputs imported before independent reproduction.
- Experiment 9's final result bundle also remains to be imported, although its implementation and specification are committed.

## Standard for future claims

A method should be called a practical near-1-bit quantizer only after it reports full-model storage, end-to-end quality, calibration cost, inference implications, and comparisons on more than one model and dataset. Until then, results here are mechanism studies and falsification evidence.
