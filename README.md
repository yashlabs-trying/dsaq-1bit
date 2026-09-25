# DSAQ: structured and seeded 1-bit weight reconstruction

This repository implements the first two falsification experiments for an adaptive, approximately one-bit LLM weight representation:

1. **DSA suitability:** determine whether compact magnitude surfaces beat sign-plus-scale on blocks whose structure can be measured before routing.
2. **Seeded residual search:** on the exact held-out blocks classified as irregular by Experiment 1, test whether reproducible generated candidates improve reconstruction enough to justify their candidate-index bits.

The primary run uses `Qwen/Qwen2.5-0.5B`, fixed WikiText-2 calibration/evaluation slices, 128x128 blocks, and all linear projections in transformer layers 0, 12, and 23. This deliberately covers early/middle/late Q/K/V/O, gate/up/down matrices without claiming whole-model quantization. The untouched BF16 model is the perplexity reference.

Every result keeps these together: perplexity, activation-sensitivity-weighted error, and true bits per weight. True BPW includes signs, FP16 parameters, DSA family IDs, routing flags, candidate indices, shared distribution parameters, and the global seed.

## Methods

The naive baseline stores one sign bit per weight and one FP16 magnitude per block. DSA fits a nonnegative magnitude surface selected from row trend, column trend, plane, and quadratic families. The family is selected by raw calibration MSE; activation second moments are used only as a diagonal Hessian proxy for reporting in Experiment 1.

The fixed Hadamard variant transforms each padded block along its input dimension, reconstructs in the transformed domain, and applies the inverse transform before scoring or inserting the weight into the model.

The suitability model is ridge regression trained on a deterministic calibration split of blocks using only pre-reconstruction statistics. Its threshold is selected on that split. All headline routing measurements use the disjoint held-out block split.

Seeded search evaluates K in 2, 4, 8, and 16 with uniform, Gaussian, robust, and globally shared magnitude distributions. Candidate streams are reproducible from a global seed, matrix name, block ID, and candidate ID. Only the chosen candidate index is stored; local distribution parameters are charged when used.

## Run

```bash
python -m pip install -r requirements.txt
python run_experiments.py --output results/run
```

For a quick implementation check:

```bash
pytest -q
python run_experiments.py --synthetic --output results/smoke
```

The run writes raw block records, per-layer and aggregate tables, perplexity results, fitted router metadata, plots, and `REPORT.md`. Run metadata records the seed, model revision, package versions, arguments, and device.

Experiment 3 tests FP16-stored low-rank corrections to the Hadamard 1-bit residual, followed by either an ordinary or diagonal-Hessian-weighted second correction:

```bash
python run_experiment3.py --output results/experiment3
```

Ranks, factor storage, residual spectra, sensitivity concentration, layer/projection tables, matched perplexity, and recovery per metadata bit are recorded automatically.

Experiment 4 tests a Hessian-aware Hadamard binary base, damage-routed low-rank correction, and individually or structurally indexed BF16 salient residual rescue:

```bash
python run_experiment4.py --output results/experiment4
```

It reports exact binary-scale, FP16 factor, BF16 value, and sparse-index storage separately.

Experiment 5 quantizes all transformer projections and selects reproducible binary-scale candidates from the actual hidden states produced by preceding quantized blocks:

```bash
python run_experiment5.py --output results/experiment5
```

It compares independent and sequential calibration, K=4/8/16, horizons 0/1/2/4, clean and current-input targets, and beam widths 1/2/4 without residual repair.

Experiment 6 Part 1 is an oracle importance study. It keeps the Hadamard 1-bit
bulk fixed, restores selected weights exactly, and compares magnitude,
activation-aware, gradient, and Fisher selectors under global and
layer-balanced budgets:

```bash
python run_experiment6.py --output results/experiment6
```

Experiment 7 scores the actual Hadamard quantization error. It compares a
conditional blockwise Q-OBS selector with static, iterative, and collapse-prior
ICQR repair driven by teacher-student KL:

```bash
python run_experiment7.py --output results/experiment7
```

## Interpretation limits

This is a controlled proof-of-concept on three representative layers of one small model. A positive result is evidence for a routing signal and a generated-candidate mechanism, not evidence that a complete model can reach the same BPW or perplexity. A negative result kills the tested DSA surfaces or seeded distributions, not every possible structured reconstruction method.
