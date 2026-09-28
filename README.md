# WaveFly: falsification-driven research toward practical 1-bit LLMs

WaveFly is an experimental research program asking a narrow question:

> **What representation can preserve transformer function at approximately one stored bit per weight?**

This repository records ten completed experiments on Qwen2.5-0.5B and one proposed follow-up. The project began with structured magnitude surfaces and candidate search, then tested low-rank residuals, second-order guidance, functional search, exact-weight importance oracles, conditional causal repair, hierarchical rescue, and activation-space reconstruction.

The central result so far is a constraint, not a compression claim: **the information required by the tested Hadamard 1-bit base is dense in ordinary weight coordinates**. Even the strongest hierarchical rescue run required **93.444921%** of the original residual to remain exact to reach the 10%-degradation boundary. That result redirects the work from finding a small set of weights to protect toward designing a more expressive binary representation.

## Research snapshot

| Experiment | Hypothesis under test | Main result | Decision |
|---|---|---|---|
| 1. DSA suitability | Measurable block structure predicts when a compact magnitude surface helps | Pearson 0.942, but held-out Spearman 0.253 missed the preregistered gate | Reject tested router |
| 2. Seeded search | A few reproducible candidates rescue irregular blocks | Best method beat DSA on only 0.08% of held-out irregular blocks | Reject candidate family |
| 3. Low-rank residuals | The 1-bit residual is compressibly low-rank | Best PPL 1,072.61 at 1.296 BPW; rank 8 captured 14.46% energy | Useful correction, insufficient representation |
| 4. Hessian + sparse rescue | Quantization damage is concentrated enough to repair cheaply | Best PPL 489.23 at 3.101 BPW | Reject sparse proxy as path to target |
| 5. Functional binary search | Activation-aware sequential selection fixes accumulated error | PPL 96,086 vs 644,467 baseline at 1.001 BPW | Search helps; representation remains limiting |
| 6. Importance oracle | A small exact subset restores function | Fisher reached PPL 649.49 at 10% protected | Small scalar subset not found |
| 7. Conditional importance | Recomputed causal repair handles interactions | Local gain correlation 0.723; best checkpoint PPL 25,740 | Local repairs do not compose globally |
| 8. Joint protection elimination | Starting from BF16 reveals removable residual information | 94.227% still required at PPL 19.687 | Residual is dense |
| 9. Hierarchical group rescue | Coarse-to-fine structure finds a much smaller protected set | 93.444921% required at PPL 19.683539 | Sparse exact-residual route rejected |
| 10. Activation-nullspace folding | Preserve function instead of reconstructing weights | Best Layer-0 PPL about 1,071; stopped at the first-layer gate | Decoder family too restrictive |

Experiments 1–7 are fully represented by code and committed result artifacts. Experiment 9 code is present; its final result is recorded from the completed pod run, while the result bundle has not yet been imported. Experiments 8 and 10 currently have research-record provenance only. See [the experiment ledger](docs/EXPERIMENTS.md) for methods, innovations, outcomes, limitations, and artifact status.

## Why the sequence matters

The experiments were designed as decision gates rather than a leaderboard:

1. **Can better local 1-bit fitting solve the problem?** Experiments 1–5 showed that preprocessing, search, and low-rank corrections produce real gains but leave a large perplexity gap.
2. **Can a small amount of original information repair the damage?** Experiments 6–9 showed that local importance is measurable, yet the information needed for BF16-like quality is dense and strongly interacting.
3. **Can one bit encode a richer function than one independent binary weight?** Experiment 10 changed the objective from weight reconstruction to activation-space fidelity, but its rank-1-like decoder lacked capacity.

This progression prevents repeated optimization of a representation family that the evidence has already ruled out.

## Experimental discipline

- **Primary quality metric:** WikiText-2 perplexity.
- **Reference model:** `Qwen/Qwen2.5-0.5B`.
- **BF16 reference:** approximately 17.90 PPL.
- **Strict quality boundary:** PPL <= 19.688 for the full-model importance studies.
- **Storage accounting:** sign bits, scales, factors, indexes, routing flags, candidate IDs, and shared seeds are charged in true bits per weight.
- **Selection/evaluation separation:** calibration data drives fitting and selection; the fixed evaluation split measures final perplexity.
- **Negative-result policy:** a failed gate changes the next hypothesis; it is not hidden by reporting only MSE.

The full protocol and limits are in [Methodology](docs/METHODOLOGY.md).

## Reproduce

Create an environment and run the fast checks:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pytest -q
python run_experiments.py --synthetic --output results/smoke
```

Run the first two representative-layer experiments:

```bash
python run_experiments.py --output results/run
```

Run later experiments independently:

```bash
python run_experiment3.py --output results/experiment3
python run_experiment4.py --output results/experiment4
python run_experiment5.py --output results/experiment5
python run_experiment6.py --output results/experiment6
python run_experiment7.py --output results/experiment7
python run_experiment9.py --output results/experiment9
```

GPU experiments download Qwen2.5-0.5B and WikiText-2 from Hugging Face. Each run writes its environment, aggregate tables, diagnostics, plots, and a generated `REPORT.md`.

## Repository map

```text
dsaq/                  Core block transforms, candidate generation, and routing
run_experiment*.py     Standalone experiment drivers
research_prompts/      Original experiment specifications
results/               Committed reports, tables, and plots for Experiments 1–7
docs/EXPERIMENTS.md    Research ledger and decision history
docs/METHODOLOGY.md    Evaluation protocol, metrics, and validity limits
tests/                  Determinism and transform-level checks
```

## Current research direction

The evidence argues against spending more effort on sparse preservation of exact scalar weights. The next useful experiment must expand decoder expressiveness while keeping amortized storage near one BPW: multiple effective magnitudes from coupled binary bits, shared filters or dictionaries, and direct activation-space objectives. The proposed Experiment 11 explores functional sigma-delta binary folding; it has not produced a result yet.

## Scope

This is active research, not a production quantizer. Results apply to the stated model, data slices, quantization base, and search budgets. Representative-layer experiments and full-model interventions are labeled separately. No result in this repository currently demonstrates BF16-like perplexity at approximately 1 BPW.
