# Handoff for the next research model

## Start here

This repository contains the complete 1-bit LLM quantization study through Experiment 7. The active scientific problem is **IMPORTANCE**. Do not move to compactness or encoding: Experiment 7 did not pass the required importance gate.

Read these files in order:

1. `README.md`
2. `research_prompts/experiment7.md`
3. `results/experiment7/REPORT.md`
4. `run_experiment7.py`
5. Earlier prompts and reports only when historical detail is needed.

The Git repository is `/workspace/dsaq`. The Hugging Face cache is `/workspace/.cache/huggingface` and already contains Qwen2.5-0.5B and WikiText-2. Environment snapshots are in `/workspace/dsaq/environment`.

## Fixed experimental setup

- Model: `Qwen/Qwen2.5-0.5B`
- Dataset: WikiText-2 raw-v1
- GPU on the current pod: NVIDIA RTX 4000 Ada Generation, 20 GB
- Python: 3.12.3
- Base quantizer: the same Hadamard + 1-bit construction used by Experiments 6 and 7
- BF16 perplexity: 17.8982
- Hadamard 1-bit perplexity: 637,559
- Hard importance gate: PPL below 100 with at most 10% exact oracle restoration

## Experiment 7 result

The hard gate failed.

- Best Experiment 6 global Fisher: PPL 649.493 at 10% protected
- Best Q-OBS: PPL 575,487 at 3%
- Best quantization-error Fisher within 10%: PPL 73,602 at 7.5%
- Best iterative ICQR checkpoint: PPL 25,740 at 0.25%
- Iterative ICQR reached 0.375% internally and then stopped because no proposed batch passed both teacher-KL and calibration-NLL acceptance.
- Collapse-prior ICQR stopped at 0.250114% for the same reason.
- Static ICQR was catastrophic.
- Direct causal score versus measured gain had Pearson r=0.723; 96.4% of tested corrections had positive local gain.
- Pair interactions ranged from -41.01 to +33.76, median -4.286.
- The tested small-critical-subset hypothesis is not supported at or below 10%.

Interpretation: local causal repair direction is predictable, but independent/local corrections do not compose into global perplexity recovery. The next importance method should operate on joint subspaces, coordinated error sets, or an optimization trajectory. Do not introduce another fixed scalar per-weight ranking without evidence addressing this failure.

The implemented Q-OBS is a tractable blockwise approximation using conditional 128x128 error groups, damped activation covariance, and inverse-covariance diagnostics. It is not a claim of exact per-weight SparseGPT/OBS.

## Reproduce and inspect

```bash
cd /workspace/dsaq
export HF_HOME=/workspace/.cache/huggingface
python -m pytest -q
python -m py_compile run_experiment7.py
python run_experiment7.py --help
```

Before rerunning an expensive experiment, inspect the CLI in the script and preserve the fixed model, dataset, seed, baseline, and evaluation definitions. Existing outputs are under `results/experiment1` through `results/experiment7` (Experiments 1 and 2 use the top-level `results` files).

## Key artifacts

- `results/experiment7/aggregate.csv`: all budget-level metrics
- `results/experiment7/causal_validation.csv`: predicted and directly measured repairs
- `results/experiment7/interactions.csv`: pairwise non-additivity
- `results/experiment7/conditional_stopping.csv`: why conditional runs stopped
- `results/experiment7/protection_allocation.csv`: layer/projection allocations
- `results/experiment7/per_layer_drift.csv`: hidden-state damage
- `results/experiment7/plots/`: required plots
- `research_prompts/`: original user specifications for Experiments 1 through 7

## Repository state at handoff

The expected branch is `main`. The Experiment 7 result commit is `5871222`. Run `git status --short` and `git log -3 --oneline` before changing anything. Keep generated results and the exact prompt that requested each experiment. Never commit credentials or access tokens.

