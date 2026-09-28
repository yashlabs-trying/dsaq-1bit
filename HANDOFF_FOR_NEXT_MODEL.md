# Research handoff

Read these first:

1. [`README.md`](README.md)
2. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
3. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)
4. the specification, script, and generated report for the experiment being extended

## Current conclusion

The tested Hadamard 1-bit base cannot be repaired compactly by preserving a sparse set of exact scalar residuals. Experiment 8 required 94.227% protected residual at the PPL boundary; Experiment 9 reduced this only to 93.444921% at PPL 19.683539. The next research question is decoder expressiveness: how can coupled binary bits represent a richer layer function at approximately one physical bit per weight?

Do not add another fixed scalar importance ranking unless it directly addresses the interaction and density evidence from Experiments 7–9.

## Artifact caveat

Experiments 1–7 are reproducible from committed code and result artifacts. Experiment 9 code is committed, but its completed result bundle still needs to be imported. Experiments 8 and 10 are currently research-record entries only. Preserve this distinction in future reports.

## Fixed reference points

- Model: `Qwen/Qwen2.5-0.5B`
- Dataset: WikiText-2 `raw-v1`
- BF16 PPL: approximately 17.90
- Strict later-stage boundary: PPL <=19.688
- Hadamard 1-bit full-model PPL: approximately 637k–644k, depending on the recorded evaluation configuration
- Experiment 6 Fisher oracle: PPL 649.493 at 10% protected
- Experiment 7 best accepted ICQR checkpoint: PPL 25,740 at 0.250043% protected
- Experiment 9 best logged safe mask: PPL 19.683539 at 93.444921% protected

## Before a new run

```bash
git status --short
git log -5 --oneline
python -m pytest -q
python -m py_compile run_experiment*.py
```

Keep the model, dataset, seed, baseline, evaluation definition, true-BPW accounting, original prompt, environment snapshot, and raw result tables with every run. Never commit credentials or model-access tokens.
