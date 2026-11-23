# WaveFly V6 pod handoff

This directory is a resumable snapshot for the V6 experiment. The V6 run completed on 33 matrices from Qwen/Qwen2.5-0.5B, facebook/opt-125m, and EleutherAI/pythia-70m.

Repository commit: `37805b3`

## Environment

- GPU: NVIDIA RTX 2000 Ada Generation, 16 GB
- PyTorch: 2.8.0+cu128
- CUDA: 12.8
- transformers: 4.56.2
- Seed: 20260923 for V6; 20260922 for V5
- TF32: disabled
- `HF_HUB_ENABLE_HF_TRANSFER=0`
- `OPENBLAS_NUM_THREADS=4`

The pod already has the downloaded model checkpoints in `/workspace/.cache/huggingface`. Do not redownload them unless the cache is missing.

## Resume commands

```sh
cd /workspace/wavefly-proof
python tests/test_v5.py
python tests/test_v6.py
python experiments/v6_report.py --cached
```

The completed V6 raw run is in `results/v6/csv/configurations.csv`; the aggregate report is `WAVEFLY_V6_REPORT.md`. To rerun the full experiment:

```sh
HF_HOME=/workspace/.cache/huggingface HF_HUB_ENABLE_HF_TRANSFER=0 OPENBLAS_NUM_THREADS=4 python -u experiments/v6_experiment.py
python experiments/v6_report.py
```

V6 result: 17.24% mean held-out error at 1.999984 BPW, 48.77% worst-layer error, 17.09% new-domain error. The best isolated component was micro-scaling: 17.56% mean and 34.01% worst-layer error at 2 BPW.

Raw CSV/JSON measurements, plots, V5 results, reports, and invariant tests are included in this handoff.
