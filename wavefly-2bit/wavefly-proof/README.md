# WaveFly numerical proof

Experiments 1–2 only. See FINAL_REPORT.md for outcomes and limitations.

Run on a CUDA host with Python, PyTorch, transformers 4.56.2, accelerate, safetensors, pandas, numpy, matplotlib:

```sh
HF_HUB_ENABLE_HF_TRANSFER=0 python experiments/run.py
HF_HUB_ENABLE_HF_TRANSFER=0 python experiments/full_layers.py
python tests/test_codec.py
python experiments/report.py
```

`wavefly/core.py` contains explicit DCT, quantization, residual serialization, deserialization, metrics, and exact byte accounting. `experiments/run.py` runs round-trip, synthetic, real weight, residual, precision, and tile-size sweeps. `full_layers.py` independently checks full linear outputs and sequential real-tile requantization. Models are downloaded from Hugging Face; hashes and hardware details are recorded under results/json. Packed bytes and all aggregate configuration measurements are retained. Random sample coordinates are reproducible from the fixed seed and run order rather than individually stored. Do not infer inference speed or end-to-end model accuracy from these tests.

The follow-up reversible channel-clustering experiment is reported in `CLUSTERING_REPORT.md`:

```sh
HF_HUB_ENABLE_HF_TRANSFER=0 python experiments/clustering_experiment.py
python experiments/clustering_report.py
```

It compares raw, row, column, joint, spherical/correlation, activation-aware, magnitude-aware, and block-local permutations. It verifies exact permutation undo, reconstructs complete matrices for functional error, counts permutation metadata, and retains all requested curves and heatmaps under `results/clustering`.

The Haar DWT follow-up uses held-out activations and exact mixed-precision BPW accounting:

```sh
HF_HUB_ENABLE_HF_TRANSFER=0 python experiments/dwt_experiment.py
python tests/test_wavelet.py
python experiments/dwt_report.py
```

See `DWT_WAVEFLY_REPORT.md` and `results/dwt`.

WaveFly V4 tests saliency-protected columns against direct and Haar grouped bulk quantization:

```sh
HF_HUB_ENABLE_HF_TRANSFER=0 python experiments/v4_experiment.py
python experiments/v4_report.py
```

See `WAVEFLY_V4_REPORT.md` and `results/v4`.
