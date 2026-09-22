# WaveFly

WaveFly is an experimental investigation into spectral compression of transformer weights. The repository contains reproducible PyTorch experiments, packed codec implementations, raw measurements, plots, and technical reports.

The current evidence is deliberately negative where appropriate:

- Full-coefficient DCT storage and reconstruction work to floating-point tolerance.
- Quantized coefficient storage, decoding, and requantization work correctly.
- A fixed DCT basis does not preserve sufficient model-layer accuracy near two bits per weight in the tested models.
- Reversible row and column clustering improves local correlation and spectral concentration, but only earns a partial pass; average functional error remains too high.
- Hardware kernel performance and end-to-end perplexity have not been measured.

Start with:

- [`wavefly-proof/FINAL_REPORT.md`](wavefly-proof/FINAL_REPORT.md) — representation and universality experiments.
- [`wavefly-proof/CLUSTERING_REPORT.md`](wavefly-proof/CLUSTERING_REPORT.md) — reversible channel-clustering experiment.
- [`wavefly-proof/DWT_WAVEFLY_REPORT.md`](wavefly-proof/DWT_WAVEFLY_REPORT.md) — clustered Haar DWT and importance-aware mixed-precision experiment.
- [`wavefly-proof/WAVEFLY_V4_REPORT.md`](wavefly-proof/WAVEFLY_V4_REPORT.md) — saliency-protected columns and grouped bulk quantization.
- [`wavefly-proof/README.md`](wavefly-proof/README.md) — environment and reproduction commands.

All measured CSV/JSON results and generated plots are committed under `wavefly-proof/results`.
