# WaveFly

## Four proof experiments

### 1. DCT representation, quantization, and requantization

We implemented explicit orthonormal DCT tiles, coefficient selection, packed integer storage, residual serialization, decoding, and FP16 → INT8 → INT4 → 2-bit requantization. Full-coefficient reconstruction is numerically reversible and packed streams decode exactly. Near 2 BPW, the best hybrid configurations still showed roughly 51–58% functional error across the tested Qwen, OPT, and Pythia layers.

### 2. Reversible row and column clustering

Normalized row/column k-means, correlation and activation-aware orderings, block-local permutations, and metadata costs were tested. The best ordering, `both_k16_norm_centroid`, raised validation Top-32 spectral energy from 51.15% to 53.69% and reduced mean functional error from 65.32% to 59.12%. This was a real but partial improvement.

### 3. Clustered Haar DWT with mixed precision

Haar wavelets were applied after clustering with exact BPW accounting for scales, indices, metadata, and mixed-precision payloads across all 33 validation matrices. Mean held-out output error reached about 13.39% near 3.99 BPW, but remained about 50.87% near 1.89 BPW.

### 4. Saliency-protected columns with grouped bulk quantization (V4)

The best configuration protected 5% of columns using activation-RMS saliency at INT8, then used direct 1.5-bit grouped quantization with group size 32 and FP8 scales shared across four groups. It achieved **1.952 BPW, 27.55% mean held-out error, and 57.86% worst-layer error**. At 2.25 BPW, mean error crossed 25% at 23.96%; at 2.50 BPW it was 22.33%. Direct grouped quantization beat the Haar controls.

The current conclusion is **PARTIAL**: WaveFly can store, decode, and requantize the tested weights, and V4 reaches the 25% error range, but it does not yet preserve production-quality model behavior at approximately 2 BPW. Perplexity, downstream accuracy, inference speed, and hardware kernel performance remain unmeasured.

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
