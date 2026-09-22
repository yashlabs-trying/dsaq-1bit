# WaveFly

WaveFly investigates compression of transformer weights through reproducible numerical experiments. Code, measurements, plots, and reports are included.

## Six proof experiments

### 1. DCT representation, quantization, and requantization

We implemented explicit orthonormal DCT tiles, coefficient selection, packed integer storage, residual serialization, decoding, and FP16 → INT8 → INT4 → 2-bit requantization. Full-coefficient reconstruction is numerically reversible and packed streams decode exactly. Near 2 BPW, the best hybrid configurations still showed roughly 51–58% functional error across the tested Qwen, OPT, and Pythia layers.

### 2. Reversible row and column clustering

Normalized row/column k-means, correlation and activation-aware orderings, block-local permutations, and metadata costs were tested. The best ordering, `both_k16_norm_centroid`, raised validation Top-32 spectral energy from 51.15% to 53.69% and reduced mean functional error from 65.32% to 59.12%. This was a real but partial improvement.

### 3. Clustered Haar DWT with mixed precision

Haar wavelets were applied after clustering with exact BPW accounting for scales, indices, metadata, and mixed-precision payloads across all 33 validation matrices. Mean held-out output error reached about 13.39% near 3.99 BPW, but remained about 50.87% near 1.89 BPW.

### 4. Saliency-protected columns with grouped bulk quantization (V4)

The best configuration protected 5% of columns using activation-RMS saliency at INT8, then used direct 1.5-bit grouped quantization with group size 32 and FP8 scales shared across four groups. It achieved **1.952 BPW, 27.55% mean held-out error, and 57.86% worst-layer error**. At 2.25 BPW, mean error crossed 25% at 23.96%; at 2.50 BPW it was 22.33%. Direct grouped quantization beat the Haar controls.

### 5. Layer-adaptive learned quantization and residual correction (V5)

V5 evaluated 3,237 configurations across the same 33 matrices on an RTX 2000 Ada GPU. Calibration-selected layer/group bit allocation, learned scalar/vector codebooks, and static residual corrections reached **18.71% mean held-out output error at 1.9998 global BPW**, with **44.37% worst-layer error**. A separate set of new code, math, and general-text passages gave 18.55% mean error. At 2.2498 BPW, mean held-out error was 16.76%; at 2.4998 BPW it was 15.22%.

The best system combines different quantizers across layers and uses low-rank or binary corrections on six matrices. Group and layer bit allocation provide the largest gains. Oracle and learned routing add no gain to the selected ≤2 BPW system. Some small hard layers receive approximately 5 BPW; the weight-count-weighted average over sampled matrices remains below 2 BPW.

**Comparison detail:** V5 follows the requested bias-excluded metric, `||X(W−W_hat)^T||F / ||XW^T||F`. Reproducing V4 gives 27.55% with bias included and 29.78% with bias excluded. Thus the fair V5 comparison is **29.78% → 18.71%**. This is relative layer-output error, not token error rate or perplexity. V5 global BPW is weighted by matrix size; earlier V4 summaries used the arithmetic mean of per-layer BPW.

The current verdict is **PARTIAL: Stage 1 (<20% mean error) reached; near-lossless behavior not reached**. V5 storage is explicitly accounted for but has not been physically packed into a deployable model. Perplexity, downstream accuracy, inference speed, and final kernel performance remain unmeasured.

### 6. Global bit-stealing and residual-aware allocation (V6)

V6 ran **24,084 configurations across all 33 matrices**. At the strict 2.00 BPW budget, the joint V6 system reached **17.24% mean held-out error**, but its worst layer was **48.77%**, worse than V5’s 44.37%. The strongest isolated component was nested micro-scaling: **17.56% mean and 34.01% worst-layer error at 2.00 BPW**. At 2.50 BPW, full V6 reached 13.34% mean error, but worst-layer error remained 48.71%.

Residual saliency and activation-aware grouping gave small improvements. Global bit-stealing alone regressed the mean to 19.15% at 2 BPW. The V6 verdict is **FAIL for its strict minimum target at ≤2 BPW**: it did not achieve both mean <15% and worst-layer <35%. The experiment is still useful because it identifies micro-scaling as the best component and shows that the remaining hard layer is not fixed by the tested allocation strategy. No perplexity or model-level accuracy claim is made.

## Reports and reproduction

- [`wavefly-proof/FINAL_REPORT.md`](wavefly-proof/FINAL_REPORT.md) — representation and universality experiments.
- [`wavefly-proof/CLUSTERING_REPORT.md`](wavefly-proof/CLUSTERING_REPORT.md) — reversible channel-clustering experiment.
- [`wavefly-proof/DWT_WAVEFLY_REPORT.md`](wavefly-proof/DWT_WAVEFLY_REPORT.md) — clustered Haar DWT and importance-aware mixed-precision experiment.
- [`wavefly-proof/WAVEFLY_V4_REPORT.md`](wavefly-proof/WAVEFLY_V4_REPORT.md) — saliency-protected columns and grouped bulk quantization.
- [`wavefly-proof/WAVEFLY_V5_REPORT.md`](wavefly-proof/WAVEFLY_V5_REPORT.md) — adaptive learned quantization, residual corrections, and routing tests.
- [`wavefly-proof/WAVEFLY_V6_REPORT.md`](wavefly-proof/WAVEFLY_V6_REPORT.md) — global bit-stealing, residual-driven saliency, activation-aware grouping, nested micro-scaling, and mixed precision.
- [`wavefly-proof/README.md`](wavefly-proof/README.md) — environment and reproduction commands.

All measured CSV/JSON results and generated plots are committed under `wavefly-proof/results`.
