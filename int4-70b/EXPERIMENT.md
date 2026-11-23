# Shared residual patterns + sparse exceptions

First hypothesis for FlashRun: define E = W_reference - W_base, with W_base reconstructed from a compressed checkpoint. Encode blocks of E with a shared PCA mean/basis and per-block coefficients. Encode the largest errors remaining after PCA reconstruction as global index + value pairs.

Reconstruction: W_hat = W_base + E_PCA + E_sparse.

The reference may be BF16 weights exported as float32 or a 4-bit reconstruction. State which: recovering a 4-bit reference does not establish BF16 quality. There is no real checkpoint or base quantizer in this prototype yet.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest -v
.\.venv\Scripts\python.exe experiment.py
.\.venv\Scripts\python.exe experiment.py --residual path/to/error.npy --block-size 64 --rank 2 --sparse-fraction 0.01 --output results/real-layer.json
```

Input arrays are flattened in C order into fixed-width blocks. Export paired reference/base arrays with identical ordering before calculating E. No checkpoint download is performed. Start with a small representative layer subset: full SVD and dense decoding are not suitable for a full 70B checkpoint.

## Accounting and interpretation

Stored PCA mean, basis, codes, and sparse values are FP16. Sparse indices are uint32 or uint64 according to array size. Shape metadata is counted. Reconstruction metrics use these rounded stored values, and the archive is round-trip checked.

Report both payload bytes and NPZ container bytes. Correction bits per weight are ADDITIONAL to base weights and their metadata. Neither number includes model runtime allocations. At block width 64 and rank 2, coefficients alone cost 0.5 bits per weight; a tiny latent is a hypothesis, not an assumption.

Compare zero correction, PCA only, sparse only at no greater payload budget, and the hybrid. The synthetic input intentionally contains low-rank structure and outliers: success on it proves plumbing, not that LLM quantization errors have this structure.

## Next gate

Measure real quantization residuals across multiple representative layers. Sweep block width, PCA rank, and sparse budget. Compare against strong low-bit and mixed-precision baselines at matched TOTAL bytes. Evaluate held-out activation/output error and ultimately model tasks. Magnitude-selected exceptions are a starting baseline, not a sensitivity estimator.

AE and VQ are future alternatives, not implemented. Runtime adaptation and custom GPU kernels follow only if the representation earns its byte budget.
