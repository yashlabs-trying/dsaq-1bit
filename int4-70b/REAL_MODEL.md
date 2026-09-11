# Real-model experiment

Run `python model_experiment.py` in a CUDA PyTorch environment after installing `requirements-model.txt`.

Then run `python -m unittest test_model_experiment -v` and `python runtime_experiment.py` to validate bit packing and the reference GPU-resident compressed runtime. On the supplied pod, set `HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/flashrun/hf-cache` before these commands. The functioning environment is `/tmp/flashrun-venv`; persistent experiment files are in `/workspace/flashrun`. A second environment at `/workspace/flashrun/.venv` also finished installing, but was not used for the successful run. The temporary environment must be recreated after a pod reset.

See [measured findings](results/REAL_MODEL_FINDINGS.md). Complete remote package versions were saved to `/workspace/flashrun/environment-freeze.txt`.

This is an end-to-end **quality experiment**, not a deployable compressed inference engine. It downloads Qwen/Qwen2.5-0.5B and WikiText-2, evaluates BF16, encodes all transformer linear weights, saves per-layer experimental archives, reconstructs each candidate into BF16 tensors, and evaluates all candidates on identical text.

The initial RunPod GPU is an NVIDIA L4, not an RTX 3090. No speed claim applies to the target hardware.

## Variants

- Original BF16.
- Groupwise affine min-max 2-bit and 4-bit weights, including FP16 scales and offsets.
- 2-bit plus rank-2 PCA residual correction, block width 64.
- 2-bit plus PCA plus corrections for the largest remaining 1% of residual elements.
- 2-bit plus sparse-only corrections at no greater correction tensor-payload budget.

PCA is fitted to weight residuals against the BF16 reference. The evaluation text is not used to fit the representation. These naive quantizers are controls, not state-of-the-art compression baselines.

Embeddings, tied output head, norms, and biases remain at original precision and their unique parameter bytes are counted. The per-layer archives contain multiple candidates to enable controlled comparisons: archive size is not deployment size. Tensor payload accounting excludes serialization metadata and runtime memory, and is explicitly labeled accordingly.

## Evaluation limitations

First 4096 tokens of WikiText-2 test text, split into independent 256-token windows (4080 next-token predictions). This is a short exploratory perplexity comparison, not a standard full-corpus benchmark or proof of downstream task accuracy. Lower perplexity is better.

Model revision, software versions, dataset fingerprint, per-layer weight errors, and measured perplexities are recorded in `real_results/report.json`. Dense BF16 reconstruction is intentional to isolate representation quality. It neither establishes 70B memory feasibility nor measures fused decompression speed.
