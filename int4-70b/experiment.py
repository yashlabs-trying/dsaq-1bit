"""CPU feasibility probe: shared block PCA + sparse residual corrections.

Input is E = reference weights - reconstructed base weights, exported as .npy.
This is not a quantizer, model evaluator, or inference kernel.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def encode(blocks, rank, count):
    """Fit a shared basis and retain largest remaining errors globally.

    FP16 storage rounding is applied before selecting sparse corrections.
    """
    if blocks.ndim != 2 or not np.isfinite(blocks).all():
        raise ValueError("Expected a finite 2-D block matrix")
    if not 0 <= rank <= min(blocks.shape) or not 0 <= count <= blocks.size:
        raise ValueError("Invalid rank or sparse count")
    mean = (blocks.mean(axis=0) if rank else np.zeros(blocks.shape[1])).astype(np.float16)
    centered = blocks - mean.astype(np.float32)
    if rank:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:rank].astype(np.float16)
        codes = (centered @ basis.astype(np.float32).T).astype(np.float16)
    else:
        basis = np.empty((0, blocks.shape[1]), dtype=np.float16)
        codes = np.empty((blocks.shape[0], 0), dtype=np.float16)
    approx = codes.astype(np.float32) @ basis.astype(np.float32) + mean.astype(np.float32)
    residual = (blocks - approx).ravel()
    indices = (np.argpartition(np.abs(residual), residual.size - count)[-count:]
               if count else np.empty(0, dtype=np.int64))
    index_dtype = np.uint32 if blocks.size <= 2**32 else np.uint64
    indices = np.sort(indices).astype(index_dtype)
    values = residual[indices].astype(np.float16)
    arrays = dict(mean=mean, basis=basis, codes=codes, indices=indices, values=values,
                  shape=np.array(blocks.shape, dtype=np.uint64))
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("FP16 overflow; normalize the input or change the format")
    return arrays


def decode(arrays):
    result = arrays['codes'].astype(np.float32) @ arrays['basis'].astype(np.float32)
    result += arrays['mean'].astype(np.float32)
    result.ravel()[arrays['indices']] += arrays['values'].astype(np.float32)
    return result


def evaluate(blocks, arrays):
    reconstructed = decode(arrays)
    error = blocks - reconstructed
    payload = sum(a.nbytes for a in arrays.values())
    return dict(payload_bytes=payload, correction_bits_per_weight=8 * payload / blocks.size,
                residual_relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(blocks), 1e-30)),
                residual_mse=float(np.mean(error.astype(np.float64)**2)),
                max_abs_error=float(np.max(np.abs(error))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--residual', type=Path)
    parser.add_argument('--block-size', type=int, default=64)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--sparse-fraction', type=float, default=.01)
    parser.add_argument('--output', type=Path, default=Path('results/synthetic.json'))
    args = parser.parse_args()
    if args.block_size <= 0 or not 0 <= args.sparse_fraction <= 1:
        parser.error('Invalid block size or sparse fraction')
    if args.residual:
        original = np.load(args.residual, allow_pickle=False).astype(np.float32)
        label = str(args.residual)
    else:
        rng = np.random.default_rng(42)
        original = (rng.normal(size=(512, 2)) @ rng.normal(size=(2, args.block_size))
                    + .03 * rng.normal(size=(512, args.block_size))).astype(np.float32)
        original.ravel()[::101] += 8
        label = 'SYNTHETIC low-rank plus outliers; favorable construction, not LLM evidence'
    if original.size == 0 or original.size % args.block_size:
        parser.error('Input must be nonempty and divisible by block size; no hidden padding')
    blocks = original.reshape(-1, args.block_size)
    count = int(blocks.size * args.sparse_fraction)
    hybrid = encode(blocks, args.rank, count)
    budget = sum(a.nbytes for a in hybrid.values())
    sparse_empty = encode(blocks, 0, 0)
    sparse_overhead = sum(a.nbytes for a in sparse_empty.values())
    entry_bytes = hybrid['indices'].dtype.itemsize + 2
    sparse_count = min(blocks.size, max(0, (budget - sparse_overhead) // entry_bytes))
    candidates = {'zero_correction': None, 'pca_only': encode(blocks, args.rank, 0),
                  'sparse_only_matched_budget': encode(blocks, 0, sparse_count), 'pca_plus_sparse': hybrid}
    report = dict(input=label, shape=list(original.shape), block_size=args.block_size,
                  rank=args.rank, sparse_count=count, results={})
    for name, arrays in candidates.items():
        if arrays is None:
            report['results'][name] = dict(payload_bytes=0, correction_bits_per_weight=0,
                                          residual_relative_l2=1.0 if np.any(blocks) else 0.0)
        else:
            report['results'][name] = evaluate(blocks, arrays)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.output.with_suffix('.npz')
    np.savez(archive, **hybrid)
    with np.load(archive, allow_pickle=False) as loaded:
        np.testing.assert_array_equal(decode(hybrid), decode(loaded))
    report['archive_bytes_including_container'] = archive.stat().st_size
    report['limitations'] = ['Basis fitted to these weights: measures weight compression, not generalization to other weights.',
        'No model quality, activation-weighted error, base quantizer metadata, KV cache, GPU memory, or inference speed measured.',
        'Full CPU SVD and dense reconstruction are feasibility tools, not a scalable 70B implementation.',
        'PCA optimizes squared reconstruction error; largest residual is not necessarily most important to model quality.']
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
