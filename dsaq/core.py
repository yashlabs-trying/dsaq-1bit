from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


FAMILY_PARAMS = {"row": 2, "column": 2, "plane": 3, "quadratic": 6}


@dataclass
class BlockLayout:
    rows: int
    cols: int
    block_size: int
    padded_rows: int
    padded_cols: int
    valid: torch.Tensor

    @property
    def block_weights(self) -> torch.Tensor:
        return self.valid.sum(dim=1)


def stable_u64(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


def split_is_calibration(matrix: str, block_ids: torch.Tensor, fraction: float = 0.5) -> torch.Tensor:
    values = [stable_u64(matrix, int(i)) % 10_000 < int(fraction * 10_000) for i in block_ids.cpu()]
    return torch.tensor(values, dtype=torch.bool, device=block_ids.device)


def matrix_to_blocks(weight: torch.Tensor, block_size: int) -> tuple[torch.Tensor, BlockLayout]:
    rows, cols = weight.shape
    pr = math.ceil(rows / block_size) * block_size
    pc = math.ceil(cols / block_size) * block_size
    padded = torch.zeros((pr, pc), dtype=weight.dtype, device=weight.device)
    padded[:rows, :cols] = weight
    valid2d = torch.zeros((pr, pc), dtype=torch.bool, device=weight.device)
    valid2d[:rows, :cols] = True
    blocks = padded.view(pr // block_size, block_size, pc // block_size, block_size)
    blocks = blocks.permute(0, 2, 1, 3).reshape(-1, block_size * block_size)
    valid = valid2d.view(pr // block_size, block_size, pc // block_size, block_size)
    valid = valid.permute(0, 2, 1, 3).reshape(-1, block_size * block_size)
    return blocks, BlockLayout(rows, cols, block_size, pr, pc, valid)


def blocks_to_matrix(blocks: torch.Tensor, layout: BlockLayout) -> torch.Tensor:
    b = layout.block_size
    grid = blocks.view(layout.padded_rows // b, layout.padded_cols // b, b, b)
    padded = grid.permute(0, 2, 1, 3).reshape(layout.padded_rows, layout.padded_cols)
    return padded[: layout.rows, : layout.cols]


def hadamard_matrix(n: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if n <= 0 or n & (n - 1):
        raise ValueError("Hadamard size must be a positive power of two")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def hadamard_blocks(blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    h = hadamard_matrix(block_size, device=blocks.device, dtype=blocks.dtype)
    return torch.matmul(blocks.view(-1, block_size, block_size), h).reshape(blocks.shape)


def design_matrices(block_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, block_size, device=device, dtype=dtype)
    rr, cc = torch.meshgrid(axis, axis, indexing="ij")
    one = torch.ones_like(rr)
    return {
        "row": torch.stack((one, rr), -1).reshape(-1, 2),
        "column": torch.stack((one, cc), -1).reshape(-1, 2),
        "plane": torch.stack((one, rr, cc), -1).reshape(-1, 3),
        "quadratic": torch.stack((one, rr, cc, rr * cc, rr.square(), cc.square()), -1).reshape(-1, 6),
    }


def _fit_surface(magnitude: torch.Tensor, valid: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Edge blocks need a masked solve; full blocks share one pseudoinverse.
    if bool(valid.all()):
        coef = magnitude @ torch.linalg.pinv(x).T
        pred = (coef @ x.T).clamp_min_(0)
        return pred, coef
    preds, coefs = [], []
    for y, mask in zip(magnitude, valid):
        xm = x[mask]
        coef = torch.linalg.lstsq(xm, y[mask].unsqueeze(1)).solution.squeeze(1)
        preds.append((x @ coef).clamp_min(0))
        coefs.append(coef)
    return torch.stack(preds), torch.stack(coefs)


def weighted_error(original: torch.Tensor, reconstructed: torch.Tensor, importance: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denom = (original.square() * importance * valid).sum(1).clamp_min(1e-12)
    return (((original - reconstructed).square() * importance * valid).sum(1) / denom)


def raw_relative_error(original: torch.Tensor, reconstructed: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    denom = (original.square() * valid).sum(1).clamp_min(1e-12)
    return (((original - reconstructed).square() * valid).sum(1) / denom)


def reconstruct_naive(blocks: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (blocks.abs() * valid).sum(1) / valid.sum(1).clamp_min(1)
    return blocks.sign() * scale[:, None], scale


def reconstruct_dsa(blocks: torch.Tensor, valid: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor]:
    magnitude = blocks.abs()
    designs = design_matrices(block_size, blocks.device, blocks.dtype)
    candidates, coefs, families = [], [], []
    for family, x in designs.items():
        pred, coef = _fit_surface(magnitude, valid, x)
        recon = blocks.sign() * pred
        err = raw_relative_error(blocks, recon, valid)
        candidates.append(recon)
        coefs.append(coef)
        families.append(family)
    errors = torch.stack([raw_relative_error(blocks, c, valid) for c in candidates], 0)
    winner = errors.argmin(0)
    output = torch.empty_like(blocks)
    param_counts = torch.empty(blocks.shape[0], dtype=torch.int64, device=blocks.device)
    labels: list[str] = []
    for i in range(blocks.shape[0]):
        j = int(winner[i])
        output[i] = candidates[j][i]
        labels.append(families[j])
        param_counts[i] = FAMILY_PARAMS[families[j]]
    return output, errors.min(0).values, labels, param_counts


def structure_features(blocks: torch.Tensor, valid: torch.Tensor, block_size: int, regression_error: torch.Tensor) -> tuple[np.ndarray, list[str]]:
    x = blocks.abs().view(-1, block_size, block_size)
    mask = valid.view_as(x)
    count = mask.sum((1, 2)).clamp_min(1)
    mean = (x * mask).sum((1, 2)) / count
    var = (((x - mean[:, None, None]) * mask).square().sum((1, 2)) / count).clamp_min(1e-12)
    cv = var.sqrt() / mean.clamp_min(1e-8)
    row_diff = ((x[:, 1:] - x[:, :-1]).abs() * (mask[:, 1:] & mask[:, :-1])).sum((1, 2))
    col_diff = ((x[:, :, 1:] - x[:, :, :-1]).abs() * (mask[:, :, 1:] & mask[:, :, :-1])).sum((1, 2))
    smooth = 1.0 / (1.0 + (row_diff + col_diff) / (count * mean.clamp_min(1e-8)))
    row_mean = (x * mask).sum(2) / mask.sum(2).clamp_min(1)
    col_mean = (x * mask).sum(1) / mask.sum(1).clamp_min(1)
    row_corr = _adjacent_corr(row_mean)
    col_corr = _adjacent_corr(col_mean)
    v = torch.ones((x.shape[0], block_size, 1), device=x.device, dtype=x.dtype) / math.sqrt(block_size)
    for _ in range(4):
        u = torch.bmm(x, v); u = u / u.norm(dim=1, keepdim=True).clamp_min(1e-8)
        v = torch.bmm(x.transpose(1, 2), u); v = v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
    sigma = torch.bmm(u.transpose(1, 2), torch.bmm(x, v)).flatten()
    spectral = sigma.square() / x.square().sum((1, 2)).clamp_min(1e-12)
    # Eight-bin normalized magnitude entropy, looped only over blocks for stable edge handling.
    entropy = []
    for block, m in zip(x, mask):
        values = block[m]
        hist = torch.histc(values, bins=8, min=0, max=float(values.max().clamp_min(1e-8)))
        p = hist / hist.sum().clamp_min(1)
        entropy.append(-(p[p > 0] * p[p > 0].log2()).sum() / 3.0)
    entropy_t = torch.stack(entropy)
    r2 = 1.0 - regression_error
    feats = torch.stack((row_corr, col_corr, spectral, smooth, cv, entropy_t, r2), 1)
    names = ["row_corr", "column_corr", "spectral_concentration", "local_smoothness", "magnitude_cv", "entropy", "regression_r2"]
    return feats.float().cpu().numpy(), names


def _adjacent_corr(values: torch.Tensor) -> torch.Tensor:
    a, b = values[:, :-1], values[:, 1:]
    a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
    return ((a * b).sum(1) / (a.square().sum(1) * b.square().sum(1)).sqrt().clamp_min(1e-8)).abs()


def fit_router(features: np.ndarray, gains: np.ndarray, calibration: np.ndarray, ridge: float = 1e-2,
               minimum_useful_gain: float = 0.01) -> dict[str, np.ndarray | float]:
    x = features[calibration]
    y = gains[calibration]
    mean, scale = x.mean(0), x.std(0) + 1e-8
    z = (x - mean) / scale
    design = np.column_stack((np.ones(len(z)), z))
    reg = np.eye(design.shape[1]) * ridge; reg[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    scores = np.column_stack((np.ones(len(features)), (features - mean) / scale)) @ coef
    thresholds = np.quantile(scores[calibration], np.linspace(0.05, 0.95, 91))
    truth = gains[calibration] >= minimum_useful_gain
    best_t, best_ba = float(thresholds[0]), -1.0
    for t in thresholds:
        pred = scores[calibration] > t
        tpr = (pred[truth].mean() if truth.any() else 0.0)
        tnr = ((~pred)[~truth].mean() if (~truth).any() else 0.0)
        if (tpr + tnr) / 2 > best_ba:
            best_ba, best_t = (tpr + tnr) / 2, float(t)
    return {"mean": mean, "scale": scale, "coef": coef, "scores": scores, "threshold": best_t,
            "minimum_useful_gain": minimum_useful_gain}


def candidate_metadata_bits(distribution: str, k: int, n_blocks: int, n_matrices: int = 1) -> int:
    index_bits = math.ceil(math.log2(k)) * n_blocks
    if distribution == "global":
        distribution_bits = 32 * n_matrices
    else:
        distribution_bits = 32 * n_blocks
    return index_bits + distribution_bits


def true_bpw(sign_weights: int, metadata_bits: int) -> float:
    return (sign_weights + metadata_bits) / max(sign_weights, 1)


def seeded_candidates(
    blocks: torch.Tensor,
    valid: torch.Tensor,
    block_ids: Iterable[int],
    matrix_name: str,
    distribution: str,
    k: int,
    global_seed: int,
    global_stats: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Return [K, B, N] signed candidates, each reproducible from block identity."""
    magnitude = blocks.abs()
    valid_f = valid.float()
    count = valid_f.sum(1).clamp_min(1)
    mean = (magnitude * valid_f).sum(1) / count
    std = (((magnitude - mean[:, None]).square() * valid_f).sum(1) / count).sqrt().clamp_min(1e-8)
    median = torch.stack([m[v].median() for m, v in zip(magnitude, valid)])
    mad = torch.stack([(m[v] - med).abs().median() for m, v, med in zip(magnitude, valid, median)]).clamp_min(1e-8)
    out = []
    ids = list(map(int, block_ids))
    max_block_id = max(ids, default=-1)
    index = torch.tensor(ids, dtype=torch.long, device=blocks.device)
    for candidate_id in range(k):
        # One counter stream per matrix/candidate. Block ID selects a stable row,
        # so a decoder can reproduce a block by seeking to its deterministic offset.
        generator = torch.Generator(device=blocks.device)
        generator.manual_seed(stable_u64(global_seed, matrix_name, candidate_id, distribution) & ((1 << 63) - 1))
        shape = (max_block_id + 1, blocks.shape[1])
        if distribution in {"gaussian", "global"}:
            random = torch.randn(shape, generator=generator, device=blocks.device, dtype=blocks.dtype)[index]
        else:
            random = torch.rand(shape, generator=generator, device=blocks.device, dtype=blocks.dtype)[index]
        if distribution == "uniform":
            sample = random * (2 * mean[:, None])
        elif distribution == "gaussian":
            sample = (random * std[:, None] + mean[:, None]).abs()
        elif distribution == "robust":
            u = random.clamp_(1e-6, 1 - 1e-6)
            laplace = torch.where(u < .5, torch.log(2 * u), -torch.log(2 * (1 - u)))
            sample = (median[:, None] + (mad[:, None] / math.log(2)) * laplace).abs()
        elif distribution == "global":
            if global_stats is None:
                raise ValueError("global distribution requires shared mean/std")
            sample = (random * global_stats[1] + global_stats[0]).abs()
        else:
            raise ValueError(distribution)
        out.append(sample * blocks.sign())
    return torch.stack(out)


def select_seeded(candidates: torch.Tensor, original: torch.Tensor, valid: torch.Tensor, importance: torch.Tensor, criterion: str) -> tuple[torch.Tensor, torch.Tensor]:
    if criterion == "raw":
        errors = torch.stack([raw_relative_error(original, c, valid) for c in candidates])
    elif criterion == "weighted":
        errors = torch.stack([weighted_error(original, c, importance, valid) for c in candidates])
    else:
        raise ValueError(criterion)
    winner = errors.argmin(0)
    selected = candidates[winner, torch.arange(original.shape[0], device=original.device)]
    return selected, winner
