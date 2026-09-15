"""Bounded per-state beam with correlated prefix distortion before pruning.

Same bit-shift topology and packed paths. This is approximate full-Hessian
search: future cross-terms can rescue a path pruned at an earlier prefix.
"""
import torch
from trellis_quant import decode_paths, viterbi


@torch.inference_mode()
def correlated_beam(x, nodes, metric, beam=2, fallback=True):
    batch, width = x.shape
    if beam not in (1, 2, 4, 8):
        raise ValueError('supported beams: 1, 2, 4, 8')
    costs = torch.full((batch, 256, beam), float('inf'), device=x.device)
    costs[:, 0, 0] = 0
    paths = torch.zeros((batch, 256, beam), dtype=torch.long, device=x.device)
    # Cache prefix error projected through H. Updating this once per step avoids
    # repeatedly decoding every surviving prefix (quadratic kernel overhead).
    projected = torch.zeros((batch, 256, beam, width), device=x.device)
    symbols = torch.arange(16, device=x.device)[None, None, :, None]
    lows = torch.arange(16, device=x.device)[None, :, None, None]
    batch_rows = torch.arange(batch, device=x.device)[:, None]
    for step in range(width // 2):
        start, end = 2 * step, 2 * step + 2
        previous_cost = costs.reshape(batch, 16, 16, beam).permute(0, 2, 1, 3).reshape(batch, 16, 16 * beam)
        previous_paths = paths.reshape(batch, 16, 16, beam).permute(0, 2, 1, 3).reshape(batch, 16, 16 * beam)
        error = (nodes[None] - x[:, None, start:end]).reshape(batch, 16, 16, 2)
        local = ((error @ metric[start:end, start:end]) * error).sum(-1)
        candidates = previous_cost[:, :, None, :] + local[..., None]
        if start:
            cross = projected[..., start:end]
            cross = cross.reshape(batch, 16, 16, beam, 2).permute(0, 2, 1, 3, 4).reshape(batch, 16, 16 * beam, 2)
            candidates = candidates + 2 * torch.einsum('blpc,blsc->blsp', cross, error)
        selected_cost, selected = candidates.topk(beam, dim=-1, largest=False)
        parent = previous_paths[:, :, None, :].expand(batch, 16, 16, 16 * beam).gather(-1, selected)
        parent_index = ((selected // beam) * 16 + lows) * beam + selected % beam
        parent_projection = projected.reshape(batch, 256 * beam, width)[batch_rows, parent_index.reshape(batch, -1)]
        increment = (error.reshape(batch, 256, 2) @ metric[start:end])[:, :, None, :]
        projected = parent_projection.reshape(batch, 256, beam, width) + increment
        paths = (parent | (symbols << (4 * step))).reshape(batch, 256, beam)
        costs = selected_cost.reshape(batch, 256, beam)
    best = costs.reshape(batch, -1).argmin(1)
    rows = torch.arange(batch, device=x.device)
    ids = paths.reshape(batch, -1)[rows, best]
    q = decode_paths(ids, nodes, width=width)
    if fallback:
        old_q, old_ids = viterbi(x, nodes, metric.diag(), full_metric=metric)
        error, old_error = x - q, x - old_q
        improved = ((error @ metric) * error).sum(1) <= ((old_error @ metric) * old_error).sum(1)
        q = torch.where(improved[:, None], q, old_q)
        ids = torch.where(improved, ids, old_ids)
    return q, ids
