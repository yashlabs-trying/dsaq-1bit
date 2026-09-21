"""Independent, reset-block bit-shift trellis prototype; not a QTIP reproduction.

256 states, 16 transitions per pair of weights, 8/16/32-weight reset blocks.
Each block stores a uint16/32/64 path. Viterbi is exact for additive pair costs
only; optional reranking scores selected endpoint winners with a full metric.
"""
import struct
import math
import os
from pathlib import Path
import numpy as np
import torch

HEADER = struct.Struct('<IIQe')
STATES, VECTOR = 256, 2
WIDTH = int(os.environ.get('FLASHRUN_TRELLIS_WIDTH', '8'))
if WIDTH not in (8, 16, 32):
    raise ValueError('supported reset widths: 8, 16, 32')
PATH_BYTES = WIDTH // 4
PATH_DTYPE = {8: '<u2', 16: '<u4', 32: '<u8'}[WIDTH]
METADATA_BYTES = HEADER.size + STATES * VECTOR * 2
FORCED_BLOCKS = math.ceil(METADATA_BYTES / PATH_BYTES)


def metadata_allocation(names, mode='per_matrix'):
    if mode == 'per_matrix':
        return {name: FORCED_BLOCKS for name in names}
    if mode != 'global_mlp':
        raise ValueError('unknown metadata budget mode')
    mlp = [name for name in names if '.mlp.' in name]
    if not mlp:
        raise ValueError('global MLP allocation requires MLP matrices')
    total = math.ceil(len(names) * METADATA_BYTES / PATH_BYTES)
    base, remainder = divmod(total, len(mlp))
    allocations = dict.fromkeys(names, 0)
    allocations.update({name: base + (i < remainder) for i, name in enumerate(mlp)})
    return allocations


def decode_paths(ids, nodes, width=WIDTH):
    state = torch.zeros_like(ids, dtype=torch.long)
    values = []
    for step in range(width // VECTOR):
        symbol = (ids.long() >> (4 * step)) & 15
        state = ((state << 4) | symbol) & 255
        values.append(nodes[state])
    return torch.cat(values, dim=-1)


def viterbi(x, nodes, diagonal=None, full_metric=None, top_paths=8):
    """Optimal additive search; optional full-metric reranking is approximate."""
    width = x.shape[1]
    if width not in (8, 16, 32):
        raise ValueError('expected eight-, sixteen-, or thirty-two-dimensional reset blocks')
    batch = x.shape[0]
    cost = torch.full((batch, STATES), float('inf'), device=x.device)
    cost[:, 0] = 0
    traces = []
    for step in range(width // VECTOR):
        # Destination s has predecessors (s >> 4) + 16*j.
        best, predecessor_high = cost.reshape(batch, 16, 16).min(dim=1)
        error = x[:, None, 2 * step:2 * step + 2] - nodes[None]
        if diagonal is not None:
            pair_diagonal = (diagonal[2 * step:2 * step + 2] if diagonal.ndim == 1 else
                             diagonal[:, None, 2 * step:2 * step + 2])
            error_cost = (error.square() * pair_diagonal).sum(-1)
        else:
            error_cost = error.square().sum(-1)
        cost = best.repeat_interleave(16, dim=1) + error_cost
        traces.append(predecessor_high)
    state = (cost.argmin(dim=1) if full_metric is None else
             cost.topk(min(top_paths, STATES), dim=1, largest=False).indices)
    ids = torch.zeros_like(state)
    rows = torch.arange(batch, device=x.device)
    if full_metric is not None:
        rows = rows[:, None].expand_as(state)
    for step in reversed(range(width // VECTOR)):
        ids |= (state & 15) << (4 * step)
        low = state >> 4
        state = 16 * traces[step][rows, low] + low
    decoded = decode_paths(ids, nodes, width)
    if full_metric is not None:
        error = decoded - x[:, None]
        losses = ((error @ full_metric) * error).sum(-1)
        selected = losses.argmin(1)
        rows = torch.arange(batch, device=x.device)
        decoded, ids = decoded[rows, selected], ids[rows, selected]
    return decoded, ids


def learn_nodes(w, seed, iterations=12, sample_pairs=32768):
    """K-means on transformed weight pairs; original model remains frozen."""
    generator = torch.Generator(device=w.device).manual_seed(seed)
    pairs = w.reshape(-1, 2)
    selection = torch.randint(pairs.shape[0], (min(sample_pairs, pairs.shape[0]),),
                              generator=generator, device=w.device)
    x = pairs[selection]
    centers = x[torch.randperm(len(x), generator=generator, device=w.device)[:STATES]].clone()
    for _ in range(iterations):
        distance = x.square().sum(1, keepdim=True) - 2 * x @ centers.t() + centers.square().sum(1)
        ids = distance.argmin(1)
        sums = torch.zeros_like(centers).index_add_(0, ids, x)
        counts = torch.bincount(ids, minlength=STATES).clamp_min(1)
        valid = torch.bincount(ids, minlength=STATES) > 0
        centers = torch.where(valid[:, None], sums / counts[:, None], centers)
    # Random node placement is held fixed; learning topology is a later upgrade.
    return centers.half().float()


def learn_outlier_nodes(w, h, seed, reserve=64, iterations=12, sample_pairs=32768):
    """Seed part of the existing node table from Hessian-important weight tails.

    This changes no serialized fields: ``reserve`` nodes come from the most
    consequential sampled pairs and the remaining nodes model the bulk.  The
    score is the diagonal-Hessian energy of a transformed weight pair.
    """
    if not 0 < reserve < STATES:
        raise ValueError('reserve must be between zero and the node count')
    rows, columns = w.shape
    if columns % VECTOR:
        raise ValueError('weight width must be divisible by the node width')
    generator = torch.Generator(device=w.device).manual_seed(seed)
    pairs = w.reshape(rows, columns // VECTOR, VECTOR)
    count = min(sample_pairs, pairs.shape[0] * pairs.shape[1])
    flat_ids = torch.randint(pairs.shape[0] * pairs.shape[1], (count,),
                             generator=generator, device=w.device)
    row_ids = flat_ids // pairs.shape[1]
    pair_ids = flat_ids % pairs.shape[1]
    x = pairs[row_ids, pair_ids]
    diagonal = h.diag().reshape(-1, VECTOR)[pair_ids].clamp_min(1e-12)
    importance = (x.square() * diagonal).sum(1)
    tail_count = max(reserve * 8, math.ceil(.05 * count))
    tail = importance.topk(min(tail_count, count)).indices

    def fit(values, weights, clusters):
        if len(values) < clusters:
            raise ValueError('not enough node-training samples')
        chosen = torch.multinomial(weights.clamp_min(1e-12), clusters,
                                   replacement=False, generator=generator)
        centers = values[chosen].clone()
        for _ in range(iterations):
            distance = (values.square().sum(1, keepdim=True) -
                        2 * values @ centers.t() + centers.square().sum(1))
            ids = distance.argmin(1)
            weighted = values * weights[:, None]
            sums = torch.zeros_like(centers).index_add_(0, ids, weighted)
            counts = torch.zeros(clusters, device=w.device).index_add_(0, ids, weights)
            centers = torch.where(counts[:, None] > 0,
                                  sums / counts.clamp_min(1e-12)[:, None], centers)
        return centers

    # Mild weighting keeps the bulk useful; the dedicated tail fit gets the
    # full importance score.  Random placement avoids concentrating tail nodes
    # in one trellis predecessor bucket.
    bulk_weights = importance.sqrt().clamp_min(1e-6)
    bulk = fit(x, bulk_weights, STATES - reserve)
    tail_nodes = fit(x[tail], importance[tail], reserve)
    nodes = torch.cat((bulk, tail_nodes))
    order = torch.randperm(STATES, generator=generator, device=w.device)
    return nodes[order].half().float()


def fixed_nodes(device, seed=123):
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(STATES, VECTOR, device=device, generator=generator).half().float()


def train_path_nodes(w, nodes, seed, iterations=8, sample_blocks=8192, neutral_zero=False):
    """Alternate Viterbi path selection and shared-node least-squares updates.

    Quantizer-only fitting under an additive Euclidean objective. Model weights
    are unchanged. This explicitly trains the table for its restricted topology.
    """
    nodes = nodes.clone()
    if neutral_zero:
        nodes[0] = 0
    generator = torch.Generator(device=w.device).manual_seed(seed)
    blocks = w.reshape(-1, WIDTH)
    selection = torch.randint(len(blocks), (min(sample_blocks, len(blocks)),),
                              generator=generator, device=w.device)
    x = blocks[selection]
    history = []
    for _ in range(iterations):
        q, paths = viterbi(x, nodes)
        history.append(float((x - q).square().mean()))
        state = torch.zeros_like(paths)
        sums = torch.zeros_like(nodes)
        counts = torch.zeros(STATES, device=w.device)
        for step in range(WIDTH // VECTOR):
            state = ((state << 4) | ((paths >> (4 * step)) & 15)) & 255
            sums.index_add_(0, state, x[:, 2 * step:2 * step + 2])
            counts.index_add_(0, state, torch.ones_like(state, dtype=torch.float))
        update = torch.where(counts[:, None] > 0, sums / counts.clamp_min(1)[:, None], nodes)
        if neutral_zero:
            update[0] = 0
        nodes = update.half().float()
    q, _ = viterbi(x, nodes)
    history.append(float((x - q).square().mean()))
    return nodes, history


def train_importance_nodes(w, h, nodes, seed, iterations=8, sample_blocks=8192,
                           neutral_zero=False):
    """Fit paths and nodes with per-position diagonal-Hessian importance."""
    nodes = nodes.clone()
    if neutral_zero:
        nodes[0] = 0
    generator = torch.Generator(device=w.device).manual_seed(seed)
    blocks = w.reshape(-1, WIDTH)
    groups = w.shape[1] // WIDTH
    selection = torch.randint(len(blocks), (min(sample_blocks, len(blocks)),),
                              generator=generator, device=w.device)
    x = blocks[selection]
    group_ids = selection % groups
    diagonal = h.diag().reshape(groups, WIDTH)[group_ids].clamp_min(1e-12)
    history = []
    for _ in range(iterations):
        q, paths = viterbi(x, nodes, diagonal=diagonal)
        history.append(float(((x - q).square() * diagonal).sum() / diagonal.sum()))
        state = torch.zeros_like(paths)
        sums = torch.zeros_like(nodes)
        counts = torch.zeros_like(nodes)
        for step in range(WIDTH // VECTOR):
            state = ((state << 4) | ((paths >> (4 * step)) & 15)) & 255
            values = x[:, 2 * step:2 * step + 2]
            weights = diagonal[:, 2 * step:2 * step + 2]
            sums.index_add_(0, state, values * weights)
            counts.index_add_(0, state, weights)
        update = torch.where(counts > 0, sums / counts.clamp_min(1e-12), nodes)
        if neutral_zero:
            update[0] = 0
        nodes = update.half().float()
    q, _ = viterbi(x, nodes, diagonal=diagonal)
    history.append(float(((x - q).square() * diagonal).sum() / diagonal.sum()))
    return nodes, history


def train_residual_specialists(w, h, nodes, seed, reserve=32, iterations=2,
                               sample_blocks=4096, neutral_zero=True):
    """Replace low-use states with topology-aware post-encoding specialists.

    Each trellis predecessor low nibble can reach only one 16-node destination
    bucket.  We therefore cluster high-impact residual targets separately for
    every reachable bucket. Candidate replacements are accepted bucket by
    bucket only when they reduce the sampled diagonal-Hessian objective.
    """
    if reserve <= 0 or reserve % 16 or reserve >= STATES:
        raise ValueError('reserve must be a positive multiple of 16 below 256')
    per_bucket = reserve // 16
    if per_bucket >= 16:
        raise ValueError('at least one ordinary state must remain per bucket')
    nodes = nodes.clone()
    if neutral_zero:
        nodes[0] = 0
    generator = torch.Generator(device=w.device).manual_seed(seed)
    blocks = w.reshape(-1, WIDTH)
    groups = w.shape[1] // WIDTH
    selection = torch.randint(len(blocks), (min(sample_blocks, len(blocks)),),
                              generator=generator, device=w.device)
    x = blocks[selection]
    group_ids = selection % groups
    diagonal = h.diag().reshape(groups, WIDTH)[group_ids].clamp_min(1e-12)

    def encode(table):
        q, paths = viterbi(x, table, diagonal=diagonal)
        loss = float(((x - q).square() * diagonal).sum() / diagonal.sum())
        return q, paths, loss

    q, paths, loss = encode(nodes)
    history = [dict(loss=loss, accepted_buckets=0)]
    for _ in range(iterations):
        state = torch.zeros_like(paths)
        usage = torch.zeros(STATES, device=w.device)
        buckets, targets, scores = [], [], []
        for step in range(WIDTH // VECTOR):
            predecessor_bucket = state & 15
            symbol = (paths >> (4 * step)) & 15
            state = ((state << 4) | symbol) & 255
            usage.index_add_(0, state, torch.ones_like(state, dtype=torch.float))
            target = x[:, 2 * step:2 * step + 2]
            score = ((target - q[:, 2 * step:2 * step + 2]).square() *
                     diagonal[:, 2 * step:2 * step + 2]).sum(1)
            buckets.append(predecessor_bucket)
            targets.append(target)
            scores.append(score)
        buckets = torch.cat(buckets)
        targets = torch.cat(targets)
        scores = torch.cat(scores)
        accepted = 0
        for bucket in range(16):
            mask = buckets == bucket
            if int(mask.sum()) < per_bucket:
                continue
            values, weights = targets[mask], scores[mask]
            tail_count = min(len(values), max(per_bucket * 64, math.ceil(.05 * len(values))))
            tail = weights.topk(tail_count).indices
            values, weights = values[tail], weights[tail].clamp_min(1e-12)
            chosen = torch.multinomial(weights, per_bucket, replacement=False,
                                       generator=generator)
            centers = values[chosen].clone()
            for _fit in range(6):
                distance = (values.square().sum(1, keepdim=True) -
                            2 * values @ centers.t() + centers.square().sum(1))
                assignment = distance.argmin(1)
                sums = torch.zeros_like(centers).index_add_(
                    0, assignment, values * weights[:, None])
                counts = torch.zeros(per_bucket, device=w.device).index_add_(
                    0, assignment, weights)
                centers = torch.where(counts[:, None] > 0,
                                      sums / counts.clamp_min(1e-12)[:, None], centers)
            candidates = torch.arange(bucket * 16, (bucket + 1) * 16, device=w.device)
            if neutral_zero and bucket == 0:
                candidates = candidates[candidates != 0]
            replace = candidates[usage[candidates].argsort()[:per_bucket]]
            proposal = nodes.clone()
            proposal[replace] = centers.half().float()
            if neutral_zero:
                proposal[0] = 0
            new_q, new_paths, new_loss = encode(proposal)
            if new_loss < loss:
                nodes, q, paths, loss = proposal, new_q, new_paths, new_loss
                accepted += 1
        history.append(dict(loss=loss, accepted_buckets=accepted))
        if not accepted:
            break
    return nodes, history


def force_mask(rows, groups, group, forced, device):
    return torch.arange(rows, device=device) * groups + group >= rows * groups - forced


def save_layer(path, ids, nodes, columns, seed, scale, forced=FORCED_BLOCKS):
    flat = ids.reshape(-1)
    if forced and torch.any(flat[-forced:] != 0):
        raise ValueError('omitted paths must be zero')
    body = flat[:-forced] if forced else flat
    data = HEADER.pack(ids.shape[0], columns, seed, scale)
    data += nodes.cpu().numpy().astype('<f2').tobytes()
    data += body.cpu().numpy().astype(PATH_DTYPE).tobytes()
    Path(path).write_bytes(data)


def load_layer(path, forced=None):
    data = Path(path).read_bytes()
    rows, columns, seed, scale = HEADER.unpack(data[:HEADER.size])
    nodes = torch.from_numpy(np.frombuffer(data[HEADER.size:METADATA_BYTES], dtype='<f2').astype(np.float32))
    nodes = nodes.reshape(STATES, VECTOR)
    stored = np.frombuffer(data[METADATA_BYTES:], dtype=PATH_DTYPE)
    ids = np.zeros(rows * columns // WIDTH, dtype=np.int64)
    if forced is None:
        forced = len(ids) - len(stored)
    if forced < 0:
        raise ValueError('stored path count exceeds matrix dimensions')
    ids[:len(stored)] = stored
    if len(stored) != len(ids) - forced:
        raise ValueError('incorrect serialized length')
    return torch.from_numpy(ids.reshape(rows, columns // WIDTH)), nodes, seed, scale
