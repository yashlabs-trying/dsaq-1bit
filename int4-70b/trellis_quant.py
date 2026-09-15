"""Independent, reset-block bit-shift trellis prototype; not a QTIP reproduction.

256 states, 16 transitions per pair of weights, four pairs per reset block.
Each block stores a uint16 path. Viterbi is exact for additive pair costs only.
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
if WIDTH not in (8, 16):
    raise ValueError('supported reset widths: 8, 16')
PATH_BYTES = WIDTH // 4
PATH_DTYPE = '<u2' if WIDTH == 8 else '<u4'
METADATA_BYTES = HEADER.size + STATES * VECTOR * 2
FORCED_BLOCKS = math.ceil(METADATA_BYTES / PATH_BYTES)


def decode_paths(ids, nodes, width=WIDTH):
    state = torch.zeros_like(ids, dtype=torch.long)
    values = []
    for step in range(width // VECTOR):
        symbol = (ids.long() >> (4 * step)) & 15
        state = ((state << 4) | symbol) & 255
        values.append(nodes[state])
    return torch.cat(values, dim=-1)


def viterbi(x, nodes, diagonal=None):
    """Optimal fixed-start path under diagonal, additive squared error."""
    width = x.shape[1]
    if width not in (8, 16):
        raise ValueError('expected eight- or sixteen-dimensional reset blocks')
    batch = x.shape[0]
    cost = torch.full((batch, STATES), float('inf'), device=x.device)
    cost[:, 0] = 0
    traces = []
    for step in range(width // VECTOR):
        # Destination s has predecessors (s >> 4) + 16*j.
        best, predecessor_high = cost.reshape(batch, 16, 16).min(dim=1)
        error = x[:, None, 2 * step:2 * step + 2] - nodes[None]
        if diagonal is not None:
            error_cost = (error.square() * diagonal[2 * step:2 * step + 2]).sum(-1)
        else:
            error_cost = error.square().sum(-1)
        cost = best.repeat_interleave(16, dim=1) + error_cost
        traces.append(predecessor_high)
    state = cost.argmin(dim=1)
    ids = torch.zeros(batch, dtype=torch.long, device=x.device)
    rows = torch.arange(batch, device=x.device)
    for step in reversed(range(width // VECTOR)):
        ids |= (state & 15) << (4 * step)
        low = state >> 4
        state = 16 * traces[step][rows, low] + low
    return decode_paths(ids, nodes, width), ids


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


def load_layer(path, forced=FORCED_BLOCKS):
    data = Path(path).read_bytes()
    rows, columns, seed, scale = HEADER.unpack(data[:HEADER.size])
    nodes = torch.from_numpy(np.frombuffer(data[HEADER.size:METADATA_BYTES], dtype='<f2').astype(np.float32))
    nodes = nodes.reshape(STATES, VECTOR)
    stored = np.frombuffer(data[METADATA_BYTES:], dtype=PATH_DTYPE)
    ids = np.zeros(rows * columns // WIDTH, dtype=np.int64)
    ids[:len(stored)] = stored
    if len(stored) != len(ids) - forced:
        raise ValueError('incorrect serialized length')
    return torch.from_numpy(ids.reshape(rows, columns // WIDTH)), nodes, seed, scale
