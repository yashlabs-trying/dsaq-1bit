"""Independent, reset-block bit-shift trellis prototype; not a QTIP reproduction.

256 states, 16 transitions per pair of weights, four pairs per reset block.
Each block stores a uint16 path. Viterbi is exact for additive pair costs only.
"""
import struct
from pathlib import Path
import numpy as np
import torch

HEADER = struct.Struct('<IIQe')
STATES, VECTOR, WIDTH = 256, 2, 8
METADATA_BYTES = HEADER.size + STATES * VECTOR * 2
FORCED_BLOCKS = METADATA_BYTES // 2


def decode_paths(ids, nodes):
    state = torch.zeros_like(ids, dtype=torch.long)
    values = []
    for step in range(4):
        symbol = (ids.long() >> (4 * step)) & 15
        state = ((state << 4) | symbol) & 255
        values.append(nodes[state])
    return torch.cat(values, dim=-1)


def viterbi(x, nodes, diagonal=None):
    """Optimal fixed-start path under diagonal, additive squared error."""
    if x.shape[1] != WIDTH:
        raise ValueError('expected eight-dimensional reset blocks')
    batch = x.shape[0]
    cost = torch.full((batch, STATES), float('inf'), device=x.device)
    cost[:, 0] = 0
    traces = []
    for step in range(4):
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
    for step in reversed(range(4)):
        ids |= (state & 15) << (4 * step)
        low = state >> 4
        state = 16 * traces[step][rows, low] + low
    return decode_paths(ids, nodes), ids


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


def force_mask(rows, groups, group, forced, device):
    return torch.arange(rows, device=device) * groups + group >= rows * groups - forced


def save_layer(path, ids, nodes, columns, seed, scale, forced=FORCED_BLOCKS):
    flat = ids.reshape(-1)
    if forced and torch.any(flat[-forced:] != 0):
        raise ValueError('omitted paths must be zero')
    body = flat[:-forced] if forced else flat
    data = HEADER.pack(ids.shape[0], columns, seed, scale)
    data += nodes.cpu().numpy().astype('<f2').tobytes()
    data += body.cpu().numpy().astype('<u2').tobytes()
    Path(path).write_bytes(data)


def load_layer(path, forced=FORCED_BLOCKS):
    data = Path(path).read_bytes()
    rows, columns, seed, scale = HEADER.unpack(data[:HEADER.size])
    nodes = torch.from_numpy(np.frombuffer(data[HEADER.size:METADATA_BYTES], dtype='<f2').astype(np.float32))
    nodes = nodes.reshape(STATES, VECTOR)
    stored = np.frombuffer(data[METADATA_BYTES:], dtype='<u2')
    ids = np.zeros(rows * columns // WIDTH, dtype=np.int64)
    ids[:len(stored)] = stored
    if len(stored) != len(ids) - forced:
        raise ValueError('incorrect serialized length')
    return torch.from_numpy(ids.reshape(rows, columns // WIDTH)), nodes, seed, scale
