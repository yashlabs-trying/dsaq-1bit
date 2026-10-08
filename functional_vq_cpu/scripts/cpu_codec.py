"""Minimal dependency-free reader/writer for the frozen Layer-0 packed codec."""
from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
import torch


SCALE_LEVELS = 16
PATTERNS_PER_FAMILY = 256


def unpack_fixed(blob, count, bits):
    raw = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[: count * bits]
    matrix = raw.reshape(count, bits).astype(np.uint16)
    values = (matrix * (1 << np.arange(bits, dtype=np.uint16))[None, :]).sum(1)
    return torch.from_numpy(values.astype(np.uint8))


def pack_fixed(values, bits):
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    matrix = ((values[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(matrix.reshape(-1), bitorder="little").tobytes()


def read_codec(path):
    path = Path(path)
    blob = path.read_bytes()
    pos = 0
    meta_len = struct.unpack_from("<I", blob, pos)[0]
    pos += 4
    meta = json.loads(blob[pos : pos + meta_len])
    pos += meta_len
    nblocks = meta["nblocks"]
    records = np.frombuffer(blob, dtype="<u2", count=nblocks, offset=pos).copy()
    pos += 2 * nblocks
    pattern_ids = torch.from_numpy((records & 0x0FFF).astype(np.int32))
    scale_ids = torch.from_numpy((records >> 12).astype(np.uint8))
    scale_count = len(meta["names"]) * SCALE_LEVELS
    scales = torch.from_numpy(
        np.frombuffer(blob, dtype="<f2", count=scale_count, offset=pos).copy()
    ).view(len(meta["names"]), SCALE_LEVELS)
    pos += 2 * scale_count
    checkpoint_count = math.prod(meta["checkpoint_shape"])
    checkpoint = torch.from_numpy(
        np.frombuffer(blob, dtype="<f2", count=checkpoint_count, offset=pos).copy()
    ).view(meta["checkpoint_shape"])
    pos += 2 * checkpoint_count
    pattern_count = math.prod(meta["pattern_shape"])
    pattern_bytes = (pattern_count * meta["pattern_bits"] + 7) // 8
    codes = unpack_fixed(blob[pos : pos + pattern_bytes], pattern_count, meta["pattern_bits"]).view(meta["pattern_shape"])
    pos += pattern_bytes
    hard = residual_ids = residual_q = residual_scale = None
    if meta["hard_count"]:
        mask_bytes = (nblocks + 7) // 8
        hard = torch.from_numpy(
            np.unpackbits(np.frombuffer(blob[pos : pos + mask_bytes], dtype=np.uint8), bitorder="little")[:nblocks].copy()
        ).bool()
        pos += mask_bytes
        residual_id_bytes = (meta["hard_count"] * meta["residual_bits"] + 7) // 8
        residual_ids = unpack_fixed(
            blob[pos : pos + residual_id_bytes], meta["hard_count"], meta["residual_bits"]
        )
        pos += residual_id_bytes
        residual_q = torch.from_numpy(
            np.frombuffer(blob, dtype=np.int8, count=meta["residual_k"] * 16, offset=pos).copy()
        ).view(meta["residual_k"], 16)
        pos += meta["residual_k"] * 16
        residual_scale = torch.from_numpy(
            np.frombuffer(blob, dtype="<f2", count=meta["residual_k"], offset=pos).copy()
        )
        pos += 2 * meta["residual_k"]
    assert pos == len(blob), (pos, len(blob))
    return meta, scales, checkpoint, codes, pattern_ids, scale_ids, hard, residual_ids, residual_q, residual_scale


def unblock(blocks, shape):
    rows, cols = shape
    return blocks.view(rows // 4, cols // 4, 4, 4).permute(0, 2, 1, 3).reshape(rows, cols)


def blockify(weight):
    rows, cols = weight.shape
    return weight.view(rows // 4, 4, cols // 4, 4).permute(0, 2, 1, 3).reshape(-1, 16)


def codec_state(path):
    state = read_codec(path)
    meta, scale_tables, checkpoint, codes, pattern_ids, scale_ids, hard, residual_ids, residual_q, residual_scale = state
    family = (torch.arange(codes.shape[0]) // PATTERNS_PER_FAMILY).long()
    patterns = checkpoint.float()[family[:, None], codes.long()]
    matrix_index = torch.empty(meta["nblocks"], dtype=torch.long)
    slices = {name: tuple(meta["slices"][name]) for name in meta["names"]}
    for index, name in enumerate(meta["names"]):
        start, end, _, _ = slices[name]
        matrix_index[start:end] = index
    scales = scale_tables.float()[matrix_index, scale_ids.long()]
    residual = torch.zeros(meta["nblocks"], 16)
    if hard is not None:
        residual[hard] = residual_q.float()[residual_ids.long()] * residual_scale.float()[residual_ids.long(), None]
    return {
        "meta": meta,
        "slices": slices,
        "patterns": patterns,
        "pattern_ids": pattern_ids,
        "scales": scales,
        "residual": residual,
    }


def decode_weights(path):
    state = codec_state(path)
    block_values = (
        state["patterns"][state["pattern_ids"].long()] * state["scales"][:, None]
        + state["residual"]
    )
    weights = {}
    for name, (start, end, rows, cols) in state["slices"].items():
        weights[name] = unblock(block_values[start:end], (rows, cols))
    return weights, state


def patch_ids(src, dst, pattern_ids):
    src, dst = Path(src), Path(dst)
    blob = bytearray(src.read_bytes())
    meta_len = struct.unpack_from("<I", blob, 0)[0]
    pos = 4 + meta_len
    records = np.frombuffer(blob, dtype="<u2", count=pattern_ids.numel(), offset=pos).copy()
    records = (records & 0xF000) | (pattern_ids.numpy().astype(np.uint16) & 0x0FFF)
    blob[pos : pos + 2 * len(records)] = records.astype("<u2").tobytes()
    dst.write_bytes(blob)


def patch_pattern_codes(src, dst, new_codes):
    src, dst = Path(src), Path(dst)
    blob = bytearray(src.read_bytes())
    meta_len = struct.unpack_from("<I", blob, 0)[0]
    meta = json.loads(blob[4 : 4 + meta_len])
    pos = 4 + meta_len + 2 * meta["nblocks"]
    pos += 2 * len(meta["names"]) * SCALE_LEVELS
    pos += 2 * math.prod(meta["checkpoint_shape"])
    count = math.prod(meta["pattern_shape"])
    byte_count = (count * meta["pattern_bits"] + 7) // 8
    packed = pack_fixed(new_codes.numpy(), meta["pattern_bits"])
    if len(packed) != byte_count:
        raise RuntimeError("Pattern-code packed size changed")
    blob[pos : pos + byte_count] = packed
    dst.write_bytes(blob)
