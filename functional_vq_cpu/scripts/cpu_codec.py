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


def unpack_uint(blob, count, bits):
    """Unpack little-endian fixed-width integers up to 32 bits."""
    raw = np.unpackbits(np.frombuffer(blob, dtype=np.uint8), bitorder="little")[: count * bits]
    matrix = raw.reshape(count, bits).astype(np.uint64)
    values = (matrix * (np.uint64(1) << np.arange(bits, dtype=np.uint64))[None, :]).sum(1)
    return torch.from_numpy(values.astype(np.int64))


def pack_fixed(values, bits):
    values = np.asarray(values, dtype=np.uint16).reshape(-1)
    matrix = ((values[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(matrix.reshape(-1), bitorder="little").tobytes()


def read_codec(path):
    return read_codec_blob(Path(path).read_bytes())


def read_codec_blob(blob):
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
    return codec_state_from_raw(read_codec(path))


def codec_state_from_raw(state):
    meta, scale_tables, checkpoint, codes, pattern_ids, scale_ids, hard, residual_ids, residual_q, residual_scale = state
    if checkpoint.shape[0] == 1:
        family = torch.zeros(codes.shape[0], dtype=torch.long)
    else:
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
    path = Path(path)
    blob = path.read_bytes()
    meta_len = struct.unpack_from("<I", blob, 0)[0]
    meta = json.loads(blob[4:4+meta_len])
    if meta.get("version") == 3 and meta.get("layout") == "selective_joint_residual":
        return decode_selective_joint_residual(blob, meta, meta_len)
    if meta.get("version") == 2 and meta.get("layout") == "joint_mlp_packages":
        return decode_joint_packages(blob, meta, meta_len)
    state = codec_state(path)
    block_values = (
        state["patterns"][state["pattern_ids"].long()] * state["scales"][:, None]
        + state["residual"]
    )
    weights = {}
    for name, (start, end, rows, cols) in state["slices"].items():
        weights[name] = unblock(block_values[start:end], (rows, cols))
    return weights, state


def decode_legacy_blob(blob):
    state = codec_state_from_raw(read_codec_blob(blob))
    block_values = (state["patterns"][state["pattern_ids"].long()]
                    * state["scales"][:, None] + state["residual"])
    weights = {name: unblock(block_values[s:e], (rows, cols))
               for name, (s, e, rows, cols) in state["slices"].items()}
    return weights, state


def decode_selective_joint_residual(blob, meta, meta_len):
    import lzma
    pos = 4+meta_len
    parent_size = meta["compressed_parent_bytes"]
    parent_blob = lzma.decompress(blob[pos:pos+parent_size])
    pos += parent_size
    weights, parent_state = decode_legacy_blob(parent_blob)
    count = meta["critical_count"]
    index_bits = meta["critical_index_bits"]
    index_bytes = (count*index_bits+7)//8
    package_indices = unpack_uint(blob[pos:pos+index_bytes], count, index_bits)
    pos += index_bytes
    correction_ids = torch.from_numpy(np.frombuffer(
        blob, dtype=np.uint8, count=count, offset=pos).copy()).long()
    pos += count
    k = meta["correction_k"]
    correction_q = torch.from_numpy(np.frombuffer(
        blob, dtype=np.int8, count=k*48, offset=pos).copy()).view(k,48).float()
    pos += k*48
    correction_scale = torch.from_numpy(np.frombuffer(
        blob, dtype="<f2", count=k, offset=pos).copy()).float()
    pos += 2*k
    pos += meta.get("padding_bytes", 0)
    assert pos == len(blob), (pos, len(blob))
    corrections = correction_q[correction_ids]*correction_scale[correction_ids,None]

    slices = parent_state["slices"]
    g_start = slices["gate_proj"][0]
    u_start = slices["up_proj"][0]
    d_start = slices["down_proj"][0]
    neuron_groups = meta["neuron_groups"]
    hidden_chunks = meta["hidden_chunks"]
    neuron = package_indices//hidden_chunks
    chunk = package_indices%hidden_chunks
    g_index = g_start+package_indices
    u_index = u_start+package_indices
    d_index = d_start+chunk*neuron_groups+neuron
    gate_blocks = blockify(weights["gate_proj"])
    up_blocks = blockify(weights["up_proj"])
    down_blocks = blockify(weights["down_proj"])
    gate_blocks[package_indices] += corrections[:,:16]*parent_state["scales"][g_index,None]
    up_blocks[package_indices] += corrections[:,16:32]*parent_state["scales"][u_index,None]
    down_blocks[d_index-d_start] += corrections[:,32:48]*parent_state["scales"][d_index,None]
    weights["gate_proj"] = unblock(gate_blocks, weights["gate_proj"].shape)
    weights["up_proj"] = unblock(up_blocks, weights["up_proj"].shape)
    weights["down_proj"] = unblock(down_blocks, weights["down_proj"].shape)
    return weights, {"meta": meta, "package_indices": package_indices,
                     "correction_ids": correction_ids}


def decode_joint_packages(blob, meta, meta_len):
    """Decode E15 attention blocks plus aligned Gate/Up/Down packages."""
    pos = 4 + meta_len
    attention_blocks = meta["attention_blocks"]
    records = np.frombuffer(blob, dtype="<u2", count=attention_blocks, offset=pos).copy()
    pos += 2 * attention_blocks
    attention_ids = torch.from_numpy((records & 0x0FFF).astype(np.int64))
    attention_scale_ids = torch.from_numpy((records >> 12).astype(np.int64))

    scale_count = len(meta["names"]) * SCALE_LEVELS
    scale_tables = torch.from_numpy(
        np.frombuffer(blob, dtype="<f2", count=scale_count, offset=pos).copy()
    ).view(len(meta["names"]), SCALE_LEVELS).float()
    pos += 2 * scale_count

    checkpoint_count = math.prod(meta["attention_checkpoint_shape"])
    checkpoint = torch.from_numpy(
        np.frombuffer(blob, dtype="<f2", count=checkpoint_count, offset=pos).copy()
    ).view(meta["attention_checkpoint_shape"])
    pos += 2 * checkpoint_count
    code_count = math.prod(meta["attention_pattern_shape"])
    code_bytes = (code_count * meta["attention_pattern_bits"] + 7) // 8
    codes = unpack_fixed(blob[pos:pos+code_bytes], code_count,
                         meta["attention_pattern_bits"]).view(meta["attention_pattern_shape"])
    pos += code_bytes
    family = (torch.arange(codes.shape[0]) // PATTERNS_PER_FAMILY).long()
    attention_patterns = checkpoint.float()[family[:, None], codes.long()]

    joint_count = math.prod(meta["joint_codebook_shape"])
    joint = torch.from_numpy(
        np.frombuffer(blob, dtype="<f2", count=joint_count, offset=pos).copy()
    ).view(meta["joint_codebook_shape"]).float()
    pos += 2 * joint_count
    component_residuals = None
    component_residual_bits = meta.get("component_residual_bits", 0)
    if component_residual_bits:
        residual_count = math.prod(meta["component_residual_shape"])
        component_residuals = torch.from_numpy(
            np.frombuffer(blob, dtype="<f2", count=residual_count, offset=pos).copy()
        ).view(meta["component_residual_shape"]).float()
        pos += 2 * residual_count

    package_count = meta["package_count"]
    record_bits = meta["package_record_bits"]
    package_bytes = (package_count * record_bits + 7) // 8
    package_records = unpack_uint(blob[pos:pos+package_bytes], package_count, record_bits)
    pos += package_bytes
    package_ids = package_records & ((1 << meta["package_id_bits"]) - 1)
    shift = meta["package_id_bits"]
    component_ids = []
    if component_residual_bits:
        component_mask = (1 << component_residual_bits) - 1
        for _ in range(3):
            component_ids.append((package_records >> shift) & component_mask)
            shift += component_residual_bits
    gate_scale_ids = (package_records >> shift) & 0xF
    up_scale_ids = (package_records >> (shift+4)) & 0xF
    down_scale_ids = (package_records >> (shift+8)) & 0xF

    nblocks = meta["nblocks"]
    mask_bytes = (nblocks + 7) // 8
    hard = torch.from_numpy(np.unpackbits(
        np.frombuffer(blob[pos:pos+mask_bytes], dtype=np.uint8),
        bitorder="little")[:nblocks].copy()).bool()
    pos += mask_bytes
    residual_id_bytes = (meta["hard_count"] * meta["residual_bits"] + 7) // 8
    residual_ids = unpack_fixed(blob[pos:pos+residual_id_bytes], meta["hard_count"],
                                meta["residual_bits"]).long()
    pos += residual_id_bytes
    residual_q = torch.from_numpy(np.frombuffer(
        blob, dtype=np.int8, count=meta["residual_k"]*16, offset=pos).copy()
    ).view(meta["residual_k"], 16).float()
    pos += meta["residual_k"]*16
    residual_scale = torch.from_numpy(np.frombuffer(
        blob, dtype="<f2", count=meta["residual_k"], offset=pos).copy()).float()
    pos += 2*meta["residual_k"]
    pos += meta.get("padding_bytes", 0)
    assert pos == len(blob), (pos, len(blob))

    slices = {name: tuple(meta["slices"][name]) for name in meta["names"]}
    blocks = torch.zeros(nblocks, 16)
    # Attention retains the original shared 4x4 codebook and records.
    matrix_index = torch.empty(attention_blocks, dtype=torch.long)
    for index, name in enumerate(meta["names"][:4]):
        start, end, _, _ = slices[name]
        matrix_index[start:end] = index
    blocks[:attention_blocks] = (attention_patterns[attention_ids]
        * scale_tables[matrix_index, attention_scale_ids][:, None])

    package_values = joint[package_ids]
    if component_residual_bits:
        package_values = package_values.clone()
        package_values[:, :16] += component_residuals[0, component_ids[0]]
        package_values[:, 16:32] += component_residuals[1, component_ids[1]]
        package_values[:, 32:48] += component_residuals[2, component_ids[2]]
    g_start = slices["gate_proj"][0]
    u_start = slices["up_proj"][0]
    d_start = slices["down_proj"][0]
    neuron_groups = meta["neuron_groups"]
    hidden_chunks = meta["hidden_chunks"]
    p = torch.arange(package_count)
    neuron = p // hidden_chunks
    chunk = p % hidden_chunks
    down_index = d_start + chunk*neuron_groups + neuron
    blocks[g_start:g_start+package_count] = (package_values[:, :16]
        * scale_tables[4, gate_scale_ids][:, None])
    blocks[u_start:u_start+package_count] = (package_values[:, 16:32]
        * scale_tables[5, up_scale_ids][:, None])
    blocks[down_index] = (package_values[:, 32:48]
        * scale_tables[6, down_scale_ids][:, None])

    residual = torch.zeros(nblocks, 16)
    residual[hard] = residual_q[residual_ids] * residual_scale[residual_ids, None]
    blocks += residual
    weights = {name: unblock(blocks[s:e], (rows, cols))
               for name, (s, e, rows, cols) in slices.items()}
    return weights, {"meta": meta, "package_ids": package_ids,
                     "joint_codebook": joint, "hard": hard}


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
