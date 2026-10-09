"""E15: train and pack a true joint Gate/Up/Down 4-neuron package codec.

Each package aligns three blocks:
  Gate[j:j+4, c:c+4], Up[j:j+4, c:c+4], Down[c:c+4, j:j+4].

The three normalized 4x4 blocks share one 13-bit package ID while retaining
their three existing 4-bit scale IDs. Attention and the complete residual
stream are copied from E14c. The output is padded to exactly the E14c byte size.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cpu_codec import blockify, codec_state, decode_weights, pack_fixed, read_codec


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(os.environ.get("E15_PARENT", ROOT / "packed/E14c_joint_ce_consistent_005.packed"))
CACHE = ROOT / "cache/e9_layer_cache.pt"
LABEL = os.environ.get("E15_LABEL", "E15_joint_package_8140")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
K = int(os.environ.get("E15_K", "8140"))
TARGET = os.environ.get("E15_TARGET", "parent")
COMPONENT_RESIDUAL_BITS = int(os.environ.get("E15_COMPONENT_RESIDUAL_BITS", "0"))
THREADS = int(os.environ.get("E15_THREADS", "4"))
COARSE_K = 32
COARSE_ITERS = int(os.environ.get("E15_COARSE_ITERS", "8"))
LOCAL_ITERS = int(os.environ.get("E15_LOCAL_ITERS", "5"))
SEED = 20261009


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack_uint(values, bits):
    values = np.asarray(values, dtype=np.uint64).reshape(-1)
    shifts = np.arange(bits, dtype=np.uint64)
    matrix = ((values[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(matrix.reshape(-1), bitorder="little").tobytes()


def assign(data, centers, chunk=2048):
    ids = torch.empty(data.shape[0], dtype=torch.long)
    center_norm = centers.square().sum(1)
    for start in range(0, data.shape[0], chunk):
        x = data[start:start+chunk]
        distance = x.square().sum(1, keepdim=True) + center_norm[None, :] - 2*(x @ centers.T)
        ids[start:start+x.shape[0]] = distance.argmin(1)
    return ids


def kmeans(data, k, iterations, generator):
    if data.shape[0] < k:
        raise RuntimeError(f"cluster has {data.shape[0]} samples for {k} centers")
    initial = torch.randperm(data.shape[0], generator=generator)[:k]
    centers = data[initial].clone()
    ids = None
    for _ in range(iterations):
        ids = assign(data, centers)
        sums = torch.zeros_like(centers)
        counts = torch.bincount(ids, minlength=k).float()
        sums.index_add_(0, ids, data)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            refill = torch.randint(data.shape[0], (int((~nonempty).sum()),), generator=generator)
            centers[~nonempty] = data[refill]
    return centers, ids


def train_hierarchical(data):
    generator = torch.Generator().manual_seed(SEED)
    sample_count = min(65536, data.shape[0])
    sample_ids = torch.randperm(data.shape[0], generator=generator)[:sample_count]
    coarse, _ = kmeans(data[sample_ids], COARSE_K, COARSE_ITERS, generator)
    coarse_ids = assign(data, coarse)
    # 8140 => twelve clusters with 255 local centers, twenty with 254.
    base, extra = divmod(K, COARSE_K)
    local_ks = [base + (1 if i < extra else 0) for i in range(COARSE_K)]
    centers_all = []
    package_ids = torch.empty(data.shape[0], dtype=torch.long)
    offset = 0
    for coarse_index, local_k in enumerate(local_ks):
        members = torch.where(coarse_ids == coarse_index)[0]
        local_data = data[members]
        local_centers, local_ids = kmeans(local_data, local_k, LOCAL_ITERS, generator)
        # Stored FP16 centers define the real codec. Reassign once to those exact values.
        local_centers = local_centers.half().float()
        local_ids = assign(local_data, local_centers)
        package_ids[members] = local_ids + offset
        centers_all.append(local_centers)
        offset += local_k
        print(json.dumps({"event": "local_cluster", "index": coarse_index+1,
                          "clusters": COARSE_K, "samples": int(members.numel()),
                          "centers": local_k, "total_centers": offset}), flush=True)
    centers = torch.cat(centers_all)
    assert centers.shape == (K, 48)
    return centers, package_ids


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)

    parent = read_codec(PARENT)
    (old_meta, scale_tables, checkpoint, codes, old_ids, scale_ids,
     hard, residual_ids, residual_q, residual_scale) = parent
    state = codec_state(PARENT)
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    originals = cache["original_weights"]
    slices = state["slices"]
    g_start, _, g_rows, g_cols = slices["gate_proj"]
    u_start, _, _, _ = slices["up_proj"]
    d_start, _, d_rows, d_cols = slices["down_proj"]
    neuron_groups, hidden_chunks = g_rows//4, g_cols//4
    package_count = neuron_groups*hidden_chunks
    p = torch.arange(package_count)
    neuron = p//hidden_chunks
    chunk = p%hidden_chunks
    g_index = g_start+p
    u_index = u_start+p
    d_index = d_start+chunk*neuron_groups+neuron

    original_blocks = {name: blockify(originals[name])
                       for name in ("gate_proj", "up_proj", "down_proj")}
    down_local = d_index-d_start
    if TARGET == "bf16":
        target_g = ((original_blocks["gate_proj"]-state["residual"][g_index])
                    / state["scales"][g_index, None])
        target_u = ((original_blocks["up_proj"]-state["residual"][u_index])
                    / state["scales"][u_index, None])
        target_d = ((original_blocks["down_proj"][down_local]-state["residual"][d_index])
                    / state["scales"][d_index, None])
    elif TARGET == "parent":
        # Preserve the functionally optimized E14c state. Its assignments are
        # deliberately not nearest-BF16 patterns, so BF16-MSE initialization
        # throws away the strongest result of E7-E14.
        parent_patterns = state["patterns"][state["pattern_ids"].long()]
        target_g = parent_patterns[g_index]
        target_u = parent_patterns[u_index]
        target_d = parent_patterns[d_index]
    else:
        raise ValueError(f"unknown E15_TARGET={TARGET!r}")
    packages = torch.cat((target_g, target_u, target_d), dim=1).contiguous()
    print(json.dumps({"event": "train_start", "packages": package_count,
                      "dimensions": 48, "k": K, "target": TARGET,
                      "parent": str(PARENT)}), flush=True)
    centers, package_ids = train_hierarchical(packages)
    package_id_bits = math.ceil(math.log2(K))
    component_codebooks = None
    component_ids = []
    reconstructed = centers[package_ids].clone()
    if COMPONENT_RESIDUAL_BITS:
        component_k = 1 << COMPONENT_RESIDUAL_BITS
        generator = torch.Generator().manual_seed(SEED+1)
        books = []
        for component in range(3):
            section = slice(component*16, (component+1)*16)
            error = packages[:, section]-reconstructed[:, section]
            book, rid = kmeans(error, component_k, 8, generator)
            book = book.half().float()
            rid = assign(error, book)
            reconstructed[:, section] += book[rid]
            books.append(book)
            component_ids.append(rid)
        component_codebooks = torch.stack(books)

    attention_blocks = g_start
    attention_records = ((scale_ids[:attention_blocks].numpy().astype(np.uint16) << 12)
                         | old_ids[:attention_blocks].numpy().astype(np.uint16))
    package_records = package_ids.numpy().astype(np.uint64)
    shift = package_id_bits
    if COMPONENT_RESIDUAL_BITS:
        for rid in component_ids:
            package_records |= rid.numpy().astype(np.uint64) << shift
            shift += COMPONENT_RESIDUAL_BITS
    package_records |= scale_ids[g_index].numpy().astype(np.uint64) << shift
    package_records |= scale_ids[u_index].numpy().astype(np.uint64) << (shift+4)
    package_records |= scale_ids[d_index].numpy().astype(np.uint64) << (shift+8)
    package_record_bits = shift+12

    payload = bytearray()
    payload += attention_records.astype("<u2").tobytes()
    payload += scale_tables.numpy().astype("<f2").tobytes()
    payload += checkpoint.numpy().astype("<f2").tobytes()
    payload += pack_fixed(codes.numpy(), old_meta["pattern_bits"])
    payload += centers.numpy().astype("<f2").tobytes()
    if component_codebooks is not None:
        payload += component_codebooks.numpy().astype("<f2").tobytes()
    payload += pack_uint(package_records, package_record_bits)
    payload += np.packbits(hard.numpy().astype(np.uint8), bitorder="little").tobytes()
    payload += pack_fixed(residual_ids.numpy(), old_meta["residual_bits"])
    payload += residual_q.numpy().astype(np.int8).tobytes()
    payload += residual_scale.numpy().astype("<f2").tobytes()

    meta = {
        "version": 2, "layout": "joint_mlp_packages",
        "method": f"E15 hierarchical {TARGET}-imitation joint Gate/Up/Down package VQ",
        "model": "Qwen/Qwen2.5-0.5B",
        "revision": "060db6499f32faf8b98477b0a26969ef7d8b9987",
        "weights": old_meta["weights"], "nblocks": old_meta["nblocks"],
        "names": old_meta["names"], "slices": old_meta["slices"],
        "attention_blocks": attention_blocks,
        "attention_checkpoint_shape": old_meta["checkpoint_shape"],
        "attention_pattern_shape": old_meta["pattern_shape"],
        "attention_pattern_bits": old_meta["pattern_bits"],
        "joint_codebook_shape": [K, 48], "joint_codebook_dtype": "float16",
        "package_count": package_count, "package_id_bits": package_id_bits,
        "package_record_bits": package_record_bits,
        "component_residual_bits": COMPONENT_RESIDUAL_BITS,
        "component_residual_shape": ([3, 1 << COMPONENT_RESIDUAL_BITS, 16]
                                     if COMPONENT_RESIDUAL_BITS else None),
        "neuron_groups": neuron_groups,
        "hidden_chunks": hidden_chunks, "hard_count": old_meta["hard_count"],
        "residual_bits": old_meta["residual_bits"],
        "residual_k": old_meta["residual_k"], "seed": SEED,
        "coarse_k": COARSE_K, "coarse_iters": COARSE_ITERS,
        "local_iters": LOCAL_ITERS, "target": TARGET,
        "parent_sha256": sha256(PARENT),
        "padding_bytes": 0,
    }
    target_bytes = PARENT.stat().st_size
    for _ in range(10):
        meta_blob = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
        padding = target_bytes - (4+len(meta_blob)+len(payload))
        if padding < 0:
            raise RuntimeError(f"E15 exceeds parent by {-padding} bytes")
        if padding == meta["padding_bytes"]:
            break
        meta["padding_bytes"] = padding
    meta_blob = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    padding = target_bytes - (4+len(meta_blob)+len(payload))
    meta["padding_bytes"] = padding
    meta_blob = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode()
    padding = target_bytes - (4+len(meta_blob)+len(payload))
    if padding != meta["padding_bytes"] or padding < 0:
        raise RuntimeError("metadata/padding length did not stabilize")
    blob = struct.pack("<I", len(meta_blob)) + meta_blob + payload + bytes(padding)
    OUTPUT.write_bytes(blob)
    assert OUTPUT.stat().st_size == target_bytes

    mse = float((packages-reconstructed).square().mean())
    parent_weights, _ = decode_weights(PARENT)
    output_weights, _ = decode_weights(OUTPUT)
    relative_weight_mse = {
        name: float((output_weights[name]-parent_weights[name]).square().sum()
                    / parent_weights[name].square().sum())
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    functional = {}
    for split in ("a", "b"):
        x = cache[f"x_{split}"]
        z_parent = F.silu(x@parent_weights["gate_proj"].T) * (x@parent_weights["up_proj"].T)
        z_output = F.silu(x@output_weights["gate_proj"].T) * (x@output_weights["up_proj"].T)
        y_parent = z_parent@parent_weights["down_proj"].T
        y_output = z_output@output_weights["down_proj"].T
        functional[f"relative_z_{split.upper()}"] = float(
            (z_output-z_parent).square().sum()/z_parent.square().sum())
        functional[f"relative_y_{split.upper()}"] = float(
            (y_output-y_parent).square().sum()/y_parent.square().sum())
    result = {"experiment": LABEL, "parent": str(PARENT),
              "parent_sha256": sha256(PARENT), "target": TARGET, "k": K,
              "package_count": package_count, "package_dimensions": 48,
              "package_id_bits": package_id_bits,
              "component_residual_bits": COMPONENT_RESIDUAL_BITS,
              "package_record_bits": package_record_bits,
              "scale_bits_per_package": 12,
              "joint_codebook_bytes": K*48*2,
              "component_codebook_bytes": (0 if component_codebooks is None
                                             else component_codebooks.numel()*2),
              "padding_bytes": padding,
              "training_mse_normalized": mse,
              "relative_weight_mse_vs_parent": relative_weight_mse,
              "functional_error_vs_parent": functional,
              "functional_gate_pass": max(functional.values()) < 0.25,
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size*8/old_meta["weights"],
              "output_sha256": sha256(OUTPUT),
              "elapsed_seconds": time.time()-started, "strict_ppl": None}
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
