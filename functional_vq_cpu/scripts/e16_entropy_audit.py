"""Lossless ID entropy audit for the frozen E14c MLP assignments."""
from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import numpy as np
import torch

from cpu_codec import codec_state, pack_fixed, read_codec


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "packed/E14c_joint_ce_consistent_005.packed"
RESULT = ROOT / "results/E16_id_entropy_audit.json"


def main():
    codec = codec_state(PARENT)
    raw_codec = read_codec(PARENT)
    scale_ids = raw_codec[5]
    hard = raw_codec[6]
    residual_ids = raw_codec[7]
    ids = codec["pattern_ids"]
    result = {"experiment": "E16_id_entropy_audit", "parent": str(PARENT),
              "projections": {}}
    total_count = total_entropy_bits = total_fixed = total_zlib = 0
    streams = []
    for name in ("gate_proj", "up_proj", "down_proj"):
        start, end, _, _ = codec["slices"][name]
        values = ids[start:end].long()
        counts = torch.bincount(values, minlength=4096).double()
        probability = counts[counts > 0]/values.numel()
        entropy = float(-(probability*torch.log2(probability)).sum())
        raw = pack_fixed(values.numpy(), 12)
        compressed = zlib.compress(raw, 9)
        streams.append(raw)
        item = {"count": values.numel(), "unique_ids": int((counts > 0).sum()),
                "entropy_bits_per_id": entropy,
                "theoretical_bytes": math.ceil(entropy*values.numel()/8),
                "fixed_bytes": len(raw), "zlib_bytes": len(compressed),
                "zlib_bits_per_id": len(compressed)*8/values.numel(),
                "zlib_savings_bytes": len(raw)-len(compressed)}
        result["projections"][name] = item
        total_count += values.numel()
        total_entropy_bits += entropy*values.numel()
        total_fixed += len(raw)
        total_zlib += len(compressed)
    combined = zlib.compress(b"".join(streams), 9)
    result["total"] = {
        "count": total_count,
        "entropy_bits_per_id": total_entropy_bits/total_count,
        "theoretical_bytes": math.ceil(total_entropy_bits/8),
        "fixed_bytes": total_fixed,
        "theoretical_savings_bytes": total_fixed-math.ceil(total_entropy_bits/8),
        "zlib_bytes_separate": total_zlib,
        "zlib_savings_bytes_separate": total_fixed-total_zlib,
        "zlib_bytes_combined": len(combined),
        "zlib_savings_bytes_combined": total_fixed-len(combined),
    }
    result["scale_ids"] = {}
    scale_total_fixed = scale_total_theory = scale_total_zlib = 0
    for name in ("gate_proj", "up_proj", "down_proj"):
        start, end, _, _ = codec["slices"][name]
        values = scale_ids[start:end].long()
        counts = torch.bincount(values, minlength=16).double()
        probability = counts[counts > 0]/values.numel()
        entropy = float(-(probability*torch.log2(probability)).sum())
        raw = pack_fixed(values.numpy(), 4)
        compressed = zlib.compress(raw, 9)
        theory = math.ceil(entropy*values.numel()/8)
        result["scale_ids"][name] = {
            "count": values.numel(), "entropy_bits_per_id": entropy,
            "theoretical_bytes": theory, "fixed_bytes": len(raw),
            "zlib_bytes": len(compressed),
            "theoretical_savings_bytes": len(raw)-theory,
            "zlib_savings_bytes": len(raw)-len(compressed),
        }
        scale_total_fixed += len(raw)
        scale_total_theory += theory
        scale_total_zlib += len(compressed)
    result["scale_ids_total"] = {
        "fixed_bytes": scale_total_fixed,
        "theoretical_bytes": scale_total_theory,
        "theoretical_savings_bytes": scale_total_fixed-scale_total_theory,
        "zlib_bytes": scale_total_zlib,
        "zlib_savings_bytes": scale_total_fixed-scale_total_zlib,
    }
    hard_raw = np.packbits(hard.numpy().astype(np.uint8), bitorder="little").tobytes()
    hard_zlib = zlib.compress(hard_raw, 9)
    rid_raw = pack_fixed(residual_ids.numpy(), 8)
    rid_zlib = zlib.compress(rid_raw, 9)
    result["other_streams"] = {
        "hard_mask_fixed_bytes": len(hard_raw),
        "hard_mask_zlib_bytes": len(hard_zlib),
        "hard_mask_zlib_savings_bytes": len(hard_raw)-len(hard_zlib),
        "residual_ids_fixed_bytes": len(rid_raw),
        "residual_ids_zlib_bytes": len(rid_zlib),
        "residual_ids_zlib_savings_bytes": len(rid_raw)-len(rid_zlib),
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
