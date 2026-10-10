"""Produce an exact theoretical payload ledger for an E19-R checkpoint.

This counts the proposed packed codec, not torch.save container overhead.  It
also makes explicit how far the current representation is from 1.20 BPW.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


TOTAL_MODEL_PARAMETERS = 494_032_768


def audit(path: Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    components = {
        "gptq_weight_ids": 0,
        "gptq_scales": 0,
        "gptq_zero_points": 0,
        "residual_ids_c1": 0,
        "residual_ids_c2": 0,
        "codebook_c1": 0,
        "codebook_c2": 0,
    }
    quantized_weights = 0
    for codec in state["codecs"].values():
        bits = int(codec["base_bits"])
        components["gptq_weight_ids"] += codec["base_codes"].numel() * bits
        components["gptq_scales"] += codec["base_scale"].numel() * 16
        components["gptq_zero_points"] += codec["base_zero"].numel() * 16
        components["residual_ids_c1"] += codec["idx1"].numel() * 4
        components["residual_ids_c2"] += codec["idx2"].numel() * 4
        components["codebook_c1"] += codec["C1"].numel() * 16
        components["codebook_c2"] += codec["C2"].numel() * 16
        quantized_weights += math.prod(codec["shape"])

    total_bits = sum(components.values())
    target_bits = round(1.20 * TOTAL_MODEL_PARAMETERS)
    rows = {
        name: {
            "bits": bits,
            "bytes": bits / 8,
            "model_bpw": bits / TOTAL_MODEL_PARAMETERS,
            "share_percent": 100 * bits / total_bits,
        }
        for name, bits in components.items()
    }
    weight_values = sum(
        codec["base_codes"].numel() for codec in state["codecs"].values()
    )
    scale_values = sum(
        codec["base_scale"].numel() for codec in state["codecs"].values()
    )
    residual_bits = (
        components["residual_ids_c1"] + components["residual_ids_c2"] +
        components["codebook_c1"] + components["codebook_c2"]
    )
    rate_descent = []
    for base_bits in (4, 3, 2, 1):
        fp16_zero_bits = (
            weight_values * base_bits + scale_values * 16 +
            scale_values * 16 + residual_bits
        )
        packed_zero_bits = (
            weight_values * base_bits + scale_values * 16 +
            scale_values * base_bits + residual_bits
        )
        rate_descent.append({
            "base_bits": base_bits,
            "bpw_with_fp16_zero_points": fp16_zero_bits / TOTAL_MODEL_PARAMETERS,
            "bpw_with_natural_width_zero_points": (
                packed_zero_bits / TOTAL_MODEL_PARAMETERS
            ),
        })
    return {
        "checkpoint": str(path),
        "codec_layers": len(state["codecs"]),
        "model_parameters_denominator": TOTAL_MODEL_PARAMETERS,
        "quantized_linear_weight_values": quantized_weights,
        "components": rows,
        "total": {
            "bits": total_bits,
            "bytes": total_bits / 8,
            "model_bpw": total_bits / TOTAL_MODEL_PARAMETERS,
        },
        "target_1_20_bpw": {
            "bits": target_bits,
            "bytes": target_bits / 8,
            "bits_to_remove": total_bits - target_bits,
            "bytes_to_remove": (total_bits - target_bits) / 8,
            "reduction_percent": 100 * (1 - target_bits / total_bits),
        },
        "rate_descent_projection": rate_descent,
        "notes": [
            "Counts packed integer IDs and FP16 metadata/codebooks.",
            "Does not count container alignment, tensor headers, names, or model config.",
            "The torch checkpoint is a research container, not the packed payload.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", nargs="?", type=Path,
        default=Path(__file__).parent / "e19r_results" / "e19r_value_best.pt",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.checkpoint)
    rendered = json.dumps(result, indent=2)
    if args.out:
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
