from __future__ import annotations

import json
from pathlib import Path

import torch

from cpu_codec import decode_weights


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "artifacts" / "functional_vq_l0"
CACHE = ROOT / "cache" / "layer0_functional_cache.pt"
PARENT = SOURCE / "packed" / "E7a_codebook_100.packed"


def mlp(weights, x):
    gate = x @ weights["gate_proj"].T
    up = x @ weights["up_proj"].T
    z = torch.nn.functional.silu(gate) * up
    return z, z @ weights["down_proj"].T


def main():
    torch.set_num_threads(4)
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    weights, _ = decode_weights(PARENT)
    output = {}
    with torch.inference_mode():
        for split in ("a", "b"):
            z, y = mlp(weights, cache[f"x_{split}"])
            zt, yt = cache[f"z_teacher_{split}"], cache[f"y_teacher_{split}"]
            output[f"Ez_{split.upper()}"] = float((z - zt).norm() / zt.norm())
            output[f"Ey_{split.upper()}"] = float((y - yt).norm() / yt.norm())
            output[f"Ly_{split.upper()}"] = float((y - yt).square().sum() / yt.square().sum())
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
