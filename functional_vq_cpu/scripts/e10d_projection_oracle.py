"""7-projection BF16 restoration oracle.

For each projection (Q/K/V/O/Gate/Up/Down), restore ONLY that one projection
to BF16 while keeping all others at E10A quantized values, then run strict PPL.

This identifies where the remaining PPL damage actually lives after E10A.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM

from cpu_codec import decode_weights


ROOT = Path(__file__).resolve().parents[1]
E10A_PACKED = ROOT / "packed/E10a_gate_up_downstream.packed"
TOKENS = ROOT.parent / "artifacts/functional_vq_l0/calibration_cache/fixed_token_ids.pt"
RESULT = ROOT / "results/E10d_projection_oracle.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SEQ = 256
THREADS = 4
REFERENCE_PPL = 22.188270476805936


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate_ppl(model, token_ids):
    """Evaluate PPL on evaluation token set."""
    losses = []
    with torch.inference_mode():
        for start in range(0, token_ids.numel() - SEQ + 1, SEQ):
            x = token_ids[start:start + SEQ].unsqueeze(0)
            loss = float(model(x, labels=x, use_cache=False).loss)
            losses.append(loss)
    nll = float(np.mean(losses))
    ppl = math.exp(min(nll, 80))
    return ppl, nll, losses


def main():
    torch.set_num_threads(THREADS)
    started = time.time()

    decoded, codec = decode_weights(E10A_PACKED)
    token_sets = torch.load(TOKENS, map_location="cpu", weights_only=True)
    evaluation = token_sets["evaluation"]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()

    # Store original BF16 weights for all projections
    bf16_weights = {}
    with torch.no_grad():
        for name, mod in module_map(model).items():
            bf16_weights[name] = mod.weight.detach().clone()

    results = {}

    # Baseline: all quantized (E10A)
    print(json.dumps({"event": "baseline"}), flush=True)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value)
    ppl, nll, losses = evaluate_ppl(model, evaluation)
    results["none"] = {"ppl": ppl, "nll": nll, "segment_losses": losses}
    print(json.dumps({"event": "result", "restored": "none", "ppl": ppl}), flush=True)

    # One projection at a time: restore to BF16
    for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        print(json.dumps({"event": "restoring", "projection": name}), flush=True)
        with torch.no_grad():
            # Start from E10A quantized
            for n, value in decoded.items():
                module_map(model)[n].weight.copy_(value)
            # Override just this one projection with BF16
            module_map(model)[name].weight.copy_(bf16_weights[name])

        ppl, nll, losses = evaluate_ppl(model, evaluation)
        results[name] = {"ppl": ppl, "nll": nll, "segment_losses": losses}
        print(json.dumps({"event": "result", "restored": name, "ppl": ppl}), flush=True)

    # Full BF16 reference
    print(json.dumps({"event": "bf16_reference"}), flush=True)
    with torch.no_grad():
        for name, w in bf16_weights.items():
            module_map(model)[name].weight.copy_(w)
    ppl, nll, losses = evaluate_ppl(model, evaluation)
    results["all_bf16"] = {"ppl": ppl, "nll": nll, "segment_losses": losses}
    print(json.dumps({"event": "result", "restored": "all_bf16", "ppl": ppl}), flush=True)

    # Summary table
    summary = []
    for proj, data in results.items():
        delta_bf16 = data["ppl"] - REFERENCE_PPL
        summary.append({"projection": proj, "ppl": data["ppl"],
                        "delta_from_bf16": round(delta_bf16, 3)})
    summary.sort(key=lambda x: x["ppl"])

    report = {
        "experiment": "E10d_projection_oracle",
        "model": MODEL, "packed": str(E10A_PACKED),
        "packed_sha256": sha256(E10A_PACKED),
        "reference_ppl": REFERENCE_PPL,
        "results": {k: {"ppl": v["ppl"], "nll": v["nll"]} for k, v in results.items()},
        "summary_table": summary,
        "elapsed_seconds": time.time() - started,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", **report}, indent=2), flush=True)


if __name__ == "__main__":
    main()
