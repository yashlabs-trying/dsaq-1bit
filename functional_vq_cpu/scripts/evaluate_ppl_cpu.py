"""Strict full-model CPU perplexity evaluation of a packed Layer-0 codec."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM

from cpu_codec import decode_weights


ROOT = Path(__file__).resolve().parents[1]
PACKED = Path(os.environ.get("PPL_PACKED", ROOT / "packed/E8g_down_polish_round5.packed"))
TOKENS = ROOT.parent / "artifacts/functional_vq_l0/calibration_cache/fixed_token_ids.pt"
RESULT = Path(os.environ.get("PPL_RESULT", ROOT / "results/E8g_strict_cpu_ppl.json"))
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SEQ = 256
THREADS = int(os.environ.get("PPL_THREADS", "4"))
REFERENCE_PPL = 22.188270476805936


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    decoded, codec = decode_weights(PACKED)
    token_sets = torch.load(TOKENS, map_location="cpu", weights_only=True)
    evaluation = token_sets["evaluation"]
    print(json.dumps({"event": "load_model", "model": MODEL, "revision": REVISION,
                      "packed": str(PACKED), "packed_sha256": sha256(PACKED),
                      "evaluation_tokens": int(evaluation.numel()), "sequence_length": SEQ,
                      "available_ram": psutil.virtual_memory().available}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value)

    losses = []
    segment_times = []
    segments = len(range(0, evaluation.numel() - SEQ + 1, SEQ))
    with torch.inference_mode():
        for index, start in enumerate(range(0, evaluation.numel() - SEQ + 1, SEQ), 1):
            segment_started = time.time()
            x = evaluation[start:start + SEQ].unsqueeze(0)
            loss = float(model(x, labels=x, use_cache=False).loss)
            losses.append(loss)
            segment_times.append(time.time() - segment_started)
            print(json.dumps({"event": "segment", "index": index, "segments": segments,
                              "loss": loss, "elapsed_seconds": time.time() - started,
                              "segment_seconds": segment_times[-1],
                              "available_ram": psutil.virtual_memory().available}), flush=True)
    nll = float(np.mean(losses))
    ppl = math.exp(min(nll, 80))
    result = {
        "experiment": "E8g_strict_full_model_cpu_ppl",
        "model": MODEL, "revision": REVISION, "dtype": "float32", "device": "cpu",
        "threads": THREADS, "packed": str(PACKED), "packed_sha256": sha256(PACKED),
        "bytes": PACKED.stat().st_size, "bpw": PACKED.stat().st_size * 8 / codec["meta"]["weights"],
        "tokens": str(TOKENS), "tokens_sha256": sha256(TOKENS),
        "evaluation_tokens": int(evaluation.numel()), "sequence_length": SEQ,
        "segment_losses": losses, "nll": nll, "ppl": ppl,
        "reference_ppl": REFERENCE_PPL,
        "delta_ppl_pct": 100 * (ppl / REFERENCE_PPL - 1),
        "elapsed_seconds": time.time() - started,
        "mean_segment_seconds": float(np.mean(segment_times)),
        "python": platform.python_version(), "torch": torch.__version__,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
