"""Compare exact Layer-0 transition errors for E7a and E8g on fixed A/B data."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from cpu_codec import decode_weights


ROOT = Path(__file__).resolve().parents[1]
TOKENS = ROOT.parent / "artifacts/functional_vq_l0/calibration_cache/fixed_token_ids.pt"
E7A = ROOT.parent / "artifacts/functional_vq_l0/packed/E7a_codebook_100.packed"
E8G = ROOT / "packed/E8g_down_polish_round5.packed"
RESULT = ROOT / "results/E9_layer_transition_diagnostic.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
THREADS = int(os.environ.get("E9_THREADS", "4"))
SEQ = 256


class LayerCaptured(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


@torch.inference_mode()
def capture_h2(model, token_ids):
    captured = []

    def stop_after_layer(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden.detach().float().cpu())
        raise LayerCaptured

    hook = model.model.layers[0].register_forward_hook(stop_after_layer)
    try:
        for start in range(0, token_ids.numel() - SEQ + 1, SEQ):
            batch = token_ids[start:start + SEQ].unsqueeze(0)
            try:
                model(batch, use_cache=False)
            except LayerCaptured:
                pass
    finally:
        hook.remove()
    return torch.cat([item.reshape(-1, item.shape[-1]) for item in captured])


@torch.inference_mode()
def next_norm(model, hidden):
    norm = model.model.layers[1].input_layernorm
    return norm(hidden)


def errors(teacher_h2, student_h2, teacher_norm, student_norm):
    layer_loss = float((student_h2-teacher_h2).square().sum()/teacher_h2.square().sum())
    norm_loss = float((student_norm-teacher_norm).square().sum()/teacher_norm.square().sum())
    return {"L_h2": layer_loss, "E_h2": layer_loss ** 0.5,
            "L_next_norm": norm_loss, "E_next_norm": norm_loss ** 0.5}


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    token_sets = torch.load(TOKENS, map_location="cpu", weights_only=True)
    heldout = token_sets["heldout"]
    ids_a, ids_b = heldout[:512], heldout[512:1024]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    original = {name: module.weight.detach().clone() for name, module in module_map(model).items()}

    teacher_h2_a, teacher_h2_b = capture_h2(model, ids_a), capture_h2(model, ids_b)
    teacher_norm_a, teacher_norm_b = next_norm(model, teacher_h2_a), next_norm(model, teacher_h2_b)
    results = {}
    for label, path in (("E7a", E7A), ("E8g", E8G)):
        decoded, state = decode_weights(path)
        with torch.no_grad():
            for name, value in decoded.items():
                module_map(model)[name].weight.copy_(value)
        h2_a, h2_b = capture_h2(model, ids_a), capture_h2(model, ids_b)
        norm_a, norm_b = next_norm(model, h2_a), next_norm(model, h2_b)
        results[label] = {
            "packed": str(path), "packed_sha256": sha256(path),
            "bytes": path.stat().st_size,
            "bpw": path.stat().st_size * 8 / state["meta"]["weights"],
            "A": errors(teacher_h2_a, h2_a, teacher_norm_a, norm_a),
            "B": errors(teacher_h2_b, h2_b, teacher_norm_b, norm_b),
        }

    with torch.no_grad():
        for name, value in original.items():
            module_map(model)[name].weight.copy_(value)
    payload = {
        "experiment": "E9_layer_transition_alignment_diagnostic",
        "model": MODEL, "revision": REVISION, "dtype": "float32", "device": "cpu",
        "threads": THREADS, "tokens_sha256": sha256(TOKENS),
        "tokens_A": int(ids_a.numel()), "tokens_B": int(ids_b.numel()),
        "sequence_length": SEQ, "results": results,
        "elapsed_seconds": time.time()-started,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    os.replace(temporary, RESULT)
    print(json.dumps({"event": "complete", **payload}, indent=2), flush=True)


if __name__ == "__main__":
    main()
