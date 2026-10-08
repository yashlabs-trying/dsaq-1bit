"""Export exact teacher/student Layer-0 transition targets for E9."""
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
PARENT = ROOT / "packed/E8g_down_polish_round5.packed"
OUTPUT = ROOT / "cache/e9_layer_cache.pt"
MANIFEST = ROOT / "cache/e9_layer_cache.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SEQ = 256


class LayerCaptured(RuntimeError):
    pass


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


@torch.inference_mode()
def capture(model, ids):
    h1_values, x_values, h2_values = [], [], []
    hooks = [
        model.model.layers[0].post_attention_layernorm.register_forward_pre_hook(
            lambda _m, inputs: h1_values.append(inputs[0].detach().float().cpu())),
        model.model.layers[0].mlp.gate_proj.register_forward_pre_hook(
            lambda _m, inputs: x_values.append(inputs[0].detach().float().cpu())),
    ]

    def stop(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        h2_values.append(hidden.detach().float().cpu())
        raise LayerCaptured

    hooks.append(model.model.layers[0].register_forward_hook(stop))
    try:
        for start in range(0, ids.numel() - SEQ + 1, SEQ):
            try:
                model(ids[start:start+SEQ].unsqueeze(0), use_cache=False)
            except LayerCaptured:
                pass
    finally:
        for hook in hooks:
            hook.remove()
    flatten = lambda values: torch.cat([v.reshape(-1, v.shape[-1]) for v in values])
    return flatten(h1_values), flatten(x_values), flatten(h2_values)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    started = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    original = {name: module.weight.detach().clone() for name, module in module_map(model).items()}
    heldout = torch.load(TOKENS, map_location="cpu", weights_only=True)["heldout"]
    ids_a, ids_b = heldout[:512], heldout[512:1024]
    _, _, teacher_h2_a = capture(model, ids_a)
    _, _, teacher_h2_b = capture(model, ids_b)
    decoded, _ = decode_weights(PARENT)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value)
    student_h1_a, student_x_a, student_h2_a = capture(model, ids_a)
    student_h1_b, student_x_b, student_h2_b = capture(model, ids_b)
    norm = model.model.layers[1].input_layernorm
    cache = {
        "model": MODEL, "revision": REVISION, "parent": str(PARENT),
        "parent_sha256": hashlib.sha256(PARENT.read_bytes()).hexdigest(),
        "x_a": student_x_a, "x_b": student_x_b,
        "h1_a": student_h1_a, "h1_b": student_h1_b,
        "h2_teacher_a": teacher_h2_a, "h2_teacher_b": teacher_h2_b,
        "h2_parent_a": student_h2_a, "h2_parent_b": student_h2_b,
        "next_norm_weight": norm.weight.detach().float().cpu().clone(),
        "next_norm_eps": float(norm.variance_epsilon),
        "original_weights": {name: value.cpu() for name, value in original.items()},
        "token_ids_a": ids_a.clone(), "token_ids_b": ids_b.clone(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    torch.save(cache, temporary)
    os.replace(temporary, OUTPUT)
    manifest = {
        "model": MODEL, "revision": REVISION, "parent": str(PARENT),
        "parent_sha256": cache["parent_sha256"], "cache": str(OUTPUT),
        "cache_bytes": OUTPUT.stat().st_size, "elapsed_seconds": time.time()-started,
        "shapes": {k: list(v.shape) for k, v in cache.items() if isinstance(v, torch.Tensor)},
        "hashes": {k: tensor_hash(v) for k, v in cache.items() if isinstance(v, torch.Tensor)},
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
