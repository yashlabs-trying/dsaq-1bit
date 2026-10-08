"""Create a self-contained Layer-0 functional-search cache on CPU.

The full model is used only to produce the exact input to Layer-0 gate_proj.
A hook aborts the forward at that point, so layers 1+ never run. The model is
then released before teacher MLP targets are constructed and saved.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "artifacts" / "functional_vq_l0"
TOKENS = SOURCE / "calibration_cache" / "fixed_token_ids.pt"
OUT = ROOT / "cache" / "layer0_functional_cache.pt"
MANIFEST = ROOT / "cache" / "layer0_functional_cache.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


class CapturedLayer0Input(RuntimeError):
    pass


def memory_state():
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    return {
        "rss_bytes": proc.memory_info().rss,
        "available_bytes": vm.available,
        "percent_used": vm.percent,
    }


@torch.inference_mode()
def capture_layer0_inputs(model, token_ids, sequence_length=256):
    captured = []

    def stop_at_gate(_module, inputs):
        captured.append(inputs[0].detach().float().cpu())
        raise CapturedLayer0Input

    hook = model.model.layers[0].mlp.gate_proj.register_forward_pre_hook(stop_at_gate)
    try:
        for start in range(0, token_ids.numel() - sequence_length + 1, sequence_length):
            batch = token_ids[start : start + sequence_length].unsqueeze(0)
            try:
                model(batch, use_cache=False)
            except CapturedLayer0Input:
                pass
            print(json.dumps({
                "event": "captured",
                "tokens": min(start + sequence_length, token_ids.numel()),
                "memory": memory_state(),
            }), flush=True)
    finally:
        hook.remove()
    return torch.cat([value.reshape(-1, value.shape[-1]) for value in captured])


@torch.inference_mode()
def teacher_targets(weights, x, token_chunk=128):
    zs, ys = [], []
    for start in range(0, x.shape[0], token_chunk):
        chunk = x[start : start + token_chunk]
        gate = chunk @ weights["gate_proj"].T
        up = chunk @ weights["up_proj"].T
        z = torch.nn.functional.silu(gate) * up
        y = z @ weights["down_proj"].T
        zs.append(z.cpu())
        ys.append(y.cpu())
    return torch.cat(zs), torch.cat(ys)


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def main():
    started = time.time()
    ROOT.joinpath("cache").mkdir(parents=True, exist_ok=True)
    print(json.dumps({"event": "loading_model", "memory": memory_state()}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        revision=REVISION,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).eval()
    layer = model.model.layers[0]
    weights = {
        "gate_proj": layer.mlp.gate_proj.weight.detach().float().cpu().clone(),
        "up_proj": layer.mlp.up_proj.weight.detach().float().cpu().clone(),
        "down_proj": layer.mlp.down_proj.weight.detach().float().cpu().clone(),
    }
    token_sets = torch.load(TOKENS, map_location="cpu", weights_only=True)
    heldout = token_sets["heldout"]
    x_a = capture_layer0_inputs(model, heldout[:512])
    x_b = capture_layer0_inputs(model, heldout[512:1024])
    del layer, model
    gc.collect()
    print(json.dumps({"event": "model_unloaded", "memory": memory_state()}), flush=True)

    z_a, y_a = teacher_targets(weights, x_a)
    z_b, y_b = teacher_targets(weights, x_b)
    cache = {
        "model": MODEL,
        "revision": REVISION,
        "x_a": x_a,
        "x_b": x_b,
        "z_teacher_a": z_a,
        "z_teacher_b": z_b,
        "y_teacher_a": y_a,
        "y_teacher_b": y_b,
        "weights": weights,
        "token_ids_a": heldout[:512].clone(),
        "token_ids_b": heldout[512:1024].clone(),
    }
    temp = OUT.with_suffix(".tmp")
    torch.save(cache, temp)
    os.replace(temp, OUT)
    hashes = {
        name: tensor_hash(value)
        for name, value in {
            "x_a": x_a,
            "x_b": x_b,
            "z_teacher_a": z_a,
            "z_teacher_b": z_b,
            "y_teacher_a": y_a,
            "y_teacher_b": y_b,
            **{f"weight_{name}": value for name, value in weights.items()},
        }.items()
    }
    manifest = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": "float32",
        "source_tokens": str(TOKENS),
        "source_tokens_sha256": hashlib.sha256(TOKENS.read_bytes()).hexdigest(),
        "cache": str(OUT),
        "cache_bytes": OUT.stat().st_size,
        "shapes": {
            key: list(value.shape)
            for key, value in cache.items()
            if isinstance(value, torch.Tensor)
        },
        "hashes": hashes,
        "elapsed_seconds": time.time() - started,
        "final_memory": memory_state(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    main()
