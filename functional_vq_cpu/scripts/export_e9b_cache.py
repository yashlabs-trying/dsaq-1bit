"""Export downstream (L1) teacher/student targets + sensitivity gradients for E9b.

Computes:
  h3_teacher = L1(h2_teacher)   — teacher hidden state after Layer 1
  h3_student = L1(h2_student)   — student (parent codec) hidden state after Layer 1
  g = ∂||h3_student - h3_teacher||² / ∂h2   — downstream sensitivity gradient w.r.t. h2

The sensitivity gradient g is used for the cheap first-order scoring:
  Δloss ≈ Σ_t (g_t^T · Δh2_t)² / ||error_total||²
"""
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
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
PARENT = ROOT / "packed/E8g_down_polish_round5.packed"
OUTPUT = ROOT / "cache/e9b_cache.pt"
MANIFEST = ROOT / "cache/e9b_cache.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def run_layer1(h2, model, token_ids):
    """Run h2 through frozen Layer 1 (RMSNorm → self_attn → residual → RMSNorm → MLP)."""
    layer1 = model.model.layers[1]
    seq_len = h2.shape[0]

    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    causal_mask = torch.full((seq_len, seq_len), torch.finfo(h2.dtype).min, device=h2.device)
    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    cos, sin = model.model.rotary_emb(h2.unsqueeze(0), position_ids)
    position_embeddings = (cos, sin)

    h = layer1.input_layernorm(h2)
    attn_out = layer1.self_attn(
        h.unsqueeze(0),
        attention_mask=causal_mask,
        position_embeddings=position_embeddings,
    )[0].squeeze(0)
    h = h2 + attn_out
    h_attn = h
    h = layer1.post_attention_layernorm(h)
    h = h_attn + layer1.mlp(h)
    return h


def main():
    started = time.time()
    torch.set_num_threads(4)

    e9_cache = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    parent_sha = sha256(PARENT)
    if e9_cache["parent_sha256"] != parent_sha:
        raise RuntimeError("E9 cache is not anchored to E8g")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()
    decoded, _ = decode_weights(PARENT)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value)

    h2t_a = e9_cache["h2_teacher_a"]
    h2t_b = e9_cache["h2_teacher_b"]
    h2s_a = e9_cache["h2_parent_a"]
    h2s_b = e9_cache["h2_parent_b"]
    token_ids_a = e9_cache["token_ids_a"]
    token_ids_b = e9_cache["token_ids_b"]

    print(json.dumps({"event": "compute_downstream_h3_teacher"}), flush=True)
    with torch.no_grad():
        h3t_a = run_layer1(h2t_a, model, token_ids_a)
        h3t_b = run_layer1(h2t_b, model, token_ids_b)

    print(json.dumps({"event": "compute_downstream_h3_student"}), flush=True)
    with torch.no_grad():
        h3s_a = run_layer1(h2s_a, model, token_ids_a)
        h3s_b = run_layer1(h2s_b, model, token_ids_b)

    print(json.dumps({"event": "compute_sensitivity_gradients"}), flush=True)
    h2s_a_grad = h2s_a.detach().clone().requires_grad_(True)
    h2s_b_grad = h2s_b.detach().clone().requires_grad_(True)
    h3s_a_grad = run_layer1(h2s_a_grad, model, token_ids_a)
    h3s_b_grad = run_layer1(h2s_b_grad, model, token_ids_b)
    loss_a = (h3s_a_grad - h3t_a).square().sum()
    loss_b = (h3s_b_grad - h3t_b).square().sum()
    ga = torch.autograd.grad(loss_a, h2s_a_grad)[0]
    gb = torch.autograd.grad(loss_b, h2s_b_grad)[0]

    cache = {
        "model": MODEL, "revision": REVISION, "parent": str(PARENT),
        "parent_sha256": parent_sha,
        "h1_a": e9_cache["h1_a"].clone(), "h1_b": e9_cache["h1_b"].clone(),
        "x_a": e9_cache["x_a"].clone(), "x_b": e9_cache["x_b"].clone(),
        "h2_teacher_a": h2t_a, "h2_teacher_b": h2t_b,
        "h2_parent_a": e9_cache["h2_parent_a"].clone(),
        "h2_parent_b": e9_cache["h2_parent_b"].clone(),
        "h3_teacher_a": h3t_a, "h3_teacher_b": h3t_b,
        "h3_parent_a": h3s_a, "h3_parent_b": h3s_b,
        "sensitivity_gradient_a": ga,
        "sensitivity_gradient_b": gb,
        "next_norm_weight": e9_cache["next_norm_weight"].clone(),
        "next_norm_eps": e9_cache["next_norm_eps"],
        "original_weights": {k: v.clone() for k, v in e9_cache["original_weights"].items()},
        "token_ids_a": token_ids_a.clone(),
        "token_ids_b": token_ids_b.clone(),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".tmp")
    torch.save(cache, temporary)
    os.replace(temporary, OUTPUT)

    with torch.no_grad():
        l3_A = float((h3s_a - h3t_a).square().sum() / h3t_a.square().sum())
        l3_B = float((h3s_b - h3t_b).square().sum() / h3t_b.square().sum())
        l2_A = float((h2s_a - h2t_a).square().sum() / h2t_a.square().sum())
        l2_B = float((h2s_b - h2t_b).square().sum() / h2t_b.square().sum())

    manifest = {
        "model": MODEL, "revision": REVISION, "parent": str(PARENT),
        "parent_sha256": parent_sha, "cache": str(OUTPUT),
        "cache_bytes": OUTPUT.stat().st_size, "elapsed_seconds": time.time() - started,
        "shapes": {k: list(v.shape) for k, v in cache.items() if isinstance(v, torch.Tensor)},
        "hashes": {k: tensor_hash(v) for k, v in cache.items() if isinstance(v, torch.Tensor)},
        "L_h2_A": l2_A, "L_h2_B": l2_B,
        "L_h3_A": l3_A, "L_h3_B": l3_B,
        "downstream_amplification_A": l3_A / max(l2_A, 1e-10),
        "downstream_amplification_B": l3_B / max(l2_B, 1e-10),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"event": "complete", **manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
