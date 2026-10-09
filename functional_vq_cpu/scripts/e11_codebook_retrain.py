"""E11: CE-aware codebook prototype retraining.

Optimizes the 4096 prototype patterns (16 families × 256 patterns × 16 values)
against the downstream CE/KL loss, keeping the same 6-bit storage.

Method: float shadow copies → gradient descent → re-quantize to 6-bit levels.
IDs, scales, residuals are frozen during prototype optimization.

After retraining, run E11b to reassign IDs (now that prototypes have moved).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import read_codec, pack_fixed, patch_pattern_codes, unblock


ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
E10A_PACKED = ROOT / "packed/E10a_gate_up_downstream.packed"
OUTPUT = ROOT / "packed" / "E11_retrained.packed"
RESULT = ROOT / "results" / "E11_retrained.json"
STATE_PATH = ROOT / "checkpoints" / "E11_retrained_state.pt"
LABEL = "E11_codebook_ce_retrain"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
LR = float(os.environ.get("E11_LR", "1e-4"))
STEPS = int(os.environ.get("E11_STEPS", "50"))
USE_CE = os.environ.get("E11_CE", "1") == "1"


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_state():
    vm = psutil.virtual_memory()
    return {"rss_bytes": psutil.Process().memory_info().rss,
            "available_bytes": vm.available, "percent_used": vm.percent}


def run_mlp(weights, x):
    gate = x @ weights["gate_proj"].T
    up = x @ weights["up_proj"].T
    z = F.silu(gate) * up
    return z, z @ weights["down_proj"].T


def run_downstream(h2, model):
    """Run h2 through L1+L2, return h4."""
    for i in (1, 2):
        layer = model.model.layers[i]
        slen = h2.shape[0]
        pos = torch.arange(slen, device=h2.device).unsqueeze(0)
        mask = torch.full((slen, slen), torch.finfo(h2.dtype).min, device=h2.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h2.unsqueeze(0), pos)
        h_norm = layer.input_layernorm(h2)
        attn = layer.self_attn(h_norm.unsqueeze(0), attention_mask=mask,
                               position_embeddings=(cos, sin))[0].squeeze(0)
        h2 = h2 + attn
        h_attn = h2
        h2 = layer.post_attention_layernorm(h2)
        h2 = h_attn + layer.mlp(h2)
    return h2


def run_full_model(h2, model, token_ids):
    """Run h2 through L1-L23 + LM head, return logits."""
    for i in range(1, 24):
        layer = model.model.layers[i]
        slen = h2.shape[0]
        pos = torch.arange(slen, device=h2.device).unsqueeze(0)
        mask = torch.full((slen, slen), torch.finfo(h2.dtype).min, device=h2.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h2.unsqueeze(0), pos)
        h_norm = layer.input_layernorm(h2)
        attn = layer.self_attn(h_norm.unsqueeze(0), attention_mask=mask,
                               position_embeddings=(cos, sin))[0].squeeze(0)
        h2 = h2 + attn
        h_attn = h2
        h2 = layer.post_attention_layernorm(h2)
        h2 = h_attn + layer.mlp(h2)
    return model.lm_head(h2)


def quantize_prototypes(float_protos, checkpoint, codes):
    """Quantize float prototypes to nearest 6-bit levels and update codes."""
    n_families, n_patterns, n_vals = float_protos.shape
    levels = checkpoint.float()  # (16, 64)
    # Vectorized: (16, 256, 16, 1) - (16, 1, 1, 64) = (16, 256, 16, 64)
    dists = (float_protos.unsqueeze(-1) - levels[:, None, None, :]).abs()
    new_codes = dists.argmin(dim=-1)  # (16, 256, 16)
    return new_codes


def decode_weights_from_protos(float_protos, ids, scales, residuals, slices, meta):
    """Decode weights from float prototypes + current metadata."""
    import math
    nblocks = ids.shape[0]
    patterns_per_family = 256
    family = torch.arange(nblocks) // patterns_per_family
    patterns = float_protos[family.long(), ids.long()]  # (nblocks, 16)
    block_vals = patterns * scales[:, None] + residuals  # (nblocks, 16)
    weights = {}
    for name, (s, e, r, c) in slices.items():
        weights[name] = unblock(block_vals[s:e], (r, c))
    return weights


def main():
    torch.set_num_threads(4)
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    e9 = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    tok_a, tok_b = e9["token_ids_a"], e9["token_ids_b"]
    h1_a, h1_b = e9["h1_a"], e9["h1_b"]
    x_a, x_b = e9["x_a"], e9["x_b"]
    h4t_a, h4t_b = e9.get("h4_teacher_a"), e9.get("h4_teacher_b")

    # Need h4 teacher for L1+L2 loss — compute from e9 cache teacher h2
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval().to(device)

    if h4t_a is None:
        print(json.dumps({"event": "computing_h4_teacher"}), flush=True)
        with torch.no_grad():
            h2t_a = e9["h2_teacher_a"].to(device)
            h2t_b = e9["h2_teacher_b"].to(device)
            h4t_a = run_downstream(h2t_a, model).cpu()
            h4t_b = run_downstream(h2t_b, model).cpu()

    # ── Load E10A codec ──
    meta, scale_tables, checkpoint, codes, pattern_ids, scale_ids, hard, rid, rq, rs = \
        read_codec(E10A_PACKED)
    print(json.dumps({"event": "loaded_codec", "nblocks": meta["nblocks"],
                       "n_patterns": meta["pattern_shape"]}), flush=True)

    # Build current prototypes (16 families × 256 patterns × 16 values)
    n_families, n_levels = checkpoint.shape
    n_per_family = meta["pattern_shape"][0] // n_families  # 256
    # codes are flat (4096, 16): first 256 = family 0, next 256 = family 1, etc.
    pf = torch.arange(codes.shape[0], dtype=torch.long) // n_per_family  # (4096,) family per code
    proto_vals = checkpoint.float()[pf[:, None], codes.long()]  # (4096, 16)
    float_protos = proto_vals.reshape(n_families, n_per_family, 16)  # (16, 256, 16)
    pid = pattern_ids.long() if isinstance(pattern_ids, torch.Tensor) else torch.from_numpy(pattern_ids).long()
    sid = scale_ids.long() if isinstance(scale_ids, torch.Tensor) else torch.from_numpy(scale_ids).long()

    matrix_index = torch.empty(meta["nblocks"], dtype=torch.long)
    slices = {}
    for idx, name in enumerate(meta["names"]):
        s, e, r, c = meta["slices"][name]
        slices[name] = (s, e, r, c)
        matrix_index[s:e] = idx

    scales_f = scale_tables.float()[matrix_index, sid]
    residual = torch.zeros(meta["nblocks"], 16)
    if hard is not None:
        residual[hard] = rq.float()[rid.long()] * rs.float()[rid.long(), None]

    # ── Training loop ──
    print(json.dumps({"event": "start_training", "steps": STEPS, "lr": LR,
                       "use_ce": USE_CE, "device": device}), flush=True)

    # We'll reconstruct current state each step from the prototype tensor
    # and run the downstream to get the loss.
    epoch = 0
    best_loss = float("inf")

    if STATE_PATH.exists():
        state = torch.load(STATE_PATH, map_location="cpu", weights_only=True)
        if state["parent_sha256"] == sha256(E10A_PACKED):
            float_protos = state["float_protos"]
            epoch = state["epoch"]
            codes = state["codes"]
            print(json.dumps({"event": "resume", "epoch": epoch}), flush=True)

    # Copy Layer-0 BF16 weights for teacher CE computation
    original_w = {n: mod.weight.detach().clone().cpu()
                  for n, mod in module_map(model).items()}

    for step in range(epoch, STEPS):
        step_started = time.time()
        float_protos = float_protos.to(device).requires_grad_(True)

        # Decode weights from current prototypes
        # For each block: family = pid // 256, pattern_in_family = pid % 256
        pid_dev = pid.to(device)
        pif = pid_dev // 256   # family index (0-15)
        pip = pid_dev % 256    # pattern within family (0-255)
        block_vals = float_protos[pif, pip]  # (nblocks, 16)
        block_vals = block_vals * scales_f.to(device)[:, None] + residual.to(device)

        weights = {}
        for name, (s, e, r, c) in slices.items():
            weights[name] = unblock(block_vals[s:e].float(), (r, c))

        # Forward through MLP
        z_a = F.silu(x_a.to(device) @ weights["gate_proj"].T) * (x_a.to(device) @ weights["up_proj"].T)
        y_a = z_a @ weights["down_proj"].T
        h2_a = e9["h1_a"].to(device) + y_a

        if USE_CE:
            logits = run_full_model(h2_a, model, tok_a.to(device))
            loss = F.cross_entropy(logits[:-1], tok_a.to(device)[1:], reduction="mean")
        else:
            h4_a = run_downstream(h2_a, model)
            loss = (h4_a - h4t_a.to(device)).square().mean()

        loss.backward()

        grad = float_protos.grad.detach().cpu()  # (16, 256, 16)
        float_protos = float_protos.detach().cpu()

        # Move each code ±1 level based on gradient sign
        move = -torch.sign(grad).long().reshape(-1, 16)  # (4096, 16)
        new_codes = (codes.long() + move).clamp(0, 63).to(torch.uint8)
        codes = new_codes

        # Restore float_protos from quantized values
        levels = checkpoint.float()
        pf = torch.arange(codes.shape[0], dtype=torch.long) // n_per_family
        proto_vals = levels[pf[:, None], codes.long()]
        float_protos = proto_vals.reshape(n_families, n_per_family, 16)

        loss_val = float(loss.detach().cpu())
        if loss_val < best_loss:
            best_loss = loss_val

        print(json.dumps({"event": "step", "step": step + 1, "loss": loss_val,
                          "best_loss": best_loss, "lr": max(LR * (1 - step / STEPS), 1e-8),
                          "elapsed_seconds": time.time() - step_started,
                          "memory": memory_state()}), flush=True)

        if (step + 1) % 10 == 0:
            state = {"parent_sha256": sha256(E10A_PACKED),
                     "epoch": step + 1, "float_protos": float_protos,
                     "codes": codes}
            STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, STATE_PATH)
            # Also write partial packed (codes need flat shape: 4096, 16)
            codes_flat = codes.reshape(-1, 16).to(torch.uint8)
            patch_pattern_codes(E10A_PACKED, OUTPUT, codes_flat)
            print(json.dumps({"event": "checkpoint", "step": step + 1,
                               "output_size": OUTPUT.stat().st_size}), flush=True)

    # ── Finalize ──
    codes_flat = codes.reshape(-1, 16).to(torch.uint8)
    patch_pattern_codes(E10A_PACKED, OUTPUT, codes_flat)
    result = {"experiment": LABEL, "steps": STEPS, "lr": LR,
              "use_ce": USE_CE, "final_loss": loss_val, "best_loss": best_loss,
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size * 8 / meta["weights"],
              "output_sha256": sha256(OUTPUT),
              "elapsed_seconds": time.time() - started, "ppl": None}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
