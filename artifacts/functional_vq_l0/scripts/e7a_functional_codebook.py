"""E7a: same-byte discrete functional adaptation of MLP-dominant prototypes.

The floating decoded patterns receive gradients from final Layer-0 MLP output
loss. Each accepted update is projected immediately onto the codec's existing
family-specific 64-level tables, so the only stored changes are 6-bit pattern
codes. Calibration A proposes updates; calibration B gates every acceptance.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import waverel_checkpoint_patterns as base


ROOT = Path("/workspace/experiments/functional_vq_l0")
INPUT = Path(os.environ.get("E7_INPUT", ROOT / "packed/E6_assignment_40pct.packed"))
OUTPUT = Path(os.environ.get("E7_OUTPUT", ROOT / "packed/E7a_codebook_short.packed"))
RESULT = Path(os.environ.get("E7_RESULT", ROOT / "results/E7a_codebook_short.json"))
TOKENS = ROOT / "calibration_cache/fixed_token_ids.pt"
STEPS = int(os.environ.get("E7_STEPS", "30"))
TOPK = int(os.environ.get("E7_TOPK", "512"))
TRUST = float(os.environ.get("E7_TRUST", "0.01"))
GU_MIN = float(os.environ.get("E7_GU_MIN", "0.62"))
ATT_MAX = float(os.environ.get("E7_ATT_MAX", "0.075"))


@torch.inference_mode()
def capture_x(model, ids):
    values = []
    module = model.model.layers[0].mlp.gate_proj
    hook = module.register_forward_pre_hook(
        lambda _module, inputs: values.append(inputs[0].detach().float().cpu())
    )
    for start in range(0, ids.numel() - 255, 256):
        model(ids[start : start + 256].unsqueeze(0).cuda(), use_cache=False)
    hook.remove()
    return torch.cat([value.reshape(-1, value.shape[-1]) for value in values])


def mlp(weights, x):
    gate = x @ weights["gate_proj"].T
    up = x @ weights["up_proj"].T
    z = torch.nn.functional.silu(gate) * up
    return z, z @ weights["down_proj"].T


def patch_pattern_codes(src: Path, dst: Path, new_codes: torch.Tensor):
    blob = bytearray(src.read_bytes())
    meta_len = struct.unpack_from("<I", blob, 0)[0]
    meta = json.loads(blob[4 : 4 + meta_len])
    pos = 4 + meta_len
    pos += 2 * meta["nblocks"]
    pos += 2 * len(meta["names"]) * base.SCALE_LEVELS
    pos += 2 * math.prod(meta["checkpoint_shape"])
    count = math.prod(meta["pattern_shape"])
    byte_count = (count * meta["pattern_bits"] + 7) // 8
    packed = base.pack_fixed(new_codes, meta["pattern_bits"])
    assert len(packed) == byte_count
    blob[pos : pos + byte_count] = packed
    dst.write_bytes(blob)


def main():
    started = time.time()
    base.seed_all()
    model = AutoModelForCausalLM.from_pretrained(
        base.MODEL, dtype=torch.float32, device_map={"": "cuda"}
    ).eval()
    original = {
        name: module.weight.detach().cpu().clone()
        for name, module in base.module_map(model).items()
    }
    token_sets = torch.load(TOKENS, map_location="cpu", weights_only=True)
    heldout = token_sets["heldout"]
    ids_a, ids_b = heldout[:512], heldout[512:1024]
    eval_ids = token_sets["evaluation"]

    base.restore(model, original)
    x_a = capture_x(model, ids_a).cuda()
    x_b = capture_x(model, ids_b).cuda()
    teacher_weights = {
        name: original[name].cuda()
        for name in ("gate_proj", "up_proj", "down_proj")
    }
    with torch.inference_mode():
        z_teacher_a, y_teacher_a = mlp(teacher_weights, x_a)
        z_teacher_b, y_teacher_b = mlp(teacher_weights, x_b)
    zden_a, zden_b = z_teacher_a.square().sum(), z_teacher_b.square().sum()
    yden_a, yden_b = y_teacher_a.square().sum(), y_teacher_b.square().sum()

    meta, scale_tables, checkpoint, codes, pattern_ids, scale_ids, hard, residual_ids, residual_q, residual_scale = base.read_codec(INPUT)
    slices = {name: tuple(meta["slices"][name]) for name in meta["names"]}
    family = (torch.arange(codes.shape[0]) // base.PATTERNS_PER_FAMILY).long()
    matrix_index = torch.empty(meta["nblocks"], dtype=torch.long)
    for index, name in enumerate(meta["names"]):
        start, end, _, _ = slices[name]
        matrix_index[start:end] = index
    scales = scale_tables[matrix_index, scale_ids.long()].float().cuda()
    residual = torch.zeros(meta["nblocks"], 16)
    if hard is not None:
        residual[hard] = residual_q.float()[residual_ids.long()] * residual_scale.float()[residual_ids.long(), None]
    residual = residual.cuda()
    pattern_ids_gpu = pattern_ids.long().cuda()
    checkpoint_gpu = checkpoint.float().cuda()
    family_gpu = family.cuda()

    # Usage-based safety mask. Prototypes are globally shared, so "MLP relevant"
    # is determined by actual E6 assignments rather than by pattern family.
    usage = {}
    for name in meta["names"]:
        start, end, _, _ = slices[name]
        usage[name] = torch.bincount(pattern_ids[start:end].long(), minlength=codes.shape[0])
    attention_usage = sum(usage[name] for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    gate_up_usage = usage["gate_proj"] + usage["up_proj"]
    down_usage = usage["down_proj"]
    total_usage = attention_usage + gate_up_usage + down_usage
    gate_up_ratio = gate_up_usage.float() / total_usage.clamp_min(1).float()
    attention_ratio = attention_usage.float() / total_usage.clamp_min(1).float()
    selected = (gate_up_ratio >= GU_MIN) & (attention_ratio <= ATT_MAX)
    selected_gpu = selected.cuda()[:, None].expand(-1, 16)

    def decoded_patterns(code_tensor):
        code_gpu = code_tensor.long().cuda()
        return checkpoint_gpu[family_gpu[:, None], code_gpu]

    def mlp_weights(patterns):
        weights = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            start, end, rows, cols = slices[name]
            block_values = (
                patterns[pattern_ids_gpu[start:end]] * scales[start:end, None]
                + residual[start:end]
            )
            weights[name] = base.unblock(block_values, (rows, cols))
        return weights

    @torch.inference_mode()
    def metrics(code_tensor):
        patterns = decoded_patterns(code_tensor)
        weights = mlp_weights(patterns)
        z_a, y_a = mlp(weights, x_a)
        z_b, y_b = mlp(weights, x_b)
        return {
            "Lz_A": float((z_a - z_teacher_a).square().sum() / zden_a),
            "Lz_B": float((z_b - z_teacher_b).square().sum() / zden_b),
            "Ly_A": float((y_a - y_teacher_a).square().sum() / yden_a),
            "Ly_B": float((y_b - y_teacher_b).square().sum() / yden_b),
            "Ez_A": float((z_a - z_teacher_a).norm() / z_teacher_a.norm()),
            "Ez_B": float((z_b - z_teacher_b).norm() / z_teacher_b.norm()),
            "Ey_A": float((y_a - y_teacher_a).norm() / y_teacher_a.norm()),
            "Ey_B": float((y_b - y_teacher_b).norm() / y_teacher_b.norm()),
        }

    base_patterns = decoded_patterns(codes).detach()
    current_codes = codes.clone()
    current = metrics(current_codes)
    initial = dict(current)
    history = []
    print(json.dumps({
        "event": "start",
        "selected_prototypes": int(selected.sum()),
        "selected_gate_up_uses": int(gate_up_usage[selected].sum()),
        "selected_attention_uses": int(attention_usage[selected].sum()),
        "selected_down_uses": int(down_usage[selected].sum()),
        "metrics": current,
    }), flush=True)

    for step in range(1, STEPS + 1):
        patterns = decoded_patterns(current_codes).detach().requires_grad_(True)
        weights = mlp_weights(patterns)
        _, y_a = mlp(weights, x_a)
        loss_y = (y_a - y_teacher_a).square().sum() / yden_a
        trust = ((patterns - base_patterns).square() * selected_gpu).sum() / selected_gpu.sum().clamp_min(1)
        objective = loss_y + TRUST * trust
        objective.backward()

        gradient = patterns.grad.detach()
        code_gpu = current_codes.long().cuda()
        direction = torch.where(gradient < 0, torch.ones_like(code_gpu), -torch.ones_like(code_gpu))
        proposed_code = (code_gpu + direction).clamp(0, base.LEVELS - 1)
        current_value = checkpoint_gpu[family_gpu[:, None], code_gpu]
        proposed_value = checkpoint_gpu[family_gpu[:, None], proposed_code]
        predicted_gain = -(gradient * (proposed_value - current_value))
        valid = selected_gpu & (proposed_code != code_gpu) & (predicted_gain > 0)
        score = torch.where(valid, predicted_gain, torch.full_like(predicted_gain, -torch.inf))
        available = int(valid.sum())
        if available == 0:
            print(json.dumps({"event": "stop", "step": step, "reason": "no_valid_projected_moves"}), flush=True)
            break

        order = torch.topk(score.flatten(), min(TOPK, available)).indices
        accepted = None
        trial_sizes = []
        trial = min(TOPK, available)
        while trial >= 8:
            trial_sizes.append(trial)
            candidate = current_codes.clone()
            flat_candidate = candidate.view(-1)
            chosen = order[:trial].cpu()
            flat_proposed = proposed_code.flatten().cpu()
            flat_candidate[chosen] = flat_proposed[chosen].to(flat_candidate.dtype)
            candidate_metrics = metrics(candidate)
            if candidate_metrics["Ly_A"] < current["Ly_A"] and candidate_metrics["Ly_B"] <= current["Ly_B"] + 1e-10:
                accepted = (candidate, candidate_metrics, trial)
                break
            trial //= 2

        event = {
            "step": step,
            "objective_A": float(objective.detach()),
            "trust": float(trust.detach()),
            "available_moves": available,
            "trial_sizes": trial_sizes,
            "accepted": accepted is not None,
        }
        if accepted is None:
            event["reason"] = "calibration_B_rejected_all_projected_batches"
            history.append(event)
            print(json.dumps(event), flush=True)
            break

        current_codes, current, moved = accepted
        event["moved_coordinates"] = moved
        event["metrics"] = current
        history.append(event)
        print(json.dumps(event), flush=True)

    patch_pattern_codes(INPUT, OUTPUT, current_codes)
    assert OUTPUT.stat().st_size == INPUT.stat().st_size
    decoded, decoded_meta = base.decode(OUTPUT)
    base.restore(model, decoded)
    ppl = base.perplexity(model, eval_ids, 256)
    hashes = {
        name: hashlib.sha256(weight.numpy().tobytes()).hexdigest()
        for name, weight in decoded.items()
    }
    result = {
        "experiment": "E7a_same_byte_functional_codebook_short",
        "input": str(INPUT),
        "output": str(OUTPUT),
        "steps_requested": STEPS,
        "steps_accepted": sum(int(item["accepted"]) for item in history),
        "topk": TOPK,
        "trust_lambda": TRUST,
        "gate_up_ratio_min": GU_MIN,
        "attention_ratio_max": ATT_MAX,
        "selected_prototypes": int(selected.sum()),
        "selected_gate_up_uses": int(gate_up_usage[selected].sum()),
        "selected_attention_uses": int(attention_usage[selected].sum()),
        "selected_down_uses": int(down_usage[selected].sum()),
        "changed_pattern_coordinates": int((current_codes != codes).sum()),
        "changed_prototypes": int((current_codes != codes).any(1).sum()),
        "initial": initial,
        "final": current,
        "history": history,
        "bytes": OUTPUT.stat().st_size,
        "bpw": OUTPUT.stat().st_size * 8 / meta["weights"],
        "ppl": ppl,
        "delta_ppl_pct": 100 * (ppl["ppl"] / 22.188270476805936 - 1),
        "decoded_hashes": hashes,
        "elapsed_seconds": time.time() - started,
    }
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
