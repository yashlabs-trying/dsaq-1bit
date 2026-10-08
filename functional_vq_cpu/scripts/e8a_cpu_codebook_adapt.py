"""E8a same-byte prototype adaptation after E7 functional reassignment."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from cpu_codec import PATTERNS_PER_FAMILY, codec_state, patch_pattern_codes, read_codec, unblock


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cache/layer0_functional_cache.pt"
PARENT = Path(os.environ.get("E8_INPUT", ROOT / "packed/E7d_down_polish_5pct.packed"))
OUTPUT = Path(os.environ.get("E8_OUTPUT", ROOT / "packed/E8a_codebook_30.packed"))
RESULT = Path(os.environ.get("E8_RESULT", ROOT / "results/E8a_codebook_30.json"))
STEPS = int(os.environ.get("E8_STEPS", "30"))
TOPK = int(os.environ.get("E8_TOPK", "512"))
MIN_TRIAL = int(os.environ.get("E8_MIN_TRIAL", "8"))
TRUST = float(os.environ.get("E8_TRUST", "0.01"))
GU_MIN = float(os.environ.get("E8_GU_MIN", "0.62"))
ATT_MAX = float(os.environ.get("E8_ATT_MAX", "0.075"))
THREADS = int(os.environ.get("E8_THREADS", "4"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mlp(weights, x):
    gate = x @ weights["gate_proj"].T
    up = x @ weights["up_proj"].T
    z = torch.nn.functional.silu(gate) * up
    return z, z @ weights["down_proj"].T


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    meta, scale_tables, checkpoint, codes, pattern_ids, scale_ids, hard, residual_ids, residual_q, residual_scale = read_codec(PARENT)
    slices = {name: tuple(meta["slices"][name]) for name in meta["names"]}
    family = (torch.arange(codes.shape[0]) // PATTERNS_PER_FAMILY).long()
    matrix_index = torch.empty(meta["nblocks"], dtype=torch.long)
    for index, name in enumerate(meta["names"]):
        start, end, _, _ = slices[name]
        matrix_index[start:end] = index
    scales = scale_tables.float()[matrix_index, scale_ids.long()]
    residual = torch.zeros(meta["nblocks"], 16)
    if hard is not None:
        residual[hard] = residual_q.float()[residual_ids.long()] * residual_scale.float()[residual_ids.long(), None]

    usage = {}
    for name in meta["names"]:
        start, end, _, _ = slices[name]
        usage[name] = torch.bincount(pattern_ids[start:end].long(), minlength=codes.shape[0])
    attention_usage = sum(usage[name] for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
    gate_up_usage = usage["gate_proj"] + usage["up_proj"]
    down_usage = usage["down_proj"]
    total_usage = attention_usage + gate_up_usage + down_usage
    selected = ((gate_up_usage.float() / total_usage.clamp_min(1)) >= GU_MIN) & ((attention_usage.float() / total_usage.clamp_min(1)) <= ATT_MAX)
    selected_coordinates = selected[:, None].expand(-1, 16)

    def decoded_patterns(code_tensor):
        return checkpoint.float()[family[:, None], code_tensor.long()]

    def mlp_weights(patterns):
        weights = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            start, end, rows, cols = slices[name]
            blocks = patterns[pattern_ids[start:end].long()] * scales[start:end, None] + residual[start:end]
            weights[name] = unblock(blocks, (rows, cols))
        return weights

    x_a, x_b = cache["x_a"], cache["x_b"]
    y_teacher_a, y_teacher_b = cache["y_teacher_a"], cache["y_teacher_b"]
    z_teacher_a, z_teacher_b = cache["z_teacher_a"], cache["z_teacher_b"]
    yden_a, yden_b = y_teacher_a.square().sum(), y_teacher_b.square().sum()
    zden_a, zden_b = z_teacher_a.square().sum(), z_teacher_b.square().sum()

    @torch.no_grad()
    def metrics(code_tensor):
        weights = mlp_weights(decoded_patterns(code_tensor))
        z_a, y_a = mlp(weights, x_a)
        z_b, y_b = mlp(weights, x_b)
        return {"Lz_A": float((z_a-z_teacher_a).square().sum()/zden_a),
                "Lz_B": float((z_b-z_teacher_b).square().sum()/zden_b),
                "Ly_A": float((y_a-y_teacher_a).square().sum()/yden_a),
                "Ly_B": float((y_b-y_teacher_b).square().sum()/yden_b),
                "Ez_A": float((z_a-z_teacher_a).norm()/z_teacher_a.norm()),
                "Ez_B": float((z_b-z_teacher_b).norm()/z_teacher_b.norm()),
                "Ey_A": float((y_a-y_teacher_a).norm()/y_teacher_a.norm()),
                "Ey_B": float((y_b-y_teacher_b).norm()/y_teacher_b.norm())}

    base_patterns = decoded_patterns(codes).detach()
    current_codes = codes.clone()
    current = metrics(current_codes)
    initial = dict(current)
    history = []
    print(json.dumps({"event": "start", "parent": str(PARENT),
                      "selected_prototypes": int(selected.sum()),
                      "selected_gate_up_uses": int(gate_up_usage[selected].sum()),
                      "selected_attention_uses": int(attention_usage[selected].sum()),
                      "selected_down_uses": int(down_usage[selected].sum()),
                      "metrics": current}), flush=True)

    for step in range(1, STEPS + 1):
        patterns = decoded_patterns(current_codes).detach().requires_grad_(True)
        weights = mlp_weights(patterns)
        _, y_a = mlp(weights, x_a)
        loss_y = (y_a - y_teacher_a).square().sum() / yden_a
        trust = ((patterns-base_patterns).square()*selected_coordinates).sum()/selected_coordinates.sum().clamp_min(1)
        objective = loss_y + TRUST * trust
        objective.backward()
        gradient = patterns.grad.detach()
        code_long = current_codes.long()
        direction = torch.where(gradient < 0, torch.ones_like(code_long), -torch.ones_like(code_long))
        proposed = (code_long + direction).clamp(0, 63)
        current_value = checkpoint.float()[family[:, None], code_long]
        proposed_value = checkpoint.float()[family[:, None], proposed]
        predicted_gain = -(gradient * (proposed_value-current_value))
        valid = selected_coordinates & (proposed != code_long) & (predicted_gain > 0)
        available = int(valid.sum())
        if available == 0:
            break
        score = torch.where(valid, predicted_gain, torch.full_like(predicted_gain, -torch.inf))
        order = torch.topk(score.flatten(), min(TOPK, available)).indices
        accepted = None
        trial_sizes = []
        trial = min(TOPK, available)
        while trial >= MIN_TRIAL:
            trial_sizes.append(trial)
            candidate = current_codes.clone()
            chosen = order[:trial]
            candidate.view(-1)[chosen] = proposed.flatten()[chosen].to(candidate.dtype)
            candidate_metrics = metrics(candidate)
            if candidate_metrics["Ly_A"] < current["Ly_A"] and candidate_metrics["Ly_B"] <= current["Ly_B"] + 1e-10:
                accepted = candidate, candidate_metrics, trial
                break
            trial //= 2
        event = {"step": step, "objective_A": float(objective.detach()), "trust": float(trust.detach()),
                 "available_moves": available, "trial_sizes": trial_sizes, "accepted": accepted is not None}
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

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    patch_pattern_codes(PARENT, OUTPUT, current_codes)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")
    result = {"experiment": "E8a_same_byte_codebook_adaptation", "input": str(PARENT),
              "input_sha256": sha256(PARENT), "output": str(OUTPUT),
              "steps_requested": STEPS, "steps_accepted": sum(int(x["accepted"]) for x in history),
              "topk": TOPK, "min_trial": MIN_TRIAL, "trust_lambda": TRUST,
              "gate_up_ratio_min": GU_MIN, "attention_ratio_max": ATT_MAX,
              "selected_prototypes": int(selected.sum()),
              "selected_gate_up_uses": int(gate_up_usage[selected].sum()),
              "selected_attention_uses": int(attention_usage[selected].sum()),
              "selected_down_uses": int(down_usage[selected].sum()),
              "changed_pattern_coordinates": int((current_codes != codes).sum()),
              "changed_prototypes": int((current_codes != codes).any(1).sum()),
              "initial": initial, "final": current, "history": history,
              "bytes": OUTPUT.stat().st_size, "bpw": OUTPUT.stat().st_size*8/meta["weights"],
              "output_sha256": sha256(OUTPUT), "elapsed_seconds": time.time()-started,
              "ppl": None, "ppl_note": "CPU worker does not claim PPL; requires strict full-model evaluation"}
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
