"""Diagnostic: compare E8g, E9a, E9b on +1-layer, +2-layer, and logit-KL metrics.

Tests which proxy correlates with the known PPL ordering:
  E9b (73.155) < E8g (76.169) < E9a (76.792)

Lower metric should correspond to lower PPL across all three checkpoints.
Does NOT modify any checkpoint — read-only evaluation on calibration B.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import decode_weights


ROOT = Path(__file__).resolve().parents[1]
TOKENS_PATH = (ROOT.parent / "artifacts/functional_vq_l0/calibration_cache/fixed_token_ids.pt").resolve()
PACKED_DIR = ROOT / "packed"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"

CHECKPOINTS = {
    "E8g": PACKED_DIR / "E8g_down_polish_round5.packed",
    "E9a": PACKED_DIR / "E9a_down_transition_5pct.packed",
    "E9b": PACKED_DIR / "E9b_downstream_l1_5pct.packed",
}

PPL_VALUES = {
    "E8g": 76.169,
    "E9a": 76.792,
    "E9b": 73.155,
}


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compute_logit_kl(student_logits, teacher_logits):
    """Per-token KL(teacher || student), averaged over tokens."""
    teacher_prob = F.softmax(teacher_logits.float(), dim=-1)
    student_log_prob = F.log_softmax(student_logits.float(), dim=-1)
    kl_per_token = (teacher_prob * (teacher_prob.log() - student_log_prob)).sum(-1)
    return float(kl_per_token.mean())


def capture_outputs(model, token_ids, seq_len=512):
    """Run model on token_ids, capture h2 (L0 out), h3 (L1 out), h4 (L2 out), logits."""
    captured = {"h2": [], "h3": [], "h4": [], "logits": []}

    def make_hook(key):
        def hook(_module, _inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[key].append(h.detach().float().cpu())
        return hook

    handles = [
        model.model.layers[0].register_forward_hook(make_hook("h2")),
        model.model.layers[1].register_forward_hook(make_hook("h3")),
        model.model.layers[2].register_forward_hook(make_hook("h4")),
    ]
    n = token_ids.numel()
    with torch.no_grad():
        for start in range(0, n, seq_len):
            end = min(start + seq_len, n)
            batch = token_ids[start:end].unsqueeze(0)
            out = model(batch, labels=batch, use_cache=False)
            captured["logits"].append(out.logits.detach().float().cpu())
    for h in handles:
        h.remove()

    def cat(key):
        if key == "logits":
            return torch.cat(captured[key], dim=1).squeeze(0)
        flat = [v.reshape(-1, v.shape[-1]) for v in captured[key]]
        return torch.cat(flat, dim=0)

    return {k: cat(k) for k in ("h2", "h3", "h4", "logits")}


def main():
    started = time.time()
    torch.set_num_threads(4)

    token_sets = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    heldout = token_sets["heldout"]
    calib_b = heldout[512:1024]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True
    ).eval()

    print(json.dumps({"event": "compute_teacher"}), flush=True)
    original_weights = {name: mod.weight.detach().clone()
                        for name, mod in module_map(model).items()}
    teacher = capture_outputs(model, calib_b)

    results = {}
    for label in ("E8g", "E9a", "E9b"):
        packed = CHECKPOINTS[label]
        if not packed.exists():
            print(json.dumps({"event": "skip", "checkpoint": label, "reason": "not found"}), flush=True)
            continue
        print(json.dumps({"event": "evaluate", "checkpoint": label, "packed": str(packed)}), flush=True)
        decoded, _ = decode_weights(packed)
        with torch.no_grad():
            for name, value in decoded.items():
                module_map(model)[name].weight.copy_(value)
        student = capture_outputs(model, calib_b)
        with torch.no_grad():
            for name, value in original_weights.items():
                module_map(model)[name].weight.copy_(value)

        h2_err = (student["h2"] - teacher["h2"]).float()
        h3_err = (student["h3"] - teacher["h3"]).float()
        h4_err = (student["h4"] - teacher["h4"]).float()

        den_h2 = teacher["h2"].float().square().sum()
        den_h3 = teacher["h3"].float().square().sum()
        den_h4 = teacher["h4"].float().square().sum()

        L_h2 = float(h2_err.square().sum() / den_h2)
        L_plus1 = float(h3_err.square().sum() / den_h3)
        L_plus2 = float(h4_err.square().sum() / den_h4)
        kl_val = compute_logit_kl(student["logits"], teacher["logits"])

        results[label] = {
            "L_h2": L_h2,
            "L_plus1": L_plus1,
            "L_plus2": L_plus2,
            "KL": kl_val,
            "ppl": PPL_VALUES[label],
        }
        print(json.dumps({"event": "result", "checkpoint": label, **results[label]}), flush=True)

    metric_names = ["L_h2", "L_plus1", "L_plus2", "KL"]
    ordering = {"E9b": 0, "E8g": 1, "E9a": 2}
    correct_ppl_order = ["E9b", "E8g", "E9a"]

    diagnostic = {"checkpoints": list(results.keys()), "known_ppl_order": correct_ppl_order,
                  "ppl_values": {k: results[k]["ppl"] for k in results}, "metrics": {}}

    for metric in metric_names:
        vals = {k: results[k][metric] for k in results}
        ordered = sorted(vals, key=lambda k: vals[k])
        matches_ppl = ordered == correct_ppl_order
        diagnostic["metrics"][metric] = {
            "values": vals,
            "ranked": ordered,
            "matches_ppl_order": matches_ppl,
        }
        print(json.dumps({"event": "metric_check", "metric": metric,
                          "values": vals, "ranked": ordered,
                          "matches_ppl_order": matches_ppl}), flush=True)

    best_metric = None
    for metric in metric_names:
        if diagnostic["metrics"][metric]["matches_ppl_order"]:
            best_metric = metric
            break

    diagnostic["recommended_metric"] = best_metric
    diagnostic["elapsed_seconds"] = time.time() - started
    print(json.dumps({"event": "diagnostic_complete", **diagnostic}), flush=True)


if __name__ == "__main__":
    main()
