"""E10C1: Tail analysis + sensitivity subspace construction.

Phase 1: Measure excess NLL concentration across calibration tokens.
Phase 2: Build sensitivity subspace M from subset-averaged CE gradients.

For M, uses 16 random subsets of 32 tokens each → 16 gradient vectors
g_i = ∂(Σ CE_t)/∂h2 for each subset. Then M_global = Σ g_i g_i^T
and M_tail = Σ w_i g_i g_i^T where w_i weights subsets by excess CE.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import decode_weights


ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
TOKENS_PATH = (ROOT.parent / "artifacts/functional_vq_l0/calibration_cache/fixed_token_ids.pt").resolve()
E10A_PACKED = ROOT / "packed/E10a_gate_up_downstream.packed"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
N_SUBSETS = 16
SUBSET_SIZE = 32
OUTPUT = ROOT / "cache/e10c_basis.pt"


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


def run_ce(model, token_ids):
    with torch.no_grad():
        out = model(token_ids.unsqueeze(0), labels=token_ids.unsqueeze(0), use_cache=False)
        logits = out.logits[0]
    shift = logits[:-1]
    labels = token_ids[1:]
    ce = F.cross_entropy(shift, labels, reduction="none")
    return ce, logits


def compute_subset_gradient(model, h2_input, token_ids):
    """Compute ∂(Σ CE_t)/∂h2 for a subset of tokens (single backward pass)."""
    device = h2_input.device
    h2 = h2_input.detach().clone().requires_grad_(True)

    h = h2
    for i in range(1, 24):
        layer = model.model.layers[i]
        pos_ids = torch.arange(h.shape[0], device=device).unsqueeze(0)
        causal_mask = torch.full((h.shape[0], h.shape[0]), torch.finfo(h.dtype).min, device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h.unsqueeze(0), pos_ids)
        pos_emb = (cos, sin)
        h_norm = layer.input_layernorm(h)
        attn_out = layer.self_attn(h_norm.unsqueeze(0), attention_mask=causal_mask,
                                   position_embeddings=pos_emb)[0].squeeze(0)
        h = h + attn_out
        h_attn = h
        h = layer.post_attention_layernorm(h)
        h = h_attn + layer.mlp(h)

    logits = model.lm_head(h)
    ce_total = F.cross_entropy(logits[:-1], token_ids[1:], reduction="sum")
    g = torch.autograd.grad(ce_total, h2, create_graph=False)[0]
    return g


def main():
    started = time.time()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy = torch.zeros(1, device=device)
    del dummy

    token_sets = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    heldout = token_sets["heldout"]
    calib_b = heldout[512:1024]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval().to(device)

    original_weights = {name: mod.weight.detach().clone()
                        for name, mod in module_map(model).items()}

    # ── Phase 1: Tail analysis ──
    print(json.dumps({"event": "teacher_ce"}), flush=True)
    teacher_ce, _ = run_ce(model, calib_b.to(device))

    print(json.dumps({"event": "student_e10a_ce"}), flush=True)
    decoded, _ = decode_weights(E10A_PACKED)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value.to(device))
    student_ce, _ = run_ce(model, calib_b.to(device))
    student_ce = student_ce.cpu()
    teacher_ce = teacher_ce.cpu()

    excess = student_ce - teacher_ce
    pos_excess = excess.clamp(min=0)
    total_pos = float(pos_excess.sum())

    report = {"checkpoint": str(E10A_PACKED), "device": device,
              "n_tokens": int(excess.shape[0]),
              "mean_excess_NLL": float(excess.mean()),
              "median_excess_NLL": float(excess.median()),
              "max_excess_NLL": float(excess.max()),
              "total_pos_excess_NLL": total_pos,
              "pct_tokens_with_positive_excess": float((pos_excess > 0).float().mean() * 100)}

    sorted_excess = pos_excess.sort(descending=True).values
    cumulative = sorted_excess.cumsum(0)
    for pct in [1, 2, 5, 10, 20, 50]:
        n = max(1, int(len(sorted_excess) * pct / 100))
        frac = float(cumulative[n - 1] / max(total_pos, 1e-10))
        report[f"tail_{pct}pct"] = frac
        print(json.dumps({"event": "tail", "pct": pct, "frac": frac}), flush=True)

    # ── Phase 2: Subset gradients ──
    print(json.dumps({"event": "restoring_student_weights"}), flush=True)
    decoded, _ = decode_weights(E10A_PACKED)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value.to(device))

    # Compute h2 for student using E9 cache
    e9_cache = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    x = e9_cache["x_b"].to(device)
    h1 = e9_cache["h1_b"].to(device)
    with torch.no_grad():
        gate = x @ model.model.layers[0].mlp.gate_proj.weight.T
        up = x @ model.model.layers[0].mlp.up_proj.weight.T
        z = F.silu(gate) * up
        y = z @ model.model.layers[0].mlp.down_proj.weight.T
    h2_student = h1 + y

    # Generate random subsets
    rng = np.random.RandomState(42)
    all_idx = np.arange(511)
    subset_grads = []

    print(json.dumps({"event": "subset_gradients", "n_subsets": N_SUBSETS,
                       "subset_size": SUBSET_SIZE}), flush=True)

    for si in range(N_SUBSETS):
        idx = rng.choice(all_idx, SUBSET_SIZE, replace=False)
        idx.sort()
        idx_t = torch.tensor(idx, device=device)
        h2_sub = h2_student[idx_t]
        tok_sub = calib_b[idx_t].to(device)
        g = compute_subset_gradient(model, h2_sub, tok_sub)
        subset_grads.append(g.cpu())
        print(json.dumps({"event": "subset_done", "subset": si + 1,
                           "grad_norm": float(g.norm())}), flush=True)

    g_stack = torch.stack(subset_grads)  # (16, 32, 896)
    g_avg = g_stack.mean(dim=1)           # (16, 896) — per-subset average

    # For tail weighting: compute excess CE per subset
    subset_excess = torch.zeros(N_SUBSETS)
    for si in range(N_SUBSETS):
        idx = rng.choice(all_idx, SUBSET_SIZE, replace=False)
        subset_excess[si] = pos_excess[torch.tensor(idx)].sum()

    # ── Build M matrices ──
    M_global = g_avg.T @ g_avg  # (896, 896)
    sw = subset_excess / max(float(subset_excess.sum()), 1e-10)
    g_weighted = g_avg * sw.sqrt().unsqueeze(1)
    M_tail = g_weighted.T @ g_weighted

    report["subset_grad_norms"] = [float(g.norm()) for g in subset_grads]

    def analyze_eigens(M, name):
        M_np = M.numpy()
        M_sym = (M_np + M_np.T) / 2
        evals, evecs = np.linalg.eigh(M_sym)
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]
        cum = np.cumsum(evals)
        cum_frac = cum / cum[-1]
        vals = {}
        for r in [1, 2, 4, 8, 16, 32, 64]:
            vals[f"top_{r}"] = float(cum_frac[min(r-1, len(cum_frac)-1)])
        vals["eval_top1"] = float(evals[0])
        vals["eval_ratio_1_to_64"] = float(evals[0] / max(evals[-1], 1e-30))
        vals["eval_entropy"] = float(-np.sum((evals/evals.sum()) * np.log(evals/evals.sum() + 1e-30)))
        print(json.dumps({"event": "subspace", "basis": name, **vals}), flush=True)
        return evals[:64], evecs[:, :64]

    evals_global, evecs_global = analyze_eigens(M_global, "global")
    evals_tail, evecs_tail = analyze_eigens(M_tail, "tail")

    # ── Save basis ──
    basis = {
        "M_global": M_global,
        "M_tail": M_tail,
        "eigenvalues_global": torch.from_numpy(evals_global),
        "eigenvectors_global": torch.from_numpy(evecs_global),
        "eigenvalues_tail": torch.from_numpy(evals_tail),
        "eigenvectors_tail": torch.from_numpy(evecs_tail),
        "per_token_excess_CE": excess,
        "h2_student": h2_student.cpu(),
        "h2_teacher": e9_cache["h2_teacher_b"].clone(),
        "x": x.cpu(),
        "h1": h1.cpu(),
        "original_weights": {k: v.cpu() for k, v in e9_cache["original_weights"].items()},
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, OUTPUT)

    report["elapsed_seconds"] = time.time() - started
    print(json.dumps({"event": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()
