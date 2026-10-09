"""FR1: Gate/Up functional residual codebook with CE gradient.

Sanity test → build correction prototypes → staged apply → B-validate → PPL.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, read_codec, unblock

ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
PARENT = ROOT / "packed/E10a_gate_up_downstream.packed"
OUTPUT = ROOT / "packed" / "FR1_gate_up_residual.packed"
RESULT = ROOT / "results" / "FR1_gate_up_residual.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TAIL_POWER = 0.5
COVERAGE = float(os.environ.get("FR1_COV", "0.02"))
ETA_TEST = float(os.environ.get("FR1_ETA_TEST", "0.0"))
N_PROTOS = int(os.environ.get("FR1_K", "64"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def km(d, k, seed=42, n_iter=50):
    n, _ = d.shape
    rng = np.random.RandomState(seed)
    c = d[rng.choice(n, k, False)].copy()
    p = None
    for _ in range(n_iter):
        ds = np.zeros((n, k))
        for i in range(k):
            df = d - c[i]; ds[:, i] = np.sum(df * df, 1)
        l = ds.argmin(1)
        if p is not None and np.array_equal(l, p): break
        p = l
        for i in range(k):
            m = l == i
            if m.sum(): c[i] = d[m].mean(0)
    return c, l


def main():
    torch.set_num_threads(4)
    t0 = time.time()

    e9 = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    x_a, x_b = e9["x_a"], e9["x_b"]
    h1_a, h1_b = e9["h1_a"], e9["h1_b"]
    h2t_a, h2t_b = e9["h2_teacher_a"], e9["h2_teacher_b"]
    tok_a, tok_b = e9["token_ids_a"], e9["token_ids_b"]

    rm, st, ckpt, oc, oi_np, si_np, hard, rid, rq, rs = read_codec(PARENT)
    slices = {n: tuple(rm["slices"][n]) for n in rm["names"]}
    pf = torch.arange(4096, dtype=torch.long) // 256
    cod = oc.clone().long() if isinstance(oc, torch.Tensor) else torch.from_numpy(oc.copy()).long()
    pv = ckpt.float()[pf[:, None], cod]
    def to_t(x): return x.clone() if isinstance(x, torch.Tensor) else torch.from_numpy(x.copy())
    oi = to_t(oi_np).long(); si = to_t(si_np).long()
    mi = torch.empty(rm["nblocks"], dtype=torch.long)
    for idx, nm in enumerate(rm["names"]):
        s, e, _, _ = slices[nm]; mi[s:e] = idx
    sf = st.float()[mi, si.clamp(0, 15)]
    res = torch.zeros(rm["nblocks"], 16)
    if hard is not None: res[hard] = rq.float()[rid.long()] * rs.float()[rid.long(), None]
    def decode(ids):
        bv = pv[ids.long()] * sf[:, None] + res
        w = {}
        for n, (s, e, r, c) in slices.items(): w[n] = unblock(bv[s:e], (r, c))
        return w, bv[:]
    w_base, bv_base = decode(oi)

    # Forward to get h2, gate, up, z
    with torch.no_grad():
        ga = x_a @ w_base["gate_proj"].T
        ua = x_a @ w_base["up_proj"].T
        za = F.silu(ga) * ua
        h2_a = h1_a + za @ w_base["down_proj"].T

    # Load model
    print(json.dumps({"event": "load_model"}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()

    def forward_full(h2, toks):
        h = h2
        for i in range(1, 24):
            ly = model.model.layers[i]; slen = h.shape[0]
            p = torch.arange(slen).unsqueeze(0)
            m = torch.triu(torch.full((slen, slen), torch.finfo(h.dtype).min), 1).unsqueeze(0).unsqueeze(0)
            cs, si_ = model.model.rotary_emb(h.unsqueeze(0), p)
            h = h + ly.self_attn(ly.input_layernorm(h).unsqueeze(0), attention_mask=m, position_embeddings=(cs, si_))[0].squeeze(0)
            ha = h; h = ly.post_attention_layernorm(h); h = ha + ly.mlp(h)
        return model.lm_head(h)

    def compute_ce(wn):
        with torch.no_grad():
            gv = x_b @ wn["gate_proj"].T; uv = x_b @ wn["up_proj"].T
            h = h1_b + (F.silu(gv) * uv) @ wn["down_proj"].T
            logits = forward_full(h, tok_b)
        return float(F.cross_entropy(logits[:-1], tok_b[1:], reduction="mean"))

    # Teacher CE for tail weights
    print(json.dumps({"event": "teacher_ce"}), flush=True)
    with torch.no_grad():
        tlog = forward_full(h2t_a, tok_a)
    tce = F.cross_entropy(tlog[:-1], tok_a[1:], reduction="none")

    print(json.dumps({"event": "student_ce_grad"}), flush=True)
    h2_g = h2_a.detach().clone().requires_grad_(True)
    slog = forward_full(h2_g, tok_a)
    sce = F.cross_entropy(slog[:-1], tok_a[1:], reduction="none")
    ex = (sce.detach() - tce).clamp(min=0)
    ex = ex / max(float(ex.sum()), 1e-10)
    tw = torch.cat([ex ** TAIL_POWER, (ex ** TAIL_POWER)[-1:]], dim=0)
    tw = tw / tw.sum()
    wce = (tw[:-1] * sce).sum()
    g_h2 = torch.autograd.grad(wce, h2_g)[0]  # (512, 896)

    # Backprop through Down to get ∂L/∂z
    down = w_base["down_proj"]  # (896, 4864)
    g_z = g_h2 @ down  # (512, 4864)

    # Parse Gate/Up block dimensions
    gs = slices["gate_proj"]; us = slices["up_proj"]
    g_rows, g_cols = gs[2], gs[3]  # 4864, 896
    u_rows, u_cols = us[2], us[3]  # 4864, 896
    g_rg = g_rows // 4; g_cg = g_cols // 4  # 1216, 224
    u_rg = u_rows // 4; u_cg = u_cols // 4  # 1216, 224

    # Compute Gate gradient: ∂L/∂W_g = x^T @ (silu'(gate) * g_z * up) per block
    print(json.dumps({"event": "gate_gradient"}), flush=True)
    x_blocks = x_a.reshape(512, g_cg, 4)  # (512, 224, 4)
    ga_blocks = ga.reshape(512, g_rg, 4)   # (512, 1216, 4)
    sig = F.silu(ga_blocks)  # (512, 1216, 4)
    sig_p = sig * (1 - sig)  # silu derivative
    ua_blocks = ua.reshape(512, u_rg, 4)   # (512, 1216, 4)
    gz_blocks = g_z.reshape(512, g_rg, 4)  # (512, 1216, 4)
    # Gate grad factor: silu' * g_z * up  → (512, 1216, 4)
    gf = sig_p * gz_blocks * ua_blocks
    # ∂L/∂W_g[br, bc] = x_cols^T @ gf_rows
    g_grad = torch.einsum('tci,trj->crij', x_blocks, gf)  # (224, 1216, 4, 4)
    g_grad_flat = g_grad.reshape(g_cg * g_rg, 16).float().numpy()  # (272384, 16)

    # Compute Up gradient: ∂L/∂W_u = x^T @ (silu(gate) * g_z) per block
    print(json.dumps({"event": "up_gradient"}), flush=True)
    uf = sig * gz_blocks  # (512, 1216, 4)
    u_grad = torch.einsum('tci,trj->crij', x_blocks, uf)  # (224, 1216, 4, 4)
    u_grad_flat = u_grad.reshape(u_cg * u_rg, 16).float().numpy()  # (272384, 16)

    # Also compute grad norms for scoring
    g_norms = np.linalg.norm(g_grad_flat, axis=1) + 1e-10
    u_norms = np.linalg.norm(u_grad_flat, axis=1) + 1e-10
    g_dir = -g_grad_flat / g_norms[:, None]  # descent direction
    u_dir = -u_grad_flat / u_norms[:, None]

    print(json.dumps({"event": "grad_stats",
                       "gate_mean_norm": float(g_norms.mean()),
                       "gate_norm_90pct": float(np.percentile(g_norms, 90)),
                       "up_mean_norm": float(u_norms.mean()),
                       "up_norm_90pct": float(np.percentile(u_norms, 90))}), flush=True)

    # ── Sanity test: apply ETA_TEST to all blocks, check CE ──
    if ETA_TEST > 0:
        print(json.dumps({"event": "sanity_test", "eta": ETA_TEST}), flush=True)
        w_test = {n: w_base[n].clone() for n in w_base}
        # Gate correction
        g_corr = torch.from_numpy(-ETA_TEST * g_grad_flat).reshape(g_rg, g_cg, 4, 4)
        g_new = blockify(w_base["gate_proj"]) + g_corr.reshape(-1, 16)
        w_test["gate_proj"] = unblock(g_new, (g_rows, g_cols))
        # Up correction
        u_corr = torch.from_numpy(-ETA_TEST * u_grad_flat).reshape(u_rg, u_cg, 4, 4)
        u_new = blockify(w_base["up_proj"]) + u_corr.reshape(-1, 16)
        w_test["up_proj"] = unblock(u_new, (u_rows, u_cols))
        ce_base = compute_ce(w_base)
        ce_test = compute_ce(w_test)
        print(json.dumps({"event": "sanity_result", "CE_base": ce_base, "CE_test": ce_test}), flush=True)
        result = {"experiment": "FR1_gate_up_sanity", "eta": ETA_TEST,
                  "CE_base": ce_base, "CE_test": ce_test,
                  "improved": ce_test < ce_base}
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(json.dumps(result, indent=2))
        print(json.dumps({"event": "complete", **result}), flush=True)
        return

    # ── Build correction prototypes from descent directions ──
    n_fix_g = max(1, int(g_grad_flat.shape[0] * COVERAGE))
    n_fix_u = max(1, int(u_grad_flat.shape[0] * COVERAGE))

    # Cluster directions (only from blocks with non-negligible gradient)
    def build_protos(dirs, norms, k):
        strong = norms > np.percentile(norms, 30)
        sub = dirs[strong]
        n_sub = min(sub.shape[0], 30000)
        idx = np.random.RandomState(42).choice(sub.shape[0], n_sub, False)
        protos, _ = km(sub[idx], k, seed=42)
        return torch.from_numpy(protos).float()

    g_protos = build_protos(g_dir, g_norms, N_PROTOS)
    u_protos = build_protos(u_dir, u_norms, N_PROTOS)
    print(json.dumps({"event": "protos", "gate": g_protos.shape[0], "up": u_protos.shape[0]}))

    # Score blocks by best alignment × grad norm
    g_score = g_norms * (torch.from_numpy(g_dir).float() @ g_protos.T).max(dim=1).values.numpy()
    u_score = u_norms * (torch.from_numpy(u_dir).float() @ u_protos.T).max(dim=1).values.numpy()

    g_fix = np.argsort(g_score)[-n_fix_g:]
    u_fix = np.argsort(u_score)[-n_fix_u:]

    # ── Apply corrections: Up first, then Gate, then combined ──
    # For each fix block: α = ETA_TEST or use grad_norm based scale
    eta = ETA_TEST if ETA_TEST > 0 else 0.1
    w_up = {n: w_base[n].clone() for n in w_base}
    up_blocks = blockify(w_base["up_proj"]).clone()
    for fi, bi in enumerate(u_fix):
        step = eta * torch.from_numpy(u_dir[bi]) * u_norms[bi]
        up_blocks[bi] -= step
    w_up["up_proj"] = unblock(up_blocks, (u_rows, u_cols))

    ce_up = compute_ce(w_up)
    print(json.dumps({"event": "stage_up", "CE": ce_up}), flush=True)

    w_gate = {n: w_up[n].clone() for n in w_up}
    gate_blocks = blockify(w_up["gate_proj"]).clone()
    for fi, bi in enumerate(g_fix):
        step = eta * torch.from_numpy(g_dir[bi]) * g_norms[bi]
        gate_blocks[bi] -= step
    w_gate["gate_proj"] = unblock(gate_blocks, (g_rows, g_cols))

    ce_gate = compute_ce(w_gate)
    print(json.dumps({"event": "stage_gate", "CE": ce_gate}), flush=True)

    # Accept if final CE improves
    ce_base = compute_ce(w_base)
    best_ce = min(ce_base, ce_up, ce_gate)
    w_best = w_gate if ce_gate < ce_base else (w_up if ce_up < ce_base else w_base)

    print(json.dumps({"event": "final", "CE_base": ce_base, "CE_up": ce_up,
                       "CE_gate": ce_gate, "best": best_ce}), flush=True)

    # PPL evaluation if improved
    if best_ce < ce_base:
        print(json.dumps({"event": "evaluate_ppl"}), flush=True)
        token_sets = torch.load(
            ROOT.parent / "artifacts" / "functional_vq_l0" / "calibration_cache" / "fixed_token_ids.pt",
            map_location="cpu", weights_only=True)
        eval_tk = token_sets["evaluation"]; sl = 256
        mod = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True).eval()
        def set_mod(m, nm, val):
            t = m.model.layers[0].self_attn if nm[:4] in ("q_pr","k_pr","v_pr","o_pr") else m.model.layers[0].mlp
            getattr(t, nm).weight.copy_(val)
        with torch.no_grad():
            for nm in w_best: set_mod(mod, nm, w_best[nm])
        losses = []
        with torch.inference_mode():
            for start in range(0, eval_tk.numel() - sl + 1, sl):
                x = eval_tk[start:start+sl].unsqueeze(0)
                losses.append(float(mod(x, labels=x, use_cache=False).loss))
        ppl = math.exp(min(float(np.mean(losses)), 80))
        print(json.dumps({"event": "ppl", "ppl": ppl}), flush=True)
        result = {"experiment": "FR1_gate_up_residual", "accepted": True,
                  "CE_base": ce_base, "CE_best": best_ce,
                  "n_fix_gate": n_fix_g, "n_fix_up": n_fix_u,
                  "eta": eta, "ppl": ppl}
    else:
        result = {"experiment": "FR1_gate_up_residual", "accepted": False,
                  "CE_base": ce_base, "CE_up": ce_up, "CE_gate": ce_gate}

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
