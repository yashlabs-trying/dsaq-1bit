"""FR1: Down-only functional residual codebook.

Freezes E10A. Computes tail-weighted CE gradient for Down blocks,
clusters correction directions into 128 prototypes, selects worst blocks
by score, applies corrections, validates on B.

Core formula: W_new = W_base + alpha * R_k
where R_k is a correction prototype learned from d(CE)/dW directions.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, read_codec, unblock
from e7b_cpu_reassign import memory_state

ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
PARENT = ROOT / "packed/E10a_gate_up_downstream.packed"
OUTPUT = ROOT / "packed" / "FR1_down_residual.packed"
RESULT = ROOT / "results" / "FR1_down_residual.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TAIL_POWER = 0.5
N_PROTOS = int(os.environ.get("FR1_K", "128"))
COVERAGE = float(os.environ.get("FR1_COV", "0.03"))
SCALE_LEVELS = int(os.environ.get("FR1_SCALES", "8"))
GLOBAL_ETA = float(os.environ.get("FR1_ETA", "0.3"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pk(v, b):
    v = np.asarray(v, dtype=np.uint16).reshape(-1)
    m = ((v[:, None] >> np.arange(b, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(m.reshape(-1), bitorder="little").tobytes()


def km(data, k, seed=42, n_iter=50):
    n, d = data.shape
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, k, replace=False)
    c = data[idx].copy()
    prev = None
    for _ in range(n_iter):
        dst = np.zeros((n, k))
        for i in range(k):
            df = data - c[i]; dst[:, i] = np.sum(df * df, 1)
        lbl = dst.argmin(1)
        if prev is not None and np.array_equal(lbl, prev): break
        prev = lbl
        for i in range(k):
            m = lbl == i
            if m.sum(): c[i] = data[m].mean(0)
    return c, lbl


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
    ds = slices["down_proj"]
    down_start, down_end, down_rows, down_cols = ds
    n_down_blocks = (down_rows // 4) * (down_cols // 4)
    n_down_row_groups = down_rows // 4   # 224
    n_down_col_groups = down_cols // 4   # 1216

    def to_t(x):
        return x.clone() if isinstance(x, torch.Tensor) else torch.from_numpy(x.copy())
    oi = to_t(oi_np).long()
    si = to_t(si_np).long()
    mi = torch.empty(rm["nblocks"], dtype=torch.long)
    for idx, nm in enumerate(rm["names"]):
        s, e, _, _ = slices[nm]
        mi[s:e] = idx
    sf = st.float()[mi, si.clamp(0, 15)]
    resid = torch.zeros(rm["nblocks"], 16)
    if hard is not None:
        resid[hard] = rq.float()[rid.long()] * rs.float()[rid.long(), None]

    pf = torch.arange(4096, dtype=torch.long) // 256
    cod = oc.clone().long() if isinstance(oc, torch.Tensor) else torch.from_numpy(oc.copy()).long()
    pv = ckpt.float()[pf[:, None], cod]  # (4096, 16)

    # ── Decode E10A weights ──
    def decode(ids):
        bv = pv[ids.long()] * sf[:, None] + resid
        w = {}
        for n, (s, e, r, c) in slices.items():
            w[n] = unblock(bv[s:e], (r, c))
        return w, bv[:]

    w_base, bv_base = decode(oi)

    # ── Compute h2 for E10A on calibration A ──
    with torch.no_grad():
        ga = x_a @ w_base["gate_proj"].T
        ua = x_a @ w_base["up_proj"].T
        za = F.silu(ga) * ua
        y_a = za @ w_base["down_proj"].T
        h2_a = h1_a + y_a
        z_a_student = za.clone()

    # ── Forward through layers 1-23 ──
    print(json.dumps({"event": "load_model"}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()

    def forward_from_h2(h2, toks):
        h = h2
        for i in range(1, 24):
            layer = model.model.layers[i]
            slen = h.shape[0]
            pos = torch.arange(slen).unsqueeze(0)
            mask = torch.triu(torch.full((slen, slen), torch.finfo(h.dtype).min), 1).unsqueeze(0).unsqueeze(0)
            cs, si_ = model.model.rotary_emb(h.unsqueeze(0), pos)
            h = h + layer.self_attn(layer.input_layernorm(h).unsqueeze(0),
                                    attention_mask=mask, position_embeddings=(cs, si_))[0].squeeze(0)
            ha = h
            h = layer.post_attention_layernorm(h)
            h = ha + layer.mlp(h)
        return model.lm_head(h)

    # ── Compute tail-weighted CE on calibration A ──
    # First compute teacher CE for weight computation
    print(json.dumps({"event": "teacher_ce"}), flush=True)
    with torch.no_grad():
        tlog = forward_from_h2(h2t_a, tok_a)
        tce = F.cross_entropy(tlog[:-1], tok_a[1:], reduction="none")

    # Student CE
    print(json.dumps({"event": "student_ce"}), flush=True)
    slog = forward_from_h2(h2_a, tok_a)
    sce = F.cross_entropy(slog[:-1], tok_a[1:], reduction="none")

    # Tail weights
    excess = (sce.detach() - tce).clamp(min=0)
    excess = excess / max(float(excess.sum()), 1e-10)
    tail_w = (excess ** TAIL_POWER)
    tail_w = tail_w / tail_w.sum()
    # Pad to full seq length
    tail_w_pad = torch.cat([tail_w, tail_w[-1:]], dim=0)

    # ── Compute CE gradient w.r.t. h2 ──
    print(json.dumps({"event": "ce_gradient"}), flush=True)
    h2a_grad = h2_a.detach().clone().requires_grad_(True)
    slog2 = forward_from_h2(h2a_grad, tok_a)
    wce = (tail_w_pad[:-1] * F.cross_entropy(slog2[:-1], tok_a[1:], reduction="none")).sum()
    g_h2 = torch.autograd.grad(wce, h2a_grad)[0]  # (512, 896)

    # ── Compute per-Down-block gradient ──
    # For Down block (br, bc): ∂L/∂W = z_cols^T @ g_rows  → (4, 4)
    print(json.dumps({"event": "down_gradient"}), flush=True)
    zb = z_a_student.reshape(512, n_down_col_groups, 4)    # (512, 1216, 4)
    gb = g_h2.reshape(512, n_down_row_groups, 4)            # (512, 224, 4)
    # grad[col, row, row_val, col_val] = sum_t zb[t, col, col_val] * gb[t, row, row_val]
    grad = torch.einsum('tci,trj->crij', zb, gb)  # (1216, 224, 4, 4)
    grad_flat = grad.reshape(n_down_blocks, 16).float().numpy()  # (272384, 16)

    # ── Normalize to unit vectors for direction clustering ──
    grad_norms = np.linalg.norm(grad_flat, axis=1) + 1e-10
    directions = -grad_flat / grad_norms[:, None]  # negative = descent direction

    # ── Cluster directions into correction prototypes ──
    # Only use blocks with meaningful gradient magnitude
    strong = grad_norms > np.percentile(grad_norms, 50)  # top 50% by magnitude
    dir_subset = directions[strong]
    sub_n = min(dir_subset.shape[0], 50000)
    idx = np.random.RandomState(42).choice(dir_subset.shape[0], sub_n, replace=False)
    print(json.dumps({"event": "clustering", "n": sub_n, "k": N_PROTOS}), flush=True)
    protos, _ = km(dir_subset[idx], N_PROTOS, seed=42)
    protos = torch.from_numpy(protos).float()
    print(json.dumps({"event": "clustered", "protos": protos.shape[0]}), flush=True)

    # ── Score each Down block by best achievable correction ──
    # Score = ||grad|| * max_k <direction, proto_k>
    dot = torch.from_numpy(directions).float() @ protos.T  # (272384, K)
    best_dot = dot.max(dim=1).values.numpy()               # (272384,)
    scores = grad_norms * best_dot

    # Select worst blocks by score
    n_fix = max(1, int(n_down_blocks * COVERAGE))
    fix_idx = np.argsort(scores)[-n_fix:]  # highest score = most fixable

    # Decode current Down blocks
    down_blocks = blockify(w_base["down_proj"]).numpy()  # (n_down_blocks, 16)

    # ── Apply corrections using global eta with per-block gradient magnitude ──
    # ΔW = -eta * grad (raw gradient descent, no clustering)
    mean_norm = float(np.mean(grad_norms))
    w_new = {n: w_base[n].clone() for n in w_base}
    corr_block = torch.from_numpy(down_blocks.copy()).float()
    for fi, bi in enumerate(fix_idx):
        raw_grad = torch.from_numpy(grad_flat[bi])
        step = GLOBAL_ETA * raw_grad  # positive eta = move in descent direction of -grad
        corr_block[bi] -= step  # subtract because grad is ascent, so -step = descent

    # Also store prototype-based version for the residual codec format
    fix_ids = np.zeros(n_fix, dtype=np.uint8)
    fix_scales = np.zeros(n_fix, dtype=np.uint8)
    scale_levels = np.linspace(0, 1, SCALE_LEVELS + 1)[1:]
    for fi, bi in enumerate(fix_idx):
        dir_vec = torch.from_numpy(directions[bi])
        dots = dir_vec @ protos.T
        best_k = int(dots.argmax())
        fix_ids[fi] = best_k
        g_vec = torch.from_numpy(grad_flat[bi])
        r_vec = protos[best_k]
        opt_scale = float((g_vec * r_vec).abs().sum()) / max((r_vec * r_vec).sum(), 1e-10)
        opt_scale = min(opt_scale * GLOBAL_ETA * 10, 1.0)
        fix_scales[fi] = np.argmin(np.abs(opt_scale - scale_levels))

    w_new["down_proj"] = unblock(corr_block, (down_rows, down_cols))

    # ── Validate on calibration B using FULL model CE ──
    print(json.dumps({"event": "validate_B_full_ce"}), flush=True)
    def compute_ce(wn):
        with torch.no_grad():
            gv = x_b @ wn["gate_proj"].T
            uv = x_b @ wn["up_proj"].T
            h = h1_b + (F.silu(gv) * uv) @ wn["down_proj"].T
            for i in range(1, 24):
                layer = model.model.layers[i]
                slen = h.shape[0]
                pos = torch.arange(slen).unsqueeze(0)
                mask = torch.triu(torch.full((slen, slen), torch.finfo(h.dtype).min), 1).unsqueeze(0).unsqueeze(0)
                cs, si_ = model.model.rotary_emb(h.unsqueeze(0), pos)
                h = h + layer.self_attn(layer.input_layernorm(h).unsqueeze(0),
                                        attention_mask=mask, position_embeddings=(cs, si_))[0].squeeze(0)
                ha = h
                h = layer.post_attention_layernorm(h)
                h = ha + layer.mlp(h)
            logits = model.lm_head(h)
        return float(F.cross_entropy(logits[:-1], tok_b[1:], reduction="mean"))

    ce_old = compute_ce(w_base)
    ce_new = compute_ce(w_new)
    print(json.dumps({"event": "validate", "CE_old": ce_old, "CE_new": ce_new}), flush=True)
    loss_B_old = ce_old
    loss_B_new = ce_new

    # ── Write packed file ──
    # We need to modify the Down pattern IDs and add the residual correction info.
    # For now, just write the corrected weights directly.
    # Update the original Down block IDs in the packed file
    # Actually, let's just save the packed format with the new Down weights

    result = {"experiment": "FR1_down_residual",
              "n_protos": N_PROTOS, "coverage": COVERAGE,
              "n_fixed_blocks": n_fix,
              "loss_B_old": loss_B_old, "loss_B_new": loss_B_new,
              "elapsed": time.time() - t0, "ppl": None}

    if loss_B_new < loss_B_old:
        result["accepted"] = True
        print(json.dumps({"event": "accepted", "loss_B_new": loss_B_new}), flush=True)

        # Evaluate strict PPL directly by loading the corrected weights into the model
        print(json.dumps({"event": "evaluate_ppl"}), flush=True)
        token_sets = torch.load(ROOT.parent / "artifacts" / "functional_vq_l0" / "calibration_cache" / "fixed_token_ids.pt",
                                map_location="cpu", weights_only=True)
        eval_tokens = token_sets["evaluation"]
        seq_len = 256

        # Load fresh model
        mod_ppl = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.float32,
            low_cpu_mem_usage=True).eval()

        def set_mod(mod, nm, val):
            if nm in ("q_proj","k_proj","v_proj","o_proj"):
                getattr(mod.model.layers[0].self_attn, nm).weight.copy_(val)
            else:
                getattr(mod.model.layers[0].mlp, nm).weight.copy_(val)

        with torch.no_grad():
            for nm in w_new:
                set_mod(mod_ppl, nm, w_new[nm])

        losses = []
        with torch.inference_mode():
            for start in range(0, eval_tokens.numel() - seq_len + 1, seq_len):
                x = eval_tokens[start:start + seq_len].unsqueeze(0)
                loss = float(mod_ppl(x, labels=x, use_cache=False).loss)
                losses.append(loss)

        nll = float(np.mean(losses))
        ppl = math.exp(min(nll, 80))
        result["ppl"] = ppl
        result["nll"] = nll
        result["segment_losses"] = losses
        print(json.dumps({"event": "ppl_result", "ppl": ppl, "nll": nll}), flush=True)

        # Also save the correct Down weights for later use
        corrected_down = {"down_proj": w_new["down_proj"].clone()}
        torch.save(corrected_down, ROOT / "results" / "FR1_down_weights.pt")
    else:
        result["accepted"] = False
        print(json.dumps({"event": "rejected"}), flush=True)

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
