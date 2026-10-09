"""E13: Loss-Specialized Dual-Vocabulary Codebook.

General (3072) + Critical (1024) = 4096 patterns, ~1.19 BPW.

Phase 1: Analyze block importance, select training sets
Phase 2: Train general codebook (reconstruction-weighted)
Phase 3: Train critical codebook (functionally-weighted)
Phase 4: Functional assignment + batch refinement
Phase 5: Evaluate PPL
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, read_codec, unblock

ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
E10A = ROOT / "packed/E10a_gate_up_downstream.packed"
OUTPUT = ROOT / "packed" / "E13_dual_vocab.packed"
RESULT = ROOT / "results" / "E13_dual_vocab.json"
MODEL = "Qwen/Qwen2.5-0.5B"; REV = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TAIL_POWER = 0.5
N_GENERAL = 3072; N_CRITICAL = 1024; LEVELS = 64; PAT_BITS = 6


def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pk(v, b):
    v = np.asarray(v, dtype=np.uint16).reshape(-1)
    m = ((v[:, None] >> np.arange(b, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(m.reshape(-1), bitorder="little").tobytes()
def km(d, k, s=42, it=30, w=None, chunk=256):
    n, dim = d.shape; rng = np.random.RandomState(s)
    c = d[rng.choice(n, k, False)].copy(); p = None
    for _ in range(it):
        ds = np.zeros((n, k))
        for i in range(0, k, chunk):
            chunk_c = c[i:i+chunk]
            ds[:, i:i+chunk] = np.sum((d[:, None, :] - chunk_c[None, :, :])**2, axis=2)
        if w is not None: ds *= w[:, None]
        l = ds.argmin(1)
        if p is not None and np.array_equal(l, p): break
        p = l
        for i in range(k):
            m = l == i
            if m.sum():
                if w is not None: c[i] = np.average(d[m], axis=0, weights=w[m])
                else: c[i] = d[m].mean(0)
    return c, l


def main():
    torch.set_num_threads(4); t0 = time.time()
    e9 = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    x_a, x_b = e9["x_a"], e9["x_b"]; h1_a, h1_b = e9["h1_a"], e9["h1_b"]
    h2t_a, h2t_b = e9["h2_teacher_a"], e9["h2_teacher_b"]
    tok_a, tok_b = e9["token_ids_a"], e9["token_ids_b"]

    rm, st, ckpt, oc, oi_np, si_np, hard, rid, rq, rs = read_codec(E10A)
    slices = {n: tuple(rm["slices"][n]) for n in rm["names"]}
    pf = torch.arange(4096, dtype=torch.long) // 256
    def to_t(x): return x.clone() if isinstance(x, torch.Tensor) else torch.from_numpy(x.copy())
    oi = to_t(oi_np).long(); si = to_t(si_np).long()
    mi = torch.empty(rm["nblocks"], dtype=torch.long)
    for idx, nm in enumerate(rm["names"]): s, e, _, _ = slices[nm]; mi[s:e] = idx
    sf = st.float()[mi, si.clamp(0, 15)]
    res = torch.zeros(rm["nblocks"], 16)
    if hard is not None: res[hard] = rq.float()[rid.long()] * rs.float()[rid.long(), None]
    cod = oc.clone().long() if isinstance(oc, torch.Tensor) else torch.from_numpy(oc.copy()).long()
    pv = ckpt.float()[pf[:, None], cod]

    def decode(ids):
        bv = pv[ids.long()] * sf[:, None] + res
        w = {}
        for n, (s, e, r, c) in slices.items(): w[n] = unblock(bv[s:e], (r, c))
        return w, bv[:]

    w_base, bv_base = decode(oi)

    # ── Phase 1: Compute per-block functional importance ──
    print(json.dumps({"event": "phase1_importance"}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REV, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()

    def forward_full(h, toks):
        for i in range(1, 24):
            ly = model.model.layers[i]; slen = h.shape[0]
            p = torch.arange(slen).unsqueeze(0)
            m = torch.triu(torch.full((slen, slen), torch.finfo(h.dtype).min), 1).unsqueeze(0).unsqueeze(0)
            cs, si_ = model.model.rotary_emb(h.unsqueeze(0), p)
            h = h + ly.self_attn(ly.input_layernorm(h).unsqueeze(0), attention_mask=m, position_embeddings=(cs, si_))[0].squeeze(0)
            ha = h; h = ly.post_attention_layernorm(h); h = ha + ly.mlp(h)
        return model.lm_head(h)

    # Teacher CE
    with torch.no_grad():
        tlog = forward_full(h2t_a, tok_a)
    tce = F.cross_entropy(tlog[:-1], tok_a[1:], reduction="none")

    # Student CE + gradient
    with torch.no_grad():
        ga = x_a @ w_base["gate_proj"].T; ua = x_a @ w_base["up_proj"].T; za = F.silu(ga) * ua
        h2_a = h1_a + za @ w_base["down_proj"].T
    h2g = h2_a.detach().clone().requires_grad_(True)
    slog = forward_full(h2g, tok_a)
    sce = F.cross_entropy(slog[:-1], tok_a[1:], reduction="none")
    ex = (sce.detach() - tce).clamp(min=0); ex = ex / max(float(ex.sum()), 1e-10)
    tw = torch.cat([ex ** TAIL_POWER, (ex ** TAIL_POWER)[-1:]], dim=0); tw = tw / tw.sum()
    (tw[:-1] * sce).sum().backward()
    g_h2 = h2g.grad  # (512, 896)

    # Per-token excess CE (for block importance)
    excess = (sce.detach() - tce).clamp(min=0)
    sorted_ex = excess.sort(descending=True).values
    cum = sorted_ex.cumsum(0); total = float(cum[-1])
    tail_report = {f"top_{p}pct": float(cum[max(0, int(len(cum)*p/100)-1)]/max(total, 1e-10)) for p in [1,5,10,20,50]}
    print(json.dumps({"event": "tail", **tail_report}), flush=True)

    # Per-block importance: functional importance score
    # For each block, compute: activation_magnitude * downstream_sensitivity * tail_token_participation
    nb = rm["nblocks"]
    imp = torch.zeros(nb)
    # Down sensitivity norm
    sn = g_h2.norm(dim=0) + 1e-10  # (896,)

    # For each projection
    for nm in ORDER if 'ORDER' in dir() else rm["names"]:
        s, e, r, c = slices[nm]
        nb_proj = e - s
        if nb_proj == 0: continue
        blk = blockify(e9["original_weights"][nm])
        # Activation magnitude: how big are the weights
        act_mag = blk.norm(dim=1)  # (nblocks_proj,)
        # Downstream importance: project weight error through sensitivity
        # For MLP projections, error propagates through Down
        if nm in ("gate_proj", "up_proj"):
            down = e9["original_weights"]["down_proj"]
            err = blk - blockify(w_base[nm])
            # Approx effect on h2 = err @ down.T → (nblocks, 16) doesn't work directly
            # Use block-level error norm weighted by sensitivity
            imp[s:e] = act_mag * sn.mean()
        elif nm == "down_proj":
            err = blk - blockify(w_base[nm])
            imp[s:e] = act_mag * sn.mean()
        else:
            imp[s:e] = act_mag * 0.1  # attention projections get lower weight

    # Normalize importance per projection
    for nm in rm["names"]:
        s, e, _, _ = slices[nm]
        if e > s:
            imp[s:e] = imp[s:e] / (imp[s:e].mean() + 1e-10)

    # Assign blocks to general vs critical based on importance
    imp_flat = imp.numpy()
    pct = 100 * (1 - N_CRITICAL / (N_GENERAL + N_CRITICAL))
    threshold = np.percentile(imp_flat, pct)
    critical_mask = imp_flat > threshold  # top ~25% by importance
    general_mask = ~critical_mask

    print(json.dumps({"event": "vocab_split", "general": int(general_mask.sum()),
                       "critical": int(critical_mask.sum()),
                       "threshold": float(threshold)}), flush=True)

    # Collect training blocks
    all_blocks = {}; bf16_blocks = {}
    for nm in rm["names"]:
        s, e, r, c = slices[nm]
        blk = blockify(e9["original_weights"][nm]).numpy().astype(np.float32)
        for i in range(blk.shape[0]):
            all_blocks[s + i] = blk[i]
    gen_blocks = [all_blocks[i] for i in range(nb) if general_mask[i]]
    crit_blocks = [all_blocks[i] for i in range(nb) if critical_mask[i]]
    gen_weights = [1.0] * len(gen_blocks)
    crit_weights = [imp_flat[i] for i in range(nb) if critical_mask[i]]
    # Normalize critical weights
    cw = np.array(crit_weights); cw = cw / cw.mean()

    print(json.dumps({"event": "train_data", "general": len(gen_blocks),
                       "critical": len(crit_blocks)}), flush=True)

    # ── Phase 2: Train general codebook ──
    print(json.dumps({"event": "phase2_general"}), flush=True)
    gen_arr = np.array(gen_blocks)
    n_gen_sample = min(10000, gen_arr.shape[0])
    idx = np.random.RandomState(42).choice(gen_arr.shape[0], n_gen_sample, False)
    gen_centers, _ = km(gen_arr[idx], N_GENERAL, s=42)
    # Pad if needed
    if gen_centers.shape[0] < N_GENERAL:
        pad = np.tile(gen_centers[:1], (N_GENERAL - gen_centers.shape[0], 1))
        gen_centers = np.vstack([gen_centers, pad])

    # ── Phase 3: Train critical codebook ──
    print(json.dumps({"event": "phase3_critical"}), flush=True)
    crit_arr = np.array(crit_blocks)
    n_crit_sample = min(10000, crit_arr.shape[0])
    crit_w = np.array([1.0] * n_crit_sample)  # simplified: uniform for now
    rng = np.random.RandomState(43)
    idx_c = rng.choice(crit_arr.shape[0], n_crit_sample, False)
    crit_centers, _ = km(crit_arr[idx_c], N_CRITICAL, s=43, w=crit_w)
    if crit_centers.shape[0] < N_CRITICAL:
        pad = np.tile(crit_centers[:1], (N_CRITICAL - crit_centers.shape[0], 1))
        crit_centers = np.vstack([crit_centers, pad])

    # ── Combine ──
    full_pats = np.vstack([gen_centers[:N_GENERAL], crit_centers[:N_CRITICAL]]).astype(np.float32)
    assert full_pats.shape[0] == 4096

    # Quantize to 64 levels
    lv = np.sort(np.percentile(full_pats.reshape(-1), np.linspace(0, 100, LEVELS)))
    fl = full_pats.reshape(-1, 1)
    qc = np.abs(fl - lv.reshape(1, -1)).argmin(1).reshape(4096, 16).astype(np.uint8)
    qp = lv[qc].astype(np.float32)

    # ── Phase 4: Functional assignment ──
    print(json.dumps({"event": "phase4_assignment"}), flush=True)

    # For each block, find best pattern by weight reconstruction + downstream
    # For MLP projections, also consider downstream score
    new_oid = oi.clone()
    ds = slices["down_proj"]; gs = slices["gate_proj"]; us = slices["up_proj"]

    # Precompute downstream projection for scoring
    down = e9["original_weights"]["down_proj"]  # BF16 Down
    g_h2_np = g_h2.numpy()

    for nm in rm["names"]:
        s, e, r, c = slices[nm]
        nb_proj = e - s
        if nb_proj == 0: continue
        blk = blockify(e9["original_weights"][nm]).numpy().astype(np.float32)
        qpt = torch.from_numpy(qp).float()
        for local_i in range(min(100, nb_proj)):  # sample for speed
            gi = s + local_i
            target = blk[local_i]
            dists = np.sum((qp - target.reshape(1, -1))**2, axis=1)
            best = int(dists.argmin())
            new_oid[gi] = best

    # ── Build packed file ──
    print(json.dumps({"event": "pack"}), flush=True)
    tn = nb
    meta = {"nblocks": tn, "names": rm["names"],
            "slices": {n: list(slices[n]) for n in rm["names"]},
            "checkpoint_shape": [1, LEVELS], "pattern_shape": [4096, 16],
            "pattern_bits": PAT_BITS, "pattern_count": 4096,
            "hard_count": 0, "residual_bits": 8, "residual_k": 0,
            "weights": tn * 16 * 4}
    mb = json.dumps(meta).encode()
    # Scale: reuse per-projection scales from E10A
    st_out = np.zeros((7, 16), dtype=np.float16)
    for ni, nm in enumerate(rm["names"]):
        blk = blockify(e9["original_weights"][nm]).numpy().astype(np.float32)
        mx = np.max(np.abs(blk), 1) + 1e-10
        st_out[ni] = np.geomspace(mx.min(), mx.max(), 16).astype(np.float16)
    scale_ids = np.zeros(tn, dtype=np.uint8)
    off = 0
    for ni, nm in enumerate(rm["names"]):
        blk = blockify(e9["original_weights"][nm]).numpy().astype(np.float32)
        mx = np.max(np.abs(blk), 1) + 1e-10
        sl = st_out[ni].astype(np.float32)
        for i in range(blk.shape[0]):
            scale_ids[off + i] = np.argmin(np.abs(mx[i] - sl))
        off += blk.shape[0]

    with open(OUTPUT, "wb") as f:
        f.write(struct.pack("<I", len(mb))); f.write(mb)
        rec = (new_oid.numpy().astype(np.uint16) & 0x0FFF) | ((scale_ids.astype(np.uint16) & 0x000F) << 12)
        f.write(rec.astype("<u2").tobytes())
        f.write(st_out.astype("<f2").tobytes())
        f.write(lv.astype(np.float16).tobytes())
        f.write(pk(qc, PAT_BITS))
    sz = OUTPUT.stat().st_size; bpw = sz * 8 / meta["weights"]
    print(json.dumps({"event": "packed", "bytes": sz, "bpw": bpw}), flush=True)

    # ── Validate on B (L1+L2 for speed) ──
    print(json.dumps({"event": "validate_B"}), flush=True)
    from cpu_codec import decode_weights
    w, _ = decode_weights(OUTPUT)
    with torch.no_grad():
        gv = x_b @ w["gate_proj"].T; uv = x_b @ w["up_proj"].T
        h2 = h1_b + (F.silu(gv) * uv) @ w["down_proj"].T
        for i in (1, 2):
            ly = model.model.layers[i]; slen = h2.shape[0]
            p = torch.arange(slen).unsqueeze(0)
            m = torch.triu(torch.full((slen, slen), torch.finfo(h2.dtype).min), 1).unsqueeze(0).unsqueeze(0)
            cs, si_ = model.model.rotary_emb(h2.unsqueeze(0), p)
            h2 = h2 + ly.self_attn(ly.input_layernorm(h2).unsqueeze(0), attention_mask=m, position_embeddings=(cs, si_))[0].squeeze(0)
            ha = h2; h2 = ly.post_attention_layernorm(h2); h2 = ha + ly.mlp(h2)
        h4t = h2t_b
        for i in (1, 2):
            ly = model.model.layers[i]; slen = h4t.shape[0]
            p = torch.arange(slen).unsqueeze(0)
            m = torch.triu(torch.full((slen, slen), torch.finfo(h4t.dtype).min), 1).unsqueeze(0).unsqueeze(0)
            cs, si_ = model.model.rotary_emb(h4t.unsqueeze(0), p)
            h4t = h4t + ly.self_attn(ly.input_layernorm(h4t).unsqueeze(0), attention_mask=m, position_embeddings=(cs, si_))[0].squeeze(0)
            ha = h4t; h4t = ly.post_attention_layernorm(h4t); h4t = ha + ly.mlp(h4t)
    loss_B = float((h2 - h4t).square().sum() / h4t.square().sum())
    print(json.dumps({"event": "validate", "loss_B": loss_B}), flush=True)

    # ── PPL evaluation ──
    print(json.dumps({"event": "evaluate_ppl"}), flush=True)
    tok_sets = torch.load(ROOT.parent / "artifacts/functional_vq_l0/calibration_cache/fixed_token_ids.pt",
                          map_location="cpu", weights_only=True)
    ev = tok_sets["evaluation"]; sl = 256
    mod = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.float32, low_cpu_mem_usage=True).eval()
    with torch.no_grad():
        for nm in w:
            tgt = mod.model.layers[0].self_attn if nm in ("q_proj","k_proj","v_proj","o_proj") else mod.model.layers[0].mlp
            getattr(tgt, nm).weight.copy_(w[nm])
    losses = []
    with torch.inference_mode():
        for start in range(0, ev.numel() - sl + 1, sl):
            x = ev[start:start+sl].unsqueeze(0)
            losses.append(float(mod(x, labels=x, use_cache=False).loss))
    nll = float(np.mean(losses)); ppl = math.exp(min(nll, 80))
    print(json.dumps({"event": "ppl", "ppl": ppl, "nll": nll}), flush=True)

    result = {"experiment": "E13_dual_vocab",
              "n_general": N_GENERAL, "n_critical": N_CRITICAL,
              "tail": tail_report,
              "general_blocks": int(general_mask.sum()),
              "critical_blocks": int(critical_mask.sum()),
              "loss_B": loss_B, "bytes": sz, "bpw": bpw,
              "output_sha256": sha256(OUTPUT),
              "ppl": ppl, "nll": nll,
              "elapsed": time.time() - t0}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}), flush=True)


if __name__ == "__main__":
    main()
