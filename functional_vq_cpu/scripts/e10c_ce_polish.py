"""E10C: CE-gradient-aware + tail-weighted Gate/Up/Down reassignment.

Uses g = d(CE)/d(h2) (full-model cross-entropy gradient) instead of L1+L2 NMSE.

Tail weighting: score_t = w_t * (g * e_t + g * dh_t)^2 where
  w_t ~ max(0, CE_s(t) - CE_t(t))^{0.5}
focuses optimization on tokens producing the most excess NLL.

Round-robin: Gate -> Up -> Down -> Gate -> Up -> Down (2 cycles, recompute g after each).

Scorer architecture supports future upgrade to M = sum w_t g_t g_t^T when GPU available.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, codec_state, decode_weights, patch_ids, unblock
from e7b_cpu_reassign import candidates, memory_state

ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
PARENT = Path(os.environ.get("E10C_PARENT", ROOT / "packed/E10a_gate_up_downstream.packed"))
COVERAGE = float(os.environ.get("E10C_COVERAGE", "0.05"))
LABEL = os.environ.get("E10C_LABEL", "E10c_ce_tail_polish")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
STATE = ROOT / "checkpoints" / f"{LABEL}_state.pt"
CHECKPOINT_ROWS = int(os.environ.get("E10C_CHECKPOINT_ROWS", "2"))
THREADS = int(os.environ.get("E10C_THREADS", "4"))
MIN_AVAILABLE_GB = float(os.environ.get("E10C_MIN_AVAILABLE_GB", "1.0"))
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TAIL_POWER = 0.5
TAIL_GLOBAL_BLEND = 0.7


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {name: unblock(blocks[s:e], (r, c)) for name, (s, e, r, c) in codec["slices"].items()}


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


# ── Scorer: tail-weighted CE gradient ──

class CEScorer:
    """Scores using g = ∂CE/∂h2 with per-token tail weighting.

    score = Σ_t w_t (g_t · e_t + g_t · Δh_t)² / den

    Where g_t is precomputed, w_t weights tail tokens.
    Supports upgrade to per-token g_t g_t^T subspace.
    """
    def __init__(self, g_a: torch.Tensor, g_b: torch.Tensor,
                 tail_weights_a: torch.Tensor, tail_weights_b: torch.Tensor,
                 den_a: float, den_b: float,
                 down_weight: torch.Tensor):
        self.g_a = g_a * tail_weights_a.sqrt().unsqueeze(1)
        self.g_b = g_b * tail_weights_b.sqrt().unsqueeze(1)
        self.den_a = den_a
        self.den_b = den_b
        self.down = down_weight
        self._precomp = {}

    def precompute_down_projection(self):
        self._precomp["g_down_a"] = self.g_a @ self.down  # (512, 4864)
        self._precomp["g_down_b"] = self.g_b @ self.down

    def score_gate_block(self, z_a, z_b, gate_a, gate_b, up_a, up_b,
                         row_slice, new_block, old_block, x_a, x_b,
                         col_slice, projet_a, projet_b):
        delta_gate_a = x_a[:, col_slice] @ (new_block - old_block).T
        delta_gate_b = x_b[:, col_slice] @ (new_block - old_block).T
        nz_a = F.silu(gate_a[:, row_slice] + delta_gate_a) * up_a[:, row_slice]
        nz_b = F.silu(gate_b[:, row_slice] + delta_gate_b) * up_b[:, row_slice]
        dz_a = nz_a - z_a[:, row_slice]
        dz_b = nz_b - z_b[:, row_slice]
        g_down_a = self._precomp["g_down_a"][:, row_slice]
        g_down_b = self._precomp["g_down_b"][:, row_slice]
        pd_a = (g_down_a * dz_a).sum(1)
        pd_b = (g_down_b * dz_b).sum(1)
        np_a = projet_a + pd_a
        np_b = projet_b + pd_b
        sa = float(np_a.square().sum())
        sb = float(np_b.square().sum())
        return sa, sb, dz_a, dz_b, pd_a, pd_b, nz_a, nz_b

    def score_up_block(self, z_a, z_b, gate_a, gate_b, up_a, up_b,
                       row_slice, new_block, old_block, x_a, x_b,
                       col_slice, projet_a, projet_b):
        delta_up_a = x_a[:, col_slice] @ (new_block - old_block).T
        delta_up_b = x_b[:, col_slice] @ (new_block - old_block).T
        nz_a = F.silu(gate_a[:, row_slice]) * (up_a[:, row_slice] + delta_up_a)
        nz_b = F.silu(gate_b[:, row_slice]) * (up_b[:, row_slice] + delta_up_b)
        dz_a = nz_a - z_a[:, row_slice]
        dz_b = nz_b - z_b[:, row_slice]
        g_down_a = self._precomp["g_down_a"][:, row_slice]
        g_down_b = self._precomp["g_down_b"][:, row_slice]
        pd_a = (g_down_a * dz_a).sum(1)
        pd_b = (g_down_b * dz_b).sum(1)
        np_a = projet_a + pd_a
        np_b = projet_b + pd_b
        sa = float(np_a.square().sum())
        sb = float(np_b.square().sum())
        return sa, sb, dz_a, dz_b, pd_a, pd_b, nz_a, nz_b

    def score_down_block(self, z_a, z_b, row_slice, col_slice,
                         new_block, old_block, h2_a, h2_b,
                         h2t_a, h2t_b, projet_a, projet_b):
        delta_a = z_a[:, col_slice] @ (new_block - old_block).T
        delta_b = z_b[:, col_slice] @ (new_block - old_block).T
        pd_a = (self.g_a[:, row_slice] * delta_a).sum(1)
        pd_b = (self.g_b[:, row_slice] * delta_b).sum(1)
        np_a = projet_a + pd_a
        np_b = projet_b + pd_b
        sa = float(np_a.square().sum())
        sb = float(np_b.square().sum())
        return sa, sb, delta_a, delta_b, pd_a, pd_b

    def compute_delta_h2(self, dz_a, dz_b, rows):
        dh_a = dz_a @ self.down[:, rows].T
        dh_b = dz_b @ self.down[:, rows].T
        return dh_a, dh_b


# ── CE gradient computation ──

def compute_ce_gradient(model, h2_a, h2_b, token_ids_a, token_ids_b):
    """Compute g = ∂(Σ w_t CE_t)/∂h2 with hard-tail weighting."""
    def run_forward(h2_input):
        h = h2_input
        for i in range(1, 24):
            layer = model.model.layers[i]
            slen = h.shape[0]
            pos_ids = torch.arange(slen, device=h.device).unsqueeze(0)
            causal_mask = torch.full((slen, slen), torch.finfo(h.dtype).min, device=h.device)
            causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
            cos, sin = model.model.rotary_emb(h.unsqueeze(0), pos_ids)
            h_norm = layer.input_layernorm(h)
            attn = layer.self_attn(h_norm.unsqueeze(0), attention_mask=causal_mask,
                                   position_embeddings=(cos, sin))[0].squeeze(0)
            h = h + attn
            h_attn = h
            h = layer.post_attention_layernorm(h)
            h = h_attn + layer.mlp(h)
        return model.lm_head(h)

    def get_g(h2, toks):
        h = h2.detach().clone().requires_grad_(True)
        logits = run_forward(h)
        ce = F.cross_entropy(logits[:-1], toks[1:], reduction="none")
        w = ce.detach()  # weight by tail power
        w = w ** TAIL_POWER
        w = w / w.sum()
        loss = (w * ce).sum()
        g = torch.autograd.grad(loss, h)[0]
        return g, ce

    ga, ce_a = get_g(h2_a, token_ids_a)
    gb, ce_b = get_g(h2_b, token_ids_b)
    # Pad CE: add dummy for last position (no next-token prediction)
    ce_a = torch.cat([ce_a, ce_a[-1:]], dim=0)
    ce_b = torch.cat([ce_b, ce_b[-1:]], dim=0)
    return ga, gb, ce_a, ce_b


# ── Projection search ──

def search_projection(proj, codec, ids, weights, cache, scorer,
                      x_a, x_b, gate_a, gate_b, up_a, up_b, z_a, z_b,
                      h2_a, h2_b, h2t_a, h2t_b, projet_a, projet_b,
                      top_rows):
    start, end, rows_total, cols_total = codec["slices"][proj]
    block_cols = cols_total // 4
    orig_block = blockify(cache["original_weights"][proj])
    changed_set = []
    searched = accepted = 0
    # activation source: for gate/up use x (896 input), for down use z (4864 input)
    act_source = z_a if proj == "down_proj" else x_a

    for pos in range(len(top_rows)):
        br = int(top_rows[pos])
        rs = slice(br * 4, br * 4 + 4)
        for bc in range(block_cols):
            searched += 1
            gb_idx = start + br * block_cols + bc
            cs = slice(bc * 4, bc * 4 + 4)
            old = weights[proj][rs, cs].clone()
            scale = float(codec["scales"][gb_idx])
            aw = act_source[:, cs].square().mean(0).repeat(4)
            cand = candidates(codec["patterns"], orig_block[gb_idx - start] / scale,
                              aw, int(ids[gb_idx]))
            best_id = int(ids[gb_idx])
            ba = float(projet_a.square().sum())
            bb = float(projet_b.square().sum())
            best = None

            for cid in cand:
                if cid == best_id:
                    continue
                nb = (codec["patterns"][cid] * scale + codec["residual"][gb_idx]).view(4, 4)

                if proj == "gate_proj":
                    sa, sb, dz_a, dz_b, pd_a, pd_b, nz_a, nz_b = \
                        scorer.score_gate_block(z_a, z_b, gate_a, gate_b, up_a, up_b,
                                                rs, nb, old, x_a, x_b, cs, projet_a, projet_b)
                    delta_extra = (nz_a, nz_b, dz_a, dz_b, "gate")
                elif proj == "up_proj":
                    sa, sb, dz_a, dz_b, pd_a, pd_b, nz_a, nz_b = \
                        scorer.score_up_block(z_a, z_b, gate_a, gate_b, up_a, up_b,
                                              rs, nb, old, x_a, x_b, cs, projet_a, projet_b)
                    delta_extra = (nz_a, nz_b, dz_a, dz_b, "up")
                else:
                    sa, sb, da, db, pd_a, pd_b = \
                        scorer.score_down_block(z_a, z_b, rs, cs, nb, old,
                                                h2_a, h2_b, h2t_a, h2t_b,
                                                projet_a, projet_b)
                    delta_extra = (da, db, pd_a, pd_b, "down")

                if sa < ba and sb <= bb + 1e-7:
                    ba, bb = sa, sb
                    best = (cid, nb, delta_extra, pd_a, pd_b)

            if best is not None:
                cid, nb, extra, pd_a, pd_b = best

                if extra[4] in ("gate", "up"):
                    nz_a, nz_b, dz_a, dz_b, _ = extra
                    weights[proj][rs, cs] = nb
                    ids[gb_idx] = cid
                    changed_set.append(gb_idx)
                    accepted += 1
                    z_a[:, rs] = nz_a
                    z_b[:, rs] = nz_b
                    dh_a, dh_b = scorer.compute_delta_h2(dz_a, dz_b, rs)
                    h2_a += dh_a
                    h2_b += dh_b
                else:
                    da, db, pd_a_, pd_b_, _ = extra
                    weights[proj][rs, cs] = nb
                    ids[gb_idx] = cid
                    changed_set.append(gb_idx)
                    accepted += 1
                    h2_a[:, rs] += da
                    h2_b[:, rs] += db

                projet_a += pd_a
                projet_b += pd_b

        if (pos + 1) % max(1, len(top_rows) // 4) == 0 or pos == len(top_rows) - 1:
            yield {"position": pos + 1, "total": len(top_rows),
                   "searched": searched, "accepted": accepted,
                   "changed": list(set(changed_set))}

    yield {"position": len(top_rows), "total": len(top_rows),
           "searched": searched, "accepted": accepted,
           "changed": list(set(changed_set))}


def save_state(ids, rnd, pos, top_rows_dict, changed, t0, init_m, cur_round):
    torch.save({"parent_sha256": sha256(PARENT), "coverage": COVERAGE,
                "ids": ids, "round": rnd, "position": pos,
                "top_rows": top_rows_dict, "changed": changed,
                "elapsed_seconds": time.time() - t0,
                "initial_metrics": init_m, "current_round": cur_round},
               STATE.parent / STATE.name)


def metrics(h2_a, h2_b, h2t_a, h2t_b, ce_a=None, ce_b=None):
    lha = float((h2_a - h2t_a).square().sum() / h2t_a.square().sum())
    lhb = float((h2_b - h2t_b).square().sum() / h2t_b.square().sum())
    m = {"L_h2_A": lha, "L_h2_B": lhb}
    if ce_a is not None:
        m["mean_CE_A"] = float(ce_a.mean())
        m["mean_CE_B"] = float(ce_b.mean()) if ce_b is not None else None
    return m


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    t0 = time.time()
    for d in (OUTPUT.parent, RESULT.parent, STATE.parent):
        d.mkdir(parents=True, exist_ok=True)

    e9 = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    x_a, x_b = e9["x_a"], e9["x_b"]
    h1_a, h1_b = e9["h1_a"], e9["h1_b"]
    h2t_a, h2t_b = e9["h2_teacher_a"], e9["h2_teacher_b"]
    tok_a, tok_b = e9["token_ids_a"], e9["token_ids_b"]
    codec = codec_state(PARENT)
    ids = codec["pattern_ids"].clone()
    w = weights_from_ids(codec, ids)

    # State vectors
    ga = x_a @ w["gate_proj"].T; gb = x_b @ w["gate_proj"].T
    ua = x_a @ w["up_proj"].T; ub = x_b @ w["up_proj"].T
    za = F.silu(ga) * ua; zb = F.silu(gb) * ub
    ya = za @ w["down_proj"].T; yb = zb @ w["down_proj"].T
    h2_a = h1_a + ya; h2_b = h1_b + yb

    # Load model for gradient
    print(json.dumps({"event": "load_model"}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()

    decoded, _ = decode_weights(PARENT)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value)

    # CE gradient (tail-weighted)
    print(json.dumps({"event": "compute_ce_gradient"}), flush=True)
    g_ce_a, g_ce_b, ce_a, ce_b = compute_ce_gradient(
        model, h2_a, h2_b, tok_a, tok_b)

    # Compute tail weights from excess CE on A (the search split)
    with torch.no_grad():
        orig_w = {n: m.weight.detach().clone() for n, m in module_map(model).items()}
        for n, mw in decoded.items():
            module_map(model)[n].weight.copy_(mw)
        student_out = model(tok_a.unsqueeze(0), labels=tok_a.unsqueeze(0), use_cache=False)
        slog_a = student_out.logits[0]
        sce_a = F.cross_entropy(slog_a[:-1], tok_a[1:], reduction="none")
        sce_a = torch.cat([sce_a, sce_a[-1:]], dim=0)
        for n, mw in orig_w.items():
            module_map(model)[n].weight.copy_(mw)
        teacher_out = model(tok_a.unsqueeze(0), labels=tok_a.unsqueeze(0), use_cache=False)
        tlog_a = teacher_out.logits[0]
        tce_a = F.cross_entropy(tlog_a[:-1], tok_a[1:], reduction="none")
        tce_a = torch.cat([tce_a, tce_a[-1:]], dim=0)
        # Restore student
        for n, mw in decoded.items():
            module_map(model)[n].weight.copy_(mw)

    excess_a = (sce_a - tce_a).clamp(min=0)
    excess_a = excess_a / max(float(excess_a.sum()), 1e-10)
    tail_w_a = excess_a ** TAIL_POWER
    tail_w_a = tail_w_a / tail_w_a.sum()

    # Also for B (just for tracking, not search)
    tail_w_b = torch.ones_like(ce_b) / ce_b.shape[0]

    den_a = float((g_ce_a * tail_w_a.sqrt().unsqueeze(1) * h2t_a).square().sum())
    den_b = float((g_ce_b * tail_w_b.sqrt().unsqueeze(1) * h2t_b).square().sum())

    scorer = CEScorer(g_ce_a, g_ce_b, tail_w_a, tail_w_b,
                      den_a, den_b, w["down_proj"])
    scorer.precompute_down_projection()

    projet_a = (scorer.g_a * (h2_a - h2t_a)).sum(1)
    projet_b = (scorer.g_b * (h2_b - h2t_b)).sum(1)

    init_m = metrics(h2_a, h2_b, h2t_a, h2t_b, ce_a, ce_b)
    print(json.dumps({"event": "start", "parent": str(PARENT),
                      "coverage": COVERAGE, "initial": init_m,
                      "memory": memory_state()}), flush=True)

    # Resume state
    rnd, pos, top_rows_d, changed, cur_round = 0, 0, {}, [], None
    if STATE.exists():
        r = torch.load(STATE, map_location="cpu", weights_only=True)
        if r.get("parent_sha256") == sha256(PARENT):
            ids = r["ids"]; rnd = int(r.get("round", 0))
            pos = int(r.get("position", 0))
            top_rows_d = r.get("top_rows", {})
            changed = list(r.get("changed", []))
            cur_round = r.get("current_round")
            w = weights_from_ids(codec, ids)
            ga = x_a @ w["gate_proj"].T; gb = x_b @ w["gate_proj"].T
            ua = x_a @ w["up_proj"].T; ub = x_b @ w["up_proj"].T
            za = F.silu(ga) * ua; zb = F.silu(gb) * ub
            ya = za @ w["down_proj"].T; yb = zb @ w["down_proj"].T
            h2_a = h1_a + ya; h2_b = h1_b + yb
            # Recompute CE gradient for resumed state
            print(json.dumps({"event": "resume_recompute_gradient"}), flush=True)
            g_ce_a, g_ce_b, ce_a, ce_b = compute_ce_gradient(
                model, h2_a, h2_b, tok_a, tok_b)
            proj_new = (g_ce_a * tail_w_a.sqrt().unsqueeze(1) * (h2_a - h2t_a)).sum(1)
            projet_a = proj_new
            projet_b = (g_ce_b * tail_w_b.sqrt().unsqueeze(1) * (h2_b - h2t_b)).sum(1)
            scorer = CEScorer(g_ce_a, g_ce_b, tail_w_a, tail_w_b,
                             den_a, den_b, w["down_proj"])
            scorer.precompute_down_projection()

    rounds = ["gate_proj", "up_proj", "down_proj"] * 2
    if cur_round is not None:
        ri = rounds.index(cur_round)
        rounds = rounds[ri:]
        rnd = ri

    for ri, proj in enumerate(rounds[rnd:], rnd):
        print(json.dumps({"event": "round_start", "round": ri + 1,
                          "proj": proj}), flush=True)
        _, _, r_total, c_total = codec["slices"][proj]
        bcols = c_total // 4

        dk = f"{proj}_rows"
        if dk not in top_rows_d:
            # Damage: projection-weighted sensitivity
            if proj == "down_proj":
                dam = ((scorer.g_a * (h2_a - h2t_a)).square() +
                       (scorer.g_b * (h2_b - h2t_b)).square()).mean(0)
            else:
                gda = scorer._precomp["g_down_a"]
                gdb = scorer._precomp["g_down_b"]
                dam = ((gda * za).square() + (gdb * zb).square()).mean(0)
            dam = dam.view(-1, 4).mean(1)
            top_rows_d[dk] = torch.topk(
                dam, max(1, int(COVERAGE * dam.numel()))).indices.tolist()

        tr = top_rows_d[dk]
        sp = pos if ri == rnd else 0

        for prog in search_projection(
                proj, codec, ids, w, e9, scorer,
                x_a, x_b, ga, gb, ua, ub, za, zb,
                h2_a, h2_b, h2t_a, h2t_b, projet_a, projet_b,
                tr[sp:]):

            changed = list(set(changed + prog["changed"]))
            save_state(ids, ri, sp + prog["position"], top_rows_d,
                       changed, t0, init_m, proj)
            patch_ids(PARENT, OUTPUT, ids)

            print(json.dumps({"event": "progress", "round": ri + 1,
                              "proj": proj,
                              "done": prog["position"], "total": prog["total"],
                              "searched": prog["searched"],
                              "accepted": prog["accepted"],
                              "changed_total": len(changed),
                              "metrics": metrics(h2_a, h2_b, h2t_a, h2t_b),
                              "memory": memory_state()}), flush=True)

            if prog["position"] == prog["total"]:
                # Rebuild and recompute gradient
                w = weights_from_ids(codec, ids)
                ga = x_a @ w["gate_proj"].T; gb = x_b @ w["gate_proj"].T
                ua = x_a @ w["up_proj"].T; ub = x_b @ w["up_proj"].T
                za = F.silu(ga) * ua; zb = F.silu(gb) * ub
                ya = za @ w["down_proj"].T; yb = zb @ w["down_proj"].T
                h2_a = h1_a + ya; h2_b = h1_b + yb
                with torch.no_grad():
                    for n in ("gate_proj", "up_proj"):
                        module_map(model)[n].weight.copy_(w[n])
                print(json.dumps({"event": "recompute_gradient",
                                  "round": ri + 1}), flush=True)
                g_ce_a, g_ce_b, ce_a, ce_b = compute_ce_gradient(
                    model, h2_a, h2_b, tok_a, tok_b)
                projet_a = (scorer.g_a * (h2_a - h2t_a)).sum(1)
                projet_b = (scorer.g_b * (h2_b - h2t_b)).sum(1)
                scorer = CEScorer(g_ce_a, g_ce_b, tail_w_a, tail_w_b,
                                 den_a, den_b, w["down_proj"])
                scorer.precompute_down_projection()
                top_rows_d = {}

    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")

    final = metrics(h2_a, h2_b, h2t_a, h2t_b, ce_a, ce_b)
    result = {"experiment": LABEL, "parent": str(PARENT),
              "parent_sha256": sha256(PARENT), "coverage": COVERAGE,
              "rounds": ri + 1,
              "initial": init_m, "final": final,
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size * 8 / codec["meta"]["weights"],
              "output_sha256": sha256(OUTPUT),
              "elapsed_seconds": time.time() - t0, "ppl": None}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULT.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
