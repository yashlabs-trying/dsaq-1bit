"""E10A: Gate + Up downstream sensitivity polish with alternating rounds.

Architecture
────────────
  SensitivityScorer  — swappable metric (single g vector → rank-r subspace later)
  ProjectionSearcher — reusable candidate search engine for any projection

Loop
────
  Round 1: Gate polish          (Up frozen)   → recompute gradient after batch
  Round 2: Up polish            (Gate frozen) → recompute gradient after batch
  Round 3: Gate polish (refine) → recompute gradient after batch
  Round 4: Up polish   (refine) → recompute gradient after batch

Uses L1+L2 downstream sensitivity:
  g = ∂||L2(L1(h2)) - h4_teacher||² / ∂h2

Candidate Δh2 = Δz @ W_down.T  where Δz = Δ(silu(gate)⊙up) for the changed rows.
Score = Σ_t (g_t · Δh2_t)²  projected through the downstream gradient.

Supports future upgrade: replace the single g vector with a rank-r subspace UΛU^T.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, codec_state, decode_weights, patch_ids, unblock
from e7b_cpu_reassign import candidates, memory_state


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cache/e9c_cache.pt"
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
PARENT = Path(os.environ.get("E10A_PARENT", ROOT / "packed/E9c_downstream_l12_5pct.packed"))
COVERAGE = float(os.environ.get("E10A_COVERAGE", "0.05"))
LABEL = os.environ.get("E10A_LABEL", "E10a_gate_up_downstream")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
STATE = ROOT / "checkpoints" / f"{LABEL}_state.pt"
CHECKPOINT_ROWS = int(os.environ.get("E10A_CHECKPOINT_ROWS", "2"))
THREADS = int(os.environ.get("E10A_THREADS", "4"))
MIN_AVAILABLE_GB = float(os.environ.get("E10A_MIN_AVAILABLE_GB", "1.0"))
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {name: unblock(blocks[s:e], (r, c)) for name, (s, e, r, c) in codec["slices"].items()}


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


# ── Sensitivity Scorer (upgradeable to subspace) ──────────────────────────

class SensitivityScorer:
    """Scores block changes by their projection onto downstream-sensitive directions.

    Current: single gradient vector g.
    Future: rank-r subspace UΛU^T (swap _compute_delta_loss method).
    """
    def __init__(self, g_a: torch.Tensor, g_b: torch.Tensor,
                 down_weight: torch.Tensor, down_slices):
        self.g_a = g_a       # (512, 896)
        self.g_b = g_b       # (512, 896)
        self.down = down_weight  # (896, 4864)
        self.down_slices = down_slices

        self._precomp = {}

    def precompute_down_projection(self):
        """Precompute g @ W_down for efficient Gate/Up scoring."""
        self._precomp["g_down_a"] = self.g_a @ self.down     # (512, 4864)
        self._precomp["g_down_b"] = self.g_b @ self.down     # (512, 4864)

    def score_gate_block(self, z_a, z_b, gate_a, gate_b, up_a, up_b,
                         row_slice, new_block, old_block, x_a, x_b,
                         col_slice, projet_a, projet_b, norm_factor_a, norm_factor_b):
        """Score a Gate block change: Δgate → Δz → Δh2 → project through g."""
        delta_gate_a = x_a[:, col_slice] @ (new_block - old_block).T  # (512, 4)
        delta_gate_b = x_b[:, col_slice] @ (new_block - old_block).T
        new_z_a = F.silu(gate_a[:, row_slice] + delta_gate_a) * up_a[:, row_slice]
        new_z_b = F.silu(gate_b[:, row_slice] + delta_gate_b) * up_b[:, row_slice]
        dz_a = new_z_a - z_a[:, row_slice]
        dz_b = new_z_b - z_b[:, row_slice]
        g_down_a = self._precomp["g_down_a"][:, row_slice]
        g_down_b = self._precomp["g_down_b"][:, row_slice]
        proj_a = (g_down_a * dz_a).sum(1)
        proj_b = (g_down_b * dz_b).sum(1)
        new_projet_a = projet_a + proj_a
        new_projet_b = projet_b + proj_b
        score_a = float(new_projet_a.square().sum() / norm_factor_a)
        score_b = float(new_projet_b.square().sum() / norm_factor_b)
        return score_a, score_b, dz_a, dz_b, proj_a, proj_b, new_z_a, new_z_b

    def score_up_block(self, z_a, z_b, gate_a, gate_b, up_a, up_b,
                       row_slice, new_block, old_block, x_a, x_b,
                       col_slice, projet_a, projet_b, norm_factor_a, norm_factor_b):
        """Score an Up block change."""
        delta_up_a = x_a[:, col_slice] @ (new_block - old_block).T
        delta_up_b = x_b[:, col_slice] @ (new_block - old_block).T
        new_z_a = F.silu(gate_a[:, row_slice]) * (up_a[:, row_slice] + delta_up_a)
        new_z_b = F.silu(gate_b[:, row_slice]) * (up_b[:, row_slice] + delta_up_b)
        dz_a = new_z_a - z_a[:, row_slice]
        dz_b = new_z_b - z_b[:, row_slice]
        g_down_a = self._precomp["g_down_a"][:, row_slice]
        g_down_b = self._precomp["g_down_b"][:, row_slice]
        proj_a = (g_down_a * dz_a).sum(1)
        proj_b = (g_down_b * dz_b).sum(1)
        new_projet_a = projet_a + proj_a
        new_projet_b = projet_b + proj_b
        score_a = float(new_projet_a.square().sum() / norm_factor_a)
        score_b = float(new_projet_b.square().sum() / norm_factor_b)
        return score_a, score_b, dz_a, dz_b, proj_a, proj_b, new_z_a, new_z_b

    def compute_delta_h2_from_dz(self, dz_a, dz_b, gate_rows_for_down):
        """Compute Δh2 = Δz @ W_down.T for indirect h2 tracking."""
        # dz: (512, 4), self.down[:, gate_rows]: (896, 4)
        delta_h2_a = dz_a @ self.down[:, gate_rows_for_down].T  # (512, 4) @ (4, 896) = (512, 896)
        delta_h2_b = dz_b @ self.down[:, gate_rows_for_down].T
        return delta_h2_a, delta_h2_b


# ── Downstream gradient recomputation ─────────────────────────────────────

def compute_downstream_gradient(model, h2_a, h2_b, token_ids_a, token_ids_b,
                                 h4_teacher_a, h4_teacher_b):
    """Compute g = ∂||L2(L1(h2)) - h4_teacher||² / ∂h2 for current student h2."""
    def run_l12(h2, tok_ids):
        seq_len = h2.shape[0]
        pos_ids = torch.arange(seq_len, device=h2.device).unsqueeze(0)
        causal_mask = torch.full((seq_len, seq_len), torch.finfo(h2.dtype).min, device=h2.device)
        causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        h = h2
        for layer_idx in (1, 2):
            layer = model.model.layers[layer_idx]
            cos, sin = model.model.rotary_emb(h.unsqueeze(0), pos_ids)
            pos_emb = (cos, sin)
            h_norm = layer.input_layernorm(h)
            attn = layer.self_attn(h_norm.unsqueeze(0), attention_mask=causal_mask,
                                   position_embeddings=pos_emb)[0].squeeze(0)
            h = h + attn
            h_attn = h
            h = layer.post_attention_layernorm(h)
            h = h_attn + layer.mlp(h)
        return h

    h2s_a = h2_a.detach().clone().requires_grad_(True)
    h2s_b = h2_b.detach().clone().requires_grad_(True)
    h4s_a = run_l12(h2s_a, token_ids_a)
    h4s_b = run_l12(h2s_b, token_ids_b)
    loss_a = (h4s_a - h4_teacher_a).square().sum()
    loss_b = (h4s_b - h4_teacher_b).square().sum()
    ga = torch.autograd.grad(loss_a, h2s_a)[0]
    gb = torch.autograd.grad(loss_b, h2s_b)[0]
    return ga, gb


# ── Projection search ────────────────────────────────────────────────────

def search_projection(projection, codec, ids, weights, cache, scorer,
                      x_a, x_b, gate_a, gate_b, up_a, up_b, z_a, z_b,
                      y_a, y_b, h2_a, h2_b, projet_a, projet_b,
                      norm_factor_a, norm_factor_b, top_rows):
    """Search one projection's worst blocks using the sensitivity scorer."""
    start, end, rows_total, cols_total = codec["slices"][projection]
    block_cols = cols_total // 4
    original_blocks = blockify(cache["original_weights"][projection])
    yden_a = cache["h2_teacher_a"].square().sum()
    yden_b = cache["h2_teacher_b"].square().sum()

    changed_blocks = []
    searched = 0
    accepted = 0

    for position in range(len(top_rows)):
        block_row = int(top_rows[position])
        row_slice = slice(block_row * 4, block_row * 4 + 4)

        for block_col in range(block_cols):
            searched += 1
            global_block = start + block_row * block_cols + block_col
            col_slice = slice(block_col * 4, block_col * 4 + 4)
            old_block = weights[projection][row_slice, col_slice].clone()
            scale = float(codec["scales"][global_block])
            activation_weight = x_a[:, col_slice].square().mean(0).repeat(4)
            candidate_ids = candidates(
                codec["patterns"],
                original_blocks[global_block - start] / scale,
                activation_weight, int(ids[global_block])
            )

            best_id = int(ids[global_block])
            best_a = float(projet_a.square().sum() / norm_factor_a)
            best_b = float(projet_b.square().sum() / norm_factor_b)
            best = None

            for candidate_id in candidate_ids:
                if candidate_id == best_id:
                    continue
                new_block = (codec["patterns"][candidate_id] * scale +
                             codec["residual"][global_block]).view(4, 4)

                if projection == "gate_proj":
                    sa, sb, dz_a, dz_b, pj_a, pj_b, nz_a, nz_b = \
                        scorer.score_gate_block(
                            z_a, z_b, gate_a, gate_b, up_a, up_b,
                            row_slice, new_block, old_block,
                            x_a, x_b, col_slice,
                            projet_a, projet_b, norm_factor_a, norm_factor_b)
                else:
                    sa, sb, dz_a, dz_b, pj_a, pj_b, nz_a, nz_b = \
                        scorer.score_up_block(
                            z_a, z_b, gate_a, gate_b, up_a, up_b,
                            row_slice, new_block, old_block,
                            x_a, x_b, col_slice,
                            projet_a, projet_b, norm_factor_a, norm_factor_b)

                if sa < best_a and sb <= best_b + 1e-7:
                    best_id, best_a, best_b = candidate_id, sa, sb
                    best = (new_block, dz_a, dz_b, pj_a, pj_b, nz_a, nz_b)

            if best is not None:
                new_block, dz_a, dz_b, pj_a, pj_b, nz_a, nz_b = best
                weights[projection][row_slice, col_slice] = new_block
                ids[global_block] = best_id
                changed_blocks.append(int(global_block))
                accepted += 1
                z_a[:, row_slice] = nz_a
                z_b[:, row_slice] = nz_b
                projet_a += pj_a
                projet_b += pj_b
                delta_h2_a, delta_h2_b = scorer.compute_delta_h2_from_dz(
                    dz_a, dz_b, row_slice)
                h2_a += delta_h2_a
                h2_b += delta_h2_b

        if (position + 1) % max(1, len(top_rows) // 4) == 0 or position == len(top_rows) - 1:
            yield {
                "position": position + 1, "total": len(top_rows),
                "searched": searched, "accepted": accepted,
                "changed_set": list(set(changed_blocks)),
            }

    yield {
        "position": len(top_rows), "total": len(top_rows),
        "searched": searched, "accepted": accepted,
        "changed_set": list(set(changed_blocks)),
    }


# ── State helpers ─────────────────────────────────────────────────────────

def save_state(ids, round_idx, position, top_rows_list, changed_list,
               started, initial_metrics, current_round):
    payload = {"parent_sha256": sha256(PARENT), "coverage": COVERAGE,
               "ids": ids, "round": round_idx, "position": position,
               "top_rows": top_rows_list, "changed": changed_list,
               "elapsed_seconds": time.time() - started,
               "initial_metrics": initial_metrics, "current_round": current_round}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, STATE)


# ── Metric helpers ────────────────────────────────────────────────────────

def compute_metrics(h2_a, h2_b, cache, ga=None, gb=None):
    lh2a = float((h2_a - cache["h2_teacher_a"]).square().sum() / cache["h2_teacher_a"].square().sum())
    lh2b = float((h2_b - cache["h2_teacher_b"]).square().sum() / cache["h2_teacher_b"].square().sum())
    metrics = {"L_h2_A": lh2a, "L_h2_B": lh2b}
    if ga is not None and gb is not None:
        pg_a = (ga * h2_a).sum(1)
        pg_b = (gb * h2_b).sum(1)
        pg_t_a = (ga * cache["h2_teacher_a"]).sum(1)
        pg_t_b = (gb * cache["h2_teacher_b"]).sum(1)
        err_a = pg_a - pg_t_a
        err_b = pg_b - pg_t_b
        metrics["L_sense_A"] = float(err_a.square().sum() / pg_t_a.square().sum())
        metrics["L_sense_B"] = float(err_b.square().sum() / pg_t_b.square().sum())
        metrics["E_sense_A"] = metrics["L_sense_A"]**0.5
        metrics["E_sense_B"] = metrics["L_sense_B"]**0.5
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    for directory in (OUTPUT.parent, RESULT.parent, STATE.parent):
        directory.mkdir(parents=True, exist_ok=True)

    # ── Load caches and codec ──
    e9c_cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    e9_cache = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    if e9c_cache["parent_sha256"] != sha256(PARENT):
        print(json.dumps({"event": "cache_mismatch", "expected": sha256(PARENT),
                          "got": e9c_cache["parent_sha256"]}), flush=True)

    codec = codec_state(PARENT)
    ids = codec["pattern_ids"].clone()
    weights = weights_from_ids(codec, ids)

    x_a, x_b = e9_cache["x_a"], e9_cache["x_b"]
    h1_a, h1_b = e9_cache["h1_a"], e9_cache["h1_b"]
    h2t_a, h2t_b = e9_cache["h2_teacher_a"], e9_cache["h2_teacher_b"]
    h4t_a = e9c_cache["h4_teacher_a"]
    h4t_b = e9c_cache["h4_teacher_b"]

    # ── Build state vectors ──
    gate_a = x_a @ weights["gate_proj"].T
    gate_b = x_b @ weights["gate_proj"].T
    up_a = x_a @ weights["up_proj"].T
    up_b = x_b @ weights["up_proj"].T
    z_a = F.silu(gate_a) * up_a
    z_b = F.silu(gate_b) * up_b
    y_a = z_a @ weights["down_proj"].T
    y_b = z_b @ weights["down_proj"].T
    h2_a = h1_a + y_a
    h2_b = h1_b + y_b

    # ── Load BF16 model for gradient recomputation ──
    print(json.dumps({"event": "load_model"}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()
    decoded, _ = decode_weights(PARENT)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value)

    # ── Resume or compute initial gradient ──
    round_idx, position, top_rows_dict, changed, initial_metrics = 0, 0, {}, [], None
    current_round = None  # None = initial gradient

    if STATE.exists():
        resume = torch.load(STATE, map_location="cpu", weights_only=True)
        if resume.get("parent_sha256") == sha256(PARENT):
            ids = resume["ids"]
            round_idx = int(resume.get("round", 0))
            position = int(resume.get("position", 0))
            top_rows_dict = resume.get("top_rows", {})
            changed = list(resume.get("changed", []))
            initial_metrics = resume.get("initial_metrics")
            current_round = resume.get("current_round", None)
            print(json.dumps({"event": "resume", "round": round_idx,
                              "position": position, "current_round": current_round}), flush=True)
            # Rebuild weights, z, h2 from loaded ids
            weights = weights_from_ids(codec, ids)
            gate_a = x_a @ weights["gate_proj"].T
            gate_b = x_b @ weights["gate_proj"].T
            up_a = x_a @ weights["up_proj"].T
            up_b = x_b @ weights["up_proj"].T
            z_a = F.silu(gate_a) * up_a
            z_b = F.silu(gate_b) * up_b
            y_a = z_a @ weights["down_proj"].T
            y_b = z_b @ weights["down_proj"].T
            h2_a = h1_a + y_a
            h2_b = h1_b + y_b

    print(json.dumps({"event": "compute_initial_gradient"}), flush=True)
    ga, gb = compute_downstream_gradient(
        model, h2_a, h2_b,
        e9_cache["token_ids_a"], e9_cache["token_ids_b"],
        h4t_a, h4t_b)
    norm_factor_a = float(h4t_a.square().sum())
    norm_factor_b = float(h4t_b.square().sum())

    scorer = SensitivityScorer(ga, gb, weights["down_proj"],
                                codec["slices"]["down_proj"])
    scorer.precompute_down_projection()

    projet_a = (ga * (h2_a - h2t_a)).sum(1)
    projet_b = (gb * (h2_b - h2t_b)).sum(1)

    if initial_metrics is None:
        initial_metrics = compute_metrics(h2_a, h2_b, e9_cache, ga, gb)

    print(json.dumps({"event": "start", "parent": str(PARENT), "coverage": COVERAGE,
                      "initial": initial_metrics, "memory": memory_state()}), flush=True)

    # ── Round schedule ──
    rounds = ["gate_proj", "up_proj", "gate_proj", "up_proj"]
    if current_round is not None:
        # Resume from where we left off
        rounds = rounds[rounds.index(current_round):]
        round_idx = rounds.index(current_round)

    for rindex, projection in enumerate(rounds[round_idx:], round_idx):
        print(json.dumps({"event": "round_start", "round": rindex + 1,
                          "projection": projection}), flush=True)

        start_s, end_s, rows_total, cols_total = codec["slices"][projection]
        block_cols = cols_total // 4

        # Determine which row groups to search
        damage_key = f"{projection}_rows"
        if damage_key not in top_rows_dict:
            # Compute damage using projection of gate/up error through the sensitivity gradient
            g_down_a = scorer._precomp["g_down_a"]  # (512, 4864)
            g_down_b = scorer._precomp["g_down_b"]
            damage = ((g_down_a * z_a).square() + (g_down_b * z_b).square()).mean(0)
            damage = damage.view(-1, 4).mean(1)
            top_rows_dict[damage_key] = torch.topk(
                damage, max(1, int(COVERAGE * damage.numel()))).indices.tolist()

        top_rows = top_rows_dict[damage_key]
        start_position = position if rindex == round_idx else 0

        # Run the search
        for progress in search_projection(
                projection, codec, ids, weights, e9_cache, scorer,
                x_a, x_b, gate_a, gate_b, up_a, up_b, z_a, z_b,
                y_a, y_b, h2_a, h2_b, projet_a, projet_b,
                norm_factor_a, norm_factor_b, top_rows[start_position:]):

            pos = progress["position"]
            total_pos = start_position + pos
            changed_set = list(set(changed + progress["changed_set"]))

            # Checkpoint
            save_state(ids, rindex, total_pos, top_rows_dict, changed_set,
                       started, initial_metrics, projection)
            patch_ids(PARENT, OUTPUT, ids)

            metrics = compute_metrics(h2_a, h2_b, e9_cache, ga, gb)
            print(json.dumps({"event": "progress", "round": rindex + 1,
                              "projection": projection,
                              "row_groups_done": total_pos,
                              "row_groups_total": len(top_rows) + start_position,
                              "searched_blocks": progress["searched"],
                              "accepted": progress["accepted"],
                              "changed_total": len(changed_set),
                              "metrics": metrics,
                              "memory": memory_state()}), flush=True)

            if pos == progress["total"]:
                # Round complete — recompute gradient
                # Rebuild weights, z, h2 from updated ids
                weights = weights_from_ids(codec, ids)
                gate_a = x_a @ weights["gate_proj"].T
                gate_b = x_b @ weights["gate_proj"].T
                up_a = x_a @ weights["up_proj"].T
                up_b = x_b @ weights["up_proj"].T
                z_a = F.silu(gate_a) * up_a
                z_b = F.silu(gate_b) * up_b
                y_a = z_a @ weights["down_proj"].T
                y_b = z_b @ weights["down_proj"].T
                h2_a = h1_a + y_a
                h2_b = h1_b + y_b

                # Also update the model weights in the BF16 copy
                with torch.no_grad():
                    for name in ("gate_proj", "up_proj"):
                        module_map(model)[name].weight.copy_(weights[name])

                print(json.dumps({"event": "recompute_gradient",
                                  "round": rindex + 1}), flush=True)
                ga, gb = compute_downstream_gradient(
                    model, h2_a, h2_b,
                    e9_cache["token_ids_a"], e9_cache["token_ids_b"],
                    h4t_a, h4t_b)
                # Update scorer
                scorer = SensitivityScorer(ga, gb, weights["down_proj"],
                                            codec["slices"]["down_proj"])
                scorer.precompute_down_projection()
                projet_a = (ga * (h2_a - h2t_a)).sum(1)
                projet_b = (gb * (h2_b - h2t_b)).sum(1)
                # Clear row cache for next round
                for k in list(top_rows_dict.keys()):
                    if k.startswith(f"{projection}_rows") or k.startswith(f"{'up_proj' if projection == 'gate_proj' else 'gate_proj'}_rows"):
                        del top_rows_dict[k]

        position = 0

    # ── Finalize ──
    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")

    final_metrics = compute_metrics(h2_a, h2_b, e9_cache, ga, gb)
    result = {"experiment": LABEL, "parent": str(PARENT),
              "parent_sha256": sha256(PARENT), "coverage": COVERAGE,
              "rounds_completed": len(rounds),
              "initial": initial_metrics, "final": final_metrics,
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size * 8 / codec["meta"]["weights"],
              "output_sha256": sha256(OUTPUT),
              "elapsed_seconds": time.time() - started, "ppl": None}
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
