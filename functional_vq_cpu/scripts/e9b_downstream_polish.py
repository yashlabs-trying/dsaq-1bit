"""Downstream-sensitive functional polish for E9b.

Instead of optimizing raw ||h2 - h2_teacher||² (which E9a showed is a poor proxy
for PPL), we optimize the sensitivity-weighted loss:

  L_sense = Σ_t (g_t · (h2_t - h2_teacher_t))²

where g_t = ∂||L1(h2) - h3_teacher||² / ∂h2 is the downstream sensitivity gradient
precomputed by export_e9b_cache.py.

This concentrates the optimization on h2 directions that actually matter to the
frozen BF16 downstream layers, ignoring irrelevant directions.

Calibration A selects; calibration B is a strict non-worsening acceptance gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import torch

from cpu_codec import blockify, codec_state, patch_ids, unblock
from e7b_cpu_reassign import candidates, memory_state


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cache/e9b_cache.pt"
PARENT = Path(os.environ.get("E9B_PARENT", ROOT / "packed/E8g_down_polish_round5.packed"))
COVERAGE = float(os.environ.get("E9B_COVERAGE", "0.05"))
LABEL = os.environ.get("E9B_LABEL", "E9b_downstream_l1_5pct")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
STATE = ROOT / "checkpoints" / f"{LABEL}_state.pt"
CHECKPOINT_ROWS = int(os.environ.get("E9B_CHECKPOINT_ROWS", "1"))
THREADS = int(os.environ.get("E9B_THREADS", "4"))
MIN_AVAILABLE_GB = float(os.environ.get("E9B_MIN_AVAILABLE_GB", "1.0"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {name: unblock(blocks[s:e], (r, c)) for name, (s, e, r, c) in codec["slices"].items()}


def downstream_metrics(h2_a, h2_b, cache):
    ga, gb = cache["sensitivity_gradient_a"], cache["sensitivity_gradient_b"]
    pg_a = (ga * h2_a).sum(1)
    pg_b = (gb * h2_b).sum(1)
    pg_t_a = (ga * cache["h2_teacher_a"]).sum(1)
    pg_t_b = (gb * cache["h2_teacher_b"]).sum(1)
    err_a = pg_a - pg_t_a
    err_b = pg_b - pg_t_b
    lh3_a = float((h2_a - cache["h2_teacher_a"]).square().sum() / cache["h2_teacher_a"].square().sum())
    lh3_b = float((h2_b - cache["h2_teacher_b"]).square().sum() / cache["h2_teacher_b"].square().sum())
    sense_a = float((err_a).square().sum() / (pg_t_a).square().sum())
    sense_b = float((err_b).square().sum() / (pg_t_b).square().sum())
    return {
        "L_h2_A": lh3_a, "L_h2_B": lh3_b,
        "L_sense_A": sense_a, "L_sense_B": sense_b,
        "E_sense_A": sense_a**0.5, "E_sense_B": sense_b**0.5,
    }


def save_state(ids, position, top_rows, changed, started, initial_metrics):
    payload = {"parent_sha256": sha256(PARENT), "coverage": COVERAGE, "ids": ids,
               "position": position, "top_rows": top_rows, "changed": changed,
               "elapsed_seconds": time.time() - started, "initial_metrics": initial_metrics}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, STATE)


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    for directory in (OUTPUT.parent, RESULT.parent, STATE.parent):
        directory.mkdir(parents=True, exist_ok=True)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    if cache["parent_sha256"] != sha256(PARENT):
        raise RuntimeError("E9b cache is not anchored to parent")

    codec = codec_state(PARENT)
    ids = codec["pattern_ids"].clone()
    position, changed, top_rows, initial_metrics = 0, [], None, None

    if STATE.exists():
        resume = torch.load(STATE, map_location="cpu", weights_only=True)
        if resume["parent_sha256"] != sha256(PARENT) or resume.get("coverage") != COVERAGE:
            raise RuntimeError("Resume checkpoint does not match")
        ids, position, top_rows = resume["ids"], int(resume["position"]), resume["top_rows"]
        changed, initial_metrics = list(resume["changed"]), resume["initial_metrics"]
        print(json.dumps({"event": "resume", "position": position}), flush=True)

    weights = weights_from_ids(codec, ids)
    x_a, x_b = cache["x_a"], cache["x_b"]
    gate_a = x_a @ weights["gate_proj"].T
    gate_b = x_b @ weights["gate_proj"].T
    up_a = x_a @ weights["up_proj"].T
    up_b = x_b @ weights["up_proj"].T
    z_a = torch.nn.functional.silu(gate_a) * up_a
    z_b = torch.nn.functional.silu(gate_b) * up_b
    y_a = z_a @ weights["down_proj"].T
    y_b = z_b @ weights["down_proj"].T
    h2_a = cache["h1_a"] + y_a
    h2_b = cache["h1_b"] + y_b

    current_metrics = downstream_metrics(h2_a, h2_b, cache)
    if initial_metrics is None:
        initial_metrics = dict(current_metrics)

    if top_rows is None:
        ga, gb = cache["sensitivity_gradient_a"], cache["sensitivity_gradient_b"]
        err_a = h2_a - cache["h2_teacher_a"]
        err_b = h2_b - cache["h2_teacher_b"]
        sense_damage = ((ga * err_a).square() + (gb * err_b).square()).mean(0).view(-1, 4).mean(1)
        top_rows = torch.topk(sense_damage, max(1, int(COVERAGE * sense_damage.numel()))).indices.tolist()

    start, _, _, cols_total = codec["slices"]["down_proj"]
    block_cols = cols_total // 4
    original = blockify(cache["original_weights"]["down_proj"])

    ga, gb = cache["sensitivity_gradient_a"], cache["sensitivity_gradient_b"]
    h2t_a, h2t_b = cache["h2_teacher_a"], cache["h2_teacher_b"]
    h3t_a, h3t_b = cache["h3_teacher_a"], cache["h3_teacher_b"]

    projet_a = (ga * (h2_a - h2t_a)).sum(1)
    projet_b = (gb * (h2_b - h2t_b)).sum(1)
    sense_den_a = float(h3t_a.square().sum())
    sense_den_b = float(h3t_b.square().sum())
    total_sense_a = projet_a.square().sum()
    total_sense_b = projet_b.square().sum()

    print(json.dumps({"event": "start", "parent": str(PARENT), "coverage": COVERAGE,
                      "row_groups": len(top_rows), "initial": initial_metrics,
                      "memory": memory_state()}), flush=True)

    searched_blocks = accepted_changes = 0
    for row_position in range(position, len(top_rows)):
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < MIN_AVAILABLE_GB:
            save_state(ids, row_position, top_rows, changed, started, initial_metrics)
            raise MemoryError(f"Only {available_gb:.2f} GB available; saved and stopped")

        block_row = int(top_rows[row_position])
        row_slice = slice(block_row * 4, block_row * 4 + 4)

        for block_col in range(block_cols):
            searched_blocks += 1
            global_block = start + block_row * block_cols + block_col
            col_slice = slice(block_col * 4, block_col * 4 + 4)
            old_block = weights["down_proj"][row_slice, col_slice].clone()
            scale = float(codec["scales"][global_block])
            activation_weight = z_a[:, col_slice].square().mean(0).repeat(4)
            candidate_ids = candidates(codec["patterns"], original[global_block - start] / scale,
                                       activation_weight, int(ids[global_block]))

            best_id = int(ids[global_block])
            best_a, best_b = float(total_sense_a), float(total_sense_b)
            best = None

            for candidate_id in candidate_ids:
                if candidate_id == best_id:
                    continue
                new_block = (codec["patterns"][candidate_id] * scale + codec["residual"][global_block]).view(4, 4)
                delta_a = z_a[:, col_slice] @ (new_block - old_block).T
                delta_b = z_b[:, col_slice] @ (new_block - old_block).T

                ga_row = ga[:, row_slice]
                gb_row = gb[:, row_slice]

                proj_delta_a = (ga_row * delta_a).sum(1)
                proj_delta_b = (gb_row * delta_b).sum(1)

                new_projet_a = projet_a + proj_delta_a
                new_projet_b = projet_b + proj_delta_b

                score_a = float(new_projet_a.square().sum())
                score_b = float(new_projet_b.square().sum())

                if score_a < best_a and score_b <= best_b + 1e-7:
                    best_id, best_a, best_b = candidate_id, score_a, score_b
                    best = (new_block, proj_delta_a, proj_delta_b, delta_a, delta_b)

            if best is not None:
                new_block, proj_delta_a, proj_delta_b, delta_a, delta_b = best
                weights["down_proj"][row_slice, col_slice] = new_block
                ids[global_block] = best_id
                changed.append(int(global_block))
                accepted_changes += 1
                h2_a[:, row_slice] += delta_a
                h2_b[:, row_slice] += delta_b
                projet_a += proj_delta_a
                projet_b += proj_delta_b
                total_sense_a = torch.tensor(best_a)
                total_sense_b = torch.tensor(best_b)

        completed = row_position + 1
        if completed % CHECKPOINT_ROWS == 0 or completed == len(top_rows):
            save_state(ids, completed, top_rows, changed, started, initial_metrics)
            patch_ids(PARENT, OUTPUT, ids)
            print(json.dumps({"event": "progress", "row_groups_done": completed,
                              "row_groups_total": len(top_rows), "searched_blocks": searched_blocks,
                              "accepted_changes": accepted_changes,
                              "blocks_per_second": searched_blocks / max(time.time() - started, 1e-9),
                              "metrics": downstream_metrics(h2_a, h2_b, cache),
                              "memory": memory_state()}), flush=True)

    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")

    result = {"experiment": LABEL, "parent": str(PARENT), "parent_sha256": sha256(PARENT),
              "coverage": COVERAGE, "threads": THREADS, "row_groups": len(top_rows),
              "searched_blocks": searched_blocks, "ids_changed": len(set(changed)),
              "initial": initial_metrics, "final": downstream_metrics(h2_a, h2_b, cache),
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size * 8 / codec["meta"]["weights"],
              "output_sha256": sha256(OUTPUT), "elapsed_seconds": time.time() - started,
              "ppl": None}
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
