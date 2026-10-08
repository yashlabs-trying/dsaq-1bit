"""Memory-bounded, resumable CPU functional reassignment for E7b.

Gate/Up candidates are generated from the adapted E7a codebook and scored by
the exact change in final MLP output SSE. Calibration A selects; calibration B
must not worsen. No full Qwen model is loaded and no PPL claim is made here.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import psutil
import torch

from cpu_codec import blockify, codec_state, decode_weights, patch_ids, unblock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "artifacts" / "functional_vq_l0"
CACHE_PATH = ROOT / "cache" / "layer0_functional_cache.pt"
PARENT = Path(os.environ.get("E7B_PARENT", SOURCE / "packed/E7a_codebook_100.packed"))
COVERAGE = float(os.environ.get("E7B_COVERAGE", "0.10"))
LABEL = os.environ.get("E7B_LABEL", f"E7b_reassign_{int(COVERAGE * 100)}pct")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
STATE = ROOT / "checkpoints" / f"{LABEL}_state.pt"
CHECKPOINT_ROWS = int(os.environ.get("E7B_CHECKPOINT_ROWS", "4"))
THREADS = int(os.environ.get("E7B_THREADS", "4"))
MIN_AVAILABLE_GB = float(os.environ.get("E7B_MIN_AVAILABLE_GB", "1.0"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def memory_state():
    vm = psutil.virtual_memory()
    return {
        "rss_bytes": psutil.Process().memory_info().rss,
        "available_bytes": vm.available,
        "percent_used": vm.percent,
    }


def mlp(weights, x):
    gate = x @ weights["gate_proj"].T
    up = x @ weights["up_proj"].T
    z = torch.nn.functional.silu(gate) * up
    return gate, up, z, z @ weights["down_proj"].T


def candidates(patterns, target, activation_weight, current):
    squared = (patterns - target).square()
    distance = squared.sum(1)
    weighted = (squared * activation_weight).sum(1)
    nearest_weight = torch.topk(distance, 3, largest=False).indices.tolist()
    nearest_activation = torch.topk(weighted, 3, largest=False).indices.tolist()
    top = torch.topk(distance, 64, largest=False).indices
    chosen = list(dict.fromkeys([int(current)] + nearest_weight + nearest_activation))
    pool = torch.nn.functional.normalize(patterns[top], dim=1)
    chosen_vectors = torch.nn.functional.normalize(patterns[torch.tensor(chosen)], dim=1)
    for _ in range(2):
        diversity = 1 - (pool @ chosen_vectors.T).abs().max(1).values
        order = torch.argsort(diversity, descending=True)
        pick = next(int(top[index]) for index in order if int(top[index]) not in chosen)
        chosen.append(pick)
        chosen_vectors = torch.nn.functional.normalize(patterns[torch.tensor(chosen)], dim=1)
    return chosen[:8]


def weights_from_ids(codec, ids):
    block_values = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    weights = {}
    for name, (start, end, rows, cols) in codec["slices"].items():
        weights[name] = unblock(block_values[start:end], (rows, cols))
    return weights


def metrics(z_a, z_b, y_a, y_b, cache):
    return {
        "Ez_A": float((z_a - cache["z_teacher_a"]).norm() / cache["z_teacher_a"].norm()),
        "Ez_B": float((z_b - cache["z_teacher_b"]).norm() / cache["z_teacher_b"].norm()),
        "Ey_A": float((y_a - cache["y_teacher_a"]).norm() / cache["y_teacher_a"].norm()),
        "Ey_B": float((y_b - cache["y_teacher_b"]).norm() / cache["y_teacher_b"].norm()),
        "Ly_A": float((y_a - cache["y_teacher_a"]).square().sum() / cache["y_teacher_a"].square().sum()),
        "Ly_B": float((y_b - cache["y_teacher_b"]).square().sum() / cache["y_teacher_b"].square().sum()),
    }


def save_state(ids, projection_index, row_position, top_rows, changed, started, initial_metrics):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "parent": str(PARENT),
        "parent_sha256": sha256(PARENT),
        "coverage": COVERAGE,
        "ids": ids,
        "projection_index": projection_index,
        "row_position": row_position,
        "top_rows": top_rows,
        "changed": changed,
        "elapsed_seconds": time.time() - started,
        "initial_metrics": initial_metrics,
    }
    temp = STATE.with_suffix(".tmp")
    torch.save(data, temp)
    os.replace(temp, STATE)


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    for directory in (OUTPUT.parent, RESULT.parent, STATE.parent, ROOT / "logs", ROOT / "benchmarks"):
        directory.mkdir(parents=True, exist_ok=True)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    codec = codec_state(PARENT)
    parent_ids = codec["pattern_ids"].clone()
    ids = parent_ids.clone()
    projection_index = row_position = 0
    changed = []
    top_rows = None
    initial_metrics = None
    if STATE.exists():
        resume = torch.load(STATE, map_location="cpu", weights_only=True)
        if resume["parent_sha256"] != sha256(PARENT) or resume["coverage"] != COVERAGE:
            raise RuntimeError("Resume checkpoint does not match parent or coverage")
        ids = resume["ids"]
        projection_index = int(resume["projection_index"])
        row_position = int(resume["row_position"])
        top_rows = resume["top_rows"]
        changed = list(resume["changed"])
        initial_metrics = resume["initial_metrics"]
        print(json.dumps({"event": "resume", "projection_index": projection_index, "row_position": row_position}), flush=True)

    weights = weights_from_ids(codec, ids)
    x_a, x_b = cache["x_a"], cache["x_b"]
    with torch.no_grad():
        gate_a, up_a, z_a, y_a = mlp(weights, x_a)
        gate_b, up_b, z_b, y_b = mlp(weights, x_b)
    current_metrics = metrics(z_a, z_b, y_a, y_b, cache)
    if initial_metrics is None:
        initial_metrics = dict(current_metrics)
    if top_rows is None:
        damage = (
            (z_a - cache["z_teacher_a"]).square().mean(0)
            * weights["down_proj"].norm(dim=0).square()
        ).view(-1, 4).mean(1)
        count = max(1, int(COVERAGE * damage.numel()))
        top_rows = torch.topk(damage, count).indices.tolist()

    yden_a = cache["y_teacher_a"].square().sum()
    yden_b = cache["y_teacher_b"].square().sum()
    total_sse_a = (y_a - cache["y_teacher_a"]).square().sum()
    total_sse_b = (y_b - cache["y_teacher_b"]).square().sum()
    original_blocks = {
        name: blockify(cache["weights"][name]) for name in ("gate_proj", "up_proj")
    }
    print(json.dumps({
        "event": "start",
        "parent": str(PARENT),
        "coverage": COVERAGE,
        "row_groups": len(top_rows),
        "initial": initial_metrics,
        "memory": memory_state(),
    }), flush=True)

    projections = ("gate_proj", "up_proj")
    searched_blocks = 0
    accepted_changes = 0
    for pindex in range(projection_index, len(projections)):
        projection = projections[pindex]
        start, end, rows_total, cols_total = codec["slices"][projection]
        block_cols = cols_total // 4
        first_row = row_position if pindex == projection_index else 0
        for position in range(first_row, len(top_rows)):
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < MIN_AVAILABLE_GB:
                save_state(ids, pindex, position, top_rows, changed, started, initial_metrics)
                raise MemoryError(f"Only {available_gb:.2f} GB available; saved and stopped")
            block_row = int(top_rows[position])
            row_slice = slice(block_row * 4, block_row * 4 + 4)
            down_columns = weights["down_proj"][:, row_slice]
            gram = down_columns.T @ down_columns
            for block_col in range(block_cols):
                searched_blocks += 1
                global_block = start + block_row * block_cols + block_col
                col_slice = slice(block_col * 4, block_col * 4 + 4)
                old_block = weights[projection][row_slice, col_slice].clone()
                scale = float(codec["scales"][global_block])
                target = original_blocks[projection][global_block - start] / scale
                activation_weight = x_a[:, col_slice].square().mean(0).repeat(4)
                candidate_ids = candidates(
                    codec["patterns"], target, activation_weight, int(ids[global_block])
                )
                error_a = y_a - cache["y_teacher_a"]
                error_b = y_b - cache["y_teacher_b"]
                error_projection_a = error_a @ down_columns
                error_projection_b = error_b @ down_columns
                best_id = int(ids[global_block])
                best_a, best_b = float(total_sse_a), float(total_sse_b)
                best = None
                for candidate_id in candidate_ids:
                    if candidate_id == best_id:
                        continue
                    new_block = (
                        codec["patterns"][candidate_id] * scale
                        + codec["residual"][global_block]
                    ).view(4, 4)
                    delta_a = x_a[:, col_slice] @ (new_block - old_block).T
                    delta_b = x_b[:, col_slice] @ (new_block - old_block).T
                    if projection == "gate_proj":
                        new_z_a = torch.nn.functional.silu(gate_a[:, row_slice] + delta_a) * up_a[:, row_slice]
                        new_z_b = torch.nn.functional.silu(gate_b[:, row_slice] + delta_b) * up_b[:, row_slice]
                    else:
                        new_z_a = torch.nn.functional.silu(gate_a[:, row_slice]) * (up_a[:, row_slice] + delta_a)
                        new_z_b = torch.nn.functional.silu(gate_b[:, row_slice]) * (up_b[:, row_slice] + delta_b)
                    dz_a = new_z_a - z_a[:, row_slice]
                    dz_b = new_z_b - z_b[:, row_slice]
                    delta_sse_a = 2 * (error_projection_a * dz_a).sum() + ((dz_a @ gram) * dz_a).sum()
                    delta_sse_b = 2 * (error_projection_b * dz_b).sum() + ((dz_b @ gram) * dz_b).sum()
                    score_a = float(total_sse_a + delta_sse_a)
                    score_b = float(total_sse_b + delta_sse_b)
                    if score_a < best_a and score_b <= best_b + 1e-7:
                        best_id, best_a, best_b = candidate_id, score_a, score_b
                        best = (new_block, delta_a, delta_b, new_z_a, new_z_b, dz_a, dz_b)
                if best is not None:
                    new_block, delta_a, delta_b, new_z_a, new_z_b, dz_a, dz_b = best
                    weights[projection][row_slice, col_slice] = new_block
                    ids[global_block] = best_id
                    changed.append(int(global_block))
                    accepted_changes += 1
                    if projection == "gate_proj":
                        gate_a[:, row_slice] += delta_a
                        gate_b[:, row_slice] += delta_b
                    else:
                        up_a[:, row_slice] += delta_a
                        up_b[:, row_slice] += delta_b
                    z_a[:, row_slice] = new_z_a
                    z_b[:, row_slice] = new_z_b
                    y_a += dz_a @ down_columns.T
                    y_b += dz_b @ down_columns.T
                    total_sse_a = torch.tensor(best_a)
                    total_sse_b = torch.tensor(best_b)

            completed = position + 1
            if completed % CHECKPOINT_ROWS == 0 or completed == len(top_rows):
                save_state(ids, pindex, completed, top_rows, changed, started, initial_metrics)
                patch_ids(PARENT, OUTPUT, ids)
                progress_metrics = metrics(z_a, z_b, y_a, y_b, cache)
                elapsed = time.time() - started
                print(json.dumps({
                    "event": "progress",
                    "projection": projection,
                    "row_groups_done": completed,
                    "row_groups_total": len(top_rows),
                    "searched_blocks": searched_blocks,
                    "accepted_changes": accepted_changes,
                    "blocks_per_second": searched_blocks / max(elapsed, 1e-9),
                    "metrics": progress_metrics,
                    "memory": memory_state(),
                }), flush=True)
        row_position = 0
        save_state(ids, pindex + 1, 0, top_rows, changed, started, initial_metrics)

    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")
    final_metrics = metrics(z_a, z_b, y_a, y_b, cache)
    result = {
        "experiment": LABEL,
        "parent": str(PARENT),
        "parent_sha256": sha256(PARENT),
        "coverage": COVERAGE,
        "threads": THREADS,
        "row_groups": len(top_rows),
        "searched_blocks": searched_blocks,
        "ids_changed": len(set(changed)),
        "initial": initial_metrics,
        "final": final_metrics,
        "bytes": OUTPUT.stat().st_size,
        "bpw": OUTPUT.stat().st_size * 8 / codec["meta"]["weights"],
        "output_sha256": sha256(OUTPUT),
        "elapsed_seconds": time.time() - started,
        "peak_memory_note": "See append-only progress log for RSS/available memory",
        "ppl": None,
        "ppl_note": "CPU worker does not claim PPL; requires strict full-model evaluation",
    }
    temp = RESULT.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    os.replace(temp, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
