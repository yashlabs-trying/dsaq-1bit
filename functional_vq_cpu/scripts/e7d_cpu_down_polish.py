"""CPU functional reassignment of Down blocks using the current student z.

Each candidate is scored by its exact change to final MLP-output SSE.  A is
the search split; B is a strict non-worsening acceptance gate.  The packed
layout, scales, residuals, and codebook are unchanged.
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
from e7b_cpu_reassign import candidates, memory_state, metrics, mlp


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cache/layer0_functional_cache.pt"
PARENT = Path(os.environ.get("E7D_PARENT", ROOT / "packed/E7c_paired_gu_5pct.packed"))
COVERAGE = float(os.environ.get("E7D_COVERAGE", "0.05"))
LABEL = os.environ.get("E7D_LABEL", f"E7d_down_polish_{int(COVERAGE * 100)}pct")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
STATE = ROOT / "checkpoints" / f"{LABEL}_state.pt"
CHECKPOINT_ROWS = int(os.environ.get("E7D_CHECKPOINT_ROWS", "1"))
THREADS = int(os.environ.get("E7D_THREADS", "4"))
MIN_AVAILABLE_GB = float(os.environ.get("E7D_MIN_AVAILABLE_GB", "1.0"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {name: unblock(blocks[s:e], (r, c)) for name, (s, e, r, c) in codec["slices"].items()}


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
    codec = codec_state(PARENT)
    ids = codec["pattern_ids"].clone()
    position, changed, top_rows, initial_metrics = 0, [], None, None
    if STATE.exists():
        resume = torch.load(STATE, map_location="cpu", weights_only=True)
        if resume["parent_sha256"] != sha256(PARENT) or resume["coverage"] != COVERAGE:
            raise RuntimeError("Resume checkpoint does not match parent or coverage")
        ids, position, top_rows = resume["ids"], int(resume["position"]), resume["top_rows"]
        changed, initial_metrics = list(resume["changed"]), resume["initial_metrics"]
        print(json.dumps({"event": "resume", "position": position}), flush=True)

    weights = weights_from_ids(codec, ids)
    x_a, x_b = cache["x_a"], cache["x_b"]
    with torch.no_grad():
        _, _, z_a, y_a = mlp(weights, x_a)
        _, _, z_b, y_b = mlp(weights, x_b)
    current_metrics = metrics(z_a, z_b, y_a, y_b, cache)
    if initial_metrics is None:
        initial_metrics = dict(current_metrics)
    if top_rows is None:
        damage = (y_a - cache["y_teacher_a"]).square().mean(0).view(-1, 4).mean(1)
        top_rows = torch.topk(damage, max(1, int(COVERAGE * damage.numel()))).indices.tolist()

    start, _, _, cols_total = codec["slices"]["down_proj"]
    block_cols = cols_total // 4
    original = blockify(cache["weights"]["down_proj"])
    total_sse_a = (y_a - cache["y_teacher_a"]).square().sum()
    total_sse_b = (y_b - cache["y_teacher_b"]).square().sum()
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
            error_a = y_a[:, row_slice] - cache["y_teacher_a"][:, row_slice]
            error_b = y_b[:, row_slice] - cache["y_teacher_b"][:, row_slice]
            best_id, best_a, best_b, best = int(ids[global_block]), float(total_sse_a), float(total_sse_b), None
            for candidate_id in candidate_ids:
                if candidate_id == best_id:
                    continue
                new_block = (codec["patterns"][candidate_id] * scale + codec["residual"][global_block]).view(4, 4)
                dy_a = z_a[:, col_slice] @ (new_block - old_block).T
                dy_b = z_b[:, col_slice] @ (new_block - old_block).T
                score_a = float(total_sse_a + 2 * (error_a * dy_a).sum() + dy_a.square().sum())
                score_b = float(total_sse_b + 2 * (error_b * dy_b).sum() + dy_b.square().sum())
                if score_a < best_a and score_b <= best_b + 1e-7:
                    best_id, best_a, best_b, best = candidate_id, score_a, score_b, (new_block, dy_a, dy_b)
            if best is not None:
                new_block, dy_a, dy_b = best
                weights["down_proj"][row_slice, col_slice] = new_block
                ids[global_block] = best_id
                changed.append(int(global_block))
                accepted_changes += 1
                y_a[:, row_slice] += dy_a
                y_b[:, row_slice] += dy_b
                total_sse_a, total_sse_b = torch.tensor(best_a), torch.tensor(best_b)

        completed = row_position + 1
        if completed % CHECKPOINT_ROWS == 0 or completed == len(top_rows):
            save_state(ids, completed, top_rows, changed, started, initial_metrics)
            patch_ids(PARENT, OUTPUT, ids)
            elapsed = time.time() - started
            print(json.dumps({"event": "progress", "row_groups_done": completed,
                              "row_groups_total": len(top_rows), "searched_blocks": searched_blocks,
                              "accepted_changes": accepted_changes,
                              "blocks_per_second": searched_blocks / max(elapsed, 1e-9),
                              "metrics": metrics(z_a, z_b, y_a, y_b, cache),
                              "memory": memory_state()}), flush=True)

    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")
    result = {"experiment": LABEL, "parent": str(PARENT), "parent_sha256": sha256(PARENT),
              "coverage": COVERAGE, "threads": THREADS, "row_groups": len(top_rows),
              "searched_blocks": searched_blocks, "ids_changed": len(set(changed)),
              "initial": initial_metrics, "final": metrics(z_a, z_b, y_a, y_b, cache),
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size * 8 / codec["meta"]["weights"],
              "output_sha256": sha256(OUTPUT), "elapsed_seconds": time.time() - started,
              "ppl": None, "ppl_note": "CPU worker does not claim PPL; requires strict full-model evaluation"}
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
