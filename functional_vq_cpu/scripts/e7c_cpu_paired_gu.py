"""Resumable CPU paired Gate/Up functional reassignment for E7c.

For each selected 4-neuron row group and input-column block, evaluate the
Cartesian product of fresh Gate and Up candidate IDs.  Pairs are selected by
exact final MLP output SSE on calibration A and accepted only when calibration
B does not worsen.  Packed layout and byte count remain unchanged.
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
CACHE_PATH = ROOT / "cache" / "layer0_functional_cache.pt"
PARENT = Path(os.environ.get("E7C_PARENT", ROOT / "packed/E7b_reassign_20pct.packed"))
COVERAGE = float(os.environ.get("E7C_COVERAGE", "0.05"))
LABEL = os.environ.get("E7C_LABEL", f"E7c_paired_gu_{int(COVERAGE * 100)}pct")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
STATE = ROOT / "checkpoints" / f"{LABEL}_state.pt"
CHECKPOINT_ROWS = int(os.environ.get("E7C_CHECKPOINT_ROWS", "2"))
THREADS = int(os.environ.get("E7C_THREADS", "4"))
MIN_AVAILABLE_GB = float(os.environ.get("E7C_MIN_AVAILABLE_GB", "1.0"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {
        name: unblock(blocks[start:end], (rows, cols))
        for name, (start, end, rows, cols) in codec["slices"].items()
    }


def save_state(ids, position, top_rows, changed, started, initial_metrics):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "parent": str(PARENT),
        "parent_sha256": sha256(PARENT),
        "coverage": COVERAGE,
        "ids": ids,
        "position": position,
        "top_rows": top_rows,
        "changed": changed,
        "elapsed_seconds": time.time() - started,
        "initial_metrics": initial_metrics,
    }
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
    position = 0
    changed = []
    top_rows = None
    initial_metrics = None
    if STATE.exists():
        resume = torch.load(STATE, map_location="cpu", weights_only=True)
        if resume["parent_sha256"] != sha256(PARENT) or resume["coverage"] != COVERAGE:
            raise RuntimeError("Resume checkpoint does not match parent or coverage")
        ids = resume["ids"]
        position = int(resume["position"])
        top_rows = resume["top_rows"]
        changed = list(resume["changed"])
        initial_metrics = resume["initial_metrics"]
        print(json.dumps({"event": "resume", "position": position}), flush=True)

    weights = weights_from_ids(codec, ids)
    x_a, x_b = cache["x_a"], cache["x_b"]
    with torch.no_grad():
        gate_a, up_a, z_a, y_a = mlp(weights, x_a)
        gate_b, up_b, z_b, y_b = mlp(weights, x_b)
    current_metrics = metrics(z_a, z_b, y_a, y_b, cache)
    if initial_metrics is None:
        initial_metrics = dict(current_metrics)
    if top_rows is None:
        # Rank remaining row groups by their contribution-weighted z error.
        damage = (
            (z_a - cache["z_teacher_a"]).square().mean(0)
            * weights["down_proj"].norm(dim=0).square()
        ).view(-1, 4).mean(1)
        count = max(1, int(COVERAGE * damage.numel()))
        top_rows = torch.topk(damage, count).indices.tolist()

    gate_start, _, _, cols_total = codec["slices"]["gate_proj"]
    up_start, _, _, up_cols = codec["slices"]["up_proj"]
    if cols_total != up_cols:
        raise RuntimeError("Gate and Up input widths differ")
    block_cols = cols_total // 4
    original_gate = blockify(cache["weights"]["gate_proj"])
    original_up = blockify(cache["weights"]["up_proj"])
    total_sse_a = (y_a - cache["y_teacher_a"]).square().sum()
    total_sse_b = (y_b - cache["y_teacher_b"]).square().sum()
    print(json.dumps({
        "event": "start",
        "parent": str(PARENT),
        "coverage": COVERAGE,
        "row_groups": len(top_rows),
        "candidate_pairs_per_block": 64,
        "initial": initial_metrics,
        "memory": memory_state(),
    }), flush=True)

    searched_pairs = 0
    searched_block_pairs = 0
    accepted_pairs = 0
    for row_position in range(position, len(top_rows)):
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < MIN_AVAILABLE_GB:
            save_state(ids, row_position, top_rows, changed, started, initial_metrics)
            raise MemoryError(f"Only {available_gb:.2f} GB available; saved and stopped")
        block_row = int(top_rows[row_position])
        row_slice = slice(block_row * 4, block_row * 4 + 4)
        down_columns = weights["down_proj"][:, row_slice]
        gram = down_columns.T @ down_columns
        for block_col in range(block_cols):
            searched_block_pairs += 1
            gate_global = gate_start + block_row * block_cols + block_col
            up_global = up_start + block_row * block_cols + block_col
            col_slice = slice(block_col * 4, block_col * 4 + 4)
            old_gate = weights["gate_proj"][row_slice, col_slice].clone()
            old_up = weights["up_proj"][row_slice, col_slice].clone()
            activation_weight = x_a[:, col_slice].square().mean(0).repeat(4)

            gate_scale = float(codec["scales"][gate_global])
            up_scale = float(codec["scales"][up_global])
            gate_ids = candidates(
                codec["patterns"], original_gate[gate_global - gate_start] / gate_scale,
                activation_weight, int(ids[gate_global]),
            )
            up_ids = candidates(
                codec["patterns"], original_up[up_global - up_start] / up_scale,
                activation_weight, int(ids[up_global]),
            )

            gate_options = []
            for candidate_id in gate_ids:
                block = (codec["patterns"][candidate_id] * gate_scale + codec["residual"][gate_global]).view(4, 4)
                gate_options.append((candidate_id, block, x_a[:, col_slice] @ (block - old_gate).T,
                                     x_b[:, col_slice] @ (block - old_gate).T))
            up_options = []
            for candidate_id in up_ids:
                block = (codec["patterns"][candidate_id] * up_scale + codec["residual"][up_global]).view(4, 4)
                up_options.append((candidate_id, block, x_a[:, col_slice] @ (block - old_up).T,
                                   x_b[:, col_slice] @ (block - old_up).T))

            error_projection_a = (y_a - cache["y_teacher_a"]) @ down_columns
            error_projection_b = (y_b - cache["y_teacher_b"]) @ down_columns
            best_a, best_b = float(total_sse_a), float(total_sse_b)
            best = None
            for gate_id, gate_block, delta_gate_a, delta_gate_b in gate_options:
                candidate_gate_a = gate_a[:, row_slice] + delta_gate_a
                candidate_gate_b = gate_b[:, row_slice] + delta_gate_b
                silu_a = torch.nn.functional.silu(candidate_gate_a)
                silu_b = torch.nn.functional.silu(candidate_gate_b)
                for up_id, up_block, delta_up_a, delta_up_b in up_options:
                    searched_pairs += 1
                    if gate_id == int(ids[gate_global]) and up_id == int(ids[up_global]):
                        continue
                    candidate_up_a = up_a[:, row_slice] + delta_up_a
                    candidate_up_b = up_b[:, row_slice] + delta_up_b
                    new_z_a = silu_a * candidate_up_a
                    new_z_b = silu_b * candidate_up_b
                    dz_a = new_z_a - z_a[:, row_slice]
                    dz_b = new_z_b - z_b[:, row_slice]
                    delta_sse_a = 2 * (error_projection_a * dz_a).sum() + ((dz_a @ gram) * dz_a).sum()
                    delta_sse_b = 2 * (error_projection_b * dz_b).sum() + ((dz_b @ gram) * dz_b).sum()
                    score_a = float(total_sse_a + delta_sse_a)
                    score_b = float(total_sse_b + delta_sse_b)
                    if score_a < best_a and score_b <= best_b + 1e-7:
                        best_a, best_b = score_a, score_b
                        best = (gate_id, up_id, gate_block, up_block, delta_gate_a,
                                delta_gate_b, delta_up_a, delta_up_b, new_z_a,
                                new_z_b, dz_a, dz_b)
            if best is not None:
                (gate_id, up_id, gate_block, up_block, delta_gate_a, delta_gate_b,
                 delta_up_a, delta_up_b, new_z_a, new_z_b, dz_a, dz_b) = best
                weights["gate_proj"][row_slice, col_slice] = gate_block
                weights["up_proj"][row_slice, col_slice] = up_block
                ids[gate_global] = gate_id
                ids[up_global] = up_id
                changed.extend((int(gate_global), int(up_global)))
                accepted_pairs += 1
                gate_a[:, row_slice] += delta_gate_a
                gate_b[:, row_slice] += delta_gate_b
                up_a[:, row_slice] += delta_up_a
                up_b[:, row_slice] += delta_up_b
                z_a[:, row_slice] = new_z_a
                z_b[:, row_slice] = new_z_b
                y_a += dz_a @ down_columns.T
                y_b += dz_b @ down_columns.T
                total_sse_a = torch.tensor(best_a)
                total_sse_b = torch.tensor(best_b)

        completed = row_position + 1
        if completed % CHECKPOINT_ROWS == 0 or completed == len(top_rows):
            save_state(ids, completed, top_rows, changed, started, initial_metrics)
            patch_ids(PARENT, OUTPUT, ids)
            elapsed = time.time() - started
            print(json.dumps({
                "event": "progress", "row_groups_done": completed,
                "row_groups_total": len(top_rows), "searched_block_pairs": searched_block_pairs,
                "searched_candidate_pairs": searched_pairs, "accepted_pairs": accepted_pairs,
                "block_pairs_per_second": searched_block_pairs / max(elapsed, 1e-9),
                "metrics": metrics(z_a, z_b, y_a, y_b, cache), "memory": memory_state(),
            }), flush=True)

    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")
    result = {
        "experiment": LABEL, "parent": str(PARENT), "parent_sha256": sha256(PARENT),
        "coverage": COVERAGE, "threads": THREADS, "row_groups": len(top_rows),
        "searched_block_pairs": searched_block_pairs, "searched_candidate_pairs": searched_pairs,
        "accepted_pairs": accepted_pairs, "ids_changed": len(set(changed)),
        "initial": initial_metrics, "final": metrics(z_a, z_b, y_a, y_b, cache),
        "bytes": OUTPUT.stat().st_size,
        "bpw": OUTPUT.stat().st_size * 8 / codec["meta"]["weights"],
        "output_sha256": sha256(OUTPUT), "elapsed_seconds": time.time() - started,
        "ppl": None, "ppl_note": "CPU worker does not claim PPL; requires strict full-model evaluation",
    }
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
