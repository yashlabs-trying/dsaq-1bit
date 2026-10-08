"""Same-byte Down reassignment against the exact Layer-0 transition target."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch

from cpu_codec import blockify, codec_state, patch_ids, unblock
from e7b_cpu_reassign import candidates, memory_state


ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "cache/e9_layer_cache.pt"
PARENT = Path(os.environ.get("E9_PARENT", ROOT / "packed/E8g_down_polish_round5.packed"))
COVERAGE = float(os.environ.get("E9_COVERAGE", "0.05"))
LABEL = os.environ.get("E9_LABEL", "E9a_down_transition_5pct")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
THREADS = int(os.environ.get("E9_THREADS", "4"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {name: unblock(blocks[s:e], (r, c)) for name, (s, e, r, c) in codec["slices"].items()}


def rmsnorm(hidden, weight, eps):
    return hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + eps) * weight


def transition_metrics(h2_a, h2_b, cache):
    teacher_a, teacher_b = cache["h2_teacher_a"], cache["h2_teacher_b"]
    norm_weight, eps = cache["next_norm_weight"], cache["next_norm_eps"]
    norm_teacher_a = rmsnorm(teacher_a, norm_weight, eps)
    norm_teacher_b = rmsnorm(teacher_b, norm_weight, eps)
    norm_a, norm_b = rmsnorm(h2_a, norm_weight, eps), rmsnorm(h2_b, norm_weight, eps)
    lh2a = float((h2_a-teacher_a).square().sum()/teacher_a.square().sum())
    lh2b = float((h2_b-teacher_b).square().sum()/teacher_b.square().sum())
    lna = float((norm_a-norm_teacher_a).square().sum()/norm_teacher_a.square().sum())
    lnb = float((norm_b-norm_teacher_b).square().sum()/norm_teacher_b.square().sum())
    return {"L_h2_A": lh2a, "L_h2_B": lh2b, "E_h2_A": lh2a**0.5, "E_h2_B": lh2b**0.5,
            "L_next_norm_A": lna, "L_next_norm_B": lnb,
            "E_next_norm_A": lna**0.5, "E_next_norm_B": lnb**0.5}


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    if cache["parent_sha256"] != sha256(ROOT / "packed/E8g_down_polish_round5.packed"):
        raise RuntimeError("E9 cache is not anchored to frozen E8g")
    codec = codec_state(PARENT)
    ids = codec["pattern_ids"].clone()
    weights = weights_from_ids(codec, ids)
    x_a, x_b = cache["x_a"], cache["x_b"]
    gate_a, gate_b = x_a @ weights["gate_proj"].T, x_b @ weights["gate_proj"].T
    up_a, up_b = x_a @ weights["up_proj"].T, x_b @ weights["up_proj"].T
    z_a = torch.nn.functional.silu(gate_a) * up_a
    z_b = torch.nn.functional.silu(gate_b) * up_b
    y_a, y_b = z_a @ weights["down_proj"].T, z_b @ weights["down_proj"].T
    h2_a, h2_b = cache["h1_a"] + y_a, cache["h1_b"] + y_b
    initial = transition_metrics(h2_a, h2_b, cache)
    den_a, den_b = cache["h2_teacher_a"].square().sum(), cache["h2_teacher_b"].square().sum()
    total_a = (h2_a-cache["h2_teacher_a"]).square().sum()
    total_b = (h2_b-cache["h2_teacher_b"]).square().sum()
    damage = (h2_a-cache["h2_teacher_a"]).square().mean(0).view(-1,4).mean(1)
    top_rows = torch.topk(damage, max(1, int(COVERAGE*damage.numel()))).indices.tolist()
    start, _, _, cols_total = codec["slices"]["down_proj"]
    block_cols = cols_total // 4
    original = blockify(cache["original_weights"]["down_proj"])
    changed = []
    print(json.dumps({"event": "start", "parent": str(PARENT), "coverage": COVERAGE,
                      "row_groups": len(top_rows), "initial": initial}), flush=True)
    for position, block_row in enumerate(top_rows, 1):
        row_slice = slice(block_row*4, block_row*4+4)
        for block_col in range(block_cols):
            global_block = start + block_row*block_cols + block_col
            col_slice = slice(block_col*4, block_col*4+4)
            old = weights["down_proj"][row_slice, col_slice].clone()
            scale = float(codec["scales"][global_block])
            aw = z_a[:, col_slice].square().mean(0).repeat(4)
            choices = candidates(codec["patterns"], original[global_block-start]/scale,
                                 aw, int(ids[global_block]))
            error_a = h2_a[:, row_slice]-cache["h2_teacher_a"][:, row_slice]
            error_b = h2_b[:, row_slice]-cache["h2_teacher_b"][:, row_slice]
            best_id, best_a, best_b, best = int(ids[global_block]), float(total_a), float(total_b), None
            for candidate_id in choices:
                if candidate_id == best_id:
                    continue
                block = (codec["patterns"][candidate_id]*scale+codec["residual"][global_block]).view(4,4)
                delta_a, delta_b = z_a[:, col_slice]@(block-old).T, z_b[:, col_slice]@(block-old).T
                score_a = float(total_a + 2*(error_a*delta_a).sum()+delta_a.square().sum())
                score_b = float(total_b + 2*(error_b*delta_b).sum()+delta_b.square().sum())
                if score_a < best_a and score_b <= best_b + 1e-7:
                    best_id, best_a, best_b, best = candidate_id, score_a, score_b, (block, delta_a, delta_b)
            if best is not None:
                block, delta_a, delta_b = best
                weights["down_proj"][row_slice, col_slice] = block
                ids[global_block] = best_id
                changed.append(int(global_block))
                h2_a[:, row_slice] += delta_a
                h2_b[:, row_slice] += delta_b
                total_a, total_b = torch.tensor(best_a), torch.tensor(best_b)
        print(json.dumps({"event": "progress", "row_groups_done": position,
                          "row_groups_total": len(top_rows), "ids_changed": len(changed),
                          "metrics": transition_metrics(h2_a, h2_b, cache),
                          "memory": memory_state()}), flush=True)
    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")
    result = {"experiment": LABEL, "parent": str(PARENT), "parent_sha256": sha256(PARENT),
              "coverage": COVERAGE, "row_groups": len(top_rows),
              "searched_blocks": len(top_rows)*block_cols, "ids_changed": len(set(changed)),
              "initial": initial, "final": transition_metrics(h2_a, h2_b, cache),
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size*8/codec["meta"]["weights"],
              "output_sha256": sha256(OUTPUT), "elapsed_seconds": time.time()-started,
              "ppl": None}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    os.replace(temporary, RESULT)
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
