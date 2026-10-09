"""E14: same-byte neuron-aligned joint Gate/Up/Down assignment search.

One move changes three existing 12-bit IDs together:
  * a Gate 4x4 block for neuron group J and input chunk C,
  * the matching Up block (same J, C), and
  * a Down block for output chunk O and input neuron group J.

The Cartesian product is scored only as a complete nonlinear move, including
the Gate/Up interaction and the Down cross-term.  Calibration A selects and B
must not worsen.  The packed format and byte count are unchanged.
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
from e10a_gate_up_polish import compute_downstream_gradient, module_map


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(os.environ.get("E14_PARENT", ROOT / "packed/E10a_gate_up_downstream.packed"))
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
E9C_CACHE = ROOT / "cache/e9c_cache.pt"
COVERAGE = float(os.environ.get("E14_COVERAGE", "0.05"))
INPUT_TOP = int(os.environ.get("E14_INPUT_TOP", "2"))
OUTPUT_TOP = int(os.environ.get("E14_OUTPUT_TOP", "2"))
LABEL = os.environ.get("E14_LABEL", "E14_neuron_joint_5pct")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
STATE = ROOT / "checkpoints" / f"{LABEL}_state.pt"
THREADS = int(os.environ.get("E14_THREADS", "4"))
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {name: unblock(blocks[s:e], (r, c)) for name, (s, e, r, c) in codec["slices"].items()}


def metrics(h2_a, h2_b, h2t_a, h2t_b, projected_a, projected_b, norm_a, norm_b):
    return {
        "L_h2_A": float((h2_a-h2t_a).square().sum()/h2t_a.square().sum()),
        "L_h2_B": float((h2_b-h2t_b).square().sum()/h2t_b.square().sum()),
        "L_sense_A": float(projected_a.square().sum()/norm_a),
        "L_sense_B": float(projected_b.square().sum()/norm_b),
    }


def make_block(codec, global_block, candidate_id):
    return (codec["patterns"][candidate_id] * float(codec["scales"][global_block])
            + codec["residual"][global_block]).view(4, 4)


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    for directory in (OUTPUT.parent, RESULT.parent, STATE.parent):
        directory.mkdir(parents=True, exist_ok=True)

    e9 = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    e9c = torch.load(E9C_CACHE, map_location="cpu", weights_only=True)
    codec = codec_state(PARENT)
    ids = codec["pattern_ids"].clone()
    weights = weights_from_ids(codec, ids)
    originals = e9["original_weights"]
    original_blocks = {name: blockify(originals[name]) for name in ("gate_proj", "up_proj", "down_proj")}
    x_a, x_b = e9["x_a"], e9["x_b"]
    h1_a, h1_b = e9["h1_a"], e9["h1_b"]
    h2t_a, h2t_b = e9["h2_teacher_a"], e9["h2_teacher_b"]
    h4t_a, h4t_b = e9c["h4_teacher_a"], e9c["h4_teacher_b"]

    gate_a, gate_b = x_a @ weights["gate_proj"].T, x_b @ weights["gate_proj"].T
    up_a, up_b = x_a @ weights["up_proj"].T, x_b @ weights["up_proj"].T
    z_a, z_b = F.silu(gate_a)*up_a, F.silu(gate_b)*up_b
    h2_a, h2_b = h1_a + z_a @ weights["down_proj"].T, h1_b + z_b @ weights["down_proj"].T

    print(json.dumps({"event": "load_model", "parent": str(PARENT)}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager").eval()
    decoded, _ = decode_weights(PARENT)
    with torch.no_grad():
        for name, value in decoded.items():
            module_map(model)[name].weight.copy_(value)
    grad_a, grad_b = compute_downstream_gradient(
        model, h2_a, h2_b, e9["token_ids_a"], e9["token_ids_b"], h4t_a, h4t_b)
    norm_a, norm_b = float(h4t_a.square().sum()), float(h4t_b.square().sum())
    projected_a = (grad_a*(h2_a-h2t_a)).sum(1)
    projected_b = (grad_b*(h2_b-h2t_b)).sum(1)
    initial = metrics(h2_a, h2_b, h2t_a, h2t_b, projected_a, projected_b, norm_a, norm_b)

    # Rank groups by sensitivity-weighted error against the original MLP on the
    # same student input distribution. Four neurons are the atomic 4x4 unit.
    with torch.no_grad():
        z_ref_a = F.silu(x_a @ originals["gate_proj"].T) * (x_a @ originals["up_proj"].T)
        z_ref_b = F.silu(x_b @ originals["gate_proj"].T) * (x_b @ originals["up_proj"].T)
    gdown_a, gdown_b = grad_a @ weights["down_proj"], grad_b @ weights["down_proj"]
    damage = ((gdown_a*(z_a-z_ref_a)).square().mean(0)
              + (gdown_b*(z_b-z_ref_b)).square().mean(0)).view(-1,4).mean(1)
    group_count = max(1, int(COVERAGE*damage.numel()))
    top_groups = torch.topk(damage, group_count).indices.tolist()

    g_start, _, _, g_cols = codec["slices"]["gate_proj"]
    u_start, _, _, u_cols = codec["slices"]["up_proj"]
    d_start, _, d_rows, d_cols = codec["slices"]["down_proj"]
    input_blocks, output_blocks = g_cols//4, d_rows//4
    down_block_cols = d_cols//4
    changed = []
    accepted_triples = evaluated_triples = locations = 0
    print(json.dumps({"event": "start", "coverage": COVERAGE,
                      "neuron_groups": group_count, "input_top": INPUT_TOP,
                      "output_top": OUTPUT_TOP, "initial": initial,
                      "memory": memory_state()}), flush=True)

    for group_position, block_row in enumerate(top_groups, 1):
        rows = slice(block_row*4, block_row*4+4)
        # Rank input chunks by current Gate+Up activation damage.
        input_scores = []
        for block_col in range(input_blocks):
            cols = slice(block_col*4, block_col*4+4)
            dg = x_a[:,cols] @ (weights["gate_proj"][rows,cols]-originals["gate_proj"][rows,cols]).T
            du = x_a[:,cols] @ (weights["up_proj"][rows,cols]-originals["up_proj"][rows,cols]).T
            input_scores.append(float(dg.square().mean()+du.square().mean()))
        selected_inputs = torch.topk(torch.tensor(input_scores), min(INPUT_TOP, input_blocks)).indices.tolist()

        # Rank output chunks by downstream-sensitive contribution of this group.
        contribution_a = z_a[:,rows] @ weights["down_proj"][:,rows].T
        contribution_b = z_b[:,rows] @ weights["down_proj"][:,rows].T
        output_score = ((grad_a*contribution_a).square().mean(0)
                        + (grad_b*contribution_b).square().mean(0)).view(-1,4).mean(1)
        selected_outputs = torch.topk(output_score, min(OUTPUT_TOP, output_blocks)).indices.tolist()

        for input_col in selected_inputs:
            cols = slice(input_col*4, input_col*4+4)
            gate_global = g_start + block_row*input_blocks + input_col
            up_global = u_start + block_row*input_blocks + input_col
            old_gate = weights["gate_proj"][rows,cols].clone()
            old_up = weights["up_proj"][rows,cols].clone()
            aw = x_a[:,cols].square().mean(0).repeat(4)
            gate_choices = candidates(codec["patterns"], original_blocks["gate_proj"][gate_global-g_start]/float(codec["scales"][gate_global]), aw, int(ids[gate_global]))
            up_choices = candidates(codec["patterns"], original_blocks["up_proj"][up_global-u_start]/float(codec["scales"][up_global]), aw, int(ids[up_global]))
            gate_options, up_options = [], []
            for cid in gate_choices:
                block = make_block(codec, gate_global, cid)
                gate_options.append((cid, block, x_a[:,cols]@(block-old_gate).T, x_b[:,cols]@(block-old_gate).T))
            for cid in up_choices:
                block = make_block(codec, up_global, cid)
                up_options.append((cid, block, x_a[:,cols]@(block-old_up).T, x_b[:,cols]@(block-old_up).T))

            for output_row in selected_outputs:
                locations += 1
                out_rows = slice(output_row*4, output_row*4+4)
                down_global = d_start + output_row*down_block_cols + block_row
                old_down = weights["down_proj"][out_rows,rows].clone()
                daw = z_a[:,rows].square().mean(0).repeat(4)
                down_choices = candidates(codec["patterns"], original_blocks["down_proj"][down_global-d_start]/float(codec["scales"][down_global]), daw, int(ids[down_global]))
                down_options = [(cid, make_block(codec, down_global, cid)) for cid in down_choices]
                old_down_columns = weights["down_proj"][:,rows].clone()
                gdown_group_a, gdown_group_b = grad_a @ old_down_columns, grad_b @ old_down_columns
                best_a, best_b, best = float(projected_a.square().sum()/norm_a), float(projected_b.square().sum()/norm_b), None

                for gid, gate_block, dg_a, dg_b in gate_options:
                    new_gate_a, new_gate_b = gate_a[:,rows]+dg_a, gate_b[:,rows]+dg_b
                    sg_a, sg_b = F.silu(new_gate_a), F.silu(new_gate_b)
                    for uid, up_block, du_a, du_b in up_options:
                        new_up_a, new_up_b = up_a[:,rows]+du_a, up_b[:,rows]+du_b
                        new_z_a, new_z_b = sg_a*new_up_a, sg_b*new_up_b
                        dz_a, dz_b = new_z_a-z_a[:,rows], new_z_b-z_b[:,rows]
                        base_proj_a = (gdown_group_a*dz_a).sum(1)
                        base_proj_b = (gdown_group_b*dz_b).sum(1)
                        for did, down_block in down_options:
                            evaluated_triples += 1
                            if gid == int(ids[gate_global]) and uid == int(ids[up_global]) and did == int(ids[down_global]):
                                continue
                            dd = down_block-old_down
                            cross_a, cross_b = new_z_a @ dd.T, new_z_b @ dd.T
                            delta_proj_a = base_proj_a + (grad_a[:,out_rows]*cross_a).sum(1)
                            delta_proj_b = base_proj_b + (grad_b[:,out_rows]*cross_b).sum(1)
                            score_a = float((projected_a+delta_proj_a).square().sum()/norm_a)
                            score_b = float((projected_b+delta_proj_b).square().sum()/norm_b)
                            if score_a < best_a and score_b <= best_b + 1e-10:
                                best_a, best_b = score_a, score_b
                                best = (gid, uid, did, gate_block, up_block, down_block,
                                        dg_a, dg_b, du_a, du_b, new_z_a, new_z_b,
                                        dz_a, dz_b, cross_a, cross_b, delta_proj_a, delta_proj_b)
                if best is not None:
                    (gid, uid, did, gate_block, up_block, down_block, dg_a, dg_b,
                     du_a, du_b, new_z_a, new_z_b, dz_a, dz_b, cross_a, cross_b,
                     delta_proj_a, delta_proj_b) = best
                    # h2 delta uses the pre-move Down columns plus the exact
                    # interaction of the new z with the changed Down block.
                    h2_a += dz_a @ old_down_columns.T
                    h2_b += dz_b @ old_down_columns.T
                    h2_a[:,out_rows] += cross_a
                    h2_b[:,out_rows] += cross_b
                    gate_a[:,rows] += dg_a; gate_b[:,rows] += dg_b
                    up_a[:,rows] += du_a; up_b[:,rows] += du_b
                    z_a[:,rows] = new_z_a; z_b[:,rows] = new_z_b
                    weights["gate_proj"][rows,cols] = gate_block
                    weights["up_proj"][rows,cols] = up_block
                    weights["down_proj"][out_rows,rows] = down_block
                    ids[gate_global], ids[up_global], ids[down_global] = gid, uid, did
                    changed.extend((int(gate_global), int(up_global), int(down_global)))
                    projected_a += delta_proj_a; projected_b += delta_proj_b
                    accepted_triples += 1

        if group_position % 2 == 0 or group_position == group_count:
            patch_ids(PARENT, OUTPUT, ids)
            state = {"parent_sha256": sha256(PARENT), "ids": ids,
                     "groups_done": group_position, "top_groups": top_groups,
                     "changed": changed, "accepted_triples": accepted_triples,
                     "evaluated_triples": evaluated_triples}
            torch.save(state, STATE.with_suffix(".tmp")); os.replace(STATE.with_suffix(".tmp"), STATE)
            print(json.dumps({"event": "progress", "groups_done": group_position,
                              "groups_total": group_count, "locations": locations,
                              "accepted_triples": accepted_triples,
                              "evaluated_triples": evaluated_triples,
                              "ids_changed": len(set(changed)),
                              "metrics": metrics(h2_a,h2_b,h2t_a,h2t_b,projected_a,projected_b,norm_a,norm_b),
                              "memory": memory_state()}), flush=True)

    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")
    result = {"experiment": LABEL, "parent": str(PARENT), "parent_sha256": sha256(PARENT),
              "coverage": COVERAGE, "neuron_groups": group_count,
              "input_top": INPUT_TOP, "output_top": OUTPUT_TOP,
              "locations": locations, "evaluated_triples": evaluated_triples,
              "accepted_triples": accepted_triples, "ids_changed": len(set(changed)),
              "initial": initial,
              "final": metrics(h2_a,h2_b,h2t_a,h2t_b,projected_a,projected_b,norm_a,norm_b),
              "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size*8/codec["meta"]["weights"],
              "output_sha256": sha256(OUTPUT), "elapsed_seconds": time.time()-started,
              "ppl": None}
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
