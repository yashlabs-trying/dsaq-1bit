"""E14B: neuron-aligned joint Gate/Up/Down search with exact CE gating.

This is a same-byte assignment experiment. Candidate triples are shortlisted by
the *signed* first-order language-model cross-entropy change, then accepted only
when exact downstream CE improves on calibration A and does not worsen on B.

Unlike E10C, the downstream path includes Qwen's final RMSNorm before lm_head.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, codec_state, patch_ids, unblock
from e7b_cpu_reassign import candidates, memory_state


ROOT = Path(__file__).resolve().parents[1]
PARENT = Path(os.environ.get("E14B_PARENT", ROOT / "packed/E10a_gate_up_downstream.packed"))
CACHE = ROOT / "cache/e9_layer_cache.pt"
COVERAGE = float(os.environ.get("E14B_COVERAGE", "0.002"))
GROUP_OFFSET = int(os.environ.get("E14B_GROUP_OFFSET", "0"))
INPUT_TOP = int(os.environ.get("E14B_INPUT_TOP", "1"))
OUTPUT_TOP = int(os.environ.get("E14B_OUTPUT_TOP", "1"))
SHORTLIST = int(os.environ.get("E14B_SHORTLIST", "8"))
LABEL = os.environ.get("E14B_LABEL", "E14b_joint_ce_proof")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
THREADS = int(os.environ.get("E14B_THREADS", "4"))
SEQ = 256
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
TOL = float(os.environ.get("E14B_B_TOL", "1e-7"))
SEGMENT_TOL = float(os.environ.get("E14B_SEGMENT_TOL", "0.0"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_from_ids(codec, ids):
    blocks = codec["patterns"][ids.long()] * codec["scales"][:, None] + codec["residual"]
    return {name: unblock(blocks[s:e], (r, c))
            for name, (s, e, r, c) in codec["slices"].items()}


def make_block(codec, global_block, candidate_id):
    return (codec["patterns"][candidate_id] * float(codec["scales"][global_block])
            + codec["residual"][global_block]).view(4, 4)


def downstream_logits(model, h2):
    """Run layers 1..23, final RMSNorm, and lm_head from Layer-0 output."""
    seq_len = h2.shape[0]
    pos_ids = torch.arange(seq_len, device=h2.device).unsqueeze(0)
    causal_mask = torch.full((seq_len, seq_len), torch.finfo(h2.dtype).min,
                             device=h2.device)
    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
    h = h2
    for layer_idx in range(1, len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        cos, sin = model.model.rotary_emb(h.unsqueeze(0), pos_ids)
        h_norm = layer.input_layernorm(h)
        attn = layer.self_attn(h_norm.unsqueeze(0), attention_mask=causal_mask,
                               position_embeddings=(cos, sin))[0].squeeze(0)
        h = h + attn
        residual = h
        h = residual + layer.mlp(layer.post_attention_layernorm(h))
    return model.lm_head(model.model.norm(h))


def exact_ce_segments(model, h2, token_ids):
    with torch.no_grad():
        values = []
        for start in range(0, token_ids.numel(), SEQ):
            hs = h2[start:start+SEQ]
            ts = token_ids[start:start+SEQ]
            if ts.numel() < 2:
                continue
            logits = downstream_logits(model, hs)
            values.append(float(F.cross_entropy(logits[:-1], ts[1:])))
        return values


def exact_ce(model, h2, token_ids):
    values = exact_ce_segments(model, h2, token_ids)
    return sum(values)/len(values)


def ce_gradient(model, h2, token_ids):
    h = h2.detach().clone().requires_grad_(True)
    losses = []
    token_count = 0
    for start in range(0, token_ids.numel(), SEQ):
        hs = h[start:start+SEQ]
        ts = token_ids[start:start+SEQ]
        if ts.numel() < 2:
            continue
        logits = downstream_logits(model, hs)
        losses.append(F.cross_entropy(logits[:-1], ts[1:], reduction="sum"))
        token_count += ts.numel()-1
    loss = torch.stack(losses).sum()/token_count
    return torch.autograd.grad(loss, h)[0].detach(), float(loss.detach())


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)

    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    codec = codec_state(PARENT)
    ids = codec["pattern_ids"].clone()
    weights = weights_from_ids(codec, ids)
    originals = cache["original_weights"]
    original_blocks = {name: blockify(originals[name])
                       for name in ("gate_proj", "up_proj", "down_proj")}
    x_a, x_b = cache["x_a"], cache["x_b"]
    h1_a, h1_b = cache["h1_a"], cache["h1_b"]
    tok_a, tok_b = cache["token_ids_a"], cache["token_ids_b"]

    gate_a, gate_b = x_a @ weights["gate_proj"].T, x_b @ weights["gate_proj"].T
    up_a, up_b = x_a @ weights["up_proj"].T, x_b @ weights["up_proj"].T
    z_a, z_b = F.silu(gate_a) * up_a, F.silu(gate_b) * up_b
    h2_a = h1_a + z_a @ weights["down_proj"].T
    h2_b = h1_b + z_b @ weights["down_proj"].T

    print(json.dumps({"event": "load_model", "memory": memory_state()}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager").eval()

    grad_a, ce_a = ce_gradient(model, h2_a, tok_a)
    grad_b, ce_b = ce_gradient(model, h2_b, tok_b)
    segments_a = exact_ce_segments(model, h2_a, tok_a)
    segments_b = exact_ce_segments(model, h2_b, tok_b)
    initial_ce = {"CE_A": ce_a, "CE_B": ce_b,
                  "PPL_A": float(torch.exp(torch.tensor(ce_a))),
                  "PPL_B": float(torch.exp(torch.tensor(ce_b))),
                  "segments_A": segments_a, "segments_B": segments_b}

    with torch.no_grad():
        z_ref_a = F.silu(x_a @ originals["gate_proj"].T) * (x_a @ originals["up_proj"].T)
        z_ref_b = F.silu(x_b @ originals["gate_proj"].T) * (x_b @ originals["up_proj"].T)
    gdown_a, gdown_b = grad_a @ weights["down_proj"], grad_b @ weights["down_proj"]
    damage = ((gdown_a * (z_a-z_ref_a)).abs().mean(0)
              + (gdown_b * (z_b-z_ref_b)).abs().mean(0)).view(-1, 4).mean(1)
    group_count = max(1, int(COVERAGE * damage.numel()))
    ranked = torch.topk(damage, min(damage.numel(), GROUP_OFFSET+group_count)).indices.tolist()
    top_groups = ranked[GROUP_OFFSET:GROUP_OFFSET+group_count]
    group_count = len(top_groups)

    g_start, _, _, g_cols = codec["slices"]["gate_proj"]
    u_start, _, _, _ = codec["slices"]["up_proj"]
    d_start, _, d_rows, d_cols = codec["slices"]["down_proj"]
    input_blocks, output_blocks, down_block_cols = g_cols//4, d_rows//4, d_cols//4
    accepted = evaluated_exact = evaluated_triples = locations = 0
    changed = []

    print(json.dumps({"event": "start", "coverage": COVERAGE,
                      "group_offset": GROUP_OFFSET,
                      "groups": group_count, "input_top": INPUT_TOP,
                      "output_top": OUTPUT_TOP, "shortlist": SHORTLIST,
                      "initial": initial_ce, "memory": memory_state()}), flush=True)

    for group_pos, block_row in enumerate(top_groups, 1):
        rows = slice(block_row*4, block_row*4+4)

        input_scores = []
        for block_col in range(input_blocks):
            cols = slice(block_col*4, block_col*4+4)
            dg = x_a[:, cols] @ (weights["gate_proj"][rows, cols]
                                  - originals["gate_proj"][rows, cols]).T
            du = x_a[:, cols] @ (weights["up_proj"][rows, cols]
                                  - originals["up_proj"][rows, cols]).T
            input_scores.append(float(dg.square().mean() + du.square().mean()))
        selected_inputs = torch.topk(torch.tensor(input_scores),
                                     min(INPUT_TOP, input_blocks)).indices.tolist()

        contribution_a = z_a[:, rows] @ weights["down_proj"][:, rows].T
        contribution_b = z_b[:, rows] @ weights["down_proj"][:, rows].T
        output_score = ((grad_a*contribution_a).abs().mean(0)
                        + (grad_b*contribution_b).abs().mean(0)).view(-1,4).mean(1)
        selected_outputs = torch.topk(output_score,
                                      min(OUTPUT_TOP, output_blocks)).indices.tolist()

        for input_col in selected_inputs:
            cols = slice(input_col*4, input_col*4+4)
            gate_global = g_start + block_row*input_blocks + input_col
            up_global = u_start + block_row*input_blocks + input_col
            old_gate = weights["gate_proj"][rows, cols].clone()
            old_up = weights["up_proj"][rows, cols].clone()
            aw = x_a[:, cols].square().mean(0).repeat(4)
            gate_ids = candidates(codec["patterns"],
                                  original_blocks["gate_proj"][gate_global-g_start]
                                  / float(codec["scales"][gate_global]), aw,
                                  int(ids[gate_global]))
            up_ids = candidates(codec["patterns"],
                                original_blocks["up_proj"][up_global-u_start]
                                / float(codec["scales"][up_global]), aw,
                                int(ids[up_global]))
            gate_options, up_options = [], []
            for cid in gate_ids:
                block = make_block(codec, gate_global, cid)
                gate_options.append((cid, block,
                    x_a[:,cols] @ (block-old_gate).T,
                    x_b[:,cols] @ (block-old_gate).T))
            for cid in up_ids:
                block = make_block(codec, up_global, cid)
                up_options.append((cid, block,
                    x_a[:,cols] @ (block-old_up).T,
                    x_b[:,cols] @ (block-old_up).T))

            for output_row in selected_outputs:
                locations += 1
                # A prior output-row move may have changed this shared Gate/Up
                # block. Rebuild deltas from the current decoded state so the
                # tracked h2 is exactly reproducible from the final packed IDs.
                old_gate = weights["gate_proj"][rows, cols].clone()
                old_up = weights["up_proj"][rows, cols].clone()
                gate_options, up_options = [], []
                for cid in gate_ids:
                    block = make_block(codec, gate_global, cid)
                    gate_options.append((cid, block,
                        x_a[:,cols] @ (block-old_gate).T,
                        x_b[:,cols] @ (block-old_gate).T))
                for cid in up_ids:
                    block = make_block(codec, up_global, cid)
                    up_options.append((cid, block,
                        x_a[:,cols] @ (block-old_up).T,
                        x_b[:,cols] @ (block-old_up).T))
                out_rows = slice(output_row*4, output_row*4+4)
                down_global = d_start + output_row*down_block_cols + block_row
                old_down = weights["down_proj"][out_rows, rows].clone()
                daw = z_a[:,rows].square().mean(0).repeat(4)
                down_ids = candidates(codec["patterns"],
                    original_blocks["down_proj"][down_global-d_start]
                    / float(codec["scales"][down_global]), daw,
                    int(ids[down_global]))
                down_options = [(cid, make_block(codec, down_global, cid))
                                for cid in down_ids]
                old_down_columns = weights["down_proj"][:, rows].clone()
                gdown_group_a = grad_a @ old_down_columns
                gdown_group_b = grad_b @ old_down_columns
                shortlist = []

                for gid, gate_block, dg_a, dg_b in gate_options:
                    new_gate_a, new_gate_b = gate_a[:,rows]+dg_a, gate_b[:,rows]+dg_b
                    sg_a, sg_b = F.silu(new_gate_a), F.silu(new_gate_b)
                    for uid, up_block, du_a, du_b in up_options:
                        new_up_a, new_up_b = up_a[:,rows]+du_a, up_b[:,rows]+du_b
                        new_z_a, new_z_b = sg_a*new_up_a, sg_b*new_up_b
                        dz_a, dz_b = new_z_a-z_a[:,rows], new_z_b-z_b[:,rows]
                        base_a = (gdown_group_a*dz_a).sum()
                        base_b = (gdown_group_b*dz_b).sum()
                        for did, down_block in down_options:
                            evaluated_triples += 1
                            if (gid == int(ids[gate_global]) and uid == int(ids[up_global])
                                    and did == int(ids[down_global])):
                                continue
                            dd = down_block-old_down
                            cross_a, cross_b = new_z_a @ dd.T, new_z_b @ dd.T
                            pred_a = float(base_a + (grad_a[:,out_rows]*cross_a).sum())
                            pred_b = float(base_b + (grad_b[:,out_rows]*cross_b).sum())
                            item = (pred_a, pred_b, gid, uid, did, gate_block,
                                    up_block, down_block, dg_a, dg_b, du_a, du_b,
                                    new_z_a, new_z_b, dz_a, dz_b, cross_a, cross_b)
                            shortlist.append(item)

                shortlist.sort(key=lambda item: item[0])
                shortlist = shortlist[:SHORTLIST]
                best = None
                best_ce_a = ce_a
                for item in shortlist:
                    evaluated_exact += 1
                    (*_, dz_a, dz_b, cross_a, cross_b) = item
                    candidate_h2_a = h2_a + dz_a @ old_down_columns.T
                    candidate_h2_b = h2_b + dz_b @ old_down_columns.T
                    candidate_h2_a[:,out_rows] += cross_a
                    candidate_h2_b[:,out_rows] += cross_b
                    test_segments_a = exact_ce_segments(model, candidate_h2_a, tok_a)
                    test_a = sum(test_segments_a)/len(test_segments_a)
                    if test_a >= best_ce_a - 1e-10:
                        continue
                    if any(new > old + SEGMENT_TOL
                           for new, old in zip(test_segments_a, segments_a)):
                        continue
                    test_segments_b = exact_ce_segments(model, candidate_h2_b, tok_b)
                    test_b = sum(test_segments_b)/len(test_segments_b)
                    if (test_b <= ce_b + TOL
                            and not any(new > old + SEGMENT_TOL
                                        for new, old in zip(test_segments_b, segments_b))):
                        best_ce_a = test_a
                        best = (item, candidate_h2_a, candidate_h2_b, test_b,
                                test_segments_a, test_segments_b)

                if best is not None:
                    item, h2_a, h2_b, ce_b, segments_a, segments_b = best
                    (pred_a, pred_b, gid, uid, did, gate_block, up_block,
                     down_block, dg_a, dg_b, du_a, du_b, new_z_a, new_z_b,
                     dz_a, dz_b, cross_a, cross_b) = item
                    ce_a = best_ce_a
                    gate_a[:,rows] += dg_a; gate_b[:,rows] += dg_b
                    up_a[:,rows] += du_a; up_b[:,rows] += du_b
                    z_a[:,rows] = new_z_a; z_b[:,rows] = new_z_b
                    weights["gate_proj"][rows,cols] = gate_block
                    weights["up_proj"][rows,cols] = up_block
                    weights["down_proj"][out_rows,rows] = down_block
                    ids[gate_global], ids[up_global], ids[down_global] = gid, uid, did
                    changed.extend((int(gate_global), int(up_global), int(down_global)))
                    accepted += 1

        # Refresh the exact CE gradient after each atomic neuron group.
        grad_a, ce_a = ce_gradient(model, h2_a, tok_a)
        grad_b, ce_b = ce_gradient(model, h2_b, tok_b)
        patch_ids(PARENT, OUTPUT, ids)
        print(json.dumps({"event": "progress", "group": group_pos,
                          "groups": group_count, "accepted": accepted,
                          "ids_changed": len(set(changed)),
                          "CE_A": ce_a, "CE_B": ce_b,
                          "segments_A": segments_a, "segments_B": segments_b,
                          "PPL_A": float(torch.exp(torch.tensor(ce_a))),
                          "PPL_B": float(torch.exp(torch.tensor(ce_b))),
                          "evaluated_exact": evaluated_exact,
                          "memory": memory_state()}), flush=True)

    patch_ids(PARENT, OUTPUT, ids)
    if OUTPUT.stat().st_size != PARENT.stat().st_size:
        raise RuntimeError("Packed size changed")
    final = {"CE_A": ce_a, "CE_B": ce_b,
             "PPL_A": float(torch.exp(torch.tensor(ce_a))),
             "PPL_B": float(torch.exp(torch.tensor(ce_b))),
             "segments_A": segments_a, "segments_B": segments_b}
    result = {"experiment": LABEL, "method": "joint_neuron_signed_ce_exact_gate",
              "parent": str(PARENT), "parent_sha256": sha256(PARENT),
              "coverage": COVERAGE, "group_offset": GROUP_OFFSET,
              "groups": group_count,
              "input_top": INPUT_TOP, "output_top": OUTPUT_TOP,
              "shortlist": SHORTLIST, "locations": locations,
              "evaluated_triples": evaluated_triples,
              "evaluated_exact": evaluated_exact, "accepted": accepted,
              "ids_changed": len(set(changed)), "initial": initial_ce,
              "final": final, "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size*8/codec["meta"]["weights"],
              "output_sha256": sha256(OUTPUT),
              "elapsed_seconds": time.time()-started, "strict_ppl": None}
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
