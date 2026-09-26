from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_experiment6 import (functional_statistics, layer_of, projection_of,
                             quantize_hadamard, safe_perplexity)
from run_experiment9 import (Group, apply_masks, cheap_metrics, cheap_teacher,
                             empty_masks, final_diagnostics, protected_count,
                             split_group, top_global_masks)
from run_experiments import activation_importance, get_text_tokens, seed_everything, target_modules


STRICT_TARGET = 19.688


def parse_args():
    p = argparse.ArgumentParser(description="Strict block refinement for Experiment 9")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--evaluation-tokens", type=int, default=4096)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--gradient-sequences", type=int, default=8)
    p.add_argument("--candidates-per-level", type=int, default=12)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output", type=Path, default=Path("results/experiment9"))
    return p.parse_args()


def name_lookup(originals):
    return {(layer_of(name), projection_of(name)): name for name in originals}


def parse_logged_group(label, lookup, originals):
    head, *parts = label.split(":")
    layer = int(head[1:]); projection = parts[0]
    name = lookup[(layer, projection)]; rows, cols = originals[name].shape
    group = Group(name, 0, rows, 0, cols, "projection", f"L{layer}:{projection}")
    for item in parts[1:]:
        if item.startswith("r") and "-" in item:
            a, b = map(int, item[1:].split("-")); group = Group(name, a, b, 0, cols, "logged", label)
        elif item.startswith("c") and "-" in item:
            a, b = map(int, item[1:].split("-")); group = Group(name, 0, rows, a, b, "logged", label)
    return group


def reconstruct_mask(history, core, originals):
    masks = {name: value.clone() for name, value in core.items()}
    lookup = name_lookup(originals)
    rows = history[history.search.eq("fisher_core15")]
    for row in rows.itertuples():
        if row.event == "macro_add":
            # All coarse Fisher additions improved full PPL and were accepted.
            for label in str(row.groups).split("|"):
                group = parse_logged_group(label, lookup, originals)
                masks[group.name][group.r0:group.r1, group.c0:group.c1] = True
        elif row.event in {"backtrack_remove", "core_remove"} and bool(row.accepted):
            group = parse_logged_group(str(row.groups), lookup, originals)
            view = masks[group.name][group.r0:group.r1, group.c0:group.c1]
            core_view = core[group.name][group.r0:group.r1, group.c0:group.c1]
            if row.event == "backtrack_remove":
                view[~core_view] = False
            else:
                view[core_view] = False
    return masks


def lowest_density_blocks(masks, fisher, originals, block, limit):
    candidates = []
    # The boundary search identified q/gate/down as the only partially retained
    # families. Refine there rather than scanning protected all-one projections.
    for name, weight in originals.items():
        if projection_of(name) not in {"q_proj", "gate_proj", "down_proj"}:
            continue
        rows, cols = weight.shape
        for r in range(0, rows, block):
            for c in range(0, cols, block):
                r1, c1 = min(r + block, rows), min(c + block, cols)
                selected = masks[name][r:r1, c:c1]
                count = int(selected.sum())
                if count == 0:
                    continue
                score = float(fisher[name][r:r1, c:c1][selected].sum()) / count
                candidates.append((score, Group(name, r, r1, c, c1, f"block{block}",
                                                f"L{layer_of(name)}:{projection_of(name)}:b{block}:{r},{c}")))
    return [group for _, group in sorted(candidates, key=lambda item: item[0])[:limit]]


def save_packed_mask(path, masks, originals):
    arrays = {}
    metadata = []
    for index, name in enumerate(originals):
        flat = masks[name].numpy().reshape(-1)
        arrays[f"mask_{index}"] = np.packbits(flat, bitorder="little")
        metadata.append(f"{index}\t{name}\t{originals[name].shape[0]}\t{originals[name].shape[1]}\t{flat.size}")
    np.savez_compressed(path, **arrays)
    path.with_suffix(".tsv").write_text("index\tname\trows\tcols\tnumel\n" + "\n".join(metadata) + "\n", encoding="utf-8")


def main():
    args = parse_args(); seed_everything(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    modules = target_modules(model, set(range(len(model.model.layers))))
    originals = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
    total = sum(w.numel() for w in originals.values())
    calibration = get_text_tokens(tokenizer, "train", args.calibration_tokens, args.sequence_length)
    evaluation = get_text_tokens(tokenizer, "test", args.evaluation_tokens, args.sequence_length)
    _, grad2 = functional_statistics(model, modules, calibration[:args.gradient_sequences], device)
    fisher = {name: originals[name].float().square() * grad2[name] for name in originals}
    core = top_global_masks(fisher, originals, 15.0)
    base = {name: quantize_hadamard(weight.to(device), 128) for name, weight in originals.items()}
    history = pd.read_csv(args.output / "rescue_history.csv")
    masks = reconstruct_mask(history, core, originals)
    apply_masks(modules, originals, base, masks, device)
    start_ppl, _ = safe_perplexity(model, evaluation, device)
    print(f"RECONSTRUCTED protected={100*protected_count(masks)/total:.6f}% ppl={start_ppl:.6f}", flush=True)
    if start_ppl > STRICT_TARGET:
        raise RuntimeError("reconstructed mask is not safe under the strict target")

    new_rows = []
    eval_index = int(history.full_eval_index.max())
    for block in (128, 64, 32, 16):
        candidates = lowest_density_blocks(masks, fisher, originals, block, args.candidates_per_level)
        for group in candidates:
            view = masks[group.name][group.r0:group.r1, group.c0:group.c1]
            old = view.clone(); removed = int(view.sum())
            if removed == 0:
                continue
            view.fill_(False); apply_masks(modules, originals, base, masks, device)
            ppl, nll = safe_perplexity(model, evaluation, device); eval_index += 1
            accepted = ppl <= STRICT_TARGET
            count = protected_count(masks)
            print(f"STRICT {eval_index} {group.label} protected={100*count/total:.6f}% ppl={ppl:.6f} accepted={accepted}", flush=True)
            new_rows.append({"search": "fisher_core15", "mode": "strict_block_refinement",
                             "core_pct": 15.0, "event": "block_remove", "groups": group.label,
                             "level": group.level, "accepted": accepted, "protected": count,
                             "protected_pct": 100*count/total, "ppl": ppl, "nll": nll,
                             "elapsed_seconds": np.nan, "full_eval_index": eval_index})
            if not accepted:
                view.copy_(old); apply_masks(modules, originals, base, masks, device)

    final_ppl, final_nll = safe_perplexity(model, evaluation, device)
    final_count = protected_count(masks)
    history = pd.concat([history, pd.DataFrame(new_rows)], ignore_index=True)
    history.to_csv(args.output / "rescue_history.csv", index=False)
    summary = pd.read_csv(args.output / "search_summary.csv")
    selector = summary.search.eq("fisher_core15")
    core_overlap = sum(int((masks[n] & core[n]).sum()) for n in masks)
    summary.loc[selector, "final_ppl"] = final_ppl
    summary.loc[selector, "final_nll"] = final_nll
    summary.loc[selector, "protected"] = final_count
    summary.loc[selector, "protected_pct"] = 100*final_count/total
    summary.loc[selector, "core_retained_pct_points"] = 100*core_overlap/total
    summary.loc[selector, "core_removed_pct_points"] = 15 - 100*core_overlap/total
    summary.loc[selector, "additional_beyond_core_pct"] = 100*sum(int((masks[n] & ~core[n]).sum()) for n in masks)/total
    summary.loc[selector, "backtracking_removed_pct_points"] = 95.503493 - 100*final_count/total
    summary.loc[selector, "full_evaluations"] = int((history.search == "fisher_core15").sum())
    summary.to_csv(args.output / "search_summary.csv", index=False)

    allocation = pd.read_csv(args.output / "final_allocation.csv")
    allocation = allocation[~allocation.search.eq("fisher_core15")]
    rows = []
    total_selected = final_count
    for name, weight in originals.items():
        rows.append({"search": "fisher_core15", "layer": layer_of(name),
                     "projection": projection_of(name), "protected": int(masks[name].sum()),
                     "total": weight.numel(), "protected_rate": float(masks[name].float().mean()),
                     "share_of_selected": int(masks[name].sum())/total_selected,
                     "core_overlap": int((masks[name] & core[name]).sum())})
    pd.concat([allocation, pd.DataFrame(rows)], ignore_index=True).to_csv(args.output / "final_allocation.csv", index=False)
    save_packed_mask(args.output / "best_safe_mask.npz", masks, originals)
    print(f"FINAL protected={100*final_count/total:.6f}% ppl={final_ppl:.6f}", flush=True)


if __name__ == "__main__":
    main()
