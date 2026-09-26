from __future__ import annotations

import argparse
import gc
import math
import platform
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from run_experiment6 import (capture_layers, functional_statistics, layer_of,
                             projection_of, quantize_hadamard, safe_perplexity)
from run_experiments import (activation_importance, dump_json, get_text_tokens,
                             seed_everything, target_modules)


TARGET_RATIO = 1.10
CORE_PCTS = (5.0, 10.0, 15.0)
MODES = ("fisher", "functional", "hybrid")


@dataclass(frozen=True)
class Group:
    name: str
    r0: int
    r1: int
    c0: int
    c1: int
    level: str
    label: str

    @property
    def area(self) -> int:
        return (self.r1 - self.r0) * (self.c1 - self.c0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 9: WaveFly hierarchical adaptive group rescue")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="main")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--evaluation-tokens", type=int, default=4096)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--gradient-sequences", type=int, default=8)
    p.add_argument("--direct-candidates", type=int, default=6)
    p.add_argument("--max-full-evals", type=int, default=360)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output", type=Path, default=Path("results/experiment9"))
    return p.parse_args()


def empty_masks(originals):
    return {name: torch.zeros(weight.shape, dtype=torch.bool) for name, weight in originals.items()}


def clone_masks(masks):
    return {name: value.clone() for name, value in masks.items()}


def protected_count(masks) -> int:
    return sum(int(mask.sum()) for mask in masks.values())


def group_view(tensor, group: Group):
    return tensor[group.r0:group.r1, group.c0:group.c1]


def group_change_count(masks, group: Group, value: bool) -> int:
    view = group_view(masks[group.name], group)
    return int((~view if value else view).sum())


def set_group(masks, group: Group, value: bool) -> int:
    changed = group_change_count(masks, group, value)
    group_view(masks[group.name], group).fill_(value)
    return changed


def projection_groups(originals):
    return [Group(name, 0, weight.shape[0], 0, weight.shape[1], "projection",
                  f"L{layer_of(name)}:{projection_of(name)}") for name, weight in originals.items()]


def split_group(group: Group, pieces: int, level: str):
    rows, cols = group.r1 - group.r0, group.c1 - group.c0
    # Attention heads and MLP intermediate channels live on the output dimension,
    # except o_proj/down_proj where the functional channels live on the input dimension.
    axis = 1 if projection_of(group.name) in {"o_proj", "down_proj"} else 0
    length = cols if axis else rows
    step = max(1, math.ceil(length / pieces))
    out = []
    for start in range(0, length, step):
        end = min(start + step, length)
        if axis:
            out.append(Group(group.name, group.r0, group.r1, group.c0 + start,
                             group.c0 + end, level, f"{group.label}:c{start}-{end}"))
        else:
            out.append(Group(group.name, group.r0 + start, group.r0 + end,
                             group.c0, group.c1, level, f"{group.label}:r{start}-{end}"))
    return out


def block_children(group: Group, block: int):
    out = []
    for r in range(group.r0, group.r1, block):
        for c in range(group.c0, group.c1, block):
            out.append(Group(group.name, r, min(r + block, group.r1), c,
                             min(c + block, group.c1), f"block{block}",
                             f"{group.label}:b{block}:{r},{c}"))
    return out


def top_global_masks(scores, originals, pct):
    total = sum(weight.numel() for weight in originals.values())
    k = max(1, round(total * pct / 100))
    # Find one exact global threshold without materializing a second full model-sized tensor.
    vector = torch.cat([scores[name].flatten() for name in originals])
    indices = torch.topk(vector, k, largest=True, sorted=False).indices
    selected = torch.zeros(vector.numel(), dtype=torch.bool)
    selected[indices] = True
    masks, cursor = {}, 0
    for name, weight in originals.items():
        masks[name] = selected[cursor:cursor + weight.numel()].view(weight.shape).clone()
        cursor += weight.numel()
    del vector, indices, selected
    return masks


def apply_masks(model_modules, originals, base, masks, device):
    with torch.no_grad():
        for name, module in model_modules.items():
            merged = base[name].clone()
            merged[masks[name]] = originals[name][masks[name]]
            module.weight.copy_(merged.to(device=device, dtype=module.weight.dtype))


def cheap_teacher(model, tokens, device):
    with torch.inference_mode():
        out = model(tokens.to(device), labels=tokens.to(device), use_cache=False,
                    output_hidden_states=True)
    return out.logits.detach().cpu().bfloat16(), out.hidden_states[-1].detach().cpu().bfloat16()


def cheap_metrics(model, tokens, teacher_logits, teacher_hidden, device):
    with torch.inference_mode():
        out = model(tokens.to(device), labels=tokens.to(device), use_cache=False,
                    output_hidden_states=True)
        logits = out.logits.float()
        target = torch.softmax(teacher_logits.to(device).float(), dim=-1)
        kl = F.kl_div(torch.log_softmax(logits, dim=-1), target, reduction="batchmean")
        hidden = out.hidden_states[-1].float()
        reference = teacher_hidden.to(device).float()
        nmse = (hidden - reference).square().sum() / reference.square().sum().clamp_min(1e-12)
    return {"nll": float(out.loss), "kl": float(kl), "hidden_nmse": float(nmse)}


def static_group_score(group, masks, fisher, quant_fisher, residual_damage, mode):
    available = ~group_view(masks[group.name], group)
    if not bool(available.any()):
        return -float("inf")
    f = float(group_view(fisher[group.name], group)[available].sum())
    if mode == "fisher":
        return f / max(int(available.sum()), 1)
    q = float(group_view(quant_fisher[group.name], group)[available].sum())
    r = float(group_view(residual_damage[group.name], group)[available].sum())
    return (math.log1p(max(q, 0)) + math.log1p(max(r, 0))) / math.sqrt(max(int(available.sum()), 1))


def normalize(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return x
    scale = np.nanstd(x)
    return (x - np.nanmedian(x)) / (scale if scale > 1e-12 else 1.0)


def direct_rank(model, modules, originals, base, masks, candidates, static_scores,
                cheap_tokens, teacher_logits, teacher_hidden, device, top_n, logs, tag):
    order = np.argsort(static_scores)[::-1][:top_n]
    baseline = cheap_metrics(model, cheap_tokens, teacher_logits, teacher_hidden, device)
    measurements = []
    for rank_index in order:
        group = candidates[int(rank_index)]
        before = group_view(masks[group.name], group).clone()
        set_group(masks, group, True)
        module = modules[group.name]
        merged = base[group.name].clone()
        merged[masks[group.name]] = originals[group.name][masks[group.name]]
        with torch.no_grad():
            module.weight.copy_(merged.to(device, module.weight.dtype))
        metric = cheap_metrics(model, cheap_tokens, teacher_logits, teacher_hidden, device)
        group_view(masks[group.name], group).copy_(before)
        merged = base[group.name].clone()
        merged[masks[group.name]] = originals[group.name][masks[group.name]]
        with torch.no_grad():
            module.weight.copy_(merged.to(device, module.weight.dtype))
        gains = (baseline["kl"] - metric["kl"], baseline["hidden_nmse"] - metric["hidden_nmse"],
                 baseline["nll"] - metric["nll"])
        measurements.append((int(rank_index), *gains))
        logs.append({"search": tag, "event": "trajectory_probe", "group": group.label,
                     "level": group.level, "accepted": False, "ppl": np.nan,
                     "kl_gain": gains[0], "hidden_gain": gains[1], "nll_gain": gains[2]})
    if not measurements:
        return list(np.argsort(static_scores)[::-1])
    composite = normalize([m[1] for m in measurements]) + normalize([m[2] for m in measurements]) + normalize([m[3] for m in measurements])
    direct_order = [measurements[i][0] for i in np.argsort(composite)[::-1]]
    rest = [int(i) for i in np.argsort(static_scores)[::-1] if int(i) not in set(direct_order)]
    return direct_order + rest


def distribution_rows(tag, masks, originals, core_masks):
    total_selected = protected_count(masks)
    rows = []
    buckets = defaultdict(lambda: [0, 0, 0])
    for name, weight in originals.items():
        key = (layer_of(name), projection_of(name))
        buckets[key][0] += int(masks[name].sum())
        buckets[key][1] += weight.numel()
        buckets[key][2] += int((masks[name] & core_masks[name]).sum())
    for (layer, projection), (selected, total, core_overlap) in buckets.items():
        rows.append({"search": tag, "layer": layer, "projection": projection,
                     "protected": selected, "total": total,
                     "protected_rate": selected / total,
                     "share_of_selected": selected / max(total_selected, 1),
                     "core_overlap": core_overlap})
    return rows


def run_search(mode, core_pct, model, modules, originals, base, core_masks, fisher,
               quant_fisher, residual_damage, evaluation, cheap_tokens,
               teacher_logits, teacher_hidden, device, total_weights, target_ppl,
               state, args):
    tag = f"{mode}_core{core_pct:g}"
    masks = clone_masks(core_masks)
    apply_masks(modules, originals, base, masks, device)
    start = time.time()

    def full_eval(event, groups=(), accepted=True):
        if state["full_evals"] >= args.max_full_evals:
            raise RuntimeError("maximum full-perplexity evaluations reached")
        ppl, nll = safe_perplexity(model, evaluation, device)
        state["full_evals"] += 1
        count = protected_count(masks)
        state["history"].append({"search": tag, "mode": mode, "core_pct": core_pct,
                                 "event": event, "groups": "|".join(g.label for g in groups),
                                 "level": groups[0].level if groups else "core",
                                 "accepted": accepted, "protected": count,
                                 "protected_pct": 100 * count / total_weights,
                                 "ppl": ppl, "nll": nll, "elapsed_seconds": time.time() - start,
                                 "full_eval_index": state["full_evals"]})
        print(f"EVAL {state['full_evals']:03d} {tag} {event} "
              f"protected={100 * count / total_weights:.4f}% ppl={ppl:.6g}", flush=True)
        return ppl, nll

    ppl, _ = full_eval("starting_core")
    start_ppl = ppl
    accepted_groups = []
    candidates = projection_groups(originals)
    score_mode = "fisher" if mode == "fisher" else "functional"
    # Projection candidates are disjoint. Activating one cannot alter the static
    # score of another, so cache these model-sized reductions once per search.
    cached_scores = {group: static_group_score(group, masks, fisher, quant_fisher,
                                               residual_damage, score_mode)
                     for group in candidates}
    iteration = 0
    while ppl > target_ppl and protected_count(masks) < total_weights:
        iteration += 1
        remaining = [g for g in candidates if group_change_count(masks, g, True) > 0]
        if not remaining:
            break
        scores = [cached_scores[g] for g in remaining]
        use_direct = mode == "functional" or (mode == "hybrid" and ppl < 30)
        if use_direct:
            order = direct_rank(model, modules, originals, base, masks, remaining, scores,
                                cheap_tokens, teacher_logits, teacher_hidden, device,
                                args.direct_candidates, state["group_log"], tag)
        else:
            order = list(np.argsort(scores)[::-1])
        increment = 5.0 if ppl > 80 else (2.0 if ppl > 30 else (1.0 if ppl > 22 else 0.5))
        target_add = round(total_weights * increment / 100)
        proposal, added = [], 0
        for index in order:
            group = remaining[index]
            change = group_change_count(masks, group, True)
            if change <= 0:
                continue
            proposal.append(group); added += change
            if added >= target_add:
                break
        previous = ppl
        for group in proposal:
            set_group(masks, group, True)
        apply_masks(modules, originals, base, masks, device)
        candidate_ppl, _ = full_eval("macro_add", proposal)
        accept = candidate_ppl < previous
        if accept:
            ppl = candidate_ppl; accepted_groups.extend(proposal)
        else:
            for group in proposal:
                set_group(masks, group, False)
            apply_masks(modules, originals, base, masks, device)
            # Backtrack the failed batch and certify its members separately.
            for group in proposal:
                set_group(masks, group, True); apply_masks(modules, originals, base, masks, device)
                trial, _ = full_eval("split_add", [group])
                if trial < ppl:
                    ppl = trial; accepted_groups.append(group)
                else:
                    set_group(masks, group, False); apply_masks(modules, originals, base, masks, device)
                    state["group_log"].append({"search": tag, "event": "split_add", "group": group.label,
                                               "level": group.level, "accepted": False, "ppl": trial})
        if iteration > 40:
            break

    reached = ppl <= target_ppl
    # Refine the most recent additions only after a certified safe state exists.
    backtrack_before = protected_count(masks)
    # Coarse ablation paths stop at their first certified safe state. The single
    # best path is refined after all nine comparisons finish.
    if reached and False:
        refine_parents = list(reversed(accepted_groups[-8:]))
        for parent in refine_parents:
            for child in split_group(parent, 8, "head_or_channel"):
                if not bool(group_view(masks[child.name], child).any()):
                    continue
                old = group_view(masks[child.name], child).clone()
                set_group(masks, child, False); apply_masks(modules, originals, base, masks, device)
                trial, _ = full_eval("backtrack_remove", [child])
                if trial <= target_ppl:
                    ppl = trial
                else:
                    group_view(masks[child.name], child).copy_(old)
                    apply_masks(modules, originals, base, masks, device)
        # Revisit weak projection-level pieces of the original core.
        core_groups = projection_groups(originals)
        core_scores = []
        for group in core_groups:
            overlap = group_view(masks[group.name] & core_masks[group.name], group)
            score = float(group_view(fisher[group.name], group)[overlap].sum()) if bool(overlap.any()) else float("inf")
            core_scores.append(score)
        for idx in np.argsort(core_scores)[:16]:
            parent = core_groups[int(idx)]
            for child in split_group(parent, 8, "core_head_or_channel"):
                region = group_view(masks[child.name] & core_masks[child.name], child)
                if not bool(region.any()):
                    continue
                old = group_view(masks[child.name], child).clone()
                core_region = group_view(core_masks[child.name], child)
                group_view(masks[child.name], child)[core_region] = False
                apply_masks(modules, originals, base, masks, device)
                trial, _ = full_eval("core_remove", [child])
                if trial <= target_ppl:
                    ppl = trial
                else:
                    group_view(masks[child.name], child).copy_(old)
                    apply_masks(modules, originals, base, masks, device)

    final_count = protected_count(masks)
    result = {"search": tag, "mode": mode, "core_pct": core_pct, "starting_ppl": start_ppl,
              "reached_target": reached, "final_ppl": ppl, "protected": final_count,
              "protected_pct": 100 * final_count / total_weights,
              "additional_beyond_core_pct": 100 * int(sum((masks[n] & ~core_masks[n]).sum() for n in masks)) / total_weights,
              "core_retained_pct_points": 100 * int(sum((masks[n] & core_masks[n]).sum() for n in masks)) / total_weights,
              "core_removed_pct_points": core_pct - 100 * int(sum((masks[n] & core_masks[n]).sum() for n in masks)) / total_weights,
              "backtracking_removed_pct_points": 100 * (backtrack_before - final_count) / total_weights,
              "runtime_seconds": time.time() - start, "full_evaluations": state["full_evals"]}
    return result, masks


def refine_best(tag, model, modules, originals, base, masks, core_masks, fisher,
                residual_damage, evaluation, device, total_weights, target_ppl,
                state, args):
    apply_masks(modules, originals, base, masks, device)
    start_count = protected_count(masks)
    current_ppl, _ = safe_perplexity(model, evaluation, device)

    def evaluate(event, group_a, group_b=None):
        if state["full_evals"] >= args.max_full_evals:
            return float("inf")
        ppl, nll = safe_perplexity(model, evaluation, device)
        state["full_evals"] += 1
        count = protected_count(masks)
        groups = group_a.label + (f" -> {group_b.label}" if group_b else "")
        state["history"].append({"search": tag, "mode": "best_refinement", "core_pct": np.nan,
                                 "event": event, "groups": groups, "level": group_a.level,
                                 "accepted": ppl <= target_ppl, "protected": count,
                                 "protected_pct": 100 * count / total_weights, "ppl": ppl,
                                 "nll": nll, "elapsed_seconds": np.nan,
                                 "full_eval_index": state["full_evals"]})
        print(f"EVAL {state['full_evals']:03d} {tag} {event} "
              f"protected={100*count/total_weights:.4f}% ppl={ppl:.6g}", flush=True)
        return ppl

    parents = projection_groups(originals)
    # First refine the weakest selected non-core projections into functional
    # head/channel groups.
    ranked_extra = []
    for parent in parents:
        region = group_view(masks[parent.name] & ~core_masks[parent.name], parent)
        if bool(region.any()):
            score = float(group_view(fisher[parent.name], parent)[region].sum()) / max(int(region.sum()), 1)
            ranked_extra.append((score, parent))
    for _, parent in sorted(ranked_extra, key=lambda x: x[0])[:4]:
        for child in split_group(parent, 8, "head_or_channel"):
            view = group_view(masks[child.name], child)
            removable = view & ~group_view(core_masks[child.name], child)
            if not bool(removable.any()):
                continue
            old = view.clone(); view[removable] = False
            apply_masks(modules, originals, base, masks, device)
            trial = evaluate("backtrack_remove", child)
            if trial <= target_ppl:
                current_ppl = trial
            else:
                view.copy_(old); apply_masks(modules, originals, base, masks, device)

    # Then revisit the weakest pieces inherited from the starting Fisher core.
    ranked_core = []
    for parent in parents:
        region = group_view(masks[parent.name] & core_masks[parent.name], parent)
        if bool(region.any()):
            score = float(group_view(fisher[parent.name], parent)[region].sum()) / max(int(region.sum()), 1)
            ranked_core.append((score, parent))
    for _, parent in sorted(ranked_core, key=lambda x: x[0])[:4]:
        for child in split_group(parent, 8, "core_head_or_channel"):
            view = group_view(masks[child.name], child)
            removable = view & group_view(core_masks[child.name], child)
            if not bool(removable.any()):
                continue
            old = view.clone(); view[removable] = False
            apply_masks(modules, originals, base, masks, device)
            trial = evaluate("core_remove", child)
            if trial <= target_ppl:
                current_ppl = trial
            else:
                view.copy_(old); apply_masks(modules, originals, base, masks, device)

    # Limited unequal-cost swap search between functional groups.
    children = [child for parent in parents for child in split_group(parent, 8, "swap_group")]
    protected, unprotected = [], []
    for child in children:
        view = group_view(masks[child.name], child)
        if bool(view.any()):
            score = float(group_view(fisher[child.name], child)[view].sum()) / max(int(view.sum()), 1)
            protected.append((score, child))
        if bool((~view).any()):
            score = float(group_view(residual_damage[child.name], child)[~view].sum()) / max(int((~view).sum()), 1)
            unprotected.append((score, child))
    protected.sort(key=lambda x: x[0]); unprotected.sort(key=lambda x: x[0], reverse=True)
    attempts = 0
    for _, remove in protected:
        if attempts >= 4:
            break
        remove_count = group_change_count(masks, remove, False)
        add = next((g for _, g in unprotected if g.name != remove.name and
                    group_change_count(masks, g, True) < remove_count), None)
        if add is None:
            continue
        old_remove = group_view(masks[remove.name], remove).clone()
        old_add = group_view(masks[add.name], add).clone()
        set_group(masks, remove, False); set_group(masks, add, True)
        apply_masks(modules, originals, base, masks, device)
        trial = evaluate("swap", remove, add); attempts += 1
        if trial <= target_ppl:
            current_ppl = trial
        else:
            group_view(masks[remove.name], remove).copy_(old_remove)
            group_view(masks[add.name], add).copy_(old_add)
            apply_masks(modules, originals, base, masks, device)
    return current_ppl, 100 * (start_count - protected_count(masks)) / total_weights


def final_diagnostics(model, tokens, teacher_outputs, device):
    _, outputs, _ = capture_layers(model, tokens, device)
    rows = []
    for layer in sorted(outputs):
        q, t = outputs[layer].float(), teacher_outputs[layer].float()
        rows.append({"layer": layer, "hidden_nmse": float((q-t).square().sum()/t.square().sum().clamp_min(1e-12)),
                     "hidden_cosine": float(F.cosine_similarity(q.flatten(), t.flatten(), dim=0))})
    return rows


def plots(out, history, summaries, allocation, target):
    plots_dir = out / "plots"; plots_dir.mkdir(exist_ok=True)
    h = history[history.ppl.notna()]
    plt.figure(figsize=(10, 6))
    for tag, frame in h.groupby("search"):
        plt.plot(frame.protected_pct, frame.ppl, marker="o", ms=3, label=tag)
    plt.axhline(target, color="black", ls="--", label=f"target {target:.3f}")
    plt.yscale("log"); plt.xlabel("Protected residual (%)"); plt.ylabel("Perplexity")
    plt.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig(plots_dir / "ppl_vs_protected.png", dpi=180); plt.close()
    best = summaries.sort_values(["reached_target", "protected_pct"], ascending=[False, True]).iloc[0]
    a = allocation[allocation.search == best.search].groupby("projection").protected.sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5)); a.plot.bar(); plt.ylabel("Protected weights"); plt.tight_layout()
    plt.savefig(plots_dir / "final_by_projection.png", dpi=180); plt.close()


def write_report(out, bf_ppl, target, summaries, history, allocation, drift, total_weights):
    reached = summaries[summaries.reached_target]
    best = reached.sort_values(["protected_pct", "final_ppl"]).iloc[0] if len(reached) else summaries.sort_values("final_ppl").iloc[0]
    best_alloc = allocation[allocation.search == best.search]
    projections = best_alloc.groupby("projection").protected.sum().sort_values(ascending=False)
    mode_best = summaries.groupby("mode").final_ppl.min()
    core_best = summaries.sort_values(["reached_target", "protected_pct", "final_ppl"], ascending=[False, True, True]).groupby("core_pct").first()
    accepted = history[(history.search == best.search) & history.accepted & history.ppl.notna()]
    biggest = accepted.assign(gain=accepted.ppl.shift(1)-accepted.ppl).sort_values("gain", ascending=False).head(1)
    biggest_level = biggest.level.iloc[0] if len(biggest) else "none"
    text = f"""# Experiment 9 — WaveFly Hierarchical Adaptive Group Rescue

BF16 PPL was **{bf_ppl:.6g}** and the certified 10% degradation ceiling was **{target:.6g}**.

## Search summary

{summaries.to_markdown(index=False)}

## Required answers

**Q1. Can hierarchical rescue reach PPL <= {target:.3f}?** {'YES' if len(reached) else 'NO'}.

**Q2. Minimum total protected percentage found:** {best.protected_pct:.6g}% (PPL {best.final_ppl:.6g}).

**Q3. Additional protection beyond its starting core:** {best.additional_beyond_core_pct:.6g} percentage points.

**Q4. Hierarchy level with the largest recorded recovery:** {biggest_level}.

**Q5. Are layers/projections more useful than individual blocks?** The build-up phase used projection groups; fine groups were used only for boundary backtracking. See `rescue_history.csv` for measured gains.

**Q6. Dominant projection families:** {', '.join(f'{k} ({int(v):,})' for k,v in projections.head(4).items())}.

**Q7. Does adaptive recomputation outperform fixed ranking?** {'YES' if mode_best.get('hybrid', np.inf) < mode_best.get('fisher', np.inf) else 'NO'} by best achieved PPL.

**Q8. Does functional scoring outperform Fisher scoring?** {'YES' if mode_best.get('functional', np.inf) < mode_best.get('fisher', np.inf) else 'NO'} by best achieved PPL.

**Q9. Backtracking reduction:** {best.backtracking_removed_pct_points:.6g} percentage points.

**Q10. CORE removed:** {best.core_removed_pct_points:.6g} percentage points from its {best.core_pct:g}% starting core.

**Q11. Is the final set structured?** Protection is reported by layer/projection in `final_allocation.csv`; the search explicitly selected projections and head/channel groups.

**Q12. Best starting core:** {best.core_pct:g}% Fisher.

**Q13. Exact protected percentage found under the ceiling:** {best.protected_pct:.6g}%.

This is a search result, not a proof of global minimality. Every accepted macro step and every final removal was judged by actual WikiText-2 perplexity. Cheap KL, hidden-state, and NLL probes only ranked proposals.
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main():
    args = parse_args(); seed_everything(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    modules = target_modules(model, set(range(len(model.model.layers))))
    originals = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
    total_weights = sum(weight.numel() for weight in originals.values())
    calibration = get_text_tokens(tokenizer, "train", args.calibration_tokens, args.sequence_length)
    evaluation = get_text_tokens(tokenizer, "test", args.evaluation_tokens, args.sequence_length)
    diag_tokens = calibration[:1]
    cheap_tokens = calibration[:1, :128]
    bf_ppl, bf_nll = safe_perplexity(model, evaluation, device)
    target_ppl = TARGET_RATIO * bf_ppl
    _, teacher_outputs, _ = capture_layers(model, diag_tokens, device)
    teacher_logits, teacher_hidden = cheap_teacher(model, cheap_tokens, device)
    activation_rms2 = activation_importance(model, modules, calibration, device)
    _, grad2 = functional_statistics(model, modules, calibration[:args.gradient_sequences], device)
    fisher = {name: originals[name].float().square() * grad2[name] for name in originals}
    base = {}
    for i, name in enumerate(originals, 1):
        print(f"Hadamard {i}/{len(originals)} {name}", flush=True)
        base[name] = quantize_hadamard(originals[name].to(device), args.block_size)
    quant_fisher = {}; residual_damage = {}
    for name in originals:
        delta2 = (originals[name].float() - base[name].float()).square()
        quant_fisher[name] = delta2 * grad2[name]
        residual_damage[name] = delta2 * activation_rms2[name][None]
    del grad2, activation_rms2; gc.collect(); torch.cuda.empty_cache()

    core_masks = {pct: top_global_masks(fisher, originals, pct) for pct in CORE_PCTS}
    state = {"full_evals": 0, "history": [], "group_log": []}
    summaries, masks_by_search, allocation_rows = [], {}, []
    try:
        for core_pct in CORE_PCTS:
            for mode in MODES:
                print(f"SEARCH {mode} core={core_pct:g}%", flush=True)
                result, masks = run_search(mode, core_pct, model, modules, originals, base,
                                           core_masks[core_pct], fisher, quant_fisher,
                                           residual_damage, evaluation, cheap_tokens,
                                           teacher_logits, teacher_hidden, device,
                                           total_weights, target_ppl, state, args)
                summaries.append(result); masks_by_search[result["search"]] = clone_masks(masks)
                allocation_rows.extend(distribution_rows(result["search"], masks, originals, core_masks[core_pct]))
                del masks; gc.collect()
    except RuntimeError as exc:
        if "maximum full-perplexity" not in str(exc):
            raise
        print(str(exc), flush=True)

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        raise RuntimeError("No Experiment 9 search completed")
    reached = summary_df[summary_df.reached_target]
    best_row = (reached.sort_values(["protected_pct", "final_ppl"]).iloc[0]
                if len(reached) else summary_df.sort_values("final_ppl").iloc[0])
    best_masks = masks_by_search[best_row.search]
    if bool(best_row.reached_target):
        refined_ppl, removed = refine_best(best_row.search, model, modules, originals, base,
                                            best_masks, core_masks[float(best_row.core_pct)],
                                            fisher, residual_damage, evaluation, device,
                                            total_weights, target_ppl, state, args)
        final_count = protected_count(best_masks)
        core_overlap = int(sum((best_masks[n] & core_masks[float(best_row.core_pct)][n]).sum()
                               for n in best_masks))
        selector = summary_df.search == best_row.search
        summary_df.loc[selector, "final_ppl"] = refined_ppl
        summary_df.loc[selector, "protected"] = final_count
        summary_df.loc[selector, "protected_pct"] = 100 * final_count / total_weights
        summary_df.loc[selector, "additional_beyond_core_pct"] = 100 * int(sum(
            (best_masks[n] & ~core_masks[float(best_row.core_pct)][n]).sum() for n in best_masks)) / total_weights
        summary_df.loc[selector, "core_retained_pct_points"] = 100 * core_overlap / total_weights
        summary_df.loc[selector, "core_removed_pct_points"] = float(best_row.core_pct) - 100 * core_overlap / total_weights
        summary_df.loc[selector, "backtracking_removed_pct_points"] = removed
    apply_masks(modules, originals, base, best_masks, device)
    final_ppl, final_nll = safe_perplexity(model, evaluation, device)
    drift = pd.DataFrame(final_diagnostics(model, diag_tokens, teacher_outputs, device))
    cheap = cheap_metrics(model, cheap_tokens, teacher_logits, teacher_hidden, device)
    summary_df.loc[summary_df.search == best_row.search, "final_ppl"] = final_ppl
    summary_df.loc[summary_df.search == best_row.search, "final_nll"] = final_nll
    summary_df.loc[summary_df.search == best_row.search, "final_kl"] = cheap["kl"]
    summary_df.loc[summary_df.search == best_row.search, "final_hidden_nmse"] = cheap["hidden_nmse"]

    history = pd.DataFrame(state["history"]); group_log = pd.DataFrame(state["group_log"])
    allocation = pd.DataFrame(allocation_rows)
    allocation = allocation[allocation.search != best_row.search]
    allocation = pd.concat([allocation, pd.DataFrame(distribution_rows(
        best_row.search, best_masks, originals, core_masks[float(best_row.core_pct)]))], ignore_index=True)
    summary_df.to_csv(args.output / "search_summary.csv", index=False)
    history.to_csv(args.output / "rescue_history.csv", index=False)
    group_log.to_csv(args.output / "group_log.csv", index=False)
    allocation.to_csv(args.output / "final_allocation.csv", index=False)
    drift.to_csv(args.output / "final_hidden_drift.csv", index=False)
    dump_json(args.output / "environment.json", {"model": args.model, "revision": args.revision,
              "seed": args.seed, "device": torch.cuda.get_device_name(), "torch": torch.__version__,
              "python": platform.python_version(), "bf16_ppl": bf_ppl, "bf16_nll": bf_nll,
              "target_ppl": target_ppl, "total_weights": total_weights,
              "full_ppl_evaluations": state["full_evals"], "arguments": vars(args)})
    plots(args.output, history, summary_df, allocation, target_ppl)
    write_report(args.output, bf_ppl, target_ppl, summary_df, history, allocation, drift, total_weights)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
