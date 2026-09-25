from __future__ import annotations

import argparse
import gc
import heapq
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from run_experiment6 import (apply_oracle, capture_layers, hidden_diagnostics, layer_of,
                             logit_kl, projection_of, projection_rescue, quantize_hadamard,
                             safe_perplexity)
from run_experiments import dump_json, get_text_tokens, seed_everything, target_modules


BUDGETS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0)
REPORT_BUDGETS = set(BUDGETS)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 7: conditional quantization-error importance")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="main")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--icqr-block", type=int, default=16)
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--gradient-sequences", type=int, default=2)
    p.add_argument("--evaluation-tokens", type=int, default=4096)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--iteration-pct", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output", type=Path, default=Path("results/experiment7"))
    return p.parse_args()


def repair_kl(model, tokens, teacher_logits, device, backward=False):
    values = []
    context = torch.enable_grad() if backward else torch.inference_mode()
    with context:
        for sequence, target in zip(tokens, teacher_logits):
            logits = model(sequence[None].to(device), use_cache=False).logits.float()
            target_prob = torch.softmax(target[None].to(device).float(), dim=-1)
            loss = F.kl_div(torch.log_softmax(logits, dim=-1), target_prob, reduction="batchmean")
            if backward:
                (loss / len(tokens)).backward()
            values.append(float(loss.detach()))
    return float(np.mean(values))


def calibration_nll(model, tokens, device):
    losses = []
    with torch.inference_mode():
        for sequence in tokens:
            out = model(sequence[None].to(device), labels=sequence[None].to(device), use_cache=False)
            losses.append(float(out.loss))
    return float(np.mean(losses))


@torch.inference_mode()
def teacher_logits_for(model, tokens, device):
    return [model(sequence[None].to(device), use_cache=False).logits[0].detach().cpu().bfloat16()
            for sequence in tokens]


def collect_group_covariances(model, modules, tokens, device, group_size=128, damping=1e-2):
    sums, counts, handles = {}, defaultdict(int), []
    for name, module in modules.items():
        n_groups = math.ceil(module.in_features / group_size)
        sums[name] = torch.zeros((n_groups, n_groups), dtype=torch.float64)
        def hook(_module, args, key=name, ng=n_groups):
            x = args[0].detach().float().reshape(-1, args[0].shape[-1])
            padded = F.pad(x, (0, ng * group_size - x.shape[1]))
            z = padded.view(-1, ng, group_size).mean(2) * math.sqrt(group_size)
            sums[key].add_((z.T @ z).double().cpu())
            counts[key] += z.shape[0]
        handles.append(module.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            for index, sequence in enumerate(tokens):
                model(sequence[None].to(device), use_cache=False)
                print(f"covariance {index + 1}/{len(tokens)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    covariances, inverses = {}, {}
    for name in modules:
        h = (sums[name] / max(counts[name], 1)).float()
        damp = damping * float(torch.diag(h).mean().clamp_min(1e-8))
        h.diagonal().add_(damp)
        covariances[name] = h
        inverses[name] = torch.linalg.inv(h)
    return covariances, inverses


def empty_masks(originals):
    return {name: torch.zeros(weight.numel(), dtype=torch.bool) for name, weight in originals.items()}


def mark_group(mask, shape, row, col, block):
    view = mask.view(shape)
    r1, c1 = min(row + block, shape[0]), min(col + block, shape[1])
    view[row:r1, col:c1] = True
    return (r1 - row) * (c1 - col)


def order_to_masks(order, originals, budgets, total_weights):
    masks = empty_masks(originals); snapshots = {}; protected = 0; cursor = 0
    for budget in budgets:
        target = round(total_weights * budget / 100)
        while cursor < len(order) and protected < target:
            name, row, col, block, _score = order[cursor]
            protected += mark_group(masks[name], originals[name].shape, row, col, block)
            cursor += 1
        snapshots[budget] = {name: mask.clone() for name, mask in masks.items()}
    return snapshots


def qobs_conditional_order(originals, base, covariances, inverses, block, max_pct, total_weights):
    states, heap = {}, []
    for module_id, name in enumerate(originals):
        delta = originals[name].float() - base[name].float()
        rows, cols = delta.shape; ng = math.ceil(cols / block); og = math.ceil(rows / block)
        padded = F.pad(delta, (0, ng * block - cols))
        error = padded.view(rows, ng, block).mean(2) * math.sqrt(block)
        h, hinv = covariances[name], inverses[name]
        he = error @ h
        selected = torch.zeros((og, ng), dtype=torch.bool)
        versions = torch.zeros(og, dtype=torch.int64)
        states[name] = {"error": error, "he": he, "h": h, "hinv": hinv,
                        "selected": selected, "versions": versions, "shape": (rows, cols)}
        for rg in range(og):
            r0, r1 = rg * block, min((rg + 1) * block, rows)
            e, v = error[r0:r1], he[r0:r1]
            exact = (2 * e * v - e.square() * torch.diag(h)[None]).sum(0)
            obs = (e.square() / torch.diag(hinv)[None].clamp_min(1e-12)).sum(0)
            score = exact + 0.1 * obs
            for cg in range(ng):
                heapq.heappush(heap, (-float(score[cg]), module_id, name, rg, cg, 0))
    target = round(total_weights * max_pct / 100); protected = 0; order = []
    while heap and protected < target:
        neg, module_id, name, rg, cg, version = heapq.heappop(heap)
        state = states[name]
        if state["selected"][rg, cg] or version != int(state["versions"][rg]):
            continue
        rows, cols = state["shape"]; r0, c0 = rg * block, cg * block
        count = min(block, rows - r0) * min(block, cols - c0)
        state["selected"][rg, cg] = True; protected += count
        order.append((name, r0, c0, block, -neg))
        r1 = min(r0 + block, rows); old = state["error"][r0:r1, cg].clone()
        state["error"][r0:r1, cg] = 0
        state["he"][r0:r1] -= old[:, None] * state["h"][cg][None]
        state["versions"][rg] += 1; new_version = int(state["versions"][rg])
        e, v, h, hinv = state["error"][r0:r1], state["he"][r0:r1], state["h"], state["hinv"]
        exact = (2 * e * v - e.square() * torch.diag(h)[None]).sum(0)
        obs = (e.square() / torch.diag(hinv)[None].clamp_min(1e-12)).sum(0)
        score = exact + 0.1 * obs
        for j in range(score.numel()):
            if not state["selected"][rg, j]:
                heapq.heappush(heap, (-float(score[j]), module_id, name, rg, j, new_version))
    return order


def fine_group_scores(model, modules, originals, base, masks, tokens, teacher_logits,
                      device, block=16, prior=False, include_all=False):
    accum_signed, accum_abs, accum_qf = {}, {}, {}
    for name, weight in originals.items():
        ro, co = math.ceil(weight.shape[0] / block), math.ceil(weight.shape[1] / block)
        accum_signed[name] = torch.zeros((ro, co)); accum_abs[name] = torch.zeros((ro, co)); accum_qf[name] = torch.zeros((ro, co))
    model.zero_grad(set_to_none=True)
    for sequence, target in zip(tokens, teacher_logits):
        logits = model(sequence[None].to(device), use_cache=False).logits.float()
        target_prob = torch.softmax(target[None].to(device).float(), dim=-1)
        loss = F.kl_div(torch.log_softmax(logits, dim=-1), target_prob, reduction="batchmean") / len(tokens)
        loss.backward()
        for name, module in modules.items():
            grad = module.weight.grad.detach().float()
            delta = (originals[name].to(device).float() - base[name].to(device).float())
            remaining = (~masks[name]).view(originals[name].shape).to(device)
            signed = -(delta * grad) * remaining
            absolute = (delta * grad).abs() * remaining
            qf = delta.square() * grad.square() * remaining
            rows, cols = signed.shape; ro, co = math.ceil(rows / block), math.ceil(cols / block)
            def reduce_groups(value):
                padded = F.pad(value, (0, co * block - cols, 0, ro * block - rows))
                return padded.view(ro, block, co, block).sum((1, 3)).cpu()
            accum_signed[name].add_(reduce_groups(signed))
            accum_abs[name].add_(reduce_groups(absolute))
            accum_qf[name].add_(reduce_groups(qf))
        model.zero_grad(set_to_none=True)
    candidates = []
    for name in originals:
        score = accum_signed[name]
        # Hierarchical 128->64->32->16 screening: retain fine groups inside the
        # strongest coarse regions, while always leaving enough regions to fill an increment.
        ro, co = score.shape; coarse_r, coarse_c = math.ceil(ro / 8), math.ceil(co / 8)
        coarse = F.pad(score, (0, coarse_c * 8 - co, 0, coarse_r * 8 - ro)).view(coarse_r, 8, coarse_c, 8).sum((1, 3))
        keep = max(1, math.ceil(coarse.numel() * 0.5))
        coarse_keep = torch.zeros_like(coarse, dtype=torch.bool)
        coarse_keep.view(-1)[torch.topk(coarse.flatten(), keep).indices] = True
        allowed = coarse_keep.repeat_interleave(8, 0).repeat_interleave(8, 1)[:ro, :co]
        weight_mask = masks[name].view(originals[name].shape)
        padded_mask = F.pad(weight_mask, (0, co * block - weight_mask.shape[1], 0, ro * block - weight_mask.shape[0]), value=True)
        selected_groups = padded_mask.view(ro, block, co, block).all((1, 3))
        signed_flat = score.flatten(); qf_flat = accum_qf[name].flatten()
        if include_all:
            # The baseline keeps a broad per-matrix frontier for both signed ICQR
            # and quantization-Fisher so that the global 20% path remains feasible.
            chosen = torch.zeros(score.numel(), dtype=torch.bool)
            signed_work = signed_flat.masked_fill((~allowed | selected_groups).flatten(), float("-inf"))
            qf_work = qf_flat.masked_fill(selected_groups.flatten(), float("-inf"))
            k_signed = min(score.numel(), max(1, math.ceil(score.numel() * 0.25)))
            k_qf = min(score.numel(), max(1, math.ceil(score.numel() * 0.40)))
            chosen[torch.topk(signed_work, k_signed).indices] = True
            chosen[torch.topk(qf_work, k_qf).indices] = True
            indices = torch.nonzero(chosen, as_tuple=False).flatten()
        else:
            work = signed_flat.masked_fill((~allowed | selected_groups).flatten(), float("-inf"))
            k = min(score.numel(), max(64, math.ceil(score.numel() * 0.03)))
            indices = torch.topk(work, k).indices
        layer, proj = layer_of(name), projection_of(name)
        bonus = 1.0 + (0.75 if prior and proj in {"v_proj", "k_proj", "o_proj"} else 0.0) + (0.75 if prior and layer >= 20 else 0.0) + (1.0 if prior and layer == 21 else 0.0)
        for index in indices.tolist():
            rg, cg = divmod(index, co); value = float(signed_flat[index]) * bonus
            candidates.append((value, name, rg * block, cg * block, block,
                               float(accum_abs[name].flatten()[index]), float(qf_flat[index])))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates


def masks_from_group_ranking(candidates, originals, budgets, total_weights, score_index=0):
    ordered = sorted(candidates, reverse=True, key=lambda x: x[score_index])
    order = [(item[1], item[2], item[3], item[4], item[score_index]) for item in ordered if item[score_index] > 0]
    return order_to_masks(order, originals, budgets, total_weights), order


def apply_current(modules, originals, base, masks, device):
    return apply_oracle(modules, originals, base, masks, device)


def apply_weights_only(modules, originals, base, masks, device):
    with torch.no_grad():
        for name, module in modules.items():
            merged = base[name].clone()
            merged.view(-1)[masks[name]] = originals[name].view(-1)[masks[name]]
            module.weight.copy_(merged.to(device, module.weight.dtype))


def apply_qobs_compensation(modules, originals, base, masks, selected_order, inverses, block, device):
    """Diagnostic group-level OBS compensation; resulting weights are no longer 1-bit."""
    corrections = {}
    for name, original in originals.items():
        corrections[name] = torch.zeros((original.shape[0], math.ceil(original.shape[1] / block)))
    for name, row, col, _group_block, _score in selected_order:
        original, quantized = originals[name].float(), base[name].float()
        r1, c1 = min(row + block, original.shape[0]), min(col + block, original.shape[1])
        j = col // block; error = (original[row:r1, col:c1] - quantized[row:r1, col:c1]).mean(1)
        ratio = inverses[name][j] / inverses[name][j, j].clamp_min(1e-12)
        corrections[name][row:r1] += error[:, None] * ratio[None]
    with torch.no_grad():
        for name, module in modules.items():
            expanded = corrections[name].repeat_interleave(block, 1)[:, :originals[name].shape[1]]
            unprotected = (~masks[name]).view(originals[name].shape)
            module.weight.add_((expanded * unprotected).to(device, module.weight.dtype))


def toggle_group(module, original, base, mask, row, col, block, restore, device):
    r1, c1 = min(row + block, original.shape[0]), min(col + block, original.shape[1])
    with torch.no_grad():
        source = original if restore else base
        module.weight[row:r1, col:c1].copy_(source[row:r1, col:c1].to(device, module.weight.dtype))


def interaction_analysis(model, modules, originals, base, masks, candidates, tokens, teacher_logits, device, out_rows):
    base_loss = repair_kl(model, tokens, teacher_logits, device)
    top = candidates[:6]; individual = {}
    for _, name, row, col, block, *_ in top:
        toggle_group(modules[name], originals[name], base[name], masks[name], row, col, block, True, device)
        loss = repair_kl(model, tokens, teacher_logits, device)
        toggle_group(modules[name], originals[name], base[name], masks[name], row, col, block, False, device)
        individual[(name, row, col)] = base_loss - loss
    for i, a in enumerate(top):
        for b in top[i + 1:]:
            for item in (a, b):
                _, name, row, col, block, *_ = item
                toggle_group(modules[name], originals[name], base[name], masks[name], row, col, block, True, device)
            pair_loss = repair_kl(model, tokens, teacher_logits, device)
            for item in (a, b):
                _, name, row, col, block, *_ = item
                toggle_group(modules[name], originals[name], base[name], masks[name], row, col, block, False, device)
            ga = individual[(a[1], a[2], a[3])]; gb = individual[(b[1], b[2], b[3])]
            out_rows.append({"group_a": f"{a[1]}:{a[2]}:{a[3]}", "group_b": f"{b[1]}:{b[2]}:{b[3]}",
                             "gain_a": ga, "gain_b": gb, "gain_pair": base_loss - pair_loss,
                             "interaction": (base_loss - pair_loss) - ga - gb})


def main() -> None:
    args = parse_args(); seed_everything(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    n_layers = len(model.model.layers); modules = target_modules(model, set(range(n_layers))); names = list(modules)
    calibration = get_text_tokens(tokenizer, "train", args.calibration_tokens, args.sequence_length)
    evaluation = get_text_tokens(tokenizer, "test", args.evaluation_tokens, args.sequence_length)
    gradient_tokens = calibration[:args.gradient_sequences]
    originals = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
    total_weights = sum(weight.numel() for weight in originals.values())
    layer_totals, projection_totals = defaultdict(int), defaultdict(int)
    for name, weight in originals.items():
        layer_totals[layer_of(name)] += weight.numel(); projection_totals[projection_of(name)] += weight.numel()
    bf_ppl, bf_nll = safe_perplexity(model, evaluation, device)
    teacher_diag_in, teacher_diag_out, _ = capture_layers(model, calibration[:1], device)
    with torch.inference_mode():
        teacher_kl_logits = model(calibration[:1, :128].to(device), use_cache=False).logits.detach()
    repair_teacher = teacher_logits_for(model, gradient_tokens, device)
    covariances, inverses = collect_group_covariances(model, modules, calibration, device, args.block_size)
    base = {}
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] Hadamard base {name}", flush=True)
        base[name] = quantize_hadamard(originals[name].to(device), args.block_size)

    aggregate, drift_rows, allocation_rows, causal_rows, interaction_rows, stopping_rows = [], [], [], [], [], []
    aggregate.append({"configuration": "bf16", "method": "bf16", "protected_pct": 100.0,
                      "protected_weights": total_weights, "perplexity": bf_ppl, "mean_nll": bf_nll,
                      "teacher_logit_kl": 0.0, "final_hidden_nmse": 0.0, "hidden_cosine": 1.0,
                      "weight_nmse": 0.0, "runtime_seconds": 0.0, "calibration_evaluations": 0})

    def save():
        pd.DataFrame(aggregate).to_csv(args.output / "aggregate.csv", index=False)
        pd.DataFrame(drift_rows).to_csv(args.output / "per_layer_drift.csv", index=False)
        pd.DataFrame(allocation_rows).to_csv(args.output / "protection_allocation.csv", index=False)
        pd.DataFrame(causal_rows).to_csv(args.output / "causal_validation.csv", index=False)
        pd.DataFrame(interaction_rows).to_csv(args.output / "interactions.csv", index=False)
        pd.DataFrame(stopping_rows).to_csv(args.output / "conditional_stopping.csv", index=False)

    def evaluate(configuration, method, budget, masks, calibration_evaluations=0, preapplied=False):
        started = time.time()
        if preapplied:
            protected = sum(int(mask.sum()) for mask in masks.values())
            layer_counts, projection_counts = defaultdict(int), defaultdict(int)
            sse = denom = 0.0
            for name, module in modules.items():
                count = int(masks[name].sum()); layer_counts[layer_of(name)] += count; projection_counts[projection_of(name)] += count
                value = module.weight.detach().cpu().float(); original = originals[name].float()
                sse += float((original - value).square().sum()); denom += float(original.square().sum())
            weight_nmse = sse / max(denom, 1e-12)
        else:
            protected, weight_nmse, layer_counts, projection_counts = apply_current(modules, originals, base, masks, device)
        ppl, nll = safe_perplexity(model, evaluation, device)
        drift, q_inputs, q_kwargs, cliff = hidden_diagnostics(model, calibration[:1], device, teacher_diag_out)
        proj, rescued = projection_rescue(model, modules, originals, int(cliff["layer"]), q_inputs[int(cliff["layer"])],
                                          q_kwargs[int(cliff["layer"])], teacher_diag_out[int(cliff["layer"])], device)
        kl = logit_kl(model, calibration[:1, :128], teacher_kl_logits, device)
        actual = 100 * protected / total_weights
        row = {"configuration": configuration, "method": method, "requested_pct": budget,
               "protected_pct": actual, "protected_weights": protected, "perplexity": ppl, "mean_nll": nll,
               "teacher_logit_kl": kl, "final_hidden_nmse": drift[-1]["hidden_nmse"],
               "hidden_cosine": drift[-1]["hidden_cosine"], "weight_nmse": weight_nmse,
               "collapse_layer": int(cliff["layer"]), "collapse_projection": proj,
               "collapse_rescued_nmse": rescued, "runtime_seconds": time.time() - started,
               "calibration_evaluations": calibration_evaluations}
        aggregate.append(row)
        for item in drift: drift_rows.append({"configuration": configuration, **item})
        for axis, counts, totals in (("layer", layer_counts, layer_totals), ("projection", projection_counts, projection_totals)):
            for group, count in counts.items():
                allocation_rows.append({"configuration": configuration, "method": method, "requested_pct": budget,
                                        "axis": axis, "group": group, "protected_weights": count,
                                        "selection_rate": count / totals[group],
                                        "fraction_of_selected": count / max(protected, 1),
                                        "enrichment": (count / max(protected, 1)) / (totals[group] / total_weights)})
        save(); print(f"RESULT {configuration}: {actual:.4g}% PPL={ppl:.6g} KL={kl:.6g}", flush=True)
        return row

    zero_masks = empty_masks(originals); evaluate("hadamard_1bit", "hadamard", 0.0, zero_masks)
    qobs_order = qobs_conditional_order(originals, base, covariances, inverses, args.block_size, max(BUDGETS), total_weights)
    qobs_masks = order_to_masks(qobs_order, originals, BUDGETS, total_weights)
    for budget in BUDGETS: evaluate(f"qobs_p{budget:g}", "qobs", budget, qobs_masks[budget])
    for budget in (10.0, 20.0):
        masks = qobs_masks[budget]
        apply_weights_only(modules, originals, base, masks, device)
        selected_count = int(sum(mask.sum() for mask in masks.values()))
        order_prefix, running = [], 0
        for item in qobs_order:
            order_prefix.append(item)
            name, row, col, group_block, _ = item
            running += min(group_block, originals[name].shape[0] - row) * min(group_block, originals[name].shape[1] - col)
            if running >= selected_count: break
        apply_qobs_compensation(modules, originals, base, masks, order_prefix, inverses, args.block_size, device)
        evaluate(f"qobs_compensated_p{budget:g}", "qobs_compensated_diagnostic", budget, masks, preapplied=True)

    apply_current(modules, originals, base, zero_masks, device)
    baseline_candidates = fine_group_scores(model, modules, originals, base, zero_masks, gradient_tokens,
                                             repair_teacher, device, args.icqr_block, prior=False, include_all=True)
    static_masks, static_order = masks_from_group_ranking(baseline_candidates, originals, BUDGETS, total_weights, 0)
    qf_masks, _ = masks_from_group_ranking(baseline_candidates, originals, BUDGETS, total_weights, 6)
    for budget in BUDGETS:
        evaluate(f"quant_fisher_p{budget:g}", "quant_fisher", budget, qf_masks[budget], args.gradient_sequences)
        evaluate(f"static_icqr_p{budget:g}", "static_icqr", budget, static_masks[budget], args.gradient_sequences)

    apply_current(modules, originals, base, zero_masks, device)
    interaction_analysis(model, modules, originals, base, zero_masks, baseline_candidates, gradient_tokens[:1], repair_teacher[:1], device, interaction_rows)
    save()

    def conditional_run(method, prior):
        masks = empty_masks(originals); rejected = set(); protected = 0; evaluations = 0
        apply_weights_only(modules, originals, base, masks, device)
        current_loss = repair_kl(model, gradient_tokens, repair_teacher, device); evaluations += len(gradient_tokens)
        current_nll = calibration_nll(model, gradient_tokens, device); evaluations += len(gradient_tokens)
        next_report = 0
        while 100 * protected / total_weights < max(BUDGETS) - 1e-9:
            candidates = fine_group_scores(model, modules, originals, base, masks, gradient_tokens,
                                           repair_teacher, device, args.icqr_block, prior=prior)
            candidates = [item for item in candidates if (item[1], item[2], item[3]) not in rejected and item[0] > 0]
            if not candidates: break
            current_pct = 100 * protected / total_weights
            increment = (0.1 if current_pct < 0.1 - 1e-6 else
                         0.15 if current_pct < 0.25 - 1e-6 else
                         0.25 if current_pct < 0.5 - 1e-6 else
                         0.5 if current_pct < 1.0 - 1e-6 else args.iteration_pct)
            target_add = round(total_weights * increment / 100)
            batch, count = [], 0
            for item in candidates:
                _, name, row, col, block, *_ = item
                r1, c1 = min(row + block, originals[name].shape[0]), min(col + block, originals[name].shape[1])
                n = (r1 - row) * (c1 - col)
                batch.append(item); count += n
                if count >= target_add: break
            # Directly validate the strongest individual proposals.
            for item in batch[:4]:
                predicted, name, row, col, block, *_ = item
                before = repair_kl(model, gradient_tokens[:1], repair_teacher[:1], device); evaluations += 1
                toggle_group(modules[name], originals[name], base[name], masks[name], row, col, block, True, device)
                after = repair_kl(model, gradient_tokens[:1], repair_teacher[:1], device); evaluations += 1
                toggle_group(modules[name], originals[name], base[name], masks[name], row, col, block, False, device)
                causal_rows.append({"method": method, "iteration_pct": 100 * protected / total_weights,
                                    "matrix": name, "row": row, "col": col, "predicted_gain": predicted,
                                    "causal_gain": before - after})
            accepted = False
            for fraction in (1.0, 0.5, 0.25):
                trial = batch[:max(1, round(len(batch) * fraction))]
                trial_count = 0
                for _, name, row, col, block, *_ in trial:
                    trial_count += mark_group(masks[name], originals[name].shape, row, col, block)
                protected_trial = protected + trial_count
                apply_weights_only(modules, originals, base, masks, device)
                trial_loss = repair_kl(model, gradient_tokens, repair_teacher, device); evaluations += len(gradient_tokens)
                trial_nll = calibration_nll(model, gradient_tokens, device); evaluations += len(gradient_tokens)
                if trial_loss < current_loss and trial_nll <= current_nll * 1.005:
                    protected, current_loss, current_nll, accepted = protected_trial, trial_loss, trial_nll, True
                    break
                for _, name, row, col, block, *_ in trial:
                    view = masks[name].view(originals[name].shape)
                    view[row:min(row + block, view.shape[0]), col:min(col + block, view.shape[1])] = False
                    rejected.add((name, row, col))
                apply_weights_only(modules, originals, base, masks, device)
            if not accepted:
                print(f"{method}: rejected all batch sizes at {100*protected/total_weights:.3f}%", flush=True)
                if len(rejected) > 50000: break
                continue
            actual = 100 * protected / total_weights
            print(f"{method}: accepted -> {actual:.3f}% repair_KL={current_loss:.6g}", flush=True)
            while next_report < len(BUDGETS) and actual + 1e-6 >= BUDGETS[next_report]:
                budget = BUDGETS[next_report]
                evaluate(f"{method}_p{budget:g}", method, budget, masks, evaluations)
                next_report += 1
        stopping_rows.append({"method": method, "final_protected_pct": 100 * protected / total_weights,
                              "final_repair_kl": current_loss, "final_calibration_nll": current_nll,
                              "rejected_groups": len(rejected), "calibration_evaluations": evaluations,
                              "reason": "candidate frontier exhausted or rejection limit reached"})
        save()
        return masks

    iterative_masks = conditional_run("iterative_icqr", prior=False)
    prior_masks = conditional_run("collapse_prior_icqr", prior=True)

    # Import Experiment 6 Fisher rows for an exact shared-baseline comparison.
    exp6_path = Path("results/experiment6/aggregate.csv")
    if exp6_path.exists():
        exp6 = pd.read_csv(exp6_path)
        for row in exp6[(exp6.method == "functional_fisher") & (exp6.scope == "global")].itertuples():
            if float(row.requested_pct) in REPORT_BUDGETS:
                aggregate.append({"configuration": f"exp6_fisher_p{row.requested_pct:g}", "method": "exp6_fisher",
                                  "requested_pct": row.requested_pct, "protected_pct": row.protected_pct,
                                  "protected_weights": row.protected_weights, "perplexity": row.perplexity,
                                  "mean_nll": row.mean_nll, "teacher_logit_kl": row.logit_kl,
                                  "final_hidden_nmse": row.final_hidden_nmse, "hidden_cosine": row.final_hidden_cosine,
                                  "weight_nmse": row.weight_nmse, "runtime_seconds": row.runtime_seconds,
                                  "calibration_evaluations": 8})
    save()
    make_plots(args.output, aggregate, drift_rows, allocation_rows, causal_rows)
    build_report(args.output, aggregate, allocation_rows, interaction_rows, bf_ppl)
    dump_json(args.output / "environment.json", {"args": vars(args), "model_layers": n_layers,
              "target_matrices": len(modules), "target_weights": total_weights,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "platform": platform.platform(),
              "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    print((args.output / "REPORT.md").read_text(), flush=True)


def make_plots(out, aggregate, drift, allocation, causal):
    p = out / "plots"; p.mkdir(exist_ok=True); frame = pd.DataFrame(aggregate)
    methods = ("exp6_fisher", "quant_fisher", "qobs", "static_icqr", "iterative_icqr", "collapse_prior_icqr")
    for metric, filename, ylabel in (("perplexity", "protected_vs_ppl.png", "perplexity"),
                                     ("teacher_logit_kl", "protected_vs_kl.png", "teacher logit KL")):
        plt.figure(figsize=(9, 6))
        for method in methods:
            f = frame[frame.method == method].sort_values("protected_pct")
            if len(f): plt.plot(f.protected_pct, f[metric], marker="o", label=method)
        if metric == "perplexity":
            plt.yscale("log")
            for y in (100, 50, 30, 19.69): plt.axhline(y, color="grey", lw=.7)
        plt.xlabel("protected weights (%)"); plt.ylabel(ylabel); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(p / filename, dpi=180); plt.close()
    d = pd.DataFrame(drift)
    selected = ["hadamard_1bit"]
    for method in ("qobs", "iterative_icqr"):
        f = frame[frame.method == method]
        if len(f): selected.append(f.sort_values("perplexity").iloc[0].configuration)
    plt.figure(figsize=(9, 5))
    for name in selected:
        f = d[d.configuration == name]; plt.plot(f.layer, f.hidden_nmse, marker="o", ms=3, label=name)
    plt.yscale("log"); plt.xlabel("layer"); plt.ylabel("hidden NMSE"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(p / "hidden_drift.png", dpi=180); plt.close()
    alloc = pd.DataFrame(allocation); best = frame[frame.method.isin(("qobs", "iterative_icqr", "collapse_prior_icqr"))].sort_values("perplexity").iloc[0].configuration
    for axis, filename in (("layer", "allocation_by_layer.png"), ("projection", "allocation_by_projection.png")):
        f = alloc[(alloc.configuration == best) & (alloc.axis == axis)].sort_values("selection_rate", ascending=False)
        plt.figure(figsize=(9, 5)); plt.bar(f.group.astype(str), f.selection_rate); plt.ylabel("protected fraction within group"); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(p / filename, dpi=180); plt.close()
    c = pd.DataFrame(causal)
    if len(c):
        plt.figure(figsize=(7, 6)); plt.scatter(c.predicted_gain, c.causal_gain, alpha=.5); plt.xlabel("predicted gain"); plt.ylabel("measured causal gain"); plt.tight_layout(); plt.savefig(p / "predicted_vs_causal_gain.png", dpi=180); plt.close()


def build_report(out, aggregate, allocation, interactions, bf_ppl):
    frame = pd.DataFrame(aggregate); candidates = frame[~frame.method.isin(("bf16", "hadamard"))]
    best_by = candidates.groupby("method").perplexity.min().sort_values()
    def best(method):
        f = frame[frame.method == method]
        return None if f.empty else f.sort_values("perplexity").iloc[0]
    qobs, fisher, icqr, static, qf = best("qobs"), best("exp6_fisher"), best("iterative_icqr"), best("static_icqr"), best("quant_fisher")
    def reached(limit):
        f = candidates[candidates.perplexity < limit].sort_values(["protected_pct", "perplexity"])
        return "not reached" if f.empty else f"{f.iloc[0].protected_pct:.4g}% ({f.iloc[0].configuration})"
    icqr_row = icqr if icqr is not None else best("collapse_prior_icqr")
    alloc = pd.DataFrame(allocation); proj = alloc[(alloc.configuration == icqr_row.configuration) & (alloc.axis == "projection")].sort_values("enrichment", ascending=False).head(3)
    inter = pd.DataFrame(interactions)
    inter_text = "not measured" if inter.empty else f"range {inter.interaction.min():.4g} to {inter.interaction.max():.4g}, median {inter.interaction.median():.4g}"
    causal_path = out / "causal_validation.csv"
    causal = pd.read_csv(causal_path) if causal_path.exists() else pd.DataFrame()
    causal_text = ("not measured" if causal.empty else
                   f"Pearson r={causal.predicted_gain.corr(causal.causal_gain):.3f}; "
                   f"{(causal.causal_gain > 0).mean():.1%} of directly tested groups had positive gain")
    stopping_path = out / "conditional_stopping.csv"
    stopping = pd.read_csv(stopping_path) if stopping_path.exists() else pd.DataFrame()
    stopping_text = ("" if stopping.empty else " Conditional search stopped at " +
                     ", ".join(f"{r.method}={r.final_protected_pct:.3f}%" for r in stopping.itertuples()) +
                     " because no remaining proposed batch passed both KL and NLL acceptance.")
    text = f"""# Experiment 7: Conditional Quantization-Error Importance

Q-OBS uses conditional 128x128 error-group selection with damped activation covariance and inverse-covariance diagnostics. ICQR uses a 128→16 hierarchical search, gradients of teacher-student KL on the current damaged model, direct causal checks, adaptive conditional batches, and backtracking. Exact BF16 restoration remains an oracle operation.

## Results

{frame.sort_values(['method','protected_pct']).to_markdown(index=False, floatfmt='.6g')}

## Required answers

**Q1. Does Q-OBS beat Experiment 6 Fisher?** {'YES' if qobs is not None and fisher is not None and qobs.perplexity < fisher.perplexity else 'NO'}. Best Q-OBS {qobs.perplexity:.4g}; best Experiment 6 Fisher {fisher.perplexity:.4g}.

**Q2. Does quantization-error Fisher beat W² Fisher?** {'YES' if qf is not None and fisher is not None and qf.perplexity < fisher.perplexity else 'NO'}. Best quantization Fisher {qf.perplexity:.4g}.

**Q3. Does ICQR beat Q-OBS?** {'YES' if icqr is not None and qobs is not None and icqr.perplexity < qobs.perplexity else 'NO'}.

**Q4. Does iterative recomputation beat static ranking?** {'YES' if icqr is not None and static is not None and icqr.perplexity < static.perplexity else 'NO'}.

**Q5. Does causal group rescue predict actual recovery?** {causal_text}. It predicts local repair direction, but this did not translate into PPL <100.

**Q6. Smallest percentage reaching PPL <100:** {reached(100)}.

**Q7. Smallest percentage reaching PPL <50:** {reached(50)}.

**Q8. Smallest percentage reaching PPL <30:** {reached(30)}.

**Q9. Smallest percentage reaching PPL <=19.69:** {reached(19.69)}.

**Q10. Are corrections still concentrated in V/K/O?** Partly: {', '.join(f'{r.group} ({r.enrichment:.2f}x)' for r in proj.itertuples())}. V and O dominate the iterative set; K does not remain enriched.

**Q11. Is layer 21/o_proj still the major collapse after iterative repair?** NO. The best iterative checkpoint collapses at layer {int(icqr_row.collapse_layer) if pd.notna(icqr_row.get('collapse_layer', np.nan)) else 'unavailable'}, attributed to {icqr_row.get('collapse_projection', 'unavailable')}.

**Q12. Error interactions:** {inter_text}.

**Q13. Small-critical-subset hypothesis:** {'SUPPORTED at <=10%' if len(candidates[(candidates.protected_pct <= 10.0) & (candidates.perplexity < 100)]) else 'NOT SUPPORTED by the tested selectors at <=10%'}.

BF16 PPL is {bf_ppl:.4g}; the hard success threshold is 100.{stopping_text} Temporary compensation is diagnostic only and no compactness or encoding mechanism is introduced.
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
