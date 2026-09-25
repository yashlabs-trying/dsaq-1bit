from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dsaq.core import blocks_to_matrix, hadamard_blocks, matrix_to_blocks, reconstruct_naive
from run_experiments import activation_importance, dump_json, get_text_tokens, perplexity, seed_everything, set_weights, target_modules


FRACTIONS = (0.0, .0001, .0005, .001, .0025, .005, .01, .02, .05)
CONCENTRATION = (.001, .005, .01, .02, .05, .10)
ADAPTIVE_BUDGETS = (.025, .05, .10)
STRATEGIES = ("magnitude", "residual", "residual_squared", "hessian_damage")
STRUCTURES = ("row", "column", "block8", "block16")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="main")
    p.add_argument("--layers", default="0,12,23")
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--evaluation-tokens", type=int, default=4096)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output", type=Path, default=Path("results/experiment4"))
    return p.parse_args()


def layer_number(name: str) -> int:
    return int(name.split(".layers.", 1)[1].split(".", 1)[0])


def projection_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def transformed_sign_basis(weight: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor, object]:
    blocks, layout = matrix_to_blocks(weight, block_size)
    transformed = hadamard_blocks(blocks, block_size)
    return blocks, transformed.sign(), layout


def binary_base(weight: torch.Tensor, importance: torch.Tensor, block_size: int, objective: str,
                granularity: str = "block") -> tuple[torch.Tensor, int]:
    """Hadamard sign code with FP16 scales optimized in transformed, raw, or weighted output space."""
    blocks, signs, layout = transformed_sign_basis(weight, block_size)
    valid = layout.valid.float()
    if objective == "naive":
        transformed = hadamard_blocks(blocks, block_size)
        _, scale = reconstruct_naive(transformed, layout.valid)
        scale = scale.half().float()
        q_t = signs * scale[:, None]
        return blocks_to_matrix(hadamard_blocks(q_t, block_size), layout), 16 * len(scale)

    sign_back = hadamard_blocks(signs, block_size)
    imp_matrix = importance[None, :].expand(weight.shape[0], -1)
    imp_blocks, _ = matrix_to_blocks(imp_matrix, block_size)
    metric = valid if objective == "magnitude" else valid * imp_blocks
    if granularity == "block":
        numerator = (blocks * sign_back * metric).sum(1)
        denominator = (sign_back.square() * metric).sum(1).clamp_min(1e-12)
        scale = (numerator / denominator).clamp_min(0).half().float()
        q = blocks_to_matrix(sign_back * scale[:, None], layout)
        return q, 16 * len(scale)
    if granularity != "row":
        raise ValueError(granularity)
    basis = blocks_to_matrix(sign_back, layout)
    metric2 = torch.ones_like(weight) if objective == "magnitude" else imp_matrix
    scale = ((weight * basis * metric2).sum(1) / (basis.square() * metric2).sum(1).clamp_min(1e-12)).clamp_min(0).half().float()
    return basis * scale[:, None], 16 * len(scale)


def matrix_metrics(name: str, original: torch.Tensor, recon: torch.Tensor, importance: torch.Tensor,
                   config: str, binary_bits: int, lrr_bits: int = 0, rescue_bits: int = 0,
                   rescued: int = 0, rank: int = 0) -> dict[str, object]:
    residual = original - recon
    raw_sse = residual.square().sum()
    w_sse = (residual.square() * importance[None, :]).sum()
    return {
        "matrix": name, "layer": layer_number(name), "projection": projection_type(name), "configuration": config,
        "weights": original.numel(), "rank": rank, "rescued_weights": rescued,
        "raw_sse": float(raw_sse), "weight_energy": float(original.square().sum()),
        "weighted_sse": float(w_sse), "weighted_energy": float((original.square() * importance[None, :]).sum()),
        "dot": float((original * recon).sum()), "original_norm2": float(original.square().sum()),
        "recon_norm2": float(recon.square().sum()), "binary_bits": binary_bits, "lrr_bits": lrr_bits,
        "rescue_bits": rescue_bits, "total_bits": binary_bits + lrr_bits + rescue_bits,
    }


def aggregate_rows(rows: list[dict[str, object]], keys: list[str]) -> pd.DataFrame:
    d = pd.DataFrame(rows)
    g = d.groupby(keys, dropna=False).agg(
        matrices=("matrix", "count"), weights=("weights", "sum"), rescued_weights=("rescued_weights", "sum"),
        raw_sse=("raw_sse", "sum"), weight_energy=("weight_energy", "sum"), weighted_sse=("weighted_sse", "sum"),
        weighted_energy=("weighted_energy", "sum"), dot=("dot", "sum"), original_norm2=("original_norm2", "sum"),
        recon_norm2=("recon_norm2", "sum"), binary_bits=("binary_bits", "sum"), lrr_bits=("lrr_bits", "sum"),
        rescue_bits=("rescue_bits", "sum"), total_bits=("total_bits", "sum"), mean_rank=("rank", "mean"),
    ).reset_index()
    g["raw_mse"] = g.raw_sse / g.weights
    g["relative_error"] = g.raw_sse / g.weight_energy
    g["weighted_error"] = g.weighted_sse / g.weighted_energy
    g["cosine"] = g["dot"] / np.sqrt(g.original_norm2 * g.recon_norm2)
    g["true_bpw"] = g.total_bits / g.weights
    g["binary_metadata_bpw"] = (g.binary_bits - g.weights) / g.weights
    g["lrr_metadata_bpw"] = g.lrr_bits / g.weights
    g["salient_rescue_bpw"] = g.rescue_bits / g.weights
    g["fraction_weights_rescued"] = g.rescued_weights / g.weights
    return g


def fp16_hessian_corrections(residual: torch.Tensor, importance: torch.Tensor, ranks=(1, 2, 3, 4)) -> dict[int, torch.Tensor]:
    root = importance.sqrt().clamp_min(1e-8)
    u, s, vh = torch.linalg.svd(residual * root[None, :], full_matrices=False)
    out = {}
    for rank in ranks:
        corr = (u[:, :rank].half().float() * s[:rank].half().float()[None, :]) @ vh[:rank].half().float()
        out[rank] = corr * root.reciprocal()[None, :]
    return out


def allocate_ranks(states: dict[str, dict], total_weights: int, budget_bpw: float) -> dict[str, int]:
    ranks = {name: 0 for name in states}
    spent = 0
    while True:
        choices = []
        for name, state in states.items():
            old = ranks[name]
            if old >= 4:
                continue
            new = old + 1
            before = state["base"] if old == 0 else state["base"] + state["corrections"][old]
            after = state["base"] + state["corrections"][new]
            imp = state["importance"]
            gain = float(((state["original"] - before).square() * imp[None, :]).sum() -
                         ((state["original"] - after).square() * imp[None, :]).sum())
            bits = state["rank_bits"]
            choices.append((gain / bits, gain, bits, name, new))
        choices.sort(reverse=True)
        selected = next((x for x in choices if x[1] > 0 and spent + x[2] <= budget_bpw * total_weights), None)
        if selected is None:
            break
        _, _, bits, name, new = selected
        ranks[name] = new; spent += bits
    return ranks


def reconstruction_from_ranks(states: dict[str, dict], ranks: dict[str, int]) -> dict[str, torch.Tensor]:
    return {name: state["base"] + (state["corrections"][ranks[name]] if ranks[name] else 0) for name, state in states.items()}


def global_order(states: dict[str, dict], recon: dict[str, torch.Tensor], strategy: str) -> tuple[torch.Tensor, list[tuple[str, int, int]], torch.Tensor]:
    scores, mapping, damage = [], [], []
    for name, state in states.items():
        residual = state["original"] - recon[name]
        if strategy == "magnitude": score = state["original"].abs()
        elif strategy == "residual": score = residual.abs()
        elif strategy == "residual_squared": score = residual.square()
        elif strategy == "hessian_damage": score = residual.square() * state["importance"][None, :]
        else: raise ValueError(strategy)
        scores.append(score.flatten().cpu())
        damage.append((residual.square() * state["importance"][None, :]).flatten().cpu())
        mapping.append((name, state["original"].shape[0], state["original"].shape[1]))
    all_scores = torch.cat(scores); all_damage = torch.cat(damage)
    max_k = math.ceil(FRACTIONS[-1] * len(all_scores))
    return torch.topk(all_scores, max_k, sorted=True).indices, mapping, all_damage


def apply_individual_rescue(states: dict[str, dict], base_recon: dict[str, torch.Tensor], order: torch.Tensor,
                            mapping: list[tuple[str, int, int]], count: int) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    out = {name: value.clone() for name, value in base_recon.items()}; per_matrix = {name: 0 for name in out}
    chosen = order[:count]
    offset = 0
    for name, rows, cols in mapping:
        size = rows * cols; mask = (chosen >= offset) & (chosen < offset + size); local = chosen[mask] - offset
        if len(local):
            flat = out[name].view(-1); original = states[name]["original"].view(-1)
            residual16 = (original[local] - flat[local]).bfloat16().float(); flat[local] += residual16
            per_matrix[name] = len(local)
        offset += size
    return out, per_matrix


def structured_units(states: dict[str, dict], recon: dict[str, torch.Tensor], structure: str) -> list[tuple[float, str, object, int, int]]:
    units = []
    matrix_bits = math.ceil(math.log2(len(states)))
    for name, state in states.items():
        residual = state["original"] - recon[name]
        damage = residual.square() * state["importance"][None, :]
        rows, cols = residual.shape
        if structure == "row":
            for i, value in enumerate(damage.sum(1).cpu()): units.append((float(value), name, i, cols, matrix_bits + math.ceil(math.log2(rows))))
        elif structure == "column":
            for j, value in enumerate(damage.sum(0).cpu()): units.append((float(value), name, j, rows, matrix_bits + math.ceil(math.log2(cols))))
        else:
            b = int(structure.replace("block", "")); br, bc = math.ceil(rows / b), math.ceil(cols / b)
            for i in range(br):
                for j in range(bc):
                    r1, c1 = min((i + 1) * b, rows), min((j + 1) * b, cols)
                    n = (r1 - i*b) * (c1 - j*b)
                    units.append((float(damage[i*b:r1, j*b:c1].sum()), name, (i, j, b), n,
                                  matrix_bits + math.ceil(math.log2(br * bc))))
    units.sort(key=lambda x: x[0], reverse=True)
    return units


def apply_structured_rescue(states: dict[str, dict], base_recon: dict[str, torch.Tensor], units: list[tuple], target: int,
                            structure: str) -> tuple[dict[str, torch.Tensor], dict[str, int], int, int]:
    out = {name: value.clone() for name, value in base_recon.items()}; selected, count, index_bits = [], 0, 0
    for unit in units:
        selected.append(unit); count += unit[3]; index_bits += unit[4]
        if count >= target: break
    per_matrix = {name: 0 for name in out}
    for _, name, loc, n, _ in selected:
        original, rec = states[name]["original"], out[name]
        if structure == "row": rec[loc] += (original[loc] - rec[loc]).bfloat16().float()
        elif structure == "column": rec[:, loc] += (original[:, loc] - rec[:, loc]).bfloat16().float()
        else:
            i, j, b = loc; r1, c1 = min((i+1)*b, rec.shape[0]), min((j+1)*b, rec.shape[1])
            rec[i*b:r1, j*b:c1] += (original[i*b:r1, j*b:c1] - rec[i*b:r1, j*b:c1]).bfloat16().float()
        per_matrix[name] += n
    return out, per_matrix, count, index_bits


def damage_concentration(states: dict[str, dict], recon: dict[str, torch.Tensor]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records, grouped = [], []
    global_units: dict[str, list[torch.Tensor]] = {"weight": [], "column": [], "row": [], "block16": []}
    for name, state in states.items():
        damage = ((state["original"] - recon[name]).square() * state["importance"][None, :]).cpu()
        total = float(damage.sum())
        for axis, values in (("weight", damage.flatten()), ("column", damage.sum(0)), ("row", damage.sum(1))):
            global_units[axis].append(values)
            ordered = values.sort(descending=True).values; cumulative = ordered.cumsum(0) / ordered.sum().clamp_min(1e-12)
            for frac in CONCENTRATION:
                k = max(1, math.ceil(frac * len(values)))
                records.append({"matrix": name, "layer": layer_number(name), "projection": projection_type(name),
                                "unit": axis, "fraction": frac, "damage_fraction": float(cumulative[k-1])})
        blocks, _ = matrix_to_blocks(damage, 16); values = blocks.sum(1); ordered = values.sort(descending=True).values
        global_units["block16"].append(values)
        cumulative = ordered.cumsum(0) / ordered.sum().clamp_min(1e-12)
        for frac in CONCENTRATION:
            k = max(1, math.ceil(frac * len(values)))
            records.append({"matrix": name, "layer": layer_number(name), "projection": projection_type(name),
                            "unit": "block16", "fraction": frac, "damage_fraction": float(cumulative[k-1])})
        grouped.append({"matrix": name, "layer": layer_number(name), "projection": projection_type(name), "weighted_damage": total})
    for axis, parts in global_units.items():
        values = torch.cat(parts); ordered = values.sort(descending=True).values
        cumulative = ordered.cumsum(0) / ordered.sum().clamp_min(1e-12)
        for frac in CONCENTRATION:
            k = max(1, math.ceil(frac * len(values)))
            records.append({"matrix": "ALL", "layer": -1, "projection": "ALL", "unit": axis,
                            "fraction": frac, "damage_fraction": float(cumulative[k-1])})
    return pd.DataFrame(records), pd.DataFrame(grouped)


def make_plots(out: Path, results: pd.DataFrame, concentration: pd.DataFrame, allocation: pd.DataFrame, damage_groups: pd.DataFrame) -> None:
    p = out / "plots"; p.mkdir(exist_ok=True)
    low = results[results.true_bpw <= 3.2]
    plt.figure(figsize=(9, 5));
    for family, f in low.groupby("family"): plt.scatter(f.true_bpw, f.perplexity, label=family, s=30)
    plt.yscale("log"); plt.xlabel("true BPW"); plt.ylabel("perplexity"); plt.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.savefig(p/"perplexity_vs_bpw.png", dpi=180); plt.close()
    plt.figure(figsize=(9, 5));
    for family, f in low.groupby("family"): plt.scatter(f.true_bpw, f.weighted_error, label=family, s=30)
    plt.xlabel("true BPW"); plt.ylabel("weighted error"); plt.legend(fontsize=8, ncol=2); plt.tight_layout(); plt.savefig(p/"weighted_error_vs_bpw.png", dpi=180); plt.close()
    c = concentration[concentration.matrix == "ALL"]
    plt.figure(figsize=(8, 5));
    for unit, f in c.groupby("unit"): plt.plot(100*f.fraction, 100*f.damage_fraction, marker="o", label=unit)
    plt.xlabel("top units (%)"); plt.ylabel("weighted damage captured (%)"); plt.legend(); plt.tight_layout(); plt.savefig(p/"damage_cumulative.png", dpi=180); plt.close()
    sparse = results[(results.family == "sparse") & (results.strategy == "hessian_damage")].sort_values("fraction_weights_rescued")
    plt.figure(figsize=(8, 5)); plt.plot(100*sparse.fraction_weights_rescued, sparse.perplexity, marker="o"); plt.yscale("log"); plt.xlabel("weights restored (%)"); plt.ylabel("perplexity"); plt.tight_layout(); plt.savefig(p/"perplexity_vs_sparse_fraction.png", dpi=180); plt.close()
    compare = results[(results.family == "sparse") & (results.fraction_weights_rescued <= .01)]
    plt.figure(figsize=(8, 5));
    for strategy, f in compare.groupby("strategy"): plt.plot(100*f.fraction_weights_rescued, f.perplexity, marker="o", label=strategy)
    plt.yscale("log"); plt.xlabel("weights restored (%)"); plt.ylabel("perplexity"); plt.legend(); plt.tight_layout(); plt.savefig(p/"selection_strategies.png", dpi=180); plt.close()
    a = allocation.groupby(["budget_bpw", "projection"]).allocated_rank.mean().unstack(fill_value=0)
    a.plot(kind="bar", figsize=(9, 5)); plt.ylabel("mean allocated rank"); plt.tight_layout(); plt.savefig(p/"adaptive_rank_by_projection.png", dpi=180); plt.close()
    d = damage_groups.groupby(["layer", "projection"]).weighted_damage.sum().unstack(fill_value=0)
    d = d.div(d.sum().sum()); d.plot(kind="bar", stacked=True, figsize=(9, 5)); plt.ylabel("fraction of weighted damage"); plt.tight_layout(); plt.savefig(p/"damage_by_layer_projection.png", dpi=180); plt.close()


def build_report(out: Path, results: pd.DataFrame, concentration: pd.DataFrame) -> None:
    bf16 = float(results.loc[results.configuration == "bf16", "perplexity"].iloc[0])
    naive = results.loc[results.configuration == "hadamard_naive_b128"].iloc[0]
    hbase = results[results.family == "hessian_base"].sort_values("perplexity").iloc[0]
    global1 = results.loc[results.configuration == "global_hessian_rank1"].iloc[0]
    adaptive = results[results.family == "adaptive_lrr"].sort_values("true_bpw")
    sparse = results[(results.family == "sparse") & (results.strategy == "hessian_damage")].sort_values("fraction_weights_rescued")
    structured = results[results.family == "structured"]
    top1 = float(concentration[(concentration.matrix == "ALL") & (concentration.unit == "weight") & (concentration.fraction == .01)].damage_fraction.iloc[0])
    h_better = hbase.weighted_error < naive.weighted_error and hbase.perplexity < naive.perplexity
    adaptive_best = adaptive.sort_values("perplexity").iloc[0]
    global_at_similar = global1
    adaptive_wins = bool(adaptive.iloc[0].perplexity < global_at_similar.perplexity) if len(adaptive) else False
    sparse1 = sparse.iloc[(sparse.fraction_weights_rescued - .01).abs().argmin()]
    sparse_recovery = (hbase.perplexity - sparse1.perplexity) / max(hbase.perplexity - bf16, 1e-12)
    functionally_concentrated = sparse_recovery >= .5
    best_sparse = results[results.family == "sparse"].sort_values("perplexity").iloc[0]
    individual_one = results[(results.family == "sparse") & (results.fraction_weights_rescued >= .0099) & (results.fraction_weights_rescued <= .0101)].sort_values("perplexity").iloc[0]
    structured_one = structured[(structured.fraction_weights_rescued >= .009) & (structured.fraction_weights_rescued <= .011)].sort_values("perplexity").iloc[0]
    thresholds = []
    for threshold in (100, 50, 30, 25, 19.7):
        hit = results[(results.configuration != "bf16") & (results.perplexity < threshold)]
        thresholds.append({"ppl_threshold": threshold, "lowest_true_bpw": float(hit.true_bpw.min()) if len(hit) else np.nan,
                           "configuration": hit.sort_values("true_bpw").iloc[0].configuration if len(hit) else "not reached"})
    budgets = []
    for cap in (1.10, 1.20, 1.30, 1.50):
        eligible = results[results.true_bpw <= cap]
        best = eligible.sort_values("perplexity").iloc[0]
        budgets.append({"bpw_cap": cap, "best_perplexity": best.perplexity, "configuration": best.configuration})
    best = results[results.configuration != "bf16"].sort_values("perplexity").iloc[0]
    q6 = bool(best.perplexity <= 1.10 * bf16)
    cols = ["configuration", "family", "perplexity", "perplexity_degradation_pct", "weighted_error", "true_bpw",
            "binary_metadata_bpw", "lrr_metadata_bpw", "salient_rescue_bpw", "fraction_weights_rescued", "damage_removed_fraction"]
    text = f"""# Experiment 4: sensitivity-directed binary, low-rank, and sparse rescue

## Result

BF16 perplexity is **{bf16:.4f}**. The best compressed configuration is **{best.configuration}** at perplexity **{best.perplexity:.4f}** and **{best.true_bpw:.6f} BPW**. The 5–10% target (perplexity <= {1.10*bf16:.4f}) was **{'reached' if q6 else 'not reached'}**.

The top 1% of individual residual weights contains **{top1:.2%}** of weighted damage after the selected Hessian-aware base.

## Principal configurations

{results[cols].sort_values(['true_bpw','perplexity']).to_markdown(index=False, floatfmt='.6g')}

## Lowest BPW reaching perplexity thresholds

{pd.DataFrame(thresholds).to_markdown(index=False, floatfmt='.6g')}

## Best result under each storage cap

{pd.DataFrame(budgets).to_markdown(index=False, floatfmt='.6g')}

## Answers

**Q1: Is Hessian-aware binary quantization better than naive binary? {'YES' if h_better else 'NO'}.** Best Hessian base: {hbase.configuration}, PPL {hbase.perplexity:.4f}, weighted error {hbase.weighted_error:.6f}; naive PPL {naive.perplexity:.4f}, weighted error {naive.weighted_error:.6f}.

**Q2: Is most model damage concentrated enough for targeted correction? {'YES' if functionally_concentrated else 'NO'}.** The diagonal proxy is concentrated (top 1% captures {top1:.2%}), but restoring that 1% recovers only {sparse_recovery:.2%} of the excess perplexity.

**Q3: What percentage causes most damage?** See `damage_concentration.csv`; top 1% weights capture {top1:.2%}, with row, column, and block curves reported separately.

**Q4: How much can sparse salient rescue lower perplexity?** Hessian-selected 1% restoration changes PPL from {hbase.perplexity:.4f} to {sparse1.perplexity:.4f} at {sparse1.true_bpw:.6f} BPW. The best sparse result is {best_sparse.configuration}, PPL {best_sparse.perplexity:.4f} at {best_sparse.true_bpw:.6f} BPW. At about 1% restored, the best individual method ({individual_one.configuration}, PPL {individual_one.perplexity:.4f}) beats the best structured method ({structured_one.configuration}, PPL {structured_one.perplexity:.4f}).

**Q5: Best perplexity below 1.10/1.20/1.30/1.50 BPW:** shown in the storage-cap table above.

**Q6: Did any tested configuration reach within 5–10% of BF16? {'YES' if q6 else 'NO'}.**

**Adaptive-LRR ablation:** {'PASS' if adaptive_wins else 'FAIL'} at the matched ~0.025 added-BPW point. Global rank-1 gives PPL {global1.perplexity:.4f} at {global1.true_bpw:.6f} BPW; the smallest adaptive budget gives PPL {adaptive.iloc[0].perplexity:.4f} at {adaptive.iloc[0].true_bpw:.6f} BPW.

**Q7: Dominant remaining error:** {'the target was reached' if q6 else 'the diagonal activation-importance proxy does not identify the weights that control perplexity, and the remaining functional loss is not isolated by the tested sparse or rank-4 repairs'}. The strongest evidence is that Hessian selection removes much more proxy damage while magnitude selection produces much lower perplexity. No new algorithm is proposed here.

Magnitude and residual-squared strategies use identical cardinalities. `residual` and `residual_squared` must produce identical orderings and are retained as an explicit control. All sparse values are BF16 residuals; individual indices use a global flat index, while structured methods charge matrix and local-unit identifiers. Every low-rank factor stores FP16 U, S, and V.
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args(); seed_everything(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16).to(device).eval()
    modules = target_modules(model, {int(x) for x in args.layers.split(",")})
    calibration = get_text_tokens(tokenizer, "train", args.calibration_tokens, args.sequence_length)
    evaluation = get_text_tokens(tokenizer, "test", args.evaluation_tokens, args.sequence_length)
    importance = activation_importance(model, modules, calibration, device)
    originals = {name: module.weight.detach().float().cpu() for name, module in modules.items()}
    total_weights = sum(x.numel() for x in originals.values()); original_gpu = {k: v.to(device) for k, v in originals.items()}
    imp_gpu = {k: importance[k].to(device).float() for k in originals}
    base_specs = [
        ("hadamard_naive_b128", "naive", 128, "block", "naive_base"),
        ("hadamard_magnitude_b128", "magnitude", 128, "block", "magnitude_base"),
        ("hadamard_hessian_b128", "hessian", 128, "block", "hessian_base"),
        ("hadamard_magnitude_b64", "magnitude", 64, "block", "magnitude_base"),
        ("hadamard_hessian_b64", "hessian", 64, "block", "hessian_base"),
        ("hadamard_magnitude_row", "magnitude", 128, "row", "magnitude_base"),
        ("hadamard_hessian_row", "hessian", 128, "row", "hessian_base"),
    ]
    base_recons, base_meta, matrix_rows, config_meta = {}, {}, [], {}
    baseline_ppl = perplexity(model, evaluation, device)
    ppl_records = [{"configuration": "bf16", "family": "baseline", "perplexity": baseline_ppl, "strategy": "none",
                    "damage_removed_fraction": 1.0, "exact_index_bits": 0}]

    def evaluate(config: str, family: str, recon_cpu: dict[str, torch.Tensor], binary_meta: dict[str, int],
                 lrr: dict[str, int] | None = None, rescue: dict[str, int] | None = None,
                 rescued: dict[str, int] | None = None, ranks: dict[str, int] | None = None,
                 strategy: str = "none", damage_removed: float = 0.0, exact_index_bits: int = 0) -> None:
        lrr = lrr or {}; rescue = rescue or {}; rescued = rescued or {}; ranks = ranks or {}
        local = []
        for name in originals:
            row = matrix_metrics(name, originals[name], recon_cpu[name], importance[name], config,
                                 originals[name].numel() + binary_meta[name], lrr.get(name, 0), rescue.get(name, 0),
                                 rescued.get(name, 0), ranks.get(name, 0))
            row.update({"family": family, "strategy": strategy, "damage_removed_fraction": damage_removed,
                        "exact_index_bits": exact_index_bits}); local.append(row)
        matrix_rows.extend(local)
        set_weights(modules, recon_cpu); value = perplexity(model, evaluation, device)
        ppl_records.append({"configuration": config, "family": family, "perplexity": value, "strategy": strategy,
                            "damage_removed_fraction": damage_removed, "exact_index_bits": exact_index_bits})
        print(f"PPL {config}: {value:.4f}", flush=True)

    for config, objective, block_size, granularity, family in base_specs:
        print(f"Building {config}", flush=True); rec, meta = {}, {}
        for name in originals:
            q, bits = binary_base(original_gpu[name], imp_gpu[name], block_size, objective, granularity)
            rec[name] = q.cpu(); meta[name] = bits
        base_recons[config] = rec; base_meta[config] = meta
        evaluate(config, family, rec, meta)
    base_aggregate = aggregate_rows(matrix_rows, ["configuration", "family", "strategy", "damage_removed_fraction", "exact_index_bits"])
    hessian_names = [x[0] for x in base_specs if x[4] == "hessian_base"]
    base_ppl = pd.DataFrame(ppl_records)
    selected_base = base_ppl[base_ppl.configuration.isin(hessian_names)].sort_values("perplexity").iloc[0].configuration
    print(f"Selected Q_H: {selected_base}", flush=True)

    states = {}
    for idx, name in enumerate(originals, 1):
        print(f"[{idx}/{len(originals)}] Hessian SVD {name}", flush=True)
        o, b, imp = original_gpu[name], base_recons[selected_base][name].to(device), imp_gpu[name]
        corr = fp16_hessian_corrections(o - b, imp)
        states[name] = {"original": originals[name], "base": base_recons[selected_base][name], "importance": importance[name],
                        "corrections": {r: c.cpu() for r, c in corr.items()},
                        "rank_bits": 16 * (o.shape[0] + o.shape[1] + 1), "binary_meta": base_meta[selected_base][name]}
    del original_gpu, imp_gpu

    rank1 = {name: 1 for name in states}; rec1 = reconstruction_from_ranks(states, rank1)
    evaluate("global_hessian_rank1", "global_lrr", rec1, base_meta[selected_base],
             {n: s["rank_bits"] for n, s in states.items()}, ranks=rank1)
    adaptive_recons, allocations = {}, []
    for budget in ADAPTIVE_BUDGETS:
        ranks = allocate_ranks(states, total_weights, budget); rec = reconstruction_from_ranks(states, ranks)
        key = f"adaptive_lrr_b{budget:.3f}"; adaptive_recons[budget] = rec
        lrr_bits = {n: states[n]["rank_bits"] * ranks[n] for n in states}
        for n in states: allocations.append({"budget_bpw": budget, "matrix": n, "layer": layer_number(n), "projection": projection_type(n), "allocated_rank": ranks[n], "lrr_bits": lrr_bits[n]})
        evaluate(key, "adaptive_lrr", rec, base_meta[selected_base], lrr_bits, ranks=ranks)

    concentration, damage_groups = damage_concentration(states, base_recons[selected_base])
    concentration.to_csv(args.output / "damage_concentration.csv", index=False); damage_groups.to_csv(args.output / "damage_by_matrix.csv", index=False)
    base_damage = sum(float(((s["original"] - s["base"]).square() * s["importance"][None, :]).sum()) for s in states.values())
    index_bits = math.ceil(math.log2(total_weights))
    for strategy in STRATEGIES:
        order, mapping, all_damage = global_order(states, base_recons[selected_base], strategy)
        for frac in FRACTIONS:
            count = math.ceil(frac * total_weights); rec, counts = apply_individual_rescue(states, base_recons[selected_base], order, mapping, count)
            rescue = {n: counts[n] * (16 + index_bits) for n in states}
            if count: rescue[next(iter(states))] += 32
            remaining = sum(float(((states[n]["original"] - rec[n]).square() * states[n]["importance"][None, :]).sum()) for n in states)
            evaluate(f"sparse_{strategy}_p{100*frac:g}", "sparse", rec, base_meta[selected_base], rescue=rescue, rescued=counts,
                     strategy=strategy, damage_removed=1-remaining/base_damage, exact_index_bits=count*index_bits + (32 if count else 0))

    combined_base = adaptive_recons[.05]
    combined_ranks = allocate_ranks(states, total_weights, .05)
    combined_lrr = {n: states[n]["rank_bits"] * combined_ranks[n] for n in states}
    combined_damage = sum(float(((states[n]["original"] - combined_base[n]).square() * states[n]["importance"][None, :]).sum()) for n in states)
    order, mapping, _ = global_order(states, combined_base, "hessian_damage")
    for frac in FRACTIONS:
        count = math.ceil(frac * total_weights); rec, counts = apply_individual_rescue(states, combined_base, order, mapping, count)
        rescue = {n: counts[n] * (16 + index_bits) for n in states}
        if count: rescue[next(iter(states))] += 32
        remaining = sum(float(((states[n]["original"] - rec[n]).square() * states[n]["importance"][None, :]).sum()) for n in states)
        evaluate(f"combined_b0.05_p{100*frac:g}", "combined", rec, base_meta[selected_base], combined_lrr, rescue, counts, combined_ranks,
                 "hessian_damage", 1-remaining/combined_damage, count*index_bits + (32 if count else 0))

    for structure in STRUCTURES:
        units = structured_units(states, base_recons[selected_base], structure)
        for frac in (.001, .005, .01):
            rec, counts, count, unit_index_bits = apply_structured_rescue(states, base_recons[selected_base], units, math.ceil(frac*total_weights), structure)
            total_rescue_bits = count*16 + unit_index_bits + 32
            rescue = {n: counts[n]*16 for n in states}; rescue[next(iter(states))] += unit_index_bits + 32
            remaining = sum(float(((states[n]["original"] - rec[n]).square() * states[n]["importance"][None, :]).sum()) for n in states)
            evaluate(f"structured_{structure}_p{100*frac:g}", "structured", rec, base_meta[selected_base], rescue=rescue, rescued=counts,
                     strategy=structure, damage_removed=1-remaining/base_damage, exact_index_bits=unit_index_bits + 32)

    set_weights(modules, originals)
    raw = pd.DataFrame(matrix_rows); raw.to_csv(args.output / "matrix_results.csv", index=False)
    keys = ["configuration", "family", "strategy", "damage_removed_fraction", "exact_index_bits"]
    aggregate = aggregate_rows(matrix_rows, keys)
    ppl_df = pd.DataFrame(ppl_records)
    results = aggregate.merge(ppl_df[ppl_df.configuration != "bf16"], on=keys, how="outer")
    bf16_row = {c: np.nan for c in results.columns}; bf16_row.update(ppl_records[0]); bf16_row.update({"true_bpw": 16., "weighted_error": 0., "raw_mse": 0., "relative_error": 0., "cosine": 1., "fraction_weights_rescued": 1., "binary_metadata_bpw": 15., "lrr_metadata_bpw": 0., "salient_rescue_bpw": 0.})
    results = pd.concat((pd.DataFrame([bf16_row]), results), ignore_index=True)
    results["perplexity_degradation_pct"] = 100*(results.perplexity/baseline_ppl - 1)
    had_ppl = float(results.loc[results.configuration == selected_base, "perplexity"].iloc[0])
    results["perplexity_recovery_per_added_bit"] = (had_ppl-results.perplexity)/(results.true_bpw-float(results.loc[results.configuration == selected_base, "true_bpw"].iloc[0])).replace(0, np.nan)
    results.to_csv(args.output / "aggregate.csv", index=False)
    ppl_df.to_csv(args.output / "perplexity.csv", index=False)
    results[results.family.isin(["naive_base", "magnitude_base", "hessian_base"])].to_csv(args.output / "binary_ablation.csv", index=False)
    results[results.family.isin(["global_lrr", "adaptive_lrr"])].to_csv(args.output / "adaptive_lrr_ablation.csv", index=False)
    results[results.family.isin(["sparse", "combined"])].to_csv(args.output / "sparse_rescue_ablation.csv", index=False)
    results[results.family == "structured"].to_csv(args.output / "structured_rescue_ablation.csv", index=False)
    aggregate_rows(matrix_rows, ["layer"] + keys).to_csv(args.output / "per_layer.csv", index=False)
    aggregate_rows(matrix_rows, ["projection"] + keys).to_csv(args.output / "per_projection.csv", index=False)
    allocation = pd.DataFrame(allocations); allocation.to_csv(args.output / "adaptive_allocation.csv", index=False)
    make_plots(args.output, results, concentration, allocation, damage_groups); build_report(args.output, results, concentration)
    dump_json(args.output / "environment.json", {"args": vars(args), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "target_weights": total_weights, "selected_hessian_base": selected_base, "timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((args.output / "REPORT.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
