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
from run_experiments import (
    activation_importance,
    dump_json,
    get_text_tokens,
    perplexity,
    seed_everything,
    set_weights,
    target_modules,
)


R1_RANKS = (1, 2, 4, 8)
RANK_PAIRS = ((1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 2), (8, 4))
METHODS2 = ("ordinary", "hessian")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="main")
    p.add_argument("--layers", default="0,12,23")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--evaluation-tokens", type=int, default=4096)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output", type=Path, default=Path("results/experiment3"))
    return p.parse_args()


def projection_type(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def layer_number(name: str) -> int:
    return int(name.split(".layers.", 1)[1].split(".", 1)[0])


def fp16_low_rank_many(residual: torch.Tensor, ranks: tuple[int, ...], weighted_columns: torch.Tensor | None = None) -> dict[int, torch.Tensor]:
    """Fit several ranks with one SVD and round every stored factor to FP16."""
    if weighted_columns is None:
        fitted = residual
        inverse = None
    else:
        root = weighted_columns.sqrt().clamp_min(1e-8)
        fitted = residual * root[None, :]
        inverse = root.reciprocal()
    u, s, vh = torch.linalg.svd(fitted.float(), full_matrices=False)
    corrections = {}
    for rank in ranks:
        u16 = u[:, :rank].half().float()
        s16 = s[:rank].half().float()
        vh16 = vh[:rank].half().float()
        correction = (u16 * s16[None, :]) @ vh16
        if inverse is not None:
            correction = correction * inverse[None, :]
        corrections[rank] = correction
    return corrections


def sensitivity_summary(name: str, configuration: str, residual: torch.Tensor, importance: torch.Tensor,
                        block_size: int, r1: int) -> dict[str, object]:
    damage = residual.square() * importance[None, :]
    total = damage.sum().clamp_min(1e-12)
    col_damage, row_damage = damage.sum(0), damage.sum(1)
    top_cols = max(1, math.ceil(.1 * len(col_damage)))
    top_rows = max(1, math.ceil(.1 * len(row_damage)))
    damage_blocks, _ = matrix_to_blocks(damage, block_size)
    block_damage = damage_blocks.sum(1)
    top_blocks = max(1, math.ceil(.1 * len(block_damage)))
    return {
        "matrix": name, "layer": layer_number(name), "projection": projection_type(name),
        "configuration": configuration, "r1": r1, "total_damage": float(total),
        "column_damage_top10_fraction": float(col_damage.topk(top_cols).values.sum() / total),
        "row_damage_top10_fraction": float(row_damage.topk(top_rows).values.sum() / total),
        "block_damage_top10_fraction": float(block_damage.topk(top_blocks).values.sum() / total),
        "column_damage_cv": float(col_damage.std() / col_damage.mean().clamp_min(1e-12)),
        "row_damage_cv": float(row_damage.std() / row_damage.mean().clamp_min(1e-12)),
    }


def metric_row(name: str, config: str, original: torch.Tensor, reconstruction: torch.Tensor,
               importance: torch.Tensor, base_bits: int, stage1_bits: int, stage2_bits: int,
               r1: int = 0, r2: int = 0, stage2_method: str = "none",
               explained_r1: float = 0.0) -> dict[str, object]:
    residual = original - reconstruction
    original_energy = original.square().sum().clamp_min(1e-12)
    weighted_denom = (original.square() * importance[None, :]).sum().clamp_min(1e-12)
    total_bits = base_bits + stage1_bits + stage2_bits
    return {
        "matrix": name,
        "layer": layer_number(name),
        "projection": projection_type(name),
        "configuration": config,
        "r1": r1,
        "r2": r2,
        "stage2_method": stage2_method,
        "weights": original.numel(),
        "raw_mse": float(residual.square().mean()),
        "relative_error": float(residual.square().sum() / original_energy),
        "weighted_error": float((residual.square() * importance[None, :]).sum() / weighted_denom),
        "cosine": float(torch.nn.functional.cosine_similarity(original.flatten(), reconstruction.flatten(), dim=0)),
        "explained_r1_energy": explained_r1,
        "base_bits": base_bits,
        "stage1_bits": stage1_bits,
        "stage2_bits": stage2_bits,
        "metadata_bits": total_bits - original.numel(),
        "total_bits": total_bits,
        "true_bpw": total_bits / original.numel(),
    }


def aggregate(rows: pd.DataFrame, group: list[str]) -> pd.DataFrame:
    work = rows.copy()
    for metric in ("relative_error", "weighted_error", "raw_mse", "cosine", "explained_r1_energy"):
        work[f"_{metric}"] = work[metric] * work.weights
    result = work.groupby(group, dropna=False).agg(
        matrices=("matrix", "count"), weights=("weights", "sum"), base_bits=("base_bits", "sum"),
        stage1_bits=("stage1_bits", "sum"), stage2_bits=("stage2_bits", "sum"), total_bits=("total_bits", "sum"),
        raw_mse=("_raw_mse", "sum"), relative_error=("_relative_error", "sum"),
        weighted_error=("_weighted_error", "sum"), cosine=("_cosine", "sum"),
        explained_r1_energy=("_explained_r1_energy", "sum"),
    ).reset_index()
    for metric in ("raw_mse", "relative_error", "weighted_error", "cosine", "explained_r1_energy"):
        result[metric] /= result.weights
    result["metadata_bits"] = result.total_bits - result.weights
    result["true_bpw"] = result.total_bits / result.weights
    return result


def plot_results(out: Path, spectra: pd.DataFrame, aggregate_table: pd.DataFrame, ppl: pd.DataFrame) -> None:
    p = out / "plots"; p.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 5))
    for proj, frame in spectra.groupby("projection"):
        curve = frame.groupby("rank").singular_ratio.median()
        plt.semilogy(curve.index, curve.values, label=proj)
    plt.xlabel("singular direction"); plt.ylabel("median singular value / first"); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(p / "residual_singular_spectrum.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 5))
    for proj, frame in spectra.groupby("projection"):
        curve = frame.groupby("rank").cumulative_energy.mean()
        plt.plot(curve.index, curve.values, label=proj)
    plt.xlabel("rank"); plt.ylabel("mean cumulative residual energy"); plt.xticks([1, 2, 4, 8, 16]); plt.legend(ncol=2); plt.tight_layout()
    plt.savefig(p / "cumulative_residual_energy.png", dpi=180); plt.close()

    first = ppl[(ppl.configuration == "hadamard_naive") | ((ppl.r1 > 0) & (ppl.r2 == 0))].sort_values("r1")
    plt.figure(figsize=(7, 5)); plt.plot(first.r1, first.perplexity, marker="o"); plt.xlabel("first-stage rank"); plt.ylabel("perplexity"); plt.xticks([0, 1, 2, 4, 8]); plt.tight_layout()
    plt.savefig(p / "perplexity_vs_first_rank.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 5))
    low_bit = ppl[ppl.true_bpw <= 2]
    for method, frame in low_bit.groupby("stage2_method"):
        plt.scatter(frame.true_bpw, frame.perplexity, label=method, s=45)
    bf16_ppl = float(ppl.loc[ppl.configuration == "bf16", "perplexity"].iloc[0])
    plt.axhline(bf16_ppl, color="black", linestyle="--", linewidth=1, label=f"BF16 ({bf16_ppl:.1f})")
    plt.xlabel("true BPW"); plt.ylabel("perplexity"); plt.yscale("log"); plt.xlim(.98, 1.32); plt.legend(); plt.tight_layout()
    plt.savefig(p / "perplexity_vs_bpw.png", dpi=180); plt.close()

    second = aggregate_table[aggregate_table.r2 > 0]
    plt.figure(figsize=(8, 5))
    labels, before, after_o, after_h = [], [], [], []
    for r1, r2 in RANK_PAIRS:
        o = second[(second.r1 == r1) & (second.r2 == r2) & (second.stage2_method == "ordinary")]
        h = second[(second.r1 == r1) & (second.r2 == r2) & (second.stage2_method == "hessian")]
        base = aggregate_table[(aggregate_table.r1 == r1) & (aggregate_table.r2 == 0)]
        if len(o) and len(h) and len(base):
            labels.append(f"{r1}+{r2}"); before.append(base.weighted_error.iloc[0]); after_o.append(o.weighted_error.iloc[0]); after_h.append(h.weighted_error.iloc[0])
    x = np.arange(len(labels)); width = .26
    plt.bar(x-width, before, width, label="before stage 2"); plt.bar(x, after_o, width, label="ordinary"); plt.bar(x+width, after_h, width, label="Hessian")
    plt.xticks(x, labels); plt.ylabel("sensitivity-weighted error"); plt.xlabel("r1+r2"); plt.legend(); plt.tight_layout()
    plt.savefig(p / "weighted_error_second_stage.png", dpi=180); plt.close()

    chosen = ppl[(ppl.configuration.isin(["hadamard_naive", "rank4", "rank4_ordinary_rank2", "rank4_hessian_rank2"]))]
    plt.figure(figsize=(8, 5)); plt.bar(chosen.configuration, chosen.perplexity); plt.yscale("log"); plt.ylabel("perplexity"); plt.xticks(rotation=18, ha="right"); plt.tight_layout()
    plt.savefig(p / "method_perplexity_comparison.png", dpi=180); plt.close()


def build_report(out: Path, aggregate_table: pd.DataFrame, projection_table: pd.DataFrame,
                 spectra: pd.DataFrame, ppl: pd.DataFrame, sensitivity: pd.DataFrame) -> None:
    energy = spectra[spectra["rank"].isin([1, 2, 4, 8, 16])].groupby("rank").cumulative_energy.mean()
    rank8_energy = float(energy.loc[8])
    had_ppl = float(ppl.loc[ppl.configuration == "hadamard_naive", "perplexity"].iloc[0])
    bf16 = float(ppl.loc[ppl.configuration == "bf16", "perplexity"].iloc[0])
    candidates = ppl[(ppl.r1 > 0)].copy()
    candidates["recovery_fraction"] = (had_ppl - candidates.perplexity) / max(had_ppl - bf16, 1e-12)
    candidates["recovery_per_added_bit"] = (had_ppl - candidates.perplexity) / candidates.added_metadata_bits.clip(lower=1)
    best_quality = candidates.sort_values("perplexity").iloc[0]
    best_efficiency = candidates.sort_values("recovery_per_added_bit", ascending=False).iloc[0]
    paired = ppl[ppl.stage2_method.isin(METHODS2)].pivot(index=["r1", "r2"], columns="stage2_method", values="perplexity").dropna()
    hessian_wins = float((paired.hessian < paired.ordinary).mean()) if len(paired) else 0.0
    harmful_top10 = float(sensitivity.loc[sensitivity.r1 == 0, "column_damage_top10_fraction"].mean())
    q1 = rank8_energy >= .20
    q2 = best_quality.recovery_fraction >= .25
    q3 = hessian_wins >= .625 and float((paired.ordinary - paired.hessian).median()) > 0
    dsa_row = {"configuration": "hadamard_dsa (Exp. 1)", "perplexity": 6482.09914360287, "true_bpw": 1.00604}
    eligible = candidates[candidates.true_bpw <= 1.05]
    best_capped = eligible.sort_values("perplexity").iloc[0] if len(eligible) else None
    q4 = bool(best_capped is not None and best_capped.perplexity < dsa_row["perplexity"])
    table_cols = ["configuration", "r1", "r2", "stage2_method", "true_bpw", "relative_error", "weighted_error", "cosine"]
    text = f"""# Experiment 3: low-rank residual reconstruction

## Scope

Qwen2.5-0.5B layers 0, 12, and 23; all Q/K/V/O and MLP gate/up/down projections; WikiText-2; seed 20260925; 128x128 Hadamard blocks. Every low-rank factor is rounded to FP16 and U, S, and V storage is charged exactly.

## Residual energy

{energy.rename('mean_cumulative_energy').reset_index().to_markdown(index=False, floatfmt='.6g')}

Mean rank-8 captured energy is **{rank8_energy:.2%}**. The mean top-10%-column share of sensitivity-weighted residual damage is **{harmful_top10:.2%}**.

## Aggregate reconstruction results

{aggregate_table[table_cols].to_markdown(index=False, floatfmt='.6g')}

## Perplexity and storage

{ppl.to_markdown(index=False, floatfmt='.6g')}

Best quality: **{best_quality.configuration}**, perplexity **{best_quality.perplexity:.4f}**, true BPW **{best_quality.true_bpw:.6f}**, recovery of the Hadamard-1-bit excess perplexity **{best_quality.recovery_fraction:.2%}**.

Best measured perplexity recovery per added metadata bit: **{best_efficiency.configuration}** ({best_efficiency.recovery_per_added_bit:.6g} perplexity points/bit).

Hessian-guided stage 2 beats ordinary stage 2 at equal ranks in **{hessian_wins:.1%}** of tested pairs.

## Answers

**Q1: Is the 1-bit residual low-rank enough to exploit? {'YES' if q1 else 'NO'}** under the preregistered practical rule that rank 8 captures at least 20% of residual energy.

**Q2: Does first LRR materially recover model quality? {'YES' if q2 else 'NO'}** under the rule that it recovers at least 25% of the Hadamard-1-bit excess perplexity.

**Q3: Does Hessian-guided second LRR recover more quality per metadata bit than ordinary LRR? {'YES' if q3 else 'NO'}** under the rule that it wins at equal storage in at least five of eight rank pairs with a positive median perplexity advantage.

**Q4: Should LRR replace DSA as the main reconstruction mechanism? {'YES' if q4 else 'NO'}** The best configuration at or below 1.05 BPW is **{best_capped.configuration if best_capped is not None else 'none'}** with perplexity **{best_capped.perplexity if best_capped is not None else float('nan'):.4f}**, compared with Experiment 1 Hadamard+DSA at **{dsa_row['perplexity']:.4f}** and {dsa_row['true_bpw']:.6f} BPW.

These conclusions apply to this representative-layer intervention and the tested FP16 factorization ranks. They are not whole-model compression claims.
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
    reconstructions: dict[str, dict[str, torch.Tensor]] = {"naive": {}, "hadamard_naive": {}}
    rows, spectra_rows, sensitivity_rows = [], [], []
    total_weights = sum(x.numel() for x in originals.values())
    total_base_metadata = 0
    for number, (name, module) in enumerate(modules.items(), 1):
        print(f"[{number}/{len(modules)}] SVD analysis: {name}", flush=True)
        original = module.weight.detach().float()
        blocks, layout = matrix_to_blocks(original, args.block_size)
        naive_blocks, _ = reconstruct_naive(blocks, layout.valid)
        naive = blocks_to_matrix(naive_blocks, layout)
        transformed = hadamard_blocks(blocks, args.block_size)
        transformed_q, _ = reconstruct_naive(transformed, layout.valid)
        had_base = blocks_to_matrix(hadamard_blocks(transformed_q, args.block_size), layout)
        reconstructions["naive"][name] = naive.cpu(); reconstructions["hadamard_naive"][name] = had_base.cpu()
        base_metadata = 16 * len(blocks); base_bits = original.numel() + base_metadata; total_base_metadata += base_metadata
        imp = importance[name].to(device).float().clamp_min(1e-12)
        rows.append(metric_row(name, "naive", original, naive, imp, base_bits, 0, 0))
        rows.append(metric_row(name, "hadamard_naive", original, had_base, imp, base_bits, 0, 0))
        r1_full = original - had_base
        u, s, vh = torch.linalg.svd(r1_full, full_matrices=False)
        energy_total = s.square().sum().clamp_min(1e-12)
        cumulative = s.square().cumsum(0) / energy_total
        for idx in range(min(64, len(s))):
            spectra_rows.append({"matrix": name, "layer": layer_number(name), "projection": projection_type(name), "rank": idx + 1,
                                 "singular_value": float(s[idx]), "singular_ratio": float(s[idx] / s[0]), "cumulative_energy": float(cumulative[idx])})
        sensitivity_rows.append(sensitivity_summary(name, "hadamard_naive", r1_full, imp, args.block_size, 0))
        for r1 in R1_RANKS:
            # Reuse exact singular directions, but quantize all stored factors to FP16.
            corr1 = (u[:, :r1].half().float() * s[:r1].half().float()[None, :]) @ vh[:r1].half().float()
            rec1 = had_base + corr1
            key1 = f"rank{r1}"; reconstructions.setdefault(key1, {})[name] = rec1.cpu()
            bits1 = 16 * r1 * (original.shape[0] + original.shape[1] + 1)
            rows.append(metric_row(name, key1, original, rec1, imp, base_bits, bits1, 0, r1=r1, explained_r1=float(cumulative[r1 - 1])))
            residual2 = original - rec1
            sensitivity_rows.append(sensitivity_summary(name, key1, residual2, imp, args.block_size, r1))
            second_ranks = tuple(r2 for pair_r1, r2 in RANK_PAIRS if pair_r1 == r1)
            corrections = {
                "ordinary": fp16_low_rank_many(residual2, second_ranks),
                "hessian": fp16_low_rank_many(residual2, second_ranks, imp),
            }
            for r2 in second_ranks:
                for method2 in METHODS2:
                    corr2 = corrections[method2][r2]
                    rec2 = rec1 + corr2
                    key2 = f"rank{r1}_{method2}_rank{r2}"; reconstructions.setdefault(key2, {})[name] = rec2.cpu()
                    bits2 = 16 * r2 * (original.shape[0] + original.shape[1] + 1)
                    rows.append(metric_row(name, key2, original, rec2, imp, base_bits, bits1, bits2,
                                           r1=r1, r2=r2, stage2_method=method2, explained_r1=float(cumulative[r1 - 1])))
    raw = pd.DataFrame(rows); spectra = pd.DataFrame(spectra_rows); sensitivity = pd.DataFrame(sensitivity_rows)
    raw.to_csv(args.output / "matrix_results.csv", index=False); spectra.to_csv(args.output / "singular_spectra.csv", index=False); sensitivity.to_csv(args.output / "sensitivity_structure.csv", index=False)
    agg = aggregate(raw, ["configuration", "r1", "r2", "stage2_method"])
    per_layer = aggregate(raw, ["layer", "configuration", "r1", "r2", "stage2_method"])
    per_projection = aggregate(raw, ["projection", "configuration", "r1", "r2", "stage2_method"])
    agg.to_csv(args.output / "aggregate.csv", index=False); per_layer.to_csv(args.output / "per_layer.csv", index=False); per_projection.to_csv(args.output / "per_projection.csv", index=False)
    baseline_ppl = perplexity(model, evaluation, device)
    ppl_rows = [{"configuration": "bf16", "r1": 0, "r2": 0, "stage2_method": "none", "perplexity": baseline_ppl,
                 "delta_bf16": 0.0, "raw_mse": 0.0, "relative_error": 0.0, "weighted_error": 0.0,
                 "cosine": 1.0, "true_bpw": 16.0, "added_metadata_bits": 0}]
    ordered = ["naive", "hadamard_naive"] + [f"rank{r}" for r in R1_RANKS]
    ordered += [f"rank{r1}_{m}_rank{r2}" for r1, r2 in RANK_PAIRS for m in METHODS2]
    for idx, config in enumerate(ordered, 1):
        print(f"[{idx}/{len(ordered)}] Perplexity: {config}", flush=True)
        set_weights(modules, reconstructions[config])
        value = perplexity(model, evaluation, device)
        info = agg[agg.configuration == config].iloc[0]
        ppl_rows.append({"configuration": config, "r1": int(info.r1), "r2": int(info.r2), "stage2_method": info.stage2_method,
            "perplexity": value, "delta_bf16": value - baseline_ppl, "raw_mse": float(info.raw_mse),
            "relative_error": float(info.relative_error), "weighted_error": float(info.weighted_error),
            "cosine": float(info.cosine), "true_bpw": float(info.true_bpw),
            "added_metadata_bits": int(info.stage1_bits + info.stage2_bits)})
    set_weights(modules, originals)
    ppl = pd.DataFrame(ppl_rows)
    had_ppl = float(ppl.loc[ppl.configuration == "hadamard_naive", "perplexity"].iloc[0])
    ppl["recovery_from_hadamard"] = had_ppl - ppl.perplexity
    ppl["recovery_fraction"] = ppl.recovery_from_hadamard / max(had_ppl - baseline_ppl, 1e-12)
    ppl["recovery_per_added_metadata_bit"] = np.where(
        ppl.added_metadata_bits > 0, ppl.recovery_from_hadamard / ppl.added_metadata_bits, np.nan)
    ppl.to_csv(args.output / "perplexity.csv", index=False)
    plot_results(args.output, spectra, agg, ppl); build_report(args.output, agg, per_projection, spectra, ppl, sensitivity)
    dump_json(args.output / "environment.json", {"args": vars(args), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "target_weights": total_weights,
        "base_metadata_bits": total_base_metadata, "timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    print((args.output / "REPORT.md").read_text())


if __name__ == "__main__":
    main()
