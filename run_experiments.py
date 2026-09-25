from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from dsaq.core import (
    FAMILY_PARAMS,
    blocks_to_matrix,
    candidate_metadata_bits,
    fit_router,
    hadamard_blocks,
    matrix_to_blocks,
    raw_relative_error,
    reconstruct_dsa,
    reconstruct_naive,
    seeded_candidates,
    select_seeded,
    split_is_calibration,
    structure_features,
    true_bpw,
    weighted_error,
)


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
    p.add_argument("--output", type=Path, default=Path("results/run"))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--skip-perplexity", action="store_true")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def dump_json(path: Path, value: object) -> None:
    def convert(x: object):
        if isinstance(x, (np.floating, np.integer)): return x.item()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, torch.Tensor): return x.detach().cpu().tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert), encoding="utf-8")


def get_text_tokens(tokenizer, split: str, count: int, seq_len: int) -> torch.Tensor:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(x for x in ds["text"] if x.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    usable = min((len(ids) // seq_len) * seq_len, (count // seq_len) * seq_len)
    return ids[:usable].view(-1, seq_len)


def target_modules(model: torch.nn.Module, layers: set[int]) -> dict[str, torch.nn.Linear]:
    suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    result = {}
    for name, module in model.named_modules():
        match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
        if match and int(match.group(1)) in layers and isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in suffixes:
            result[name] = module
    if not result:
        raise RuntimeError("No target projection modules found")
    return result


@torch.inference_mode()
def activation_importance(model, modules: dict[str, torch.nn.Linear], tokens: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}; counts: dict[str, int] = defaultdict(int); handles = []
    for name, module in modules.items():
        def hook(_module, args, key=name):
            x = args[0].detach().float()
            dims = tuple(range(x.ndim - 1))
            value = x.square().sum(dims).cpu()
            sums[key] = sums.get(key, torch.zeros_like(value)) + value
            counts[key] += int(np.prod(x.shape[:-1]))
        handles.append(module.register_forward_pre_hook(hook))
    try:
        for batch in tokens:
            model(batch[None].to(device), use_cache=False)
    finally:
        for h in handles: h.remove()
    return {k: (sums[k] / counts[k]).clamp_min(1e-12) for k in modules}


def importance_blocks(column_importance: torch.Tensor, layout, device: torch.device) -> torch.Tensor:
    b = layout.block_size
    padded = torch.zeros(layout.padded_cols, dtype=torch.float32, device=device)
    padded[: layout.cols] = column_importance.to(device)
    cols = padded.view(layout.padded_cols // b, b)
    grid = cols[None].expand(layout.padded_rows // b, -1, -1)
    return grid[:, :, None, :].expand(-1, -1, b, -1).reshape(-1, b * b)


def block_row(name: str, block_id: int, method: str, original: torch.Tensor, recon: torch.Tensor, valid: torch.Tensor,
              importance: torch.Tensor, metadata: int, nweights: int) -> dict[str, object]:
    raw = float(raw_relative_error(original, recon, valid)[0])
    weighted = float(weighted_error(original, recon, importance, valid)[0])
    mse = float((((original - recon).square() * valid).sum() / valid.sum().clamp_min(1)))
    cos = float(torch.nn.functional.cosine_similarity(original * valid, recon * valid, dim=1)[0])
    return {"matrix": name, "block_id": block_id, "method": method, "weights": nweights, "raw_mse": mse,
            "relative_error": raw, "weighted_error": weighted, "cosine": cos, "metadata_bits": metadata,
            "true_bpw": true_bpw(nweights, metadata)}


def analyze_matrix(name: str, weight: torch.Tensor, column_importance: torch.Tensor, block_size: int) -> tuple[list[dict], dict]:
    blocks, layout = matrix_to_blocks(weight.float(), block_size)
    valid = layout.valid
    imp = importance_blocks(column_importance, layout, weight.device)
    naive, _ = reconstruct_naive(blocks, valid)
    dsa, dsa_err, families, param_counts = reconstruct_dsa(blocks, valid, block_size)
    transformed = hadamard_blocks(blocks, block_size)
    h_naive_t, _ = reconstruct_naive(transformed, valid)
    h_dsa_t, _, h_families, h_params = reconstruct_dsa(transformed, valid, block_size)
    h_naive = hadamard_blocks(h_naive_t, block_size)
    h_dsa = hadamard_blocks(h_dsa_t, block_size)
    features, feature_names = structure_features(blocks, valid, block_size, dsa_err)
    naive_err = raw_relative_error(blocks, naive, valid)
    gain = (naive_err - raw_relative_error(blocks, dsa, valid)).cpu().numpy()
    rows: list[dict] = []
    for i in range(len(blocks)):
        v = valid[i:i+1]; o = blocks[i:i+1]; ii = imp[i:i+1]; n = int(v.sum())
        entries = [
            ("naive", naive[i:i+1], 16, "constant", 1),
            ("hadamard_naive", h_naive[i:i+1], 16, "constant", 1),
            ("dsa", dsa[i:i+1], 3 + 16 * int(param_counts[i]), families[i], int(param_counts[i])),
            ("hadamard_dsa", h_dsa[i:i+1], 3 + 16 * int(h_params[i]), h_families[i], int(h_params[i])),
        ]
        for method, recon, metadata, family, params in entries:
            row = block_row(name, i, method, o, recon, v, ii, metadata, n)
            row.update({"family": family, "parameter_count": params, "calibration_block": bool(split_is_calibration(name, torch.tensor([i], device=weight.device))[0])})
            for col, value in zip(feature_names, features[i]): row[col] = float(value)
            row["dsa_gain"] = float(gain[i])
            rows.append(row)
    cache = {"layout": layout, "blocks": blocks.cpu(), "valid": valid.cpu(), "importance": imp.cpu(),
             "naive": naive.cpu(), "dsa": dsa.cpu(), "hadamard_naive": h_naive.cpu(), "hadamard_dsa": h_dsa.cpu(),
             "features": features, "feature_names": feature_names, "gain": gain}
    return rows, cache


def fit_global_router(exp1: pd.DataFrame, caches: dict[str, dict]) -> dict:
    feature_names = next(iter(caches.values()))["feature_names"]
    features, gains, cal, keys = [], [], [], []
    for name, cache in caches.items():
        features.append(cache["features"]); gains.append(cache["gain"])
        block_ids = torch.arange(len(cache["gain"]))
        cal.append(split_is_calibration(name, block_ids).cpu().numpy()); keys.extend((name, int(i)) for i in block_ids)
    x, y, split = np.concatenate(features), np.concatenate(gains), np.concatenate(cal)
    fit = fit_router(x, y, split)
    score_map = {key: float(score) for key, score in zip(keys, fit["scores"])}
    truth = y[~split] >= fit["minimum_useful_gain"]
    pred = fit["scores"][~split] > fit["threshold"]
    metrics = {
        "feature_names": feature_names, "mean": fit["mean"], "scale": fit["scale"], "coef": fit["coef"],
        "threshold": fit["threshold"], "minimum_useful_gain": fit["minimum_useful_gain"], "heldout_n": int((~split).sum()),
        "heldout_pearson_r": float(stats.pearsonr(fit["scores"][~split], y[~split]).statistic),
        "heldout_spearman_r": float(stats.spearmanr(fit["scores"][~split], y[~split]).statistic),
        "heldout_accuracy": float((truth == pred).mean()),
        "heldout_balanced_accuracy": float(((pred[truth].mean() if truth.any() else 0) + ((~pred)[~truth].mean() if (~truth).any() else 0)) / 2),
    }
    return {"fit": fit, "metrics": metrics, "score_map": score_map}


def run_seeded(caches: dict[str, dict], router: dict, seed: int) -> tuple[pd.DataFrame, dict[str, dict[str, torch.Tensor]]]:
    rows, recon_cache = [], {}
    for name, cache in caches.items():
        blocks = cache["blocks"].cuda(); valid = cache["valid"].cuda(); imp = cache["importance"].cuda()
        scores = np.asarray([router["score_map"][(name, i)] for i in range(len(blocks))])
        heldout = ~split_is_calibration(name, torch.arange(len(blocks))).cpu().numpy()
        irregular = scores <= router["metrics"]["threshold"]
        ids = np.where(irregular)[0]
        heldout_ids = np.where(irregular & heldout)[0]
        recon_cache[name] = {}
        if not len(ids): continue
        global_values = blocks.abs()[valid]
        global_stats = (float(global_values.mean()), float(global_values.std()))
        for distribution in ("uniform", "gaussian", "robust", "global"):
            for k in (2, 4, 8, 16):
                candidates = seeded_candidates(blocks[ids], valid[ids], ids, name, distribution, k, seed, global_stats)
                for criterion in ("raw", "weighted"):
                    selected, winners = select_seeded(candidates, blocks[ids], valid[ids], imp[ids], criterion)
                    key = f"{distribution}_k{k}_{criterion}"
                    hybrid = cache["dsa"].clone(); hybrid[ids] = selected.cpu(); recon_cache[name][key] = hybrid
                    local_lookup = {block_id: pos for pos, block_id in enumerate(ids)}
                    for block_id in heldout_ids:
                        pos = local_lookup[block_id]; v = valid[block_id:block_id+1]; o = blocks[block_id:block_id+1]; ii = imp[block_id:block_id+1]
                        n = int(v.sum()); metadata = math.ceil(math.log2(k)) + (0 if distribution == "global" else 32) + 1
                        if distribution == "global" and block_id == heldout_ids[0]:
                            metadata += 32  # one shared FP16 mean/std pair per participating matrix
                        row = block_row(name, int(block_id), key, o, selected[pos:pos+1], v, ii, metadata, n)
                        row.update({"distribution": distribution, "k": k, "criterion": criterion, "candidate_index": int(winners[pos]),
                                    "dsa_relative_error": float(raw_relative_error(o, cache["dsa"][block_id:block_id+1].cuda(), v)[0]),
                                    "dsa_weighted_error": float(weighted_error(o, cache["dsa"][block_id:block_id+1].cuda(), ii, v)[0])})
                        rows.append(row)
                del candidates
    return pd.DataFrame(rows), recon_cache


@torch.inference_mode()
def perplexity(model, tokens: torch.Tensor, device: torch.device) -> float:
    losses = []
    for sequence in tokens:
        out = model(sequence[None].to(device), labels=sequence[None].to(device), use_cache=False)
        losses.append(float(out.loss))
    return float(math.exp(np.mean(losses)))


def set_weights(modules: dict[str, torch.nn.Linear], matrices: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in matrices.items(): modules[name].weight.copy_(value.to(modules[name].weight.device, modules[name].weight.dtype))


def summarize(exp1: pd.DataFrame, exp2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def aggregate(frame: pd.DataFrame, group: list[str]) -> pd.DataFrame:
        temp = frame.copy()
        temp["weighted_sse"] = temp.weighted_error * temp.weights
        temp["relative_sse"] = temp.relative_error * temp.weights
        temp["sign_bits"] = temp.weights
        return temp.groupby(group, dropna=False).agg(blocks=("block_id", "count"), weights=("weights", "sum"), raw_mse=("raw_mse", "mean"),
            relative_error=("relative_sse", lambda x: x.sum() / temp.loc[x.index, "weights"].sum()),
            weighted_error=("weighted_sse", lambda x: x.sum() / temp.loc[x.index, "weights"].sum()),
            metadata_bits=("metadata_bits", "sum"), sign_bits=("sign_bits", "sum"), cosine=("cosine", "mean")).reset_index()
    a1 = aggregate(exp1, ["method"]); a1["true_bpw"] = (a1.sign_bits + a1.metadata_bits) / a1.weights
    a2 = aggregate(exp2, ["distribution", "k", "criterion"]); a2["true_bpw"] = (a2.sign_bits + a2.metadata_bits + 64) / a2.weights
    return a1, a2


def make_plots(out: Path, exp1: pd.DataFrame, exp2: pd.DataFrame, router: dict) -> None:
    plot_dir = out / "plots"; plot_dir.mkdir(exist_ok=True)
    held = exp1[(exp1.method == "dsa") & (~exp1.calibration_block)]
    scores = np.asarray([router["score_map"][(r.matrix, int(r.block_id))] for r in held.itertuples()])
    plt.figure(figsize=(7, 5)); plt.hexbin(scores, held.dsa_gain, gridsize=45, bins="log", mincnt=1); plt.axhline(0, color="black", lw=1); plt.axvline(router["metrics"]["threshold"], color="red", ls="--")
    plt.xlabel("DSA suitability score (held-out)"); plt.ylabel("relative-error gain over naive"); plt.colorbar(label="log count"); plt.tight_layout(); plt.savefig(plot_dir / "structure_score_vs_dsa_gain.png", dpi=180); plt.close()
    methods = ["naive", "hadamard_naive", "dsa", "hadamard_dsa"]
    values = [exp1[exp1.method == m].relative_error.values for m in methods]
    plt.figure(figsize=(8, 5)); plt.boxplot(values, tick_labels=methods, showfliers=False); plt.ylabel("relative reconstruction error"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(plot_dir / "experiment1_error_distributions.png", dpi=180); plt.close()
    grouped = exp2.groupby(["k", "criterion"]).weighted_error.mean().reset_index()
    plt.figure(figsize=(7, 5))
    for criterion, g in grouped.groupby("criterion"): plt.plot(g.k, g.weighted_error, marker="o", label=criterion)
    plt.xlabel("candidate count K"); plt.ylabel("mean sensitivity-weighted error"); plt.xticks([2, 4, 8, 16]); plt.legend(); plt.tight_layout(); plt.savefig(plot_dir / "seeded_search_by_k.png", dpi=180); plt.close()


def write_report(out: Path, aggregate1: pd.DataFrame, aggregate2: pd.DataFrame, router: dict, exp2: pd.DataFrame, ppl: pd.DataFrame) -> None:
    m = router["metrics"]
    best = aggregate2.sort_values(["weighted_error", "true_bpw"]).iloc[0]
    irregular = exp2[(exp2.distribution == best.distribution) & (exp2.k == best.k) & (exp2.criterion == best.criterion)]
    wins = float((irregular.weighted_error < irregular.dsa_weighted_error).mean())
    q1 = m["heldout_spearman_r"] > .3 and m["heldout_balanced_accuracy"] > .6
    q2 = wins > .5 and best.weighted_error < (irregular.dsa_weighted_error * irregular.weights).sum() / irregular.weights.sum()
    text = f"""# Experiments 1 and 2 report

## Scope

Qwen2.5-0.5B layers 0, 12, and 23; every Q/K/V/O and MLP gate/up/down projection; fixed 128x128 blocks and WikiText-2 splits. This is a representative-layer intervention, not a whole-model compression result.

## Experiment 1

{aggregate1.to_markdown(index=False, floatfmt='.6g')}

Held-out suitability Pearson r: **{m['heldout_pearson_r']:.4f}**. Spearman r: **{m['heldout_spearman_r']:.4f}**. Balanced classification accuracy: **{m['heldout_balanced_accuracy']:.4f}** on **{m['heldout_n']}** blocks.

## Experiment 2

{aggregate2.to_markdown(index=False, floatfmt='.6g')}

Best sensitivity-weighted configuration is **{best.distribution}, K={int(best.k)}, selection={best.criterion}**, true BPW **{best.true_bpw:.6f}** on the irregular-block payload. It beats DSA weighted error on **{wins:.2%}** of held-out irregular blocks.

## Perplexity

{ppl.to_markdown(index=False, floatfmt='.6g') if len(ppl) else 'Perplexity was skipped.'}

## Answers

**Q1: Can we reliably detect blocks where DSA is worth using? {'YES' if q1 else 'NO'} under the preregistered practical rule (held-out Spearman > 0.30 and balanced accuracy > 0.60).**

**Q2: Does seeded search justify itself on DSA failures? {'YES' if q2 else 'NO'} under the preregistered rule (wins on >50% of held-out irregular blocks and lowers weighted aggregate error).**

These answers apply only to the tested DSA surfaces, distributions, model, block size, and representative layers.
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def synthetic_run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    rng = torch.Generator().manual_seed(args.seed)
    records, caches = [], {}
    b = args.block_size
    for layer in range(3):
        axis = torch.linspace(-1, 1, b)
        rr, cc = torch.meshgrid(axis, axis, indexing="ij")
        blocks = []
        for i in range(12):
            mag = (1 + .3 * rr + .2 * cc).clamp_min(.05) if i < 6 else torch.rand((b, b), generator=rng) * 2
            blocks.append((torch.randint(0, 2, (b, b), generator=rng) * 2 - 1) * mag)
        matrix = torch.cat([torch.cat(blocks[:6], 1), torch.cat(blocks[6:], 1)], 0)
        name = f"synthetic.layers.{layer}.q_proj"
        rows, cache = analyze_matrix(name, matrix.cuda() if torch.cuda.is_available() else matrix, torch.ones(matrix.shape[1]), b)
        records.extend(rows); caches[name] = cache
    exp1 = pd.DataFrame(records); router = fit_global_router(exp1, caches); exp2, _ = run_seeded(caches, router, args.seed) if torch.cuda.is_available() else (pd.DataFrame(), {})
    exp1.to_csv(args.output / "experiment1_blocks.csv", index=False); dump_json(args.output / "router.json", router["metrics"])
    if len(exp2): exp2.to_csv(args.output / "experiment2_blocks.csv", index=False)
    print((args.output / "router.json").read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args(); seed_everything(args.seed)
    if args.synthetic: synthetic_run(args); return
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for the primary run")
    args.output.mkdir(parents=True, exist_ok=True); (args.output / "plots").mkdir(exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16).to(device).eval()
    layers = {int(x) for x in args.layers.split(",")}; modules = target_modules(model, layers)
    calibration = get_text_tokens(tokenizer, "train", args.calibration_tokens, args.sequence_length)
    evaluation = get_text_tokens(tokenizer, "test", args.evaluation_tokens, args.sequence_length)
    importance = activation_importance(model, modules, calibration, device)
    originals = {name: module.weight.detach().float().cpu() for name, module in modules.items()}
    all_rows, caches = [], {}
    for number, (name, module) in enumerate(modules.items(), 1):
        print(f"[{number}/{len(modules)}] Experiment 1: {name}", flush=True)
        rows, cache = analyze_matrix(name, module.weight.detach(), importance[name], args.block_size)
        all_rows.extend(rows); caches[name] = cache
    exp1 = pd.DataFrame(all_rows); exp1.to_csv(args.output / "experiment1_blocks.csv", index=False)
    router = fit_global_router(exp1, caches); dump_json(args.output / "router.json", router["metrics"])
    exp2, seeded_cache = run_seeded(caches, router, args.seed); exp2.to_csv(args.output / "experiment2_blocks.csv", index=False)
    aggregate1, aggregate2 = summarize(exp1, exp2)
    aggregate1.to_csv(args.output / "experiment1_aggregate.csv", index=False); aggregate2.to_csv(args.output / "experiment2_aggregate.csv", index=False)
    layer1 = exp1.groupby(["matrix", "method"]).agg(blocks=("block_id", "count"), weights=("weights", "sum"), relative_error=("relative_error", "mean"), weighted_error=("weighted_error", "mean"), metadata_bits=("metadata_bits", "sum"), true_bpw=("true_bpw", "mean")).reset_index()
    layer2 = exp2.groupby(["matrix", "distribution", "k", "criterion"]).agg(blocks=("block_id", "count"), weights=("weights", "sum"), relative_error=("relative_error", "mean"), weighted_error=("weighted_error", "mean"), metadata_bits=("metadata_bits", "sum"), true_bpw=("true_bpw", "mean")).reset_index()
    layer1.to_csv(args.output / "experiment1_per_layer.csv", index=False); layer2.to_csv(args.output / "experiment2_per_layer.csv", index=False)
    make_plots(args.output, exp1, exp2, router)
    ppl_rows = []
    if not args.skip_perplexity:
        baseline = perplexity(model, evaluation, device); ppl_rows.append({"configuration": "bf16", "perplexity": baseline, "delta": 0.0})
        for method in ("naive", "hadamard_naive", "dsa", "hadamard_dsa"):
            set_weights(modules, {name: blocks_to_matrix(cache[method].cuda(), cache["layout"]) for name, cache in caches.items()})
            value = perplexity(model, evaluation, device); ppl_rows.append({"configuration": method, "perplexity": value, "delta": value - baseline})
        # PPL for each K and selection rule, using the distribution with lowest held-out weighted error for that pair.
        best_by_pair = aggregate2.sort_values("weighted_error").groupby(["k", "criterion"], as_index=False).first()
        for row in best_by_pair.itertuples():
            key = f"{row.distribution}_k{int(row.k)}_{row.criterion}"
            set_weights(modules, {name: blocks_to_matrix(seeded_cache[name].get(key, caches[name]["dsa"]).cuda(), caches[name]["layout"]) for name in modules})
            value = perplexity(model, evaluation, device); ppl_rows.append({"configuration": key, "perplexity": value, "delta": value - baseline})
        set_weights(modules, originals)
    ppl = pd.DataFrame(ppl_rows); ppl.to_csv(args.output / "perplexity.csv", index=False)
    write_report(args.output, aggregate1, aggregate2, router, exp2, ppl)
    metadata = {"args": vars(args), "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(), "target_matrices": list(modules), "timestamp_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    dump_json(args.output / "environment.json", metadata)
    print((args.output / "REPORT.md").read_text(encoding="utf-8"))


if __name__ == "__main__": main()
