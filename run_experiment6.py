from __future__ import annotations

import argparse
import gc
import math
import platform
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dsaq.core import blocks_to_matrix, hadamard_blocks, matrix_to_blocks, reconstruct_naive
from run_experiments import activation_importance, dump_json, get_text_tokens, seed_everything, target_modules


PCTS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
RANK_PCTS = tuple(sorted(set(PCTS + (0.33,))))
PRIMARY = ("magnitude", "activation", "functional_fisher")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6 Part 1: oracle importance selection")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="main")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--evaluation-tokens", type=int, default=4096)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output", type=Path, default=Path("results/experiment6"))
    return p.parse_args()


def layer_of(name: str) -> int:
    return int(name.split(".layers.", 1)[1].split(".", 1)[0])


def projection_of(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def quantize_hadamard(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    blocks, layout = matrix_to_blocks(weight.float(), block_size)
    transformed = hadamard_blocks(blocks, block_size)
    _, scales = reconstruct_naive(transformed, layout.valid)
    scales = scales.half().float()
    reconstructed = hadamard_blocks(transformed.sign() * scales[:, None], block_size)
    return blocks_to_matrix(reconstructed, layout).bfloat16().cpu()


def safe_perplexity(model, tokens: torch.Tensor, device: torch.device) -> tuple[float, float]:
    losses = []
    with torch.inference_mode():
        for sequence in tokens:
            out = model(sequence[None].to(device), labels=sequence[None].to(device), use_cache=False)
            losses.append(float(out.loss))
    nll = float(np.mean(losses))
    return (float(math.exp(nll)) if nll < 709 else float("inf")), nll


def capture_layers(model, tokens: torch.Tensor, device: torch.device):
    inputs, outputs, kwargs_by_layer, handles = {}, {}, {}, []
    for index, layer in enumerate(model.model.layers):
        def pre(_m, args, kwargs, i=index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            inputs[i] = hidden.detach()
            kwargs_by_layer[i] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        def post(_m, _args, out, i=index):
            outputs[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(layer.register_forward_hook(post))
    try:
        with torch.inference_mode():
            model(tokens.to(device), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return inputs, outputs, kwargs_by_layer


def functional_statistics(model, modules, tokens, device):
    grad_sum = {name: torch.zeros_like(module.weight, dtype=torch.float32, device="cpu") for name, module in modules.items()}
    grad_sq = {name: torch.zeros_like(value) for name, value in grad_sum.items()}
    for index, sequence in enumerate(tokens):
        model.zero_grad(set_to_none=True)
        sequence = sequence[None].to(device)
        model(sequence, labels=sequence, use_cache=False).loss.backward()
        for name, module in modules.items():
            grad = module.weight.grad.detach().float().cpu()
            grad_sum[name].add_(grad)
            grad_sq[name].addcmul_(grad, grad)
        print(f"functional gradients {index + 1}/{len(tokens)}", flush=True)
    model.zero_grad(set_to_none=True)
    n = float(len(tokens))
    return ({name: value / n for name, value in grad_sum.items()},
            {name: value / n for name, value in grad_sq.items()})


def build_score_vectors(names, originals, activation_rms2, mean_grad, mean_grad2):
    offsets, cursor = {}, 0
    for name in names:
        count = originals[name].numel()
        offsets[name] = (cursor, cursor + count)
        cursor += count
    vectors = {}
    vectors["magnitude"] = torch.cat([originals[name].float().abs().flatten() for name in names])
    vectors["activation"] = torch.cat([
        (originals[name].float().abs() * activation_rms2[name].sqrt()[None]).flatten() for name in names
    ])
    vectors["functional_gradient"] = torch.cat([
        (originals[name].float() * mean_grad[name]).abs().flatten() for name in names
    ])
    vectors["functional_fisher"] = torch.cat([
        (originals[name].float().square() * mean_grad2[name]).flatten() for name in names
    ])
    return offsets, vectors


def thresholds_for_scores(scores, names, offsets, percentages, scope):
    thresholds = {}
    if scope == "global":
        max_n = max(1, round(scores.numel() * max(percentages) / 100))
        values = torch.topk(scores, max_n, largest=True, sorted=True).values
        for pct in percentages:
            n = round(scores.numel() * pct / 100)
            thresholds[pct] = float("inf") if n == 0 else float(values[n - 1])
        return thresholds
    layers = sorted({layer_of(name) for name in names})
    for layer in layers:
        pieces = [scores[offsets[name][0]:offsets[name][1]] for name in names if layer_of(name) == layer]
        layer_scores = torch.cat(pieces)
        max_n = max(1, round(layer_scores.numel() * max(percentages) / 100))
        values = torch.topk(layer_scores, max_n, largest=True, sorted=True).values
        for pct in percentages:
            n = round(layer_scores.numel() * pct / 100)
            thresholds[(layer, pct)] = float("inf") if n == 0 else float(values[n - 1])
    return thresholds


def masks_from_thresholds(scores, names, offsets, thresholds, pct, scope):
    result = {}
    for name in names:
        start, end = offsets[name]
        threshold = thresholds[pct] if scope == "global" else thresholds[(layer_of(name), pct)]
        result[name] = scores[start:end] >= threshold
    return result


def apply_oracle(modules, originals, base, masks, device):
    protected = 0
    sse = denom = 0.0
    layer_counts, projection_counts = defaultdict(int), defaultdict(int)
    with torch.no_grad():
        for name, module in modules.items():
            original = originals[name]
            quantized = base[name]
            mask = masks[name]
            merged = quantized.clone()
            merged.view(-1)[mask] = original.view(-1)[mask]
            module.weight.copy_(merged.to(device, module.weight.dtype))
            protected += int(mask.sum())
            layer_counts[layer_of(name)] += int(mask.sum())
            projection_counts[projection_of(name)] += int(mask.sum())
            error = (original.float() - merged.float()).square()
            sse += float(error.sum())
            denom += float(original.float().square().sum())
    return protected, sse / max(denom, 1e-12), layer_counts, projection_counts


def hidden_diagnostics(model, diag_tokens, device, teacher_outputs):
    inputs, outputs, kwargs = capture_layers(model, diag_tokens, device)
    rows, previous = [], 0.0
    for layer in sorted(outputs):
        target, value = teacher_outputs[layer].float(), outputs[layer].float()
        nmse = float((target - value).square().sum() / target.square().sum().clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(target.flatten(), value.flatten(), dim=0))
        rows.append({"layer": layer, "hidden_nmse": nmse, "hidden_cosine": cosine,
                     "drift_jump": nmse - previous})
        previous = nmse
    cliff = max((row for row in rows if row["layer"] > 0), key=lambda row: row["drift_jump"])
    return rows, inputs, kwargs, cliff


@torch.inference_mode()
def projection_rescue(model, modules, originals, layer, hidden, kwargs, teacher_output, device):
    candidates = []
    for name, module in modules.items():
        if layer_of(name) != layer:
            continue
        saved = module.weight.detach().clone()
        module.weight.copy_(originals[name].to(device, module.weight.dtype))
        out = model.model.layers[layer](hidden_states=hidden, **kwargs)
        out = out[0] if isinstance(out, (tuple, list)) else out
        target = teacher_output.float(); value = out.float()
        nmse = float((target - value).square().sum() / target.square().sum().clamp_min(1e-12))
        module.weight.copy_(saved)
        candidates.append((nmse, projection_of(name)))
    return min(candidates)[1], min(candidates)[0]


@torch.inference_mode()
def logit_kl(model, tokens, teacher_logits, device):
    logits = model(tokens.to(device), use_cache=False).logits.float()
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    return float(torch.nn.functional.kl_div(torch.log_softmax(logits, dim=-1), teacher_logp.exp(), reduction="batchmean"))


def write_outputs(out, aggregate, drift, distribution, overlap):
    pd.DataFrame(aggregate).to_csv(out / "aggregate.csv", index=False)
    pd.DataFrame(drift).to_csv(out / "per_layer_drift.csv", index=False)
    pd.DataFrame(distribution).to_csv(out / "importance_distribution.csv", index=False)
    pd.DataFrame(overlap).to_csv(out / "overlap_and_unions.csv", index=False)


def make_plots(out, aggregate, drift, distribution):
    plot_dir = out / "plots"; plot_dir.mkdir(exist_ok=True)
    frame = pd.DataFrame(aggregate)
    plt.figure(figsize=(9, 6))
    for method in PRIMARY:
        for scope, style in (("global", "-"), ("layer_balanced", "--")):
            f = frame[(frame.method == method) & (frame.scope == scope)].sort_values("protected_pct")
            plt.plot(f.protected_pct, f.perplexity, style, marker="o", label=f"{method} / {scope}")
    plt.yscale("log"); plt.xlabel("weights preserved exactly (%)"); plt.ylabel("perplexity")
    plt.axhline(100, color="grey", lw=.8); plt.axhline(50, color="grey", lw=.8); plt.axhline(30, color="grey", lw=.8)
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(plot_dir / "oracle_recovery_curves.png", dpi=180); plt.close()
    best = frame[(frame.method != "bf16")].sort_values("perplexity").iloc[0]
    d = pd.DataFrame(drift); d = d[d.configuration == best.configuration]
    plt.figure(figsize=(8, 5)); plt.plot(d.layer, d.hidden_nmse, marker="o")
    plt.yscale("log"); plt.xlabel("transformer layer"); plt.ylabel("hidden-state NMSE")
    plt.tight_layout(); plt.savefig(plot_dir / "best_hidden_state_drift.png", dpi=180); plt.close()
    dist = pd.DataFrame(distribution)
    top = dist[(dist.configuration == best.configuration) & (dist.axis == "projection")].sort_values("enrichment", ascending=False)
    plt.figure(figsize=(8, 5)); plt.bar(top.group, top.enrichment)
    plt.axhline(1.0, color="grey", lw=.8); plt.ylabel("selection enrichment vs projection size")
    plt.xticks(rotation=30); plt.tight_layout()
    plt.savefig(plot_dir / "best_projection_distribution.png", dpi=180); plt.close()


def build_report(out, aggregate, overlap, bf_ppl):
    frame = pd.DataFrame(aggregate)
    oracle = frame[frame.method != "bf16"].sort_values("perplexity")
    best = oracle.iloc[0]
    primary = frame[frame.method.isin(PRIMARY)]
    method_best = primary.groupby("method").perplexity.min().sort_values()
    strongest = method_best.index[0]
    def threshold_answer(limit):
        eligible = primary[primary.perplexity < limit].sort_values(["protected_pct", "perplexity"])
        return "not reached at <=10%" if eligible.empty else f"{eligible.iloc[0].protected_pct:.4g}% ({eligible.iloc[0].configuration})"
    near = primary[primary.perplexity <= 1.10 * bf_ppl].sort_values(["protected_pct", "perplexity"])
    q5 = "not reached at <=10%" if near.empty else f"{near.iloc[0].protected_pct:.4g}% ({near.iloc[0].configuration})"
    one = primary[np.isclose(primary.protected_pct, 1.0)]
    layer_conc = pd.DataFrame(pd.read_csv(out / "importance_distribution.csv"))
    best_layers = layer_conc[(layer_conc.configuration == best.configuration) & (layer_conc.axis == "layer")].sort_values("fraction_of_selected", ascending=False).head(3)
    best_proj = layer_conc[(layer_conc.configuration == best.configuration) & (layer_conc.axis == "projection")].sort_values("enrichment", ascending=False).head(3)
    overlaps = pd.DataFrame(overlap); jac = overlaps[overlaps.kind == "jaccard"]
    hybrids = frame[frame.method.str.startswith("hybrid")]
    best_single_1 = one.perplexity.min(); best_hybrid = hybrids.perplexity.min() if len(hybrids) else float("inf")
    zero = primary[np.isclose(primary.protected_pct, 0)].iloc[0]
    text = f"""# Experiment 6 Part 1: Importance Oracle

All 24 transformer layers and all Q/K/V/O/gate/up/down weights use the same fixed Hadamard + 1-bit destructive baseline. Selected oracle weights alone are restored to their original BF16 values. Protected BF16 values are diagnostic and are not proposed as an encoding.

## Results

{frame.sort_values(['method','scope','protected_pct']).to_markdown(index=False, floatfmt='.6g')}

## Required answers

**Q1. Which importance method is strongest?** {strongest}. Its best tested PPL is {method_best.iloc[0]:.4g}.

**Q2. Minimum percentage for PPL <100:** {threshold_answer(100)}.

**Q3. Minimum percentage for PPL <50:** {threshold_answer(50)}.

**Q4. Minimum percentage for PPL <30:** {threshold_answer(30)}.

**Q5. Minimum percentage for <=10% BF16 degradation (PPL <= {1.10*bf_ppl:.4g}):** {q5}.

**Q6. Layer concentration:** For the best configuration, the three largest shares are {', '.join(f'layer {int(r.group)} ({r.fraction_of_selected:.2%})' for r in best_layers.itertuples())}.

**Q7. Projection concentration:** Relative to each projection family's size, the strongest enrichments are {', '.join(f'{r.group} ({r.enrichment:.2f}x; {r.fraction_of_selected:.2%} of selected weights)' for r in best_proj.itertuples())}.

**Q8. Top-1% overlap:** {', '.join(f'{r.selector_a} vs {r.selector_b}: Jaccard {r.value:.4f}' for r in jac.itertuples())}.

**Q9. Do selector unions outperform the best single selector at 1%?** {'YES' if best_hybrid < best_single_1 else 'NO'}. Best hybrid PPL {best_hybrid:.4g}; best single-method 1% PPL {best_single_1:.4g}.

**Q10. Collapse without protection:** the 0%-protected Hadamard model's largest positive inter-layer NMSE jump is at layer {int(zero.collapse_layer)}, attributed by single-projection rescue to {zero.collapse_projection}.

The activation scores `|W|*RMS(X)` and `W^2*E[X^2]` have identical order because the second is the square of the first. Both requested formulas are therefore represented by the same activation ranking. `functional_gradient` and `functional_fisher` are both reported; the three-family comparison uses Fisher as the primary functional selector.
"""
    (out / "REPORT.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args(); seed_everything(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").to(device).eval()
    n_layers = len(model.model.layers)
    modules = target_modules(model, set(range(n_layers))); names = list(modules)
    calibration = get_text_tokens(tokenizer, "train", args.calibration_tokens, args.sequence_length)
    evaluation = get_text_tokens(tokenizer, "test", args.evaluation_tokens, args.sequence_length)
    diag_tokens = calibration[:1]
    originals = {name: module.weight.detach().cpu().clone() for name, module in modules.items()}
    total_weights = sum(weight.numel() for weight in originals.values())
    layer_totals, projection_totals = defaultdict(int), defaultdict(int)
    for name, weight in originals.items():
        layer_totals[layer_of(name)] += weight.numel()
        projection_totals[projection_of(name)] += weight.numel()

    bf_ppl, bf_nll = safe_perplexity(model, evaluation, device)
    teacher_inputs, teacher_outputs, teacher_kwargs = capture_layers(model, diag_tokens, device)
    with torch.inference_mode():
        kl_tokens = diag_tokens[:, :128]
        teacher_logits = model(kl_tokens.to(device), use_cache=False).logits.detach()
    activation_rms2 = activation_importance(model, modules, calibration, device)
    mean_grad, mean_grad2 = functional_statistics(model, modules, calibration, device)
    offsets, score_vectors = build_score_vectors(names, originals, activation_rms2, mean_grad, mean_grad2)
    del mean_grad, mean_grad2, activation_rms2; model.zero_grad(set_to_none=True); torch.cuda.empty_cache(); gc.collect()

    base = {}
    for index, name in enumerate(names, 1):
        print(f"[{index}/{len(names)}] Hadamard base {name}", flush=True)
        base[name] = quantize_hadamard(originals[name].to(device), args.block_size)

    aggregate = [{"configuration": "bf16", "method": "bf16", "scope": "none", "requested_pct": 100.0,
                  "protected_pct": 100.0, "protected_weights": total_weights, "perplexity": bf_ppl,
                  "mean_nll": bf_nll, "ppl_degradation_pct": 0.0, "weight_nmse": 0.0,
                  "final_hidden_nmse": 0.0, "final_hidden_cosine": 1.0, "logit_kl": 0.0,
                  "collapse_layer": np.nan, "collapse_projection": "none", "runtime_seconds": 0.0,
                  "recovery_per_protected_pct": np.nan}]
    drift_rows, distribution_rows, overlap_rows = [], [], []

    def evaluate_config(configuration, method, scope, requested_pct, masks):
        start = time.time()
        protected, weight_nmse, layer_counts, projection_counts = apply_oracle(modules, originals, base, masks, device)
        ppl, nll = safe_perplexity(model, evaluation, device)
        drift, q_inputs, q_kwargs, cliff = hidden_diagnostics(model, diag_tokens, device, teacher_outputs)
        projection, rescued_nmse = projection_rescue(model, modules, originals, int(cliff["layer"]),
                                                     q_inputs[int(cliff["layer"])], q_kwargs[int(cliff["layer"])],
                                                     teacher_outputs[int(cliff["layer"])], device)
        kl = logit_kl(model, kl_tokens, teacher_logits, device)
        actual_pct = 100.0 * protected / total_weights
        bulk_ppl = next((item["perplexity"] for item in aggregate
                         if item["method"] != "bf16" and item["protected_pct"] == 0), np.nan)
        row = {"configuration": configuration, "method": method, "scope": scope,
               "requested_pct": requested_pct, "protected_pct": actual_pct,
               "protected_weights": protected, "perplexity": ppl, "mean_nll": nll,
               "ppl_degradation_pct": 100.0 * (ppl / bf_ppl - 1), "weight_nmse": weight_nmse,
               "final_hidden_nmse": drift[-1]["hidden_nmse"], "final_hidden_cosine": drift[-1]["hidden_cosine"],
               "logit_kl": kl, "collapse_layer": int(cliff["layer"]), "collapse_projection": projection,
               "collapse_rescued_nmse": rescued_nmse, "runtime_seconds": time.time() - start,
               "recovery_per_protected_pct": ((bulk_ppl - ppl) / actual_pct
                                                if actual_pct > 0 and np.isfinite(bulk_ppl) else np.nan)}
        aggregate.append(row)
        for item in drift:
            drift_rows.append({"configuration": configuration, **item})
        for axis, counts, totals in (("layer", layer_counts, layer_totals),
                                     ("projection", projection_counts, projection_totals)):
            for group, count in counts.items():
                baseline_fraction = totals[group] / total_weights
                distribution_rows.append({"configuration": configuration, "method": method, "scope": scope,
                                          "requested_pct": requested_pct, "axis": axis, "group": group,
                                          "protected_weights": count,
                                          "fraction_of_selected": count / max(protected, 1),
                                          "group_weights": totals[group],
                                          "selection_rate": count / totals[group],
                                          "enrichment": ((count / max(protected, 1)) / baseline_fraction
                                                         if protected else 0.0)})
        write_outputs(args.output, aggregate, drift_rows, distribution_rows, overlap_rows)
        print(f"RESULT {configuration}: protected={actual_pct:.6g}% PPL={ppl:.6g} hidden_NMSE={drift[-1]['hidden_nmse']:.6g}", flush=True)
        return row

    selector_thresholds = {}
    for method, scores in score_vectors.items():
        for scope in ("global", "layer_balanced"):
            print(f"ranking {method} / {scope}", flush=True)
            thresholds = thresholds_for_scores(scores, names, offsets, RANK_PCTS, scope)
            selector_thresholds[(method, scope)] = thresholds
            for pct in PCTS:
                masks = masks_from_thresholds(scores, names, offsets, thresholds, pct, scope)
                evaluate_config(f"{method}_{scope}_p{pct:g}", method, scope, pct, masks)

    one_pct_masks = {}
    for method in PRIMARY:
        scores = score_vectors[method]
        one_pct_masks[method] = masks_from_thresholds(scores, names, offsets,
                                                       selector_thresholds[(method, "global")], 1.0, "global")
    for a_index, a in enumerate(PRIMARY):
        for b in PRIMARY[a_index + 1:]:
            inter = union = 0
            for name in names:
                inter += int((one_pct_masks[a][name] & one_pct_masks[b][name]).sum())
                union += int((one_pct_masks[a][name] | one_pct_masks[b][name]).sum())
            overlap_rows.append({"kind": "jaccard", "selector_a": a, "selector_b": b,
                                 "value": inter / max(union, 1), "intersection": inter, "union": union})
    all_inter = all_union = 0
    for name in names:
        all_inter += int((one_pct_masks[PRIMARY[0]][name] & one_pct_masks[PRIMARY[1]][name] & one_pct_masks[PRIMARY[2]][name]).sum())
        all_union += int((one_pct_masks[PRIMARY[0]][name] | one_pct_masks[PRIMARY[1]][name] | one_pct_masks[PRIMARY[2]][name]).sum())
    overlap_rows.append({"kind": "jaccard_all_three", "selector_a": "all_three", "selector_b": "all_three",
                         "value": all_inter / max(all_union, 1), "intersection": all_inter, "union": all_union})

    hybrid_specs = (("hybrid_mag_func", (("magnitude", .5), ("functional_fisher", .5))),
                    ("hybrid_all_three", (("magnitude", .33), ("activation", .33), ("functional_fisher", .33))))
    for hybrid, parts in hybrid_specs:
        masks = {name: torch.zeros(originals[name].numel(), dtype=torch.bool) for name in names}
        for method, pct in parts:
            part = masks_from_thresholds(score_vectors[method], names, offsets,
                                         selector_thresholds[(method, "global")], pct, "global")
            for name in names:
                masks[name] |= part[name]
        row = evaluate_config(hybrid, hybrid, "global_union", sum(p[1] for p in parts), masks)
        overlap_rows.append({"kind": "union_result", "selector_a": hybrid, "selector_b": "",
                             "value": row["perplexity"], "intersection": np.nan,
                             "union": row["protected_weights"]})

    write_outputs(args.output, aggregate, drift_rows, distribution_rows, overlap_rows)
    make_plots(args.output, aggregate, drift_rows, distribution_rows)
    build_report(args.output, aggregate, overlap_rows, bf_ppl)
    dump_json(args.output / "environment.json", {"args": vars(args), "model_layers": n_layers,
              "target_matrices": len(modules), "target_weights": total_weights,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "platform": platform.platform(), "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with torch.no_grad():
        for name, module in modules.items():
            module.weight.copy_(originals[name].to(device, module.weight.dtype))
    print((args.output / "REPORT.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
