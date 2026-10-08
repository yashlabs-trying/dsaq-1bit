"""Diagnose a failed frozen VQ4 baseline without changing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from run_vq4_repro import module_map, perplexity, reconstruct, restore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("/workspace/vq4_repro/frozen_baseline.pt"))
    p.add_argument("--output", type=Path, default=Path("/workspace/vq4_repro/diagnostics.json"))
    a = p.parse_args()
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["requested_revision"],
        torch_dtype=torch.bfloat16, device_map={"": "cuda"}, low_cpu_mem_usage=True).eval()
    modules = module_map(model)
    originals = {n: m.weight.detach().cpu().bfloat16().clone() for n, m in modules.items()}
    centers = state["codebook_q_int8"].float() * state["codebook_scale_fp16"].float()[:, None]
    ids = state["assignments_int32"]
    eval_ids = state["tokens"]["evaluation"]
    seq = cfg["sequence_length"]
    rows = []
    recon = {}
    for name, weight in originals.items():
        q = reconstruct(ids[name], centers, weight.shape, torch.float32)
        recon[name] = q
        e = q - weight.float()
        f = state["fisher_fp32"][name]
        rows.append({"name": name, "layer": int(name.split(".")[2]), "family": name.rsplit(".", 1)[-1],
            "relative_mse": float(e.square().sum() / weight.float().square().sum().clamp_min(1e-20)),
            "fisher_relative_mse": float((e.square() * f[None, :]).sum() /
                (weight.float().square() * f[None, :]).sum().clamp_min(1e-20)),
            "max_abs_error": float(e.abs().max())})
    # Only layer zero is quantized. Restore each projection family temporarily.
    layer0 = [n for n in modules if ".layers.0." in n]
    restore(modules, originals)
    with torch.no_grad():
        for n in layer0: modules[n].weight.copy_(recon[n].to("cuda", modules[n].weight.dtype))
    all_quant = perplexity(model, eval_ids, seq)
    family = []
    for name in layer0:
        with torch.no_grad(): modules[name].weight.copy_(originals[name].to("cuda", modules[name].weight.dtype))
        metric = perplexity(model, eval_ids, seq)
        family.append({"restored": name.rsplit(".", 1)[-1], **metric,
                       "ppl_reduction": all_quant["ppl"] - metric["ppl"]})
        with torch.no_grad(): modules[name].weight.copy_(recon[name].to("cuda", modules[name].weight.dtype))
    result = {"all_layer0_quantized": all_quant, "matrix_error": rows,
              "layer0_family_restore": family,
              "worst_by_fisher_error": sorted(rows, key=lambda x: x["fisher_relative_mse"], reverse=True)[:10]}
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
