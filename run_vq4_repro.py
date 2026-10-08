"""Reproducible 4x4 Fisher-weighted VQ baseline for Qwen2.5-0.5B.

The script deliberately gates later refinement on baseline quality.  It builds
three deterministic K=65536 codebooks, selects a seed only by held-out layer-4
hidden-state NMSE, evaluates prefix PPL once for the winner, and saves all
scientific state needed to reproduce the run.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import struct
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="main")
    p.add_argument("--output", type=Path, default=Path("/workspace/vq4_repro"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    p.add_argument("--k", type=int, default=65536)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--sample-blocks", type=int, default=196608)
    p.add_argument("--lloyd-iters", type=int, default=1)
    p.add_argument("--candidate-k", type=int, default=8)
    p.add_argument("--distance-chunk", type=int, default=768)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--fit-tokens", type=int, default=4096)
    p.add_argument("--heldout-tokens", type=int, default=2048)
    p.add_argument("--eval-tokens", type=int, default=4096)
    p.add_argument("--one-layer-max-degradation", type=float, default=25.0,
                   help="Stop before refinement when winning 1L delta-PPL exceeds this percentage")
    return p.parse_args()


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def sha256_tensor(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def module_map(model):
    out = OrderedDict()
    for li in range(4):
        layer = model.model.layers[li]
        for family in PROJECTIONS:
            parent = layer.self_attn if family in PROJECTIONS[:4] else layer.mlp
            out[f"model.layers.{li}.{'self_attn' if family in PROJECTIONS[:4] else 'mlp'}.{family}"] = getattr(parent, family)
    return out


def blocks4(weight: torch.Tensor) -> torch.Tensor:
    r, c = weight.shape
    assert r % 4 == 0 and c % 4 == 0, (r, c)
    return weight.view(r // 4, 4, c // 4, 4).permute(0, 2, 1, 3).reshape(-1, 16)


def unblock4(blocks: torch.Tensor, shape) -> torch.Tensor:
    r, c = shape
    return blocks.view(r // 4, c // 4, 4, 4).permute(0, 2, 1, 3).reshape(r, c)


def tokenize_fixed(tokenizer, split: str, count: int, offset: int = 0):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(x for x in ds["text"] if x.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    if ids.numel() < offset + count:
        raise RuntimeError(f"{split} has only {ids.numel()} tokens")
    return ids[offset:offset + count].clone()


@torch.inference_mode()
def reference_hidden(model, ids, seq_len, hidden_index=4):
    values = []
    for start in range(0, ids.numel() - seq_len + 1, seq_len):
        x = ids[start:start + seq_len].unsqueeze(0).cuda()
        y = model(x, use_cache=False, output_hidden_states=True).hidden_states[hidden_index]
        values.append(y.float().cpu())
    return torch.cat(values, dim=1)


@torch.inference_mode()
def hidden_metrics(model, ids, target, seq_len, hidden_index=4):
    pred = reference_hidden(model, ids, seq_len, hidden_index)
    d = pred - target
    nmse = float(d.square().sum() / target.square().sum().clamp_min(1e-20))
    cosine = float(torch.nn.functional.cosine_similarity(pred.flatten(), target.flatten(), dim=0))
    return {"nmse": nmse, "cosine": cosine}


@torch.inference_mode()
def perplexity(model, ids, seq_len):
    losses, counts = [], []
    for start in range(0, ids.numel() - seq_len + 1, seq_len):
        x = ids[start:start + seq_len].unsqueeze(0).cuda()
        loss = model(x, labels=x, use_cache=False).loss.float()
        losses.append(float(loss)); counts.append(seq_len - 1)
    nll = float(np.average(losses, weights=counts))
    return {"nll": nll, "ppl": float(math.exp(min(nll, 80.0)))}


@torch.inference_mode()
def collect_fisher(model, modules, ids, seq_len):
    sums = {n: torch.zeros(m.in_features, dtype=torch.float64) for n, m in modules.items()}
    counts = {n: 0 for n in modules}
    hooks = []
    for name, module in modules.items():
        def hook(_m, inp, name=name):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            sums[name].add_(x.square().sum(0).cpu().double())
            counts[name] += x.shape[0]
        hooks.append(module.register_forward_pre_hook(hook))
    try:
        for start in range(0, ids.numel() - seq_len + 1, seq_len):
            model(ids[start:start + seq_len].unsqueeze(0).cuda(), use_cache=False)
    finally:
        for h in hooks: h.remove()
    return {n: (sums[n] / max(counts[n], 1)).float().clamp_min(1e-8) for n in modules}


def sample_training_blocks(originals, total: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    counts = {n: blocks4(w).shape[0] for n, w in originals.items()}
    all_count = sum(counts.values())
    parts = []
    allocated = 0
    names = list(originals)
    for i, name in enumerate(names):
        take = total - allocated if i == len(names) - 1 else max(1, round(total * counts[name] / all_count))
        b = blocks4(originals[name].float())
        idx = torch.randint(b.shape[0], (take,), generator=gen)
        parts.append(b[idx]); allocated += take
    x = torch.cat(parts)[:total]
    order = torch.randperm(x.shape[0], generator=gen)
    return x[order].contiguous()


@torch.inference_mode()
def nearest(x, centers, chunk, topk=1):
    centers = centers.cuda().half()
    c2 = centers.square().sum(1).unsqueeze(0)
    outs = []
    for s in range(0, x.shape[0], chunk):
        q = x[s:s + chunk].cuda().half()
        dist = q.square().sum(1, keepdim=True) + c2 - 2 * (q @ centers.T)
        outs.append(torch.topk(dist, topk, largest=False, sorted=True).indices.cpu())
    return torch.cat(outs)


def train_codebook(sample, k, seed, iters, chunk):
    if sample.shape[0] < k: raise ValueError("sample-blocks must be >= K")
    gen = torch.Generator().manual_seed(seed)
    centers = sample[torch.randperm(sample.shape[0], generator=gen)[:k]].clone()
    for it in range(iters):
        ids = nearest(sample, centers, chunk, 1).flatten()
        sums = torch.zeros(k, 16, dtype=torch.float32)
        counts = torch.zeros(k, dtype=torch.int64)
        sums.index_add_(0, ids, sample.float())
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=torch.int64))
        used = counts > 0
        centers[used] = sums[used] / counts[used, None]
        print(f"  Lloyd {it+1}/{iters}: used={int(used.sum())}/{k}", flush=True)
    # The deployed shared codebook is INT8 with one FP16 scale per centroid.
    scales = (centers.abs().amax(1) / 127).clamp_min(2 ** -24).half()
    q = torch.round(centers / scales.float()[:, None]).clamp(-127, 127).to(torch.int8)
    deq = q.float() * scales.float()[:, None]
    return q, scales, deq


@torch.inference_mode()
def fisher_assign(weight, fisher, centers, candidate_k, chunk):
    x = blocks4(weight.float())
    candidates = nearest(x, centers, chunk, candidate_k)
    chosen_parts = []
    col_blocks = weight.shape[1] // 4
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk].cuda().float()
        cb_ids = candidates[s:s + chunk].cuda().long()
        cb = centers.cuda().float()[cb_ids]
        pos = torch.arange(s, min(s + chunk, x.shape[0])) % col_blocks
        h4 = fisher.view(-1, 4)[pos].cuda().float()
        metric = h4[:, None, :].expand(-1, 4, -1).reshape(-1, 16)
        score = ((xb[:, None, :] - cb).square() * metric[:, None, :]).sum(2)
        local = score.argmin(1)
        chosen_parts.append(cb_ids[torch.arange(cb_ids.shape[0], device="cuda"), local].cpu())
    return torch.cat(chosen_parts).to(torch.int32)


def reconstruct(ids, centers, shape, dtype=torch.bfloat16):
    return unblock4(centers[ids.long()], shape).to(dtype)


def restore(modules, originals, names=None):
    names = list(modules) if names is None else names
    with torch.no_grad():
        for n in names: modules[n].weight.copy_(originals[n].to("cuda", modules[n].weight.dtype))


def apply_quantized(modules, assignments, centers, shapes, max_layer=3):
    with torch.no_grad():
        for name, module in modules.items():
            layer = int(name.split(".")[2])
            if layer <= max_layer:
                module.weight.copy_(reconstruct(assignments[name], centers, shapes[name], module.weight.dtype).cuda())


def payload_accounting(assignments, q, scales, modules, total_weights):
    id_bits = sum(v.numel() * 16 for v in assignments.values())
    codebook_bits = q.numel() * 8
    scale_bits = scales.numel() * 16
    # Fixed binary decoder metadata: magic/version/K/block/count (32 B), plus each
    # tensor's UTF-8 name, NUL, two uint32 dimensions and uint64 ID offset.
    metadata_bytes = 32 + sum(len(n.encode()) + 1 + 4 + 4 + 8 for n in modules)
    metadata_bits = metadata_bytes * 8
    total = id_bits + codebook_bits + scale_bits + metadata_bits
    return {"weights": total_weights, "id_bits": id_bits, "base_codebook_bits": codebook_bits,
            "base_codebook_scale_bits": scale_bits, "delta_bits": 0, "residual_bits": 0,
            "metadata_bits": metadata_bits, "total_bits": total, "bpw": total / total_weights}


def save_frozen(out, seed, q, scales, assignments, fisher, originals, tokens, config, accounting):
    out.mkdir(parents=True, exist_ok=True)
    # Scientific checkpoint (not the compact deployment stream): includes Fisher
    # and fixed token IDs. Logical codec bits are reported separately.
    state = {"format": "vq4-fisher-repro-v1", "seed": seed, "codebook_q_int8": q,
             "codebook_scale_fp16": scales, "assignments_int32": assignments,
             "fisher_fp32": fisher, "shapes": {n: list(w.shape) for n, w in originals.items()},
             "tokens": tokens, "config": config, "accounting": accounting}
    torch.save(state, out / "frozen_baseline.pt")
    dump(out / "frozen_baseline_manifest.json", {
        "seed": seed, "config": config, "accounting": accounting,
        "checkpoint_bytes": (out / "frozen_baseline.pt").stat().st_size,
        "codebook_sha256": sha256_tensor(q),
        "token_sha256": {k: sha256_tensor(v) for k, v in tokens.items()},
        "assignment_sha256": {k: sha256_tensor(v) for k, v in assignments.items()},
        "note": "checkpoint_bytes includes scientific Fisher/token sidecars and is not deployment BPW"
    })


def main():
    a = args(); a.output.mkdir(parents=True, exist_ok=True); seed_all(0)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(a.model, revision=a.revision, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision,
        torch_dtype=torch.bfloat16, device_map={"": "cuda"}, low_cpu_mem_usage=True)
    model.eval(); modules = module_map(model)
    originals = {n: m.weight.detach().cpu().bfloat16().clone() for n, m in modules.items()}
    shapes = {n: tuple(w.shape) for n, w in originals.items()}
    total_weights = sum(w.numel() for w in originals.values())
    fit = tokenize_fixed(tokenizer, "train", a.fit_tokens, 0)
    heldout = tokenize_fixed(tokenizer, "train", a.heldout_tokens, a.fit_tokens)
    evaluation = tokenize_fixed(tokenizer, "test", a.eval_tokens, 0)
    tokens = {"fit": fit, "heldout": heldout, "evaluation": evaluation}
    torch.save(tokens, a.output / "fixed_token_ids.pt")
    print("Capturing BF16 references and Fisher statistics", flush=True)
    teacher_h4 = reference_hidden(model, heldout, a.sequence_length, 4)
    bf16_eval = perplexity(model, evaluation, a.sequence_length)
    fisher = collect_fisher(model, modules, fit, a.sequence_length)
    torch.save(fisher, a.output / "fisher_fp32.pt")
    config = {"model": a.model, "requested_revision": a.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None), "block": a.block, "k": a.k,
        "seeds": a.seeds, "sample_blocks": a.sample_blocks, "lloyd_iters": a.lloyd_iters,
        "candidate_k": a.candidate_k, "sequence_length": a.sequence_length,
        "fit_tokens": a.fit_tokens, "heldout_tokens": a.heldout_tokens,
        "eval_tokens": a.eval_tokens, "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "gpu": torch.cuda.get_device_name(), "platform": platform.platform()}
    dump(a.output / "environment.json", config)
    seed_rows, artifacts = [], {}
    for seed in a.seeds:
        print(f"\nSEED {seed}: sample/codebook", flush=True); seed_all(seed); restore(modules, originals)
        sample = sample_training_blocks(originals, a.sample_blocks, seed)
        q, scales, centers = train_codebook(sample, a.k, seed, a.lloyd_iters, a.distance_chunk)
        del sample; gc.collect(); torch.cuda.empty_cache()
        assignments = {}
        for i, (name, weight) in enumerate(originals.items(), 1):
            print(f"  assign {i:02d}/{len(originals)} {name}", flush=True)
            assignments[name] = fisher_assign(weight, fisher[name], centers, a.candidate_k, a.distance_chunk)
        apply_quantized(modules, assignments, centers, shapes, 3)
        metric = hidden_metrics(model, heldout, teacher_h4, a.sequence_length, 4)
        accounting = payload_accounting(assignments, q, scales, modules, total_weights)
        row = {"seed": seed, **metric, "bpw": accounting["bpw"]}
        seed_rows.append(row); artifacts[seed] = (q, scales, assignments, accounting)
        print("  heldout", row, flush=True)
        dump(a.output / "seed_selection.json", seed_rows)
    winner = min(seed_rows, key=lambda x: x["nmse"])["seed"]
    q, scales, assignments, accounting = artifacts[winner]
    centers = q.float() * scales.float()[:, None]
    save_frozen(a.output, winner, q, scales, assignments, fisher, originals, tokens, config, accounting)
    prefix = []
    for n in range(0, 5):
        restore(modules, originals)
        if n: apply_quantized(modules, assignments, centers, shapes, n - 1)
        result = bf16_eval if n == 0 else perplexity(model, evaluation, a.sequence_length)
        delta = 0.0 if n == 0 else 100 * (result["ppl"] / bf16_eval["ppl"] - 1)
        prefix.append({"layers_quantized": n, **result, "delta_ppl_pct": delta})
        print("PREFIX", prefix[-1], flush=True)
    gate_pass = prefix[1]["delta_ppl_pct"] <= a.one_layer_max_degradation
    report = {"winner_seed": winner, "selection_metric": "heldout_h4_nmse", "seed_selection": seed_rows,
              "bf16": bf16_eval, "prefix_results": prefix, "accounting": accounting,
              "baseline_gate_pass": gate_pass, "refinement_allowed": gate_pass,
              "elapsed_seconds": time.time() - started}
    dump(a.output / "baseline_results.json", report)
    print("BASELINE_GATE", "PASS" if gate_pass else "FAIL", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
