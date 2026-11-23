"""E19-R value-only functional-QAT sanity run.

This deliberately preserves the recovered E19 initialization and evaluator, then
changes only the broken optimization path.  It is an algorithm-recovery run at
the recovered codec's real ~3.39 BPW, not a 1.20-BPW result.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

import e19_repaired as legacy
from gptq import decode as gptq_decode
from gptq import gptq_quantize


MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 42
WINDOW = 128
TRAIN_TOKENS = int(os.environ.get("E19R_TRAIN_TOKENS", "4096"))
VALID_TOKENS = int(os.environ.get("E19R_VALID_TOKENS", "4096"))
TEST_TOKENS = int(os.environ.get("E19R_TEST_TOKENS", "4096"))
VALID_CHECK_TOKENS = int(os.environ.get("E19R_VALID_CHECK_TOKENS", "1024"))
STEPS = int(os.environ.get("E19R_STEPS", "25"))
CHECK_EVERY = int(os.environ.get("E19R_CHECK_EVERY", "5"))
LR = float(os.environ.get("E19R_LR", "1e-4"))
KL_WEIGHT = float(os.environ.get("E19R_KL_WEIGHT", "0.20"))
TRUST_WEIGHT = float(os.environ.get("E19R_TRUST_WEIGHT", "1e-4"))
TOPK = int(os.environ.get("E19R_TOPK", "64"))
HESSIAN_WINDOWS = int(os.environ.get("E19R_HESSIAN_WINDOWS", "32"))
BASE_BITS = int(os.environ.get("E19R_BASE_BITS", "4"))
if BASE_BITS not in (1, 2, 3, 4):
    raise ValueError(f"E19R_BASE_BITS must be 1, 2, 3, or 4; got {BASE_BITS}")
OUT = Path(os.environ.get(
    "E19R_OUT", f"/workspace/e19r/results/E19R_{BASE_BITS}bit_value_sanity"
))


def tokenize(tokenizer, text: str, limit: int, device: torch.device) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0, :limit]
    if ids.numel() < limit:
        raise RuntimeError(f"requested {limit} tokens but split only supplied {ids.numel()}")
    return ids.to(device)


def split_windows(ids: torch.Tensor):
    return [w for w in ids.split(WINDOW) if w.numel() >= 2]


@torch.inference_mode()
def evaluate(model, ids: torch.Tensor) -> dict:
    weighted = 0.0
    count = 0
    for window in split_windows(ids):
        loss = model(window[None], labels=window[None], use_cache=False).loss
        n = window.numel() - 1
        weighted += float(loss) * n
        count += n
    nll = weighted / count
    return {"nll": nll, "ppl": math.exp(nll), "tokens": count}


@torch.inference_mode()
def teacher_cache(model, ids: torch.Tensor) -> list[dict]:
    cache = []
    for window in split_windows(ids):
        logits = model(window[None], use_cache=False).logits[:, :-1].float()
        values, indices = logits.topk(TOPK, dim=-1)
        cache.append({
            "tokens": window.detach().cpu(),
            "top_ids": indices.squeeze(0).cpu(),
            "top_logp": values.squeeze(0).log_softmax(-1).cpu(),
        })
    return cache


def stored_fp16_ste(x: torch.Tensor) -> torch.Tensor:
    stored = x.to(torch.float16).to(torch.float32)
    return x + (stored - x).detach()


def decoded_weight(codec: legacy.E19Codec, shape, ste: bool) -> torch.Tensor:
    c1 = stored_fp16_ste(codec.C1) if ste else codec.C1.to(torch.float16).float()
    c2 = stored_fp16_ste(codec.C2) if ste else codec.C2.to(torch.float16).float()
    residual = (c1[codec.idx1.long()] + c2[codec.idx2.long()]).reshape(-1)
    residual = residual[: math.prod(shape)].reshape(shape)
    base = codec._base_rec.reshape(shape).float()
    return (base + residual).to(torch.bfloat16)


def mlp_overrides(codecs: dict, modules: dict, ste: bool) -> dict:
    return {
        f"{name}.weight": decoded_weight(codec, modules[name].weight.shape, ste)
        for name, codec in codecs.items()
        if any(p in name for p in ("gate_proj", "up_proj", "down_proj"))
    }


def functional_objective(model, overrides: dict, item: dict) -> tuple[torch.Tensor, dict]:
    window = item["tokens"].to(model.device_holder)
    logits = functional_call(
        model, overrides, (window[None],), {"use_cache": False}, strict=False
    ).logits[:, :-1].float()
    labels = window[None, 1:]
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

    top_ids = item["top_ids"].to(logits.device)
    teacher_logp = item["top_logp"].to(logits.device)
    student_top_logp = logits.squeeze(0).gather(-1, top_ids).log_softmax(-1)
    teacher_p = teacher_logp.exp()
    kl = (teacher_p * (teacher_logp - student_top_logp)).sum(-1).mean()
    return ce + KL_WEIGHT * kl, {"ce": float(ce.detach()), "kl": float(kl.detach())}


@torch.inference_mode()
def evaluate_overrides(model, codecs: dict, modules: dict, ids: torch.Tensor) -> dict:
    overrides = mlp_overrides(codecs, modules, ste=False)
    weighted = 0.0
    count = 0
    for window in split_windows(ids):
        logits = functional_call(
            model, overrides, (window[None],), {"use_cache": False}, strict=False
        ).logits
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            window[None, 1:].reshape(-1),
        )
        n = window.numel() - 1
        weighted += float(loss) * n
        count += n
    nll = weighted / count
    return {"nll": nll, "ppl": math.exp(nll), "tokens": count}


def center_snapshot(codecs: dict) -> dict:
    return {
        name: (codec.C1.detach().cpu().to(torch.float16).clone(),
               codec.C2.detach().cpu().to(torch.float16).clone())
        for name, codec in codecs.items()
        if any(p in name for p in ("gate_proj", "up_proj", "down_proj"))
    }


def restore_centers(codecs: dict, snapshot: dict, device: torch.device, trainable: bool):
    for name, (c1, c2) in snapshot.items():
        codecs[name].C1 = c1.to(device=device, dtype=torch.float32).detach().requires_grad_(trainable)
        codecs[name].C2 = c2.to(device=device, dtype=torch.float32).detach().requires_grad_(trainable)


def save_codec(path: Path, codecs: dict, report_meta: dict):
    state = {
        "format": "E19R_research_checkpoint_not_bitpacked",
        "metadata": report_meta,
        "codecs": {
            name: {
                "C1": codec.C1.detach().cpu().to(torch.float16),
                "C2": codec.C2.detach().cpu().to(torch.float16),
                "idx1": codec.idx1.detach().cpu().to(torch.uint8),
                "idx2": codec.idx2.detach().cpu().to(torch.uint8),
                "base_codes": codec.base_codes.detach().cpu(),
                "base_scale": codec.base_scale.detach().cpu().to(torch.float16),
                "base_zero": codec.base_zero.detach().cpu().to(torch.float16),
                "base_bits": codec.base_bits,
                "shape": tuple(codec._base_rec.shape),
            }
            for name, codec in codecs.items()
        },
    }
    torch.save(state, path)
    return hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_size


def main():
    started = time.time()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)

    print(json.dumps({"event": "start", "gpu": torch.cuda.get_device_name(0),
                      "steps": STEPS, "train_tokens": TRAIN_TOKENS,
                      "base_bits": BASE_BITS}), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    model.device_holder = device

    train_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    valid_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    train_text = "\n\n".join(x for x in train_ds["text"] if x.strip())
    valid_text = "\n\n".join(x for x in valid_ds["text"] if x.strip())
    test_text = "\n\n".join(x for x in test_ds["text"] if x.strip())
    train_ids = tokenize(tokenizer, train_text, TRAIN_TOKENS, device)
    valid_ids = tokenize(tokenizer, valid_text, VALID_TOKENS, device)
    test_ids = tokenize(tokenizer, test_text, TEST_TOKENS, device)

    bf16_valid = evaluate(model, valid_ids)
    bf16_test = evaluate(model, test_ids)
    print(json.dumps({"event": "bf16", "valid": bf16_valid, "test": bf16_test}), flush=True)
    cache = teacher_cache(model, train_ids)

    linear = [(n, m) for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n != "lm_head"]
    modules = dict(linear)
    print(json.dumps({"event": "hessian_start", "linear_layers": len(linear)}), flush=True)
    H = legacy.collect_H(model, linear, tokenizer, train_text,
                         nsamples=min(HESSIAN_WINDOWS,
                                      TRAIN_TOKENS // legacy.SEQLEN))

    originals = {name: module.weight.detach().float().clone() for name, module in linear}
    codecs = {}
    for idx, (name, module) in enumerate(linear):
        w = originals[name]
        _, codes, scale, zero = gptq_quantize(
            w, H[name], BASE_BITS, groupwise=True, groupsize=64, sym=False
        )
        # The stored FP16 metadata is the codec.  Build both the residual target
        # and the cached base from that exact representation; retaining the
        # transient FP32 GPTQ reconstruction here caused E19-R's 0.042-PPL
        # in-process/fresh-process discrepancy.
        stored_scale = scale.to(torch.float16)
        stored_zero = zero.to(torch.float16)
        stored_base = gptq_decode(codes, stored_scale, stored_zero).float()
        residual = w.reshape(-1) - stored_base.reshape(-1)
        C1, C2, idx1, idx2, _ = legacy.init_factorized_residual(
            residual, 64, 48, 16, SEED
        )
        codec = legacy.E19Codec(
            codes, stored_scale, stored_zero, BASE_BITS, C1, C2, idx1, idx2
        )
        # Cache the exact decoder output.  Do not round this reconstruction a
        # second time: scale/zero are already stored at FP16 precision.
        codec._base_rec = stored_base.reshape(-1)
        codecs[name] = codec
        module.weight.data.copy_(stored_base.to(module.weight.dtype))
        if (idx + 1) % 20 == 0:
            print(json.dumps({"event": "build_progress", "done": idx + 1,
                              "total": len(linear)}), flush=True)

    del H, originals
    torch.cuda.empty_cache()
    stored_base_max_abs_diff = max(
        float((codec._base_rec.reshape(codec.base_codes.shape).float() -
               gptq_decode(codec.base_codes, codec.base_scale, codec.base_zero).float())
              .abs().max())
        for codec in codecs.values()
    )
    if stored_base_max_abs_diff != 0.0:
        raise RuntimeError(
            f"cached base is not the stored decode: max diff={stored_base_max_abs_diff}"
        )
    print(json.dumps({"event": "stored_decode_preflight",
                      "max_abs_difference": stored_base_max_abs_diff}), flush=True)
    base_test = evaluate(model, test_ids)

    for name, codec in codecs.items():
        modules[name].weight.data.copy_(decoded_weight(codec, modules[name].weight.shape, False))
    init_valid = evaluate(model, valid_ids)
    init_test = evaluate(model, test_ids)
    print(json.dumps({"event": "initial", "base_test": base_test,
                      "valid": init_valid, "test": init_test}), flush=True)

    theoretical_bytes = sum(codec.payload_bytes(sum(p.numel() for p in model.parameters()))[0]
                            for codec in codecs.values())
    total_params = sum(p.numel() for p in model.parameters())
    effective_bpw = 8 * theoretical_bytes / total_params
    init_meta = {"effective_bpw": effective_bpw, "test": init_test,
                 "valid": init_valid, "seed": SEED}
    init_sha, init_file_bytes = save_codec(OUT / "e19r_init.pt", codecs, init_meta)

    for p in model.parameters():
        p.requires_grad_(False)
    mlp_names = [n for n in codecs if any(x in n for x in ("gate_proj", "up_proj", "down_proj"))]
    for name in mlp_names:
        codecs[name].C1 = codecs[name].C1.float().detach().requires_grad_(True)
        codecs[name].C2 = codecs[name].C2.float().detach().requires_grad_(True)
    params = [x for n in mlp_names for x in (codecs[n].C1, codecs[n].C2)]
    initial_centers = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

    overrides0 = mlp_overrides(codecs, modules, ste=False)
    first = cache[0]
    with torch.no_grad():
        regular = evaluate(model, first["tokens"].to(device))["nll"]
        functional = evaluate_overrides(model, codecs, modules, first["tokens"].to(device))["nll"]
    if abs(regular - functional) > 2e-3:
        raise RuntimeError(f"no-op functional mismatch: regular={regular}, functional={functional}")
    print(json.dumps({"event": "preflight", "regular_ce": regular,
                      "functional_ce": functional, "difference": abs(regular-functional),
                      "model_trainable_params": sum(p.requires_grad for p in model.parameters()),
                      "center_tensors": len(params)}), flush=True)

    baseline_valid_check = evaluate_overrides(
        model, codecs, modules, valid_ids[:VALID_CHECK_TOKENS]
    )
    best_snapshot = center_snapshot(codecs)
    best_valid = baseline_valid_check["nll"]
    history = []
    generator = torch.Generator().manual_seed(SEED + 1)
    order = torch.randperm(len(cache), generator=generator).tolist()

    for step in range(1, STEPS + 1):
        item = cache[order[(step - 1) % len(order)]]
        optimizer.zero_grad(set_to_none=True)
        overrides = mlp_overrides(codecs, modules, ste=True)
        loss, metrics = functional_objective(model, overrides, item)
        trust = torch.stack([(p - p0).square().mean() for p, p0 in zip(params, initial_centers)]).mean()
        total = loss + TRUST_WEIGHT * trust
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        nonzero = sum(int(p.grad is not None and torch.count_nonzero(p.grad).item() > 0) for p in params)
        if nonzero == 0:
            raise RuntimeError("all codebook gradients are zero; functional path is disconnected")
        optimizer.step()
        row = {"step": step, **metrics, "loss": float(total.detach()),
               "trust": float(trust.detach()), "grad_norm": grad_norm,
               "nonzero_grad_tensors": nonzero}

        if step % CHECK_EVERY == 0 or step == STEPS:
            valid_check = evaluate_overrides(
                model, codecs, modules, valid_ids[:VALID_CHECK_TOKENS]
            )
            row["valid_check"] = valid_check
            if valid_check["nll"] < best_valid:
                best_valid = valid_check["nll"]
                best_snapshot = center_snapshot(codecs)
                row["accepted"] = True
            else:
                row["accepted"] = False
        history.append(row)
        print(json.dumps({"event": "step", **row}), flush=True)

    restore_centers(codecs, best_snapshot, device, trainable=False)
    for name, codec in codecs.items():
        modules[name].weight.data.copy_(decoded_weight(codec, modules[name].weight.shape, False))
    final_valid = evaluate(model, valid_ids)
    final_test = evaluate(model, test_ids)
    final_meta = {"effective_bpw": effective_bpw, "valid": final_valid, "test": final_test,
                  "parent_init_test": init_test, "selection": "validation_only"}
    final_sha, final_file_bytes = save_codec(OUT / "e19r_value_best.pt", codecs, final_meta)

    report = {
        "experiment": "E19-R value-only functional-QAT sanity",
        "representation": (
            f"E19-R {BASE_BITS}-bit GPTQ + 2x16 additive residual"
        ),
        "base_bits": BASE_BITS,
        "claimed_target_bpw": 1.20,
        "actual_theoretical_bpw": effective_bpw,
        "bf16_valid": bf16_valid,
        "bf16_test": bf16_test,
        "gptq_base_test": base_test,
        "stored_base_max_abs_difference": stored_base_max_abs_diff,
        "init_valid": init_valid,
        "baseline_valid_check": baseline_valid_check,
        "init_test": init_test,
        "final_valid": final_valid,
        "final_test": final_test,
        "delta_ppl_vs_bf16_percent": 100 * (final_test["ppl"] / bf16_test["ppl"] - 1),
        "history": history,
        "init_checkpoint": {"sha256": init_sha, "file_bytes": init_file_bytes},
        "final_checkpoint": {"sha256": final_sha, "file_bytes": final_file_bytes},
        "runtime_seconds": time.time() - started,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", **report}, default=str), flush=True)


if __name__ == "__main__":
    main()
