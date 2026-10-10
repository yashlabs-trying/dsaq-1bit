"""E19-R functional assignment refresh from a frozen trained checkpoint.

Ranks every C1/C2 replacement with a first-order LM-loss estimate, then exactly
evaluates small coordinated coverage levels on disjoint internal splits.  The
official validation split is used only for macro acceptance and test only after
that decision.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.func import functional_call
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import decode as gptq_decode


MODEL = "Qwen/Qwen2.5-0.5B"
WINDOW = 128
SEED = 42
TRAIN_TOKENS = 131_072
PROPOSAL_WINDOWS = 128
ACCEPT_WINDOWS = 128
SANITY_WINDOWS = 120
CHANGE_COUNTS = (1, 4, 16, 64, 256)
MLP_MARKERS = ("gate_proj", "up_proj", "down_proj")


def windows(ids: torch.Tensor) -> list[torch.Tensor]:
    return [x for x in ids.split(WINDOW) if x.numel() >= 2]


def nll_from_logits(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        tokens[None, 1:].reshape(-1),
    )


@torch.inference_mode()
def evaluate(model, items: list[torch.Tensor]) -> dict:
    total = 0.0
    count = 0
    for item in items:
        item = item.to(model.device_holder)
        loss = model(item[None], use_cache=False).logits
        n = item.numel() - 1
        total += float(nll_from_logits(loss, item)) * n
        count += n
    nll = total / count
    return {"nll": nll, "ppl": math.exp(nll), "tokens": count}


def decode_weight(data: dict, shape, device: torch.device) -> torch.Tensor:
    base = gptq_decode(
        data["base_codes"].to(device),
        data["base_scale"].to(device),
        data["base_zero"].to(device),
    ).reshape(shape).float()
    c1 = data["C1"].to(device).float()
    c2 = data["C2"].to(device).float()
    idx1 = data["idx1"].to(device).long()
    idx2 = data["idx2"].to(device).long()
    residual = (c1[idx1] + c2[idx2]).reshape(-1)[: math.prod(shape)].reshape(shape)
    return (base + residual).to(torch.bfloat16)


@torch.inference_mode()
def install_state(model, state: dict, names: list[str] | None = None) -> None:
    modules = dict(model.named_modules())
    selected = state["codecs"].keys() if names is None else names
    for name in selected:
        module = modules[name]
        module.weight.copy_(decode_weight(
            state["codecs"][name], module.weight.shape, model.device_holder
        ))


def collect_weight_gradients(model, state: dict, items: list[torch.Tensor]):
    modules = dict(model.named_modules())
    names = [n for n in state["codecs"] if any(x in n for x in MLP_MARKERS)]
    overrides = {
        f"{name}.weight": decode_weight(
            state["codecs"][name], modules[name].weight.shape, model.device_holder
        ).detach().requires_grad_(True)
        for name in names
    }
    for k, item in enumerate(items, 1):
        item = item.to(model.device_holder)
        logits = functional_call(
            model, overrides, (item[None],), {"use_cache": False}, strict=False
        ).logits
        nll_from_logits(logits, item).backward()
        if k % 16 == 0:
            print(json.dumps({"event": "gradient_progress", "done": k,
                              "total": len(items)}), flush=True)
    scale = 1.0 / len(items)
    gradients = {
        name: overrides[f"{name}.weight"].grad.detach().float().mul_(scale)
        for name in names
    }
    return names, gradients


@torch.inference_mode()
def rank_all_replacements(state: dict, names: list[str], gradients: dict,
                          device: torch.device) -> dict:
    score_parts = []
    layer_parts = []
    local_parts = []
    kind_parts = []
    id_parts = []
    for layer_id, name in enumerate(names):
        data = state["codecs"][name]
        c1 = data["C1"].to(device).float()
        c2 = data["C2"].to(device).float()
        idx1 = data["idx1"].to(device).long()
        idx2 = data["idx2"].to(device).long()
        package_dim = c1.shape[1]
        grad = gradients.pop(name).reshape(-1)
        padded = idx1.numel() * package_dim
        if grad.numel() < padded:
            grad = F.pad(grad, (0, padded - grad.numel()))
        grad = grad.reshape(-1, package_dim)

        raw1 = grad @ c1.t()
        current1 = raw1.gather(1, idx1[:, None])
        delta1 = raw1 - current1
        delta1.scatter_(1, idx1[:, None], float("inf"))
        best1, cand1 = delta1.min(1)

        raw2 = grad @ c2.t()
        current2 = raw2.gather(1, idx2[:, None])
        delta2 = raw2 - current2
        delta2.scatter_(1, idx2[:, None], float("inf"))
        best2, cand2 = delta2.min(1)

        choose2 = best2 < best1
        best = torch.where(choose2, best2, best1)
        candidate = torch.where(choose2, cand2, cand1)
        count = best.numel()
        score_parts.append(best.cpu())
        layer_parts.append(torch.full((count,), layer_id, dtype=torch.int16))
        local_parts.append(torch.arange(count, dtype=torch.int32))
        kind_parts.append(choose2.cpu().to(torch.uint8))
        id_parts.append(candidate.cpu().to(torch.uint8))
        print(json.dumps({"event": "rank_progress", "done": layer_id + 1,
                          "total": len(names), "name": name,
                          "packages": count}), flush=True)

    scores = torch.cat(score_parts)
    result = {
        "scores": scores,
        "layers": torch.cat(layer_parts),
        "locals": torch.cat(local_parts),
        "kinds": torch.cat(kind_parts),
        "ids": torch.cat(id_parts),
        "negative_candidates": int((scores < 0).sum()),
    }
    return result


def apply_topk(base_state: dict, ranking: dict, names: list[str], k: int) -> dict:
    state = copy.deepcopy(base_state)
    eligible = torch.nonzero(ranking["scores"] < 0, as_tuple=False).flatten()
    if eligible.numel() < k:
        k = eligible.numel()
    chosen_local = torch.topk(
        ranking["scores"][eligible], k, largest=False, sorted=True
    ).indices
    chosen = eligible[chosen_local]
    layers = ranking["layers"][chosen].long()
    locals_ = ranking["locals"][chosen].long()
    kinds = ranking["kinds"][chosen]
    ids = ranking["ids"][chosen]
    for layer_id in layers.unique(sorted=True).tolist():
        mask = layers == layer_id
        name = names[layer_id]
        positions = locals_[mask]
        layer_kinds = kinds[mask]
        layer_ids = ids[mask]
        m1 = layer_kinds == 0
        m2 = ~m1
        if m1.any():
            state["codecs"][name]["idx1"][positions[m1]] = layer_ids[m1]
        if m2.any():
            state["codecs"][name]["idx2"][positions[m2]] = layer_ids[m2]
    state["metadata"] = dict(state.get("metadata", {}))
    state["metadata"]["assignment_refresh"] = {
        "changed_packages": int(k),
        "coverage": k / ranking["scores"].numel(),
        "selection": "disjoint train acceptance+sanity, validation macro gate",
    }
    return state


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = torch.device("cuda")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).to(device).eval()
    model.device_holder = device
    for p in model.parameters():
        p.requires_grad_(False)
    install_state(model, state)

    train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(x for x in train["text"] if x.strip())
    ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0, :TRAIN_TOKENS]
    all_windows = windows(ids)
    order = torch.randperm(len(all_windows), generator=torch.Generator().manual_seed(SEED + 1))
    # The parent 200-step run consumed order[:200]. Everything below is disjoint.
    cursor = 200
    proposal = [all_windows[i] for i in order[cursor:cursor + PROPOSAL_WINDOWS]]
    cursor += PROPOSAL_WINDOWS
    acceptance = [all_windows[i] for i in order[cursor:cursor + ACCEPT_WINDOWS]]
    cursor += ACCEPT_WINDOWS
    sanity = [all_windows[i] for i in order[cursor:cursor + SANITY_WINDOWS]]

    valid_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    valid_text = "\n\n".join(x for x in valid_ds["text"] if x.strip())
    valid_ids = tokenizer(valid_text, return_tensors="pt", truncation=False)["input_ids"][0, :4096]
    valid_windows = windows(valid_ids)

    baseline = {
        "acceptance": evaluate(model, acceptance),
        "sanity": evaluate(model, sanity),
        "validation": evaluate(model, valid_windows),
    }
    print(json.dumps({"event": "baseline", **baseline}), flush=True)

    names, gradients = collect_weight_gradients(model, state, proposal)
    ranking = rank_all_replacements(state, names, gradients, device)
    total_packages = ranking["scores"].numel()
    trials = []
    trial_states = []
    for requested in CHANGE_COUNTS:
        coverage = requested / total_packages
        candidate = apply_topk(state, ranking, names, requested)
        install_state(model, candidate, names)
        accept_result = evaluate(model, acceptance)
        sanity_result = evaluate(model, sanity)
        row = {
            "coverage": coverage,
            "changed_packages": candidate["metadata"]["assignment_refresh"]["changed_packages"],
            "acceptance": accept_result,
            "sanity": sanity_result,
            "mean_internal_nll": (accept_result["nll"] + sanity_result["nll"]) / 2,
        }
        trials.append(row)
        trial_states.append(candidate)
        print(json.dumps({"event": "trial", **row}), flush=True)

    viable = [
        i for i, row in enumerate(trials)
        if row["acceptance"]["nll"] < baseline["acceptance"]["nll"]
        and row["sanity"]["nll"] < baseline["sanity"]["nll"]
    ]
    selected_index = min(viable, key=lambda i: trials[i]["mean_internal_nll"]) if viable else None
    accepted = False
    selected_state = state
    selected_valid = baseline["validation"]
    if selected_index is not None:
        candidate = trial_states[selected_index]
        install_state(model, candidate, names)
        candidate_valid = evaluate(model, valid_windows)
        trials[selected_index]["validation"] = candidate_valid
        if candidate_valid["nll"] < baseline["validation"]["nll"]:
            accepted = True
            selected_state = candidate
            selected_valid = candidate_valid

    # The test split is touched only after the internal and macro decisions.
    install_state(model, selected_state)
    test_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_text = "\n\n".join(x for x in test_ds["text"] if x.strip())
    test_ids = tokenizer(test_text, return_tensors="pt", truncation=False)["input_ids"][0, :4096]
    final_test = evaluate(model, windows(test_ids))

    selected_state["metadata"] = dict(selected_state.get("metadata", {}))
    selected_state["metadata"].update({
        "validation": selected_valid,
        "test": final_test,
        "assignment_refresh_accepted": accepted,
    })
    checkpoint = args.out / "e19r_3bit_assignment_best.pt"
    torch.save(selected_state, checkpoint)
    report = {
        "experiment": "E19-R 3-bit functional assignment refresh",
        "parent": str(args.checkpoint),
        "parent_sha256": sha256(args.checkpoint),
        "total_mlp_packages": total_packages,
        "negative_first_order_candidates": ranking["negative_candidates"],
        "split_windows": {
            "parent_value_qat": 200,
            "proposal": len(proposal),
            "acceptance": len(acceptance),
            "sanity": len(sanity),
        },
        "baseline": baseline,
        "trials": trials,
        "selected_trial_index": selected_index,
        "accepted": accepted,
        "final_validation": selected_valid,
        "final_test": final_test,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
            "file_bytes": checkpoint.stat().st_size,
        },
        "runtime_seconds": time.time() - started,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"event": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()
