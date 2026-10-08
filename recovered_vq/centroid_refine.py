"""Strict, reproducible centroid refinement for the recovered 4x4 Fisher VQ.

This extends ``weighted_vq.py`` without modifying the recovered source.  It:

* rebuilds the legacy seed-123 K=65536 baseline with a K=16 selective residual;
* simulates the stored INT8 codebooks (per-centroid FP16 scale);
* counts residual locations using an exact Rice-coded bit length;
* freezes fixed WikiText-2 fit/held-out/evaluation token IDs;
* sequentially optimizes sparse layer-specific INT4 centroid deltas against
  clean teacher block outputs from the actual student input trajectory; and
* accepts steps only when held-out hidden-state NMSE improves.

The scientific checkpoint includes extra reproducibility sidecars (Fisher and
token IDs).  Those sidecars are reported separately from deployable codec bits.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weighted_vq as legacy  # noqa: E402


SEED = 123
K = 65536
K_RESIDUAL = 16
BLOCK = 4
TARGET_BPW = 1.20
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/workspace/centroid_refine"))
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--fit-tokens", type=int, default=2048)
    p.add_argument("--heldout-tokens", type=int, default=1024)
    p.add_argument("--eval-tokens", type=int, default=4096)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--lambda-local", type=float, default=0.1)
    p.add_argument("--beta", type=float, default=1e-4)
    p.add_argument("--baseline-gate", type=float, default=50.0)
    return p.parse_args()


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def sha(x):
    return hashlib.sha256(x.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def tokens(tokenizer, split, count, offset=0):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return ids[offset:offset + count].clone()


def blocks(w):
    r, c = w.shape
    assert r % 4 == 0 and c % 4 == 0
    return w.view(r // 4, 4, c // 4, 4).permute(0, 2, 1, 3).reshape(-1, 16)


def unblock(x, shape):
    r, c = shape
    return x.view(r // 4, c // 4, 4, 4).permute(0, 2, 1, 3).reshape(r, c)


def layer_of(name):
    return int(name.split(".")[0][1:])


def get_module(model, name):
    li = layer_of(name); family = name.split(".")[2]
    parent = model.model.layers[li].self_attn if ".SA." in name else model.model.layers[li].mlp
    return getattr(parent, family)


def quantize_int8_rows(c):
    scale = (c.abs().amax(1) / 127).clamp_min(torch.finfo(torch.float16).tiny).half()
    q = torch.round(c / scale.float()[:, None]).clamp(-127, 127).to(torch.int8)
    return q, scale, q.float() * scale.float()[:, None]


def quantize_int4_rows(delta):
    scale = (delta.abs().amax(1) / 7).clamp_min(torch.finfo(torch.float16).tiny).half()
    q = torch.round(delta / scale.float()[:, None]).clamp(-7, 7).to(torch.int8)
    return q, scale, q.float() * scale.float()[:, None]


def rice_bits(positions):
    """Exact bits for sorted positions with Rice-coded nonnegative gaps."""
    if positions.numel() == 0:
        return 32 + 5, 0
    p = positions.long().sort().values
    gaps = torch.empty_like(p)
    gaps[0] = p[0]
    gaps[1:] = p[1:] - p[:-1] - 1
    scores = []
    for k in range(17):
        scores.append(int(((gaps >> k) + 1 + k).sum()) + 32 + 5)
    k = int(np.argmin(scores))
    return scores[k], k


class Codec:
    pass


def build_baseline(w, fisher):
    codec = Codec()
    codec.names = list(w)
    codec.slices = {}
    vecs, imps, cursor = [], [], 0
    for name in codec.names:
        vb, ib = blocks(w[name]), blocks(fisher[name])
        codec.slices[name] = (cursor, cursor + vb.shape[0])
        cursor += vb.shape[0]; vecs.append(vb); imps.append(ib)
    vecs = torch.cat(vecs).float(); imps = torch.cat(imps).float()
    codec.total_weights = sum(x.numel() for x in w.values())
    codec.nblocks = vecs.shape[0]
    print(f"Blocks={codec.nblocks:,}; training legacy K=65536 codebook", flush=True)
    idx = torch.randperm(codec.nblocks)[:min(codec.nblocks, 2_000_000)]
    centers, _ = legacy.fast_kmeans(vecs[idx], K, n_iter=25)
    codec.base_q, codec.base_scale, codec.centers = quantize_int8_rows(centers.float())
    print("Reassigning against stored INT8 codebook", flush=True)
    codec.assign = legacy.weighted_assign(vecs, imps, codec.centers)
    recon = codec.centers[codec.assign]
    residual = vecs - recon
    perr = (imps * residual.square()).sum(1) / imps.sum(1).clamp_min(1e-20)
    imp_mean = imps.mean(1)
    need = (perr > torch.quantile(perr, 0.95)) & (imp_mean > torch.quantile(imp_mean, 0.30))
    codec.res_positions = need.nonzero().flatten().to(torch.int32)
    print(f"Residual blocks={int(need.sum()):,} ({100*float(need.float().mean()):.3f}%)", flush=True)
    rc, _ = legacy.fast_kmeans(residual[need], K_RESIDUAL, n_iter=15)
    codec.res_q, codec.res_scale, codec.res_centers = quantize_int8_rows(rc.float())
    codec.res_ids = legacy.weighted_assign(residual[need], imps[need], codec.res_centers, verbose=False).to(torch.int8)
    rice, codec.rice_k = rice_bits(codec.res_positions)
    codec.base_bits = codec.nblocks * 16 + K * 16 * 8 + K * 16
    codec.residual_bits = rice + codec.res_ids.numel() * 4 + K_RESIDUAL * 16 * 8 + K_RESIDUAL * 16
    codec.metadata_bits = 512 + sum((len(n.encode()) + 1 + 24) * 8 for n in codec.names)
    codec.total_bits = codec.base_bits + codec.residual_bits + codec.metadata_bits
    codec.bpw = codec.total_bits / codec.total_weights
    codec._res_global = torch.full((codec.nblocks,), -1, dtype=torch.int16)
    codec._res_global[codec.res_positions.long()] = codec.res_ids.short()
    codec.baseline_wmse = float((imps * (vecs - codec.centers[codec.assign] -
        torch.where(codec._res_global[:, None] >= 0,
                    codec.res_centers[codec._res_global.clamp_min(0).long()], 0)).square()).sum() / imps.sum())
    return codec


def reconstruct_matrix(codec, name, layer_delta=None, assignment=None):
    s, e = codec.slices[name]
    ids = codec.assign[s:e] if assignment is None else assignment[s:e]
    x = codec.centers[ids.long()].clone()
    if layer_delta is not None:
        x += layer_delta[ids.long()]
    rid = codec._res_global[s:e]
    mask = rid >= 0
    if mask.any(): x[mask] += codec.res_centers[rid[mask].long()]
    return unblock(x, get_module_shape[name])


get_module_shape = {}


def restore(model, originals, layers=None):
    with torch.no_grad():
        for name, value in originals.items():
            if layers is None or layer_of(name) in layers:
                get_module(model, name).weight.copy_(value.cuda())


def apply_codec(model, codec, deltas, through_layer, assignment=None):
    with torch.no_grad():
        for name in codec.names:
            li = layer_of(name)
            if li <= through_layer:
                q = reconstruct_matrix(codec, name, deltas.get(li), assignment)
                get_module(model, name).weight.copy_(q.cuda())


class StopLayer(Exception):
    pass


def run_to_layer(model, ids, layer, grad=False):
    box = {}
    def hook(_m, _inp, out):
        box["y"] = out[0] if isinstance(out, tuple) else out
        raise StopLayer
    h = model.model.layers[layer].register_forward_hook(hook)
    try:
        ctx = torch.enable_grad() if grad else torch.inference_mode()
        with ctx:
            model(ids.cuda(), use_cache=False)
    except StopLayer:
        pass
    finally:
        h.remove()
    return box["y"]


@torch.inference_mode()
def teacher_targets(model, ids, seq, layers=4):
    out = {i: [] for i in range(layers)}
    for start in range(0, ids.numel() - seq + 1, seq):
        x = ids[start:start + seq].unsqueeze(0).cuda()
        hs = model(x, use_cache=False, output_hidden_states=True).hidden_states
        for i in range(layers): out[i].append(hs[i + 1].float().cpu())
    return out


@torch.inference_mode()
def hidden_error(model, ids, targets, seq, layer):
    se = energy = dot = pn = tn = 0.0; ix = 0
    for start in range(0, ids.numel() - seq + 1, seq):
        x = ids[start:start + seq].unsqueeze(0)
        pred = run_to_layer(model, x, layer).float().cpu(); target = targets[layer][ix]
        se += float((pred-target).square().sum()); energy += float(target.square().sum())
        dot += float((pred*target).sum()); pn += float(pred.square().sum()); tn += float(target.square().sum()); ix += 1
    return {"nmse": se/max(energy, 1e-20), "cosine": dot/max(math.sqrt(pn*tn), 1e-20)}


@torch.inference_mode()
def ppl(model, ids, seq):
    losses = []
    for start in range(0, ids.numel() - seq + 1, seq):
        x = ids[start:start+seq].unsqueeze(0).cuda()
        losses.append(float(model(x, labels=x, use_cache=False).loss))
    nll = float(np.mean(losses)); return {"nll": nll, "ppl": math.exp(min(nll, 80))}


def aggregate_centroid_grad(codec, model, layer):
    sums = torch.zeros(K, 16); counts = torch.zeros(K)
    per_family = {}
    for name in codec.names:
        if layer_of(name) != layer: continue
        g = blocks(get_module(model, name).weight.grad.detach().cpu().float())
        s, e = codec.slices[name]; ids = codec.assign[s:e].long()
        local = torch.zeros(K, 16); local.index_add_(0, ids, g)
        cnt = torch.bincount(ids, minlength=K).float()
        sums += local; counts += cnt
        per_family[name.split(".")[2]] = float(g.square().sum())
    return sums / counts.clamp_min(1)[:, None], counts, per_family


def set_trainable(model, layer, enabled):
    for p in model.parameters(): p.requires_grad_(False)
    if enabled:
        for name in codec_global.names:
            if layer_of(name) == layer: get_module(model, name).weight.requires_grad_(True)


codec_global = None


def refine_layer(model, codec, originals, deltas, fit_ids, held_ids, fit_targets,
                 held_targets, seq, layer, entries, steps, lr, lam, beta):
    restore(model, originals); apply_codec(model, codec, deltas, layer)
    before = hidden_error(model, held_ids, held_targets, seq, layer)
    history = []; selected = None; delta = torch.zeros(K, 16); m = torch.zeros_like(delta); v = torch.zeros_like(delta)
    best_delta = delta.clone(); best = before["nmse"]
    for step in range(1, steps + 1):
        restore(model, originals, [layer]); apply_codec(model, codec, deltas, layer - 1)
        local_targets = []
        for start in range(0, fit_ids.numel() - seq + 1, seq):
            local_targets.append(run_to_layer(model, fit_ids[start:start+seq].unsqueeze(0), layer).detach())
        apply_codec(model, codec, {**deltas, layer: best_delta}, layer)
        set_trainable(model, layer, True)
        for name in codec.names:
            if layer_of(name) == layer: get_module(model, name).weight.grad = None
        fit_teacher = fit_local = 0.0; ix = 0
        for start in range(0, fit_ids.numel() - seq + 1, seq):
            x = fit_ids[start:start+seq].unsqueeze(0)
            pred = run_to_layer(model, x, layer, grad=True).float()
            teacher = fit_targets[layer][ix].cuda(); local = local_targets[ix].float()
            lt = (pred-teacher).square().sum()/teacher.square().sum().clamp_min(1e-20)
            ll = (pred-local).square().sum()/local.square().sum().clamp_min(1e-20)
            ((lt + lam*ll) / max(len(local_targets), 1)).backward()
            fit_teacher += float(lt.detach()); fit_local += float(ll.detach()); ix += 1
        g, counts, family = aggregate_centroid_grad(codec, model, layer)
        if selected is None:
            used = counts > 0
            score = g.square().sum(1); score[~used] = -1
            selected = torch.topk(score, min(entries, int(used.sum()))).indices
        mask = torch.zeros(K, dtype=torch.bool); mask[selected] = True
        g += 2*beta*best_delta/max(int(mask.sum())*16, 1); g[~mask] = 0
        m = .9*m + .1*g; v = .999*v + .001*g.square()
        cand = best_delta - lr*(m/(1-.9**step))/(v/(1-.999**step)).sqrt().add(1e-8)
        cand[~mask] = 0; cand.clamp_(-0.05, 0.05)
        q, scale, deq = quantize_int4_rows(cand[selected]); quant = torch.zeros_like(cand); quant[selected] = deq
        restore(model, originals); apply_codec(model, codec, {**deltas, layer: quant}, layer)
        val = hidden_error(model, held_ids, held_targets, seq, layer)
        accepted = val["nmse"] < best
        if accepted: best = val["nmse"]; best_delta = quant
        else: lr *= .5
        row = {"step": step, "fit_teacher": fit_teacher/max(ix,1), "fit_local": fit_local/max(ix,1),
               "validation_nmse": val["nmse"], "validation_cosine": val["cosine"],
               "accepted": accepted, "learning_rate_next": lr, "projection_gradient_energy": family}
        history.append(row); print(f"L{layer} STEP {step}: {row}", flush=True)
        set_trainable(model, layer, False)
    selected = selected[best_delta[selected].abs().amax(1) > 0]
    q, scale, deq = quantize_int4_rows(best_delta[selected])
    final = torch.zeros_like(best_delta); final[selected] = deq
    restore(model, originals); apply_codec(model, codec, {**deltas, layer: final}, layer)
    after = hidden_error(model, held_ids, held_targets, seq, layer)
    return final, selected, q, scale, before, after, history


def save_checkpoint(out, codec, fisher, token_ids, delta_records, config, results):
    state = {"format": "strict-vq4-centroid-v1", "base_codebook_q": codec.base_q,
        "base_codebook_scale": codec.base_scale, "assignments": codec.assign.to(torch.int32),
        "residual_positions": codec.res_positions, "residual_ids": codec.res_ids,
        "residual_codebook_q": codec.res_q, "residual_codebook_scale": codec.res_scale,
        "rice_k": codec.rice_k, "slices": codec.slices, "fisher": fisher, "tokens": token_ids,
        "centroid_deltas": delta_records, "config": config, "results": results}
    torch.save(state, out / "centroid_refined_checkpoint.pt")
    dump(out / "manifest.json", {"format": state["format"], "config": config, "results": results,
        "checkpoint_bytes": (out/"centroid_refined_checkpoint.pt").stat().st_size,
        "codebook_sha256": sha(codec.base_q), "token_sha256": {k:sha(v) for k,v in token_ids.items()}})


def main():
    global codec_global, get_module_shape
    a = parse_args(); a.output.mkdir(parents=True, exist_ok=True); seed_all()
    started = time.time()
    model, w = legacy.load(4); model.to("cuda").eval()
    originals = {n:x.detach().cpu().clone() for n,x in w.items()}
    get_module_shape = {n:tuple(x.shape) for n,x in originals.items()}
    tokenizer = AutoTokenizer.from_pretrained(legacy.MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    fit = tokens(tokenizer, "train", a.fit_tokens, 0)
    held = tokens(tokenizer, "train", a.heldout_tokens, a.fit_tokens)
    evaluation = tokens(tokenizer, "test", a.eval_tokens, 0)
    token_ids = {"fit":fit, "heldout":held, "evaluation":evaluation}
    restore(model, originals)
    print("Computing recovered gradient-squared Fisher", flush=True)
    fisher = {n:x.cpu() for n,x in legacy.compute_fisher(model, originals).items()}
    seed_all(); codec = build_baseline(originals, fisher); codec_global = codec
    print(f"Strict baseline BPW={codec.bpw:.6f}; wMSE={codec.baseline_wmse:.8g}", flush=True)
    model.to("cuda")
    restore(model, originals); fit_targets = teacher_targets(model, fit, a.sequence_length)
    held_targets = teacher_targets(model, held, a.sequence_length)
    bf16 = ppl(model, evaluation, a.sequence_length)
    deltas = {}; baseline_prefix = []
    for n in range(1, 5):
        restore(model, originals); apply_codec(model, codec, deltas, n-1)
        metric = ppl(model, evaluation, a.sequence_length)
        baseline_prefix.append({"layers":n, **metric,
            "delta_ppl_pct":100*(metric["ppl"]/bf16["ppl"]-1)})
        print("WIKITEXT BASELINE", baseline_prefix[-1], flush=True)
    baseline_delta = baseline_prefix[-1]["delta_ppl_pct"]
    baseline_1l_delta = baseline_prefix[0]["delta_ppl_pct"]
    # Sparse delta entry budget: 16-bit centroid ID + 16-bit scale + 64 INT4 data bits.
    remaining = math.floor(TARGET_BPW*codec.total_weights - codec.total_bits)
    max_entries = max(0, remaining // 96)
    shares = [0.15, 0.20, 0.40, 0.25]
    entries = [int(max_entries*s) for s in shares]; entries[-1] += max_entries-sum(entries)
    print(f"Delta budget={max_entries} centroid-layer entries; allocation={entries}", flush=True)
    delta_records = {}; layer_rows = []
    if baseline_1l_delta <= a.baseline_gate:
        for li in range(4):
            d, selected, q, scale, before, after, hist = refine_layer(model, codec, originals, deltas,
                fit, held, fit_targets, held_targets, a.sequence_length, li, entries[li], a.steps,
                a.learning_rate, a.lambda_local, a.beta)
            deltas[li] = d
            delta_records[li] = {"centroid_ids":selected.to(torch.int32), "q_int4_in_int8":q, "scale_fp16":scale}
            layer_rows.append({"layer":li, "before":before, "after":after,
                "modified_centroids":int(selected.numel()), "history":hist})
    else:
        print(f"BASELINE GATE FAIL: WikiText 1L delta {baseline_1l_delta:.3f}% > {a.baseline_gate}%", flush=True)
    delta_bits = sum(v["centroid_ids"].numel()*96 for v in delta_records.values())
    final_bits = codec.total_bits + delta_bits
    prefix = []
    for n in range(5):
        restore(model, originals)
        if n: apply_codec(model, codec, deltas, n-1)
        metric = bf16 if n==0 else ppl(model, evaluation, a.sequence_length)
        prefix.append({"layers":n, **metric, "delta_ppl_pct":0 if n==0 else 100*(metric["ppl"]/bf16["ppl"]-1)})
        print("PREFIX", prefix[-1], flush=True)
    config = {"seed":SEED, "model":legacy.MODEL_NAME, "K":K, "K_residual":K_RESIDUAL,
        "block":"4x4", "fit_tokens":a.fit_tokens, "heldout_tokens":a.heldout_tokens,
        "eval_tokens":a.eval_tokens, "sequence_length":a.sequence_length,
        "lambda_local":a.lambda_local, "beta":a.beta, "steps":a.steps,
        "gpu":torch.cuda.get_device_name(), "torch":torch.__version__, "platform":platform.platform()}
    results = {"baseline_bpw":codec.bpw, "base_bits":codec.base_bits,
        "residual_bits":codec.residual_bits, "metadata_bits":codec.metadata_bits,
        "delta_bits":delta_bits, "total_bits":final_bits, "total_bpw":final_bits/codec.total_weights,
        "baseline_prefix_results":baseline_prefix,
        "baseline_wikitext_1l_delta_ppl_pct":baseline_1l_delta,
        "baseline_wikitext_4l_delta_ppl_pct":baseline_delta, "layer_refinement":layer_rows,
        "prefix_results":prefix, "elapsed_seconds":time.time()-started,
        "success":final_bits/codec.total_weights <=1.20 and prefix[4]["delta_ppl_pct"]<50,
        "breakthrough":final_bits/codec.total_weights <=1.20 and prefix[4]["delta_ppl_pct"]<10}
    dump(a.output/"results.json", results); save_checkpoint(a.output, codec, fisher, token_ids, delta_records, config, results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__": main()
