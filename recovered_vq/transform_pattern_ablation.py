"""Strict Layer-0 transform ablation for the 4x4 WaveRel pattern codec.

Only the orthogonal transform varies. Pattern count, scale table, residual
fraction/codebook, calibration split, evaluation split, and packed format are
held fixed. Every reported model is decoded from its on-disk payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import waverel_checkpoint_patterns as base


GROUP = 128
CANDIDATE_SEEDS = (11, 29, 47, 83)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/workspace/transform_patterns"))
    p.add_argument("--tokens", type=Path,
                   default=Path("/workspace/waverel_patterns/fixed_token_ids.pt"))
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--hard-fraction", type=float, default=0.20)
    return p.parse_args()


def hadamard(n=GROUP):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / math.sqrt(n)


def dct(n=GROUP):
    k = torch.arange(n, dtype=torch.float64)[:, None]
    j = torch.arange(n, dtype=torch.float64)[None, :]
    out = torch.cos(math.pi / n * (j + 0.5) * k) * math.sqrt(2 / n)
    out[0] /= math.sqrt(2)
    return out.float()


def seeded_rotation(seed):
    """Canonical P/S/H recipe, generated entirely from its stored seed."""
    g = torch.Generator().manual_seed(seed)
    p = torch.randperm(GROUP, generator=g)
    s = torch.randint(0, 2, (GROUP,), generator=g).float().mul_(2).sub_(1)
    # R = P S H: permute rows of S H. Orthogonal and seed-decodable.
    return (s[:, None] * hadamard())[p]


def left_groups(w, mat):
    r, c = w.shape
    assert r % GROUP == 0
    device = "cuda" if torch.cuda.is_available() else w.device
    out = torch.einsum("ij,gjk->gik", mat.to(device),
                       w.to(device).view(r // GROUP, GROUP, c)).reshape(r, c)
    return out.cpu()


def right_groups(w, mat):
    r, c = w.shape
    assert c % GROUP == 0
    device = "cuda" if torch.cuda.is_available() else w.device
    x = w.to(device).view(r, c // GROUP, GROUP)
    return torch.einsum("rgj,ij->rgi", x, mat.to(device)).reshape(r, c).cpu()


def transform_one(w, spec, inverse=False):
    kind = spec["kind"]
    if kind == "none":
        return w.clone()
    if kind in ("hadamard", "hadamard_row", "dct"):
        m = hadamard() if kind.startswith("hadamard") else dct()
        if inverse:
            # Forward Z=L W R^T; inverse W=L^T Z R.
            out = left_groups(w, m.T)
            if kind != "hadamard_row":
                out = right_groups(out, m.T)
            return out
        out = left_groups(w, m)
        if kind != "hadamard_row":
            out = right_groups(out, m)
        return out
    if kind == "activation_aware":
        r = seeded_rotation(int(spec["seed"]))
        # Forward Z=W R; right_groups computes W R^T, hence pass R.T.
        return right_groups(w, r if inverse else r.T)
    raise ValueError(kind)


def transform_fisher(f, spec):
    """Diagonal Hessian under the orthogonal coordinate transform."""
    kind = spec["kind"]
    if kind == "none":
        return f.clone()
    if kind in ("hadamard", "hadamard_row", "dct"):
        m = hadamard() if kind.startswith("hadamard") else dct()
        out = left_groups(f, m.square())
        if kind != "hadamard_row":
            out = right_groups(out, m.square().T)
        return out.clamp_min_(1e-20)
    if kind == "activation_aware":
        r = seeded_rotation(int(spec["seed"]))
        # G = F R^2 for Z=W R and W=Z R^T.
        return right_groups(f, r.square().T).clamp_min_(1e-20)
    raise ValueError(kind)


def transform_dict(values, specs, inverse=False):
    return {n: transform_one(w, specs[n], inverse) for n, w in values.items()}


@torch.inference_mode()
def proxy_error(original, transformed, activations, specs):
    """Cheap 6-bit scalar proxy used only to select activation-aware seeds."""
    scores = {}
    for name, z in transformed.items():
        scale = z.abs().amax(1, keepdim=True).clamp_min(1e-12) / 31
        zq = torch.round(z / scale).clamp(-31, 31) * scale
        wq = transform_one(zq, specs[name], inverse=True)
        x = activations[name][:256].cuda()
        y = x @ original[name].cuda().T
        e = x @ (original[name] - wq).cuda().T
        scores[name] = float(e.square().sum() / y.square().sum().clamp_min(1e-30))
    return scores


def choose_activation_specs(originals, activations):
    chosen, audit = {}, {}
    for name, w in originals.items():
        rows = []
        for seed in CANDIDATE_SEEDS:
            spec = {"kind": "activation_aware", "group": GROUP, "seed": seed,
                    "recipe": "R=P*S*H; Z=W*R"}
            z = transform_one(w, spec)
            score = proxy_error({name: w}, {name: z}, {name: activations[name]}, {name: spec})[name]
            rows.append({"seed": seed, "proxy_activation_mse_ratio": score})
        winner = min(rows, key=lambda x: x["proxy_activation_mse_ratio"])
        chosen[name] = {"kind": "activation_aware", "group": GROUP,
                        "seed": winner["seed"], "recipe": "R=P*S*H; Z=W*R"}
        audit[name] = rows
    return chosen, audit


def reconstruct_from_disk(path):
    transformed, meta = base.decode(path)
    specs = meta["transform_specs"]
    return transform_dict(transformed, specs, inverse=True), meta


def nearest_error(vecs, imp, recon):
    num = (imp * (vecs - recon).square()).sum()
    den = imp.sum().clamp_min(1e-30)
    return float(num / den)


def train_candidate(name, specs, originals, fisher, model, bf16, eval_ids,
                    activations, seq, outdir, hard_fraction):
    started = time.time()
    transformed = transform_dict(originals, specs)
    tfisher = {n: transform_fisher(fisher[n], specs[n]) for n in originals}
    vecs, imp, matrix_index, slices = base.extract(transformed, tfisher)
    raw_scale = vecs.abs().amax(1).clamp_min(1e-12)
    scale_tables, scale_ids, decoded_scale = base.quantize_scales(
        raw_scale, matrix_index, len(originals))
    normalized = vecs / raw_scale[:, None]
    target = vecs / decoded_scale[:, None]

    path = outdir / (name.lower().replace(" ", "_").replace("-", "_") + ".bin")
    if path.exists():
        meta0, stored_scales, checkpoint, codes, ids, stored_sids, mask, rids, rq, rs = base.read_codec(path)
        if meta0.get("transform_specs") != specs:
            raise RuntimeError(f"Refusing incompatible resume payload: {path}")
        if abs(meta0["hard_count"] / meta0["nblocks"] - hard_fraction) > 1e-6:
            raise RuntimeError(f"Residual budget differs in resume payload: {path}")
        if checkpoint.shape[0] == 1:
            patterns = checkpoint[0][codes.long()]
        else:
            fam = (torch.arange(codes.shape[0]) // base.PATTERNS_PER_FAMILY).long()
            patterns = checkpoint[fam[:, None], codes.long()]
        stored_scale = stored_scales[matrix_index.long(), stored_sids.long()].float()
        pattern_blocks = patterns[ids.long()].float() * stored_scale[:, None]
        residual = rq.float() * rs.float()[:, None]
        final_blocks = pattern_blocks.clone()
        final_blocks[mask] += residual[rids.long()]
        print(f"RESUME verified payload {path}", flush=True)
    else:
        # Reset the RNG so each transform gets the same stochastic training stream.
        base.seed_all()
        checkpoint, codes, patterns, ids, _ = base.train_waverel_patterns(
            normalized, target, imp)
        pattern_blocks = patterns[ids.long()] * decoded_scale[:, None]
        mask, rids, rq, rs, final_blocks, _ = base.train_hard_residual(
            vecs, imp, pattern_blocks, hard_fraction)
        base.write_codec(
            path, name, slices, scale_tables, scale_ids, checkpoint, codes, ids,
            mask, rids, rq, rs,
            meta_extra={"transform_specs": specs, "transform_group": GROUP,
                        "codec_experiment": "layer0_transform_ablation_v1"})
    total_weights = sum(w.numel() for w in originals.values())
    bpw = path.stat().st_size * 8 / total_weights
    if bpw > 1.20:
        raise RuntimeError(f"{name} exceeds budget: {bpw:.8f} BPW")

    # Verify the serialized transformed reconstruction exactly, then invert.
    disk_z, _ = base.decode(path)
    for proj, (s, e, shape) in slices.items():
        expected = base.unblock(final_blocks[s:e], shape)
        assert torch.equal(disk_z[proj], expected), proj
    recon, meta = reconstruct_from_disk(path)
    base.apply(model, recon)
    ppl = base.perplexity(model, eval_ids, seq)
    act, per_projection = base.activation_metrics(originals, recon, activations)

    reuse = torch.bincount(ids.long(), minlength=base.PATTERNS)
    matrix_ne = []
    for proj, (s, e, _) in slices.items():
        matrix_ne.append({"family": proj,
                          "nearest_pattern_error": nearest_error(
                              vecs[s:e], imp[s:e], pattern_blocks[s:e])})
    for row in per_projection:
        row["nearest_pattern_error"] = next(
            x["nearest_pattern_error"] for x in matrix_ne if x["family"] == row["family"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "method": name,
        "bpw": bpw,
        "file_bytes": path.stat().st_size,
        "sha256": digest,
        "ppl": ppl["ppl"],
        "delta_ppl_pct": 100 * (ppl["ppl"] / bf16["ppl"] - 1),
        "activation_error": act,
        "nearest_pattern_error": nearest_error(vecs, imp, pattern_blocks),
        "pattern_reuse": {"unused": int((reuse == 0).sum()),
                          "median": float(reuse.float().median()),
                          "max": int(reuse.max())},
        "projection_metrics": per_projection,
        "hard_block_pct": 100 * int(mask.sum()) / int(mask.numel()),
        "transform_specs": meta["transform_specs"],
        "elapsed_seconds": time.time() - started,
    }


def main():
    a = arguments()
    a.output.mkdir(parents=True, exist_ok=True)
    base.seed_all()
    tokenizer = AutoTokenizer.from_pretrained(base.MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        base.MODEL, torch_dtype=torch.float32, device_map={"": "cuda"}).eval()
    originals = {n: m.weight.detach().cpu().clone() for n, m in base.module_map(model).items()}

    token_state = torch.load(a.tokens, map_location="cpu", weights_only=True)
    fit, held, evaluation = token_state["fit"], token_state["heldout"], token_state["evaluation"]
    token_hashes = {k: hashlib.sha256(v.numpy().tobytes()).hexdigest()
                    for k, v in token_state.items()}
    base.restore(model, originals)
    bf16 = base.perplexity(model, evaluation, a.sequence_length)
    activations = base.capture_inputs(model, held, a.sequence_length)

    legacy_names = {f"L0.{'SA' if n in base.PROJS[:4] else 'MLP'}.{n}": w
                    for n, w in originals.items()}
    fisher_legacy = base.legacy.compute_fisher(model, legacy_names)
    model.to("cuda")
    fisher = {n: fisher_legacy[f"L0.{'SA' if n in base.PROJS[:4] else 'MLP'}.{n}"].cpu()
              for n in originals}

    fixed = lambda kind: {n: {"kind": kind, "group": GROUP} for n in originals}
    activation_specs, selection_audit = choose_activation_specs(originals, activations)
    candidates = [
        ("None", fixed("none")),
        ("Hadamard", fixed("hadamard")),
        ("Hadamard row-only", fixed("hadamard_row")),
        ("DCT", fixed("dct")),
        ("Activation-aware", activation_specs),
    ]
    results = []
    for name, specs in candidates:
        print(f"\n===== {name} =====", flush=True)
        base.restore(model, originals)
        result = train_candidate(name, specs, originals, fisher, model, bf16,
                                 evaluation, activations, a.sequence_length,
                                 a.output, a.hard_fraction)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    baseline = results[0]
    promising = [r for r in results[1:]
                 if r["activation_error"] < baseline["activation_error"]
                 and r["ppl"] < baseline["ppl"]]
    best = min(promising, key=lambda r: r["ppl"])["method"] if promising else None
    report = {
        "model": base.MODEL,
        "layer": 0,
        "bf16": bf16,
        "constraints": {"max_bpw": 1.20, "block": "4x4",
                        "patterns": base.PATTERNS, "scale_levels": base.SCALE_LEVELS,
                        "hard_fraction": a.hard_fraction,
                        "residual_k": base.K_RESIDUAL,
                        "transform_group": GROUP},
        "fixed_token_hashes": token_hashes,
        "activation_aware_selection": selection_audit,
        "results": results,
        "best_promising_transform": best,
        "decision_rule": "Must beat None on both held-out activation error and eval PPL.",
    }
    base.dump(a.output / "results.json", report)
    print("\nFINAL\n" + json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
