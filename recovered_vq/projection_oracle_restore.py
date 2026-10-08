"""Layer-0 one-at-a-time BF16/original-weight oracle restoration map."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import waverel_checkpoint_patterns as base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--codec", type=Path,
                   default=Path("/workspace/transform_patterns/none.bin"))
    p.add_argument("--tokens", type=Path,
                   default=Path("/workspace/waverel_patterns/fixed_token_ids.pt"))
    p.add_argument("--output", type=Path,
                   default=Path("/workspace/projection_oracle_restore/results.json"))
    p.add_argument("--sequence-length", type=int, default=256)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    base.seed_all()

    AutoTokenizer.from_pretrained(base.MODEL)  # Verify the frozen model identity/cache.
    model = AutoModelForCausalLM.from_pretrained(
        base.MODEL, torch_dtype=torch.float32, device_map={"": "cuda"}).eval()
    originals = {n: m.weight.detach().cpu().clone()
                 for n, m in base.module_map(model).items()}
    compressed, meta = base.decode(a.codec)
    if meta["method"] != "None":
        raise RuntimeError(f"Expected frozen None codec, got {meta['method']!r}")
    token_state = torch.load(a.tokens, map_location="cpu", weights_only=True)
    evaluation = token_state["evaluation"]

    base.restore(model, originals)
    reference = base.perplexity(model, evaluation, a.sequence_length)
    base.apply(model, compressed)
    fully_compressed = base.perplexity(model, evaluation, a.sequence_length)

    rows = []
    for name in base.PROJS:
        candidate = dict(compressed)
        candidate[name] = originals[name]
        base.apply(model, candidate)
        metric = base.perplexity(model, evaluation, a.sequence_length)
        rows.append({
            "restored_projection": name,
            "ppl": metric["ppl"],
            "delta_ppl_vs_reference_pct": 100 * (metric["ppl"] / reference["ppl"] - 1),
            "ppl_reduction_vs_compressed_pct": 100 * (1 - metric["ppl"] / fully_compressed["ppl"]),
            "nll_reduction_vs_compressed": fully_compressed["nll"] - metric["nll"],
            "weights_restored": originals[name].numel(),
        })
    rows.sort(key=lambda x: x["nll_reduction_vs_compressed"], reverse=True)
    result = {
        "model": base.MODEL,
        "layer": 0,
        "codec": str(a.codec),
        "codec_sha256": hashlib.sha256(a.codec.read_bytes()).hexdigest(),
        "codec_bpw": a.codec.stat().st_size * 8 / sum(w.numel() for w in originals.values()),
        "evaluation_token_sha256": hashlib.sha256(evaluation.numpy().tobytes()).hexdigest(),
        "reference": reference,
        "fully_compressed": fully_compressed,
        "oracle_restorations_ranked": rows,
        "ranking_metric": "NLL reduction versus fully compressed; additive and stable when PPL is large",
    }
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
