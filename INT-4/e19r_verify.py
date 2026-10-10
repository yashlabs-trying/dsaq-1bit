"""Fresh-process verification for an E19-R research checkpoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptq import decode as gptq_decode


MODEL = "Qwen/Qwen2.5-0.5B"
WINDOW = 128


@torch.inference_mode()
def evaluate(model, ids):
    total = 0.0
    count = 0
    for window in ids.split(WINDOW):
        if window.numel() < 2:
            continue
        logits = model(window[None], use_cache=False).logits
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            window[None, 1:].reshape(-1),
        )
        n = window.numel() - 1
        total += float(loss) * n
        count += n
    nll = total / count
    return {"nll": nll, "ppl": math.exp(nll), "tokens": count}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--tokens", type=int, default=4096)
    args = parser.parse_args()
    path = Path(args.checkpoint)
    device = torch.device("cuda")
    state = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()

    modules = dict(model.named_modules())
    for name, data in state["codecs"].items():
        module = modules[name]
        shape = module.weight.shape
        base = gptq_decode(
            data["base_codes"].to(device),
            data["base_scale"].to(device),
            data["base_zero"].to(device),
        ).reshape(shape).float()
        c1 = data["C1"].to(device).float()
        c2 = data["C2"].to(device).float()
        idx1 = data["idx1"].to(device).long()
        idx2 = data["idx2"].to(device).long()
        residual = (c1[idx1] + c2[idx2]).reshape(-1)[: module.weight.numel()].reshape(shape)
        module.weight.data.copy_((base + residual).to(module.weight.dtype))

    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(x for x in test["text"] if x.strip())
    ids = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0, :args.tokens].to(device)
    result = {
        "checkpoint": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "file_bytes": path.stat().st_size,
        "metadata": state["metadata"],
        "fresh_decode": evaluate(model, ids),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
