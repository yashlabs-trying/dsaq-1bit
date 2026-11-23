"""Allocate a strict global 2-bit budget using measured Hessian distortion."""
import json, math, os, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from additive_vq import additive_vq, decode, payload_bytes
from aq_block_experiment import collect_H, evaluate

MODEL = 'Qwen/Qwen2.5-0.5B'
OUT = Path(os.environ.get('FLASHRUN_OUTPUT', 'global_alloc_results'))
OUT.mkdir(exist_ok=True)
SEARCH_ITERS = int(os.environ.get('FLASHRUN_SEARCH_ITERS', '3'))
FINAL_ITERS = int(os.environ.get('FLASHRUN_ITERS', '10'))
FORMATS = ((4, 128), (2, 32), (2, 64))  # 1.75, 2.5, and 3 index bpw


@torch.inference_mode()
def quantize_matrix(w, h, group_size, codebook_size, iters):
    out_features, in_features = w.shape
    scale = w.abs().amax(1).clamp_min(1e-6)
    normalized = w / scale[:, None]
    blocks = normalized.reshape(-1, group_size)
    blocks_per_row = in_features // group_size
    positions = torch.arange(blocks.shape[0], device=w.device) % blocks_per_row
    h_blocks = h.view(
        blocks_per_row, group_size, blocks_per_row, group_size
    ).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
    codebooks, indices = additive_vq(
        blocks, 1, codebook_size, init_iters=6, iters=iters,
        H_blocks=h_blocks, block_pos=positions)
    codebooks = codebooks.half()
    rec = decode(codebooks.float(), indices, (out_features, in_features))

    # Optimize and store FP16 affine metadata in the original weight domain.
    rec_h = rec @ h
    ones_h = h.sum(0)
    a11 = (rec_h * rec).sum(1)
    a12 = rec_h.sum(1)
    a22 = h.sum()
    b1 = (rec_h * w).sum(1)
    b2 = (w * ones_h[None, :]).sum(1)
    determinant = (a11 * a22 - a12.square()).clamp_min(1e-12)
    scale = ((b1 * a22 - b2 * a12) / determinant).half().float()
    offset = ((a11 * b2 - a12 * b1) / determinant).half().float()
    rec = rec * scale[:, None] + offset[:, None]
    error = w - rec
    distortion = float(((error @ h) * error).sum())
    bits = int(8 * payload_bytes(indices, codebooks) + 32 * out_features)
    return rec, distortion, bits


def choose_formats(candidates, budget_bits):
    chosen = {name: 0 for name in candidates}
    used = sum(options[0]['bits'] for options in candidates.values())
    while True:
        best = None
        for name, options in candidates.items():
            current = chosen[name]
            for target in range(current + 1, len(options)):
                extra = options[target]['bits'] - options[current]['bits']
                gain = options[current]['distortion'] - options[target]['distortion']
                if extra <= 0 or gain <= 0 or used + extra > budget_bits:
                    continue
                score = gain / extra
                if best is None or score > best[0]:
                    best = (score, name, target, extra)
        if best is None:
            break
        _, name, target, extra = best
        chosen[name] = target
        used += extra
    return chosen, used


@torch.inference_mode()
def main():
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    train = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    cal_text = '\n\n'.join(t for t in train['text'] if t.strip())
    eval_text = '\n\n'.join(t for t in test['text'] if t.strip())
    ids = tokenizer(eval_text, return_tensors='pt')['input_ids'][0][:4096].cuda()
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    compressed_weights = sum(m.weight.numel() for _, m in layers)
    baseline = evaluate(model, ids)
    print('bf16', baseline, flush=True)
    hessians = collect_H(model, layers, tokenizer, cal_text, 2048, 128)

    candidates = {}
    for layer_index, (name, module) in enumerate(layers):
        w = module.weight.detach().float().cuda()
        h = hessians[name].cuda()
        h += 0.01 * h.diag().mean() * torch.eye(h.shape[0], device=h.device)
        options = []
        for group_size, codebook_size in FORMATS:
            _, distortion, bits = quantize_matrix(
                w, h, group_size, codebook_size, SEARCH_ITERS)
            options.append(dict(group_size=group_size,
                                codebook_size=codebook_size,
                                distortion=distortion, bits=bits))
        options.sort(key=lambda option: option['bits'])
        candidates[name] = options
        if layer_index % 20 == 0:
            print(f'profiled {layer_index}/{len(layers)}', flush=True)

    chosen, used_bits = choose_formats(candidates, 2 * compressed_weights)
    counts = {}
    for name, selected in chosen.items():
        option = candidates[name][selected]
        key = f"g{option['group_size']}_k{option['codebook_size']}"
        counts[key] = counts.get(key, 0) + 1
    print('allocation', counts, 'planned_bpw', used_bits / compressed_weights,
          flush=True)

    for layer_index, (name, module) in enumerate(layers):
        option = candidates[name][chosen[name]]
        w = module.weight.detach().float().cuda()
        h = hessians[name].cuda()
        h += 0.01 * h.diag().mean() * torch.eye(h.shape[0], device=h.device)
        rec, _, _ = quantize_matrix(
            w, h, option['group_size'], option['codebook_size'], FINAL_ITERS)
        module.weight.copy_(rec.to(module.weight.dtype))
        if layer_index % 30 == 0:
            print(f'encoded {layer_index}/{len(layers)}', flush=True)

    result = evaluate(model, ids)
    result['bits_per_compressed_weight'] = used_bits / compressed_weights
    report = dict(model=MODEL, method='measured_global_hessian_allocation',
                  formats=FORMATS, search_iters=SEARCH_ITERS,
                  final_iters=FINAL_ITERS, allocation=counts,
                  bf16=baseline, quantized=result,
                  choices={name: candidates[name][selected]
                           for name, selected in chosen.items()},
                  elapsed_seconds=time.time() - started)
    print('global_alloc_2bit', result, flush=True)
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
