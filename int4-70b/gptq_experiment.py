"""Real-model strong-base experiment: GPTQ low-bit bases vs the naive baseline.

Replaces the naive min-max base with an activation-aware GPTQ base and reports
perplexity, payload, and how much rank-2 structure the residual now contains.
"""
import json
import math
import platform
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from gptq import gptq_quantize, decode, payload_bytes

torch.manual_seed(42)
torch.set_num_threads(8)
OUT = Path('gptq_results')
OUT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen2.5-0.5B'
GROUP = 64
NSAMPLES = 2048
SEQLEN = 128


def naive_groupwise(w, bits, group=GROUP):
    """The old naive min-max base, flattened into fixed-width groups."""
    x = w.float().reshape(-1, group)
    lo = x.amin(1)
    hi = x.amax(1)
    scale = ((hi - lo) / (2 ** bits - 1)).clamp_min(torch.finfo(torch.float16).tiny)
    q = (x - lo[:, None]).div(scale[:, None]).round().clamp(0, 2 ** bits - 1)
    return scale[:, None] * q + lo[:, None]


def collect_H(model, layers, tokenizer, text, nsamples=NSAMPLES, seqlen=SEQLEN):
    buffers = {name: [] for name, _ in layers}
    handles = []
    for name, mod in layers:
        def hook(mod, args, output, name=name):
            buffers[name].append(args[0].reshape(-1, args[0].shape[-1]).detach().float().cpu())
        handles.append(mod.register_forward_hook(hook))
    ids = tokenizer(text, return_tensors='pt')['input_ids'][0]
    total = ids.numel()
    nwin = min(nsamples // seqlen, total // seqlen)
    stride = (total - seqlen) // max(nwin, 1)
    for k in range(nwin):
        start = k * stride
        window = ids[start:start + seqlen][None].cuda()
        model(window, use_cache=False)
    for h in handles:
        h.remove()
    H = {}
    for name, _ in layers:
        xs = torch.cat(buffers[name], 0)[:nsamples]
        H[name] = 2 * xs.t() @ xs
    return H


def rank2_recovery(E):
    """Fraction of residual variance a shared rank-2 basis captures (64-wide blocks)."""
    flat = E.reshape(-1)
    pad = (-flat.numel()) % GROUP
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=E.device)])
    blocks = flat.reshape(-1, GROUP)
    centered = blocks - blocks.mean(0)
    total = (centered ** 2).sum()
    if total == 0:
        return 1.0
    _, _, vt = torch.linalg.svd(centered, full_matrices=False)
    approx = centered @ vt[:2].T @ vt[:2]
    return float(1 - ((centered - approx) ** 2).sum() / total)


@torch.inference_mode()
def evaluate(model, ids):
    losses = []
    for window in ids.split(256):
        if window.numel() < 2:
            continue
        loss = model(window[None], labels=window[None], use_cache=False).loss
        losses.append((float(loss), window.numel() - 1))
    n = sum(n for _, n in losses)
    nll = sum(v * n for v, n in losses) / n
    return dict(nll=nll, perplexity=math.exp(nll), scored_tokens=n)


@torch.inference_mode()
def main():
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    train = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    cal_text = '\n\n'.join(t for t in train['text'] if t.strip())
    eval_text = '\n\n'.join(t for t in test['text'] if t.strip())
    ids = tokenizer(eval_text, return_tensors='pt')['input_ids'][0][:4096].cuda()

    names = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    total_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(m.weight.numel() for _, m in names)
    uncompressed_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) - compressed_params * 2
    original = {name: m.weight.detach().float().clone() for name, m in names}

    report = dict(model=MODEL, revision=getattr(model.config, '_commit_hash', 'unknown'),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  transformers=transformers.__version__, python=platform.python_version(),
                  nsamples=NSAMPLES, seqlen=SEQLEN,
                  evaluation='First 4096 WikiText-2 test tokens, disjoint 256-token windows',
                  variants={})

    report['variants']['bf16'] = evaluate(model, ids)
    print('bf16', report['variants']['bf16'], flush=True)

    # Old naive 2-bit base, for the residual-structure comparison.
    naive_payload = uncompressed_bytes
    naive_r2_num = naive_r2_den = 0.0
    for name, module in names:
        w = original[name]
        rec = naive_groupwise(w, 2)
        naive_payload += (w.numel() * 2 + w.numel() // GROUP * 4) / 8
        resid = (w.reshape(-1, GROUP) - rec)
        den = (resid ** 2).sum().item()
        naive_r2_den += den
        naive_r2_num += den * rank2_recovery(resid)
    report['naive_2bit_rank2_recovery'] = naive_r2_num / naive_r2_den if naive_r2_den else 0.0
    print('naive_2bit', dict(payload_bytes=naive_payload, rank2_recovery=report['naive_2bit_rank2_recovery']), flush=True)

    H = collect_H(model, names, tokenizer, cal_text)
    configs = [
        ('gptq_4bit_pc', dict(bits=4, groupwise=False)),
        ('gptq_4bit_g128', dict(bits=4, groupwise=True, groupsize=128)),
        ('gptq_3bit_g128', dict(bits=3, groupwise=True, groupsize=128)),
        ('gptq_2bit_g128', dict(bits=2, groupwise=True, groupsize=128)),
        ('gptq_2bit_pc', dict(bits=2, groupwise=False)),
    ]
    for variant, cfg in configs:
        payload = uncompressed_bytes
        abs_resid = 0.0
        abs_w = 0.0
        r2_num = r2_den = 0.0
        for name, module in names:
            w = original[name]
            Q, codes, scale, zero = gptq_quantize(
                w, H[name], cfg['bits'], groupwise=cfg['groupwise'],
                groupsize=cfg.get('groupsize', -1))
            module.weight.copy_(Q.to(module.weight.dtype).view(module.weight.shape))
            payload += payload_bytes(codes, scale, zero, cfg['bits'])
            abs_resid += (w - Q).abs().sum().item()
            abs_w += w.abs().sum().item()
            if cfg['bits'] == 2:
                resid = w - Q
                den = (resid ** 2).sum().item()
                r2_den += den
                r2_num += den * rank2_recovery(resid)
        result = evaluate(model, ids)
        result.update(dict(
            tensor_payload_bytes=payload,
            effective_bits_per_parameter=8 * payload / total_params,
            mean_abs_residual_frac=abs_resid / abs_w,
        ))
        if cfg['bits'] == 2:
            result['rank2_recovery'] = r2_num / r2_den if r2_den else 0.0
        report['variants'][variant] = result
        for name, module in names:
            module.weight.copy_(original[name].to(module.weight.dtype).view(module.weight.shape))
        print(variant, result, flush=True)
        (OUT / 'report.json').write_text(json.dumps(report, indent=2))

    report['elapsed_seconds'] = time.time() - start
    report['limitations'] = [
        'Dense BF16 reconstructed execution; NOT compressed inference or proof of 24GB fit for 70B.',
        'Exploratory short perplexity test only; no BF16 quality-equivalence claim.',
        'Payload counts tensor storage only, excluding archive headers, runtime, and KV cache.',
        'GPTQ is activation-aware but not SOTA (no QuIP# rotation, lattice codebooks, or learned VQ).',
    ]
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
