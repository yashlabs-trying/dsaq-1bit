"""Real-model additive-VQ (2-bit) experiment."""
import json
import math
import platform
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from additive_vq import additive_vq, decode, payload_bytes

torch.manual_seed(42)
OUT = Path('aq_results')
OUT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen2.5-0.5B'
M, K, G = 2, 256, 8
INIT_ITERS, ITERS = 10, 20


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
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    eval_text = '\n\n'.join(t for t in test['text'] if t.strip())
    ids = tokenizer(eval_text, return_tensors='pt')['input_ids'][0][:4096].cuda()

    names = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    total_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(m.weight.numel() for _, m in names)
    uncompressed_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) - compressed_params * 2

    report = dict(model=MODEL, revision=getattr(model.config, '_commit_hash', 'unknown'),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  transformers=transformers.__version__, M=M, K=K, g=G,
                  evaluation='First 4096 WikiText-2 test tokens', variants={})
    report['variants']['bf16'] = evaluate(model, ids)
    print('bf16', report['variants']['bf16'], flush=True)

    payload = uncompressed_bytes
    abs_resid = 0.0
    abs_w = 0.0
    for i, (name, module) in enumerate(names):
        w = module.weight.detach().float().cuda()
        out_, in_ = w.shape
        pad = (-in_) % G
        if pad:
            w = torch.cat([w, torch.zeros(out_, pad, device=w.device)], 1)
        blocks = w.reshape(-1, G)
        codebooks, idx = additive_vq(blocks, M, K, INIT_ITERS, ITERS)
        rec = decode(codebooks, idx, w.shape)[:, :in_]
        module.weight.copy_(rec.to(module.weight.dtype))
        payload += payload_bytes(idx, codebooks)
        abs_resid += (w[:, :in_] - rec).abs().sum().item()
        abs_w += w[:, :in_].abs().sum().item()
        if i % 30 == 0:
            print(f'encoded {i}/{len(names)} {name}', flush=True)
    result = evaluate(model, ids)
    result.update(dict(
        tensor_payload_bytes=payload,
        effective_bits_per_parameter=8 * payload / total_params,
        mean_abs_residual_frac=abs_resid / abs_w,
        bits_per_compressed_weight=8 * (payload - uncompressed_bytes) / compressed_params,
    ))
    report['variants']['aq_2bit'] = result
    print('aq_2bit', result, flush=True)
    report['elapsed_seconds'] = time.time() - start
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
