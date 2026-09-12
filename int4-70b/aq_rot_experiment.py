"""Rotation (QuIP#) + Hessian-weighted additive VQ at 2-bit."""
import json
import math
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from additive_vq import additive_vq, decode, payload_bytes

torch.manual_seed(42)
OUT = Path('aqrot_results')
OUT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen2.5-0.5B'
M, K, G = 2, 256, 8
INIT_ITERS, ITERS, BEAM = 10, 20, 8
NSAMPLES = 2048
SEQLEN = 128


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
        model(ids[start:start + seqlen][None].cuda(), use_cache=False)
    for h in handles:
        h.remove()
    H = {}
    for name, _ in layers:
        xs = torch.cat(buffers[name], 0)[:nsamples]
        H[name] = 2 * xs.t() @ xs
    return H


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

    report = dict(model=MODEL, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  M=M, K=K, g=G, nsamples=NSAMPLES, rotation=True, hessian=True,
                  evaluation='First 4096 WikiText-2 test tokens', variants={})
    report['variants']['bf16'] = evaluate(model, ids)
    print('bf16', report['variants']['bf16'], flush=True)

    H = collect_H(model, names, tokenizer, cal_text)

    payload = uncompressed_bytes
    for i, (name, module) in enumerate(names):
        w = module.weight.detach().float().cuda()
        out_, in_ = w.shape
        R, _ = torch.linalg.qr(torch.randn(in_, in_, device=w.device))
        HR = H[name].cuda() @ R
        d = (R * HR).sum(0).clamp_min(1e-6)
        s = w.abs().amax(1).clamp_min(1e-6)
        ww = (w / s[:, None]) @ R
        ww = ww * d.sqrt()[None, :]
        pad = (-in_) % G
        if pad:
            ww = torch.cat([ww, torch.zeros(out_, pad, device=ww.device)], 1)
        blocks = ww.reshape(-1, G)
        codebooks, idx = additive_vq(blocks, M, K, INIT_ITERS, ITERS, BEAM)
        rec = decode(codebooks, idx, ww.shape)[:, :in_]
        rec = rec / d.sqrt()[None, :]
        rec = rec @ R.t()
        rec = rec * s[:, None]
        module.weight.copy_(rec.to(module.weight.dtype))
        payload += payload_bytes(idx, codebooks)
        del R, HR, ww, blocks, codebooks, idx, rec
        if i % 30 == 0:
            print(f'encoded {i}/{len(names)}', flush=True)
    result = evaluate(model, ids)
    result.update(dict(tensor_payload_bytes=payload,
                       effective_bits_per_parameter=8 * payload / total_params,
                       bits_per_compressed_weight=8 * (payload - uncompressed_bytes) / compressed_params))
    report['variants']['aq_rot_2bit'] = result
    print('aq_rot_2bit', result, flush=True)
    report['elapsed_seconds'] = time.time() - start
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
