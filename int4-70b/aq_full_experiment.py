"""Full 2-bit additive VQ: rotation (QuIP#) + full Hessian whitening + beam search."""
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
OUT = Path('aqfull_results')
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
                  M=M, K=K, g=G, nsamples=NSAMPLES, rotation=True, full_hessian=True,
                  beam=BEAM, evaluation='First 4096 WikiText-2 test tokens', variants={})
    report['variants']['bf16'] = evaluate(model, ids)
    print('bf16', report['variants']['bf16'], flush=True)

    H = collect_H(model, names, tokenizer, cal_text)

    payload = uncompressed_bytes
    for i, (name, module) in enumerate(names):
        w = module.weight.detach().float().cuda()
        out_, in_ = w.shape
        dev = w.device
        h = H[name].cuda()
        damp = 0.01 * torch.mean(torch.diag(h))
        hd = h + damp * torch.eye(in_, device=dev)
        R, _ = torch.linalg.qr(torch.randn(in_, in_, device=dev))
        Hrot = R.t() @ hd @ R
        T2 = torch.linalg.cholesky(Hrot)
        T = R @ T2
        T2_inv = torch.linalg.inv(T2)
        T_inv = T2_inv @ R.t()
        s = w.abs().amax(1).clamp_min(1e-6)
        wt = (w / s[:, None]) @ T
        pad = (-in_) % G
        if pad:
            wt = torch.cat([wt, torch.zeros(out_, pad, device=dev)], 1)
        blocks = wt.reshape(-1, G)
        codebooks, idx = additive_vq(blocks, M, K, INIT_ITERS, ITERS, BEAM)
        rec_t = decode(codebooks, idx, wt.shape)[:, :in_]
        rec = (rec_t @ T_inv) * s[:, None]
        module.weight.copy_(rec.to(module.weight.dtype))
        payload += payload_bytes(idx, codebooks)
        del h, hd, R, Hrot, T, T2, T2_inv, T_inv, wt, blocks, codebooks, idx, rec_t, rec
        if i % 30 == 0:
            print(f'encoded {i}/{len(names)}', flush=True)
    result = evaluate(model, ids)
    result.update(dict(tensor_payload_bytes=payload,
                       effective_bits_per_parameter=8 * payload / total_params,
                       bits_per_compressed_weight=8 * (payload - uncompressed_bytes) / compressed_params))
    report['variants']['aq_full_2bit'] = result
    print('aq_full_2bit', result, flush=True)
    report['elapsed_seconds'] = time.time() - start
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
