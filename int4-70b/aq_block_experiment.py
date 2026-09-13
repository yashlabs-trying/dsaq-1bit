"""Full block-Hessian weighted additive VQ at 2-bit."""
import json, math, time
from pathlib import Path
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from additive_vq import additive_vq, decode, payload_bytes

torch.manual_seed(42)
OUT = Path('aqblock_results')
OUT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen2.5-0.5B'
M, K, G = 2, 256, 8
INIT_ITERS, ITERS, BEAM = 10, 20, 8
NSAMPLES = 2048
SEQLEN = 128

def collect_H(model, layers, tokenizer, text, nsamples, seqlen):
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
        model(ids[k * stride:k * stride + seqlen][None].cuda(), use_cache=False)
    for h in handles: h.remove()
    return {name: 2 * torch.cat(buffers[name], 0).t() @ torch.cat(buffers[name], 0) for name, _ in layers}

@torch.inference_mode()
def evaluate(model, ids):
    losses = []
    for window in ids.split(256):
        if window.numel() < 2: continue
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
    uncomp = sum(p.numel() * p.element_size() for p in model.parameters()) - compressed_params * 2
    report = dict(model=MODEL, gpu=torch.cuda.get_device_name(), torch=torch.__version__, M=M, K=K, g=G,
                  beam=BEAM, full_hessian=True, evaluation='First 4096 WikiText-2 test tokens', variants={})
    report['variants']['bf16'] = evaluate(model, ids)
    print('bf16', report['variants']['bf16'], flush=True)
    H = collect_H(model, names, tokenizer, cal_text, NSAMPLES, SEQLEN)
    payload = uncomp
    for i, (name, module) in enumerate(names):
        w = module.weight.detach().float().cuda()
        out_, in_ = w.shape
        h = H[name].cuda()
        damp = 0.01 * torch.mean(torch.diag(h))
        h = h + damp * torch.eye(in_, device=w.device)
        s = w.abs().amax(1).clamp_min(1e-6)
        w_norm = (w / s[:, None]).reshape(-1, G)
        N = w_norm.shape[0]
        nblocks_per_row = in_ // G
        block_pos = torch.arange(N, device=w.device) % nblocks_per_row
        H_blocks = h.view(nblocks_per_row, G, nblocks_per_row, G).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        codebooks, idx = additive_vq(w_norm, M, K, INIT_ITERS, ITERS, BEAM, H_blocks=H_blocks, block_pos=block_pos)
        rec = decode(codebooks, idx, (out_, -(-in_ // G) * G))[:, :in_] * s[:, None]
        module.weight.copy_(rec.to(module.weight.dtype))
        payload += payload_bytes(idx, codebooks)
        if i % 30 == 0: print(f'encoded {i}/{len(names)}', flush=True)
    result = evaluate(model, ids)
    result.update(tensor_payload_bytes=payload, effective_bits_per_parameter=8 * payload / total_params,
                  bits_per_compressed_weight=8 * (payload - uncomp) / compressed_params)
    report['variants']['aq_block_2bit'] = result
    print('aq_block_2bit', result, flush=True)
    report['elapsed_seconds'] = time.time() - start
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)

if __name__ == '__main__':
    main()