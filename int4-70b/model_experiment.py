"""Real-model quality probe. Dense reconstruction is NOT compressed inference."""
import json
import math
import platform
import time
from pathlib import Path
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

torch.manual_seed(42)
torch.set_num_threads(8)
OUT = Path('real_results')
OUT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen2.5-0.5B'
GROUP, RANK, FRACTION = 64, 2, .01


def packed_quant(w, bits):
    x = w.float().reshape(-1, GROUP)
    offset = x.amin(1).half()
    scale = ((x.amax(1) - offset.float()) / (2**bits - 1)).clamp_min(1e-8).half()
    scale = scale.clamp_min(torch.finfo(torch.float16).tiny)
    q = ((x - offset.float()[:, None]) / scale.float()[:, None]).round().clamp(0, 2**bits-1).byte()
    factor = 8 // bits
    groups = q.flatten().reshape(-1, factor)
    packed = torch.zeros(groups.shape[0], device=w.device, dtype=torch.uint8)
    for i in range(factor):
        packed |= groups[:, i] << (i * bits)
    data = dict(packed=packed.cpu(), scale=scale.cpu(), offset=offset.cpu())
    return data, unpack(data, bits, w.device)


def unpack(data, bits, device):
    p = data['packed'].to(device)
    q = torch.stack([(p >> (i*bits)) & (2**bits-1) for i in range(8//bits)], 1)
    return q.float().reshape(-1, GROUP) * data['scale'].to(device).float()[:, None] + data['offset'].to(device).float()[:, None]


def size(data):
    return sum(v.numel() * v.element_size() for v in data.values() if isinstance(v, torch.Tensor))


@torch.inference_mode()
def main():
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).cuda().eval()
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(t for t in dataset['text'] if t.strip())
    ids = tokenizer(text, return_tensors='pt')['input_ids'][0][:4096].cuda()
    # Fixed disjoint windows, identical for every variant. No use in codec fitting.
    def evaluate():
        losses = []
        for window in ids.split(256):
            if window.numel() < 2:
                continue
            loss = model(window[None], labels=window[None], use_cache=False).loss
            losses.append((float(loss), window.numel()-1))
        n = sum(n for _, n in losses)
        return dict(nll=sum(v*n for v,n in losses)/n,
                    perplexity=math.exp(sum(v*n for v,n in losses)/n), scored_tokens=n)
    names = [(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n != 'lm_head']
    report = dict(model=MODEL, revision=model.config._commit_hash, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                  dataset_fingerprint=dataset._fingerprint, group=GROUP, rank=RANK, sparse_fraction=FRACTION,
                  evaluation='First 4096 WikiText-2 test tokens, disjoint 256-token windows; exploratory, no task accuracy claim',
                  variants={}, layers=[])
    report['variants']['bf16'] = evaluate()
    print('BF16', report['variants']['bf16'], flush=True)
    total_params = sum(p.numel() for p in model.parameters())
    compressed_params = sum(m.weight.numel() for _,m in names)
    uncompressed_bytes = sum(p.numel()*p.element_size() for p in model.parameters()) - compressed_params*2
    # All residuals are relative to original BF16, not a 4-bit intermediate.
    payloads = {k:uncompressed_bytes for k in ['q2','q4','pca','hybrid','sparse_matched']}
    for i,(name,module) in enumerate(names):
        w = module.weight.detach().float()
        q2, base = packed_quant(w,2)
        q4, base4 = packed_quant(w,4)
        e = w.reshape(-1,GROUP) - base
        mean = e.mean(0).half()
        centered = e - mean.float()
        # Shared PCA across rows, covariance avoids full tall-matrix SVD.
        _, vectors = torch.linalg.eigh(centered.T @ centered)
        basis = vectors[:, -RANK:].T.contiguous().half()
        codes = (centered @ basis.float().T).half()
        common = codes.float() @ basis.float() + mean.float()
        residual = (e-common).flatten()
        count = int(e.numel()*FRACTION)
        index = torch.topk(residual.abs(),count, sorted=False).indices
        sparse = dict(index=index.int().cpu(), value=residual[index].half().cpu())
        pca = dict(mean=mean.cpu(),basis=basis.cpu(),codes=codes.cpu())
        correction_budget = size(pca)+size(sparse)
        count_matched = min(e.numel(), correction_budget//6)
        idx = torch.topk(e.flatten().abs(),count_matched,sorted=False).indices
        matched = dict(index=idx.int().cpu(),value=e.flatten()[idx].half().cpu())
        obj = dict(name=name,shape=list(w.shape),q2=q2,q4=q4,pca=pca,sparse=sparse,matched=matched)
        torch.save(obj,OUT/f'layer_{i:03d}.pt')
        correction = common.clone().flatten()
        correction[index] += sparse['value'].cuda().float()
        sparse_e = torch.zeros_like(e).flatten()
        sparse_e[idx] = matched['value'].cuda().float()
        errors = dict(q2=e,pca=e-common,hybrid=e-correction.reshape_as(e),sparse_matched=e-sparse_e.reshape_as(e),q4=w.reshape_as(e)-base4)
        report['layers'].append(dict(name=name,numel=w.numel(),mse={k:float(v.square().mean()) for k,v in errors.items()}))
        for k,extra in [('q2',0),('q4',0),('pca',size(pca)),('hybrid',correction_budget),('sparse_matched',size(matched))]:
            payloads[k] += size(q4 if k=='q4' else q2)+extra
        if i%14==0:
            print('Encoded',i+1,'/',len(names),flush=True)
    for variant in payloads:
        for i,(_,module) in enumerate(names):
            obj = torch.load(OUT/f'layer_{i:03d}.pt',weights_only=True)
            rec = unpack(obj['q4'] if variant=='q4' else obj['q2'],4 if variant=='q4' else 2,'cuda')
            if variant in ['pca','hybrid']:
                p=obj['pca']
                rec += p['codes'].cuda().float() @ p['basis'].cuda().float() + p['mean'].cuda().float()
            if variant in ['hybrid','sparse_matched']:
                s=obj['sparse'] if variant=='hybrid' else obj['matched']
                rec.flatten()[s['index'].cuda().long()] += s['value'].cuda().float()
            module.weight.copy_(rec.reshape(module.weight.shape).bfloat16())
        report['variants'][variant] = evaluate() | dict(tensor_payload_bytes=payloads[variant],effective_bits_per_parameter=8*payloads[variant]/total_params)
        (OUT/'report.json').write_text(json.dumps(report,indent=2))
        print(variant,report['variants'][variant],flush=True)
    report['elapsed_seconds']=time.time()-start
    report['limitations']=['Naive groupwise affine min-max baseline, not GPTQ/AQLM/SOTA.',
        'Dense BF16 reconstructed execution; NOT compressed inference or proof of 24GB fit for 70B.',
        'Payload counts tensor storage including unchanged parameters, excluding archive headers, runtime, and KV cache.',
        'Archives contain multiple experimental variants, not a deployable checkpoint.',
        'Short exploratory perplexity test only; no BF16 quality-equivalence claim.']
    (OUT/'report.json').write_text(json.dumps(report,indent=2))
    print('DONE',flush=True)


if __name__=='__main__':
    main()
