"""Full block-Hessian weighted additive VQ at 2-bit."""
import json, math, os, time
from pathlib import Path
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from additive_vq import (additive_vq, block_ldlq, decode, hadamard_right,
                         payload_bytes, transform_hessian)

torch.manual_seed(42)
OUT = Path(os.environ.get('FLASHRUN_OUTPUT', 'aqblock_results'))
OUT.mkdir(exist_ok=True)
MODEL = 'Qwen/Qwen2.5-0.5B'
M = int(os.environ.get('FLASHRUN_CODEBOOKS', '2'))
K = int(os.environ.get('FLASHRUN_CODEBOOK_SIZE', '256'))
G = int(os.environ.get('FLASHRUN_GROUP_SIZE', '8'))
INIT_ITERS = int(os.environ.get('FLASHRUN_INIT_ITERS', '8'))
ITERS = int(os.environ.get('FLASHRUN_ITERS', '10'))
BEAM = int(os.environ.get('FLASHRUN_BEAM', '1'))
NSAMPLES = int(os.environ.get('FLASHRUN_CAL_TOKENS', '2048'))
SEQLEN = int(os.environ.get('FLASHRUN_CAL_SEQUENCE_LENGTH', '128'))
TARGET_BPW = float(os.environ.get('FLASHRUN_TARGET_BPW', '0'))
OPTIMIZE_ROW_SCALE = os.environ.get('FLASHRUN_OPTIMIZE_ROW_SCALE', '0') == '1'
OPTIMIZE_ROW_AFFINE = os.environ.get('FLASHRUN_OPTIMIZE_ROW_AFFINE', '0') == '1'
SCALE_WEIGHT_CODEBOOK = os.environ.get('FLASHRUN_SCALE_WEIGHT_CODEBOOK', '0') == '1'
RESIDUAL_RANK = int(os.environ.get('FLASHRUN_RESIDUAL_RANK', '0'))
PROMOTE_3BIT_SUFFIXES = tuple(filter(None, os.environ.get(
    'FLASHRUN_PROMOTE_3BIT_SUFFIXES', '').split(',')))
STRICT_MLP_MIXED = os.environ.get('FLASHRUN_STRICT_MLP_MIXED', '0') == '1'
USE_BLOCK_LDLQ = os.environ.get('FLASHRUN_BLOCK_LDLQ', '0') == '1'
USE_RHT = os.environ.get('FLASHRUN_RHT', '0') == '1'
RHT_BLOCK_SIZE = int(os.environ.get('FLASHRUN_RHT_BLOCK_SIZE', '128'))
SCALE_GROUP = int(os.environ.get('FLASHRUN_SCALE_GROUP', '0'))
SCALE_BITS = int(os.environ.get('FLASHRUN_SCALE_BITS', '16'))
ENTROPY_INDICES = os.environ.get('FLASHRUN_ENTROPY_INDICES', '0') == '1'
HESSIAN_INIT = os.environ.get('FLASHRUN_HESSIAN_INIT', '0') == '1'
INIT_RESTARTS = int(os.environ.get('FLASHRUN_INIT_RESTARTS', '1'))
SEQUENTIAL_CAL = os.environ.get('FLASHRUN_SEQUENTIAL_CAL', '0') == '1'
GPU_HESSIAN = os.environ.get('FLASHRUN_GPU_HESSIAN', '0') == '1'


def encoded_payload_bytes(indices, codebooks):
    if not ENTROPY_INDICES:
        return payload_bytes(indices, codebooks)
    total_bits = codebooks.numel() * 16
    for column in range(indices.shape[1]):
        counts = torch.bincount(
            indices[:, column], minlength=codebooks.shape[1]).double()
        probabilities = counts[counts > 0] / counts.sum()
        entropy = float(-(probabilities * probabilities.log2()).sum())
        # Ideal arithmetic stream plus a uint32 frequency table and termination.
        total_bits += math.ceil(indices.shape[0] * entropy) + 32 * counts.numel() + 32
    return total_bits / 8

class _CalibrationComplete(Exception):
    pass


def collect_H(model, layers, tokenizer, text, nsamples, seqlen,
              gpu_hessian=False, stop_after_last=False):
    buffers = {name: [] for name, _ in layers}
    handles = []
    for name, mod in layers:
        def hook(mod, args, output, name=name):
            buffers[name].append(args[0].reshape(-1, args[0].shape[-1]).detach().float().cpu())
            if stop_after_last and name == layers[-1][0]:
                raise _CalibrationComplete()
        handles.append(mod.register_forward_hook(hook))
    ids = tokenizer(text, return_tensors='pt')['input_ids'][0]
    total = ids.numel()
    nwin = min(nsamples // seqlen, total // seqlen)
    stride = (total - seqlen) // max(nwin, 1)
    try:
        for k in range(nwin):
            try:
                model(ids[k * stride:k * stride + seqlen][None].cuda(), use_cache=False)
            except _CalibrationComplete:
                pass
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for name, _ in layers:
        activations = torch.cat(buffers[name], 0)
        if gpu_hessian:
            activations = activations.cuda()
        result[name] = (2 * activations.t() @ activations).cpu()
    return result

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
    report = dict(model=MODEL, model_revision=model.config._commit_hash,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, M=M, K=K, g=G,
                  beam=BEAM, full_hessian=True, calibration_tokens=NSAMPLES,
                  calibration_sequence_length=SEQLEN,
                  nominal_index_bits_per_weight=M * math.log2(K) / G,
                  sparse_target_bits_per_weight=TARGET_BPW or None,
                  optimize_row_scale=OPTIMIZE_ROW_SCALE,
                  optimize_row_affine=OPTIMIZE_ROW_AFFINE,
                  scale_weight_codebook=SCALE_WEIGHT_CODEBOOK,
                  residual_rank=RESIDUAL_RANK,
                  promote_3bit_suffixes=PROMOTE_3BIT_SUFFIXES,
                  strict_mlp_mixed=STRICT_MLP_MIXED,
                  block_ldlq=USE_BLOCK_LDLQ,
                  randomized_hadamard=USE_RHT,
                  rht_block_size=RHT_BLOCK_SIZE if USE_RHT else None,
                  scale_group=SCALE_GROUP or None,
                  scale_bits=SCALE_BITS if SCALE_GROUP else 16,
                  entropy_coded_indices=ENTROPY_INDICES,
                  hessian_initialization=HESSIAN_INIT,
                  initialization_restarts=INIT_RESTARTS,
                  sequential_block_calibration=SEQUENTIAL_CAL,
                  gpu_hessian_products=GPU_HESSIAN,
                  evaluation='First 4096 WikiText-2 test tokens', variants={})
    report['variants']['bf16'] = evaluate(model, ids)
    print('bf16', report['variants']['bf16'], flush=True)
    H = {} if SEQUENTIAL_CAL else collect_H(
        model, names, tokenizer, cal_text, NSAMPLES, SEQLEN,
        gpu_hessian=GPU_HESSIAN)
    payload = uncomp
    sparse_corrections = 0
    for i, (name, module) in enumerate(names):
        if SEQUENTIAL_CAL and name not in H:
            # Earlier blocks are already quantized, so these Hessians describe
            # the actual deployment inputs rather than only BF16 inputs.
            prefix = name.rsplit('.', 2)[0]
            block_layers = [(n, m) for n, m in names
                            if n.rsplit('.', 2)[0] == prefix]
            H = collect_H(model, block_layers, tokenizer, cal_text,
                          NSAMPLES, SEQLEN, gpu_hessian=GPU_HESSIAN,
                          stop_after_last=True)
            print('calibrated', prefix, flush=True)
        w = module.weight.detach().float().cuda()
        out_, in_ = w.shape
        layer_m, layer_k, layer_g, layer_beam = M, K, G, BEAM
        if STRICT_MLP_MIXED:
            if name.endswith('mlp.down_proj'):
                # 5-bit index for each pair of weights = 2.5 bpw.
                layer_m, layer_k, layer_g, layer_beam = 1, 32, 2, 1
            else:
                # 7-bit index for four weights = 1.75 bpw. The saved
                # quarter-bit pays for the promoted down projection and metadata.
                layer_m, layer_k, layer_g, layer_beam = 1, 128, 4, 1
        elif PROMOTE_3BIT_SUFFIXES and name.endswith(PROMOTE_3BIT_SUFFIXES):
            layer_m, layer_k, layer_g, layer_beam = 1, 64, 2, 1
        h = H[name].cuda()
        damp = 0.01 * torch.mean(torch.diag(h))
        h = h + damp * torch.eye(in_, device=w.device)
        if SCALE_GROUP:
            if in_ % SCALE_GROUP:
                raise ValueError('input width must be divisible by scale group')
            scale_shape = (out_, in_ // SCALE_GROUP)
            s = w.reshape(*scale_shape, SCALE_GROUP).abs().amax(2).clamp_min(1e-6)
            if SCALE_BITS == 8:
                s = s.to(torch.float8_e4m3fn).float()
            elif SCALE_BITS == 16:
                s = s.half().float()
            else:
                raise ValueError('scale bits must be 8 or 16')
            expanded_scale = s.repeat_interleave(SCALE_GROUP, dim=1)
            normalized_matrix = w / expanded_scale
        else:
            s = w.abs().amax(1).clamp_min(1e-6)
            expanded_scale = s[:, None]
            normalized_matrix = w / expanded_scale
        rht_signs = None
        quant_h = h
        if USE_RHT:
            generator = torch.Generator(device=w.device)
            generator.manual_seed(424242 + i)
            rht_signs = torch.randint(
                0, 2, (in_,), generator=generator, device=w.device,
                dtype=torch.int8).float().mul_(2).sub_(1)
            normalized_matrix = hadamard_right(
                normalized_matrix, rht_signs, RHT_BLOCK_SIZE)
            quant_h = transform_hessian(h, rht_signs, RHT_BLOCK_SIZE)
        w_norm = normalized_matrix.reshape(-1, layer_g)
        N = w_norm.shape[0]
        nblocks_per_row = in_ // layer_g
        block_pos = torch.arange(N, device=w.device) % nblocks_per_row
        sample_weight = None
        if SCALE_WEIGHT_CODEBOOK:
            sample_weight = s.square().repeat_interleave(nblocks_per_row)
        H_blocks = quant_h.view(
            nblocks_per_row, layer_g, nblocks_per_row, layer_g
        ).diagonal(dim1=0, dim2=2).permute(2, 0, 1)
        codebooks, idx = additive_vq(
            w_norm, layer_m, layer_k, INIT_ITERS, ITERS, layer_beam,
            H_blocks=H_blocks, block_pos=block_pos,
            sample_weight=sample_weight, hessian_init=HESSIAN_INIT,
            init_restarts=INIT_RESTARTS)
        if USE_BLOCK_LDLQ:
            if layer_m != 1:
                raise ValueError('BlockLDLQ currently requires one codebook')
            normalized_rec, matrix_idx = block_ldlq(
                normalized_matrix, quant_h, codebooks[0])
            idx = matrix_idx.reshape(-1, 1)
        codebooks = codebooks.half()
        if USE_BLOCK_LDLQ:
            normalized_rec = decode(
                codebooks.float(), idx, (out_, in_))
        else:
            normalized_rec = decode(
                codebooks.float(), idx,
                (out_, -(-in_ // layer_g) * layer_g))[:, :in_]
        if USE_RHT:
            normalized_rec = hadamard_right(
                normalized_rec, rht_signs, RHT_BLOCK_SIZE, inverse=True)
        row_offset = None
        if SCALE_GROUP:
            # Refit every local scale using diagonal activation sensitivity,
            # then round it to its actual stored precision before evaluation.
            diagonal = h.diag().reshape(1, *scale_shape[1:], SCALE_GROUP)
            q = normalized_rec.reshape(*scale_shape, SCALE_GROUP)
            target = w.reshape(*scale_shape, SCALE_GROUP)
            numerator = (q * target * diagonal).sum(2)
            denominator = (q.square() * diagonal).sum(2).clamp_min(1e-12)
            s = numerator / denominator
            if SCALE_BITS == 8:
                s = s.to(torch.float8_e4m3fn).float()
            else:
                s = s.half().float()
            expanded_scale = s.repeat_interleave(SCALE_GROUP, dim=1)
        elif OPTIMIZE_ROW_AFFINE:
            rec_h = normalized_rec @ h
            ones_h = h.sum(0)
            a11 = (rec_h * normalized_rec).sum(1)
            a12 = rec_h.sum(1)
            a22 = h.sum()
            b1 = (rec_h * w).sum(1)
            b2 = (w * ones_h[None, :]).sum(1)
            determinant = (a11 * a22 - a12.square()).clamp_min(1e-12)
            s = (b1 * a22 - b2 * a12) / determinant
            row_offset = (a11 * b2 - a12 * b1) / determinant
        elif OPTIMIZE_ROW_SCALE:
            rec_h = normalized_rec @ h
            numerator = (rec_h * w).sum(1)
            denominator = (rec_h * normalized_rec).sum(1).clamp_min(1e-12)
            s = (numerator / denominator).clamp_min(1e-8)
        if not SCALE_GROUP:
            s = s.half().float()
            expanded_scale = s[:, None]
        if row_offset is not None:
            row_offset = row_offset.half().float()
        rec = normalized_rec * expanded_scale
        if row_offset is not None:
            rec += row_offset[:, None]
        # FP16 scale is required by every representation; affine rows also store
        # one FP16 offset. Previous reports omitted this small but real metadata.
        if SCALE_GROUP:
            layer_payload = encoded_payload_bytes(idx, codebooks) + SCALE_BITS * s.numel() / 8
        else:
            layer_payload = encoded_payload_bytes(idx, codebooks) + 2 * out_
        if row_offset is not None:
            layer_payload += 2 * out_
        if USE_RHT:
            # A 64-bit seed regenerates every sign; the transform itself has no
            # per-weight payload and is fused with the activation kernel.
            layer_payload += 8
        if RESIDUAL_RANK:
            diagonal_scale = h.diag().clamp_min(1e-12).sqrt()
            for rank_index in range(RESIDUAL_RANK):
                weighted_residual = (w - rec) * diagonal_scale[None, :]
                generator = torch.Generator(device=w.device)
                generator.manual_seed(1000 + i * 31 + rank_index)
                right = torch.randn(in_, generator=generator, device=w.device)
                right /= right.norm().clamp_min(1e-12)
                for _ in range(6):
                    left = weighted_residual @ right
                    left /= left.norm().clamp_min(1e-12)
                    right = weighted_residual.t() @ left
                    right /= right.norm().clamp_min(1e-12)
                left = weighted_residual @ right
                left = left.half().float()
                right_unscaled = (right / diagonal_scale).half().float()
                rec += left[:, None] * right_unscaled[None, :]
                layer_payload += 2 * (out_ + in_)
        if TARGET_BPW:
            index_bits = max(1, math.ceil(math.log2(w.numel())))
            correction_bits = index_bits + 16
            remaining_bits = max(0, int(TARGET_BPW * w.numel() - 8 * layer_payload))
            keep = min(w.numel(), remaining_bits // correction_bits)
            if keep:
                residual = w - rec
                importance = residual.square() * h.diag()[None, :]
                selected = importance.flatten().topk(keep, sorted=False).indices
                rec_flat = rec.flatten()
                correction = (w.flatten()[selected] - rec_flat[selected]).half().float()
                rec_flat[selected] += correction
                layer_payload += keep * correction_bits / 8
                sparse_corrections += keep
        module.weight.copy_(rec.to(module.weight.dtype))
        payload += layer_payload
        if i % 30 == 0: print(f'encoded {i}/{len(names)}', flush=True)
    result = evaluate(model, ids)
    result.update(tensor_payload_bytes=payload, effective_bits_per_parameter=8 * payload / total_params,
                  bits_per_compressed_weight=8 * (payload - uncomp) / compressed_params,
                  sparse_corrections=sparse_corrections)
    report['variants']['aq_block_2bit'] = result
    print('aq_block_2bit', result, flush=True)
    report['elapsed_seconds'] = time.time() - start
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print('DONE', flush=True)

if __name__ == '__main__':
    main()
