"""NoFT E8P + full two-sided DCT/Hadamard mixing + BlockLDLQ.

This is an adaptation, NOT an exact QuIP# reproduction. The official QuIP#
Hadamard utility does not support Qwen2.5-0.5B's intermediate width (4864).
It uses the external official GPL-3.0 E8P quantizer, without CUDA extensions.
"""
import json, math, os, struct, subprocess, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from additive_vq import block_ldlq, hadamard_right
from aq_block_experiment import collect_H, evaluate


HEADER = struct.Struct('<IIQe')
MODEL_REVISION = '060db6499f32faf8b98477b0a26969ef7d8b9987'


def save_codes(path, indices, columns, seed, scale, forced):
    ids = indices.cpu().numpy().astype('<u2')
    if forced and np.any(ids[0, -forced:] != 0):
        raise ValueError('implicit indices must be code zero')
    first = ids[0, :ids.shape[1] - forced]
    body = np.concatenate((first, ids[1:].reshape(-1)))
    path.write_bytes(HEADER.pack(ids.shape[0], columns, seed, scale) + body.tobytes())


def load_codes(path, forced):
    data = path.read_bytes()
    rows, columns, seed, scale = HEADER.unpack(data[:HEADER.size])
    groups = columns // 8
    stored = np.frombuffer(data[HEADER.size:], dtype='<u2')
    ids = np.zeros((rows, groups), dtype=np.uint16)
    first_length = groups - forced
    ids[0, :first_length] = stored[:first_length]
    ids[1:] = stored[first_length:].reshape(rows - 1, groups)
    return torch.from_numpy(ids.astype(np.int64)), seed, scale


def decode_e8p(indices, packed_grid):
    """Decode the documented E8P 16-bit IDs from the stored absolute grid."""
    ids = indices.long()
    signs = ids & 255
    bits = torch.arange(8, device=ids.device)
    parity = ((signs[..., None] >> bits) & 1).sum(-1) % 2
    signs = signs ^ parity
    order = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=ids.device)
    packed = packed_grid.long()[ids >> 8]
    values = (((packed[..., None] >> (4 * order)) & 15) - 8).float() * .5
    negative = (signs[..., None] >> order) & 1
    values *= 1 - 2 * negative
    return values + (.25 - .5 * parity[..., None])


def reconstruct_layer(path, forced, packed_grid, device='cuda'):
    indices, seed, scale = load_codes(path, forced)
    rows, groups = indices.shape
    columns = groups * 8
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    su = torch.randint(0, 2, (columns,), device=device,
                       generator=generator).float() * 2 - 1
    sv = torch.randint(0, 2, (rows,), device=device,
                       generator=generator).float() * 2 - 1
    qr = decode_e8p(indices.to(device), packed_grid.to(device)).reshape(rows, columns)
    rec = mixed_right(qr * scale, su, inverse=True)
    return mixed_right(rec.t(), sv, inverse=True).t()


def refine_codes(w, h, q, indices, cb, forced, iters):
    """Discrete coordinate updates with original weights kept frozen."""
    groups = w.shape[1] // 8
    for _ in range(iters):
        error_h = (w - q) @ h
        for group in reversed(range(groups)):
            start, end = 8 * group, 8 * (group + 1)
            inverse = torch.linalg.inv(h[start:end, start:end])
            candidate = q[:, start:end] + error_h[:, start:end] @ inverse
            updated, ids = cb.quantize(candidate)
            if forced and group >= groups - forced:
                updated[0] = cb.grid[0]
                ids[0] = 0
            old = q[:, start:end]
            old_error = old - candidate
            new_error = updated - candidate
            metric = h[start:end, start:end]
            old_loss = ((old_error @ metric) * old_error).sum(1)
            new_loss = ((new_error @ metric) * new_error).sum(1)
            improve = new_loss <= old_loss
            updated = torch.where(improve[:, None], updated, old)
            ids = torch.where(improve, ids, indices[:, group])
            delta = updated - q[:, start:end]
            error_h -= delta @ h[start:end]
            q[:, start:end] = updated
            indices[:, group] = ids
    return q, indices


def mixed_right(x, signs, inverse=False):
    """Orthogonal DCT on the odd factor times Walsh-Hadamard on the 2-factor."""
    n = x.shape[-1]
    power = n & -n
    odd = n // power
    shape = x.shape
    work = x if inverse else x * signs
    work = work.reshape(-1, odd, power)
    positions = torch.arange(odd, device=x.device, dtype=x.dtype)
    frequencies = positions[:, None]
    dct = torch.cos(math.pi / odd * frequencies * (positions[None, :] + .5))
    dct *= math.sqrt(2 / odd)
    dct[0] /= math.sqrt(2)
    if inverse:
        work = torch.einsum('bkp,kl->blp', work, dct.t())
        work = hadamard_right(work, block_size=power)
    else:
        work = hadamard_right(work, block_size=power)
        work = torch.einsum('bkp,kl->blp', work, dct)
    work = work.reshape(shape)
    return work * signs if inverse else work


def official_codebook(root):
    source = root / 'lib/codebook/latticee8_padded12.py'
    text = source.read_text().split('class QuantizedE8P12Linear')[0]
    # Only the pure-PyTorch quantizer is needed. The removed imports are used by
    # inference methods below; no CUDA extension or vendored source is shipped.
    text = text.replace('import quiptools_cuda', '')
    text = text.replace('from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda', '')
    namespace = {'__name__': 'external_e8p_quantizer'}
    exec(compile(text, str(source), 'exec'), namespace)
    return namespace['E8P12_codebook']().cuda()


@torch.inference_mode()
def main():
    started = time.time()
    out = Path(os.environ.get('FLASHRUN_OUTPUT', 'lattice_results'))
    out.mkdir(exist_ok=True)
    root = Path(os.environ.get('FLASHRUN_QUIP_ROOT', '/workspace/quip-sharp'))
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16).cuda().eval()
    train = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    cal = '\n\n'.join(t for t in train['text'] if t.strip())
    text = '\n\n'.join(t for t in test['text'] if t.strip())
    ids = tokenizer(text, return_tensors='pt')['input_ids'][0][:4096].cuda()
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    bf16 = evaluate(model, ids)
    print('bf16', bf16, flush=True)
    cal_tokens = int(os.environ.get('FLASHRUN_CAL_TOKENS', '2048'))
    cal_length = int(os.environ.get('FLASHRUN_CAL_SEQUENCE_LENGTH', '128'))
    sequential = os.environ.get('FLASHRUN_SEQUENTIAL_CAL', '0') == '1'
    gpu_hessian = os.environ.get('FLASHRUN_GPU_HESSIAN', '0') == '1'
    scale_factor = float(os.environ.get('FLASHRUN_LATTICE_SCALE_FACTOR', '1'))
    cache = os.environ.get('FLASHRUN_HESSIAN_CACHE')
    if cache and sequential:
        raise ValueError('cached BF16 Hessians cannot be used with sequential calibration')
    if cache:
        expected = dict(model_revision=MODEL_REVISION, calibration_tokens=cal_tokens,
                        calibration_length=cal_length, gpu_hessian=gpu_hessian)
        if json.loads(Path(cache + '.json').read_text()) != expected:
            raise ValueError('Hessian cache does not match the requested calibration')
    hessians = (torch.load(cache, weights_only=True) if cache else
                {} if sequential else collect_H(
                    model, layers, tokenizer, cal, cal_tokens, cal_length,
                    gpu_hessian=gpu_hessian))
    print('building official E8P codebook', flush=True)
    cb = official_codebook(root)
    print('E8P ready', flush=True)
    total_weights = sum(m.weight.numel() for _, m in layers)
    strict = os.environ.get('FLASHRUN_LATTICE_STRICT', '0') == '1'
    tune_iters = int(os.environ.get('FLASHRUN_LATTICE_TUNE_ITERS', '0'))
    # Shared fixed 256x8 int4 absolute grid; it is algorithmic, counted once.
    payload_bits = 256 * 8 * 4
    implicit_total = math.ceil((payload_bits + len(layers) * HEADER.size * 8) / 16) if strict else 0
    implicit_base, implicit_remainder = divmod(implicit_total, len(layers))
    packed_grid_path = out / 'e8p_grid.bin'
    packed_grid_path.write_bytes(cb.grid_packed_abs.cpu().numpy().astype('<u4').tobytes())
    actual_bytes = packed_grid_path.stat().st_size
    for i, (name, module) in enumerate(layers):
        if sequential and name not in hessians:
            prefix = name.rsplit('.', 2)[0]
            block_layers = [(n, m) for n, m in layers
                            if n.rsplit('.', 2)[0] == prefix]
            hessians = collect_H(model, block_layers, tokenizer, cal,
                                 cal_tokens, cal_length,
                                 gpu_hessian=gpu_hessian, stop_after_last=True)
            print('calibrated', prefix, flush=True)
        w = module.weight.detach().float()
        rows, columns = w.shape
        generator = torch.Generator(device=w.device)
        generator.manual_seed(424242 + i)
        forced = implicit_base + (i < implicit_remainder)
        su = torch.randint(0, 2, (columns,), device=w.device,
                           generator=generator).float() * 2 - 1
        sv = torch.randint(0, 2, (rows,), device=w.device,
                           generator=generator).float() * 2 - 1
        h = hessians[name].cuda()
        h += .01 * h.diag().mean() * torch.eye(columns, device=w.device)
        hr = mixed_right(mixed_right(h, su).t(), su).t().contiguous()
        wr = mixed_right(mixed_right(w.t(), sv).t(), su)
        scale = (scale_factor * wr.square().mean().sqrt() / cb.opt_scale).half().float()
        calls = 0
        def quantize(candidate):
            nonlocal calls
            values, ids = cb.quantize(candidate)
            if calls < forced:
                values[0] = cb.grid[0]
                ids[0] = 0
            calls += 1
            return values, ids
        qr, indices = block_ldlq(wr / scale, hr, cb.grid,
                                 quantize_fn=quantize)
        qr, indices = refine_codes(wr / scale, hr, qr, indices, cb,
                                    forced, tune_iters)
        # Verify the selected IDs decode to the exact lattice vectors.
        decoded = cb.grid[indices].reshape_as(qr)
        torch.testing.assert_close(qr, decoded, atol=0, rtol=0)
        checkpoint = out / f'layer_{i:03d}.bin'
        save_codes(checkpoint, indices, columns, 424242 + i, float(scale), forced)
        restored_indices, restored_seed, restored_scale = load_codes(checkpoint, forced)
        torch.testing.assert_close(restored_indices.to(indices.device), indices, atol=0, rtol=0)
        assert restored_seed == 424242 + i and restored_scale == float(scale)
        actual_bytes += checkpoint.stat().st_size
        rec = mixed_right(qr * scale, su, inverse=True)
        rec = mixed_right(rec.t(), sv, inverse=True).t()
        module.weight.copy_(rec.to(module.weight.dtype))
        # 16 bits per 8-vector, a 64-bit sign seed, FP16 scalar and shape header.
        payload_bits += 2 * w.numel() + HEADER.size * 8 - 16 * forced
        if i % 20 == 0:
            print(f'encoded {i}/{len(layers)}', flush=True)
    result = evaluate(model, ids)
    result['bits_per_compressed_weight'] = payload_bits / total_weights
    result['serialized_bits_per_compressed_weight'] = 8 * actual_bytes / total_weights
    assert actual_bytes * 8 == payload_bits
    report = dict(method='E8P_two_sided_DCT_Hadamard_BlockLDLQ_NoFT',
                  exact_quip_reproduction=False, official_code_commit=commit,
                  model_revision=model.config._commit_hash,
                  torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                  strict_payload=strict, implicit_lattice_codes=implicit_total,
                  discrete_refinement_sweeps=tune_iters,
                  calibration_tokens=cal_tokens, calibration_length=cal_length,
                  hessian_cache=cache,
                  sequential_calibration=sequential, scale_factor=scale_factor,
                  bf16=bf16, quantized=result,
                  elapsed_seconds=time.time() - started)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('lattice', result, flush=True)


if __name__ == '__main__':
    main()
