"""Controlled learned-trellis quality experiment, with exact two-bit files.

The maze resets every 8, 16 or 32 weights. This bounds search time but is not the
long-chain QTIP algorithm. Fixed and learned tables have identical byte budgets.
"""
import json
import os
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from aq_block_experiment import collect_H, evaluate
from additive_vq import block_ldlq
from lattice_experiment import MODEL_REVISION, mixed_right, official_codebook, refine_codes
from trellis_quant import (WIDTH, FORCED_BLOCKS, fixed_nodes, learn_nodes, train_path_nodes,
                           metadata_allocation, viterbi, decode_paths, force_mask, save_layer, load_layer)


def reconstruct(path, device='cuda'):
    ids, nodes, seed, scale = load_layer(path)
    rows, groups = ids.shape
    columns = groups * WIDTH
    generator = torch.Generator(device=device).manual_seed(seed)
    su = torch.randint(2, (columns,), device=device, generator=generator).float() * 2 - 1
    sv = torch.randint(2, (rows,), device=device, generator=generator).float() * 2 - 1
    q = decode_paths(ids.to(device), nodes.to(device)).reshape(rows, columns) * scale
    return mixed_right(mixed_right(q, su, inverse=True).t(), sv, inverse=True).t()


def trellis_encode(w, h, nodes, weighted=False, sweeps=1, rerank=False, forced=FORCED_BLOCKS):
    rows, columns = w.shape
    groups = columns // WIDTH
    calls = 0
    def quantize(candidate, metric):
        nonlocal calls
        group = groups - 1 - calls
        diagonal = metric.diag() if weighted else None
        q, ids = viterbi(candidate, nodes, diagonal, full_metric=metric if rerank else None)
        mask = force_mask(rows, groups, group, forced, w.device)
        q[mask] = decode_paths(torch.zeros(int(mask.sum()), device=w.device, dtype=torch.long), nodes)
        ids[mask] = 0
        calls += 1
        return q, ids
    q, ids = block_ldlq(w, h, torch.empty(1, WIDTH, device=w.device),
                       quantize_fn=quantize, callback_metric=True)
    for _ in range(sweeps):
        error_h = (w - q) @ h
        for group in reversed(range(groups)):
            start, end = group * WIDTH, (group + 1) * WIDTH
            metric = h[start:end, start:end]
            candidate = q[:, start:end] + error_h[:, start:end] @ torch.linalg.inv(metric)
            update, paths = viterbi(candidate, nodes, metric.diag() if weighted else None,
                                    full_metric=metric if rerank else None)
            mask = force_mask(rows, groups, group, forced, w.device)
            update[mask] = q[mask, start:end]
            paths[mask] = ids[mask, group]
            old_error = q[:, start:end] - candidate
            new_error = update - candidate
            improve = ((new_error @ metric) * new_error).sum(1) <= ((old_error @ metric) * old_error).sum(1)
            update = torch.where(improve[:, None], update, q[:, start:end])
            paths = torch.where(improve, paths, ids[:, group])
            delta = update - q[:, start:end]
            error_h -= delta @ h[start:end]
            q[:, start:end] = update
            ids[:, group] = paths
    return q, ids


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.time()
    out = Path(os.environ.get('FLASHRUN_OUTPUT', '/workspace/trellis_gate'))
    out.mkdir(parents=True, exist_ok=True)
    full = os.environ.get('FLASHRUN_TRELLIS_FULL', '0') == '1'
    variant = os.environ.get('FLASHRUN_TRELLIS_VARIANT', 'learned_weighted')
    if full and variant == 'path_trained_no_implicit':
        raise ValueError('no-implicit variant is a matrix diagnostic above the two-bit budget')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION,
                                               torch_dtype=torch.bfloat16).cuda().eval()
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    gate_names = ['model.layers.0.self_attn.q_proj', 'model.layers.0.mlp.down_proj',
                  'model.layers.12.self_attn.q_proj']
    selected = layers if full else [(n, m) for n, m in layers if n in gate_names]
    budget_mode = os.environ.get('FLASHRUN_TRELLIS_BUDGET', 'per_matrix')
    allocations = metadata_allocation([name for name, _ in layers], budget_mode)
    if any(allocations[n] > m.weight.numel() // WIDTH for n, m in layers):
        raise ValueError('a matrix is too small to pay the assigned metadata budget')
    train = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    cal = '\n\n'.join(t for t in train['text'] if t.strip())
    cache = Path('/workspace/hessians_full.pt' if full else '/workspace/hessians_gate.pt')
    cache_config = dict(model_revision=MODEL_REVISION, calibration_tokens=8192,
                        calibration_length=256, gpu_hessian=True)
    cache_metadata = Path(str(cache) + '.json')
    if cache.exists():
        if not cache_metadata.exists() or json.loads(cache_metadata.read_text()) != cache_config:
            raise ValueError('Hessian cache configuration is missing or inconsistent')
        hessians = torch.load(cache, weights_only=True)
    else:
        hessians = collect_H(model, selected, tokenizer, cal, 8192, 256,
                            gpu_hessian=True, stop_after_last=not full)
        torch.save(hessians, cache)
        cache_metadata.write_text(json.dumps(cache_config))
    print('Hessians ready', flush=True)
    cb = official_codebook(Path('/workspace/quip-sharp')) if not full else None
    report = dict(model_revision=MODEL_REVISION, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, calibration_tokens=8192, calibration_length=256,
                  states=256, vector_width=2, reset_width=WIDTH, index_bits_per_weight=2,
                  forced_blocks_per_matrix=FORCED_BLOCKS, full_model=full,
                  budget_mode=budget_mode,
                  original_model_finetuned=False, matrices=[])
    ids_eval = None
    if full:
        test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        text = '\n\n'.join(t for t in test['text'] if t.strip())
        ids_eval = tokenizer(text, return_tensors='pt')['input_ids'][0][:8192].cuda()
        report['bf16'] = dict(first_4096=evaluate(model, ids_eval[:4096]),
                              next_4096=evaluate(model, ids_eval[4096:]))
    total_bytes, total_weights = 0, 0
    for name, module in selected:
        index = next(i for i, (n, _) in enumerate(layers) if n == name)
        w = module.weight.detach().float()
        rows, columns = w.shape
        seed = 424242 + index
        generator = torch.Generator(device='cuda').manual_seed(seed)
        su = torch.randint(2, (columns,), device='cuda', generator=generator).float() * 2 - 1
        sv = torch.randint(2, (rows,), device='cuda', generator=generator).float() * 2 - 1
        h = hessians[name].cuda().clone()
        h += .01 * h.diag().mean() * torch.eye(columns, device='cuda')
        hr = mixed_right(mixed_right(h, su).t(), su).t().contiguous()
        wr = mixed_right(mixed_right(w.t(), sv).t(), su)
        scale = wr.square().mean().sqrt().half().float()
        normalized = wr / scale
        nodes_start = time.time()
        learned = learn_nodes(normalized, seed + 10000)
        print('learned nodes', name, round(time.time() - nodes_start, 2), flush=True)
        matrix = dict(name=name, shape=list(w.shape), variants={})
        denominator = float(((wr @ hr) * wr).sum())
        variants = [variant] if full else os.environ.get(
            'FLASHRUN_TRELLIS_GATE_VARIANTS',
            'e8p_current,fixed,learned,learned_weighted,path_trained,path_trained_zero').split(',')
        for method in variants:
            tick = time.time()
            if method == 'e8p_current':
                e8scale = (wr.square().mean().sqrt() / cb.opt_scale).half().float()
                forced = 12 + (index < 8)
                calls = 0
                def e8quant(candidate):
                    nonlocal calls
                    values, indices = cb.quantize(candidate)
                    if calls < forced:
                        values[0], indices[0] = cb.grid[0], 0
                    calls += 1
                    return values, indices
                q, paths = block_ldlq(wr / e8scale, hr, cb.grid, quantize_fn=e8quant)
                q, paths = refine_codes(wr / e8scale, hr, q, paths, cb, forced, 1)
                reconstructed = q * e8scale
                byte_count = w.numel() // 4 + 18 - 2 * forced
                # E8 shared grid accounted over the whole model, as in baseline.
            else:
                nodes = fixed_nodes('cuda') if method == 'fixed' else learned
                implicit = 0 if method == 'path_trained_no_implicit' else allocations[name]
                if method in ('path_trained', 'path_trained_zero', 'path_trained_rerank', 'path_trained_no_implicit'):
                    nodes, history = train_path_nodes(normalized, nodes, seed + 20000,
                                                      neutral_zero=method != 'path_trained')
                    matrix[method + '_training_mse'] = history
                q, paths = trellis_encode(normalized, hr, nodes,
                                         weighted=method in ('learned_weighted', 'path_trained', 'path_trained_zero', 'path_trained_rerank', 'path_trained_no_implicit'),
                                         rerank=method in ('path_trained_rerank', 'path_trained_no_implicit'), forced=implicit)
                path = out / f'{method}_layer_{index:03d}.bin'
                save_layer(path, paths, nodes, columns, seed, float(scale), forced=implicit)
                restored, restored_nodes, restored_seed, restored_scale = load_layer(path, forced=implicit)
                torch.testing.assert_close(restored.cuda(), paths, atol=0, rtol=0)
                torch.testing.assert_close(restored_nodes.cuda(), nodes, atol=0, rtol=0)
                decoded = decode_paths(restored.cuda(), restored_nodes.cuda()).reshape_as(q)
                torch.testing.assert_close(decoded, q, atol=0, rtol=0)
                reconstructed = decoded * restored_scale
                byte_count = path.stat().st_size
                if budget_mode == 'per_matrix' and implicit:
                    assert byte_count * 8 / w.numel() <= 2
                if full:
                    rec = reconstruct(path)
                    module.weight.copy_(rec.to(module.weight.dtype))
                    total_bytes += byte_count
                    total_weights += w.numel()
            error = wr - reconstructed
            measured = dict(relative_hessian_error=float(((error @ hr) * error).sum()) / denominator,
                            normalized_mse=float(error.square().mean() / wr.square().mean()),
                            encoding_seconds=time.time() - tick,
                            serialized_bytes=byte_count)
            if method != 'e8p_current':
                measured['serialized_bpw'] = byte_count * 8 / w.numel()
                measured['implicit_blocks'] = implicit
            matrix['variants'][method] = measured
            print(name, method, measured, flush=True)
        report['matrices'].append(matrix)
        (out / 'report.json').write_text(json.dumps(report, indent=2))
    if full:
        report['quantized'] = dict(first_4096=evaluate(model, ids_eval[:4096]),
                                    next_4096=evaluate(model, ids_eval[4096:]))
        report['serialized_bpw'] = total_bytes * 8 / total_weights
        assert report['serialized_bpw'] <= 2
        all_weights = sum(p.numel() for p in model.parameters())
        report['whole_model_bpw'] = (8 * total_bytes + 16 * (all_weights - total_weights)) / all_weights
        report['variant'] = variant
    report['elapsed_seconds'] = time.time() - started
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('FINAL', json.dumps({k: v for k, v in report.items() if k != 'matrices'}), flush=True)


if __name__ == '__main__':
    main()
