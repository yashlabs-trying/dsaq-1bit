"""2x2 experiment: refreshed input Hessians x output-aware node fitting.

All arms start from the same original BF16 weights and baseline FP16 tables.
Refreshed Hessians come from ONE fixed baseline quantized model, not different
models in each arm. Only node fitting and Hessian source vary. No model FT.
"""
import os
os.environ['FLASHRUN_TRELLIS_WIDTH'] = '32'
import json
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from lattice_experiment import MODEL_REVISION, mixed_right
from aq_block_experiment import collect_H, evaluate
from trellis_quant import load_layer, save_layer, decode_paths, metadata_allocation
from trellis_experiment import trellis_encode, reconstruct
from trellis_node_fit import fit_output_nodes


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    baseline = Path('/workspace/ablation_current')
    base_report = json.loads((baseline / 'report.json').read_text())
    assert base_report['reset_width'] == 32 and base_report['budget_mode'] == 'global_mlp'
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION,
                                               torch_dtype=torch.bfloat16).cuda().eval()
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    original = {name: module.weight.detach().cpu().clone() for name, module in layers}
    h_original = torch.load('/workspace/hessians_full.pt', weights_only=True)
    for i, (name, module) in enumerate(layers):
        module.weight.copy_(reconstruct(baseline / f'path_trained_rerank_layer_{i:03d}.bin').to(module.weight.dtype))
    train = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    cal = '\n\n'.join(t for t in train['text'] if t.strip())
    started = time.time()
    h_refreshed = collect_H(model, layers, tokenizer, cal, 8192, 256, gpu_hessian=True)
    torch.save(h_refreshed, '/workspace/hessians_refreshed.pt')
    refreshed_seconds = time.time() - started
    print('refreshed Hessians ready', refreshed_seconds, flush=True)
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(t for t in test['text'] if t.strip())
    ids_eval = tokenizer(text, return_tensors='pt')['input_ids'][0][:8192].cuda()
    allocations = metadata_allocation([name for name, _ in layers], 'global_mlp')
    aggregate = dict(current=base_report, controls=dict(
        calibration_tokens=8192, calibration_length=256, refreshed_hessian_reference='current',
        hadamard_signs='same stored seeds', normalization='same stored FP16 matrix scales',
        node_initialization='same stored current FP16 tables', original_model_finetuned=False,
        reset_width=32, states=256, endpoint_rerank_candidates=8, discrete_refinement_sweeps=1,
        node_solver_iterations=12, budget_mode='global_mlp'),
        refreshed_calibration_seconds=refreshed_seconds)
    for method, refreshed, fit_nodes in [('hessian_only', True, False),
                                         ('layer_output_only', False, True),
                                         ('both', True, True)]:
        started = time.time()
        out = Path('/workspace/ablation_' + method)
        out.mkdir(exist_ok=True)
        report = dict(model_revision=MODEL_REVISION, gpu=torch.cuda.get_device_name(),
                      torch=torch.__version__, calibration_tokens=8192, calibration_length=256,
                      reset_width=32, states=256, vector_width=2, full_model=True,
                      budget_mode='global_mlp', original_model_finetuned=False,
                      variant=method, bf16=base_report['bf16'],
                      refreshed_hessian=refreshed, output_node_fitting=fit_nodes, matrices=[])
        payload_bytes = 0
        for i, (name, module) in enumerate(layers):
            base_paths, nodes, seed, scale = load_layer(baseline / f'path_trained_rerank_layer_{i:03d}.bin')
            w = original[name].cuda().float()
            rows, columns = w.shape
            generator = torch.Generator(device='cuda').manual_seed(seed)
            su = torch.randint(2, (columns,), device='cuda', generator=generator).float() * 2 - 1
            sv = torch.randint(2, (rows,), device='cuda', generator=generator).float() * 2 - 1
            h = (h_refreshed if refreshed else h_original)[name].cuda().clone()
            h += .01 * h.diag().mean() * torch.eye(columns, device='cuda')
            hr = mixed_right(mixed_right(h, su).t(), su).t().contiguous()
            wr = mixed_right(mixed_right(w.t(), sv).t(), su)
            normalized = wr / scale
            nodes = nodes.cuda()
            node_fit = None
            if fit_nodes:
                nodes, node_fit = fit_output_nodes(normalized, hr, base_paths.cuda(), nodes,
                                                   width=32, iterations=12)
            tick = time.time()
            q, paths = trellis_encode(normalized, hr, nodes, weighted=True, rerank=True,
                                     sweeps=1, forced=allocations[name])
            path = out / f'{method}_layer_{i:03d}.bin'
            save_layer(path, paths, nodes, columns, seed, scale, forced=allocations[name])
            restored_paths, restored_nodes, restored_seed, restored_scale = load_layer(path)
            decoded = decode_paths(restored_paths.cuda(), restored_nodes.cuda()).reshape_as(q)
            torch.testing.assert_close(decoded, q, atol=0, rtol=0)
            module.weight.copy_(reconstruct(path).to(module.weight.dtype))
            error = normalized - q
            matrix = dict(name=name, shape=list(w.shape), node_fit=node_fit,
                          variants={method: dict(relative_hessian_error=float(((error @ hr) * error).sum()
                                                  / ((normalized @ hr) * normalized).sum()),
                              serialized_bytes=path.stat().st_size, implicit_blocks=allocations[name],
                              encoding_seconds=time.time() - tick)})
            report['matrices'].append(matrix)
            payload_bytes += path.stat().st_size
            if i % 20 == 0:
                print(method, i, 'node_fit', node_fit, flush=True)
            (out / 'report.json').write_text(json.dumps(report, indent=2))
        compressed = sum(m.weight.numel() for _, m in layers)
        all_weights = sum(p.numel() for p in model.parameters())
        report['serialized_bpw'] = 8 * payload_bytes / compressed
        assert report['serialized_bpw'] == 2
        report['whole_model_bpw'] = (8 * payload_bytes + 16 * (all_weights - compressed)) / all_weights
        report['quantized'] = dict(first_4096=evaluate(model, ids_eval[:4096]),
                                    next_4096=evaluate(model, ids_eval[4096:]))
        report['elapsed_seconds'] = time.time() - started
        (out / 'report.json').write_text(json.dumps(report, indent=2))
        aggregate[method] = report
        Path('/workspace/ablation_report.json').write_text(json.dumps(aggregate, indent=2))
        print('RESULT', method, json.dumps(report['quantized']), report['elapsed_seconds'], flush=True)
    print('ALL FOUR ARMS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
