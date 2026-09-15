"""Evaluate a fresh model reconstructed only from serialized lattice files."""
import json, math, os
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from aq_block_experiment import evaluate
from lattice_experiment import HEADER, reconstruct_layer


@torch.inference_mode()
def main():
    checkpoint = Path(os.environ['FLASHRUN_CHECKPOINT'])
    report = json.loads((checkpoint / 'report.json').read_text())
    revision = report['model_revision']
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', revision=revision, torch_dtype=torch.bfloat16).cuda().eval()
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(t for t in test['text'] if t.strip())
    ids = tokenizer(text, return_tensors='pt')['input_ids'][0][:8192].cuda()
    bf16 = {'first_4096': evaluate(model, ids[:4096]),
            'next_4096': evaluate(model, ids[4096:8192])}
    layers = [(n, m) for n, m in model.named_modules()
              if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    grid_file = checkpoint / 'e8p_grid.bin'
    packed_grid = torch.from_numpy(np.frombuffer(grid_file.read_bytes(), dtype='<u4').astype(np.int64)).cuda()
    implicit_total = math.ceil((grid_file.stat().st_size * 8 + len(layers) * HEADER.size * 8) / 16) if report['strict_payload'] else 0
    base, remainder = divmod(implicit_total, len(layers))
    payload_bytes = grid_file.stat().st_size
    for i, (name, module) in enumerate(layers):
        path = checkpoint / f'layer_{i:03d}.bin'
        rec = reconstruct_layer(path, base + (i < remainder), packed_grid)
        module.weight.copy_(rec.to(module.weight.dtype))
        payload_bytes += path.stat().st_size
    compressed_weights = sum(m.weight.numel() for _, m in layers)
    total_weights = sum(p.numel() for p in model.parameters())
    uncompressed_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) - 2 * compressed_weights
    result = dict(bf16=bf16, quantized={
        'first_4096': evaluate(model, ids[:4096]),
        'next_4096': evaluate(model, ids[4096:8192])},
        compressed_payload_bytes=payload_bytes,
        serialized_bpw=8 * payload_bytes / compressed_weights,
        whole_model_bpw=8 * (payload_bytes + uncompressed_bytes) / total_weights)
    print('fresh_serialized_eval', result, flush=True)
    (checkpoint / 'fresh_eval.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
