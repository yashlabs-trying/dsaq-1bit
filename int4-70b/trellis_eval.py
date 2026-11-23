"""Fresh-model evaluation using only serialized trellis layer files."""
import json
import os
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.inference_mode()
def main():
    checkpoint = Path(os.environ['FLASHRUN_CHECKPOINT'])
    report = json.loads((checkpoint / 'report.json').read_text())
    if not report['full_model']:
        raise ValueError('evaluation requires a complete transformer checkpoint')
    os.environ['FLASHRUN_TRELLIS_WIDTH'] = str(report['reset_width'])
    # Import after selecting the explicitly recorded codec format.
    from trellis_experiment import reconstruct
    from aq_block_experiment import evaluate
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', revision=report['model_revision'])
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', revision=report['model_revision'],
                                               torch_dtype=torch.bfloat16).cuda().eval()
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != 'lm_head']
    payload_bytes = 0
    for i, (name, module) in enumerate(layers):
        assert report['matrices'][i]['name'] == name
        path = checkpoint / f"{report['variant']}_layer_{i:03d}.bin"
        module.weight.copy_(reconstruct(path).to(module.weight.dtype))
        payload_bytes += path.stat().st_size
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    text = '\n\n'.join(t for t in test['text'] if t.strip())
    ids = tokenizer(text, return_tensors='pt')['input_ids'][0][:8192].cuda()
    result = dict(first_4096=evaluate(model, ids[:4096]), next_4096=evaluate(model, ids[4096:]))
    assert result == report['quantized'], 'fresh serialized perplexity does not reproduce'
    compressed_weights = sum(m.weight.numel() for _, m in layers)
    all_weights = sum(p.numel() for p in model.parameters())
    fresh = dict(quantized=result, compressed_payload_bytes=payload_bytes,
                 serialized_bpw=8 * payload_bytes / compressed_weights,
                 whole_model_bpw=(8 * payload_bytes + 16 * (all_weights - compressed_weights)) / all_weights,
                 exact_perplexity_reproduction=True)
    (checkpoint / 'fresh_eval.json').write_text(json.dumps(fresh, indent=2))
    print('fresh_serialized_eval', json.dumps(fresh), flush=True)


if __name__ == '__main__':
    main()
