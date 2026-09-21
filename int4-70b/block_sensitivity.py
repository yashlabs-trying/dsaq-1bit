"""Oracle map of where a serialized trellis checkpoint loses model quality.

This is a diagnostic, not a deployable mixed-precision result: selected linear
groups are temporarily restored to BF16 and every extra bit is reported.
Ranking uses WikiText-2 validation; test slices are never used for selection.
"""
import json
import os
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from aq_block_experiment import evaluate
from lattice_experiment import MODEL_REVISION


def score(model, ids):
    return evaluate(model, ids[:4096])


def layer_number(name):
    match = re.match(r'model\.layers\.(\d+)\.', name)
    if not match:
        raise ValueError('unexpected transformer linear name: ' + name)
    return int(match.group(1))


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    checkpoint = Path(os.environ.get('FLASHRUN_CHECKPOINT', '/workspace/ablation_both'))
    out = Path(os.environ.get('FLASHRUN_SENSITIVITY_OUTPUT', '/workspace/block_sensitivity.json'))
    source = json.loads((checkpoint / 'report.json').read_text())
    os.environ['FLASHRUN_TRELLIS_WIDTH'] = str(source['reset_width'])
    from trellis_experiment import reconstruct

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16).cuda().eval()
    linears = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name != 'lm_head']
    original = {name: module.weight.detach().cpu().clone() for name, module in linears}
    quantized = {}
    payload = 0
    for index, (name, module) in enumerate(linears):
        if source['matrices'][index]['name'] != name:
            raise ValueError('checkpoint matrix order mismatch')
        path = checkpoint / f"{source['variant']}_layer_{index:03d}.bin"
        restored = reconstruct(path).to(torch.bfloat16).cpu()
        quantized[name] = restored
        module.weight.copy_(restored.cuda())
        payload += path.stat().st_size

    def ids_for(split, offset=0):
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split)
        text = '\n\n'.join(value for value in data['text'] if value.strip())
        return tokenizer(text, return_tensors='pt')['input_ids'][0][offset:offset + 4096].cuda()

    validation_ids = ids_for('validation')
    test_ids = torch.cat((ids_for('test'), ids_for('test', 4096)))

    layer_names = sorted({layer_number(name) for name, _ in linears})
    units = []
    for layer in layer_names:
        names = [name for name, _ in linears if layer_number(name) == layer]
        attention = [name for name in names if '.self_attn.' in name]
        mlp = [name for name in names if '.mlp.' in name]
        units.extend([
            dict(label=f'layer_{layer:02d}_whole', layer=layer, kind='whole', names=names),
            dict(label=f'layer_{layer:02d}_attention', layer=layer, kind='attention', names=attention),
            dict(label=f'layer_{layer:02d}_mlp', layer=layer, kind='mlp', names=mlp),
        ])

    modules = dict(linears)
    def set_names(names, values):
        for name in names:
            modules[name].weight.copy_(values[name].to('cuda'))

    def evaluate_all():
        return dict(validation=score(model, validation_ids),
                    test_first=score(model, test_ids[:4096]),
                    test_next=score(model, test_ids[4096:]))

    baseline = evaluate_all()
    unit_results = []
    for index, unit in enumerate(units):
        set_names(unit['names'], original)
        validation = score(model, validation_ids)
        set_names(unit['names'], quantized)
        parameters = sum(original[name].numel() for name in unit['names'])
        gain = baseline['validation']['nll'] - validation['nll']
        result = dict(label=unit['label'], layer=unit['layer'], kind=unit['kind'],
                      names=unit['names'], parameters=parameters,
                      extra_bits=14 * parameters, validation=validation,
                      validation_nll_gain=gain,
                      gain_per_billion_extra_bits=gain * 1e9 / (14 * parameters))
        unit_results.append(result)
        print('unit', index + 1, len(units), unit['label'], gain, flush=True)

    compressed_weights = sum(weight.numel() for weight in quantized.values())
    all_weights = sum(parameter.numel() for parameter in model.parameters())
    base_whole_bits = 8 * payload + 16 * (all_weights - compressed_weights)

    def cumulative(kind, counts, rank_by_cost=False):
        candidates = [result for result in unit_results
                      if result['kind'] == kind or
                      (kind == 'attention_and_mlp' and result['kind'] in ('attention', 'mlp'))]
        key = 'gain_per_billion_extra_bits' if rank_by_cost else 'validation_nll_gain'
        candidates.sort(key=lambda item: item[key], reverse=True)
        rows = []
        active = []
        for count in range(1, max(counts) + 1):
            active.extend(candidates[count - 1]['names'])
            set_names(candidates[count - 1]['names'], original)
            if count in counts:
                extra = 14 * sum(original[name].numel() for name in active)
                rows.append(dict(count=count,
                                 selected=[entry['label'] for entry in candidates[:count]],
                                 restored_parameters=sum(original[name].numel() for name in active),
                                 compressed_linear_equivalent_bpw=(8 * payload + extra) / compressed_weights,
                                 whole_model_equivalent_bpw=(base_whole_bits + extra) / all_weights,
                                 scores=evaluate_all()))
                print('cumulative', kind, count, rows[-1]['scores']['test_first']['perplexity'], flush=True)
        set_names(active, quantized)
        return rows

    whole_counts = [1, 2, 3, 4, 6, 8, 12, 16, 20, 24]
    group_counts = [1, 2, 4, 8, 12, 16, 24]
    combined_counts = [1, 2, 4, 8, 12, 16, 24, 32, 36, 40, 44, 48]
    report = dict(
        experiment='bf16_restoration_oracle', diagnostic_only=True,
        selection_split='WikiText-2 validation first 4096 tokens',
        test_splits=['WikiText-2 test 0:4096', 'WikiText-2 test 4096:8192'],
        model_revision=MODEL_REVISION, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        checkpoint=str(checkpoint), source_variant=source['variant'], source_scores=source.get('quantized'),
        baseline=baseline, compressed_payload_bytes=payload,
        compressed_weights=compressed_weights, all_weights=all_weights,
        serialized_compressed_linear_bpw=8 * payload / compressed_weights,
        whole_model_bpw=base_whole_bits / all_weights,
        unit_results=unit_results,
        cumulative_whole_layers=cumulative('whole', whole_counts),
        cumulative_attention_groups=cumulative('attention', group_counts, rank_by_cost=True),
        cumulative_mlp_groups=cumulative('mlp', group_counts, rank_by_cost=True),
        cumulative_all_groups=cumulative('attention_and_mlp', combined_counts, rank_by_cost=True),
    )
    report['elapsed_seconds'] = time.time() - started
    out.write_text(json.dumps(report, indent=2))
    print('FINAL', json.dumps(dict(baseline=baseline,
          whole=report['cumulative_whole_layers'],
          attention=report['cumulative_attention_groups'],
          mlp=report['cumulative_mlp_groups'],
          all_groups=report['cumulative_all_groups'])), flush=True)


if __name__ == '__main__':
    main()
