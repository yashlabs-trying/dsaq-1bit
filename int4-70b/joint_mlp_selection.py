"""Joint nonlinear MLP selection from exact-two-bit candidate checkpoints.

For each transformer layer, choose gate/up/down codes jointly by true post-SiLU
MLP output error on quantized-rollout inputs. A full validation-NLL check accepts
or rejects the proposed triple before moving to the next layer. Model weights
are never trained and the serialized format/budget is unchanged.
"""
import itertools
import json
import os
import shutil
import time
from pathlib import Path

os.environ['FLASHRUN_TRELLIS_WIDTH'] = '32'
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from aq_block_experiment import evaluate
from lattice_experiment import MODEL_REVISION
from trellis_experiment import reconstruct


ARMS = {
    'current': (Path('/workspace/ablation_current'), 'path_trained_rerank'),
    'hessian': (Path('/workspace/ablation_hessian_only'), 'hessian_only'),
    'node': (Path('/workspace/ablation_layer_output_only'), 'layer_output_only'),
    'both': (Path('/workspace/ablation_both'), 'both'),
}
extra = os.environ.get('FLASHRUN_JOINT_EXTRA')
if extra:
    extra_path = Path(extra)
    extra_report = json.loads((extra_path / 'report.json').read_text())
    ARMS['previous_pass'] = (extra_path, extra_report['variant'])
extras = os.environ.get('FLASHRUN_JOINT_EXTRAS')
if extras:
    for specification in extras.split(','):
        arm, raw_path = specification.split('=', 1)
        if arm in ARMS:
            raise ValueError('duplicate candidate arm: ' + arm)
        extra_path = Path(raw_path)
        extra_report = json.loads((extra_path / 'report.json').read_text())
        ARMS[arm] = (extra_path, extra_report['variant'])
BASE_ARM = os.environ.get('FLASHRUN_JOINT_BASE', 'current')
if BASE_ARM not in ARMS:
    raise ValueError('unknown base arm: ' + BASE_ARM)


def linear_layers(model):
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name != 'lm_head']


def file_for(arm, index):
    directory, variant = ARMS[arm]
    return directory / f'{variant}_layer_{index:03d}.bin'


def validate_selection_chain(rows, tolerance=1e-12):
    previous = None
    for row in rows:
        before = row['validation_before']['nll']
        after = row['validation_after']['nll']
        if previous is not None and abs(before - previous) > tolerance:
            raise ValueError('validation chain is discontinuous')
        if after > before + tolerance:
            raise ValueError('accepted state increased validation NLL')
        previous = after


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.time()
    out = Path(os.environ.get('FLASHRUN_JOINT_OUTPUT', '/workspace/mlp_joint_select'))
    out.mkdir(parents=True, exist_ok=True)
    reports = {arm: json.loads((path / 'report.json').read_text())
               for arm, (path, _) in ARMS.items()}

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION)
    teacher = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16).cuda().eval()
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-0.5B', revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16).cuda().eval()
    layers = linear_layers(model)
    index_by_name = {name: index for index, (name, _) in enumerate(layers)}
    module_by_name = dict(layers)
    for index, (name, module) in enumerate(layers):
        if reports['current']['matrices'][index]['name'] != name:
            raise ValueError('matrix order mismatch')
        module.weight.copy_(reconstruct(file_for(BASE_ARM, index)).to(torch.bfloat16))

    validation = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')
    validation_text = '\n\n'.join(text for text in validation['text'] if text.strip())
    validation_offset = int(os.environ.get('FLASHRUN_VALIDATION_OFFSET', '0'))
    all_validation_ids = tokenizer(validation_text, return_tensors='pt')['input_ids'][0]
    validation_ids = all_validation_ids[validation_offset:validation_offset + 4096].cuda()
    if len(validation_ids) != 4096:
        raise ValueError('validation window is shorter than 4096 tokens')
    test = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    test_text = '\n\n'.join(text for text in test['text'] if text.strip())
    test_ids = tokenizer(test_text, return_tensors='pt')['input_ids'][0][:8192].cuda()

    selections = []
    accepted_sources = {}
    current_validation = None
    arm_names = list(ARMS)
    for layer_index in range(len(model.model.layers)):
        block = model.model.layers[layer_index]
        captured = []

        def capture(_module, args):
            values = args[0].detach().reshape(-1, args[0].shape[-1])
            captured.append(values[::max(1, len(values) // 16)][:16].cpu())

        hook = block.mlp.register_forward_pre_hook(capture)
        before = evaluate(model, validation_ids)
        hook.remove()
        current_validation = before
        x = torch.cat(captured)[:256].cuda()
        teacher_target = teacher.model.layers[layer_index].mlp(x).float()

        names = {kind: f'model.layers.{layer_index}.mlp.{kind}_proj'
                 for kind in ('gate', 'up', 'down')}
        indices = {kind: index_by_name[name] for kind, name in names.items()}
        candidates = {}
        for kind in ('gate', 'up', 'down'):
            candidates[kind] = {
                arm: reconstruct(file_for(arm, indices[kind])).to(torch.bfloat16)
                for arm in arm_names
            }

        gate_outputs = [F.linear(x, candidates['gate'][arm]) for arm in arm_names]
        up_outputs = [F.linear(x, candidates['up'][arm]) for arm in arm_names]
        hidden = torch.stack([
            F.silu(gate_outputs[g]) * up_outputs[u]
            for g in range(len(arm_names)) for u in range(len(arm_names))
        ])
        losses = {}
        target_energy = float(teacher_target.square().mean())
        for down_index, down_arm in enumerate(arm_names):
            decoded = F.linear(hidden.flatten(0, 1), candidates['down'][down_arm])
            decoded = decoded.reshape(len(arm_names), len(arm_names), len(x), -1).float()
            for gate_index, gate_arm in enumerate(arm_names):
                for up_index, up_arm in enumerate(arm_names):
                    key = (gate_arm, up_arm, down_arm)
                    losses[key] = float((decoded[gate_index, up_index] - teacher_target).square().mean())

        choice = min(losses, key=losses.get)
        baseline_choice = (BASE_ARM, BASE_ARM, BASE_ARM)
        relative_gain = 1 - losses[choice] / losses[baseline_choice]
        proposed_choice = choice
        for kind, arm in zip(('gate', 'up', 'down'), choice):
            module_by_name[names[kind]].weight.copy_(candidates[kind][arm])
        proposed = evaluate(model, validation_ids)
        accepted = proposed['nll'] < before['nll']
        if not accepted:
            for kind in ('gate', 'up', 'down'):
                module_by_name[names[kind]].weight.copy_(candidates[kind][BASE_ARM])
            final_validation = before
            choice = baseline_choice
        else:
            final_validation = proposed
            for kind, arm in zip(('gate', 'up', 'down'), choice):
                accepted_sources[names[kind]] = arm
        selections.append(dict(
            layer=layer_index, sampled_rollout_tokens=len(x),
            proposed=dict(zip(('gate', 'up', 'down'), proposed_choice)),
            selected=dict(zip(('gate', 'up', 'down'), choice)), accepted=accepted,
            teacher_output_mse_current=losses[baseline_choice],
            teacher_output_mse_best=min(losses.values()),
            teacher_output_energy=target_energy,
            relative_teacher_mse_reduction=relative_gain,
            validation_before=before, validation_proposed=proposed,
            validation_after=final_validation))
        print('layer', layer_index, choice, 'proxy_gain', relative_gain,
              'nll', before['nll'], proposed['nll'], 'accepted', accepted, flush=True)
        del candidates, gate_outputs, up_outputs, hidden, teacher_target, x
        torch.cuda.empty_cache()

    variant = 'mlp_joint_select'
    payload = 0
    matrix_rows = []
    for index, (name, _) in enumerate(layers):
        arm = accepted_sources.get(name, BASE_ARM)
        source = file_for(arm, index)
        destination = out / f'{variant}_layer_{index:03d}.bin'
        shutil.copyfile(source, destination)
        payload += destination.stat().st_size
        matrix_rows.append(dict(name=name, source_arm=arm,
                                serialized_bytes=destination.stat().st_size))

    quantized = dict(first_4096=evaluate(model, test_ids[:4096]),
                     next_4096=evaluate(model, test_ids[4096:]))
    validate_selection_chain(selections)
    compressed = sum(module.weight.numel() for _, module in layers)
    all_weights = sum(parameter.numel() for parameter in model.parameters())
    report = dict(
        model_revision=MODEL_REVISION, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        full_model=True, variant=variant, reset_width=32, budget_mode='global_mlp',
        original_model_finetuned=False,
        selection_split=f'WikiText-2 validation tokens {validation_offset}:{validation_offset + 4096}',
        validation_offset=validation_offset,
        candidate_arms={arm: str(path) for arm, (path, _) in ARMS.items()},
        base_arm=BASE_ARM, candidate_combinations_per_layer=len(ARMS) ** 3,
        rollout_samples_per_layer=256,
        baseline_validation=selections[0]['validation_before'],
        final_validation=selections[-1]['validation_after'],
        source_quantized=reports[BASE_ARM]['quantized'], bf16=reports['current']['bf16'],
        quantized=quantized, selections=selections, matrices=matrix_rows,
        accepted_layers=sum(row['accepted'] for row in selections),
        compressed_payload_bytes=payload, serialized_bpw=8 * payload / compressed,
        whole_model_bpw=(8 * payload + 16 * (all_weights - compressed)) / all_weights,
        elapsed_seconds=time.time() - started)
    if report['serialized_bpw'] != 2:
        raise ValueError('serialized budget changed')
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print('FINAL', json.dumps(dict(accepted_layers=report['accepted_layers'],
          baseline_validation=report['baseline_validation'], final_validation=report['final_validation'],
          source=report['source_quantized'], quantized=quantized,
          serialized_bpw=report['serialized_bpw'])), flush=True)


if __name__ == '__main__':
    main()
