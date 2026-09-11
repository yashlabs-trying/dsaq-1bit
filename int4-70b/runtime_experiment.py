"""Validate reference compressed execution against saved dense hybrid perplexity."""
import gc
import json
import math
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from datasets import load_dataset
from flashrun_runtime import CompressedLinear


@torch.inference_mode()
def main():
    folder=Path('real_results')
    previous=json.loads((folder/'report.json').read_text())
    torch.set_num_threads(8)
    tokenizer=AutoTokenizer.from_pretrained(previous['model'],revision=previous['revision'])
    model=AutoModelForCausalLM.from_pretrained(previous['model'],revision=previous['revision'],torch_dtype=torch.bfloat16).cuda().eval()
    names=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n!='lm_head']
    for i,(name,module) in enumerate(names):
        archive=torch.load(folder/f'layer_{i:03d}.pt',weights_only=True)
        assert archive['name']==name
        replacement=CompressedLinear(archive,module.bias)
        # Verify exact BF16 reconstruction versus independent dense formula.
        from model_experiment import unpack
        p=archive['pca']; s=archive['sparse']
        reference=unpack(archive['q2'],2,'cuda')
        reference += p['codes'].cuda().float() @ p['basis'].cuda().float()+p['mean'].cuda().float()
        reference.flatten()[s['index'].cuda().long()] += s['value'].cuda().float()
        torch.testing.assert_close(replacement.reconstructed_weight(torch.bfloat16),reference.reshape(module.weight.shape).bfloat16(),rtol=0,atol=0)
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent),child,replacement)
    del names,module,reference,replacement,archive,p,s
    gc.collect(); torch.cuda.empty_cache()
    resident=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    data=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    ids=tokenizer('\n\n'.join(t for t in data['text'] if t.strip()),return_tensors='pt')['input_ids'][0][:4096].cuda()
    total=0.; count=0
    started=time.time()
    for window in ids.split(256):
        n=window.numel()-1
        total+=float(model(window[None],labels=window[None],use_cache=False).loss)*n
        count+=n
    perplexity=math.exp(total/count)
    report=dict(perplexity=perplexity,dense_hybrid_perplexity=previous['variants']['hybrid']['perplexity'],
        resident_allocated_bytes=resident,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),evaluation_seconds=time.time()-started,
        implementation='GPU-resident compressed linear buffers; one full linear matrix reconstructed temporarily per call; not tile-fused or optimized',
        limitations='0.5B only; no 70B fit or 3090 speed claim; prefill loss evaluation uses no KV cache')
    report['perplexity_matches_dense']=abs(perplexity-report['dense_hybrid_perplexity'])<1e-6
    prompt=tokenizer('The capital of France is',return_tensors='pt').to('cuda')
    torch.cuda.synchronize(); generation_start=time.time()
    output=model.generate(**prompt,max_new_tokens=32,do_sample=False,pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    report['generation_smoke_test']=dict(prompt='The capital of France is',
        continuation=tokenizer.decode(output[0,prompt.input_ids.shape[1]:],skip_special_tokens=True),
        new_tokens=int(output.shape[1]-prompt.input_ids.shape[1]),seconds=time.time()-generation_start,
        note='Functional smoke test only; no performance benchmark or quality claim')
    report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
    report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    (folder/'runtime_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
