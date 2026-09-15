"""Full-model correlated beam, optionally alternating paths and node fitting."""
import os
os.environ['FLASHRUN_TRELLIS_WIDTH']='32'
import json,time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
from lattice_experiment import MODEL_REVISION,mixed_right
from trellis_quant import load_layer,save_layer,decode_paths,metadata_allocation
from trellis_experiment import trellis_encode,reconstruct
from trellis_node_fit import fit_output_nodes
from trellis_beam import correlated_beam
from aq_block_experiment import evaluate


@torch.inference_mode()
def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    started=time.time();beam=int(os.environ.get('FLASHRUN_BEAM','2'))
    rounds=int(os.environ.get('FLASHRUN_NODE_ROUNDS','0'))
    batch_size=int(os.environ.get('FLASHRUN_BEAM_BATCH','2048'))
    variant=f'beam{beam}_rounds{rounds}'
    out=Path(os.environ.get('FLASHRUN_OUTPUT','/workspace/'+variant));out.mkdir(exist_ok=True)
    base=Path('/workspace/ablation_both')
    base_report=json.loads((base/'report.json').read_text())
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B',revision=MODEL_REVISION,
                                            torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B',revision=MODEL_REVISION)
    layers=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n!='lm_head']
    allocations=metadata_allocation([n for n,_ in layers],'global_mlp')
    hs=torch.load('/workspace/hessians_refreshed.pt',weights_only=True)
    report=dict(model_revision=MODEL_REVISION,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
                full_model=True,variant=variant,reset_width=32,budget_mode='global_mlp',
                beam_per_state=beam,node_fitting_rounds=rounds,beam_row_batch=batch_size,
                original_model_finetuned=False,reference=base_report['quantized'],
                bf16=base_report['bf16'],calibration_tokens=8192,calibration_length=256,matrices=[])
    if os.environ.get('FLASHRUN_RESUME','0') == '1' and (out/'report.json').exists():
        saved=json.loads((out/'report.json').read_text())
        for key in ('model_revision','variant','beam_per_state','node_fitting_rounds','budget_mode'):
            if saved[key] != report[key]:raise ValueError('incompatible resumed experiment: '+key)
        report['matrices']=saved['matrices']
    completed=len(report['matrices'])
    payload=0
    for i,(name,module) in enumerate(layers):
        tick=time.time()
        if i < completed:
            path=out/f'{variant}_layer_{i:03d}.bin'
            if report['matrices'][i]['name'] != name:raise ValueError('resumed matrix order mismatch')
            module.weight.copy_(reconstruct(path).to(module.weight.dtype));payload+=path.stat().st_size
            continue
        paths,nodes,seed,scale=load_layer(base/f'both_layer_{i:03d}.bin')
        w=module.weight.float();rows,cols=w.shape
        gen=torch.Generator(device='cuda').manual_seed(seed)
        su=torch.randint(2,(cols,),device='cuda',generator=gen).float()*2-1
        sv=torch.randint(2,(rows,),device='cuda',generator=gen).float()*2-1
        h=hs[name].cuda().clone();h+=.01*h.diag().mean()*torch.eye(cols,device='cuda')
        hr=mixed_right(mixed_right(h,su).t(),su).t().contiguous()
        wr=mixed_right(mixed_right(w.t(),sv).t(),su)/scale
        nodes=nodes.cuda()
        def search(candidate,metric):
            values,indices=[],[]
            for chunk in candidate.split(batch_size):
                q,ids=correlated_beam(chunk,nodes,metric,beam=beam)
                values.append(q);indices.append(ids)
            return torch.cat(values),torch.cat(indices)
        q,paths=trellis_encode(wr,hr,nodes,weighted=True,rerank=True,forced=allocations[name],search_fn=search)
        histories=[]
        for _ in range(rounds):
            old_nodes,old_q,old_paths=nodes,q,paths
            old_error=wr-old_q;old_loss=float(((old_error@hr)*old_error).sum())
            nodes,fit=fit_output_nodes(wr,hr,paths,nodes,width=32,iterations=12)
            q,paths=trellis_encode(wr,hr,nodes,weighted=True,rerank=True,forced=allocations[name],search_fn=search)
            error=wr-q;new_loss=float(((error@hr)*error).sum())
            fit['post_search_output_loss']=new_loss;fit['previous_output_loss']=old_loss
            fit['accepted_after_search']=new_loss<=old_loss
            if new_loss>old_loss:nodes,q,paths=old_nodes,old_q,old_paths
            histories.append(fit)
        path=out/f'{variant}_layer_{i:03d}.bin'
        save_layer(path,paths,nodes,cols,seed,scale,forced=allocations[name])
        restored,table,_,_=load_layer(path)
        torch.testing.assert_close(decode_paths(restored.cuda(),table.cuda()).reshape_as(q),q,atol=0,rtol=0)
        module.weight.copy_(reconstruct(path).to(module.weight.dtype));payload+=path.stat().st_size
        error=wr-q
        matrix=dict(name=name,shape=list(w.shape),node_rounds=histories,
                    relative_hessian_error=float(((error@hr)*error).sum()/((wr@hr)*wr).sum()),
                    encoding_seconds=time.time()-tick,serialized_bytes=path.stat().st_size)
        report['matrices'].append(matrix)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        if i%10==0:print('encoded',i,'seconds',matrix['encoding_seconds'],flush=True)
    test=load_dataset('wikitext','wikitext-2-raw-v1',split='test')
    text='\n\n'.join(t for t in test['text'] if t.strip())
    ids=tokenizer(text,return_tensors='pt')['input_ids'][0][:8192].cuda()
    report['quantized']=dict(first_4096=evaluate(model,ids[:4096]),next_4096=evaluate(model,ids[4096:]))
    compressed=sum(m.weight.numel() for _,m in layers);all_weights=sum(p.numel() for p in model.parameters())
    report['serialized_bpw']=8*payload/compressed;assert report['serialized_bpw']==2
    report['whole_model_bpw']=(8*payload+16*(all_weights-compressed))/all_weights
    report['elapsed_seconds']=time.time()-started
    report['total_matrix_encoding_seconds']=sum(m['encoding_seconds'] for m in report['matrices'])
    (out/'report.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps(report['quantized']),flush=True)


if __name__=='__main__':main()
