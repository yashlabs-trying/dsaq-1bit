"""Probe 2/4/8 per-state beams on current model's refinement candidates."""
import os
os.environ['FLASHRUN_TRELLIS_WIDTH'] = '32'
import json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from lattice_experiment import MODEL_REVISION,mixed_right
from trellis_quant import load_layer,decode_paths,viterbi
from trellis_beam import correlated_beam


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B',revision=MODEL_REVISION,
                                             torch_dtype=torch.bfloat16).cuda().eval()
    layers=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n!='lm_head']
    hs=torch.load('/workspace/hessians_refreshed.pt',weights_only=True)
    names=['model.layers.0.self_attn.q_proj','model.layers.0.mlp.down_proj',
           'model.layers.12.self_attn.q_proj','model.layers.12.mlp.down_proj']
    report=dict(reference='ablation_both',reference_perplexity=[41.5642589010658,59.252194375266015],
                gpu=torch.cuda.get_device_name(),model_revision=MODEL_REVISION,probes=[])
    for i,(name,module) in enumerate(layers):
        if name not in names: continue
        paths,nodes,seed,scale=load_layer(Path('/workspace/ablation_both')/f'both_layer_{i:03d}.bin')
        w=module.weight.float(); rows,cols=w.shape
        gen=torch.Generator(device='cuda').manual_seed(seed)
        su=torch.randint(2,(cols,),device='cuda',generator=gen).float()*2-1
        sv=torch.randint(2,(rows,),device='cuda',generator=gen).float()*2-1
        h=hs[name].cuda().clone();h+=.01*h.diag().mean()*torch.eye(cols,device='cuda')
        hr=mixed_right(mixed_right(h,su).t(),su).t().contiguous()
        wr=mixed_right(mixed_right(w.t(),sv).t(),su)/scale
        nodes=nodes.cuda();q=decode_paths(paths.cuda(),nodes).reshape_as(wr)
        eh=(wr-q)@hr
        groups=cols//32
        sampled=torch.linspace(0,rows-2,16,device='cuda').long()
        for group in [groups//4,groups//2,3*groups//4]:
            start,end=group*32,(group+1)*32; metric=hr[start:end,start:end]
            candidate=(q[:,start:end]+eh[:,start:end]@torch.linalg.inv(metric))[sampled]
            probe=dict(name=name,group=group,sampled_rows=16,variants={})
            for beam in (0,2,4,8):
                torch.cuda.synchronize();started=time.time()
                out,ids=(viterbi(candidate,nodes,metric.diag(),full_metric=metric) if beam==0 else
                         correlated_beam(candidate,nodes,metric,beam=beam))
                torch.cuda.synchronize();elapsed=time.time()-started
                error=candidate-out;loss=float(((error@metric)*error).sum())
                torch.testing.assert_close(out,decode_paths(ids,nodes))
                probe['variants'][str(beam)]=dict(loss=loss,seconds=elapsed)
            report['probes'].append(probe)
            print(name,group,{b:round(v['loss']/probe['variants']['0']['loss'],4)
                              for b,v in probe['variants'].items()},flush=True)
    report['summary']={}
    for beam in (2,4,8):
        ratios=[p['variants'][str(beam)]['loss']/p['variants']['0']['loss'] for p in report['probes']]
        report['summary'][str(beam)]=dict(mean_fractional_error_reduction=1-sum(ratios)/len(ratios),
              seconds=sum(p['variants'][str(beam)]['seconds'] for p in report['probes']))
    Path('/workspace/beam_gate.json').write_text(json.dumps(report,indent=2))
    print('SUMMARY',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()
