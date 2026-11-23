"""Frozen-path node-fit diagnostic after stronger search; no PPL claim."""
import os
os.environ['FLASHRUN_TRELLIS_WIDTH']='32'
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from lattice_experiment import MODEL_REVISION,mixed_right
from trellis_quant import load_layer,decode_paths
from trellis_node_fit import fit_output_nodes


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    base=Path('/workspace/beam2_rounds0')
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B',revision=MODEL_REVISION,
                                            torch_dtype=torch.bfloat16).cuda().eval()
    layers=[(n,m) for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and n!='lm_head']
    hs=torch.load('/workspace/hessians_refreshed.pt',weights_only=True)
    results=[]
    for i in (0,1,3,6):
        name,module=layers[i]
        paths,nodes,seed,scale=load_layer(base/f'beam2_rounds0_layer_{i:03d}.bin')
        w=module.weight.float();rows,cols=w.shape
        gen=torch.Generator(device='cuda').manual_seed(seed)
        su=torch.randint(2,(cols,),device='cuda',generator=gen).float()*2-1
        sv=torch.randint(2,(rows,),device='cuda',generator=gen).float()*2-1
        h=hs[name].cuda().clone();h+=.01*h.diag().mean()*torch.eye(cols,device='cuda')
        hr=mixed_right(mixed_right(h,su).t(),su).t().contiguous()
        wr=mixed_right(mixed_right(w.t(),sv).t(),su)/scale
        nodes=nodes.cuda();paths=paths.cuda()
        table,diagnostics=fit_output_nodes(wr,hr,paths,nodes,width=32,iterations=12)
        old=wr-decode_paths(paths,nodes).reshape_as(wr)
        new=wr-decode_paths(paths,table).reshape_as(wr)
        before=float(((old@hr)*old).sum());after=float(((new@hr)*new).sum())
        result=dict(name=name,old_output_loss=before,new_output_loss=after,
                    fractional_reduction=1-after/before,solver=diagnostics)
        results.append(result);print(name,result['fractional_reduction'],flush=True)
    report=dict(kind='fixed_path_node_fit_only_not_model_eval',model_revision=MODEL_REVISION,
                gpu=torch.cuda.get_device_name(),reference='beam2_rounds0',probes=results)
    Path('/workspace/beam_node_gate.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
