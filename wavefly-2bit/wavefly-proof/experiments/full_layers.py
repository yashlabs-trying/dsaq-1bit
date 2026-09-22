"""Offline numerical validation only: materializes reconstructed layers, not an inference kernel."""
import run as r
from run import *
from transformers import AutoModelForCausalLM,AutoTokenizer

def reconstruct(w,method,k,bits,rescount):
    b=16; nr,nc=w.shape[0]//b,w.shape[1]//b
    assert nr*b==w.shape[0] and nc*b==w.shape[1]
    tiles=w.reshape(nr,b,nc,b).permute(0,2,1,3).reshape(-1,b,b)
    result=torch.empty_like(tiles);total=32;d=basis(b)
    for start in range(0,len(tiles),4096):
        v=tiles[start:start+4096];t=len(v)
        if method=='scalar':
            h,_,_=quant(v.flatten(1),bits);h=h.reshape_as(v)
            total+=t*(b*b*bits//8+(2 if bits!=16 else 0))
        else:
            c=d@v@d.T;ids=select(c,k,method);values=c.flatten(1).gather(1,ids);q,_,_=quant(values,bits)
            cc=torch.zeros(t,b*b,device='cuda').scatter_(1,ids,q);h=d.T@cc.reshape_as(v)@d
            re=v-h;ix=re.flatten(1).abs().topk(rescount,dim=1).indices
            h.flatten(1).scatter_add_(1,ix,re.flatten(1).gather(1,ix).half().float())
            total+=storage(t,b,k,bits,method,rescount)['total_bytes']-32
        result[start:start+t]=h
    return result.reshape(nr,nc,b,b).permute(0,2,1,3).reshape_as(w),total

configs=[('scalar',256,16,0),('scalar',256,4,0),('scalar',256,2,0),('fixed',32,4,0),('topk',32,4,0),('topk',32,4,3),('topk',32,4,13),('topk',16,16,0),('topk',256,16,0)]
rows=[]; codec=[]
for mid in ['Qwen/Qwen2.5-0.5B','facebook/opt-125m','EleutherAI/pythia-70m']:
    print('FULL LOAD',mid,flush=True)
    tok=AutoTokenizer.from_pretrained(mid);model=AutoModelForCausalLM.from_pretrained(mid,torch_dtype=torch.float32,attn_implementation='eager').cuda().eval()
    count=model.config.num_hidden_layers; chosen=[0,count//2,count-1];targets={};acts={};hooks=[]
    for name,m in model.named_modules():
        li=next((int(p) for p in name.split('.') if p.isdigit()),None)
        if isinstance(m,torch.nn.Linear) and (li in chosen or name=='lm_head'):
            targets[name]=(m,li);acts[name]=[]
            def hook(mod,args,out,n=name):acts[n].append(args[0].detach().reshape(-1,args[0].shape[-1])[:64].cpu())
            hooks.append(m.register_forward_hook(hook))
    with torch.no_grad():
        for text in TEXTS:model(**tok(text,return_tensors='pt',max_length=128,truncation=True).to('cuda'))
    for h in hooks:h.remove()
    with torch.no_grad():
        for name,(m,li) in targets.items():
            w=m.weight.detach();x=torch.cat(acts[name])[:256].cuda(); y=x@w.T
            if name==next(iter(targets)):
                original=w[:16,:16][None].clone(); current=original.clone(); d=basis(16)
                for bits in [16,8,4,2]:
                    c=d@current@d.T; ids=select(c,32,'topk');vals=c.flatten(1).gather(1,ids)
                    rx=torch.empty(1,0,dtype=torch.long,device='cuda');rv=torch.empty(1,0,device='cuda')
                    data=serialize(vals,ids,16,bits,'topk',rx,rv);current=deserialize(data)
                    fn=mid.replace('/','_')+f'_requant_{bits}.bin';(OUT/'packed'/fn).write_bytes(data)
                    codec.append(dict(model=mid,layer=name,bits=bits,bytes=len(data),bpw=len(data)*8/256,sequence='FP16 -> INT8 -> INT4 -> 2-bit',**metrics(original,current,x[:32,:16][None])))
                (OUT/'json/real_requantization.json').write_text(json.dumps(codec,indent=2))
            for method,k,bits,res in configs:
                h,byt=reconstruct(w,method,k,bits,res);yh=x@h.T
                rows.append(dict(model=mid,layer=name,stage=['early','middle','late'][chosen.index(li)] if li in chosen else 'head',method=method,k=k,bits=bits,residual_count=res,bpw=byt*8/w.numel(),total_bytes=byt,weight_error=((w-h).norm()/w.norm()).item(),output_error=((y-yh).norm()/y.norm()).item(),tokens=len(x),bias_included=False))
            pd.DataFrame(rows).to_csv(OUT/'csv/full_layers.csv',index=False);print('FULL DONE',mid,name,flush=True)
    del model,targets,acts,m,w,x,h,y,yh
    torch.cuda.empty_cache()
print('FULL ALL DONE',flush=True)
