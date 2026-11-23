import os,sys,json,math,time,platform,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from wavefly.core import *
torch.manual_seed(20260922); np.random.seed(20260922)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.set_num_threads(8)
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results'
for p in ['csv','json','plots','packed']: (OUT/p).mkdir(parents=True,exist_ok=True)
ROWS=[]; ROUND=[]; FAIL=[]; VERIFY=[]; MODELINFO=[]
KS=[256,192,128,96,64,48,32,24,16,8,4]
DENS=[0,.001,.0025,.005,.01,.02,.05]

def save():
    pd.DataFrame(ROWS).to_csv(OUT/'csv/measurements.csv',index=False)
    pd.DataFrame(ROUND).to_csv(OUT/'csv/roundtrip.csv',index=False)
    (OUT/'json/failures.json').write_text(json.dumps(FAIL,indent=2))
    (OUT/'json/serialization.json').write_text(json.dumps(VERIFY,indent=2))
    (OUT/'json/models.json').write_text(json.dumps(MODELINFO,indent=2))

def families(b,t=128):
    z=lambda:torch.randn(t,b,b,device='cuda')
    d=basis(b); c=z(); mask=torch.zeros(b,b,device='cuda'); mask[:max(2,b//8),:max(2,b//8)]=1
    smooth=d.T@(c*mask)@d
    f={'gaussian':z(),'uniform':torch.rand(t,b,b,device='cuda')*2-1,'smooth':smooth}
    for rank in [1,2,4,8]: f['rank'+str(rank)]=torch.randn(t,b,rank,device='cuda')@torch.randn(t,rank,b,device='cuda')/math.sqrt(rank)
    for sp in [.5,.75,.9,.95]: f['sparse'+str(sp)]=z()*(torch.rand(t,b,b,device='cuda')>sp)
    for p in [.001,.01,.05]: f['outlier'+str(p)]=z()+20*z()*(torch.rand(t,b,b,device='cuda')<p)
    c=torch.zeros(t,b,b,device='cuda');c[:,1,2]=torch.randn(t,device='cuda');c[:,2,1]=torch.randn(t,device='cuda')
    f['sinusoidal']=d.T@c@d
    f['mixed']=smooth+.1*z()+2*z()*(torch.rand(t,b,b,device='cuda')<.01)
    return f

def roundtrip(w,label,b):
    for dt in [torch.float64,torch.float32,torch.float16]:
        v=w[:1].to(dt); d=basis(b,dt); c=d@v@d.T; h=d.T@c@d
        x=torch.randn(1,32,b,device='cuda')
        row=dict(family=label,tile=b,dtype=str(dt),orthogonality=(d.double()@d.double().T-torch.eye(b,device='cuda')).abs().max().item(),**metrics(v,h,x))
        ROUND.append(row)

def evaluate(w,x,meta,full=True):
    t,b,_=w.shape; n=b*b; d=basis(b); c=d@w@d.T
    kvals=KS if b==16 else sorted(set([n,3*n//4,n//2,n//4,n//8,n//16]),reverse=True)
    energies=c.square().sum((1,2)).clamp_min(1e-30)
    base=dict(meta,tile=b,tiles=t,weights=t*n)
    # Baseline: one group per tile, FP16 scale for integer codes, 32-byte header.
    for bits in [16,4,2]:
        h,_,_=quant(w.flatten(1),bits);h=h.reshape_as(w)
        byt=t*math.ceil(n*bits/8)+(2*t if bits!=16 else 0)+32
        ROWS.append(dict(base,method='scalar',k=n,bits=bits,residual_requested=0,residual_count=0,residual_actual=0,selection='none',energy=1,bpw=byt*8/(t*n),total_bytes=byt,**metrics(w,h,x)))
    for method in ['fixed','topk']:
        for k in kvals:
            ids=select(c,k,method); vals=c.flatten(1).gather(1,ids)
            energy=(vals.square().sum(1)/energies).mean().item()
            for bits in ([16,8,4,2] if full else [16]):
                q,_,_=quant(vals,bits); cq=torch.zeros(t,n,device='cuda').scatter_(1,ids,q)
                h=d.T@cq.reshape(t,b,b)@d; residual=w-h
                for selection in (['magnitude','activation'] if meta['source']=='real' else ['magnitude']):
                    importance=residual.abs()*(x.abs().mean(1)[:,None,:] if selection=='activation' else 1)
                    rid=importance.flatten(1).argsort(dim=1,descending=True)
                    for density in (DENS if full else [0,.01,.05]):
                        if selection=='activation' and density==0: continue
                        # Fixed slots: nearest integer number per tile. Explicit actual density.
                        r=int(math.floor(density*n+.5)); rx=rid[:,:r]
                        rv=residual.flatten(1).gather(1,rx).half().float()
                        hh=h.clone();hh.flatten(1).scatter_add_(1,rx,rv)
                        row=dict(base,method=method,k=k,bits=bits,residual_requested=density,residual_count=r,residual_actual=r/n,selection=selection,energy=energy,
                                 idct_flops_per_tile=4*b**3,**storage(t,b,k,bits,method,r),**metrics(w,hh,x))
                        ROWS.append(row)
                        if meta.get('pack_test') and k==32 and density==.01 and selection=='magnitude':
                            data=serialize(vals,ids,b,bits,method,rx,rv); decoded=deserialize(data)
                            err=(decoded-hh).abs().max().item(); assert err<1e-6
                            fn=f"{method}_{bits}bit.bin"; (OUT/'packed'/fn).write_bytes(data)
                            VERIFY.append(dict(file=fn,bytes=len(data),bpw=len(data)*8/(t*n),decode_max_abs=err,**metrics(w,decoded,x)))
    return c

def synthetic():
    for b in [8,16,32,64]:
        for name,w in families(b).items():
            if b!=64: roundtrip(w,name,b)
            x=torch.randn(len(w),16,b,device='cuda')
            evaluate(w,x,dict(source='synthetic',model='synthetic',layer=name,stage='na',shape=f'{b}x{b}',pack_test=(b==16 and name=='gaussian')),full=(b==16))
        save();print('SYNTHETIC DONE',b,len(ROWS),flush=True)

TEXTS=[
 'The study compared numerical methods for storing neural network weights. Measurements must include errors, memory, and computation. A reliable experiment uses reproducible random seeds and reports unsuccessful results.',
 'A traveler took a train from the city to a quiet village. Rain fell across the fields while she read a book about the history of mathematics and the development of modern science.',
 'Python functions can calculate the sum of a sequence. For example, define a function that loops through a list, adds each value to an accumulator, and returns the result. Testing includes empty lists and negative numbers.',
 'A company sold twelve products on Monday and eighteen on Tuesday. The total number sold was thirty. If each product costs five dollars, the combined revenue equals one hundred and fifty dollars.',
 'Plants use sunlight, water, and carbon dioxide to produce sugars. Ecosystems connect plants, animals, fungi, and microbes through complex exchanges of energy and nutrients.'
]

def real():
    from transformers import AutoModelForCausalLM,AutoTokenizer
    specs=[('Qwen/Qwen2.5-0.5B','qwen'),('facebook/opt-125m','opt'),('EleutherAI/pythia-70m','pythia')]
    for modelid,family in specs:
        print('LOADING',modelid,flush=True)
        try:
            tok=AutoTokenizer.from_pretrained(modelid)
            model=AutoModelForCausalLM.from_pretrained(modelid,torch_dtype=torch.float32,attn_implementation='eager').to('cuda').eval()
            cfg=model.config; count=getattr(cfg,'num_hidden_layers',getattr(cfg,'n_layer',None)); chosen=[0,count//2,count-1]
            MODELINFO.append(dict(id=modelid,family=family,revision=getattr(cfg,'_commit_hash',None),layers=count,chosen=chosen,parameters=sum(p.numel() for p in model.parameters())))
            targets={}; stages={}
            for name,m in model.named_modules():
                if not isinstance(m,torch.nn.Linear):continue
                parts=name.split('.'); layernum=next((int(p) for p in parts if p.isdigit()),None)
                if layernum in chosen or name=='lm_head':
                    targets[name]=m;stages[name]=['early','middle','late'][chosen.index(layernum)] if layernum in chosen else 'head'
            acts={name:[] for name in targets}; hooks=[]
            def hook(name):
                def fn(m,args,out): acts[name].append(args[0].detach().reshape(-1,args[0].shape[-1])[:64].cpu())
                return fn
            for name,m in targets.items():hooks.append(m.register_forward_hook(hook(name)))
            with torch.no_grad():
                for text in TEXTS:
                    inp=tok(text,return_tensors='pt',truncation=True,max_length=128).to('cuda');model(**inp)
            for h in hooks:h.remove()
            matrices=[]
            for name,m in targets.items():
                x=torch.cat(acts[name])[:256].to('cuda')
                if name.endswith('query_key_value'):
                    # GPT-NeoX fused rows interleave Q,K,V within each attention head.
                    heads=cfg.num_attention_heads; hd=m.weight.shape[1]//heads
                    v=m.weight.detach().reshape(heads,3,hd,-1)
                    for j,label in enumerate(['q_proj','k_proj','v_proj']):matrices.append((name+'.'+label,v[:,j].reshape(-1,v.shape[-1]),x,stages[name]))
                else:matrices.append((name,m.weight.detach(),x,stages[name]))
            emb=model.get_input_embeddings().weight.detach()
            matrices.append(('embedding',emb,None,'embedding'))
            for name,w,x,stage in matrices:
                for b in [8,16,32,64]:
                    nr,nc=w.shape[0]//b,w.shape[1]//b
                    # 128 uniformly sampled aligned tiles; fixed seed, with replacement.
                    rr=torch.randint(nr,(128,),device='cuda');cc=torch.randint(nc,(128,),device='cuda');ar=torch.arange(b,device='cuda')
                    tiles=w[rr[:,None,None]*b+ar[None,:,None],cc[:,None,None]*b+ar[None,None,:]].float()
                    # Linear weight layout [output,input], metrics use X W^T.
                    xx=(x[:32,cc[:,None]*b+ar[None,:]].permute(1,0,2).contiguous() if x is not None else torch.eye(b,device='cuda')[None].expand(128,-1,-1))
                    meta=dict(source='real',model=modelid,layer=name,stage=stage,shape=str(list(w.shape)),activation='calibration_local_contribution' if x is not None else 'identity_embedding_reconstruction')
                    if b in [8,16,32]:roundtrip(tiles,modelid+'/'+name,b)
                    c=evaluate(tiles,xx,meta,full=(b==16))
                    if b==16:
                        heat=c.abs().mean(0).cpu().numpy(); fn=(family+'_'+name).replace('.','_')
                        np.save(OUT/'json'/(fn+'.npy'),heat)
                save();print('REAL DONE',modelid,name,len(ROWS),flush=True)
            del matrices,model,targets,acts,w,x,tiles,c
            torch.cuda.empty_cache()
        except Exception:
            FAIL.append(dict(model=modelid,traceback=traceback.format_exc()));save();print(FAIL[-1],flush=True)

if __name__=='__main__':
    env=dict(seed=20260922,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,python=sys.version,cpu=platform.processor(),cpu_info=Path('/proc/cpuinfo').read_text().splitlines()[:30],tf32=False)
    try:import triton;env['triton']=triton.__version__
    except ImportError:env['triton']=None
    (OUT/'json/environment.json').write_text(json.dumps(env,indent=2));(OUT/'json/calibration.json').write_text(json.dumps(TEXTS,indent=2))
    with torch.no_grad(): synthetic(); real()
    save();print('ALL DONE',len(ROWS),flush=True)
