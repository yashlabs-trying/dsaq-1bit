"""Layer-0 joint MLP-neuron package codec.

One latent ID describes (gate row, up row, down column) jointly.  The decoder
is a small shared embedding/MLP; the remaining residual is packed 1-bit signs
with row scales.  Attention is the frozen packed attention codec.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct, time
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM
import waverel_checkpoint_patterns as base

MLP=("gate_proj","up_proj","down_proj"); HIDDEN=64; PACKAGE_BITS=(8,10)

def pack_bits(x): return np.packbits(x.detach().cpu().numpy().astype(np.uint8).reshape(-1),bitorder='little').tobytes()
def unpack_bits(blob,n): return torch.from_numpy(np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder='little')[:n].copy())
def pack_ids(x,bits):
    a=x.detach().cpu().numpy().astype(np.uint16); b=((a[:,None]>>np.arange(bits,dtype=np.uint16))&1).astype(np.uint8)
    return np.packbits(b.reshape(-1),bitorder='little').tobytes()
def unpack_ids(blob,n,bits):
    b=np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder='little')[:n*bits].reshape(n,bits).astype(np.uint16)
    return torch.from_numpy((b*(1<<np.arange(bits,dtype=np.uint16))).sum(1).astype(np.int32))
def qtensor(x):
    s=max(float(x.detach().abs().max())/127,np.finfo(np.float16).tiny);q=torch.round(x.detach().cpu()/s).clamp(-127,127).to(torch.int8);return q,np.float16(s)

class PackageDecoder(nn.Module):
    def __init__(self,k):
        super().__init__();self.code=nn.Embedding(k,8);self.fc1=nn.Linear(8,HIDDEN);self.fc2=nn.Linear(HIDDEN,2688)
    def forward(self,z): return self.fc2(torch.nn.functional.silu(self.fc1(self.code(z))))

def kmeans_features(x,k,iters=12):
    # Package count is small; use a compact PCA feature for stable assignment.
    q=min(64,x.shape[1],x.shape[0]-1);u,s,v=torch.pca_lowrank(x,q=q,niter=2);feat=x@v[:,:q]
    centers=feat[torch.linspace(0,len(feat)-1,k).long()].clone()
    for _ in range(iters):
        d=feat.square().sum(1,keepdim=True)+centers.square().sum(1)-2*feat@centers.T;ids=d.argmin(1);new=centers.clone()
        for j in range(k):
            m=ids==j
            if m.any():new[j]=feat[m].mean(0)
        centers=new
    return feat,centers,ids

def package_metrics(originals,recon,acts):
    x=acts['gate_proj'].cuda();og,ou,od=(originals[n].cuda() for n in MLP);qg,qu,qd=(recon[n].cuda() for n in MLP)
    yg,yu=x@og.T,x@ou.T;yqg,yqu=x@qg.T,x@qu.T
    om=torch.nn.functional.silu(yg)*yu;qm=torch.nn.functional.silu(yqg)*yqu;oo=om@od.T;qo=qm@qd.T
    return (float((yqg-yg).square().sum()/yg.square().sum()),float((yqu-yu).square().sum()/yu.square().sum()),float((qm-om).square().sum()/om.square().sum()),float((qo-oo).square().sum()/oo.square().sum()))

def write_codec(path,bits,att_blob,ids,norm,decoder_state,residuals,total):
    params=[]
    for n,(q,s) in decoder_state.items():params.append({'name':n,'shape':list(q.shape),'count':q.numel()})
    meta={'version':1,'method':'Joint neuron package','package_bits':bits,'packages':len(ids),'weights':total,'attention_bytes':len(att_blob),'norm_count':len(norm),'params':params,'matrix_shapes':{n:list(residuals[n].shape) for n in MLP},'decoder_hidden':HIDDEN,'residual':'packed signs + fp16 row scales'}
    mh=json.dumps(meta,separators=(',',':'),sort_keys=True).encode();parts=[struct.pack('<I',len(mh)),mh,att_blob,norm.numpy().astype('<f2').tobytes(),pack_ids(ids,bits)]
    for n,(q,s) in decoder_state.items():parts += [q.numpy().tobytes(),np.asarray(s,dtype='<f2').tobytes()]
    for n in MLP:
        r=residuals[n].detach();parts += [pack_bits(r>=0),r.abs().mean(1).clamp_min(torch.finfo(torch.float16).tiny).half().numpy().astype('<f2').tobytes()]
    path.write_bytes(b''.join(parts))

def decode_codec(path):
    blob=path.read_bytes();pos=0;ml=struct.unpack_from('<I',blob,pos)[0];pos+=4;meta=json.loads(blob[pos:pos+ml]);pos+=ml
    ap=path.with_suffix('.att.tmp');ap.write_bytes(blob[pos:pos+meta['attention_bytes']]);pos+=meta['attention_bytes'];att,_=base.decode(ap);ap.unlink()
    norm=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=meta['norm_count'],offset=pos).copy());pos+=2*meta['norm_count'];n=meta['packages'];bits=meta['package_bits'];ib=(n*bits+7)//8;ids=unpack_ids(blob[pos:pos+ib],n,bits);pos+=ib
    st=OrderedDict()
    for p in meta['params']:
        q=torch.from_numpy(np.frombuffer(blob,dtype=np.int8,count=p['count'],offset=pos).copy()).view(p['shape']);pos+=p['count'];s=np.frombuffer(blob,dtype='<f2',count=1,offset=pos)[0];pos+=2;st[p['name']]=(q,s)
    dec=PackageDecoder(1<<bits).cuda();dec.load_state_dict({n:q.float()*float(s) for n,(q,s) in st.items()});z=dec(ids.cuda()).cpu()*norm.float()[None,:]
    out={};off=0
    for n in MLP:
        r,c=meta['matrix_shapes'][n];
        # Package layout: gate row, up row, down column.
        if n=='gate_proj':out[n]=z[:,:896].reshape(r,c)
        elif n=='up_proj':out[n]=z[:,896:1792].reshape(r,c)
        else:out[n]=z[:,1792:2688].reshape(c,r).T
    for n in MLP:
        r,c=meta['matrix_shapes'][n];nb=(r*c+7)//8;sg=unpack_bits(blob[pos:pos+nb],r*c).view(r,c);pos+=nb;rs=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=r,offset=pos).copy());pos+=2*r;out[n]+=((sg.float()*2-1)*rs.float()[:,None]) if n!='down_proj' else ((sg.float()*2-1)*rs.float()[:,None])
    assert pos==len(blob),(pos,len(blob));return {**att,**out},meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--attention',type=Path,default=Path('/workspace/tiny_mlp_decoder/attention.bin'));p.add_argument('--tokens',type=Path,default=Path('/workspace/waverel_patterns/fixed_token_ids.pt'));p.add_argument('--output',type=Path,default=Path('/workspace/joint_neuron_package'));p.add_argument('--sequence-length',type=int,default=256);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);base.seed_all();start=time.time()
    model=AutoModelForCausalLM.from_pretrained(base.MODEL,torch_dtype=torch.float32,device_map={'':'cuda'}).eval();orig={n:m.weight.detach().cpu().clone() for n,m in base.module_map(model).items()};total=sum(w.numel() for w in orig.values());tok=torch.load(a.tokens,map_location='cpu',weights_only=True);held,evaluation=tok['heldout'],tok['evaluation'];base.restore(model,orig);ref=base.perplexity(model,evaluation,a.sequence_length);acts=base.capture_inputs(model,held,a.sequence_length);att_blob=a.attention.read_bytes()
    # A package is [gate row | up row | down column], all length 896.
    pack=torch.cat((orig['gate_proj'],orig['up_proj'],orig['down_proj'].T),1).float();norm=pack.std(0).clamp_min(1e-5);target=pack/norm
    # Fit latent IDs in a compact PCA space; decoder fitting uses the full package.
    results=[]
    for bits in PACKAGE_BITS:
        print('BITS',bits,flush=True);k=1<<bits;feat,centers,ids=kmeans_features(target,k);dec=PackageDecoder(k).cuda();opt=torch.optim.AdamW(dec.parameters(),lr=3e-3,weight_decay=1e-6);zids=ids.cuda();goal=target.cuda()
        for _ in range(600):
            pred=dec(zids);loss=(pred-goal).square().mean();opt.zero_grad();loss.backward();opt.step()
        # Assignment refinement against the learned functional package decoder.
        with torch.no_grad():ptable=dec(torch.arange(k,device='cuda')).cpu();dist=target.square().sum(1,keepdim=True)+ptable.square().sum(1)-2*target@ptable.T;ids=dist.argmin(1);zids=ids.cuda()
        x=acts['gate_proj'].cuda();og,ou,od=(orig[n].cuda() for n in MLP)
        for _ in range(100):
            pred=dec(zids)*norm.cuda()[None,:];g,u,d=pred[:,:896],pred[:,896:1792],pred[:,1792:].T;yg,yu=x@og.T,x@ou.T;ygq,yuq=x@g.T,x@u.T;om=torch.nn.functional.silu(yg)*yu;qm=torch.nn.functional.silu(ygq)*yuq;oo=om@od.T;qo=qm@d.T
            loss=.1*(pred-goal).square().mean()+.5*(ygq-yg).square().mean()/yg.square().mean()+.5*(yuq-yu).square().mean()/yu.square().mean()+2*(qm-om).square().mean()/om.square().mean()+3*(qo-oo).square().mean()/oo.square().mean();opt.zero_grad();loss.backward();opt.step()
        state=OrderedDict((n,qtensor(v)) for n,v in dec.state_dict().items());dec.load_state_dict({n:q.float()*float(s) for n,(q,s) in state.items()});with_ids=dec(ids.cuda()).cpu()*norm[None,:];res={'gate_proj':orig['gate_proj'].float()-with_ids[:,:896].reshape_as(orig['gate_proj']),'up_proj':orig['up_proj'].float()-with_ids[:,896:1792].reshape_as(orig['up_proj']),'down_proj':orig['down_proj'].float()-with_ids[:,1792:].reshape_as(orig['down_proj'].T).T};path=a.output/f'joint_package_{bits}bit.bin';write_codec(path,bits,att_blob,ids,norm,state,res,total);recon,meta=decode_codec(path);base.apply(model,recon);ppl=base.perplexity(model,evaluation,a.sequence_length);ge,ue,ie,re=package_metrics(orig,recon,acts);bpw=path.stat().st_size*8/total;params=sum(t[0].numel() for t in state.values());dbytes=sum(t[0].numel()+2 for t in state.values());row={'bits':bits,'bpw':bpw,'ppl':ppl['ppl'],'delta_ppl_pct':100*(ppl['ppl']/ref['ppl']-1),'gate_error':math.sqrt(ge),'up_error':math.sqrt(ue),'gate_up_interaction_error':math.sqrt(ie),'down_residual_stream_error':math.sqrt(re),'decoder_parameter_count':params,'decoder_storage_bpw':dbytes*8/total,'package_id_bpw':bits*len(ids)/total,'attention_bpw':len(att_blob)*8/total,'mlp_residual_bpw':sum(r.numel() for r in res.values())/total,'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};results.append(row);print(json.dumps(row,indent=2),flush=True)
    aw,_=base.decode(a.attention);base.apply(model,{**aw,**{n:orig[n] for n in MLP}});oracle=base.perplexity(model,evaluation,a.sequence_length);report={'model':base.MODEL,'layer':0,'reference':ref,'attention_plus_original_mlp':oracle,'results':results,'max_bpw':1.2,'elapsed_seconds':time.time()-start};base.dump(a.output/'results.json',report);print('FINAL\n'+json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
