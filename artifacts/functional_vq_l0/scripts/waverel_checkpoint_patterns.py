"""Layer-0 WaveRel-guided checkpoint-pattern experiment.

The output codecs are real bit-packed files.  BPW is their actual byte size
(including the JSON header) divided by the number of Layer-0 linear weights.
Every candidate is decoded from disk and compared bit-for-bit with its in-memory
quantized reconstruction before perplexity is evaluated.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import weighted_vq as legacy  # noqa: E402


MODEL = "Qwen/Qwen2.5-0.5B"
SEED = 20261007
FAMILIES = 16
PATTERNS_PER_FAMILY = 256
PATTERNS = FAMILIES * PATTERNS_PER_FAMILY
LEVELS = 64
PATTERN_BITS = 6
SCALE_LEVELS = 16
BLOCK = 4
PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("/workspace/waverel_patterns"))
    p.add_argument("--strict-vq", type=Path,
                   default=Path("/workspace/centroid_refine/centroid_refined_checkpoint.pt"))
    p.add_argument("--sequence-length", type=int, default=256)
    p.add_argument("--fit-tokens", type=int, default=2048)
    p.add_argument("--heldout-tokens", type=int, default=1024)
    p.add_argument("--eval-tokens", type=int, default=4096)
    p.add_argument("--hard-fraction", type=float, default=0.20)
    return p.parse_args()


def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def blocks(w):
    r, c = w.shape
    assert r % 4 == 0 and c % 4 == 0
    return w.view(r//4, 4, c//4, 4).permute(0, 2, 1, 3).reshape(-1, 16)


def unblock(x, shape):
    r, c = shape
    return x.view(r//4, c//4, 4, 4).permute(0, 2, 1, 3).reshape(r, c)


def module_map(model):
    layer = model.model.layers[0]
    out = OrderedDict()
    for family in PROJS:
        parent = layer.self_attn if family in PROJS[:4] else layer.mlp
        out[family] = getattr(parent, family)
    return out


def fixed_tokens(tokenizer, split, count, offset=0):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    return ids[offset:offset+count].clone()


def extract(originals, fisher):
    vecs, imp, matrix_index, slices = [], [], [], {}
    cursor = 0
    for mi, (name, weight) in enumerate(originals.items()):
        vb, ib = blocks(weight.float()), blocks(fisher[name].float())
        slices[name] = (cursor, cursor+vb.shape[0], tuple(weight.shape))
        cursor += vb.shape[0]; vecs.append(vb); imp.append(ib)
        matrix_index.append(torch.full((vb.shape[0],), mi, dtype=torch.int16))
    return torch.cat(vecs), torch.cat(imp), torch.cat(matrix_index), slices


def kmeans_1d(values, k=16, iters=30):
    x = values.float().flatten()
    if x.numel() > 2_000_000:
        x = x[torch.randperm(x.numel())[:2_000_000]]
    centers = torch.quantile(x, torch.linspace(0, 1, k)).unique()
    if centers.numel() < k:
        centers = torch.linspace(float(x.min()), float(x.max()), k)
    centers = centers[:k].clone()
    for _ in range(iters):
        ids = (x[:, None]-centers[None, :]).abs().argmin(1)
        new = centers.clone()
        for j in range(k):
            if (ids==j).any(): new[j] = x[ids==j].mean()
        if torch.max(torch.abs(new-centers)) < 1e-7: break
        centers = new
    return centers.sort().values


def quantize_scales(raw_scales, matrix_index, matrices):
    tables = torch.empty(matrices, SCALE_LEVELS)
    ids = torch.empty(raw_scales.numel(), dtype=torch.uint8)
    decoded = torch.empty_like(raw_scales)
    for mi in range(matrices):
        mask = matrix_index == mi
        logv = raw_scales[mask].clamp_min(1e-12).log()
        levels = kmeans_1d(logv, SCALE_LEVELS).exp()
        tables[mi] = levels
        local = (logv[:, None]-levels.log()[None, :]).abs().argmin(1)
        ids[mask] = local.byte()
    stored = tables.half()
    decoded = stored.float()[matrix_index.long(), ids.long()]
    return stored, ids, decoded


def haar4(x):
    # Orthonormal 4-point Haar transform, applied on both axes.
    h = torch.tensor([[.5,.5,.5,.5], [.5,.5,-.5,-.5],
                      [2**-.5,-2**-.5,0,0], [0,0,2**-.5,-2**-.5]])
    z = x.view(-1,4,4)
    return (h @ z @ h.T).reshape(-1,16)


def waverel_features(z):
    m = z.abs(); grid = z.view(-1,4,4); mag = m.view(-1,4,4)
    sign = z.sign()
    sorted_mag = m.sort(1).values
    row = mag.mean(2); col = mag.mean(1)
    hg = (grid[:,:,1:]-grid[:,:,:-1]).reshape(-1,12)
    vg = (grid[:,1:,:]-grid[:,:-1,:]).reshape(-1,12)
    return torch.cat([z, sign, sorted_mag, row, col, haar4(z), hg, vg], 1)


@torch.inference_mode()
def nearest(data, centers, batch=4096):
    out=[]; centers=centers.cuda(); cn=centers.square().sum(1)
    for s in range(0,data.shape[0],batch):
        x=data[s:s+batch].cuda(); d=x.square().sum(1,keepdim=True)+cn-2*x@centers.T
        out.append(d.argmin(1).cpu())
    return torch.cat(out)


def quantize_patterns(centers, family_for_pattern, normalized, family_for_block):
    checkpoint = torch.empty(FAMILIES, LEVELS)
    codes = torch.empty(centers.shape, dtype=torch.uint8)
    decoded = torch.empty_like(centers)
    for f in range(FAMILIES):
        vals = normalized[family_for_block==f]
        levels = kmeans_1d(vals, LEVELS)
        checkpoint[f] = levels
        pmask = family_for_pattern == f
        local = (centers[pmask,:,None]-levels[None,None,:]).abs().argmin(2)
        codes[pmask] = local.byte()
    stored = checkpoint.half()
    for f in range(FAMILIES):
        pmask = family_for_pattern == f
        decoded[pmask] = stored[f].float()[codes[pmask].long()]
    return stored, codes, decoded


def fisher_assign(target, imp, patterns, batch=2048, topk=64):
    # Functional assignment: shortlist geometrically, then Fisher re-rank.
    ids=[]; p=patterns.cuda(); pn=p.square().sum(1)
    for s in range(0,target.shape[0],batch):
        x=target[s:s+batch].cuda(); h=imp[s:s+batch].cuda()
        d=x.square().sum(1,keepdim=True)+pn-2*x@p.T
        cand=torch.topk(d,min(topk,p.shape[0]),largest=False).indices
        cp=p[cand]; score=((x[:,None,:]-cp).square()*h[:,None,:]).sum(2)
        best=score.argmin(1); ids.append(cand[torch.arange(x.shape[0],device='cuda'),best].cpu())
    return torch.cat(ids).to(torch.int32)


def train_global_patterns(z, target, imp):
    sample=z[torch.randperm(z.shape[0])[:min(600_000,z.shape[0])]]
    centers,_=legacy.fast_kmeans(sample,PATTERNS,n_iter=20,batch=4096)
    # One checkpoint family: duplicate its levels only for a uniform decoder shape.
    levels=kmeans_1d(z,LEVELS).half().float()
    codes=(centers[:,:,None]-levels[None,None,:]).abs().argmin(2).byte()
    decoded=levels[codes.long()]
    ids=fisher_assign(target,imp,decoded)
    return levels[None,:].half(),codes,decoded,ids,torch.zeros(PATTERNS,dtype=torch.int16)


def train_waverel_patterns(z, target, imp):
    feat=waverel_features(z)
    sample_idx=torch.randperm(z.shape[0])[:min(150_000,z.shape[0])]
    fcent,_=legacy.fast_kmeans(feat[sample_idx],FAMILIES,n_iter=12,batch=4096)
    family=nearest(feat,fcent,batch=8192)
    centers=[]; pattern_family=[]
    for f in range(FAMILIES):
        data=z[family==f]
        if data.shape[0] < PATTERNS_PER_FAMILY:
            data=z[torch.randperm(z.shape[0])[:PATTERNS_PER_FAMILY]]
        sample=data[torch.randperm(data.shape[0])[:min(100_000,data.shape[0])]]
        c,_=legacy.fast_kmeans(sample,PATTERNS_PER_FAMILY,n_iter=15,batch=4096)
        centers.append(c); pattern_family.extend([f]*PATTERNS_PER_FAMILY)
    centers=torch.cat(centers); pattern_family=torch.tensor(pattern_family,dtype=torch.int16)
    checkpoint,codes,decoded=quantize_patterns(centers,pattern_family,z,family)
    ids=fisher_assign(target,imp,decoded)
    return checkpoint,codes,decoded,ids,pattern_family


def pack_nibbles(x):
    a=x.detach().cpu().numpy().astype(np.uint8).reshape(-1)
    if len(a)%2: a=np.concatenate([a,np.zeros(1,dtype=np.uint8)])
    return (a[0::2] | (a[1::2]<<4)).tobytes()


def unpack_nibbles(blob,count):
    a=np.frombuffer(blob,dtype=np.uint8)
    out=np.empty(a.size*2,dtype=np.uint8); out[0::2]=a&15; out[1::2]=a>>4
    return torch.from_numpy(out[:count].copy())


def pack_fixed(x, bits):
    a=x.detach().cpu().numpy().astype(np.uint16).reshape(-1)
    bit_matrix=((a[:,None] >> np.arange(bits,dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1),bitorder='little').tobytes()


def unpack_fixed(blob, count, bits):
    raw=np.unpackbits(np.frombuffer(blob,dtype=np.uint8),bitorder='little')[:count*bits]
    matrix=raw.reshape(count,bits).astype(np.uint16)
    values=(matrix * (1 << np.arange(bits,dtype=np.uint16))[None,:]).sum(1)
    return torch.from_numpy(values.astype(np.uint8))


def write_codec(path, method, slices, scale_tables, scale_ids, checkpoint, pattern_codes,
                pattern_ids, hard_mask=None, residual_ids=None, residual_q=None,
                residual_scale=None, meta_extra=None):
    names=list(slices)
    meta={"version":1,"method":method,"weights":sum(slices[n][2][0]*slices[n][2][1] for n in names),
          "nblocks":int(pattern_ids.numel()),"names":names,
          "slices":{n:[int(slices[n][0]),int(slices[n][1]),*map(int,slices[n][2])] for n in names},
          "checkpoint_shape":list(checkpoint.shape),"pattern_shape":list(pattern_codes.shape),
          "pattern_bits":PATTERN_BITS,"hard_count":0 if hard_mask is None else int(hard_mask.sum()),
          "residual_k":0 if residual_q is None else int(residual_q.shape[0]),
          "residual_bits":0 if residual_q is None else int(math.ceil(math.log2(residual_q.shape[0])))}
    if meta_extra:
        meta.update(meta_extra)
    m=json.dumps(meta,separators=(",",":"),sort_keys=True).encode()
    record=((pattern_ids.numpy().astype(np.uint16)&0x0fff) |
            (scale_ids.numpy().astype(np.uint16)<<12)).astype('<u2').tobytes()
    parts=[struct.pack('<I',len(m)),m,record,scale_tables.numpy().astype('<f2').tobytes(),
           checkpoint.numpy().astype('<f2').tobytes(),pack_fixed(pattern_codes,PATTERN_BITS)]
    if hard_mask is not None:
        parts.append(np.packbits(hard_mask.numpy().astype(np.uint8),bitorder='little').tobytes())
        parts.append(pack_fixed(residual_ids,meta["residual_bits"]))
        parts.append(residual_q.numpy().astype(np.int8).tobytes())
        parts.append(residual_scale.numpy().astype('<f2').tobytes())
    path.write_bytes(b''.join(parts))


def read_codec(path):
    blob=path.read_bytes(); pos=0
    ml=struct.unpack_from('<I',blob,pos)[0]; pos+=4
    meta=json.loads(blob[pos:pos+ml]); pos+=ml; n=meta['nblocks']
    rec=np.frombuffer(blob,dtype='<u2',count=n,offset=pos).copy(); pos+=2*n
    pattern_ids=torch.from_numpy((rec&0xfff).astype(np.int32)); scale_ids=torch.from_numpy((rec>>12).astype(np.uint8))
    nm=len(meta['names']); count=nm*SCALE_LEVELS
    scales=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=count,offset=pos).copy()).view(nm,SCALE_LEVELS); pos+=2*count
    cs=math.prod(meta['checkpoint_shape']); checkpoint=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=cs,offset=pos).copy()).view(meta['checkpoint_shape']); pos+=2*cs
    pc=math.prod(meta['pattern_shape']); pbytes=(pc*meta['pattern_bits']+7)//8; codes=unpack_fixed(blob[pos:pos+pbytes],pc,meta['pattern_bits']).view(meta['pattern_shape']); pos+=pbytes
    hard=None; rid=None; rq=None; rs=None
    if meta['hard_count']:
        mb=(n+7)//8; hard=torch.from_numpy(np.unpackbits(np.frombuffer(blob[pos:pos+mb],dtype=np.uint8),bitorder='little')[:n].copy()).bool(); pos+=mb
        rb=(meta['hard_count']*meta['residual_bits']+7)//8; rid=unpack_fixed(blob[pos:pos+rb],meta['hard_count'],meta['residual_bits']); pos+=rb
        rk=meta['residual_k']; rq=torch.from_numpy(np.frombuffer(blob,dtype=np.int8,count=rk*16,offset=pos).copy()).view(rk,16); pos+=rk*16
        rs=torch.from_numpy(np.frombuffer(blob,dtype='<f2',count=rk,offset=pos).copy()); pos+=2*rk
    assert pos==len(blob),(pos,len(blob))
    return meta,scales,checkpoint,codes,pattern_ids,scale_ids,hard,rid,rq,rs


K_RESIDUAL=256


def decode(path):
    meta,scale_tables,checkpoint,codes,pids,sids,hard,rids,rq,rs=read_codec(path)
    if checkpoint.shape[0]==1:
        patterns=checkpoint[0][codes.long()]
    else:
        fam=(torch.arange(codes.shape[0])//PATTERNS_PER_FAMILY).long()
        patterns=checkpoint[fam[:,None],codes.long()]
    matrix_index=torch.empty(meta['nblocks'],dtype=torch.long)
    for mi,name in enumerate(meta['names']):
        s,e,_,_=meta['slices'][name]; matrix_index[s:e]=mi
    scale=scale_tables[matrix_index,sids.long()]
    out=patterns[pids.long()].float()*scale[:,None].float()
    if hard is not None:
        residual=rq.float()*rs.float()[:,None]
        out[hard]+=residual[rids.long()]
    weights={}
    for name in meta['names']:
        s,e,r,c=meta['slices'][name]; weights[name]=unblock(out[s:e],(r,c))
    return weights,meta


@torch.inference_mode()
def perplexity(model,ids,seq):
    losses=[]
    for s in range(0,ids.numel()-seq+1,seq):
        x=ids[s:s+seq].unsqueeze(0).cuda(); losses.append(float(model(x,labels=x,use_cache=False).loss))
    nll=float(np.mean(losses)); return {"nll":nll,"ppl":math.exp(min(nll,80))}


def apply(model,weights):
    with torch.no_grad():
        for name,value in weights.items(): module_map(model)[name].weight.copy_(value.cuda())


def restore(model,originals): apply(model,originals)


@torch.inference_mode()
def capture_inputs(model,ids,seq):
    values={n:[] for n in module_map(model)}; hooks=[]
    for name,module in module_map(model).items():
        hooks.append(module.register_forward_pre_hook(lambda _m,inp,name=name: values[name].append(inp[0].detach().float().cpu())))
    for s in range(0,ids.numel()-seq+1,seq): model(ids[s:s+seq].unsqueeze(0).cuda(),use_cache=False)
    for h in hooks:h.remove()
    return {n:torch.cat([x.reshape(-1,x.shape[-1]) for x in xs]) for n,xs in values.items()}


@torch.inference_mode()
def activation_metrics(originals,recon,activations):
    rows=[]; total_n=total_d=0.0
    for name in originals:
        x=activations[name].cuda(); w=originals[name].cuda(); q=recon[name].cuda()
        y=x@w.T; e=x@(w-q).T
        num=float(e.square().sum()); den=float(y.square().sum()); total_n+=num; total_d+=den
        rows.append({"family":name,"activation_error":math.sqrt(num/max(den,1e-30))})
    return math.sqrt(total_n/max(total_d,1e-30)),rows


def coverage_curve(error):
    q=torch.quantile(error,torch.tensor([.1,.25,.5,.75,.9]))
    return [{"allowed_error":float(v),"blocks_covered_pct":p} for v,p in zip(q,[10,25,50,75,90])]


def train_hard_residual(vecs,imp,recon,fraction):
    residual=vecs-recon; err=(imp*residual.square()).sum(1)/imp.sum(1).clamp_min(1e-20)
    count=max(1,round(fraction*vecs.shape[0])); hard_ids=torch.topk(err,count).indices
    mask=torch.zeros(vecs.shape[0],dtype=torch.bool); mask[hard_ids]=True
    centers,_=legacy.fast_kmeans(residual[mask],K_RESIDUAL,n_iter=20,batch=4096)
    scale=(centers.abs().amax(1)/127).clamp_min(torch.finfo(torch.float16).tiny).half()
    q=torch.round(centers/scale.float()[:,None]).clamp(-127,127).to(torch.int8)
    deq=q.float()*scale.float()[:,None]
    ids=legacy.weighted_assign(residual[mask],imp[mask],deq,verbose=False).byte()
    final=recon.clone(); final[mask]+=deq[ids.long()]
    return mask,ids,q,scale,final,err


def method_stats(path,model,originals,bf16,eval_ids,seq,activations):
    recon,meta=decode(path); apply(model,recon); metric=perplexity(model,eval_ids,seq)
    act,by_family=activation_metrics(originals,recon,activations)
    return {"method":meta['method'],"bpw":path.stat().st_size*8/meta['weights'],
            "file_bytes":path.stat().st_size,"ppl":metric['ppl'],
            "delta_ppl_pct":100*(metric['ppl']/bf16['ppl']-1),"activation_error":act,
            "hard_block_pct":100*meta['hard_count']/meta['nblocks'],
            "projection_activation_error":by_family}


def strict_vq_baseline(path,outfile,slices):
    s=torch.load(path,map_location='cpu',weights_only=False); names=list(slices)
    base_q=s['base_codebook_q']; base_scale=s['base_codebook_scale']; assign=s['assignments']
    rq=s['residual_codebook_q']; rs=s['residual_codebook_scale']; rp=s['residual_positions'].long(); ri=s['residual_ids']
    # Extract only Layer-0 assignment and residual state, but retain the entire shared table.
    end=max(slices[n][1] for n in names); a=assign[:end].numpy().astype('<u2').tobytes()
    mask=torch.zeros(end,dtype=torch.bool); rid_global=torch.full((end,),-1,dtype=torch.int16)
    inside=rp<end; mask[rp[inside]]=True; rid_global[rp[inside]]=ri[inside].short()
    meta=json.dumps({"version":1,"method":"Strict Fisher VQ","weights":sum(r*c for _,_,(r,c) in slices.values()),
                     "nblocks":end,"hard_count":int(mask.sum()),"note":"Layer-0 standalone cost; full shared K=65536 table retained"},
                    separators=(',',':'),sort_keys=True).encode()
    outfile.write_bytes(struct.pack('<I',len(meta))+meta+a+base_q.numpy().tobytes()+base_scale.numpy().astype('<f2').tobytes()+
                        np.packbits(mask.numpy(),bitorder='little').tobytes()+pack_nibbles(rid_global[mask].byte())+
                        rq.numpy().tobytes()+rs.numpy().astype('<f2').tobytes())
    centers=base_q.float()*base_scale.float()[:,None]; residual=rq.float()*rs.float()[:,None]
    b=centers[assign[:end].long()]; b[mask]+=residual[rid_global[mask].long()]
    recon={}
    for name,(st,en,shape) in slices.items(): recon[name]=unblock(b[st:en],shape)
    return recon,outfile.stat().st_size*8/sum(x.numel() for x in recon.values())


def main():
    a=arguments();a.output.mkdir(parents=True,exist_ok=True);seed_all();started=time.time()
    tokenizer=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32,device_map={"":"cuda"}).eval()
    mods=module_map(model); originals={n:m.weight.detach().cpu().clone() for n,m in mods.items()}
    fit=fixed_tokens(tokenizer,'train',a.fit_tokens); held=fixed_tokens(tokenizer,'train',a.heldout_tokens,a.fit_tokens); evaluation=fixed_tokens(tokenizer,'test',a.eval_tokens)
    restore(model,originals); bf16=perplexity(model,evaluation,a.sequence_length); activations=capture_inputs(model,held,a.sequence_length)
    # Exact recovered Fisher definition, restricted to L0.
    legacy_names={f"L0.{'SA' if n in PROJS[:4] else 'MLP'}.{n}":w for n,w in originals.items()}
    fisher_legacy=legacy.compute_fisher(model,legacy_names)
    model.to('cuda'); fisher={n:fisher_legacy[f"L0.{'SA' if n in PROJS[:4] else 'MLP'}.{n}"].cpu() for n in originals}
    vecs,imp,matrix_index,slices=extract(originals,fisher); total_weights=sum(x.numel() for x in originals.values())
    raw_scale=vecs.abs().amax(1).clamp_min(1e-12); scale_tables,scale_ids,decoded_scale=quantize_scales(raw_scale,matrix_index,len(originals))
    z=vecs/raw_scale[:,None]; target=vecs/decoded_scale[:,None]
    results=[]
    # Current strict baseline: real Layer-0 standalone payload size.
    if a.strict_vq.exists():
        strict_file=a.output/'strict_fisher_vq_layer0.bin'; strict_recon,strict_bpw=strict_vq_baseline(a.strict_vq,strict_file,slices)
        apply(model,strict_recon); sm=perplexity(model,evaluation,a.sequence_length); sa,saf=activation_metrics(originals,strict_recon,activations)
        results.append({"method":"Strict Fisher VQ","bpw":strict_bpw,"file_bytes":strict_file.stat().st_size,
                        "ppl":sm['ppl'],"delta_ppl_pct":100*(sm['ppl']/bf16['ppl']-1),"activation_error":sa,
                        "hard_block_pct":100*json.loads(strict_file.read_bytes()[4:4+struct.unpack_from('<I',strict_file.read_bytes(),0)[0]])['hard_count']/(total_weights/16),
                        "projection_activation_error":saf})
    else:
        print(f'SKIP recovered strict baseline: {a.strict_vq} not present', flush=True)
    # Checkpoint patterns without relationship grouping.
    print('TRAIN global checkpoint patterns',flush=True); cp0,codes0,pats0,ids0,pf0=train_global_patterns(z,target,imp)
    global_file=a.output/'checkpoint_patterns.bin'; write_codec(global_file,'Checkpoint patterns',slices,scale_tables,scale_ids,cp0,codes0,ids0)
    decoded0,_=decode(global_file); expected0={};
    for n,(s,e,shape) in slices.items(): expected0[n]=unblock(pats0[ids0[s:e].long()]*decoded_scale[s:e,None],shape)
    assert all(torch.equal(decoded0[n],expected0[n]) for n in originals)
    results.append(method_stats(global_file,model,originals,bf16,evaluation,a.sequence_length,activations))
    # WaveRel-guided family construction.
    print('TRAIN WaveRel family patterns',flush=True); cp,codes,pats,ids,pfamily=train_waverel_patterns(z,target,imp)
    wave_file=a.output/'waverel_patterns.bin'; write_codec(wave_file,'WaveRel + patterns',slices,scale_tables,scale_ids,cp,codes,ids)
    decoded,_=decode(wave_file); expected={}; base_blocks=pats[ids.long()]*decoded_scale[:,None]
    for n,(s,e,shape) in slices.items(): expected[n]=unblock(base_blocks[s:e],shape)
    assert all(torch.equal(decoded[n],expected[n]) for n in originals)
    results.append(method_stats(wave_file,model,originals,bf16,evaluation,a.sequence_length,activations))
    # Spend the remaining budget on a block mask + 4-bit residual ID.
    mask,rids,rq,rs,hard_blocks,block_error=train_hard_residual(vecs,imp,base_blocks,a.hard_fraction)
    hard_file=a.output/'waverel_patterns_hard.bin'; write_codec(hard_file,'WaveRel + patterns + hard refinement',slices,scale_tables,scale_ids,cp,codes,ids,mask,rids,rq,rs)
    if hard_file.stat().st_size*8/total_weights>1.20: raise RuntimeError('hard refinement exceeds 1.20 BPW')
    decoded_h,_=decode(hard_file); expected_h={}
    for n,(s,e,shape) in slices.items(): expected_h[n]=unblock(hard_blocks[s:e],shape)
    assert all(torch.equal(decoded_h[n],expected_h[n]) for n in originals)
    results.append(method_stats(hard_file,model,originals,bf16,evaluation,a.sequence_length,activations))
    usage=torch.bincount((ids.long()//PATTERNS_PER_FAMILY),minlength=FAMILIES)
    reuse=torch.bincount(ids.long(),minlength=PATTERNS)
    worst=max(results[-1]['projection_activation_error'],key=lambda x:x['activation_error'])
    report={"bf16":bf16,"results":results,"total_weights":total_weights,"blocks":int(vecs.shape[0]),
            "families":FAMILIES,"patterns":PATTERNS,"patterns_per_family":PATTERNS_PER_FAMILY,
            "family_usage":[int(x) for x in usage],"pattern_reuse":{"unused":int((reuse==0).sum()),"median":float(reuse.float().median()),"max":int(reuse.max())},
            "coverage_curve":coverage_curve(block_error),"hard_fraction":a.hard_fraction,
            "scale_cost_bits":int(vecs.shape[0]*4+len(originals)*SCALE_LEVELS*16),
            "wave_pattern_table_bits":int(codes.numel()*PATTERN_BITS+cp.numel()*16),
            "worst_projection_family":worst,"elapsed_seconds":time.time()-started,
            "success":results[-1]['bpw']<=1.20 and results[-1]['delta_ppl_pct']<50,
            "breakthrough":results[-1]['bpw']<=1.20 and results[-1]['delta_ppl_pct']<10}
    dump(a.output/'results.json',report); torch.save({"fit":fit,"heldout":held,"evaluation":evaluation},a.output/'fixed_token_ids.pt')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
