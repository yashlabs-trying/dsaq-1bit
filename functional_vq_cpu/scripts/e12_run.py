"""E12: Projection-specialized codec - clean implementation."""
import numpy as np, torch, json, struct, time, hashlib
from pathlib import Path
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from cpu_codec import blockify, read_codec, unblock

ROOT = Path(__file__).resolve().parents[1]
E9 = torch.load(ROOT/"cache/e9_layer_cache.pt", map_location="cpu", weights_only=True)
PAR = ROOT/"packed/E10a_gate_up_downstream.packed"
RM, ST, CK, OC, OI_np, SI_np, H, RID, RQ, RS = read_codec(PAR)
def to_t(x):
    return x.clone() if isinstance(x, torch.Tensor) else torch.from_numpy(x.copy())
OI = to_t(OI_np).long()
SI = to_t(SI_np).long()
SL = {n: tuple(RM["slices"][n]) for n in RM["names"]}
ORD = ["down_proj","gate_proj","up_proj","o_proj","v_proj","q_proj","k_proj"]
NP = [1408, 1024, 1024, 256, 160, 128, 96]

def km(d, k, seed=42):
    n,_=d.shape; rng=np.random.RandomState(seed)
    idx=rng.choice(n,k,0); c=d[idx].copy(); p=None
    for _ in range(30):
        ds=np.zeros((n,k))
        for i in range(k): df=d-c[i]; ds[:,i]=np.sum(df*df,1)
        l=ds.argmin(1)
        if p is not None and np.array_equal(l,p): break
        p=l
        for i in range(k):
            m=l==i
            if m.sum(): c[i]=d[m].mean(0)
    return c,l

t0=time.time()
x=E9["x_b"]; h1=E9["h1_b"]; h2t=E9["h2_teacher_b"]
pb={}
for n in ORD:
    s,e,_,_=SL[n]
    pb[n]=blockify(E9["original_weights"][n]).numpy().astype(np.float32)
tn=sum(b.shape[0] for b in pb.values())
print(json.dumps({"event":"collect","blocks":tn}))

# Per-proj k-means
ap=[]; po={}; off=0
for ni,n in enumerate(ORD):
    blk=pb[n]; nb=blk.shape[0]; k=NP[ni]
    sub=blk[np.random.RandomState(42).choice(nb,min(10000,nb),0)]
    c,_=km(sub,min(k,sub.shape[0]),42+ni)
    if c.shape[0]<k: c=np.vstack([c,np.tile(c[:1],(k-c.shape[0],1))])
    ap.append(c[:k]); po[n]=(off,off+k); off+=k
    print(json.dumps({"event":"kmeans","proj":n,"k":k}))

fp=np.vstack(ap).astype(np.float32)
lv=np.sort(np.percentile(fp.reshape(-1),np.linspace(0,100,64)))
qc=np.abs(fp.reshape(-1,1)-lv.reshape(1,-1)).argmin(1).reshape(4096,16).astype(np.uint8)
qp=lv[qc].astype(np.float32)

# Assign
fi=np.zeros(tn,dtype=np.int32); oi=0
for ni,n in enumerate(ORD):
    blk=pb[n]; nb=blk.shape[0]; ps,pe=po[n]
    qpt=torch.from_numpy(qp[ps:pe]); bt=torch.from_numpy(blk)
    fi[oi:oi+nb]=(torch.cdist(bt,qpt).argmin(1).numpy()+ps).astype(np.int32)
    oi+=nb
    print(json.dumps({"event":"assign","proj":n,"blocks":nb}))

# Use original scale table + residuals unchanged
mi=torch.empty(tn,dtype=torch.long)
for idx,n in enumerate(RM["names"]):
    s,e,_,_=SL[n]; mi[s:e]=idx
sf=ST.float()[mi,SI.clamp(0,15)]
resid=torch.zeros(tn,16)
if H is not None:
    resid[H]=RQ.float()[RID.long()]*RS.float()[RID.long(),None]
dec=torch.from_numpy(qp[fi]).float()*sf[:,None]+resid

weights={}
for n in ORD:
    s,e,r,c=SL[n]
    weights[n]=unblock(dec[s:e].float(),(r,c))

# Write packed (no-frills, just patterns+scale+residual from original)
def pk(v,b):
    v=np.asarray(v,dtype=np.uint16).reshape(-1)
    m=((v[:,None]>>np.arange(b,dtype=np.uint16))&1).astype(np.uint8)
    return np.packbits(m.reshape(-1),bitorder="little").tobytes()

meta = {"nblocks":tn,"names":ORD,
        "slices":{n:list(SL[n]) for n in ORD},
        "checkpoint_shape":[1,64],"pattern_shape":[4096,16],
        "pattern_bits":6,"pattern_count":4096,
        "hard_count":0,"residual_bits":8,"residual_k":0,
        "weights":tn*16*4}
mb=json.dumps(meta).encode()
OUT=ROOT/"packed/E12_specialized.packed"
with open(OUT,"wb") as f:
    f.write(struct.pack("<I",len(mb))); f.write(mb)
    rec=(fi.astype(np.uint16)&0x0FFF)|((np.zeros(tn,dtype=np.uint16)&0x000F)<<12)
    f.write(rec.astype("<u2").tobytes())
    f.write(np.zeros((7,16),dtype=np.float16).astype("<f2").tobytes())
    f.write(lv.astype(np.float16).tobytes())
    f.write(pk(qc,6))

sz=OUT.stat().st_size; bpw=sz*8/meta["weights"]
print(json.dumps({"event":"built","bytes":sz,"bpw":bpw}))

# Evaluate
print(json.dumps({"event":"evaluate"}))
mod=AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",revision="060db6499f32faf8b98477b0a26969ef7d8b9987",
    dtype=torch.float32,low_cpu_mem_usage=True,attn_implementation="eager").eval()

with torch.no_grad():
    g=x@weights["gate_proj"].T; u=x@weights["up_proj"].T
    h=h1+(F.silu(g)*u)@weights["down_proj"].T
    for i in (1,2):
        ly=mod.model.layers[i]; slen=h.shape[0]
        p=torch.arange(slen).unsqueeze(0)
        m=torch.triu(torch.full((slen,slen),torch.finfo(h.dtype).min),1).unsqueeze(0).unsqueeze(0)
        cs,si=mod.model.rotary_emb(h.unsqueeze(0),p)
        h=h+ly.self_attn(ly.input_layernorm(h).unsqueeze(0),
                          attention_mask=m,position_embeddings=(cs,si))[0].squeeze(0)
        ha=h; h=ly.post_attention_layernorm(h); h=ha+ly.mlp(h)
    h4=h
    h4t=h2t
    for i in (1,2):
        ly=mod.model.layers[i]; slen=h4t.shape[0]
        p=torch.arange(slen).unsqueeze(0)
        m=torch.triu(torch.full((slen,slen),torch.finfo(h4t.dtype).min),1).unsqueeze(0).unsqueeze(0)
        cs,si=mod.model.rotary_emb(h4t.unsqueeze(0),p)
        h4t=h4t+ly.self_attn(ly.input_layernorm(h4t).unsqueeze(0),
                              attention_mask=m,position_embeddings=(cs,si))[0].squeeze(0)
        ha=h4t; h4t=ly.post_attention_layernorm(h4t); h4t=ha+ly.mlp(h4t)
lb=float((h4-h4t).square().sum()/h4t.square().sum())
print(json.dumps({"event":"result","loss_B":lb}))
