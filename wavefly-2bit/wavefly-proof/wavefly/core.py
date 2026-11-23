import math, struct
import numpy as np
import torch

def basis(n, dtype=torch.float32, device='cuda'):
    k=torch.arange(n,dtype=torch.float64,device=device)[:,None]
    j=torch.arange(n,dtype=torch.float64,device=device)[None,:]
    d=torch.cos(math.pi/n*(j+.5)*k)*math.sqrt(2/n)
    d[0]*=1/math.sqrt(2)
    return d.to(dtype)

def metrics(w, h, x):
    w,h,x=w.double(),h.double(),x.double()
    e=w-h; y=x@w.transpose(-1,-2); yh=x@h.transpose(-1,-2)
    return dict(max_abs=e.abs().max().item(), weight_error=(e.norm()/w.norm().clamp_min(1e-30)).item(),
                cosine=torch.nn.functional.cosine_similarity(w.flatten(),h.flatten(),dim=0).item(),
                output_error=((y-yh).norm()/y.norm().clamp_min(1e-30)).item())

def quant(v,bits):
    if bits==16: return v.half().float(), None, None
    # Per-tile symmetric mid-rise levels, including four levels for two bits.
    qmax=(2**bits-1)/2
    scale=(v.abs().amax(-1,keepdim=True)/qmax).clamp_min(6e-8).half().float()
    code=torch.round(v/scale+qmax).clamp(0,2**bits-1).to(torch.uint8)
    return (code.float()-qmax)*scale, code, scale

def select(c,k,method):
    t,b,_=c.shape
    if method=='topk': return c.flatten(1).abs().topk(k,dim=1).indices
    # Exact-area rectangle anchored at DC; choose most square factorization.
    pairs=[(h,k//h) for h in range(1,b+1) if k%h==0 and k//h<=b]
    h,w=min(pairs,key=lambda p:abs(p[0]-p[1]))
    return (torch.arange(h,device=c.device)[:,None]*b+torch.arange(w,device=c.device)[None,:]).flatten()[None,:].expand(t,-1)

def pack_codes(a,bits):
    a=np.asarray(a,dtype=np.uint64).ravel()
    return np.packbits(((a[:,None]>>np.arange(bits,dtype=np.uint64))&1).astype('uint8').ravel(),bitorder='little').tobytes()

def unpack_codes(data,count,bits):
    z=np.unpackbits(np.frombuffer(data,dtype=np.uint8),bitorder='little')[:count*bits].reshape(count,bits)
    return (z.astype(np.uint64)*(1<<np.arange(bits,dtype=np.uint64))).sum(1)

def storage(t,b,k,bits,method,r):
    # One 32-byte tensor header. Each tile has independently byte-aligned streams.
    idx=math.ceil(math.log2(b*b))
    parts=dict(coefficient_bytes=t*math.ceil(k*bits/8),index_bytes=t*math.ceil(k*idx/8) if method=='topk' else 0,
               residual_bytes=t*r*2,residual_index_bytes=t*math.ceil(r*idx/8),scale_bytes=t*2 if bits!=16 else 0,metadata_bytes=32)
    parts['total_bytes']=sum(parts.values()); parts['bpw']=8*parts['total_bytes']/(t*b*b)
    return parts

def serialize(c,indices,b,bits,method,ridx,rval):
    t,k=c.shape; r=ridx.shape[1]; _,codes,scales=quant(c,bits)
    data=bytearray(struct.pack('<8I',0x57415645,t,b,k,bits,int(method=='topk'),r,1))
    ib=math.ceil(math.log2(b*b))
    for i in range(t):
        data.extend(c[i].half().cpu().numpy().astype('<f2').tobytes() if bits==16 else pack_codes(codes[i].cpu().numpy(),bits))
        if method=='topk': data.extend(pack_codes(indices[i].cpu().numpy(),ib))
        data.extend(rval[i].half().cpu().numpy().astype('<f2').tobytes())
        data.extend(pack_codes(ridx[i].cpu().numpy(),ib))
        if bits!=16: data.extend(scales[i].half().cpu().numpy().astype('<f2').tobytes())
    assert len(data)==storage(t,b,k,bits,method,r)['total_bytes']
    return bytes(data)

def deserialize(data,device='cuda'):
    _,t,b,k,bits,top,r,_=struct.unpack('<8I',data[:32]); off=32; ib=math.ceil(math.log2(b*b))
    cs=[]; ids=[]; rs=[]; ri=[]
    for i in range(t):
        n=math.ceil(k*bits/8); raw=data[off:off+n]; off+=n
        c=np.frombuffer(raw,dtype='<f2').copy().astype('float32') if bits==16 else unpack_codes(raw,k,bits).astype('float32')
        if top:
            n=math.ceil(k*ib/8); ix=unpack_codes(data[off:off+n],k,ib); off+=n
        else: ix=select(torch.zeros(1,b,b),k,'fixed')[0].numpy()
        rv=np.frombuffer(data[off:off+2*r],dtype='<f2').copy(); off+=2*r
        n=math.ceil(r*ib/8); rx=unpack_codes(data[off:off+n],r,ib); off+=n
        if bits!=16:
            sc=np.frombuffer(data[off:off+2],dtype='<f2')[0]; off+=2
            c=(c-(2**bits-1)/2)*sc
        cs.append(c);ids.append(ix);rs.append(rv);ri.append(rx)
    assert off==len(data)
    c=torch.zeros(t,b*b,device=device).scatter_(1,torch.tensor(np.array(ids).astype('int64'),device=device),torch.tensor(np.array(cs),device=device))
    d=basis(b,device=device); w=d.T@c.reshape(t,b,b)@d
    w.flatten(1).scatter_add_(1,torch.tensor(np.array(ri).astype('int64'),device=device),torch.tensor(np.array(rs).astype('float32'),device=device))
    return w
