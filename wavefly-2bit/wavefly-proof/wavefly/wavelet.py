import math
import torch
_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def haar_basis(n,levels,dtype=torch.float32,device=_DEV):
    """Packed orthonormal Haar analysis matrix; recursively transforms the low half."""
    if n&(n-1) or levels<1 or levels>int(math.log2(n)):raise ValueError((n,levels))
    h=torch.eye(n,dtype=torch.float64,device=device);m=n;s=math.sqrt(.5)
    for _ in range(levels):
        stage=torch.eye(n,dtype=torch.float64,device=device)
        block=torch.zeros(m,m,dtype=torch.float64,device=device)
        j=torch.arange(m//2,device=device);block[j,2*j]=s;block[j,2*j+1]=s
        block[m//2+j,2*j]=s;block[m//2+j,2*j+1]=-s
        stage[:m,:m]=block;h=stage@h;m//=2
    return h.to(dtype)

def band_slices(n,levels):
    """Non-overlapping packed 2D bands after a multilevel transform."""
    bands=[];m=n
    for level in range(1,levels+1):
        q=m//2
        bands += [(f'LH{level}',(slice(0,q),slice(q,m))),
                  (f'HL{level}',(slice(q,m),slice(0,q))),
                  (f'HH{level}',(slice(q,m),slice(q,m)))]
        m=q
    bands.insert(0,(f'LL{levels}',(slice(0,m),slice(0,m))))
    return bands

def band_index(n,levels,device=_DEV):
    z=torch.empty(n,n,dtype=torch.long,device=device);names=[]
    for i,(name,(a,b)) in enumerate(band_slices(n,levels)):z[a,b]=i;names.append(name)
    return z,names

def transform2(w,h):return h@w@h.T
def inverse2(c,h):return h.T@c@h
