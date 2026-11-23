import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wavefly.core import *
torch.manual_seed(17)
for b in [8,16,32,64]:
    for method in ['fixed','topk']:
        for bits in [2,4,8,16]:
            w=torch.randn(3,b,b,device='cuda');d=basis(b);c=d@w@d.T;k=b*b//8;idx=select(c,k,method)
            v=c.flatten(1).gather(1,idx);q,_,_=quant(v,bits)
            h=d.T@torch.zeros(3,b*b,device='cuda').scatter_(1,idx,q).reshape_as(w)@d
            ri=(w-h).flatten(1).abs().topk(3,dim=1).indices;rv=(w-h).flatten(1).gather(1,ri).half().float()
            h.flatten(1).scatter_add_(1,ri,rv)
            data=serialize(v,idx,b,bits,method,ri,rv)
            assert len(data)==storage(3,b,k,bits,method,3)['total_bytes']
            assert torch.equal(h,deserialize(data)),(b,method,bits)
print('PASS: 32 codec configurations, exact decoded agreement and byte accounting')
