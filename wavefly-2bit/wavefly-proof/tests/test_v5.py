"""Numerical invariants for the new allocation and correction paths."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
import v5_experiment as v5
torch.manual_seed(7)
device=_DEV
v=torch.randn(64,32,device=device)*.03
weight=torch.rand_like(v)
opts,sbits=v5.scalar_options(v,weight,True)
for avg in [1,1.25,1.5,2,3]:
    q,bits,metadata=v5.adaptive(opts,v,avg)
    assert bits<=avg*v.numel(),(bits,avg)
    assert q.shape==v.shape and torch.isfinite(q).all()
    assert metadata==2*len(v)
# Two orthogonal stored paths can exactly reconstruct a known residual.
x=torch.eye(2,device=device); w=torch.eye(2,device=device); base=torch.zeros_like(w)
atoms=[torch.diag(torch.tensor([1.,0.],device=device)),torch.diag(torch.tensor([0.,1.],device=device))]
e,mask=v5.oracle(x,w,base,atoms,2)
assert e==0 and int(mask.sum(1).max())<=2
for k in [1,2]:
    e,mask=v5.oracle(x,w,base,atoms,k)
    assert e<=1 and int(mask.sum(1).max())<=k
q,bits=v5.quant_factor(v)
assert bits==v5.aligned(v.numel()*8)+16 and torch.isfinite(q).all()
print('V5 invariant checks PASS')
