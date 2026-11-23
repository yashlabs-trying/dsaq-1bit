import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import torch
_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
import pytest
if not torch.cuda.is_available():
    pytest.skip('V6 experiment driver requires CUDA', allow_module_level=True)
import v6_experiment as v6
torch.manual_seed(9)
w=torch.randn(8,32,device=_DEV)*.02; x=torch.randn(17,32,device=_DEV)
perm=torch.randperm(32,device=_DEV); p=v6.pack_groups(w,x,perm)
assert torch.equal(v6.restore(p['z'],p),w)
e=torch.randn_like(p['z'])*.01
actual=(x[:,perm]@e.T).square().sum()/len(x)
assert torch.allclose(v6.functional_loss(e,p).sum(),actual,rtol=2e-5,atol=1e-7)
opts,st=v6.options(p,torch.tensor([.5,.75,1.,1.5],device=_DEV),2,'functional')
assert st['micro_scales']==len(p['z'])*2
assert st['zero_points']==len(p['z'])*2
for avg in [1,1.5,2,3,4]:
    q,bits,meta=v6.allocate_groups(opts,p,avg,True)
    assert bits<=avg*p['z'].numel() and meta==2*len(p['z'])
    assert torch.isfinite(q).all()
h=v6.restore(v6.allocate_groups(opts,p,1.5,True)[0],p)
row=v6.record('test',w,[x,x,x],h,'control','test',st)
v6.residual_sweep('test',w,[x,x,x],h,row,'residual',('functional',),False)
assert all(r['bits']==sum(r.get(k,0) for k in v6.FIELDS) for r in v6.ROWS)
assert all(r['bpw']==r['bits']/w.numel() for r in v6.ROWS)
print('V6 covariance, permutation, precision-budget, and storage checks PASS')
