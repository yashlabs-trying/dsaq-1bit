import sys,torch
_DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wavefly.wavelet import *
for n in [8,16,32]:
 for levels in range(1,min(3,int(torch.log2(torch.tensor(n))))+1):
  for dtype,tol in [(torch.float64,1e-12),(torch.float32,1e-5)]:
   h=haar_basis(n,levels,dtype);w=torch.randn(5,n,n,device=_DEV,dtype=dtype);c=transform2(w,h);r=inverse2(c,h)
   assert (h.double()@h.double().T-torch.eye(n,device=_DEV)).abs().max()<tol
   assert (w-r).abs().max()<tol
   bi,names=band_index(n,levels);assert len(torch.unique(bi))==1+3*levels
print('PASS: Haar correctness for B=8/16/32, L=1/2/3, float64/float32')
