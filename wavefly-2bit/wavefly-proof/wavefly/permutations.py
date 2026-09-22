import math
import torch

def inverse_permutation(p):
    inv=torch.empty_like(p);inv[p]=torch.arange(len(p),device=p.device)
    return inv

def apply_permutation(w,rows=None,cols=None):
    if rows is not None:w=w[rows]
    if cols is not None:w=w[:,cols]
    return w

def undo_permutation(w,rows=None,cols=None):
    if cols is not None:w=w[:,inverse_permutation(cols)]
    if rows is not None:w=w[inverse_permutation(rows)]
    return w

def permutation_bytes(rows,cols,macro=None):
    total=0
    for p in (rows,cols):
        if p is None:continue
        domain=min(len(p),macro) if macro else len(p)
        total+=math.ceil(len(p)*math.ceil(math.log2(max(2,domain)))/8)
    return total

def verify(w,rows=None,cols=None):
    h=undo_permutation(apply_permutation(w,rows,cols),rows,cols)
    return torch.equal(w,h),float((w-h).abs().max())
