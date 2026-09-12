"""Additive (multi-codebook) vector quantization for low-bit weights.

Each row of a weight matrix is split into blocks of `g` consecutive input
features. Every block is approximated as the sum of M codebook vectors, one
picked from each of M shared codebooks of size K. Storage is M*log2(K)/g bits
per weight (2 bits for M=2, K=256, g=8) plus a tiny shared codebook.
"""
import math
import torch


def _assign(X, centers):
    """Squared distances and nearest-center index for each block in X."""
    xn = (X ** 2).sum(-1, keepdim=True)
    cn = (centers ** 2).sum(-1)
    d2 = xn - 2 * X @ centers.t() + cn
    return d2.argmin(1)


def _kmeans(X, K, iters, seed):
    N = X.shape[0]
    torch.manual_seed(seed)
    init = torch.randint(0, N, (K,))
    centers = X[init].clone()
    for _ in range(iters):
        a = _assign(X, centers)
        A = torch.nn.functional.one_hot(a, K).float()
        cnt = A.sum(0)
        mask = cnt > 0
        centers[mask] = (A[:, mask].t() @ X) / cnt[mask].unsqueeze(1)
        if not mask.all():
            empty = (~mask).nonzero().view(-1)
            re = torch.randint(0, N, (empty.numel(),))
            centers[empty] = X[re]
    return centers


def _assign_beam2(X, C0, C1, beam):
    """Joint beam assignment for M=2: best (k0, k1) per block."""
    xn = (X ** 2).sum(-1, keepdim=True)
    d0 = xn - 2 * X @ C0.t() + (C0 ** 2).sum(-1)  # [N, K]
    top0 = torch.topk(d0, beam, dim=1, largest=False)  # [N, beam]
    c1n = (C1 ** 2).sum(-1)
    best_val = torch.full((X.shape[0],), float('inf'), device=X.device)
    best_i0 = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    best_i1 = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for b in range(beam):
        k0 = top0.indices[:, b]
        r = X - C0[k0]
        d1 = (r ** 2).sum(-1, keepdim=True) - 2 * r @ C1.t() + c1n
        v, k1 = d1.min(1)
        better = v < best_val
        best_val = torch.where(better, v, best_val)
        best_i0 = torch.where(better, k0, best_i0)
        best_i1 = torch.where(better, k1, best_i1)
    return best_i0, best_i1


def additive_vq(X, M, K, init_iters=10, iters=20, beam=1, seed=0):
    """Train M shared codebooks (each K entries of dim g) and per-block indices.

    X: [N, g] float32. Returns codebooks [M, K, g] and indices [N, M] (long).
    beam=1 is greedy; beam>1 enables joint beam assignment (M=2 only).
    """
    N = X.shape[0]
    dev = X.device
    codebooks = []
    idx = torch.zeros(N, M, dtype=torch.long, device=dev)
    resid = X.clone()
    for m in range(M):
        cb = _kmeans(resid, K, init_iters, seed + m)
        codebooks.append(cb)
        a = _assign(resid, cb)
        idx[:, m] = a
        resid = resid - cb[a]

    for _ in range(iters):
        if M == 2 and beam > 1:
            i0, i1 = _assign_beam2(X, codebooks[0], codebooks[1], beam)
            idx[:, 0] = i0
            idx[:, 1] = i1
        else:
            for m in range(M):
                rec = torch.zeros_like(X)
                for m2 in range(M):
                    if m2 != m:
                        rec += codebooks[m2][idx[:, m2]]
                idx[:, m] = _assign(X - rec, codebooks[m])
        for m in range(M):
            rec = torch.zeros_like(X)
            for m2 in range(M):
                if m2 != m:
                    rec += codebooks[m2][idx[:, m2]]
            r = X - rec
            A = torch.nn.functional.one_hot(idx[:, m], K).float()
            cnt = A.sum(0).clamp_min(1)
            codebooks[m] = (A.t() @ r) / cnt.unsqueeze(1)

    return torch.stack(codebooks), idx


def decode(codebooks, idx, shape):
    """Reconstruct a flat tensor from codebooks [M, K, g] and indices [N, M]."""
    rec = torch.zeros(idx.shape[0], codebooks.shape[-1], device=codebooks.device)
    for m in range(codebooks.shape[0]):
        rec += codebooks[m][idx[:, m]]
    return rec.reshape(shape)


def payload_bytes(idx, codebooks):
    """Storage estimate: packed indices at log2(K) bits plus FP16 codebooks."""
    M, K, g = codebooks.shape
    idx_bits = idx.numel() * int(math.log2(K))
    cb_bits = codebooks.numel() * 16
    return (idx_bits + cb_bits) / 8
