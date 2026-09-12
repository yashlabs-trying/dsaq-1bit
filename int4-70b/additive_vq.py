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


def additive_vq(X, M, K, init_iters=10, iters=20, seed=0):
    """Train M shared codebooks (each K entries of dim g) and per-block indices.

    X: [N, g] float32. Returns codebooks [M, K, g] and indices [N, M] (long).
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
        for m in range(M):
            rec = torch.zeros_like(X)
            for m2 in range(M):
                if m2 != m:
                    rec += codebooks[m2][idx[:, m2]]
            r = X - rec
            a = _assign(r, codebooks[m])
            idx[:, m] = a
            A = torch.nn.functional.one_hot(a, K).float()
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
