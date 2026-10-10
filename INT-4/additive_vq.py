"""Additive VQ with optional full block-level Hessian weighting for activation-aware 2-bit quantization."""
import math
import torch


def _assign(X, centers, H_blocks=None, block_pos=None):
    if H_blocks is None:
        xn = (X ** 2).sum(-1, keepdim=True)
        cn = (centers ** 2).sum(-1)
        return (xn - 2 * X @ centers.t() + cn).argmin(1)
    nb = H_blocks.shape[0]
    assign = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for b in range(nb):
        mask = block_pos == b
        if not mask.any():
            continue
        xb = X[mask]
        rw = xb @ H_blocks[b]
        ct = centers @ H_blocks[b]
        cn = (centers * ct).sum(-1)
        assign[mask] = (-2 * rw @ centers.t() + cn).argmin(1)
    return assign


def _assign_beam2(X, C0, C1, beam, H_blocks=None, block_pos=None):
    if H_blocks is None:
        xn = (X ** 2).sum(-1, keepdim=True)
        d0 = xn - 2 * X @ C0.t() + (C0 ** 2).sum(-1)
        top0 = torch.topk(d0, beam, dim=1, largest=False)
        c1n = (C1 ** 2).sum(-1)
        best_val = torch.full((X.shape[0],), float('inf'), device=X.device)
        best_i0 = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        best_i1 = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
        for bi in range(beam):
            k0 = top0.indices[:, bi]
            r = X - C0[k0]
            d1 = (r ** 2).sum(-1, keepdim=True) - 2 * r @ C1.t() + c1n
            v, k1 = d1.min(1)
            better = v < best_val
            best_val = torch.where(better, v, best_val)
            best_i0 = torch.where(better, k0, best_i0)
            best_i1 = torch.where(better, k1, best_i1)
        return best_i0, best_i1
    nb = H_blocks.shape[0]
    best_val = torch.full((X.shape[0],), float('inf'), device=X.device)
    best_i0 = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    best_i1 = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for b in range(nb):
        mask = block_pos == b
        if not mask.any():
            continue
        xb = X[mask]
        Hb = H_blocks[b]
        xw = xb @ Hb
        Ct0 = C0 @ Hb
        cn0 = (C0 * Ct0).sum(-1)
        d0 = -2 * xw @ C0.t() + cn0
        top0 = torch.topk(d0, beam, dim=1, largest=False)
        c1n = (C1 ** 2).sum(-1)
        for bi in range(beam):
            k0 = top0.indices[:, bi]
            r = xb - C0[k0]
            rw = r @ Hb
            Ct1 = C1 @ Hb
            cn1 = (C1 * Ct1).sum(-1)
            d1 = -2 * rw @ C1.t() + cn1
            v, k1 = d1.min(1)
            better = v < best_val[mask]
            mi = mask.nonzero().view(-1)
            update = better.view(-1)
            if update.any():
                best_val[mi[update]] = v[update]
                best_i0[mi[update]] = k0[update]
                best_i1[mi[update]] = k1[update]
    return best_i0, best_i1


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


def additive_vq(X, M, K, init_iters=10, iters=20, beam=1, seed=0, H_blocks=None, block_pos=None):
    N = X.shape[0]
    dev = X.device
    codebooks = []
    idx = torch.zeros(N, M, dtype=torch.long, device=dev)
    resid = X.clone()
    for m in range(M):
        cb = _kmeans(resid, K, init_iters, seed + m)
        codebooks.append(cb)
        a = _assign(resid, cb, H_blocks, block_pos)
        idx[:, m] = a
        resid = resid - cb[a]
    for _ in range(iters):
        if M == 2 and beam > 1:
            i0, i1 = _assign_beam2(X, codebooks[0], codebooks[1], beam, H_blocks, block_pos)
            idx[:, 0] = i0; idx[:, 1] = i1
        else:
            for m in range(M):
                rec = torch.zeros_like(X)
                for m2 in range(M):
                    if m2 != m:
                        rec += codebooks[m2][idx[:, m2]]
                idx[:, m] = _assign(X - rec, codebooks[m], H_blocks, block_pos)
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
    rec = torch.zeros(idx.shape[0], codebooks.shape[-1], device=codebooks.device)
    for m in range(codebooks.shape[0]):
        rec += codebooks[m][idx[:, m]]
    return rec.reshape(shape)


def payload_bytes(idx, codebooks):
    M, K, g = codebooks.shape
    idx_bits = idx.numel() * int(math.log2(K))
    cb_bits = codebooks.numel() * 16
    return (idx_bits + cb_bits) / 8

