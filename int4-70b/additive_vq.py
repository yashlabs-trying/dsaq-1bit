"""Additive VQ with optional full block-level Hessian weighting for activation-aware 2-bit quantization."""
import math
import torch


def hadamard_right(X, signs=None, block_size=128, inverse=False):
    """Multiply rows by a seeded block-diagonal randomized Hadamard matrix.

    Forward is ``X @ (D H)`` and inverse is ``X @ (H D)``.  H is the
    orthonormal Walsh-Hadamard matrix and D is a diagonal sign matrix.
    """
    if block_size <= 0 or block_size & (block_size - 1):
        raise ValueError('block_size must be a positive power of two')
    if X.shape[-1] % block_size:
        raise ValueError('last dimension must be divisible by block_size')
    if signs is None:
        signs = torch.ones(X.shape[-1], dtype=X.dtype, device=X.device)
    signs = signs.to(dtype=X.dtype, device=X.device)
    work = X if inverse else X * signs
    shape = work.shape
    work = work.reshape(-1, shape[-1] // block_size, block_size).clone()
    width = 1
    while width < block_size:
        paired = work.reshape(*work.shape[:-1], -1, 2, width)
        left = paired[..., 0, :].clone()
        right = paired[..., 1, :].clone()
        paired[..., 0, :] = left + right
        paired[..., 1, :] = left - right
        work = paired.reshape(*work.shape[:-1], block_size)
        width *= 2
    work = work / math.sqrt(block_size)
    work = work.reshape(shape)
    return work * signs if inverse else work


def transform_hessian(H, signs, block_size=128):
    """Return R.T @ H @ R for R = D H without materializing R."""
    right = hadamard_right(H, signs, block_size)
    return hadamard_right(right.t(), signs, block_size).t()


def _assign(X, centers, H_blocks=None, block_pos=None):
    if H_blocks is None:
        xn = (X ** 2).sum(-1, keepdim=True)
        cn = (centers ** 2).sum(-1)
        return (xn - 2 * X @ centers.t() + cn).argmin(1)
    nb = H_blocks.shape[0]
    if X.shape[0] % nb == 0 and torch.equal(
            block_pos, torch.arange(X.shape[0], device=X.device) % nb):
        # Linear weights arrive row-major with repeating column positions.
        # Batch those positions instead of scanning N entries nb times.
        rows = X.shape[0] // nb
        grouped = X.reshape(rows, nb, X.shape[1]).transpose(0, 1)
        grouped_assignments = torch.empty(
            nb, rows, dtype=torch.long, device=X.device)
        for start in range(0, nb, 32):
            end = min(start + 32, nb)
            metric = H_blocks[start:end]
            weighted_x = torch.bmm(grouped[start:end], metric)
            weighted_centers = torch.matmul(centers, metric)
            norms = (weighted_centers * centers).sum(-1)
            distances = -2 * torch.matmul(weighted_x, centers.t())
            distances += norms[:, None, :]
            grouped_assignments[start:end] = distances.argmin(-1)
        return grouped_assignments.t().reshape(-1)
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


def _assign_metric(X, centers, metric):
    weighted_x = X @ metric
    weighted_centers = centers @ metric
    center_norm = (centers * weighted_centers).sum(-1)
    return (-2 * weighted_x @ centers.t() + center_norm).argmin(1)


def block_ldlq(W, H, centers, buf_cols=128, quantize_fn=None, callback_metric=False):
    """BlockLDLQ error propagation for one g-dimensional codebook."""
    out_features, in_features = W.shape
    g = centers.shape[1]
    if in_features % g:
        raise ValueError('input dimension must be divisible by codebook width')
    if in_features % buf_cols:
        raise ValueError('input dimension must be divisible by buffer width')

    num_groups = in_features // g
    chol = torch.linalg.cholesky(H)
    diagonal_blocks = torch.diagonal(
        chol.reshape(num_groups, g, num_groups, g), dim1=0, dim2=2
    ).permute(2, 0, 1)
    D = diagonal_blocks @ diagonal_blocks.transpose(1, 2)
    inverse_blocks = torch.linalg.inv(diagonal_blocks)
    L = chol.view(in_features, num_groups, g)
    for group in range(num_groups):
        L[:, group, :] = L[:, group, :] @ inverse_blocks[group]
    L = L.reshape(in_features, in_features)

    W_t = W.t().contiguous()
    quantized_t = torch.zeros_like(W_t)
    indices_t = torch.zeros(
        num_groups, out_features, dtype=torch.long, device=W.device)
    propagated = torch.zeros_like(W_t)

    for end in range(in_features, 0, -buf_cols):
        start = end - buf_cols
        block_w = W_t[start:end]
        block_q = quantized_t[start:end]
        block_l = L[start:end]
        block_propagated = propagated[start:end]
        for local_end in range(buf_cols, 0, -g):
            local_start = local_end - g
            column_start = start + local_start
            correction = block_propagated[local_start:local_end]
            if local_end < buf_cols:
                correction = correction + block_l[
                    local_end:, column_start:column_start + g
                ].t() @ (block_w[local_end:] - block_q[local_end:])
            candidate = (block_w[local_start:local_end] + correction).t()
            group = column_start // g
            if quantize_fn is None:
                assigned = _assign_metric(candidate, centers, D[group])
                selected_centers = centers[assigned]
            else:
                selected_centers, assigned = (quantize_fn(candidate, D[group])
                                              if callback_metric else quantize_fn(candidate))
            block_q[local_start:local_end] = selected_centers.t()
            indices_t[group] = assigned
        propagated += block_l.t() @ (block_w - block_q)
        quantized_t[start:end] = block_q

    return quantized_t.t().contiguous(), indices_t.t().contiguous()


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
        for bi in range(beam):
            k0 = top0.indices[:, bi]
            r = xb - C0[k0]
            rw = r @ Hb
            Ct1 = C1 @ Hb
            cn1 = (C1 * Ct1).sum(-1)
            # r^T H r depends on the first-codebook candidate. Omitting it
            # compares incompatible partial distances and can select a worse pair.
            rn = (r * rw).sum(-1, keepdim=True)
            d1 = rn - 2 * rw @ C1.t() + cn1
            v, k1 = d1.min(1)
            better = v < best_val[mask]
            mi = mask.nonzero().view(-1)
            update = better.view(-1)
            if update.any():
                best_val[mi[update]] = v[update]
                best_i0[mi[update]] = k0[update]
                best_i1[mi[update]] = k1[update]
    return best_i0, best_i1


def _update_codebook(residual, assignments, K, old_centers,
                     H_blocks=None, block_pos=None, sample_weight=None):
    """Minimize the same metric used by assignment for fixed assignments."""
    if H_blocks is None:
        centers = old_centers.clone()
        sums = torch.zeros_like(centers)
        if sample_weight is None:
            sample_weight = torch.ones(
                residual.shape[0], dtype=residual.dtype, device=residual.device)
        sums.index_add_(0, assignments, residual * sample_weight[:, None])
        weight_sum = torch.zeros(K, dtype=residual.dtype, device=residual.device)
        weight_sum.index_add_(0, assignments, sample_weight)
        active = weight_sum > 0
        centers[active] = sums[active] / weight_sum[active, None]
        return centers

    g = residual.shape[1]
    normal_flat = torch.zeros(
        K, g * g, dtype=residual.dtype, device=residual.device)
    rhs = torch.zeros(K, g, dtype=residual.dtype, device=residual.device)
    row_hessians = H_blocks[block_pos]
    if sample_weight is None:
        sample_weight = torch.ones(
            residual.shape[0], dtype=residual.dtype, device=residual.device)
    weighted_hessians = row_hessians * sample_weight[:, None, None]
    normal_flat.index_add_(0, assignments, weighted_hessians.flatten(1))
    weighted_residual = torch.bmm(
        residual.unsqueeze(1), weighted_hessians).squeeze(1)
    rhs.index_add_(0, assignments, weighted_residual)
    counts = torch.bincount(assignments, minlength=K)

    centers = old_centers.clone()
    active = counts > 0
    if active.any():
        systems = normal_flat.view(K, g, g)[active]
        scale = systems.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
        eye = torch.eye(g, dtype=residual.dtype, device=residual.device)
        systems = systems + (1e-7 * scale)[:, None, None] * eye
        centers[active] = torch.linalg.solve(
            systems, rhs[active].unsqueeze(-1)).squeeze(-1)
    return centers


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


def _metric_kmeans(X, K, iters, seed, H_blocks, block_pos,
                   sample_weight=None, restarts=1):
    """Fit initial centers under the same activation metric as refinement."""
    best_centers, best_error = None, float('inf')
    for restart in range(restarts):
        generator = torch.Generator(device=X.device)
        generator.manual_seed(seed + 1009 * restart)
        selected = torch.randperm(X.shape[0], generator=generator,
                                  device=X.device)[:K]
        centers = X[selected].clone()
        for _ in range(iters):
            assignments = _assign(X, centers, H_blocks, block_pos)
            centers = _update_codebook(
                X, assignments, K, centers, H_blocks, block_pos, sample_weight)
        assignments = _assign(X, centers, H_blocks, block_pos)
        residual = X - centers[assignments]
        weighted = torch.bmm(residual[:, None, :],
                             H_blocks[block_pos]).squeeze(1)
        errors = (residual * weighted).sum(1)
        if sample_weight is not None:
            errors = errors * sample_weight
        error = float(errors.sum())
        if error < best_error:
            best_centers, best_error = centers, error
    return best_centers


def additive_vq(X, M, K, init_iters=10, iters=20, beam=1, seed=0,
                H_blocks=None, block_pos=None, sample_weight=None,
                hessian_init=False, init_restarts=1):
    N = X.shape[0]
    dev = X.device
    codebooks = []
    idx = torch.zeros(N, M, dtype=torch.long, device=dev)
    resid = X.clone()
    for m in range(M):
        if hessian_init:
            if H_blocks is None or block_pos is None:
                raise ValueError('Hessian initialization requires block metrics')
            cb = _metric_kmeans(
                resid, K, init_iters, seed + m, H_blocks, block_pos,
                sample_weight, init_restarts)
        else:
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
            codebooks[m] = _update_codebook(
                r, idx[:, m], K, codebooks[m], H_blocks, block_pos,
                sample_weight)
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
