"""Activation-aware GPTQ-style quantizer: a strong low-bit base for residual codecs.

Quantizes a linear weight matrix column by column and distributes the rounding
error of each already-quantized column into the still-unquantized columns using
the inverse input Hessian (2 * X^T X). This keeps the layer *output* close to
the reference even though the weights themselves are heavily rounded.
"""
import torch


def _scale_zero(x, maxq, mse, norm, grid, maxshrink, sym):
    """Per-row (per output channel) asymmetric scale/zero, with optional MSE search."""
    dev = x.device
    xmin = torch.minimum(x.min(1)[0], torch.zeros(x.shape[0], device=dev))
    xmax = torch.maximum(x.max(1)[0], torch.zeros(x.shape[0], device=dev))
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = -xmax
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    if not mse:
        return scale, zero
    best = torch.full([x.shape[0]], float('inf'), device=dev)
    best_scale = scale.clone()
    best_zero = zero.clone()
    for i in range(1, int(maxshrink * grid) + 1):
        p = 1 - i / grid
        s = p * scale
        z = torch.round(-(p * xmin) / s)
        q = torch.clamp(torch.round(x / s.unsqueeze(1)) + z.unsqueeze(1), 0, maxq)
        q = s.unsqueeze(1) * (q - z.unsqueeze(1))
        err = (q - x).abs().pow(norm).sum(1)
        better = err < best
        if better.any():
            best = torch.where(better, err, best)
            best_scale = torch.where(better, s, best_scale)
            best_zero = torch.where(better, z, best_zero)
    return best_scale, best_zero


def gptq_quantize(W, H, bits, groupsize=-1, groupwise=False, sym=False, mse=True,
                  percdamp=0.01, norm=2.4, grid=100, maxshrink=0.8):
    """Quantize W [out, cols] against input Hessian H [cols, cols].

    Returns Q (dequantized float), codes (uint8 in [0, 2^bits-1]), and scale/zero
    (either [out] per-channel, or [out, cols//groupsize] when groupwise).
    """
    if W.dim() != 2:
        raise ValueError('W must be a 2-D [out_features, in_features] tensor')
    out, cols = W.shape
    dev = W.device
    maxq = 2 ** bits - 1

    H = H.to(device=dev, dtype=torch.float32).clone()
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(cols, device=dev)
    H[diag, diag] += damp
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H

    W = W.float().clone()
    Q = torch.zeros_like(W)
    codes = torch.empty_like(W, dtype=torch.uint8)

    gsize = groupsize if groupsize > 0 else cols
    scales = []
    zeros = []

    def quant_col(x, scale, zero):
        code = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (code - zero), code

    def run_loop(i1, i2, scale, zero):
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(count):
            w = W1[:, i]
            d = Hinv1[i, i]
            q_val, code = quant_col(w, scale, zero)
            err = (w - q_val) / d
            Q1[:, i] = q_val
            codes[:, i1 + i] = code.to(torch.uint8)
            W1[:, i] = q_val
            W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    if groupwise:
        for i1 in range(0, cols, gsize):
            i2 = min(i1 + gsize, cols)
            scale, zero = _scale_zero(W[:, i1:i2], maxq, mse, norm, grid, maxshrink, sym)
            scales.append(scale)
            zeros.append(zero)
            run_loop(i1, i2, scale, zero)
        scale = torch.stack(scales, 1)
        zero = torch.stack(zeros, 1)
    else:
        scale, zero = _scale_zero(W, maxq, mse, norm, grid, maxshrink, sym)
        for i1 in range(0, cols, gsize):
            run_loop(i1, min(i1 + gsize, cols), scale, zero)

    return Q, codes, scale, zero


def decode(codes, scale, zero):
    """Reconstruct the dequantized float tensor from integer codes and scale/zero."""
    codes = codes.float()
    if scale.dim() == 2:
        g = codes.shape[1] // scale.shape[1]
        s = scale.repeat_interleave(g, 1)
        z = zero.repeat_interleave(g, 1)
    else:
        s, z = scale.unsqueeze(1), zero.unsqueeze(1)
    return s * (codes - z)


def payload_bytes(codes, scale, zero, bits):
    """Storage estimate: packed integer codes at `bits` each plus FP16 metadata."""
    packed = codes.numel() * bits
    meta = (scale.numel() + zero.numel()) * 16
    return (packed + meta) / 8


