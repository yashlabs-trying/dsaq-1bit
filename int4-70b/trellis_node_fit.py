"""Frozen-path shared-node fitting for exact linear-layer output error.

Only the 256x2 quantizer table changes. Original weights, transform, scale,
path budget and code topology are fixed. Full input correlations enter the
normal equations through a matrix-free conjugate-gradient operator.
"""
import torch


def parameter_map(paths, width):
    state = torch.zeros_like(paths, dtype=torch.long)
    states = []
    for step in range(width // 2):
        state = ((state << 4) | ((paths.long() >> (4 * step)) & 15)) & 255
        states.append(state)
    states = torch.stack(states, dim=-1)
    return (2 * states[..., None] + torch.arange(2, device=paths.device)).reshape(paths.shape[0], -1)


@torch.inference_mode()
def fit_output_nodes(w, h, paths, nodes, width=32, iterations=12):
    mapping = parameter_map(paths, width)
    if mapping.shape != w.shape:
        raise ValueError('path dimensions do not match original weights')
    flat = mapping.reshape(-1)
    theta = nodes.reshape(-1)
    def reduce(values):
        out = torch.zeros_like(theta).scatter_add_(0, flat, values.reshape(-1))
        out[:2] = 0  # Neutral node zero remains fixed, including implicit paths.
        return out
    q = theta[mapping]
    error = w - q
    before = float(((error @ h) * error).sum())
    b = reduce(error @ h)
    diagonal = torch.zeros_like(theta).scatter_add_(
        0, flat, h.diag()[None].expand_as(w).reshape(-1))
    ridge = diagonal.max().clamp_min(1e-12) * 1e-6
    preconditioner = (diagonal + ridge).clamp_min(1e-12)
    def operator(delta):
        return reduce(delta[mapping] @ h) + ridge * delta
    delta = torch.zeros_like(theta)
    residual = b.clone()
    z = residual / preconditioner
    direction = z.clone()
    rz = residual.dot(z)
    bnorm = float(b.square().sum())
    done = 0
    for _ in range(iterations):
        ap = operator(direction)
        denominator = direction.dot(ap)
        if not torch.isfinite(denominator) or float(denominator) <= 0:
            break
        alpha = rz / denominator
        delta += alpha * direction
        residual -= alpha * ap
        done += 1
        if float(residual.square().sum()) <= max(bnorm * 1e-8, 1e-20):
            break
        z = residual / preconditioner
        next_rz = residual.dot(z)
        direction = z + (next_rz / rz) * direction
        rz = next_rz
    # FP16 storage rounding and finite solver precision must not worsen loss.
    accepted, after, step_size = nodes.clone(), before, 0.0
    for fraction in (1., .5, .25, .125):
        candidate = (theta + fraction * delta).reshape_as(nodes).half().float()
        candidate[0] = nodes[0]
        candidate_error = w - candidate.reshape(-1)[mapping]
        loss = float(((candidate_error @ h) * candidate_error).sum())
        if loss <= after:
            accepted, after, step_size = candidate, loss, fraction
    return accepted, dict(fixed_path_output_loss_before=before,
                           fixed_path_output_loss_after=after,
                           cg_iterations=done, accepted_step=step_size)
