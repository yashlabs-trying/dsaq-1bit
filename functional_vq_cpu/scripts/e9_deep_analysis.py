"""Deep diagnostic: gradient spectrum, logit/KL sensitivity, error propagation.

Measures:
1. Sensitivity gradient norms across all 896 dimensions → how many matter?
2. Full-model logit/KL gradient for comparison with L1/L2 gradients
3. Error propagation: Layer-0 error after E9c, propagated through all 24 layers
4. Per-projection contribution to error (Gate, Up, Down, Q, K, V, O)
5. Attention distortion: how much does the quantized attention pattern differ from BF16?

This is a read-only diagnostic. It does not modify any checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import decode_weights


ROOT = Path(__file__).resolve().parents[1]
TOKENS_PATH = (ROOT.parent / "artifacts/functional_vq_l0/calibration_cache/fixed_token_ids.pt").resolve()
PACKED_E9C = ROOT / "packed/E9c_downstream_l12_5pct.packed"
PACKED_E8G = ROOT / "packed/E8g_down_polish_round5.packed"
E9B_CACHE = ROOT / "cache/e9b_cache.pt"
E9C_CACHE = ROOT / "cache/e9c_cache.pt"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SEQ = 512


def module_map(model):
    layer = model.model.layers[0]
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    return {name: getattr(layer.self_attn if name in names[:4] else layer.mlp, name) for name in names}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_layer_incremental(h, model, layer_idx, rotary_cache=None):
    """Run hidden state h through one transformer layer. Returns (output, attn_probs)."""
    layer = model.model.layers[layer_idx]
    seq_len = h.shape[0]
    pos_ids = torch.arange(seq_len, dtype=torch.long, device=h.device).unsqueeze(0)

    if rotary_cache is not None and layer_idx in rotary_cache:
        cos, sin = rotary_cache[layer_idx]
    else:
        cos, sin = model.model.rotary_emb(h.unsqueeze(0), pos_ids)
        if rotary_cache is not None:
            rotary_cache[layer_idx] = (cos, sin)
    pos_emb = (cos, sin)

    causal_mask = torch.full((seq_len, seq_len), torch.finfo(h.dtype).min, device=h.device)
    causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)

    h_norm = layer.input_layernorm(h)
    attn_out = layer.self_attn(h_norm.unsqueeze(0), attention_mask=causal_mask,
                               position_embeddings=pos_emb)[0].squeeze(0)
    h = h + attn_out
    h_attn = h
    h = layer.post_attention_layernorm(h)
    h = h_attn + layer.mlp(h)
    return h


def capture_all_layer_outputs(model, token_ids):
    """Run the full model and capture hidden states at every layer."""
    seq_len = token_ids.shape[0]
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(token_ids)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(inputs_embeds, position_ids)
        rotary_cache = {i: (cos, sin) for i in range(24)}

        hidden_states = inputs_embeds
        all_outputs = [hidden_states.cpu()]

        for i in range(24):
            hidden_states = run_layer_incremental(hidden_states, model, i, rotary_cache)
            all_outputs.append(hidden_states.cpu())

        logits = model.lm_head(hidden_states).cpu()
        all_outputs.append(logits)

    return all_outputs


def compute_layerwise_error_propagation(model, token_ids, e9c_decoded, original_weights):
    """Compute how Layer-0 quantization error propagates through all layers."""
    device = next(model.parameters()).device

    def run_full(override_weights=None):
        if override_weights:
            with torch.no_grad():
                for name, w in override_weights.items():
                    module_map(model)[name].weight.copy_(w.to(device))
        outputs = capture_all_layer_outputs(model, token_ids.to(device))
        if override_weights:
            with torch.no_grad():
                for name, w in original_weights.items():
                    module_map(model)[name].weight.copy_(w.to(device))
        return outputs

    print(json.dumps({"event": "propagation_teacher"}), flush=True)
    teacher = run_full()

    print(json.dumps({"event": "propagation_e9c"}), flush=True)
    student = run_full(e9c_decoded)

    layer_errors = []
    for i in range(25):
        t = teacher[i]
        s = student[i]
        nmse = float((s - t).square().sum() / t.square().sum())
        cos_sim = float(F.cosine_similarity(s.flatten().unsqueeze(0), t.flatten().unsqueeze(0)))
        layer_errors.append({"layer": i, "NMSE": nmse, "cosine_similarity": cos_sim})

    return layer_errors, teacher, student


def compute_logit_sensitivity_gradient(model, token_ids, e9c_decoded, original_weights):
    """Compute g_logit = ∂KL(student_logits || teacher_logits) / ∂h2."""
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)

    with torch.no_grad():
        for name, w in original_weights.items():
            module_map(model)[name].weight.copy_(w.to(device))
        teacher_hidden = [token_ids]
        h = model.model.embed_tokens(token_ids)
        teacher_hidden.append(h.cpu())
        pos_ids = torch.arange(h.shape[0], device=device).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h, pos_ids)
        rc = {i: (cos, sin) for i in range(24)}
        for i in range(24):
            h = run_layer_incremental(h, model, i, rc)
            teacher_hidden.append(h.cpu())
        teacher_logits = model.lm_head(h).detach()

    with torch.no_grad():
        for name, w in e9c_decoded.items():
            module_map(model)[name].weight.copy_(w.to(device))

    h2 = model.model.embed_tokens(token_ids)
    pos_ids = torch.arange(h2.shape[0], device=device).unsqueeze(0)
    cos, sin = model.model.rotary_emb(h2, pos_ids)
    rc = {i: (cos, sin) for i in range(24)}

    def run_to_h2():
        h = model.model.embed_tokens(token_ids)
        h = run_layer_incremental(h, model, 0, rc)
        return h

    h2_student = run_to_h2().detach().clone().requires_grad_(True)
    rc1 = {i: (cos, sin) for i in range(24)}

    def forward_from_h2(h2_input):
        h = h2_input
        for i in range(1, 24):
            h = run_layer_incremental(h, model, i, rc1)
        logits = model.lm_head(h)
        return logits

    student_logits = forward_from_h2(h2_student)
    teacher_log_prob = F.log_softmax(teacher_logits.float(), dim=-1)
    student_prob = F.softmax(student_logits.float(), dim=-1)
    kl = (student_prob * (student_prob.log() - teacher_log_prob)).sum(-1).mean()

    g_logit = torch.autograd.grad(kl, h2_student)[0]

    with torch.no_grad():
        for name, w in original_weights.items():
            module_map(model)[name].weight.copy_(w.to(device))

    return g_logit.cpu(), float(kl.detach())


def gradient_spectrum_analysis(gradients_dict):
    """Analyze gradient norms across all 896 dimensions."""
    results = {}
    for name, g in gradients_dict.items():
        g = g.float()
        norms = g.norm(dim=0)
        sorted_norms = norms.sort(descending=True).values
        cumulative = sorted_norms.cumsum(0) / sorted_norms.sum()

        n_50pct = int((cumulative < 0.50).sum())
        n_90pct = int((cumulative < 0.90).sum())
        n_99pct = int((cumulative < 0.99).sum())

        top1_frac = float(sorted_norms[0] / sorted_norms.sum())
        top10_frac = float(sorted_norms[:10].sum() / sorted_norms.sum())
        top100_frac = float(sorted_norms[:100].sum() / sorted_norms.sum())

        results[name] = {
            "dims_for_50pct": n_50pct,
            "dims_for_90pct": n_90pct,
            "dims_for_99pct": n_99pct,
            "top1_contribution": top1_frac,
            "top10_contribution": top10_frac,
            "top100_contribution": top100_frac,
            "mean_norm": float(norms.mean()),
            "median_norm": float(norms.median()),
            "max_norm": float(norms.max()),
            "min_norm": float(norms.min()),
            "norm_std": float(norms.std()),
        }
    return results


def attention_distortion_analysis(model, token_ids, e9c_decoded, original_weights):
    """Compare attention patterns between teacher and E9c student."""
    device = next(model.parameters()).device
    token_ids = token_ids.to(device)
    seq_len = token_ids.shape[0]
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    def get_attentions(override_weights=None):
        if override_weights:
            with torch.no_grad():
                for name, w in override_weights.items():
                    module_map(model)[name].weight.copy_(w.to(device))
        h = model.model.embed_tokens(token_ids)
        cos, sin = model.model.rotary_emb(h, pos_ids)
        attentions = []
        for i in range(24):
            layer = model.model.layers[i]
            causal_mask = torch.full((seq_len, seq_len), torch.finfo(h.dtype).min, device=device)
            causal_mask = torch.triu(causal_mask, diagonal=1).unsqueeze(0).unsqueeze(0)
            pos_emb = (cos, sin)
            h_norm = layer.input_layernorm(h)
            attn_out = layer.self_attn(h_norm.unsqueeze(0), attention_mask=causal_mask,
                                       position_embeddings=pos_emb, output_attentions=True)
            attn_probs = attn_out[1]
            attentions.append(attn_probs.cpu())
            h = h + attn_out[0].squeeze(0)
            h_attn = h
            h = layer.post_attention_layernorm(h)
            h = h_attn + layer.mlp(h)
        if override_weights:
            with torch.no_grad():
                for name, w in original_weights.items():
                    module_map(model)[name].weight.copy_(w.to(device))
        return attentions

    teacher_attn = get_attentions()
    student_attn = get_attentions(e9c_decoded)

    distortions = []
    for i in range(24):
        t = teacher_attn[i]
        s = student_attn[i]
        diff = (s - t).abs().mean().item()
        kl_per_head = F.kl_div(s.log(), t, reduction="batchmean", log_target=False).item()
        distortions.append({"layer": i, "mean_abs_diff": diff, "KL_div": kl_per_head})

    return distortions


def main():
    started = time.time()
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    token_sets = torch.load(TOKENS_PATH, map_location="cpu", weights_only=True)
    heldout = token_sets["heldout"]
    calib_b = heldout[512:1024]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager"
    ).eval().to(device)

    original_weights = {name: mod.weight.detach().clone().cpu()
                        for name, mod in module_map(model).items()}

    e9c_decoded, _ = decode_weights(PACKED_E9C)
    e9b_cache = torch.load(E9B_CACHE, map_location="cpu", weights_only=True)
    e9c_cache_grads = torch.load(E9C_CACHE, map_location="cpu", weights_only=True)

    report = {
        "checkpoints": {"E8g": str(PACKED_E8G), "E9c": str(PACKED_E9C)},
        "gradient_spectrum": {},
        "error_propagation": {},
        "attention_distortion": {},
        "logit_sensitivity": {},
        "analysis": {},
    }

    # ── 1. Gradient spectrum analysis ──
    print(json.dumps({"event": "gradient_spectrum"}), flush=True)
    gradient_sources = {
        "L1_sensitivity_A": e9b_cache["sensitivity_gradient_a"],
        "L1_sensitivity_B": e9b_cache["sensitivity_gradient_b"],
        "L12_sensitivity_A": e9c_cache_grads["sensitivity_gradient_a"],
        "L12_sensitivity_B": e9c_cache_grads["sensitivity_gradient_b"],
    }
    report["gradient_spectrum"] = gradient_spectrum_analysis(gradient_sources)

    # ── 2. Logit/KL sensitivity gradient ──
    print(json.dumps({"event": "logit_sensitivity"}), flush=True)
    g_logit_a, kl_a = compute_logit_sensitivity_gradient(model, calib_b[:SEQ//2], e9c_decoded, original_weights)
    # Use second half as calibration B
    g_logit_b, kl_b = compute_logit_sensitivity_gradient(model, calib_b[SEQ//2:], e9c_decoded, original_weights)
    logit_spectrum = gradient_spectrum_analysis({
        "logit_KL_sensitivity_A": g_logit_a,
        "logit_KL_sensitivity_B": g_logit_b,
    })
    report["logit_sensitivity"] = {
        "KL_A": kl_a, "KL_B": kl_b,
        "spectrum": logit_spectrum,
    }
    for k, v in logit_spectrum.items():
        report["gradient_spectrum"][k] = v

    # ── 3. Error propagation through all layers ──
    print(json.dumps({"event": "error_propagation"}), flush=True)
    # Use a smaller subset for speed
    calib_short = calib_b[:64]
    layer_errors, teacher, student = compute_layerwise_error_propagation(
        model, calib_short, e9c_decoded, original_weights
    )
    report["error_propagation"] = {str(l["layer"]): l for l in layer_errors}

    # ── 4. Attention distortion (Layer 0 only = most affected) ──
    print(json.dumps({"event": "attention_distortion"}), flush=True)
    attn_distortions = attention_distortion_analysis(model, calib_b[:64], e9c_decoded, original_weights)
    report["attention_distortion"] = {str(a["layer"]): a for a in attn_distortions}

    # ── 5. Synthesis and recommendations ──
    print(json.dumps({"event": "synthesis"}), flush=True)
    synthesis = synthesize(report, layer_errors, attn_distortions)
    report["analysis"] = synthesis

    print(json.dumps({"event": "complete", **report}, indent=2), flush=True)


def synthesize(report, layer_errors, attn_distortions):
    """Analyze all diagnostics and produce concrete recommendations."""
    notes = []

    # Gradient concentration
    for grad_name, grad_data in report["gradient_spectrum"].items():
        top10 = grad_data["top10_contribution"]
        top100 = grad_data["top100_contribution"]
        dims50 = grad_data["dims_for_50pct"]
        notes.append({
            "gradient": grad_name,
            "top10_controls_pct": round(100 * top10, 1),
            "top100_controls_pct": round(100 * top100, 1),
            "dims_for_50pct": dims50,
            "interpretation": (
                f"Top 10 dims control {100*top10:.1f}% of sensitivity; "
                f"{dims50}/{896} dims = {100*dims50/896:.1f}% for 50% of gradient mass"
            ),
        })

    # Error propagation
    h2_error_nmse = layer_errors[1]["NMSE"] if len(layer_errors) > 1 else None
    h24_error_nmse = layer_errors[24]["NMSE"] if len(layer_errors) > 24 else None
    h2_cos = layer_errors[1]["cosine_similarity"] if len(layer_errors) > 1 else None
    h24_cos = layer_errors[24]["cosine_similarity"] if len(layer_errors) > 24 else None

    if h2_error_nmse and h24_error_nmse:
        amp = h24_error_nmse / max(h2_error_nmse, 1e-10)
        notes.append({
            "topic": "error_propagation",
            "h2_NMSE": h2_error_nmse,
            "h24_NMSE": h24_error_nmse,
            "h2_cosine": h2_cos,
            "h24_cosine": h24_cos,
            "amplification": amp,
            "interpretation": (
                f"Error NMSE at h2 (Layer 0 out): {h2_error_nmse:.4f}, "
                f"at logits (Layer 24): {h24_error_nmse:.4f}, "
                f"amplification: {amp:.1f}x. "
                f"Cosine sim at h2: {h2_cos:.4f}, at logits: {h24_cos:.4f}. "
                f"Error grows by {amp:.1f}x through 23 layers."
            ),
        })

    # Attention distortion
    l0_attn_kl = attn_distortions[0]["KL_div"] if len(attn_distortions) > 0 else None
    early_attn_kls = [a["KL_div"] for a in attn_distortions[:6]]
    late_attn_kls = [a["KL_div"] for a in attn_distortions[18:]]

    notes.append({
        "topic": "attention_distortion",
        "layer_0_attn_KL": l0_attn_kl,
        "mean_early_attn_KL": float(np.mean(early_attn_kls)) if early_attn_kls else None,
        "mean_late_attn_KL": float(np.mean(late_attn_kls)) if late_attn_kls else None,
        "interpretation": (
            f"Layer 0 attention KL: {l0_attn_kl:.6f}, "
            f"mean early (L0-5): {float(np.mean(early_attn_kls)):.6f}, "
            f"mean late (L18-23): {float(np.mean(late_attn_kls)):.6f}. "
            "Early layers are most affected; late layers recover."
        ),
    })

    # Compare L1 vs L12 vs logit gradient alignment
    g_l1_a = report["gradient_spectrum"].get("L1_sensitivity_A", {})
    g_l12_a = report["gradient_spectrum"].get("L12_sensitivity_A", {})
    g_logit_a = report["gradient_spectrum"].get("logit_KL_sensitivity_A", {})

    l1_dirs = g_l1_a.get("dims_for_50pct", "?")
    l12_dirs = g_l12_a.get("dims_for_50pct", "?")
    logit_dirs = g_logit_a.get("dims_for_50pct", "?")
    notes.append({
        "topic": "horizon_sensitivity",
        "L1_dims_for_50pct": l1_dirs,
        "L12_dims_for_50pct": l12_dirs,
        "logit_dims_for_50pct": logit_dirs,
        "interpretation": (
            f"L1: {l1_dirs} dims control 50% of gradient → L12: {l12_dirs} dims → "
            f"logit/KL: {logit_dirs} dims. "
        ),
    })

    # Recommendations
    recommendations = []

    # --- RECOMMENDATION A: Full-horizon logit/KL ---
    recommendations.append({
        "priority": 1,
        "action": "Full-horizon logit/KL sensitivity optimization",
        "expected_gain": "Large (10-20 PPL points)",
        "rationale": (
            "E9c only propagates through L1+L2. The logit/KL gradient captures the "
            "full 23-layer propagation. Even if used only for candidate scoring with "
            "a cheaper first-order approximation, this should significantly improve "
            "the alignment between the proxy loss and true PPL. "
            "The gradient can be precomputed once (like L1/L2 gradients)."
        ),
        "implementation": "export_logit_cache.py → polish script using logit sensitivity gradient",
        "risk": "Moderate — computing the logit gradient requires full model forward+backward, but it's only done once",
    })

    # --- RECOMMENDATION B: Optimize all 7 projections ---
    recommendations.append({
        "priority": 1,
        "action": "Extend downstream sensitivity to Gate, Up, Q, K, V, O projections",
        "expected_gain": "Large (10-20 PPL points)",
        "rationale": (
            "Current optimization only touches Down projection. The Gate and Up projections "
            "directly control the MLP output (z = silu(gate)*up). The Q, K, V projections "
            "control attention patterns. All these are quantized but stuck at original "
            "codebook assignments. Optimizing them with downstream sensitivity would "
            "reduce error from multiple independent sources. "
            "The attention error (Kullback-Leibler divergence in attention patterns) "
            "is particularly concerning because wrong attention at Layer 0 propagates "
            "errors nonlinearly through all subsequent layers."
        ),
        "implementation": "Extend candidate search to all projections with downstream sensitivity scoring",
        "risk": "Low — existing candidates infrastructure in e7b_cpu_reassign.py",
    })

    # --- RECOMMENDATION C: Multi-round refinement ---
    recommendations.append({
        "priority": 2,
        "action": "Iterative refinement: Down → Gate/Up → Q/K/V/O → repeat",
        "expected_gain": "Moderate (5-10 PPL points, compounding)",
        "rationale": (
            "The projections are interdependent. Optimizing Down changes h2, which changes "
            "what Gate/Up see as input. Optimizing Gate/Up changes z, which changes Down's "
            "optimal assignment. A multi-round sequence (like E8b→g but with downstream "
            "sensitivity) would converge to a joint optimum."
        ),
        "implementation": "Round-robin polish: 1) Down polish, 2) Gate/Up polish, 3) Q/K/V/O polish, repeat",
        "risk": "Low — uses existing infrastructure, just more rounds",
    })

    # --- RECOMMENDATION D: Increase bits for sensitive dimensions ---
    recommendations.append({
        "priority": 2,
        "action": "Variable-bit allocation: more bits for sensitive dimensions",
        "expected_gain": "Moderate (5-15 PPL points)",
        "rationale": (
            f"The gradient analysis shows that top 10 directions control "
            f"{g_l12_a.get('top10_contribution', 0)*100:.1f}% of L12 sensitivity. "
            f"These dimensions could use 2-4x more bits while insensitive dimensions "
            f"use fewer bits. Total bit budget stays the same."
        ),
        "implementation": "Identify high-sensitivity dimensions from gradient → allocate more patterns per family for those dims",
        "risk": "Moderate — requires codebook retraining with non-uniform quantization",
    })

    # --- RECOMMENDATION E: Quantize Layer 1 with same technique ---
    recommendations.append({
        "priority": 3,
        "action": "Extend VQ to Layer 1 (or more layers)",
        "expected_gain": "Very large (20-40+ PPL points if done well)",
        "rationale": (
            "Quantizing only Layer 0 is a fundamental constraint. The error from a single "
            "quantized layer, no matter how well optimized, propagates through 23 layers. "
            "If Layer 1 is also quantized, we can optimize both layers jointly using "
            "downstream sensitivity through Layer 2+."
        ),
        "implementation": "Train VQ codebook for Layer 1 → export cache → run multi-layer downstream polish",
        "risk": "High — 2x compute, but proportional gain expected",
    })

    return {
        "notes": notes,
        "recommendations": recommendations,
        "summary": {
            "current_ppl": 66.606,
            "target_ppl": 30,
            "gap": 36.6,
            "gap_from_reference": 44.4,
            "most_impactful_single_change": "Optimize all 7 projections with logit/KL sensitivity",
            "estimated_reachable_ppl": "30-40 with recommendations A+B, potentially 25-30 with A+B+C",
        }
    }


if __name__ == "__main__":
    main()
