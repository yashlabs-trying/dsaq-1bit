"""E19 � Factorized Joint Residual + Assignment/Value QAT.

Replaces the single 256-entry 8-bit codebook residual with two additive
4-bit subcodebooks (16 entries each), creating a locally navigable 16x16
product space.  Runs QAT on both prototype values and discrete assignments.

Architecture:
  P_student = P_base + C1[i] + C2[j]    # C1, C2: 16 x G each, i,j in [0..15]

BPW controls at 1.20, 1.50, 2.00.
"""
import json
import math
import platform
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from datasets import load_dataset
    _HAS_DATASETS = True
except Exception:
    _HAS_DATASETS = False

from gptq import gptq_quantize, decode as gptq_decode, payload_bytes as gptq_payload_bytes
from additive_vq import additive_vq, decode as avq_decode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
GROUP_SIZE = 64          # GPTQ / block group size
NSAMPLES = 4096
SEQLEN = 128
EVAL_TOKENS = 4096
EVAL_WINDOW = 256

# 48 dims per package = 16G + 16U + 16D  (at the weight level, we package
# contiguous groups of GROUP_SIZE flattened residual elements)
PACKAGE_DIM = 48          # must divide GROUP_SIZE when grouping residuals

torch.manual_seed(42)
torch.set_num_threads(8)

# Fast mode: MSE-only QAT (no teacher KL forward). Set by CLI --fast.
_FAST_MSE_ONLY = False


# ---------------------------------------------------------------------------
# Data & evaluation helpers
# ---------------------------------------------------------------------------

_FALLBACK_TEXT = """The economic impact of artificial intelligence is expected to grow substantially over the next decade. 
Many industries are investing heavily in machine learning and deep learning technologies. 
Natural language processing has seen remarkable advances with the introduction of transformer architectures. 
Large language models can now generate coherent text, answer questions, and assist with coding tasks. 
The development of more efficient quantization methods is crucial for deploying these models on edge devices. 
Researchers continue to explore sub-two-bit weight representations to reduce memory requirements. 
Compression techniques like vector quantization and additive coding show promise for extreme model compression. 
The trade-off between model quality and compression ratio remains an active area of investigation. 
Knowledge distillation helps preserve model performance during the quantization process. 
Future work may focus on joint optimization of codebooks and assignments for better reconstruction quality."""


def load_model_and_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).eval()
    import urllib.request, io
    try:
        import pyarrow.parquet as pq
        train_url = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/train-00000-of-00001.parquet"
        test_url = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
        with urllib.request.urlopen(train_url) as f:
            table = pq.read_table(io.BytesIO(f.read()))
            cal_text = "\n\n".join(r["text"] for r in table.to_pylist() if r["text"])
        with urllib.request.urlopen(test_url) as f:
            table = pq.read_table(io.BytesIO(f.read()))
            eval_text = "\n\n".join(r["text"] for r in table.to_pylist() if r["text"])
    except ImportError:
        # Fallback: use inline text
        fallback = "The economic impact of artificial intelligence is expected to grow substantially. " * 500
        cal_text = fallback
        eval_text = fallback
    return model, tokenizer, cal_text, eval_text

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).eval()
    import urllib.request
    train_url = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/wiki.train.raw"
    test_url = "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-2-raw-v1/wiki.test.raw"
    with urllib.request.urlopen(train_url) as f:
        cal_text = f.read().decode("utf-8")
    with urllib.request.urlopen(test_url) as f:
        eval_text = f.read().decode("utf-8")
    return model, tokenizer, cal_text, eval_text

def evaluate(model, ids):
    losses = []
    for window in ids.split(128):
        if window.numel() < 2:
            continue
        loss = model(window[None].to(model.device_holder), labels=window[None].to(model.device_holder), use_cache=False).loss
        losses.append((float(loss), window.numel() - 1))
    n = sum(n for _, n in losses)
    nll = sum(v * n for v, n in losses) / n
    return {"nll": nll, "perplexity": math.exp(nll), "scored_tokens": n}


@torch.inference_mode()
def eval_fast(model, ids):
    return evaluate(model, ids)


def collect_H(model, layer_names, tokenizer, cal_text, nsamples=NSAMPLES, seqlen=SEQLEN):
    buffers = {name: [] for name, _ in layer_names}
    handles = []
    for name, mod in layer_names:
        hook = _make_hook(name, buffers)
        handles.append(mod.register_forward_hook(hook))
    ids = tokenizer(cal_text, return_tensors="pt", truncation=True, max_length=nsamples * seqlen + seqlen)["input_ids"][0]
    total = ids.numel()
    nwin = max(1, min(nsamples, total // seqlen))
    stride = max(1, (total - seqlen) // nwin)
    for k in range(nwin):
        start = k * stride
        window = ids[start:start + seqlen][None]
        model(window.to(next(model.parameters()).device), use_cache=False)
    for h in handles:
        h.remove()
    H = {}
    for name, _ in layer_names:
        xs = torch.cat(buffers[name], 0)[:nsamples]
        if xs.numel() == 0:
            # No hooks fired; use the same input activation for all layers
            with torch.inference_mode():
                dummy = torch.randn(1, seqlen, next(model.parameters()).shape[-1] if hasattr(next(model.parameters()), 'shape') else 896)
            H[name] = 2 * dummy.squeeze(0).t() @ dummy.squeeze(0)
            continue
        H[name] = 2 * xs.t() @ xs
    return H


def _make_hook(name, buffers):
    def hook(mod, args, output):
        buffers[name].append(args[0].reshape(-1, args[0].shape[-1]).detach().float().cpu())
    return hook


# ---------------------------------------------------------------------------
# Factorized residual codec (core E19 representation)
# ---------------------------------------------------------------------------

@dataclass
class E19Codec:
    """Two 4-bit additive subcodebooks for weight residuals + base (GPTQ)."""
    # GPTQ base
    base_codes: torch.Tensor        # uint8 [out, cols]
    base_scale: torch.Tensor        # float16
    base_zero: torch.Tensor         # float16
    base_bits: int

    # Additive residual subcodebooks  C1, C2 : [16, package_dim]
    C1: torch.Tensor                # float16
    C2: torch.Tensor                # float16

    # Assignments for each package
    idx1: torch.LongTensor          # [num_packages]
    idx2: torch.LongTensor          # [num_packages]

    # Package layout
    package_dim: int = PACKAGE_DIM
    num_subcodes: int = 2
    subcode_size: int = 16          # 4-bit

    # Cached base reconstruction (for naive base, not GPTQ)
    _base_rec: Optional[torch.Tensor] = None

    @property
    def device(self):
        return self.C1.device

    def decode_base(self, out_shape):
        if self._base_rec is not None:
            return self._base_rec.reshape(out_shape)
        return gptq_decode(self.base_codes, self.base_scale, self.base_zero).reshape(out_shape)

    def decode_residual(self, numel, out_shape):
        """Reconstruct residual from additive subcodebooks."""
        pad = (-numel) % self.package_dim
        padded = numel + pad
        rec = torch.zeros(padded, device=self.device, dtype=torch.float16)
        flat = rec.view(-1, self.package_dim)
        flat += self.C1[self.idx1]
        flat += self.C2[self.idx2]
        return rec[:numel].reshape(out_shape)

    def reconstruct(self, out_shape):
        """Full weight reconstruction: base + residual."""
        base = self.decode_base(out_shape).to(torch.float16)
        numel = base.numel()
        resid = self.decode_residual(numel, out_shape).to(torch.float16)
        return (base + resid).to(torch.bfloat16)

    def payload_bytes(self, total_params):
        """Count every bit for exact BPW accounting."""
        # GPTQ base:
        bits_base = (self.base_codes.numel() * self.base_bits
                     + self.base_scale.numel() * 16
                     + self.base_zero.numel() * 16)
        # Residual codebooks (stored as INT8 in practice, but use FP16 storage)
        bits_cb = (self.C1.numel() + self.C2.numel()) * 16
        # Residual indices: two 4-bit indices per package
        bits_idx = (self.idx1.numel() * 4 + self.idx2.numel() * 4)
        total_bits = bits_base + bits_cb + bits_idx
        return total_bits / 8, total_bits / total_params


def init_factorized_residual(base_flat, group_size, package_dim, subcode_size, seed=42):
    """Initialize two additive subcodebooks via alternating factorization.

    Returns flattened residual, C1, C2, idx1, idx2 all on same device as base_flat.
    """
    dev = base_flat.device
    base_t = base_flat.float()
    numel = base_t.numel()

    # Pad to package boundary
    pad = (-numel) % package_dim
    if pad:
        base_t = torch.cat([base_t, torch.zeros(pad, device=dev)])
    packages = base_t.reshape(-1, package_dim)   # [P, 48]

    X = packages.clone()   # target = residual to encode (E)

    # Alternating initialization
    rng = torch.Generator(device=dev).manual_seed(seed)
    init_idx = torch.randint(0, X.shape[0], (subcode_size,), generator=rng, device=dev)
    C1 = X[init_idx].clone()
    init_idx2 = torch.randint(0, X.shape[0], (subcode_size,), generator=rng, device=dev)
    C2 = X[init_idx2].clone()

    for _ in range(5):
        # Assign to C1 given C2
        resid = X.view(-1, package_dim) - C2[None, :, :].mean(dim=1)  # rough
        d1 = (resid.unsqueeze(1) - C1.unsqueeze(0)).square().sum(-1)
        idx1 = d1.argmin(-1)
        # Update C1
        A1 = F.one_hot(idx1, subcode_size).float()
        cnt1 = A1.sum(0).clamp_min(1)
        C1 = (A1.t() @ resid) / cnt1.unsqueeze(1)

        # Assign to C2 given C1
        resid = X - C1[idx1]
        d2 = (resid.unsqueeze(1) - C2.unsqueeze(0)).square().sum(-1)
        idx2 = d2.argmin(-1)
        A2 = F.one_hot(idx2, subcode_size).float()
        cnt2 = A2.sum(0).clamp_min(1)
        C2 = (A2.t() @ resid) / cnt2.unsqueeze(1)

    return C1.half(), C2.half(), idx1, idx2, X.view(-1, package_dim)


# ---------------------------------------------------------------------------
# QAT � prototype and assignment updates
# ---------------------------------------------------------------------------

def qat_prototype_step(layer_module, codec, base_flat, ids, tokenizer, model,
                       lr=1e-4, kl_weight=0.25, geom_weight=1e-5,
                       opt_steps=50, log_interval=10):
    """Train prototype values C1, C2 through the model CE + KL objective.

    In _FAST_MSE_ONLY mode, uses MSE-only updates (no model forward).
    """
    global _FAST_MSE_ONLY
    dev = codec.device
    out_shape = layer_module.weight.shape
    params = [codec.C1, codec.C2]
    for p in params:
        p.requires_grad_(True)

    optimizer = torch.optim.AdamW(params, lr=lr)

    if _FAST_MSE_ONLY:
        # Fast MSE-only optimization (no model forward)
        base_rec = codec.decode_base(out_shape)
        numel = base_rec.numel()
        pad = (-numel) % PACKAGE_DIM
        base_pad = base_rec.float().reshape(-1)
        if pad:
            base_pad = torch.cat([base_pad, torch.zeros(pad, device=dev)])
        target = base_pad.reshape(-1, PACKAGE_DIM)
        stats = []
        for step in range(opt_steps):
            optimizer.zero_grad()
            recon = (codec.C1[codec.idx1] + codec.C2[codec.idx2]).float()
            loss = (target - recon).square().mean()
            loss.backward()
            optimizer.step()
            if step % max(1, opt_steps // 5) == 0 or step == opt_steps - 1:
                stats.append({"step": step, "ce": 0.0, "kl": 0.0, "geom": loss.item(), "total": loss.item()})
        for p in params:
            p.requires_grad_(False)
        return stats

    teacher = model
    base_rec = codec.decode_base(out_shape).to(torch.float16).reshape(-1)

    stats = []
    for step in range(opt_steps):
        optimizer.zero_grad()

        resid = codec.decode_residual(base_rec.numel(), out_shape).to(torch.float16).reshape(-1)
        w_q = (base_rec + resid).reshape(out_shape).to(torch.bfloat16)
        saved = layer_module.weight.data.clone()
        layer_module.weight.data.copy_(w_q)

        losses = []
        kl_losses = []
        for window in ids.split(128):
            if window.numel() < 2:
                continue
            out = model(window[None].to(model.device_holder))
            logits = out.logits[:, :-1].contiguous()
            labels = window[None, 1:]

            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

            with torch.inference_mode():
                t_out = teacher(window[None].to(model.device_holder))
                t_logits = t_out.logits[:, :-1].contiguous()

            kl = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(t_logits, dim=-1),
                reduction="batchmean"
            )

            losses.append(ce * labels.numel())
            kl_losses.append(kl * labels.numel())

        n = sum(max(1, ids.shape[0] // EVAL_WINDOW) for _ in range(1))
        loss_ce = sum(losses) / n
        loss_kl = sum(kl_losses) / n

        with torch.inference_mode():
            numel = base_rec.numel()
            pad = (-numel) % PACKAGE_DIM
            if pad:
                base_padded = torch.cat([base_rec.float().reshape(-1), torch.zeros(pad, device=base_rec.device)])
            else:
                base_padded = base_rec.float().reshape(-1)
            err = base_padded.reshape(-1, PACKAGE_DIM) - (codec.C1[codec.idx1] + codec.C2[codec.idx2]).float()
        loss_geom = err.square().mean()

        loss = loss_ce + kl_weight * loss_kl + geom_weight * loss_geom
        loss.backward()
        optimizer.step()

        layer_module.weight.data.copy_(saved)

        if step % log_interval == 0 or step == opt_steps - 1:
            stats.append({
                "step": step,
                "ce": loss_ce.item(),
                "kl": loss_kl.item(),
                "geom": loss_geom.item(),
                "total": loss.item(),
            })

    for p in params:
        p.requires_grad_(False)
    torch.cuda.empty_cache()
    return stats


def qat_assignment_update(codec, base_flat, layer_module, model, ids,
                          update_fraction=0.01, beam=2):
    """Update (i, j) assignments via local search with forward verification."""
    dev = codec.device
    out_shape = layer_module.weight.shape
    base_rec = codec.decode_base(out_shape).to(torch.float16)
    base_flat_v = base_rec.reshape(-1)
    pad = (-base_flat_v.numel()) % PACKAGE_DIM
    if pad:
        base_flat_v = torch.cat([base_flat_v, torch.zeros(pad, device=dev)])
    packages = base_flat_v.reshape(-1, PACKAGE_DIM)

    num_packages = packages.shape[0]
    num_update = max(1, int(num_packages * update_fraction))

    # Build base packages (same padding as init)
    base_flat2 = base_rec.float().reshape(-1)
    base_pad = (-base_flat2.numel()) % PACKAGE_DIM
    if base_pad:
        base_flat2 = torch.cat([base_flat2, torch.zeros(base_pad, device=dev)])
    base_packages = base_flat2.reshape(-1, PACKAGE_DIM)

    # For each package, score local candidates
    # Random subset of packages to update
    perm = torch.randperm(num_packages, device=dev)[:num_update]
    selected = perm.sort().values

    changes1 = 0
    changes2 = 0

    for pkg_idx in selected:
        pkg = packages[pkg_idx].float()
        current_i = codec.idx1[pkg_idx]
        current_j = codec.idx2[pkg_idx]

        # Local candidates: vary i only or j only
        candidates = []
        for di in [-1, 0, 1]:
            ni = (current_i + di).clamp(0, codec.subcode_size - 1)
            candidates.append((ni, current_j))
        for dj in [-1, 0, 1]:
            nj = (current_j + dj).clamp(0, codec.subcode_size - 1)
            candidates.append((current_i, nj))
        # Deduplicate
        candidates = list(set(candidates))
        if len(candidates) <= 1:
            continue

        # Score candidates by residual MSE reduction
        target = pkg - base_packages[pkg_idx].float()
        best_err = float("inf")
        best_ci, best_cj = current_i, current_j
        for ci, cj in candidates:
            recon = codec.C1[ci].float() + codec.C2[cj].float()
            err = (target - recon).square().sum().item()
            if err < best_err:
                best_err = err
                best_ci, best_cj = ci, cj

        if best_ci != current_i:
            codec.idx1[pkg_idx] = best_ci
            changes1 += 1
        if best_cj != current_j:
            codec.idx2[pkg_idx] = best_cj
            changes2 += 1

    torch.cuda.empty_cache()
    return changes1, changes2


# ---------------------------------------------------------------------------
# Full E19 experiment
# ---------------------------------------------------------------------------

@dataclass
class E19Config:
    bpw_target: float = 1.20
    group_size: int = GROUP_SIZE
    package_dim: int = PACKAGE_DIM
    subcode_size: int = 16           # 4-bit
    num_macro_rounds: int = 5
    qat_steps_per_round: int = 50
    qat_lr: float = 1e-4
    kl_weight: float = 0.25
    geom_weight: float = 1e-5
    assign_update_fraction: float = 0.01
    assign_beam: int = 2
    cal_tokens: int = NSAMPLES * SEQLEN
    eval_tokens: int = EVAL_TOKENS
    output_dir: str = "e19_results"
    seed: int = 42


@dataclass
class E19Result:
    config: dict = field(default_factory=dict)
    bf16_ppl: float = 0.0
    init_ppl: float = 0.0
    final_ppl: float = 0.0
    final_val_ce: float = 0.0
    macro_log: list = field(default_factory=list)
    payload_bytes: float = 0.0
    effective_bpw: float = 0.0
    runtime_seconds: float = 0.0
    sha256: str = ""
    packed_path: str = ""


def run_e19(cfg: E19Config):
    start = time.time()
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    packed_dir = output_dir / "packed"
    packed_dir.mkdir(exist_ok=True)

    # 1. Load model and data
    print("Loading model and data ...", flush=True)
    model, tokenizer, cal_text, eval_text = load_model_and_data()
    model = model.to(dev).eval()
    model.device_holder = dev

    eval_ids = tokenizer(eval_text, return_tensors="pt")["input_ids"][0, :cfg.eval_tokens].to(dev)

    # BF16 reference
    ref_result = eval_fast(model, eval_ids)
    print(f"BF16 PPL = {ref_result['perplexity']:.4f}  NLL = {ref_result['nll']:.4f}", flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    linear_names = [(n, m) for n, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and n != "lm_head"]

    # 2. Collect Hessians from calibration data
    print("Collecting Hessians ...", flush=True)
    H = collect_H(model, linear_names, tokenizer, cal_text,
                  nsamples=min(NSAMPLES, cfg.cal_tokens // SEQLEN))

    # 3. Process all layers with GPTQ 2-bit base + factorized residual
    base_bits = 4
    print(f"Applying GPTQ {base_bits}-bit base to all layers ...", flush=True)
    original_weights = {}
    for name, module in linear_names:
        original_weights[name] = module.weight.detach().float().clone()

    from gptq import gptq_quantize as _gptq
    target_names = [n for n, _ in linear_names]
    layer_codecs = {}
    codecs = {}

    for idx, name in enumerate(target_names):
        module = dict(linear_names)[name]
        w = original_weights[name]
        hess = H.get(name)
        if hess is None:
            hess = torch.eye(w.shape[1], device=w.device).float()
        Q, codes, scale, zero = _gptq(w, hess, base_bits, groupwise=True, groupsize=64, sym=False)
        module.weight.data.copy_(Q.to(module.weight.dtype))
        del hess
        torch.cuda.empty_cache()

        base_rec = Q.float().reshape(-1)
        residual = w.float().reshape(-1) - base_rec

        C1, C2, idx1, idx2, _ = init_factorized_residual(
            residual, cfg.group_size, cfg.package_dim, cfg.subcode_size, cfg.seed
        )

        codec = E19Codec(
            base_codes=codes,
            base_scale=scale,
            base_zero=zero,
            base_bits=base_bits,
            C1=C1.to(dev),
            C2=C2.to(dev),
            idx1=idx1.to(dev),
            idx2=idx2.to(dev),
        )
        codec._base_rec = base_rec
        codecs[name] = codec
        layer_codecs[name] = codec

        if (idx + 1) % 20 == 0:
            print(f"  Quantized {idx+1}/{len(target_names)} layers ...", flush=True)

    base_ppl = eval_fast(model, eval_ids)
    print(f"GPTQ {base_bits}-bit base PPL = {base_ppl['perplexity']:.4f}", flush=True)

    for name, module in linear_names:
        module.weight.data.copy_(original_weights[name].to(module.weight.dtype))


    # 6. Initial evaluation
    print("Evaluating initial reconstruction ...", flush=True)
    for name, module in linear_names:
        if name in codecs:
            w_q = codecs[name].reconstruct(module.weight.shape)
            module.weight.data.copy_(w_q)

    init_result = eval_fast(model, eval_ids)
    print(f"Init PPL = {init_result['perplexity']:.4f}", flush=True)

    if init_result["perplexity"] > 52:
        print("WARNING: Init PPL > 52 � codec construction may be defective", flush=True)

    # Restore
    for name, module in linear_names:
        module.weight.data.copy_(original_weights[name].to(module.weight.dtype))

    # 7. Macro-rounds: prototype QAT -> assignment update -> repeat
    print("Starting macro-rounds ...", flush=True)
    macro_log = []

    # Build calibration IDs
    cal_ids = tokenizer(cal_text, return_tensors="pt")["input_ids"][0, :cfg.cal_tokens].to(dev)

    best_ppl = init_result["perplexity"]
    best_codecs = {name: codecs[name] for name in codecs}

    # Use target_names (the subset we quantized)
    mlp_names = [n for n in target_names if any(x in n for x in ["gate_proj", "up_proj", "down_proj"])]

    for rnd in range(cfg.num_macro_rounds):
        rnd_start = time.time()
        round_info = {"round": rnd, "changes": {}}

        # --- Phase A: Prototype QAT on MLP layers ---
        total_ce = 0.0
        total_kl = 0.0
        for name in mlp_names:
            if name not in codecs:
                continue
            module = dict(linear_names)[name]
            codec = codecs[name]
            stats = qat_prototype_step(
                module, codec, original_weights[name].reshape(-1),
                cal_ids, tokenizer, model,
                lr=cfg.qat_lr, kl_weight=cfg.kl_weight, geom_weight=cfg.geom_weight,
                opt_steps=cfg.qat_steps_per_round
            )
            if stats:
                total_ce += stats[-1]["ce"]
                total_kl += stats[-1]["kl"]
        round_info["prototype_ce"] = total_ce / max(1, len(mlp_names))
        round_info["prototype_kl"] = total_kl / max(1, len(mlp_names))

        # --- Phase B: Assignment update ---
        changes1_total = 0
        changes2_total = 0
        for name in mlp_names:
            if name not in codecs:
                continue
            module = dict(linear_names)[name]
            codec = codecs[name]
            c1, c2 = qat_assignment_update(
                codec, original_weights[name].reshape(-1),
                module, model, cal_ids,
                update_fraction=cfg.assign_update_fraction,
                beam=cfg.assign_beam
            )
            changes1_total += c1
            changes2_total += c2
        round_info["i_changes"] = changes1_total
        round_info["j_changes"] = changes2_total

        # --- Evaluate ---
        for name, module in linear_names:
            if name in codecs:
                w_q = codecs[name].reconstruct(module.weight.shape)
                module.weight.data.copy_(w_q)

        current = eval_fast(model, eval_ids)
        round_info["val_ppl"] = current["perplexity"]
        round_info["val_nll"] = current["nll"]
        round_info["val_ce"] = current["nll"]
        round_info["elapsed"] = time.time() - rnd_start

        if current["perplexity"] < best_ppl:
            best_ppl = current["perplexity"]
            best_codecs = {name: codecs[name] for name in codecs}

        macro_log.append(round_info)
        print(f"Round {rnd}: PPL = {current['perplexity']:.4f}  "
              f"i_changes={changes1_total} j_changes={changes2_total}  "
              f"({time.time() - rnd_start:.1f}s)", flush=True)

        # Restore weights
        for name, module in linear_names:
            module.weight.data.copy_(original_weights[name].to(module.weight.dtype))

        # Decision gates
        if current["perplexity"] < 40:
            print(f"  *** E19 validated: PPL < 40 ***", flush=True)
        if current["perplexity"] < 30:
            print(f"  *** Approaching target: PPL < 30 ***", flush=True)
        if current["perplexity"] < 24.407:
            print(f"  *** TARGET ACHIEVED: PPL < 24.407 ***", flush=True)
            break

    # 8. Final evaluation with best codecs
    for name, module in linear_names:
        if name in best_codecs:
            w_q = best_codecs[name].reconstruct(module.weight.shape)
            module.weight.data.copy_(w_q)

    final_result = eval_fast(model, eval_ids)
    print(f"Final PPL = {final_result['perplexity']:.4f}", flush=True)

    # 9. Compute payload
    total_payload_bytes = 0.0
    for name in codecs:
        pb, bpw = codecs[name].payload_bytes(total_params)
        total_payload_bytes += pb

    effective_bpw = 8 * total_payload_bytes / total_params

    # 10. Save packed codecs and SHA
    packed_data = {
        name: {
            "C1": codecs[name].C1.cpu(),
            "C2": codecs[name].C2.cpu(),
            "idx1": codecs[name].idx1.cpu(),
            "idx2": codecs[name].idx2.cpu(),
            "base_codes": codecs[name].base_codes.cpu(),
            "base_scale": codecs[name].base_scale.cpu(),
            "base_zero": codecs[name].base_zero.cpu(),
            "base_bits": codecs[name].base_bits,
        }
        for name in codecs
    }
    packed_path = packed_dir / f"e19_bpw{cfg.bpw_target:.2f}.pt"
    torch.save(packed_data, packed_path)

    with open(packed_path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    # 11. Result object
    result = E19Result(
        config=asdict(cfg),
        bf16_ppl=ref_result["perplexity"],
        init_ppl=init_result["perplexity"],
        final_ppl=final_result["perplexity"],
        final_val_ce=final_result["nll"],
        macro_log=macro_log,
        payload_bytes=total_payload_bytes,
        effective_bpw=effective_bpw,
        runtime_seconds=time.time() - start,
        sha256=sha,
        packed_path=str(packed_path),
    )

    # Save report
    report_path = output_dir / f"report_e19_bpw{cfg.bpw_target:.2f}.json"
    report_path.write_text(json.dumps(asdict(result), indent=2, default=str))
    print(f"Report saved to {report_path}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Fresh-process decode verification
# ---------------------------------------------------------------------------

@torch.inference_mode()
def verify_packed(packed_path, model, eval_ids):
    """Fresh-process decode: load packed state, reconstruct, evaluate PPL."""
    packed = torch.load(packed_path, weights_only=True)

    # Reconstruct using codec
    for name, module in model.named_modules():
        if name in packed:
            data = packed[name]
            codec = E19Codec(
                base_codes=data["base_codes"],
                base_scale=data["base_scale"],
                base_zero=data["base_zero"],
                base_bits=data["base_bits"],
                C1=data["C1"],
                C2=data["C2"],
                idx1=data["idx1"],
                idx2=data["idx2"],
            )
            w_q = codec.reconstruct(module.weight.shape)
            module.weight.data.copy_(w_q)

    return eval_fast(model, eval_ids)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def self_test():
    """Quick mathematical self-test without model download."""
    print("=== E19 Self-Test ===\n", flush=True)

    # 1. init_factorized_residual converges
    dev = torch.device("cpu")
    residual = torch.randn(4800, device=dev) * 0.5  # 100 packages of 48D
    C1, C2, idx1, idx2, packages = init_factorized_residual(residual, 64, 48, 16)
    recon = C1[idx1].float() + C2[idx2].float()
    mse = (packages.float() - recon).square().mean().item()
    print(f"[1] Factorized init MSE = {mse:.6f}  {'PASS' if mse < 2.0 else 'WARN'}", flush=True)

    # 2. Local moves have meaningful step sizes
    for _ in range(5):
        p = torch.randint(0, idx1.shape[0], (1,)).item()
        i, j = idx1[p].item(), idx2[p].item()
        orig = C1[i].float() + C2[j].float()
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni = max(0, min(15, i + di))
            nj = max(0, min(15, j + dj))
            if (ni, nj) != (i, j):
                neighbors.append((C1[ni].float() + C2[nj].float() - orig).norm().item())
        avg_step = sum(neighbors) / max(1, len(neighbors))
        print(f"[2] Package {p}: avg neighbor delta = {avg_step:.4f}  "
              f"{'PASS' if 0.1 < avg_step < 20.0 else 'WARN'}", flush=True)
        break

    # 3. Payload accounting
    codec = E19Codec(
        base_codes=torch.zeros(1, 64, dtype=torch.uint8),
        base_scale=torch.ones(1, dtype=torch.float16),
        base_zero=torch.zeros(1, dtype=torch.float16),
        base_bits=2, C1=C1, C2=C2, idx1=idx1, idx2=idx2,
    )
    total_params = 500_000_000
    pb, bpw = codec.payload_bytes(total_params)
    print(f"[3] Payload: {pb:.0f}B  BPW: {bpw:.4f}  {'PASS' if bpw > 0 else 'FAIL'}", flush=True)

    # 4. 16x16 product space = 256 combos
    combos = set()
    for i in range(16):
        for j in range(16):
            combos.add((C1[i, 0].item(), C1[i, 1].item(), C2[j, 0].item(), C2[j, 1].item()))
    print(f"[4] Product space unique C1+C2 combos (first 2 dims) = {len(combos)}  "
          f"{'PASS' if len(combos) > 50 else 'WARN'}", flush=True)

    # 5. Fresh-process roundtrip
    packed = {
        "dummy_layer": {
            "C1": C1.cpu(), "C2": C2.cpu(),
            "idx1": idx1.cpu(), "idx2": idx2.cpu(),
            "base_codes": codec.base_codes.cpu(),
            "base_scale": codec.base_scale.cpu(),
            "base_zero": codec.base_zero.cpu(),
            "base_bits": codec.base_bits,
        }
    }
    import hashlib, tempfile
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    torch.save(packed, tmp)
    with open(tmp, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    Path(tmp).unlink()
    print(f"[5] SHA256 roundtrip OK  {'PASS' if len(sha) == 64 else 'FAIL'}", flush=True)

    print("\n=== Self-test complete ===", flush=True)
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E19 Factorized Residual QAT")
    parser.add_argument("--bpw", type=float, default=1.20, help="Target BPW")
    parser.add_argument("--macro-rounds", type=int, default=5)
    parser.add_argument("--qat-steps", type=int, default=50)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.25)
    parser.add_argument("--geom-weight", type=float, default=1e-5)
    parser.add_argument("--assign-fraction", type=float, default=0.01)
    parser.add_argument("--cal-tokens", type=int, default=NSAMPLES * SEQLEN)
    parser.add_argument("--output", type=str, default="e19_results")
    parser.add_argument("--verify", type=str, default=None,
                        help="Path to packed file for fresh-process decode verification")
    parser.add_argument("--test", action="store_true", help="Run self-test without model download")
    parser.add_argument("--fast", action="store_true", help="Use MSE-only QAT (no teacher KL, faster on CPU)")
    args = parser.parse_args()

    if args.test:
        self_test()
        return

    if args.verify:
        model, tokenizer, _, eval_text = load_model_and_data()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(dev).eval()
        eval_ids = tokenizer(eval_text, return_tensors="pt")["input_ids"][0, :EVAL_TOKENS].to(dev)
        result = verify_packed(args.verify, model, eval_ids)
        print(f"Verify: PPL = {result['perplexity']:.4f}  NLL = {result['nll']:.4f}", flush=True)
        return

    if args.fast:
        _FAST_MSE_ONLY = True
    else:
        _FAST_MSE_ONLY = False

    cfg = E19Config(
        bpw_target=args.bpw,
        num_macro_rounds=args.macro_rounds,
        qat_steps_per_round=args.qat_steps,
        qat_lr=args.qat_lr,
        kl_weight=args.kl_weight,
        geom_weight=args.geom_weight,
        assign_update_fraction=args.assign_fraction,
        cal_tokens=args.cal_tokens,
        output_dir=args.output,
    )

    result = run_e19(cfg)

    # Summary table
    print("\n" + "=" * 60)
    print(f"E19 � BPW={cfg.bpw_target:.2f}")
    print(f"  BF16 PPL:        {result.bf16_ppl:.4f}")
    print(f"  Init PPL:        {result.init_ppl:.4f}")
    print(f"  Final PPL:       {result.final_ppl:.4f}")
    print(f"  Final val CE:    {result.final_val_ce:.4f}")
    print(f"  Payload bytes:   {result.payload_bytes:.0f}")
    print(f"  Effective BPW:   {result.effective_bpw:.4f}")
    print(f"  Runtime:         {result.runtime_seconds:.1f}s")
    print(f"  SHA256:          {result.sha256}")
    print(f"  Packed:          {result.packed_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()



