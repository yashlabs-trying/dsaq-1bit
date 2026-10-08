"""
Whole-block pattern codebook codec for Qwen2.5-0.5B.
Split weights into 4x4 blocks (16 weights each), cluster into codebooks
per projection family, reconstruct from codebook IDs.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import time
import gc
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_H, BLOCK_W = 4, 4  # 16 weights per block
K_CODEBOOK = 65536        # 16-bit IDs
N_TRAIN_SAMPLES = 500000  # per family
N_KMEANS_ITER = 30
BATCH_SIZE_ASSIGN = 1024   # blocks per assignment batch
CODEBOOK_DTYPE = torch.float16
STORE_CODEBOOK_BITS = 8   # per prototype value (quantized for storage)
SHARE_CODEBOOK = True      # share one codebook across all families
USE_BLOCK_SCALES = False   # per-block scale factor (0.5 BPW extra)

print(f"Device: {DEVICE}")
print(f"Block: {BLOCK_H}x{BLOCK_W} = {BLOCK_H*BLOCK_W} weights")
print(f"Codebook K: {K_CODEBOOK} ({int(math.log2(K_CODEBOOK))} bits/block = {int(math.log2(K_CODEBOOK))/(BLOCK_H*BLOCK_W):.4f} BPW)")

# ──────────────────────────────────────────────
# 1. MODEL LOADING
# ──────────────────────────────────────────────
def load_model_and_extract_weights(num_layers=4):
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")
    model.eval()
    weights = {}
    for i in range(num_layers):
        layer = model.model.layers[i]
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            w = getattr(layer.self_attn, proj).weight.data.float()
            weights[f"layer_{i}.self_attn.{proj}"] = w
        for proj in ['gate_proj', 'up_proj', 'down_proj']:
            w = getattr(layer.mlp, proj).weight.data.float()
            weights[f"layer_{i}.mlp.{proj}"] = w
    return model, weights

# ──────────────────────────────────────────────
# 2. BLOCK EXTRACTION
# ──────────────────────────────────────────────
def extract_blocks(weights, bh=4, bw=4):
    """
    Split each weight matrix into bh x bw blocks.
    Returns dict: family -> list of (block_tensor, metadata)
    metadata = (name, row_idx, col_idx, H, W)
    """
    families = defaultdict(list)
    
    for name, W in weights.items():
        H, W_shape = W.shape
        parts = name.split(".")
        proj = parts[2]
        
        if proj in ['q_proj', 'k_proj', 'v_proj']:
            family = 'qkv'
        elif proj == 'o_proj':
            family = 'o'
        elif proj in ['gate_proj', 'up_proj']:
            family = 'gateup'
        elif proj == 'down_proj':
            family = 'down'
        else:
            family = 'other'
        
        # Split into blocks
        for i in range(0, H, bh):
            for j in range(0, W_shape, bw):
                block = W[i:i+bh, j:j+bw]
                # Pad if necessary (shouldn't be needed for divisible dims)
                if block.shape != (bh, bw):
                    padded = torch.zeros(bh, bw)
                    h, w = block.shape
                    padded[:h, :w] = block
                    block = padded
                families[family].append((block.flatten(), name, i, j))
    
    return dict(families)

def blocks_to_tensor(family_blocks):
    """Convert list of (block_tensor, meta) to (N, 16) tensor."""
    return torch.stack([b[0] for b in family_blocks])

# ──────────────────────────────────────────────
# 3. WAVEREL FEATURES PER BLOCK
# ──────────────────────────────────────────────
def compute_wavarel_features(block):
    """
    Compute WaveRel features for a 4x4 block (flattened to 16 values).
    Returns dict of features for analysis.
    """
    x = block.view(4, 4)
    
    # Sign pattern (16-bit mask)
    sign_mask = (x >= 0).int()
    sign_int = 0
    for k in range(16):
        sign_int |= int(sign_mask.flatten()[k].item()) << k
    
    # Magnitudes
    mag = x.abs()
    mag_norm = mag / (mag.sum() + 1e-8)
    
    # Statistics
    mean = x.mean().item()
    std = x.std().item()
    
    # Normalized magnitudes (rescaled to [0,1])
    if mag.max() > 0:
        mag_scaled = (mag / mag.max()).flatten()
    else:
        mag_scaled = mag.flatten()
    
    # Small Haar features (1-level on 4x4)
    LL = mag[:2, :2].mean().item()
    LH = (mag[:2, :2] - mag[:2, 2:]).abs().mean().item()
    HL = (mag[:2, :2] - mag[2:, :2]).abs().mean().item()
    HH = (mag[:2, :2] - mag[2:, 2:]).abs().mean().item()
    
    # Direction (normalized vector)
    norm = x.norm().item()
    direction = (x / (norm + 1e-8)).flatten() if norm > 0 else torch.zeros(16)
    
    # Pairwise ratios (first few)
    ratios = []
    for k in range(7):
        if mag.flatten()[k+1].item() > 1e-8:
            ratios.append((mag.flatten()[k] / mag.flatten()[k+1]).item())
        else:
            ratios.append(0.0)
    
    return {
        'sign_int': sign_int,
        'sign_mask': sign_mask,
        'mag_norm': mag_norm,
        'mean': mean,
        'std': std,
        'mag_scaled': mag_scaled,
        'haar': [LL, LH, HL, HH],
        'norm': norm,
        'direction': direction,
        'ratios': ratios,
    }

# ──────────────────────────────────────────────
# 4. K-MEANS ON GPU
# ──────────────────────────────────────────────
def gpu_kmeans(data, K, n_iter=30, batch_size=4096, verbose=True):
    """
    K-means on GPU with batched assignment.
    data: (N, D) tensor on CPU
    Returns centroids: (K, D), assignments: (N,) indices
    """
    N, D = data.shape
    device = DEVICE
    
    # Move data to GPU
    data_gpu = data.to(device)
    
    # Initialize centroids with random data points
    idxs = torch.randperm(N, device=device)[:K]
    centroids = data_gpu[idxs].clone()
    
    for it in range(n_iter):
        t0 = time.time()
        
        # Assignment step (batched)
        assignments = torch.empty(N, dtype=torch.long, device=device)
        
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = data_gpu[start:end]  # (B, D)
            
            # Compute distances: ||batch - c||^2 for all centroids
            # Using: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            batch_norm = (batch ** 2).sum(dim=1, keepdim=True)  # (B, 1)
            cent_norm = (centroids ** 2).sum(dim=1, keepdim=True).T  # (1, K)
            dot = batch @ centroids.T  # (B, K)
            
            dists = batch_norm + cent_norm - 2 * dot  # (B, K)
            assignments[start:end] = dists.argmin(dim=1)
        
        # Update step - compute per-dimension sums using scatter_add
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device, dtype=torch.long)
        ones = torch.ones(N, device=device, dtype=torch.long)
        counts.scatter_add_(0, assignments, ones)
        
        # Compute sum per centroid per dimension
        for d in range(centroids.shape[1]):
            col_vals = data_gpu[:, d].contiguous()
            summed = torch.zeros(K, device=device)
            summed.scatter_add_(0, assignments, col_vals)
            new_centroids[:, d] = summed / counts.float().clamp(min=1)
        
        # Keep dead centroids unchanged
        dead = counts == 0
        new_centroids[dead] = centroids[dead]
        
        # Check convergence
        shift = ((centroids - new_centroids) ** 2).sum().item()
        centroids = new_centroids
        
        if verbose:
            # Count live centroids
            n_live = (counts > 0).sum().item()
            print(f"  K-means iter {it+1}: shift={shift:.6f}, live={n_live}/{K}, time={time.time()-t0:.1f}s")
        
        if shift < 1e-8:
            if verbose: print(f"  Converged at iter {it+1}")
            break
    
    return centroids.cpu(), assignments.cpu()


def encode_blocks(blocks_tensor, centroids, batch_size=1024):
    """
    Assign each block to nearest centroid.
    blocks_tensor: (N, D)
    centroids: (K, D)
    Returns assignments: (N,)
    """
    N = blocks_tensor.shape[0]
    K = centroids.shape[0]
    device = DEVICE
    
    blocks_gpu = blocks_tensor.to(device)
    cents_gpu = centroids.to(device)
    
    # Precompute centroid norms
    cent_norm = (cents_gpu ** 2).sum(dim=1)  # (K,)
    
    assignments = torch.empty(N, dtype=torch.long, device='cpu')
    
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = blocks_gpu[start:end]  # (B, D)
            
            batch_norm = (batch ** 2).sum(dim=1, keepdim=True)  # (B, 1)
            dot = batch @ cents_gpu.T  # (B, K)
            dists = batch_norm + cent_norm.unsqueeze(0) - 2 * dot
            assignments[start:end] = dists.argmin(dim=1).cpu()
    
    return assignments


def compute_reconstruction_error(blocks_tensor, centroids, assignments):
    """Compute MSE between original blocks and their centroid reconstructions."""
    recon = centroids[assignments]
    mse = ((blocks_tensor - recon) ** 2).mean().item()
    return mse, recon


# ──────────────────────────────────────────────
# 5. FULL PIPELINE
# ──────────────────────────────────────────────
def quantize_prototypes(centroids, n_bits=8):
    """Quantize prototype values to n_bits for storage.
    Returns: q_centroids (int8 or int16), scale, offset"""
    c = centroids.float()
    c_min = c.min()
    c_max = c.max()
    q_max = 2 ** (n_bits - 1) - 1
    scale = (c_max - c_min) / (2 * q_max) + 1e-8
    offset = (c_max + c_min) / 2
    q = torch.round((c - offset) / scale).clamp(-q_max, q_max).to(torch.int8 if n_bits <= 8 else torch.int16)
    return q, scale.item(), offset.item()

def dequantize_prototypes(q, scale, offset):
    return q.float() * scale + offset


class BlockCodec:
    def __init__(self, K=65536, share_codebook=True, use_scales=False):
        self.K = K
        self.share_codebook = share_codebook
        self.use_scales = use_scales
        self.codebook = None           # shared centroids (K, D)
        self.codebook_q = None         # quantized for storage
        self.codebook_scale = None
        self.codebook_offset = None
        self.family_codebooks = {}     # per-family codebooks (only if not shared)
        self.family_assignments = {}   # family -> assignments tensor
        self.family_scales = {}        # family -> per-block scales (if use_scales)
        self.block_meta = {}           # family -> list of (name, row, col)
        self.bpw_report = {}
        self.total_elements = 0
    
    def train(self, weights, n_train_samples=N_TRAIN_SAMPLES):
        print("\nExtracting blocks...")
        families = extract_blocks(weights)
        
        self.total_elements = sum(w.numel() for w in weights.values())
        self.block_meta = {}
        all_blocks = []
        all_family_ids = []
        
        # First pass: collect all blocks
        for family, blocks in families.items():
            block_tensor = blocks_to_tensor(blocks)
            meta = [(b[1], b[2], b[3]) for b in blocks]  # (name, row, col)
            self.block_meta[family] = (meta, block_tensor.shape[1], len(blocks))
            all_blocks.append(block_tensor)
            all_family_ids.extend([family] * len(blocks))
        
        # Concatenate all blocks for shared training
        all_data = torch.cat(all_blocks, dim=0)
        N_total, D = all_data.shape
        print(f"Total blocks: {N_total}, Dim: {D}")
        
        # Train shared codebook
        train_N = min(N_total, n_train_samples * 4)
        if train_N < N_total:
            idxs = torch.randperm(N_total)[:train_N]
            train_data = all_data[idxs]
        else:
            train_data = all_data
        
        print(f"\nTraining shared codebook (K={self.K}) on {len(train_data)} blocks...")
        centroids, _ = gpu_kmeans(train_data, self.K, n_iter=N_KMEANS_ITER, batch_size=4096)
        self.codebook = centroids.to(torch.float16)
        
        # Quantize codebook for storage
        self.codebook_q, self.codebook_scale, self.codebook_offset = quantize_prototypes(centroids, STORE_CODEBOOK_BITS)
        
        # Encode all blocks per family
        print(f"\nEncoding all blocks...")
        for family in families:
            _, D, N = self.block_meta[family]
            print(f"  Family {family}: {N} blocks")
            
            mask = torch.tensor([f == family for f in all_family_ids], device='cpu')
            family_data = all_data[mask]
            
            assign = encode_blocks(family_data, centroids.float())
            self.family_assignments[family] = assign
            
            # Compute metrics
            recon_blocks = centroids.float()[assign]
            mse = ((family_data - recon_blocks) ** 2).mean().item()
            cos_sim = (family_data * recon_blocks).sum(dim=1) / (
                family_data.norm(dim=1) * recon_blocks.norm(dim=1) + 1e-8
            )
            unique_ids = len(assign.unique())
            
            # Per-block scales
            if self.use_scales:
                # Optimal scale: minimize ||data - scale * centroid||^2
                cent = centroids.float()[assign]
                dot = (family_data * cent).sum(dim=1)
                norm2 = (cent ** 2).sum(dim=1)
                scales = dot / (norm2 + 1e-8)
                # Quantize scale to 8 bits
                qs, ss, os = quantize_prototypes(scales.unsqueeze(1), 8)
                self.family_scales[family] = (qs, ss, os)
            
            # Nearest-prototype distances
            dists = ((family_data - recon_blocks) ** 2).sum(dim=1)
            
            print(f"    MSE: {mse:.8f}, Avg cos: {cos_sim.mean():.4f}, Util: {unique_ids}/{self.K}")
            print(f"    Avg dist: {dists.mean():.8f}")
        
        # Report overall metrics  
        # Compute overall cosine
        all_recon = []
        for family in families:
            mask = torch.tensor([f == family for f in all_family_ids])
            fd = all_data[mask]
            ar = centroids.float()[self.family_assignments[family]]
            if self.use_scales:
                s = dequantize_prototypes(self.family_scales[family][0], 
                                           self.family_scales[family][1],
                                           self.family_scales[family][2]).squeeze()
                ar = ar * s.unsqueeze(1)
            all_recon.append(ar)
        all_recon = torch.cat(all_recon, dim=0)
        
        overall_cos = (all_data * all_recon).sum(dim=1) / (
            all_data.norm(dim=1) * all_recon.norm(dim=1) + 1e-8
        )
        print(f"\n  Overall avg block cosine: {overall_cos.mean():.4f}")
        print(f"  Overall MSE: {((all_data - all_recon) ** 2).mean():.8f}")
    
    def reconstruct_weights(self, weights_orig):
        """Reconstruct all weight matrices from codebooks."""
        recon = {}
        centroids = self.codebook.float()
        
        for family in self.family_assignments:
            assign = self.family_assignments[family]
            meta, D, N = self.block_meta[family]
            
            recon_blocks = centroids[assign]
            
            if self.use_scales and family in self.family_scales:
                s = dequantize_prototypes(self.family_scales[family][0],
                                           self.family_scales[family][1],
                                           self.family_scales[family][2]).squeeze()
                recon_blocks = recon_blocks * s.unsqueeze(1)
            
            idx = 0
            for (name, row, col), block_vec in zip(meta, recon_blocks):
                if name not in recon:
                    H, W = weights_orig[name].shape
                    recon[name] = torch.zeros(H, W)
                
                block = block_vec.view(BLOCK_H, BLOCK_W)
                h = min(BLOCK_H, recon[name].shape[0] - row)
                w = min(BLOCK_W, recon[name].shape[1] - col)
                recon[name][row:row+h, col:col+w] = block[:h, :w]
                idx += 1
        
        return recon
    
    def compute_bpw(self):
        """Compute bit budget breakdown."""
        total_elements = self.total_elements
        n_blocks = sum(len(a) for a in self.family_assignments.values())
        id_bits_per_block = math.log2(self.K)
        id_bits = n_blocks * id_bits_per_block
        
        # Shared codebook storage
        K, D = self.codebook.shape
        cb_bits = K * D * STORE_CODEBOOK_BITS
        
        # Scale bits
        scale_bits = 0
        if self.use_scales:
            scale_bits = n_blocks * self.family_scales[list(self.family_scales.keys())[0]][0].shape[1] * 8
        
        # Metadata (minimal: just codebook scale/offset for dequant)
        meta_bits = 32 * 2  # scale + offset as float32
        
        total_bits = id_bits + cb_bits + scale_bits + meta_bits
        bpw = total_bits / total_elements
        
        self.bpw_report = {
            'total_bpw': bpw,
            'id_bpw': id_bits / total_elements,
            'codebook_bpw': cb_bits / total_elements,
            'scale_bpw': scale_bits / total_elements,
            'meta_bpw': meta_bits / total_elements,
            'n_blocks': n_blocks,
            'K': K,
            'D': D,
        }
        
        return self.bpw_report


# ──────────────────────────────────────────────
# 6. EVALUATION
# ──────────────────────────────────────────────
def evaluate_layer_ppls(model, tokenizer, weights_orig, weights_recon, device='cuda'):
    """Evaluate PPL after replacing weights layer by layer."""
    test_text = (
        "The transformer architecture has become the foundation of modern natural language processing. "
        "It processes sequences using attention mechanisms. "
        "This design eliminates the sequential bottleneck of recurrent networks. "
        "Large language models built on this architecture have achieved remarkable results."
    )
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        base_ppl = torch.exp(model(**inputs, labels=inputs["input_ids"]).loss).item()
    print(f"  Base PPL: {base_ppl:.2f}")
    
    ppls = [base_ppl]
    delta_ppls = [0.0]
    cosines = []
    
    with torch.no_grad():
        for i in range(4):
            layer = model.model.layers[i]
            
            # Replace weights
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                key = f"layer_{i}.self_attn.{proj}"
                if key in weights_recon:
                    target = getattr(layer.self_attn, proj)
                    target.weight.data = weights_recon[key].to(target.weight.dtype).to(device)
            
            for proj in ['gate_proj', 'up_proj', 'down_proj']:
                key = f"layer_{i}.mlp.{proj}"
                if key in weights_recon:
                    target = getattr(layer.mlp, proj)
                    target.weight.data = weights_recon[key].to(target.weight.dtype).to(device)
            
            # PPL
            outputs = model(**inputs, labels=inputs["input_ids"])
            ppl = torch.exp(outputs.loss).item()
            ppls.append(ppl)
            delta = (ppl - base_ppl) / base_ppl * 100
            delta_ppls.append(delta)
            
            # Cosine for this layer's weights
            layer_cos = []
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']:
                key = f"layer_{i}.mlp.{proj}" if 'mlp' in str(type(getattr(layer, 'mlp', None))) else f"layer_{i}.self_attn.{proj}"
                # Let me just check both
                for prefix in ['self_attn', 'mlp']:
                    key = f"layer_{i}.{prefix}.{proj}"
                    if key in weights_recon and key in weights_orig:
                        w_orig = weights_orig[key]
                        w_recon = weights_recon[key]
                        c = nn.functional.cosine_similarity(
                            w_orig.flatten().unsqueeze(0),
                            w_recon.flatten().unsqueeze(0)
                        ).item()
                        layer_cos.append(c)
            
            avg_cos = sum(layer_cos) / len(layer_cos) if layer_cos else 0
            cosines.append(avg_cos)
            
            print(f"  Layer {i+1}: PPL={ppl:.2f}, Δ={delta:.1f}%, cos={avg_cos:.4f}")
    
    return ppls, delta_ppls, cosines


def compute_overall_cosine(weights_orig, weights_recon):
    """Compute cosine similarity for all weight matrices."""
    cosines = {}
    for name in weights_orig:
        if name in weights_recon:
            c = nn.functional.cosine_similarity(
                weights_orig[name].flatten().unsqueeze(0),
                weights_recon[name].flatten().unsqueeze(0)
            ).item()
            cosines[name] = c
    avg = sum(cosines.values()) / len(cosines)
    return cosines, avg


# ──────────────────────────────────────────────
# 7. MAIN EXPERIMENT
# ──────────────────────────────────────────────
def run_single_config(config_name, K, share, scales, weights, weights_orig, model, tokenizer):
    print(f"\n{'='*70}")
    print(f"Config: {config_name}")
    print(f"  K={K}, share={share}, scales={scales}")
    print(f"{'='*70}")
    
    codec = BlockCodec(K=K, share_codebook=share, use_scales=scales)
    codec.train(weights)
    
    bpw_report = codec.compute_bpw()
    print(f"\n[BPW]")
    for k, v in bpw_report.items():
        if 'bpw' in k:
            print(f"  {k}: {v:.6f}")
    
    weights_recon = codec.reconstruct_weights(weights_orig)
    
    cos_per_matrix, avg_cos = compute_overall_cosine(weights, weights_recon)
    print(f"  Avg cosine: {avg_cos:.4f}")
    
    ppls, delta_ppls, layer_cos = evaluate_layer_ppls(
        model, tokenizer, weights_orig, weights_recon, DEVICE
    )
    
    # Restore model
    for name, w in weights_orig.items():
        parts = name.split(".")
        li = int(parts[0].split("_")[1])
        layer = model.model.layers[li]
        if 'self_attn' in parts:
            target = getattr(layer.self_attn, parts[2])
        else:
            target = getattr(layer.mlp, parts[2])
        target.weight.data = w.to(target.weight.dtype)
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        'config': config_name,
        'bpw': bpw_report['total_bpw'],
        'id_bpw': bpw_report['id_bpw'],
        'cb_bpw': bpw_report['codebook_bpw'],
        'scale_bpw': bpw_report.get('scale_bpw', 0),
        'avg_cos': avg_cos,
        'deltas': delta_ppls[1:],
        'layer_cos': layer_cos,
        'block_mse': None,  # computed above
        'codebook_util': {f: f"{len(codec.family_assignments[f].unique())}/{len(codec.family_assignments[f])}" 
                         for f in codec.family_assignments},
    }


def run_experiment():
    print("=" * 70)
    print(f"Whole-Block Pattern Codebook Experiment")
    print(f"Block: {BLOCK_H}x{BLOCK_W} = {BLOCK_H*BLOCK_W} weights")
    print(f"Target: ≤1.20 BPW")
    print("=" * 70)
    
    device = DEVICE
    model, weights = load_model_and_extract_weights(num_layers=4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    weights_orig = {k: v.clone() for k, v in weights.items()}
    
    # Baseline PPL
    test_text = (
        "The transformer architecture has become the foundation of modern natural language processing. "
        "It processes sequences using attention mechanisms. "
        "This design eliminates the sequential bottleneck of recurrent networks. "
        "Large language models built on this architecture have achieved remarkable results."
    )
    model.to(device)
    model.eval()
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        base_ppl = torch.exp(model(**inputs, labels=inputs["input_ids"]).loss).item()
    print(f"\nBaseline PPL: {base_ppl:.2f}")
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run configs
    configs = [
        ("Shared K=64K no scales", 65536, True, False),
        ("Shared K=64K + scales",  65536, True, True),
        ("Per-family K=64K",      65536, False, False),
        ("Shared K=32K no scales", 32768, True, False),
    ]
    
    results = []
    for cfg_name, K, share, scales in configs:
        result = run_single_config(cfg_name, K, share, scales, 
                                    weights, weights_orig, model, tokenizer)
        results.append(result)
    
    # Results table
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    header = f"{'Config':<25} {'BPW':<8} {'ID_BPW':<8} {'CB_BPW':<8} {'Cos':<8} {'1LΔ%':<8} {'2LΔ%':<8} {'3LΔ%':<8} {'4LΔ%':<8}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        d = r['deltas']
        print(f"{r['config']:<25} {r['bpw']:<8.4f} {r['id_bpw']:<8.4f} {r['cb_bpw']:<8.4f} "
              f"{r['avg_cos']:<8.4f} {d[0]:<8.1f} {d[1]:<8.1f} {d[2]:<8.1f} {d[3]:<8.1f}")
    
    print(f"\nPrevious ALRW (1.2 BPW): cos~0.79, ΔPPL ~millions%")
    print(f"Original ALRW (12 BPW):  cos=0.919, ΔPPL=36%")
    print(f"Target: ≤1.20 BPW with ΔPPL < 10%")
    
    for r in results:
        if r['bpw'] <= 1.20:
            print(f"\n  ✓ {r['config']}: BPW={r['bpw']:.4f} ≤ 1.20, cos={r['avg_cos']:.4f}, "
                  f"4L Δ={r['deltas'][3]:.1f}%")
            if r['deltas'][3] < 10:
                print(f"    ★ MAJOR BREAKTHROUGH: ΔPPL < 10%!")
    
    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\nDone.")
