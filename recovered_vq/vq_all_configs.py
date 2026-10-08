"""
Config 1-4 runner for Fisher-weighted VQ with residual and adaptive refinement.
Block shapes: 4x4, 1x16, 16x1.
Config 4: adaptive sequential (student-aware) refinement.
"""

import torch, torch.nn as nn
import numpy as np, math, time, gc, json
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
K = 65536
STORE_CB_BITS = 8
K_RESIDUAL = 16       # 4-bit residual correction
RESIDUAL_COVERAGE = 0.05  # ~5% of blocks
CALIB_TEXTS = [
    "The transformer architecture uses attention mechanisms for sequence modeling.",
    "Large language models generate coherent text from input prompts and context.",
    "Neural networks learn hierarchical representations through backpropagation.",
    "Deep learning requires significant computational resources for training models.",
    "Attention mechanisms allow models to focus on relevant parts of input sequences.",
    "Natural language processing tasks include translation summarization and question answering.",
    "Machine learning models are trained on large datasets to generalize well.",
    "The model architecture consists of multiple transformer layers with attention.",
]

# ── 1. Utilities ──
def load(num_layers=4):
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="cpu")
    m.eval()
    w = {}
    for i in range(num_layers):
        ly = m.model.layers[i]
        for p in ['q_proj','k_proj','v_proj','o_proj']:
            w[f"L{i}.SA.{p}"] = getattr(ly.self_attn, p).weight.data.float()
        for p in ['gate_proj','up_proj','down_proj']:
            w[f"L{i}.MLP.{p}"] = getattr(ly.mlp, p).weight.data.float()
    return m, w

def restore_model(m, w_orig):
    with torch.no_grad():
        for name, w in w_orig.items():
            li = int(name.split(".")[0].split("L")[1])
            ly = m.model.layers[li]
            block = ly.self_attn if "SA" in name else ly.mlp
            pname = name.split(".")[2]
            getattr(block, pname).weight.data = w.clone().to(getattr(block, pname).weight.dtype)

def apply_weights(m, w_dict):
    with torch.no_grad():
        for name, w in w_dict.items():
            li = int(name.split(".")[0].split("L")[1])
            ly = m.model.layers[li]
            block = ly.self_attn if "SA" in name else ly.mlp
            pname = name.split(".")[2]
            getattr(block, pname).weight.data = w.to(getattr(block, pname).weight.dtype).to(DEVICE)

# ── 2. Fisher diagonal ──
def compute_fisher(m, w_dict):
    m.to(DEVICE); m.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    F = {}
    for name in w_dict:
        li = int(name.split(".")[0].split("L")[1])
        ly = m.model.layers[li]
        block = ly.self_attn if "SA" in name else ly.mlp
        pname = name.split(".")[2]
        F[name] = torch.zeros_like(getattr(block, pname).weight.data, device=DEVICE)
    for txt in CALIB_TEXTS:
        inp = tok(txt, return_tensors="pt").to(DEVICE)
        m.zero_grad()
        m(**inp, labels=inp["input_ids"]).loss.backward()
        with torch.no_grad():
            for name in w_dict:
                li = int(name.split(".")[0].split("L")[1])
                ly = m.model.layers[li]
                block = ly.self_attn if "SA" in name else ly.mlp
                pname = name.split(".")[2]
                g = getattr(block, pname).weight.grad
                if g is not None: F[name] += g.float() ** 2
    for name in F:
        f = F[name]; f /= len(CALIB_TEXTS)
        f = f / (f.mean() + 1e-8); F[name] = f.clamp(0.01, 100)
    m.to('cpu')
    return F

# ── 3. Block extraction ──
def extract_blocks(w_dict, F, bh=4, bw=4):
    blk_names, vecs, imp = [], [], []
    for name, W in w_dict.items():
        H, Ws = W.shape; f = F.get(name, torch.ones_like(W))
        for i in range(0, H, bh):
            for j in range(0, Ws, bw):
                b = W[i:i+bh, j:j+bw]; fb = f[i:i+bh, j:j+bw]
                if b.shape != (bh, bw):
                    p = torch.zeros(bh, bw); p[:b.shape[0],:b.shape[1]] = b; b = p
                    fp = torch.ones(bh, bw); fp[:fb.shape[0],:fb.shape[1]] = fb; fb = fp
                blk_names.append((name, i, j))
                vecs.append(b.flatten())
                imp.append(fb.flatten())
    return blk_names, torch.stack(vecs), torch.stack(imp)

def reconstruct_from_blocks(blk_names, blocks, w_ref, bh, bw):
    recon = {}
    for idx, (name, r, c) in enumerate(blk_names):
        if name not in recon:
            H, W = w_ref[name].shape
            recon[name] = torch.zeros(H, W)
        bk = blocks[idx].view(bh, bw)
        h = min(bh, recon[name].shape[0]-r)
        w = min(bw, recon[name].shape[1]-c)
        recon[name][r:r+h, c:c+w] = bk[:h, :w]
    return recon

# ── 4. K-means ──
def fast_kmeans(data, K, n_iter=25, batch=4096):
    N, D = data.shape; dev = DEVICE
    dg = data.to(dev)
    idxs = torch.randperm(N, device=dev)[:K]; cents = dg[idxs].clone()
    for it in range(n_iter):
        t0 = time.time()
        assign = torch.empty(N, dtype=torch.long, device=dev)
        for s in range(0, N, batch):
            e = min(s+batch, N); b = dg[s:e]
            bn = (b**2).sum(dim=1, keepdim=True); cn = (cents**2).sum(dim=1)
            assign[s:e] = (bn + cn.unsqueeze(0) - 2*(b@cents.T)).argmin(dim=1)
        new = torch.zeros_like(cents); cnt = torch.zeros(K, device=dev, dtype=torch.long)
        cnt.scatter_add_(0, assign, torch.ones(N, device=dev, dtype=torch.long))
        for d in range(D):
            s = torch.zeros(K, device=dev); s.scatter_add_(0, assign, dg[:,d]); new[:,d] = s / cnt.float().clamp(min=1)
        new[cnt==0] = cents[cnt==0]; shift = ((cents - new)**2).sum().item(); cents = new
        print(f"  KM iter {it+1}: shift={shift:.4f}, live={(cnt>0).sum().item()}/{K}, t={time.time()-t0:.1f}s")
        if shift < 1e-6: break
    return cents.cpu(), assign.cpu()

# ── 5. Fast top-K + weighted re-rank ──
def fast_unweighted_assign(data, cents, topk=64, batch=4096):
    N, D = data.shape; K = cents.shape[0]; dev = DEVICE
    dg = data.to(dev); cg = cents.to(dev)
    bn = (dg**2).sum(dim=1); cn = (cg**2).sum(dim=1)
    topk_idx = torch.empty(N, topk, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s+batch, N); b = dg[s:e]; bb = bn[s:e]
            dist = bb.unsqueeze(1) + cn.unsqueeze(0) - 2*(b@cg.T)
            topk_idx[s:e] = torch.topk(dist, topk, dim=1, largest=False).indices.cpu()
    return topk_idx

def weighted_assign_from_topk(data, imp, cents, topk_idx, batch=4096):
    N = data.shape[0]; topk = topk_idx.shape[1]; dev = DEVICE
    dg = data.to(dev); ig = imp.to(dev); cg = cents.to(dev)
    assign = torch.empty(N, dtype=torch.long, device='cpu')
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s+batch, N); b = dg[s:e]; ib = ig[s:e]; tk = topk_idx[s:e].to(dev)
            B = e-s
            cand = cg[tk]
            diff = b.unsqueeze(1) - cand
            wdist = (diff**2 * ib.unsqueeze(1)).sum(dim=2)
            best = wdist.argmin(dim=1)
            assign[s:e] = tk[torch.arange(B, device=dev), best].cpu()
    return assign

def weighted_assign(data, imp, cents, verbose=True):
    K = cents.shape[0]; topk = min(64, K)
    if verbose: print(f"  Top-{topk} unweighted...")
    tk = fast_unweighted_assign(data, cents, topk)
    if verbose: print(f"  Weighted re-rank...")
    return weighted_assign_from_topk(data, imp, cents, tk)

# ── 6. Activation-aware encoding (config 4) ──
def compute_activation_covariances(m, w_dict, bh, bw, num_layers=4):
    """
    For each block, compute the input activation covariance matrix H ≈ E[xx^T].
    This is used for activation-weighted VQ.
    However, for 4x4 blocks the input x is a slice of the full activation,
    so we need to compute per-weight-position importance.
    
    Instead: for each weight matrix, collect input activations from calibration.
    Then for each block position, compute importance from activation norm.
    """
    m.to(DEVICE); m.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    # We'll collect activation norms per output neuron (for column VQ)
    # and per input neuron (for row VQ)
    act_importance = {}
    
    with torch.no_grad():
        for name in w_dict:
            li = int(name.split(".")[0].split("L")[1])
            ly = m.model.layers[li]
            act_importance[name] = None  # will fill with running sum
        
        # Hook to capture input activations for each linear layer
        activations = {}
        handles = []
        
        def make_hook(name):
            def hook(module, inp, out):
                # inp[0] is the input tensor to the linear layer
                if name not in activations:
                    activations[name] = []
                # Only keep the norm per feature across batch+seq
                x = inp[0].detach().float()
                # x: (batch, seq, in_features)
                # Importance: ||x||^2 per input feature
                x_norm = (x ** 2).sum(dim=(0, 1))  # (in_features,)
                activations[name].append(x_norm)
            return hook
        
        for name in w_dict:
            li = int(name.split(".")[0].split("L")[1])
            ly = m.model.layers[li]
            block = ly.self_attn if "SA" in name else ly.mlp
            pname = name.split(".")[2]
            h = getattr(block, pname).register_forward_hook(make_hook(name))
            handles.append(h)
        
        for txt in CALIB_TEXTS:
            inp = tok(txt, return_tensors="pt").to(DEVICE)
            m(**inp, labels=inp["input_ids"])
        
        for h in handles:
            h.remove()
        
        for name in w_dict:
            if name in activations and len(activations[name]) > 0:
                x_norm = torch.stack(activations[name]).sum(dim=0)  # (in_features,)
                x_norm = x_norm / (x_norm.mean() + 1e-8)
                act_importance[name] = x_norm.clamp(0.01, 100).cpu()
    
    m.to('cpu')
    return act_importance


def get_student_activations(m, quantized_layers, num_layers=4):
    """
    After quantizing some layers, get the student's input activations
    for the next un-quantized layer.
    Returns: h_l for each layer l, where h_l is the input to layer l.
    """
    m.to(DEVICE); m.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    
    layer_inputs = {}
    
    # Apply quantized weights
    apply_weights(m, quantized_layers)
    
    with torch.no_grad():
        def make_hook(li):
            def hook(module, inp, out):
                if li not in layer_inputs:
                    layer_inputs[li] = []
                x = inp[0].detach().float().cpu()
                layer_inputs[li].append(x)
            return hook
        
        handles = []
        for li in range(num_layers):
            h = m.model.layers[li].register_forward_hook(make_hook(li))
            handles.append(h)
        
        for txt in CALIB_TEXTS:
            inp = tok(txt, return_tensors="pt").to(DEVICE)
            m(**inp, labels=inp["input_ids"])
        
        for h in handles:
            h.remove()
    
    m.to('cpu')
    
    # Compute activation importance per feature (average across all tokens)
    result = {}
    for li in range(num_layers):
        if li in layer_inputs:
            all_x = torch.cat([x.view(-1, x.shape[-1]) for x in layer_inputs[li]], dim=0)
            result[li] = all_x
    
    return result


def activation_weighted_encoding(block_vecs, block_imp, act_imp, blk_names, cents, bh, bw):
    """
    Encode blocks using activation-weighted importance.
    act_imp: dict of name -> per-input-feature importance (for row VQ) or
             per-output-feature importance (for column VQ)
    Combined importance = Fisher * activation_importance
    """
    # Combine Fisher and activation importance
    combined_imp = block_imp.clone()
    
    for idx, (name, r, c) in enumerate(blk_names):
        if name in act_imp:
            act = act_imp[name]  # (in_features,) for row-oriented
            # Map activation importance to block weights
            # For 4x4 block: row r, col c
            # If act_imp is per-input: it has in_features elements, matching columns
            # If act_imp is per-output: it has out_features elements, matching rows
            
            # For row VQ (1x16): block is 1 row x 16 cols
            # The 16 weights correspond to 16 consecutive input features
            if bw == 16:  # row block
                for k in range(16):
                    if c + k < len(act):
                        combined_imp[idx, k] *= act[c + k].item()
            elif bh == 16:  # column block
                for k in range(16):
                    if r + k < len(act):
                        combined_imp[idx, k] *= act[r + k].item()
            else:  # 4x4
                # Per-weight activation importance depends on both row and column
                out_act = act  # re-use (might be off but directionally correct)
                for ki in range(4):
                    for kj in range(4):
                        if r + ki < len(act):
                            combined_imp[idx, ki * 4 + kj] *= act[r + ki].item()
    
    return weighted_assign(block_vecs, combined_imp, cents, verbose=True)


# ── 7. Full VQ pipeline ──
def train_vq(w_dict, F, bh=4, bw=4, use_residual=True, act_imp=None):
    """Train VQ codec for given block shape. Returns (cents, assign, blk_names, recon_blocks, bpw)"""
    D = bh * bw
    total_el = sum(x.numel() for x in w_dict.values())
    
    blk_names, vecs, imp = extract_blocks(w_dict, F, bh, bw)
    N = len(vecs)
    print(f"  Blocks: {N}, Dim: {D}")
    
    # Train centroids
    train_n = min(N, 2000000)
    idxs = torch.randperm(N)[:train_n]
    print("  K-means...")
    cents, _ = fast_kmeans(vecs[idxs], K)
    cents = cents.float()
    
    # Encode
    if act_imp is not None:
        print("  Activation-weighted encoding...")
        assign = activation_weighted_encoding(vecs, imp, act_imp, blk_names, cents, bh, bw)
    else:
        print("  Fisher-weighted encoding...")
        assign = weighted_assign(vecs, imp, cents)
    
    # Ensure everything is on CPU
    recon = cents[assign].cpu()
    vecs = vecs.cpu()
    imp = imp.cpu()
    
    # Residual correction
    need_fix = torch.zeros(N, dtype=torch.bool)
    n_fix_val = 0
    if use_residual and K_RESIDUAL > 0:
        residual = vecs - recon
        perr = (imp * residual**2).sum(dim=1) / imp.sum(dim=1)
        err_thresh = perr.quantile(1.0 - RESIDUAL_COVERAGE).item()
        imp_mean = imp.mean(dim=1)
        imp_thresh = imp_mean.quantile(0.3).item()
        need_fix = (perr > err_thresh) & (imp_mean > imp_thresh)
        n_fix_val = need_fix.sum().item()
        print(f"  Residual: {n_fix_val}/{N} blocks ({n_fix_val/N*100:.1f}%)")
        
        if n_fix_val > 0:
            res_data = residual[need_fix]
            res_imp = imp[need_fix]
            r_cents, _ = fast_kmeans(res_data, K_RESIDUAL, n_iter=15)
            r_cents = r_cents.float()
            r_assign = weighted_assign(res_data, res_imp, r_cents, verbose=False)
            corr = torch.zeros(N, D)
            corr[need_fix] = r_cents[r_assign]
            recon = recon + corr
    
    # Metrics
    mse = ((vecs - recon)**2).mean().item()
    cos = (vecs * recon).sum(1) / (vecs.norm(p=2, dim=1) * recon.norm(p=2, dim=1) + 1e-8)
    avg_cos = cos.mean().item()
    wmse = ((imp * (vecs - recon)**2).sum() / imp.sum()).item()
    print(f"  MSE: {mse:.8f}, wMSE: {wmse:.8f}, Cos: {avg_cos:.4f}")
    id_bits = N * math.log2(K)
    cb_bits = K * D * STORE_CB_BITS
    is_res = (use_residual and K_RESIDUAL > 0 and n_fix_val > 0)
    res_bits = (n_fix_val * math.log2(K_RESIDUAL) + K_RESIDUAL * D * STORE_CB_BITS) if is_res else 0
    total_bits = id_bits + cb_bits + res_bits + 64
    bpw = total_bits / total_el
    
    return cents, assign, blk_names, recon.cpu(), bpw, avg_cos


# ── 8. Evaluation ──
def evaluate_ppl(m, tok, w_recon, w_orig, num_layers=4):
    """Evaluate PPL layer by layer."""
    test_text = ("The transformer architecture has become the foundation of modern natural language processing. "
                 "It processes sequences using attention mechanisms. "
                 "This design eliminates the sequential bottleneck of recurrent networks. "
                 "Large language models built on this architecture have achieved remarkable results.")
    inp = tok(test_text, return_tensors="pt").to(DEVICE)
    
    m.to(DEVICE); m.eval()
    
    # Restore all weights to device
    with torch.no_grad():
        for name, w in w_orig.items():
            li = int(name.split(".")[0].split("L")[1])
            ly = m.model.layers[li]
            block = ly.self_attn if "SA" in name else ly.mlp
            pname = name.split(".")[2]
            getattr(block, pname).weight.data = w.to(getattr(block, pname).weight.dtype).to(DEVICE)
    
    with torch.no_grad():
        base_ppl = torch.exp(m(**inp, labels=inp["input_ids"]).loss).item()
    print(f"  Base PPL: {base_ppl:.2f}")
    
    ppls, deltas, coss = [base_ppl], [0.0], []
    
    with torch.no_grad():
        for i in range(num_layers):
            ly = m.model.layers[i]
            for p in ['q_proj','k_proj','v_proj','o_proj']:
                k = f"L{i}.SA.{p}"
                if k in w_recon:
                    getattr(ly.self_attn, p).weight.data = w_recon[k].to(
                        getattr(ly.self_attn, p).weight.dtype).to(DEVICE)
            for p in ['gate_proj','up_proj','down_proj']:
                k = f"L{i}.MLP.{p}"
                if k in w_recon:
                    getattr(ly.mlp, p).weight.data = w_recon[k].to(
                        getattr(ly.mlp, p).weight.dtype).to(DEVICE)
            
            out = m(**inp, labels=inp["input_ids"])
            ppl = torch.exp(out.loss).item()
            ppls.append(ppl); deltas.append((ppl-base_ppl)/base_ppl*100)
            
            lc = []
            for k in w_recon:
                if f"L{i}" in k and k in w_orig:
                    lc.append(nn.functional.cosine_similarity(
                        w_orig[k].flatten().unsqueeze(0), w_recon[k].flatten().unsqueeze(0)).item())
            coss.append(sum(lc)/len(lc) if lc else 0)
            print(f"  L{i+1}: PPL={ppl:.2f}, Δ={deltas[-1]:.1f}%, Cos={coss[-1]:.4f}")
            if deltas[-1] > 100:
                print(f"    ** Major jump at layer {i+1}! **")
    
    return ppls, deltas, coss


# ── 9. Config 4: Adaptive sequential refinement ──
def train_adaptive_vq(m, w_dict, F, bh, bw, w_orig, tok):
    """
    Train VQ sequentially layer-by-layer using student input distribution.
    """
    D = bh * bw
    total_el = sum(x.numel() for x in w_dict.values())
    device = DEVICE
    
    # Extract all blocks first
    blk_names, vecs, imp = extract_blocks(w_dict, F, bh, bw)
    N = len(vecs)
    print(f"\n  Total blocks: {N}, Dim: {D}")
    
    # Train initial centroids on all data
    train_n = min(N, 2000000)
    idxs = torch.randperm(N)[:train_n]
    print("  Initial global k-means...")
    cents, _ = fast_kmeans(vecs[idxs], K)
    cents = cents.float()
    
    # Assign using Fisher-weighted encoding
    assign = weighted_assign(vecs, imp, cents)
    recon_global = cents[assign]
    
    # Group blocks by layer
    layer_blocks = {i: {'idx': [], 'vecs': [], 'imp': []} for i in range(4)}
    for idx, (name, r, c) in enumerate(blk_names):
        li = int(name.split(".")[0].split("L")[1])
        layer_blocks[li]['idx'].append(idx)
        layer_blocks[li]['vecs'].append(vecs[idx])
        layer_blocks[li]['imp'].append(imp[idx])
    
    for li in range(4):
        layer_blocks[li]['vecs'] = torch.stack(layer_blocks[li]['vecs'])
        layer_blocks[li]['imp'] = torch.stack(layer_blocks[li]['imp'])
    
    # ── Sequential refinement ──
    final_recon = recon_global.clone()
    current_quantized = {}
    
    for li in range(4):
        print(f"\n  --- Layer {li} adaptive refinement ---")
        
        # Get student input for this layer
        if li > 0:
            # Apply current quantized layers to model
            restore_model(m, w_orig)
            apply_weights(m, current_quantized)
            student_h = get_student_activations(m, current_quantized, num_layers=li+1)
            # h_{li} is input to layer li from student
            if li in student_h:
                h_student = student_h[li].to(device)  # (total_seq, hidden_dim)
                print(f"    Student input shape: {h_student.shape}")
            else:
                h_student = None
                print(f"    No student input captured")
        else:
            # Layer 0: use clean inputs
            tok = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tok.pad_token is None: tok.pad_token = tok.eos_token
            # Get clean forward activations for layer 0
            restore_model(m, w_orig)
            m.to(device); m.eval()
            student_h_layer0 = get_student_activations(m, {}, num_layers=1)
            h_student = student_h_layer0.get(0, None)
            if h_student is not None:
                h_student = h_student.to(device)
        
        if h_student is not None:
            act_imp_layer = {}
            for key in w_dict:
                if f"L{li}" in key:
                    W = w_dict[key]
                    x = h_student
                    in_feat = W.shape[1]
                    x_sub = x[:, :min(x.shape[1], in_feat)]
                    imp_per_input = (x_sub ** 2).sum(dim=0)
                    imp_per_input = imp_per_input / (imp_per_input.mean() + 1e-8)
                    act_imp_layer[key] = imp_per_input.clamp(0.01, 100).cpu()
            
            # Re-encode this layer's blocks using combined importance
            li_vecs = layer_blocks[li]['vecs']
            li_imp = layer_blocks[li]['imp']
            li_idx = layer_blocks[li]['idx']
            
            # Combine Fisher and activation importance
            combined_imp = li_imp.clone()
            for j, idx in enumerate(li_idx):
                name, r, c = blk_names[idx]
                if name in act_imp_layer:
                    act = act_imp_layer[name]
                    if bw == 16:  # row block
                        for k in range(16):
                            if c + k < len(act):
                                combined_imp[j, k] *= act[c + k].item()
                    elif bh == 16:  # column block
                        for k in range(16):
                            if r + k < len(act):
                                combined_imp[j, k] *= act[r + k].item()
                    else:  # 4x4
                        for ki in range(4):
                            for kj in range(4):
                                if r + ki < len(act):
                                    combined_imp[j, ki*4+kj] *= act[r + ki].item()
            
            # Re-assign with student-aware importance
            print(f"    Re-encoding with student-aware importance...")
            new_assign = weighted_assign(li_vecs, combined_imp, cents, verbose=True)
            
            # Update assignments
            assign_updated = assign.clone()
            for j, idx in enumerate(li_idx):
                assign_updated[idx] = new_assign[j]
            
            # Update reconstruction
            final_recon = cents[assign_updated]
            
            # Build quantized weights for this layer
            lay_recon = reconstruct_from_blocks(
                [blk_names[i] for i in li_idx],
                final_recon[li_idx],
                w_dict, bh, bw
            )
            for k, v in lay_recon.items():
                current_quantized[k] = v
        
        else:
            # No student input: just use Fisher weights
            lay_recon = reconstruct_from_blocks(
                [blk_names[i] for i in layer_blocks[li]['idx']],
                final_recon[layer_blocks[li]['idx']],
                w_dict, bh, bw
            )
            for k, v in lay_recon.items():
                current_quantized[k] = v
    
    # Metrics
    vecs_cpu = vecs.cpu()
    imp_cpu = imp.cpu()
    mse = ((vecs_cpu - final_recon)**2).mean().item()
    cos = (vecs_cpu * final_recon).sum(1) / (vecs_cpu.norm(2) * final_recon.norm(2) + 1e-8)
    avg_cos = cos.mean().item()
    wmse = ((imp_cpu * (vecs_cpu - final_recon)**2).sum() / imp_cpu.sum()).item()
    print(f"\n  Final: MSE={mse:.8f}, wMSE={wmse:.8f}, Cos={avg_cos:.4f}")
    
    # BPW
    id_bits = N * math.log2(K)
    cb_bits = K * D * STORE_CB_BITS
    total_bits = id_bits + cb_bits + 64
    bpw = total_bits / total_el
    
    return final_recon, bpw, avg_cos, blk_names


# ── 10. Main runner ──
def run_experiment():
    print("=" * 70)
    print("Fisher-Weighted VQ: Block Shape Comparison + Adaptive Refinement")
    print("=" * 70)
    
    m, w = load(4)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    w_orig = {k: v.clone() for k, v in w.items()}
    
    # Fisher diagonal
    print("\n[Fisher]")
    F = compute_fisher(m, w)
    all_f = torch.cat([f.flatten() for f in F.values()])
    print(f"  Mean={all_f.mean():.4f}, Std={all_f.std():.4f}, Range=[{all_f.min():.4f},{all_f.max():.4f}]")
    
    results = []
    
    # ── Config 1: 4×4 blocks ──
    print("\n" + "="*60)
    print("CONFIG 1: 4×4 Fisher VQ")
    print("="*60)
    cents1, assign1, names1, recon1, bpw1, cos1 = train_vq(w, F, 4, 4, use_residual=True)
    wr1 = reconstruct_from_blocks(names1, recon1, w, 4, 4)
    restore_model(m, w_orig)
    print("\n  PPL Evaluation:")
    p1, d1, c1 = evaluate_ppl(m, tok, wr1, w_orig)
    results.append({'config': '4×4 Fisher', 'bpw': bpw1, 'cos': cos1, 'deltas': d1, 'coses': c1})
    
    # ── Config 2: 1×16 row blocks ──
    print("\n" + "="*60)
    print("CONFIG 2: Row-aligned 1×16 Fisher VQ")
    print("="*60)
    cents2, assign2, names2, recon2, bpw2, cos2 = train_vq(w, F, 1, 16, use_residual=True)
    wr2 = reconstruct_from_blocks(names2, recon2, w, 1, 16)
    restore_model(m, w_orig)
    print("\n  PPL Evaluation:")
    p2, d2, c2 = evaluate_ppl(m, tok, wr2, w_orig)
    results.append({'config': 'Row 1×16 Fisher', 'bpw': bpw2, 'cos': cos2, 'deltas': d2, 'coses': c2})
    
    # ── Config 3: 16×1 column blocks ──
    print("\n" + "="*60)
    print("CONFIG 3: Column-aligned 16×1 Fisher VQ")
    print("="*60)
    cents3, assign3, names3, recon3, bpw3, cos3 = train_vq(w, F, 16, 1, use_residual=True)
    wr3 = reconstruct_from_blocks(names3, recon3, w, 16, 1)
    restore_model(m, w_orig)
    print("\n  PPL Evaluation:")
    p3, d3, c3 = evaluate_ppl(m, tok, wr3, w_orig)
    results.append({'config': 'Col 16×1 Fisher', 'bpw': bpw3, 'cos': cos3, 'deltas': d3, 'coses': c3})
    
    # ── Config 4: Best shape + adaptive refinement ──
    # Pick best shape from 1-3 (lowest 4L Δ)
    best_idx = min(range(3), key=lambda i: results[i]['deltas'][4])
    best_shape = [(4,4), (1,16), (16,1)][best_idx]
    print(f"\n" + "="*60)
    print(f"CONFIG 4: Adaptive sequential ({best_shape[0]}×{best_shape[1]})")
    print("="*60)
    bh, bw = best_shape
    recon4, bpw4, cos4, names4 = train_adaptive_vq(m, w, F, bh, bw, w_orig, tok)
    wr4 = reconstruct_from_blocks(names4, recon4, w, bh, bw)
    restore_model(m, w_orig)
    print("\n  PPL Evaluation:")
    p4, d4, c4 = evaluate_ppl(m, tok, wr4, w_orig)
    results.append({'config': f'Adaptive {bh}×{bw}', 'bpw': bpw4, 'cos': cos4, 'deltas': d4, 'coses': c4})
    
    # ── Results table ──
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    header = f"{'Method':<22} {'BPW':<8} {'Cos':<8} {'1LΔ%':<8} {'2LΔ%':<8} {'3LΔ%':<8} {'4LΔ%':<8}"
    print(header)
    print("-" * len(header))
    
    for r in results:
        d = r['deltas']
        print(f"{r['config']:<22} {r['bpw']:<8.4f} {r['cos']:<8.4f} "
              f"{d[1]:<8.1f} {d[2]:<8.1f} {d[3]:<8.1f} {d[4]:<8.1f}")
        # Find first major jump
        for i in range(1, 5):
            if d[i] > 100:
                print(f"  {'':>22} {'':>8} {'':>8} ^ Major jump at L{i}")
                break
    
    # Success check
    print(f"\nSuccess targets: 1L<10%, 2L<15%, 3L<30%, 4L<50%")
    for r in results:
        d = r['deltas']
        ok = all([d[1] < 10, d[2] < 15, d[3] < 30, d[4] < 50])
        if ok:
            print(f"  ✓ {r['config']}: ALL TARGETS MET! BPW={r['bpw']:.4f}")
    
    # Major success check
    for r in results:
        if r['bpw'] <= 1.20 and r['deltas'][4] < 10:
            print(f"\n  ★ MAJOR SUCCESS: {r['config']}: 4L ΔPPL={r['deltas'][4]:.1f}% at BPW={r['bpw']:.4f}!")
    
    return results

if __name__ == "__main__":
    results = run_experiment()
