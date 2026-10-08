"""
Practical Hessian-weighted VQ: train with fast unweighted k-means,
encode with Fisher-weighted nearest-centroid selection.
"""

import torch, torch.nn as nn
import numpy as np, math, time, gc
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEVICE = "cuda"
K = 65536
STORE_CB_BITS = 8

# ── Load ──
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

# ── Fisher diagonal ──
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
    
    texts = [
        "The transformer architecture uses attention mechanisms for sequence modeling.",
        "Large language models generate coherent text from input prompts and context.",
        "Neural networks learn hierarchical representations through backpropagation.",
        "Deep learning requires significant computational resources for training models.",
        "Attention mechanisms allow models to focus on relevant parts of input sequences.",
        "Natural language processing tasks include translation summarization and question answering.",
        "Machine learning models are trained on large datasets to generalize well.",
        "The model architecture consists of multiple transformer layers with attention.",
    ]
    
    for txt in texts:
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
        f = F[name]
        f /= len(texts)
        f = f / (f.mean() + 1e-8)
        F[name] = f.clamp(0.01, 100)
    
    m.to('cpu')
    return F

# ── Fast unweighted k-means ──
def fast_kmeans(data, K, n_iter=25, batch=4096):
    N, D = data.shape
    dev = DEVICE
    dg = data.to(dev)
    idxs = torch.randperm(N, device=dev)[:K]
    cents = dg[idxs].clone()
    for it in range(n_iter):
        t0 = time.time()
        assign = torch.empty(N, dtype=torch.long, device=dev)
        for s in range(0, N, batch):
            e = min(s+batch, N); b = dg[s:e]
            bn = (b**2).sum(dim=1, keepdim=True)
            cn = (cents**2).sum(dim=1)
            assign[s:e] = (bn + cn.unsqueeze(0) - 2*(b@cents.T)).argmin(dim=1)
        new = torch.zeros_like(cents)
        cnt = torch.zeros(K, device=dev, dtype=torch.long)
        ones = torch.ones(N, device=dev, dtype=torch.long)
        cnt.scatter_add_(0, assign, ones)
        for d in range(D):
            s = torch.zeros(K, device=dev); s.scatter_add_(0, assign, dg[:,d]); new[:,d] = s / cnt.float().clamp(min=1)
        new[cnt==0] = cents[cnt==0]
        shift = ((cents - new)**2).sum().item(); cents = new
        print(f"  KM iter {it+1}: shift={shift:.4f}, live={(cnt>0).sum().item()}/{K}, t={time.time()-t0:.1f}s")
        if shift < 1e-6: break
    return cents.cpu(), assign.cpu()

# ── Weighted k-means (per-block importance) ──
def weighted_kmeans(data, imp_mean, K, n_iter=25, batch=4096):
    """K-means where each block is weighted by its mean Fisher importance."""
    N, D = data.shape
    dev = DEVICE
    dg = data.to(dev)
    w = imp_mean.to(dev)        # (N,) - per-block importance weight
    w_sum = w.sum()
    
    # Initialize with weighted random sampling
    probs = w / w.sum()
    idxs = torch.multinomial(probs, K, replacement=False)
    cents = dg[idxs].clone()
    
    for it in range(n_iter):
        t0 = time.time()
        
        # Assignment: weighted nearest centroid
        assign = torch.empty(N, dtype=torch.long, device=dev)
        for s in range(0, N, batch):
            e = min(s+batch, N)
            b = dg[s:e]
            bn = (b**2).sum(dim=1, keepdim=True)
            cn = (cents**2).sum(dim=1)
            dot = b @ cents.T
            dist = bn + cn.unsqueeze(0) - 2*dot
            assign[s:e] = dist.argmin(dim=1)
        
        # Weighted update: c_k = sum(w_i * x_i) / sum(w_i) for i in cluster k
        new = torch.zeros_like(cents)
        ws = torch.zeros(K, device=dev)  # weight sum per cluster
        
        for d in range(D):
            wxd = w * dg[:, d]  # (N,)
            summed = torch.zeros(K, device=dev)
            summed.scatter_add_(0, assign, wxd)
            new[:, d] = summed
        
        ws.scatter_add_(0, assign, w)
        
        new = new / ws.unsqueeze(1).clamp(min=1e-8)
        
        # Keep dead centroids
        dead = ws == 0
        new[dead] = cents[dead]
        
        shift = ((cents - new)**2).sum().item()
        cents = new
        live = (ws > 0).sum().item()
        print(f"  KM iter {it+1}: shift={shift:.4f}, live={live}/{K}, t={time.time()-t0:.1f}s")
        if shift < 1e-6: break
    
    return cents.cpu(), assign.cpu()

# ── Weighted encoding ──
def fast_unweighted_assign(data, cents, batch=4096, topk=64):
    """Find top-K nearest centroids by unweighted distance (fast)."""
    N, D = data.shape
    K = cents.shape[0]
    dev = DEVICE
    dg = data.to(dev)
    cg = cents.to(dev)
    
    bn = (dg**2).sum(dim=1)  # (N,)
    cn = (cg**2).sum(dim=1)  # (K,)
    
    topk_idx = torch.empty(N, topk, dtype=torch.long, device='cpu')
    topk_dist = torch.empty(N, topk, device='cpu')
    
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s+batch, N)
            b = dg[s:e]
            bb = bn[s:e]
            
            # Fast: pairwise distance using ||x-c||^2 = ||x||^2 + ||c||^2 - 2xc^T
            dot = b @ cg.T  # (B, K)
            dist = bb.unsqueeze(1) + cn.unsqueeze(0) - 2*dot  # (B, K)
            
            # Top-K
            vals, idxs = torch.topk(dist, topk, dim=1, largest=False)
            topk_idx[s:e] = idxs.cpu()
            topk_dist[s:e] = vals.cpu()
    
    return topk_idx, topk_dist


def weighted_assign_from_topk(data, imp, cents, topk_idx, batch=4096):
    """Re-rank top-K candidates by Fisher-weighted distance."""
    N = data.shape[0]
    topk = topk_idx.shape[1]
    dev = DEVICE
    dg = data.to(dev)
    ig = imp.to(dev)
    cg = cents.to(dev)
    
    assign = torch.empty(N, dtype=torch.long, device='cpu')
    
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s+batch, N)
            b = dg[s:e]; ib = ig[s:e]
            tk = topk_idx[s:e].to(dev)  # (B, topk)
            
            B = e - s
            D = b.shape[1]
            
            # Gather candidate centroids
            cand = cg[tk]  # (B, topk, D)
            
            # Weighted distance: sum_j w_j * (x_j - c_j)^2
            diff = b.unsqueeze(1) - cand  # (B, topk, D)
            wdiff = diff**2 * ib.unsqueeze(1)  # (B, topk, D)
            wdist = wdiff.sum(dim=2)  # (B, topk)
            
            best_in_topk = wdist.argmin(dim=1)  # (B,)
            assign[s:e] = tk[torch.arange(B, device=dev), best_in_topk].cpu()
    
    return assign


def weighted_assign(data, imp, cents, verbose=True):
    """Two-stage weighted encoding: fast top-K + weighted re-rank."""
    K = cents.shape[0]
    topk = min(64, K)  # Can't ask for more than K candidates
    if verbose: print(f"  Stage 1: Fast unweighted top-{topk} from K={K}...")
    topk_idx, _ = fast_unweighted_assign(data, cents, topk=topk)
    if verbose: print(f"  Stage 2: Weighted re-rank...")
    return weighted_assign_from_topk(data, imp, cents, topk_idx)

# ── Block extraction ──
def extract_blocks(w_dict, F, bh=4, bw=4):
    blk_names = []  # (name, row, col)
    all_vecs = []
    all_imp = []
    
    for name, W in w_dict.items():
        H, Ws = W.shape
        f = F.get(name, torch.ones_like(W))
        for i in range(0, H, bh):
            for j in range(0, Ws, bw):
                b = W[i:i+bh, j:j+bw]
                fb = f[i:i+bh, j:j+bw]
                if b.shape != (bh, bw):
                    p = torch.zeros(bh, bw); p[:b.shape[0],:b.shape[1]] = b; b = p
                    fp = torch.ones(bh, bw); fp[:fb.shape[0],:fb.shape[1]] = fb; fb = fp
                blk_names.append((name, i, j))
                all_vecs.append(b.flatten())
                all_imp.append(fb.flatten())
    
    return blk_names, torch.stack(all_vecs), torch.stack(all_imp)

# ── Codec ──
class VQCodec:
    def __init__(self, K=65536, K_residual=0):
        self.K = K
        self.K_residual = K_residual
        self.cents = None
        self.assign = None
        self.blk_names = None
        self.total_el = 0
        self.res_assign = None  # residual codebook IDs (None if not used)
        self.res_cents = None   # residual centroids
    
    def train(self, w_dict, F):
        self.total_el = sum(x.numel() for x in w_dict.values())
        blk_names, vecs, imp = extract_blocks(w_dict, F)
        self.blk_names = blk_names
        N = len(vecs)
        
        print(f"Blocks: {N}")
        train_n = min(N, 2000000)
        idxs = torch.randperm(N)[:train_n]
        
        # ── Stage 1: Main codebook ──
        print("Stage 1: Unweighted k-means...")
        cents, _ = fast_kmeans(vecs[idxs], self.K)
        self.cents = cents.float()
        
        print("Stage 1: Fisher-weighted encoding...")
        self.assign = weighted_assign(vecs, imp, self.cents)
        
        recon = self.cents[self.assign].cpu()
        vecs_cpu = vecs.cpu()
        imp_cpu = imp.cpu()
        
        # ── Stage 2: Selective residual correction ──
        if self.K_residual > 0:
            print(f"Stage 2: Residual codebook (K={self.K_residual})...")
            residual = vecs_cpu - recon  # (N, 16)
            
            # Per-block weighted error
            perr = (imp_cpu * residual**2).sum(dim=1) / imp_cpu.sum(dim=1)
            
            # Train residual codebook on the worst blocks
            imp_mean = imp_cpu.mean(dim=1)
            
            err_thresh = perr.quantile(0.80).item()  # top 20%
            imp_thresh = imp_mean.quantile(0.3).item()  # top 70% importance
            need_residual = (perr > err_thresh) & (imp_mean > imp_thresh)
            n_res = need_residual.sum().item()
            print(f"  {n_res}/{N} blocks ({n_res/N*100:.1f}%) get residual")
            
            if n_res > 0:
                res_train = residual[need_residual]
                r_cents, _ = fast_kmeans(res_train, self.K_residual, n_iter=15)
                self.res_cents = r_cents.float()
                
                # Encode residual with weighted distance
                res_imp = imp_cpu[need_residual]
                self.res_assign = torch.full((N,), -1, dtype=torch.long)
                self.res_assign[need_residual] = weighted_assign(
                    res_train, res_imp, self.res_cents, verbose=False
                )
                
                # Apply correction
                corr = torch.zeros(N, 16)
                corr[need_residual] = self.res_cents[self.res_assign[need_residual]]
                recon = recon + corr
        
        # ── Metrics ──
        mse = ((vecs_cpu - recon)**2).mean().item()
        cos = (vecs_cpu*recon).sum(1) / (vecs_cpu.norm(2)*recon.norm(2)+1e-8)
        avg_cos = cos.mean().item()
        
        wsum = ((imp_cpu*(vecs_cpu-recon)**2).sum() / imp_cpu.sum()).item()
        
        print(f"  MSE: {mse:.8f}, wMSE: {wsum:.8f}, Cos: {avg_cos:.4f}")
        
        perr = (imp_cpu*(vecs_cpu-recon)**2).sum(1)/(imp_cpu.sum(1)+1e-8)
        top10 = perr.sort()[0][int(0.9*N):].mean().item()
        print(f"  Worst 10% blocks (weighted error): {top10:.8f}")
        
        # Store final reconstruction on CPU
        self.recon_blocks = recon
        
    def reconstruct(self, w_dict):
        recon = {}
        rb = self.recon_blocks if hasattr(self, 'recon_blocks') else self.cents[self.assign]
        for idx, (name, r, c) in enumerate(self.blk_names):
            if name not in recon:
                H, W = w_dict[name].shape
                recon[name] = torch.zeros(H, W)
            bk = rb[idx].view(4, 4)
            h = min(4, recon[name].shape[0]-r)
            w = min(4, recon[name].shape[1]-c)
            recon[name][r:r+h, c:c+w] = bk[:h, :w]
        return recon
    
    def bpw(self):
        nb = len(self.assign)
        idb = nb * math.log2(self.K)
        cbb = self.K * 16 * STORE_CB_BITS
        total = idb + cbb + 64
        if self.res_assign is not None:
            n_res = (self.res_assign >= 0).sum().item()
            res_id_bits = n_res * math.log2(self.K_residual)
            res_cb_bits = self.K_residual * 16 * STORE_CB_BITS
            total += res_id_bits + res_cb_bits
        return total / self.total_el

# ── Eval ──
def eval_model(m, tok, wo, wr):
    txt = ("The transformer architecture has become the foundation of modern natural language processing. "
           "It processes sequences using attention mechanisms. "
           "This design eliminates the sequential bottleneck of recurrent networks. "
           "Large language models built on this architecture have achieved remarkable results.")
    inp = tok(txt, return_tensors="pt").to(DEVICE)
    m.to(DEVICE); m.eval()
    with torch.no_grad():
        base = torch.exp(m(**inp, labels=inp["input_ids"]).loss).item()
    print(f"  Base: {base:.2f}")
    
    ppls, dels, coss = [base], [0.0], []
    
    with torch.no_grad():
        for i in range(4):
            ly = m.model.layers[i]
            for p in ['q_proj','k_proj','v_proj','o_proj']:
                k = f"L{i}.SA.{p}"
                if k in wr:
                    getattr(ly.self_attn, p).weight.data = wr[k].to(
                        getattr(ly.self_attn, p).weight.dtype).to(DEVICE)
            for p in ['gate_proj','up_proj','down_proj']:
                k = f"L{i}.MLP.{p}"
                if k in wr:
                    getattr(ly.mlp, p).weight.data = wr[k].to(
                        getattr(ly.mlp, p).weight.dtype).to(DEVICE)
            
            out = m(**inp, labels=inp["input_ids"])
            ppl = torch.exp(out.loss).item()
            ppls.append(ppl)
            dels.append((ppl-base)/base*100)
            
            lc = []
            for k in wr:
                if f"L{i}" in k and k in wo:
                    lc.append(nn.functional.cosine_similarity(
                        wo[k].flatten().unsqueeze(0), wr[k].flatten().unsqueeze(0)).item())
            coss.append(sum(lc)/len(lc) if lc else 0)
            print(f"  L{i+1}: PPL={ppl:.2f}, Δ={dels[-1]:.1f}%, Cos={coss[-1]:.4f}")
    
    return ppls, dels, coss

# ── Main ──
def main():
    print("="*70)
    print("Fisher-Weighted Block VQ")
    print(f"K={K}")
    print("="*70)
    
    m, w = load(4)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    wo = {k: v.clone() for k, v in w.items()}
    
    print("\n[Fisher]")
    F = compute_fisher(m, w)
    all_f = torch.cat([f.flatten() for f in F.values()])
    print(f"  Fisher: mean={all_f.mean():.4f}, std={all_f.std():.4f}, "
          f"min={all_f.min():.4f}, max={all_f.max():.4f}")
    
    print("\n[Training VQ]")
    codec = VQCodec(K=K, K_residual=256)  # 8-bit residual correction
    codec.train(w, F)
    b = codec.bpw()
    print(f"  BPW: {b:.4f}")
    
    print("\n[Reconstruct]")
    wr = codec.reconstruct(w)
    cos_vals = [nn.functional.cosine_similarity(w[n].flatten().unsqueeze(0), wr[n].flatten().unsqueeze(0)).item() for n in w]
    avg_cos = sum(cos_vals)/len(cos_vals)
    print(f"  Overall Cos: {avg_cos:.4f}")
    
    print("\n[PPL]")
    ppls, dels, coss = eval_model(m, tok, wo, wr)
    
    print(f"\n{'='*70}")
    print("RESULT")
    print(f"{'='*70}")
    print(f"  BPW: {b:.4f}")
    print(f"  Cos: {avg_cos:.4f}")
    print(f"  1L Δ: {dels[1]:.1f}%, 4L Δ: {dels[4]:.1f}%")
    print(f"\n  Previous (unweighted K=64K): Cos=0.831, 4L Δ=53342%")
    print(f"  Previous (Fisher K=64K):     Cos=0.774, 4L Δ=2697%")
    
    if dels[4] < 10000:
        print(f"  ★ Improvement: 4L Δ dropped from 53342% to {dels[4]:.1f}%")
    if dels[4] < 30:
        print(f"  ★ MAJOR BREAKTHROUGH: 4L ΔPPL < 30%!")

if __name__ == "__main__":
    main()
