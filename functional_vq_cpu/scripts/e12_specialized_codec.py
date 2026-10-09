"""E12: Projection-specialized 4x4 VQ codec at ~1.19 BPW.

Per-projection k-means (10K subset each), scale quantization, residual.
Standard packed format with single-family checkpoint. Same file size as E10A.
"""
from __future__ import annotations

import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, read_codec, unblock

ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
PARENT = ROOT / "packed/E10a_gate_up_downstream.packed"
OUTPUT = ROOT / "packed" / "E12_specialized.packed"
RESULT = ROOT / "results" / "E12_specialized.json"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
LEVELS = 64
PATTERN_BITS = 6
SCALE_LEVELS = 16



ALLOC = {"down_proj": 1408, "gate_proj": 1024, "up_proj": 1024,
         "o_proj": 256, "v_proj": 160, "q_proj": 128, "k_proj": 96}
ORDER = ["down_proj", "gate_proj", "up_proj", "o_proj", "v_proj", "q_proj", "k_proj"]


def kmeans(data, k, seed=42, n_iter=50):
    n, d = data.shape; rng = np.random.RandomState(seed)
    idx = rng.choice(n, k, replace=False); c = data[idx].copy()
    prev = None
    for _ in range(n_iter):
        dst = np.zeros((n, k))
        for i in range(k):
            df = data - c[i]; dst[:, i] = np.sum(df*df, 1)
        lb = dst.argmin(1)
        if prev is not None and np.array_equal(lb, prev): break
        prev = lb
        for i in range(k):
            m = lb == i
            if m.sum(): c[i] = data[m].mean(0)
    return c, lb


def pack_fixed(values, bits):
    v = np.asarray(values, dtype=np.uint16).reshape(-1)
    m = ((v[:, None] >> np.arange(bits, dtype=np.uint16)) & 1).astype(np.uint8)
    return np.packbits(m.reshape(-1), bitorder="little").tobytes()


def main():
    torch.set_num_threads(4); t0 = time.time()
    e9 = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    rm, _, _, _, _, _, _, _, _, _ = read_codec(PARENT)
    sl = {n: tuple(rm["slices"][n]) for n in rm["names"]}
    x_b = e9["x_b"]; h1 = e9["h1_b"]; h2t = e9["h2_teacher_b"]

    # Collect blocks per projection
    pb = {}
    for n in ORDER:
        s, e, _, _ = sl[n]
        pb[n] = blockify(e9["original_weights"][n]).numpy().astype(np.float32)
    tn = sum(b.shape[0] for b in pb.values())
    print(json.dumps({"event":"collect","blocks":tn}), flush=True)

    # Per-projection k-means + assignment
    n_pats = [1408, 1024, 1024, 256, 160, 128, 96]
    ap = []; po = {}; off = 0
    for ni, n in enumerate(ORDER):
        blk = pb[n]; nb = blk.shape[0]; k = n_pats[ni]
        sub = blk[np.random.RandomState(42+hash(n)%999).choice(nb, min(10000,nb), False)]
        c, _ = kmeans(sub, min(k,sub.shape[0]), seed=42+hash(n)%1000)
        if c.shape[0] < k:
            c = np.vstack([c, np.tile(c[:1], (k-c.shape[0],1))])
        ap.append(c[:k]); po[n] = (off, off+k); off += k
        print(json.dumps({"event":"kmeans","proj":n,"k":k}), flush=True)
    fp = np.vstack(ap).astype(np.float32)

    # Quantize to 64 levels
    lv = np.sort(np.percentile(fp.reshape(-1), np.linspace(0,100,LEVELS)))
    fl = fp.reshape(-1,1)
    qc = np.abs(fl - lv.reshape(1,-1)).argmin(1).reshape(4096,16).astype(np.uint8)
    qp = lv[qc].astype(np.float32)

    # Scale: max-abs per block, 16 geom levels per projection
    si = np.zeros(tn, dtype=np.uint8)
    st = np.zeros((7, SCALE_LEVELS), dtype=np.float16)
    fi = np.zeros(tn, dtype=np.int32); oi = 0
    for ni, n in enumerate(ORDER):
        blk = pb[n]; nb = blk.shape[0]
        mx = np.max(np.abs(blk), 1) + 1e-10
        slv = np.geomspace(mx.min(), mx.max(), SCALE_LEVELS).astype(np.float16)
        st[ni] = slv
        ps, pe = po[n]
        qpn = torch.from_numpy(qp[ps:pe])
        bt = torch.from_numpy(blk)
        dst = torch.cdist(bt, qpn)
        fi[oi:oi+nb] = (dst.argmin(1).numpy() + ps).astype(np.int32)
        for i in range(nb):
            si[oi+i] = np.argmin(np.abs(mx[i] - slv.astype(np.float32)))
        oi += nb
        print(json.dumps({"event":"assign","proj":n,"blocks":nb}), flush=True)

    # Apply scales to compute residuals
    sc = st[np.repeat(np.arange(7), [pb[n].shape[0] for n in ORDER]), si]  # (tn,)
    pv = qp[fi]  # (tn, 16)
    dec = pv * sc[:, None]

    # Residual for worst blocks (target about 300KB)
    all_orig = np.vstack([pb[n] for n in ORDER])
    r_err = all_orig - dec
    r_norm = np.linalg.norm(r_err.reshape(tn, -1), axis=1)
    threshold = np.percentile(r_norm, 85)
    hard_mask = r_norm > threshold
    hard_idx = np.where(hard_mask)[0]
    n_hard = len(hard_idx)

    # Simple INT8 residual quantization
    resid_vals = r_err[hard_idx]
    r_scale = np.max(np.abs(resid_vals), axis=1) + 1e-10
    r_q = np.clip(np.round(resid_vals / r_scale[:, None] * 127), -128, 127).astype(np.int8)
    r_scale_fp16 = r_scale.astype(np.float16)

    # K-means on residuals (128 clusters)
    n_res_clusters = min(128, n_hard)
    if n_hard > n_res_clusters:
        sub_r = resid_vals[np.random.RandomState(42).choice(n_hard, 2000, False)]
        rc, rl = kmeans(sub_r, n_res_clusters, seed=42)
        r_ids = np.zeros(n_hard, dtype=np.uint8)
        for i in range(n_hard):
            r_ids[i] = np.argmin(np.sum((resid_vals[i] - rc)**2, 1))
        r_q = np.clip(np.round(rc / np.max(np.abs(rc),1)[:,None]*127), -128,127).astype(np.int8)
    else:
        rc = resid_vals
        r_ids = np.arange(n_hard, dtype=np.uint8)
        n_res_clusters = n_hard

    r_scale_fp16 = np.max(np.abs(rc), 1).astype(np.float16)

    print(json.dumps({"event":"residual","n_hard":n_hard,"n_clusters":n_res_clusters}), flush=True)

    # Write packed file
    meta = {
        "nblocks": tn, "names": ORDER,
        "slices": {n: list(sl[n]) for n in ORDER},
        "checkpoint_shape": [1, LEVELS],
        "pattern_shape": [4096, 16],
        "pattern_bits": PATTERN_BITS, "pattern_count": 4096,
        "hard_count": n_hard, "residual_bits": 8, "residual_k": n_res_clusters,
        "weights": tn * 16 * 4,
    }
    mb = json.dumps(meta).encode()
    patb = (4096*16*PATTERN_BITS+7)//8

    with open(OUTPUT, "wb") as f:
        f.write(struct.pack("<I", len(mb)))
        f.write(mb)
        rec = (fi.astype(np.uint16)&0x0FFF) | ((si.astype(np.uint16)&0x000F)<<12)
        f.write(rec.astype("<u2").tobytes())
        f.write(st.astype("<f2").tobytes())
        f.write(lv.astype(np.float16).tobytes())
        f.write(pack_fixed(qc, PATTERN_BITS))
        # Hard mask + residual
        mask_bits = np.packbits(np.where(hard_mask, 1, 0).astype(np.uint8), bitorder="little")
        f.write(mask_bits.tobytes())
        r_id_bytes = (n_hard * 8 + 7) // 8
        r_id_packed = pack_fixed(r_ids, 8)
        f.write(r_id_packed)
        f.write(r_q.astype(np.int8).tobytes())
        f.write(r_scale_fp16.astype("<f2").tobytes())

    size = OUTPUT.stat().st_size
    bpw = size * 8 / meta["weights"]
    print(json.dumps({"event":"built","bytes":size,"bpw":bpw}), flush=True)

    # Evaluate
    print(json.dumps({"event":"evaluate"}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()
    from cpu_codec import decode_weights
    w, _ = decode_weights(OUTPUT)
    with torch.no_grad():
        g = x_b @ w["gate_proj"].T; u = x_b @ w["up_proj"].T
        h = h1 + (F.silu(g)*u) @ w["down_proj"].T
        for i in (1,2):
            ly = model.model.layers[i]
            slen = h.shape[0]; p = torch.arange(slen).unsqueeze(0)
            m = torch.full((slen,slen), torch.finfo(h.dtype).min)
            m = torch.triu(m,1).unsqueeze(0).unsqueeze(0)
            co, si = model.model.rotary_emb(h.unsqueeze(0), p)
            h = h + ly.self_attn(ly.input_layernorm(h).unsqueeze(0),
                                attention_mask=m, position_embeddings=(co,si))[0].squeeze(0)
            ha = h; h = ly.post_attention_layernorm(h); h = ha + ly.mlp(h)
        h4t = h2t
        for i in (1,2):
            ly = model.model.layers[i]
            slen = h4t.shape[0]; p = torch.arange(slen).unsqueeze(0)
            m = torch.full((slen,slen), torch.finfo(h4t.dtype).min)
            m = torch.triu(m,1).unsqueeze(0).unsqueeze(0)
            co, si = model.model.rotary_emb(h4t.unsqueeze(0), p)
            h4t = h4t + ly.self_attn(ly.input_layernorm(h4t).unsqueeze(0),
                                    attention_mask=m, position_embeddings=(co,si))[0].squeeze(0)
            ha = h4t; h4t = ly.post_attention_layernorm(h4t); h4t = ha + ly.mlp(h4t)
    lb = float((h - h4t).square().sum() / h4t.square().sum())
    print(json.dumps({"event":"result","loss_B":lb,"bpw":bpw}), flush=True)
    result = {"experiment":"E12_specialized","allocation":ALLOC,
              "bytes":size,"bpw":bpw,"loss_B":lb,"sha":sha256(OUTPUT),
              "elapsed":time.time()-t0,"ppl":None}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event":"complete",**result}), flush=True)


if __name__ == "__main__":
    main()
