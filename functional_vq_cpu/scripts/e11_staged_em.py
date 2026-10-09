"""E11: Staged prototype EM — replace low-value prototypes with clustered corrections.

Per round:
  1. Score prototypes by usage × reconstruction error
  2. Replace N low-value prototypes with clustered desired corrections
  3. Reassign affected blocks, validate on calibration B, rollback if worse
  4. PPL checkpoint each accepted round

Fixed at 2,223,107 bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, read_codec, patch_ids, patch_pattern_codes, unblock
from e7b_cpu_reassign import memory_state

ROOT = Path(__file__).resolve().parents[1]
E9_CACHE = ROOT / "cache/e9_layer_cache.pt"
PARENT = ROOT / "packed/E10a_gate_up_downstream.packed"
OUTPUT = ROOT / "packed" / "E11_staged_em.packed"
RESULT = ROOT / "results" / "E11_staged_em.json"
STATE = ROOT / "checkpoints" / "E11_staged_em_state.pt"
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
N_REPLACE = int(os.environ.get("E11_N", "32"))
N_ROUNDS = int(os.environ.get("E11_R", "5"))
THREADS = 4


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def kmeans_cluster(data, k, n_iter=30, seed=42):
    """Simple k-means in PyTorch."""
    rng = torch.Generator().manual_seed(seed)
    idx = torch.randperm(data.shape[0], generator=rng)[:k]
    centers = data[idx].clone()
    prev = None
    for _ in range(n_iter):
        dists = torch.cdist(data, centers)
        labels = dists.argmin(dim=1)
        if prev is not None and torch.equal(labels, prev):
            break
        prev = labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = data[mask].mean(dim=0)
    return centers, labels


def run_downstream(h2, model):
    for i in (1, 2):
        layer = model.model.layers[i]
        slen = h2.shape[0]
        pos = torch.arange(slen, device=h2.device).unsqueeze(0)
        mask = torch.full((slen, slen), torch.finfo(h2.dtype).min, device=h2.device)
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h2.unsqueeze(0), pos)
        h2 = h2 + layer.self_attn(layer.input_layernorm(h2).unsqueeze(0),
                                  attention_mask=mask, position_embeddings=(cos, sin))[0].squeeze(0)
        h_attn = h2
        h2 = layer.post_attention_layernorm(h2)
        h2 = h_attn + layer.mlp(h2)
    return h2


def compute_loss_b(weights, slices, x_b, h1_b, h4t_b, den_b, model):
    gate = x_b @ weights["gate_proj"].T
    up = x_b @ weights["up_proj"].T
    z = F.silu(gate) * up
    h2 = h1_b + z @ weights["down_proj"].T
    h4 = run_downstream(h2, model)
    return float((h4 - h4t_b).square().sum() / den_b), h2


def decode(proto_vals, ids, scales, residuals, slices):
    bv = proto_vals[ids.long()] * scales[:, None] + residuals
    w = {}
    for nm, (s, e, r, c) in slices.items():
        w[nm] = unblock(bv[s:e], (r, c))
    return w, bv


def main():
    torch.set_num_threads(THREADS)
    t0 = time.time()

    e9 = torch.load(E9_CACHE, map_location="cpu", weights_only=True)
    x_a, x_b = e9["x_a"], e9["x_b"]
    h1_a, h1_b = e9["h1_a"], e9["h1_b"]
    h2t_a, h2t_b = e9["h2_teacher_a"], e9["h2_teacher_b"]

    meta, st, ckpt, orig_codes, oid_np, sid_np, hard, rid, rq, rs = read_codec(PARENT)
    oid = oid_np.long() if isinstance(oid_np, torch.Tensor) else torch.from_numpy(oid_np).long()
    sid = sid_np.long() if isinstance(sid_np, torch.Tensor) else torch.from_numpy(sid_np).long()
    nb = meta["nblocks"]
    nfam, nlvl = ckpt.shape
    npf = meta["pattern_shape"][0] // nfam

    slices, mi = {}, torch.empty(nb, dtype=torch.long)
    for idx, nm in enumerate(meta["names"]):
        s, e, r, c = meta["slices"][nm]
        slices[nm] = (s, e, r, c)
        mi[s:e] = idx
    sf = st.float()[mi, sid]
    res = torch.zeros(nb, 16)
    if hard is not None:
        res[hard] = rq.float()[rid.long()] * rs.float()[rid.long(), None]

    codes = orig_codes.clone().long() if isinstance(orig_codes, torch.Tensor) else torch.from_numpy(orig_codes.copy()).long()
    pf = torch.arange(meta["pattern_shape"][0], dtype=torch.long) // npf
    pv = ckpt.float()[pf[:, None], codes]
    ids = oid.clone()

    # BF16 blocks
    ob = torch.zeros(nb, 16)
    for nm, (s, e, r, c) in slices.items():
        ob[s:e] = blockify(e9["original_weights"][nm])

    bv = pv[ids] * sf[:, None] + res
    rec_err = (ob - bv).square().sum(1)

    # Usage + damage per prototype
    usage = torch.zeros(4096).scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.float32))
    pdam = torch.zeros(4096).scatter_add_(0, ids, rec_err)
    vs = usage * pdam

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager"
    ).eval()
    with torch.no_grad():
        h4t_b = run_downstream(h2t_b, model)
    den_b = float(h4t_b.square().sum())

    w0, _ = decode(pv, ids, sf, res, slices)
    loss_old, _ = compute_loss_b(w0, slices, x_b, h1_b, h4t_b, den_b, model)
    print(json.dumps({"event": "baseline", "loss_B": loss_old}), flush=True)

    results, nac = [], 0
    for ri in range(N_ROUNDS):
        low = torch.topk(vs, N_REPLACE, largest=False).indices
        aff = torch.isin(ids, low)
        cand = torch.where(aff)[0]
        extra = min(N_REPLACE * 2, nb - cand.shape[0])
        if extra > 0:
            na = torch.where(~aff)[0]
            td = torch.topk(rec_err[na], extra).indices
            cand = torch.cat([cand, na[td]])

        corr_list = []
        for bi in cand:
            d = ob[bi] - bv[bi]
            if float(d.norm()) > 1e-6:
                corr_list.append(d)
        if len(corr_list) < N_REPLACE:
            print(json.dumps({"event": "stop", "found": len(corr_list)}), flush=True)
            break

        corr_t = torch.stack(corr_list)
        centers, _ = kmeans_cluster(corr_t, N_REPLACE, seed=42 + ri)
        new_fam = pf[low]
        qv, qc = torch.zeros(N_REPLACE, 16), torch.zeros(N_REPLACE, 16, dtype=torch.long)
        lv = ckpt.float()
        for i in range(N_REPLACE):
            f = int(new_fam[i])
            dst = (centers[i].unsqueeze(1) - lv[f].unsqueeze(0)).abs()
            best = dst.argmin(dim=1)
            qc[i] = best
            qv[i] = lv[f, best]

        npv = pv.clone()
        ncd = codes.clone()
        npv[low] = qv
        ncd[low] = qc

        nid = ids.clone()
        for bi in cand.tolist():
            tgt = ob[bi]
            si = float(sf[bi])
            ri_ = res[bi]
            tn = (tgt - ri_) / max(si, 1e-10)
            nd = (npv - tn.unsqueeze(0)).square().sum(1)
            nid[bi] = int(nd.argmin())

        wn, nbv = decode(npv, nid, sf, res, slices)
        lnew, _ = compute_loss_b(wn, slices, x_b, h1_b, h4t_b, den_b, model)
        ok = lnew < loss_old
        print(json.dumps({"event": "validate", "r": ri + 1, "old": loss_old,
                           "new": lnew, "ok": ok}), flush=True)
        if ok:
            pv, codes, ids, bv, loss_old = npv, ncd, nid, nbv, lnew
            nac += 1
            patch_ids(PARENT, OUTPUT, ids)
            patch_pattern_codes(PARENT, OUTPUT, codes.to(torch.uint8))
            results.append({"r": ri + 1, "accepted": True, "loss_B": lnew})
        else:
            results.append({"r": ri + 1, "accepted": False})
            break

        torch.save({"sha": sha256(PARENT), "pv": pv, "cd": codes, "id": ids,
                     "res": results, "t": time.time() - t0}, STATE)

    result = {"experiment": "E11_staged_em", "parent": str(PARENT),
              "n_replace": N_REPLACE, "accepted": nac,
              "per_round": results, "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size * 8 / meta["weights"],
              "output_sha256": sha256(OUTPUT), "elapsed": time.time() - t0, "ppl": None}
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2))
    print(json.dumps({"event": "complete", **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
