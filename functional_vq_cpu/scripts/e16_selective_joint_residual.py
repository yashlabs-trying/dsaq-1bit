"""E16: exact-byte selective joint Gate/Up/Down residuals on E14c.

The complete E14c codec is preserved losslessly inside an LZMA stream. The
reclaimed bytes fund one 8-bit joint correction ID for every package belonging
to the 5% most CE-sensitive four-neuron groups. All other packages decode
bit-identically to E14c.
"""
from __future__ import annotations

import hashlib
import json
import lzma
import math
import os
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from cpu_codec import blockify, codec_state, decode_weights
from e14b_neuron_joint_ce import ce_gradient
from e15_joint_package_codec import assign, kmeans, pack_uint


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "packed/E14c_joint_ce_consistent_005.packed"
CACHE = ROOT / "cache/e9_layer_cache.pt"
LABEL = os.environ.get("E16_LABEL", "E16_selective_joint_residual_5pct")
OUTPUT = ROOT / "packed" / f"{LABEL}.packed"
RESULT = ROOT / "results" / f"{LABEL}.json"
COVERAGE = float(os.environ.get("E16_COVERAGE", "0.05"))
K = 256
THREADS = int(os.environ.get("E16_THREADS", "4"))
MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
SEED = 20261010
ZERO_INIT = os.environ.get("E16_ZERO_INIT", "0") == "1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mlp_outputs(x, weights):
    z = F.silu(x@weights["gate_proj"].T)*(x@weights["up_proj"].T)
    return z, z@weights["down_proj"].T


def main():
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(1)
    started = time.time()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    cache = torch.load(CACHE, map_location="cpu", weights_only=True)
    parent_weights, parent_state = decode_weights(PARENT)
    original = cache["original_weights"]
    x_a, x_b = cache["x_a"], cache["x_b"]
    z_a, y_a = mlp_outputs(x_a, parent_weights)
    z_b, y_b = mlp_outputs(x_b, parent_weights)
    h2_a, h2_b = cache["h1_a"]+y_a, cache["h1_b"]+y_b

    print(json.dumps({"event": "load_model"}), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.float32,
        low_cpu_mem_usage=True, attn_implementation="eager").eval()
    grad_a, ce_a = ce_gradient(model, h2_a, cache["token_ids_a"])
    grad_b, ce_b = ce_gradient(model, h2_b, cache["token_ids_b"])

    z_ref_a, y_ref_a = mlp_outputs(x_a, original)
    z_ref_b, y_ref_b = mlp_outputs(x_b, original)
    gdown_a = grad_a@parent_weights["down_proj"]
    gdown_b = grad_b@parent_weights["down_proj"]
    group_damage = ((gdown_a*(z_a-z_ref_a)).abs().mean(0)
                    +(gdown_b*(z_b-z_ref_b)).abs().mean(0)).view(-1,4).mean(1)
    neuron_groups = group_damage.numel()
    hidden_chunks = original["gate_proj"].shape[1]//4
    critical_groups = max(1, math.ceil(COVERAGE*neuron_groups))
    top_groups = torch.topk(group_damage, critical_groups).indices.sort().values
    package_indices = (top_groups[:,None]*hidden_chunks
                       +torch.arange(hidden_chunks)[None,:]).reshape(-1).sort().values
    critical_count = package_indices.numel()

    slices = parent_state["slices"]
    g_start = slices["gate_proj"][0]
    u_start = slices["up_proj"][0]
    d_start = slices["down_proj"][0]
    neuron = package_indices//hidden_chunks
    chunk = package_indices%hidden_chunks
    g_index = g_start+package_indices
    u_index = u_start+package_indices
    d_index = d_start+chunk*neuron_groups+neuron
    parent_blocks = {name: blockify(parent_weights[name])
                     for name in ("gate_proj","up_proj","down_proj")}
    original_blocks = {name: blockify(original[name])
                       for name in ("gate_proj","up_proj","down_proj")}
    corrections = torch.cat((
        (original_blocks["gate_proj"][package_indices]
         -parent_blocks["gate_proj"][package_indices])/parent_state["scales"][g_index,None],
        (original_blocks["up_proj"][package_indices]
         -parent_blocks["up_proj"][package_indices])/parent_state["scales"][u_index,None],
        (original_blocks["down_proj"][d_index-d_start]
         -parent_blocks["down_proj"][d_index-d_start])/parent_state["scales"][d_index,None],
    ), dim=1)
    generator = torch.Generator().manual_seed(SEED)
    trained_k = K-1 if ZERO_INIT else K
    centers, correction_ids = kmeans(corrections, trained_k, 10, generator)
    if ZERO_INIT:
        centers = torch.cat((torch.zeros(1,48), centers), dim=0)
    center_scale = centers.abs().amax(1).clamp_min(1e-10)/127
    center_q = torch.round(centers/center_scale[:,None]).clamp(-127,127).to(torch.int8)
    stored_centers = center_q.float()*center_scale[:,None]
    correction_ids = (torch.zeros(critical_count,dtype=torch.long) if ZERO_INIT
                      else assign(corrections, stored_centers))

    compressed_parent = lzma.compress(PARENT.read_bytes(), preset=9)
    index_bits = math.ceil(math.log2(neuron_groups*hidden_chunks))
    payload = bytearray(compressed_parent)
    payload += pack_uint(package_indices.numpy(), index_bits)
    payload += correction_ids.numpy().astype(np.uint8).tobytes()
    payload += center_q.numpy().astype(np.int8).tobytes()
    payload += center_scale.half().numpy().astype("<f2").tobytes()
    meta = {
        "version": 3, "layout": "selective_joint_residual",
        "method": "E16 selective CE-ranked joint G/U/D correction",
        "model": MODEL, "revision": REVISION,
        "weights": parent_state["meta"]["weights"],
        "coverage": COVERAGE, "neuron_groups": neuron_groups,
        "hidden_chunks": hidden_chunks, "critical_groups": critical_groups,
        "critical_count": critical_count, "critical_index_bits": index_bits,
        "correction_k": K, "correction_id_bits": 8,
        "correction_dtype": "int8_per_prototype_fp16_scale",
        "compressed_parent_codec": "lzma_preset9",
        "compressed_parent_bytes": len(compressed_parent),
        "parent_sha256": sha256(PARENT), "seed": SEED,
        "zero_initialized": ZERO_INIT,
        "calibration_ce_A": ce_a, "calibration_ce_B": ce_b,
        "padding_bytes": 0,
    }
    target_bytes = PARENT.stat().st_size
    for _ in range(10):
        mb = json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
        padding = target_bytes-(4+len(mb)+len(payload))
        if padding < 0:
            raise RuntimeError(f"E16 exceeds parent by {-padding} bytes")
        if padding == meta["padding_bytes"]:
            break
        meta["padding_bytes"] = padding
    mb = json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    padding = target_bytes-(4+len(mb)+len(payload))
    meta["padding_bytes"] = padding
    mb = json.dumps(meta,sort_keys=True,separators=(",",":")).encode()
    padding = target_bytes-(4+len(mb)+len(payload))
    if padding != meta["padding_bytes"] or padding < 0:
        raise RuntimeError("metadata/padding did not stabilize")
    OUTPUT.write_bytes(struct.pack("<I",len(mb))+mb+payload+bytes(padding))
    output_weights, _ = decode_weights(OUTPUT)
    functional = {}
    for split, x, z_ref, y_ref, z_parent, y_parent in (
        ("A",x_a,z_ref_a,y_ref_a,z_a,y_a),("B",x_b,z_ref_b,y_ref_b,z_b,y_b)):
        z_out,y_out = mlp_outputs(x,output_weights)
        functional[f"parent_z_error_{split}"] = float((z_parent-z_ref).square().sum()/z_ref.square().sum())
        functional[f"output_z_error_{split}"] = float((z_out-z_ref).square().sum()/z_ref.square().sum())
        functional[f"parent_y_error_{split}"] = float((y_parent-y_ref).square().sum()/y_ref.square().sum())
        functional[f"output_y_error_{split}"] = float((y_out-y_ref).square().sum()/y_ref.square().sum())
    gate_pass = (functional["output_y_error_A"] < functional["parent_y_error_A"]
                 and functional["output_y_error_B"] < functional["parent_y_error_B"])
    result = {"experiment": LABEL, "parent": str(PARENT),
              "parent_sha256": sha256(PARENT), "coverage": COVERAGE,
              "critical_groups": critical_groups, "critical_count": critical_count,
              "compressed_parent_bytes": len(compressed_parent),
              "correction_index_bytes": (critical_count*index_bits+7)//8,
              "correction_id_bytes": critical_count,
              "correction_codebook_bytes": K*48+K*2,
              "padding_bytes": padding, "functional": functional,
              "functional_gate_pass": gate_pass, "bytes": OUTPUT.stat().st_size,
              "bpw": OUTPUT.stat().st_size*8/parent_state["meta"]["weights"],
              "output_sha256": sha256(OUTPUT),
              "elapsed_seconds": time.time()-started, "strict_ppl": None}
    RESULT.write_text(json.dumps(result,indent=2))
    print(json.dumps({"event":"complete",**result},indent=2),flush=True)


if __name__ == "__main__":
    main()
