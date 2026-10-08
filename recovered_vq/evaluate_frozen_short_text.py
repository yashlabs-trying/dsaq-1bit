"""Evaluate the strict frozen codec on the legacy handcrafted PPL paragraph."""
import sys
import torch

sys.path.insert(0, "/tmp")
import centroid_refine as cr


state = torch.load("/workspace/centroid_refine/centroid_refined_checkpoint.pt",
                   map_location="cpu", weights_only=False)
model, weights = cr.legacy.load(4)
model.to("cuda").eval()
cr.get_module_shape = {name: tuple(value.shape) for name, value in weights.items()}
codec = cr.Codec()
codec.names = list(state["slices"])
codec.slices = state["slices"]
codec.base_q = state["base_codebook_q"]
codec.base_scale = state["base_codebook_scale"]
codec.centers = codec.base_q.float() * codec.base_scale.float()[:, None]
codec.assign = state["assignments"]
codec.res_positions = state["residual_positions"]
codec.res_ids = state["residual_ids"]
codec.res_q = state["residual_codebook_q"]
codec.res_scale = state["residual_codebook_scale"]
codec.res_centers = codec.res_q.float() * codec.res_scale.float()[:, None]
codec.nblocks = codec.assign.numel()
codec._res_global = torch.full((codec.nblocks,), -1, dtype=torch.int16)
codec._res_global[codec.res_positions.long()] = codec.res_ids.short()
cr.codec_global = codec
originals = {name: value.clone() for name, value in weights.items()}
tokenizer = cr.AutoTokenizer.from_pretrained(cr.legacy.MODEL_NAME)
text = ("The transformer architecture has become the foundation of modern natural language processing. "
        "It processes sequences using attention mechanisms. "
        "This design eliminates the sequential bottleneck of recurrent networks. "
        "Large language models built on this architecture have achieved remarkable results.")
ids = tokenizer(text, return_tensors="pt").input_ids[0]
base = cr.ppl(model, ids, ids.numel())
print("BF16", base)
for count in range(1, 5):
    cr.restore(model, originals)
    cr.apply_codec(model, codec, {}, count - 1)
    result = cr.ppl(model, ids, ids.numel())
    print(count, result, 100 * (result["ppl"] / base["ppl"] - 1))
