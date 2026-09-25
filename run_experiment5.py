from __future__ import annotations

import argparse
import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dsaq.core import blocks_to_matrix, hadamard_blocks, matrix_to_blocks, reconstruct_naive
from run_experiments import dump_json, get_text_tokens, seed_everything, target_modules


PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--revision", default="main")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--calibration-tokens", type=int, default=4096)
    p.add_argument("--evaluation-tokens", type=int, default=4096)
    p.add_argument("--sequence-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260925)
    p.add_argument("--output", type=Path, default=Path("results/experiment5"))
    return p.parse_args()


def candidate_patterns() -> list[tuple[str, dict[str, float]]]:
    def all_factor(value: float) -> dict[str, float]: return {p: value for p in PROJECTIONS}
    def group(items: tuple[str, ...], value: float) -> dict[str, float]: return {p: (value if p in items else 1.0) for p in PROJECTIONS}
    return [
        ("base_1.00", all_factor(1.0)), ("global_0.85", all_factor(.85)), ("global_1.15", all_factor(1.15)),
        ("attention_0.85", group(PROJECTIONS[:4], .85)), ("global_0.70", all_factor(.70)),
        ("global_1.30", all_factor(1.30)), ("attention_1.15", group(PROJECTIONS[:4], 1.15)),
        ("mlp_0.85", group(PROJECTIONS[4:], .85)), ("mlp_1.15", group(PROJECTIONS[4:], 1.15)),
        ("qk_0.85", group(("q_proj", "k_proj"), .85)), ("qk_1.15", group(("q_proj", "k_proj"), 1.15)),
        ("vo_0.85", group(("v_proj", "o_proj"), .85)), ("vo_1.15", group(("v_proj", "o_proj"), 1.15)),
        ("gate_up_0.85", group(("gate_proj", "up_proj"), .85)),
        ("gate_up_1.15", group(("gate_proj", "up_proj"), 1.15)),
        ("down_1.15", group(("down_proj",), 1.15)),
    ]


def layer_of(name: str) -> int:
    return int(name.split(".layers.", 1)[1].split(".", 1)[0])


def suffix(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def quantize_matrix(weight: torch.Tensor, block_size: int, hadamard: bool) -> tuple[torch.Tensor, int]:
    blocks, layout = matrix_to_blocks(weight.float(), block_size)
    if hadamard:
        transformed = hadamard_blocks(blocks, block_size)
        q, scales = reconstruct_naive(transformed, layout.valid)
        q = hadamard_blocks(q, block_size)
    else:
        q, scales = reconstruct_naive(blocks, layout.valid)
    scales = scales.half().float()
    if hadamard:
        q = hadamard_blocks(transformed.sign() * scales[:, None], block_size)
    else:
        q = blocks.sign() * scales[:, None]
    return blocks_to_matrix(q, layout).bfloat16().cpu(), 16 * len(scales)


def tensor_metrics(target: torch.Tensor, value: torch.Tensor) -> tuple[float, float]:
    t, v = target.float(), value.float()
    nmse = float((t-v).square().sum() / t.square().sum().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(t.flatten(), v.flatten(), dim=0))
    return nmse, cosine


@dataclass
class PathState:
    hidden: torch.Tensor
    score: float
    choices: list[int]
    history: list[dict[str, float]]


class SFBS:
    def __init__(self, model, modules, originals, hadamard_base, teacher_inputs, teacher_outputs, layer_kwargs, patterns):
        self.model, self.modules, self.originals, self.base = model, modules, originals, hadamard_base
        self.teacher_inputs, self.teacher_outputs, self.kwargs, self.patterns = teacher_inputs, teacher_outputs, layer_kwargs, patterns
        self.layers = list(model.model.layers)
        self.by_layer = {i: {suffix(n): (n, m) for n, m in modules.items() if layer_of(n) == i} for i in range(len(self.layers))}
        self.evaluations = 0

    def restore_layer(self, layer: int) -> None:
        with torch.no_grad():
            for _, (name, module) in self.by_layer[layer].items(): module.weight.copy_(self.originals[name].to(module.weight.device))

    def apply_candidate(self, layer: int, candidate: int) -> None:
        factors = self.patterns[candidate][1]
        with torch.no_grad():
            for proj, (name, module) in self.by_layer[layer].items():
                module.weight.copy_((self.base[name] * factors[proj]).to(module.weight.device, module.weight.dtype))

    def call(self, layer: int, hidden: torch.Tensor) -> torch.Tensor:
        result = self.layers[layer](hidden_states=hidden, **self.kwargs[layer])
        return result[0] if isinstance(result, (tuple, list)) else result

    def fp_trajectory(self, start: int, hidden: torch.Tensor, horizon: int) -> torch.Tensor:
        value = hidden
        for j in range(start, min(start+horizon, len(self.layers))):
            self.restore_layer(j); value = self.call(j, value)
        return value

    def evaluate_candidate(self, layer: int, hidden: torch.Tensor, candidate: int, target_kind: str, horizon: int) -> tuple[torch.Tensor, float, float, float, float]:
        self.restore_layer(layer); fp_local = self.call(layer, hidden)
        self.apply_candidate(layer, candidate); q_local = self.call(layer, hidden); self.evaluations += 1
        local_nmse, local_cos = tensor_metrics(fp_local if target_kind == "current" else self.teacher_outputs[layer], q_local)
        if horizon:
            q_future = self.fp_trajectory(layer+1, q_local, horizon)
            if target_kind == "current": target_future = self.fp_trajectory(layer+1, fp_local, horizon)
            else: target_future = self.teacher_outputs[min(layer+horizon, len(self.layers)-1)]
            future_nmse, future_cos = tensor_metrics(target_future, q_future)
        else:
            future_nmse, future_cos = local_nmse, local_cos
        return q_local, local_nmse, local_cos, future_nmse, future_cos

    @torch.inference_mode()
    def independent(self, k: int) -> tuple[list[int], list[dict], int]:
        choices, records, start_eval = [], [], self.evaluations
        for layer in range(len(self.layers)):
            scored = []
            for candidate in range(k):
                q, local, cos, future, fcos = self.evaluate_candidate(layer, self.teacher_inputs[layer], candidate, "clean", 0)
                scored.append((local, candidate, cos, future, fcos))
            scored.sort(); best = scored[0]; choices.append(best[1])
            records.append({"layer": layer, "candidate": best[1], "local_nmse": best[0], "local_cosine": best[2], "horizon_nmse": best[3], "horizon_cosine": best[4]})
            self.restore_layer(layer)
        return choices, records, self.evaluations-start_eval

    @torch.inference_mode()
    def search(self, k: int, beam: int, horizon: int, target_kind: str, lambda_local: float, lambda_horizon: float,
               candidate_log: list[dict], config: str) -> tuple[list[int], list[dict], int]:
        paths = [PathState(self.teacher_inputs[0], 0.0, [], [])]; start_eval = self.evaluations
        for layer in range(len(self.layers)):
            expanded = []
            for path_id, path in enumerate(paths):
                candidates = []
                for candidate in range(k):
                    q, local, cos, future, fcos = self.evaluate_candidate(layer, path.hidden, candidate, target_kind, horizon)
                    step = lambda_local*local + lambda_horizon*future
                    candidates.append((candidate, local, cos, future, fcos, step, q))
                if path_id == 0:
                    for candidate, local, cos, future, fcos, step, _ in candidates:
                        candidate_log.append({"configuration": config, "layer": layer, "candidate": candidate,
                                              "candidate_name": self.patterns[candidate][0], "local_nmse": local,
                                              "local_cosine": cos, "horizon_nmse": future, "horizon_cosine": fcos,
                                              "step_score": step})
                for candidate, local, cos, future, fcos, step, q in candidates:
                    row = {"layer": layer, "candidate": candidate, "local_nmse": local, "local_cosine": cos,
                           "horizon_nmse": future, "horizon_cosine": fcos, "step_score": step}
                    expanded.append(PathState(q, path.score+step, path.choices+[candidate], path.history+[row]))
            expanded.sort(key=lambda x: x.score); paths = expanded[:beam]
            self.restore_layer(layer)
            print(f"{config}: layer {layer+1}/{len(self.layers)} best_score={paths[0].score:.6g}", flush=True)
        return paths[0].choices, paths[0].history, self.evaluations-start_eval

    @torch.inference_mode()
    def drift(self, choices: list[int], config: str) -> tuple[list[dict], list[torch.Tensor]]:
        hidden = self.teacher_inputs[0]; rows, inputs = [], []
        previous_nmse = 0.0
        for layer, candidate in enumerate(choices):
            inputs.append(hidden.detach())
            self.restore_layer(layer); fp_local = self.call(layer, hidden)
            self.apply_candidate(layer, candidate); q = self.call(layer, hidden)
            teacher_nmse, teacher_cos = tensor_metrics(self.teacher_outputs[layer], q)
            local_nmse, local_cos = tensor_metrics(fp_local, q)
            rows.append({"configuration": config, "layer": layer, "teacher_nmse": teacher_nmse,
                         "teacher_cosine": teacher_cos, "local_functional_nmse": local_nmse,
                         "local_functional_cosine": local_cos, "drift_jump": teacher_nmse-previous_nmse,
                         "candidate": candidate, "candidate_name": self.patterns[candidate][0]})
            previous_nmse = teacher_nmse; hidden = q
        return rows, inputs


def capture_teacher(model, tokens: torch.Tensor, device: torch.device):
    layers = list(model.model.layers); inputs, outputs, kwargs_by_layer = {}, {}, {}
    handles = []
    for index, layer in enumerate(layers):
        def pre(_m, args, kwargs, i=index):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            inputs[i] = hidden.detach()
            kwargs_by_layer[i] = {k: v for k, v in kwargs.items() if k != "hidden_states"}
        def post(_m, _args, out, i=index): outputs[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        handles.append(layer.register_forward_pre_hook(pre, with_kwargs=True)); handles.append(layer.register_forward_hook(post))
    try:
        with torch.inference_mode(): model(tokens.to(device), use_cache=False)
    finally:
        for h in handles: h.remove()
    return inputs, outputs, kwargs_by_layer


def apply_full(modules, base, originals, choices, patterns) -> None:
    with torch.no_grad():
        for name, module in modules.items():
            factor = patterns[choices[layer_of(name)]][1][suffix(name)]
            module.weight.copy_((base[name]*factor).to(module.weight.device, module.weight.dtype))


def safe_perplexity(model, tokens: torch.Tensor, device: torch.device) -> tuple[float, float]:
    losses = []
    with torch.inference_mode():
        for sequence in tokens:
            out = model(sequence[None].to(device), labels=sequence[None].to(device), use_cache=False)
            losses.append(float(out.loss))
    nll = float(np.mean(losses)); return (float(math.exp(nll)) if nll < 709 else float("inf")), nll


def weight_stats(originals, base, patterns, n_layers: int) -> dict[tuple[int, int], float]:
    stats = {}
    components = {}
    for name, w in originals.items():
        q = base[name].float(); wf = w.float(); components[(layer_of(name), suffix(name))] = (float(wf.square().sum()), float((wf*q).sum()), float(q.square().sum()), w.numel())
    for layer in range(n_layers):
        for candidate, (_, factors) in enumerate(patterns):
            sse = n = 0
            for proj in PROJECTIONS:
                w2, wq, q2, count = components[(layer, proj)]; f = factors[proj]
                sse += w2 - 2*f*wq + f*f*q2; n += count
            stats[(layer, candidate)] = sse/n
    return stats


def make_plots(out: Path, aggregate: pd.DataFrame, drift: pd.DataFrame, candidate: pd.DataFrame) -> None:
    p = out/"plots"; p.mkdir(exist_ok=True)
    plt.figure(figsize=(8,5)); plt.scatter(aggregate.true_bpw, aggregate.perplexity); plt.yscale("log")
    for r in aggregate.itertuples(): plt.annotate(r.configuration.replace("sequential_", "seq_"), (r.true_bpw, r.perplexity), fontsize=6)
    plt.xlabel("true BPW"); plt.ylabel("perplexity"); plt.tight_layout(); plt.savefig(p/"perplexity_vs_bpw.png",dpi=180); plt.close()
    selected = [x for x in ("hadamard_naive", "independent_k8", "sequential_k8_h0_current", "beam4_k8_h1_l1_1") if x in set(drift.configuration)]
    plt.figure(figsize=(9,5))
    for name in selected:
        f=drift[drift.configuration==name]; plt.plot(f.layer, f.teacher_nmse, marker="o", ms=3, label=name)
    plt.yscale("log"); plt.xlabel("layer"); plt.ylabel("teacher-trajectory NMSE"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(p/"hidden_drift_by_layer.png",dpi=180); plt.close()
    plt.figure(figsize=(9,5))
    for name in selected:
        f=drift[drift.configuration==name]; plt.plot(f.layer, f.local_functional_nmse, marker="o", ms=3, label=name)
    plt.yscale("log"); plt.xlabel("layer"); plt.ylabel("local functional NMSE"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(p/"functional_error_by_layer.png",dpi=180); plt.close()
    for column, filename, xlabel in (("k","perplexity_vs_k.png","candidate count K"),("beam","perplexity_vs_beam.png","beam width"),("horizon","perplexity_vs_horizon.png","downstream horizon")):
        f=aggregate[aggregate[column].notna()].sort_values(column); plt.figure(figsize=(7,5)); plt.plot(f[column],f.perplexity,marker="o"); plt.yscale("log"); plt.xlabel(xlabel); plt.ylabel("perplexity"); plt.tight_layout(); plt.savefig(p/filename,dpi=180); plt.close()
    plt.figure(figsize=(8,5)); plt.scatter(candidate.weight_mse, candidate.local_nmse, alpha=.35, s=12); plt.xscale("log"); plt.yscale("log"); plt.xlabel("candidate weight MSE"); plt.ylabel("functional layer NMSE"); plt.tight_layout(); plt.savefig(p/"weight_mse_vs_functional_loss.png",dpi=180); plt.close()


def build_report(out: Path, aggregate: pd.DataFrame, drift: pd.DataFrame, projection: pd.DataFrame) -> None:
    compressed = aggregate[aggregate.configuration!="bf16"]; eligible=compressed[compressed.true_bpw<=1.05]
    best=eligible.sort_values("perplexity").iloc[0]; had=aggregate[aggregate.configuration=="hadamard_naive"].iloc[0]
    independent=aggregate[aggregate.configuration=="independent_k8"].iloc[0]; sequential=aggregate[aggregate.configuration=="sequential_k8_h0_current"].iloc[0]
    beams=aggregate[aggregate.beam>1]; best_beam=beams.sort_values("perplexity").iloc[0]
    best_drift=drift[drift.configuration==best.configuration]; cliff=best_drift.sort_values("drift_jump",ascending=False).iloc[0]
    q1=best.perplexity<had.perplexity; q2=sequential.perplexity<independent.perplexity; q3=best_beam.perplexity<sequential.perplexity; q5=best.perplexity<100
    cols=["configuration","perplexity","mean_nll","true_bpw","k","beam","horizon","target","candidate_evaluations","runtime_seconds"]
    text=f"""# Experiment 5: Sequential Functional Binary Search

## Scope

All 24 Qwen2.5-0.5B transformer blocks and all 168 Q/K/V/O/gate/up/down matrices were quantized. Embeddings, normalization, and LM head remain BF16. Candidates use one reproducible scale-compensation pattern per transformer layer; storage includes one sign per target weight, FP16 block scales, and `ceil(log2(K))` candidate-ID bits per layer. No LRR, DSA, sparse rescue, or high-precision correction is present.

## Results

{aggregate[cols].sort_values('perplexity').to_markdown(index=False,floatfmt='.6g')}

Best result at <=1.05 BPW: **{best.configuration}**, PPL **{best.perplexity:.4g}**, true BPW **{best.true_bpw:.8f}**. Hadamard naive PPL: **{had.perplexity:.4g}**.

## Answers

**Q1: Did functional optimization beat weight-space optimization? {'YES' if q1 else 'NO'}.** Best functional PPL {best.perplexity:.4g} versus Hadamard naive {had.perplexity:.4g}.

**Q2: Did sequential calibration reduce accumulated distortion? {'YES' if q2 else 'NO'}.** Independent K=8 PPL {independent.perplexity:.4g}; sequential K=8 local/current-target PPL {sequential.perplexity:.4g}.

**Q3: Did beam search help? {'YES' if q3 else 'NO'}.** Best beam result: {best_beam.configuration}, PPL {best_beam.perplexity:.4g}.

**Q4: What was the best PPL at <=1.05 BPW? {best.perplexity:.4g}.**

**Q5: Did we reach PPL <100? {'YES' if q5 else 'NO'}.**

**Q6: Where does functional information begin collapsing?** The largest positive teacher-drift jump for the best configuration occurs at transformer layer **{int(cliff.layer)}**: NMSE changes by {cliff.drift_jump:.4g} to {cliff.teacher_nmse:.4g}. Projection rescue diagnostics for that layer are in `per_projection.csv`.

**Q7: Next bottleneck:** {'the milestone was reached' if q5 else 'the candidate family / fundamental one-bit representation, if horizon and beam ablations do not materially close the gap; the tables distinguish this from search failure'}. No repair mechanism is introduced.

The experiment uses calibration hidden states to select candidates and the unchanged WikiText-2 test split for final perplexity. Candidate patterns are coarse layer-level choices, which keeps metadata negligible but bounds the claim about richer per-matrix binary families.
"""
    (out/"REPORT.md").write_text(text,encoding="utf-8")


def main() -> None:
    args=parse_args(); seed_everything(args.seed); args.output.mkdir(parents=True,exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device=torch.device("cuda"); tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,dtype=torch.bfloat16,attn_implementation="sdpa").to(device).eval()
    n_layers=len(model.model.layers); modules=target_modules(model,set(range(n_layers)))
    calibration=get_text_tokens(tokenizer,"train",args.calibration_tokens,args.sequence_length)
    evaluation=get_text_tokens(tokenizer,"test",args.evaluation_tokens,args.sequence_length)
    originals={n:m.weight.detach().cpu().clone() for n,m in modules.items()}; patterns=candidate_patterns()
    total_weights=sum(x.numel() for x in originals.values()); base, naive, scale_bits={}, {}, {}
    for idx,(name,module) in enumerate(modules.items(),1):
        print(f"[{idx}/{len(modules)}] bases {name}",flush=True)
        q,bits=quantize_matrix(module.weight.detach(),args.block_size,True); qn,_=quantize_matrix(module.weight.detach(),args.block_size,False)
        base[name]=q; naive[name]=qn; scale_bits[name]=bits
    teacher_inputs,teacher_outputs,layer_kwargs=capture_teacher(model,calibration,device)
    sfbs=SFBS(model,modules,originals,base,teacher_inputs,teacher_outputs,layer_kwargs,patterns)
    wstats=weight_stats(originals,base,patterns,n_layers); candidate_rows=[]; configs=[]

    def add_config(name,choices,history,evaluations,runtime,k,beam,horizon,target,base_kind="hadamard"):
        source=base if base_kind=="hadamard" else naive
        apply_full(modules,source,originals,choices,patterns); ppl,nll=safe_perplexity(model,evaluation,device)
        drift_rows,inputs=sfbs.drift(choices,name) if base_kind=="hadamard" else ([],[])
        id_bits=0 if k<=1 else n_layers*math.ceil(math.log2(k)); bits=total_weights+sum(scale_bits.values())+id_bits
        configs.append({"configuration":name,"perplexity":ppl,"mean_nll":nll,"true_bpw":bits/total_weights,
                        "sign_bits":total_weights,"scale_bits":sum(scale_bits.values()),"candidate_id_bits":id_bits,
                        "k":k,"beam":beam,"horizon":horizon,"target":target,"candidate_evaluations":evaluations,
                        "runtime_seconds":runtime,"choices":",".join(map(str,choices))})
        for row in drift_rows: row.update({"perplexity":ppl}); all_drift.append(row)
        for row in history:
            row.update({"configuration":name,"selected":True,"weight_mse":wstats[(row["layer"],row["candidate"])]})
        print(f"RESULT {name}: PPL={ppl} NLL={nll:.4f} BPW={bits/total_weights:.8f}",flush=True)
        return inputs

    all_drift=[]
    bf_ppl,bf_nll=safe_perplexity(model,evaluation,device)
    configs.append({"configuration":"bf16","perplexity":bf_ppl,"mean_nll":bf_nll,"true_bpw":16.,"sign_bits":0,"scale_bits":0,"candidate_id_bits":0,"k":np.nan,"beam":np.nan,"horizon":np.nan,"target":"none","candidate_evaluations":0,"runtime_seconds":0,"choices":""})
    zero=[0]*n_layers
    t=time.time(); add_config("standard_naive",zero,[],0,time.time()-t,1,1,0,"weight",base_kind="naive")
    t=time.time(); add_config("hadamard_naive",zero,[],0,time.time()-t,1,1,0,"weight")
    t=time.time(); choices,hist,ev=sfbs.independent(8); add_config("independent_k8",choices,hist,ev,time.time()-t,8,1,0,"clean")

    searches=[
        ("sequential_k4_h0_current",4,1,0,"current",1.,0.),
        ("sequential_k8_h0_current",8,1,0,"current",1.,0.),
        ("sequential_k16_h0_current",16,1,0,"current",1.,0.),
        ("sequential_k8_h1_current",8,1,1,"current",1.,1.),
        ("sequential_k8_h2_current",8,1,2,"current",1.,1.),
        ("sequential_k8_h4_current",8,1,4,"current",1.,1.),
        ("sequential_k8_h1_clean",8,1,1,"clean",1.,1.),
        ("beam2_k8_h1_l1_0",8,2,1,"current",1.,0.),
        ("beam2_k8_h1_l1_1",8,2,1,"current",1.,1.),
        ("beam2_k8_h1_l05_1",8,2,1,"current",.5,1.),
        ("beam4_k8_h1_l1_1",8,4,1,"current",1.,1.),
    ]
    choice_map={"hadamard_naive":zero,"independent_k8":choices}
    for name,k,beam_width,horizon,target,ll,lh in searches:
        t=time.time(); choices,hist,ev=sfbs.search(k,beam_width,horizon,target,ll,lh,candidate_rows,name)
        add_config(name,choices,hist,ev,time.time()-t,k,beam_width,horizon,target); choice_map[name]=choices

    aggregate=pd.DataFrame(configs); drift=pd.DataFrame(all_drift)
    cand=pd.DataFrame(candidate_rows)
    if len(cand):
        cand["weight_mse"]=[wstats[(int(r.layer),int(r.candidate))] for r in cand.itertuples()]
        cand["weight_mse_rank"]=cand.groupby(["configuration","layer"]).weight_mse.rank(method="min")
        cand["functional_rank"]=cand.groupby(["configuration","layer"]).local_nmse.rank(method="min")
        selected={(config, layer): int(choice) for config, choices_i in choice_map.items() for layer, choice in enumerate(choices_i)}
        cand["selected"]=[selected.get((r.configuration,int(r.layer)))==int(r.candidate) for r in cand.itertuples()]
    best_name=aggregate[(aggregate.configuration!="bf16")&(aggregate.true_bpw<=1.05)].sort_values("perplexity").iloc[0].configuration
    best_choices=choice_map.get(best_name,zero); apply_full(modules,base,originals,best_choices,patterns)
    best_drift,best_inputs=sfbs.drift(best_choices,best_name+"_diagnostic")
    cliff=max(best_drift,key=lambda x:x["drift_jump"]); layer=int(cliff["layer"]); projection_rows=[]
    sfbs.apply_candidate(layer,best_choices[layer]); hidden=best_inputs[layer]
    with torch.no_grad():
        qout=sfbs.call(layer,hidden)
    base_nmse,_=tensor_metrics(teacher_outputs[layer],qout)
    for proj,(name,module) in sfbs.by_layer[layer].items():
        saved=module.weight.detach().clone()
        with torch.no_grad(): module.weight.copy_(originals[name].to(device))
        with torch.no_grad():
            rescued=sfbs.call(layer,hidden)
        nmse,cos=tensor_metrics(teacher_outputs[layer],rescued)
        with torch.no_grad(): module.weight.copy_(saved)
        projection_rows.append({"layer":layer,"projection":proj,"matrix":name,"base_layer_nmse":base_nmse,"rescued_layer_nmse":nmse,"nmse_reduction":base_nmse-nmse,"rescued_cosine":cos})
    for name,module in modules.items():
        layer_i=layer_of(name); candidate_i=best_choices[layer_i]; f=patterns[candidate_i][1][suffix(name)]
        w=originals[name].float(); q=base[name].float()*f
        projection_rows.append({"layer":layer_i,"projection":suffix(name),"matrix":name,"base_layer_nmse":np.nan,"rescued_layer_nmse":np.nan,"nmse_reduction":np.nan,"rescued_cosine":np.nan,"selected_candidate":candidate_i,"scale_factor":f,"weight_mse":float((w-q).square().mean())})

    aggregate["perplexity_degradation_pct"]=100*(aggregate.perplexity/bf_ppl-1)
    aggregate.to_csv(args.output/"aggregate.csv",index=False); drift.to_csv(args.output/"per_layer_drift.csv",index=False)
    cand.to_csv(args.output/"candidate_diagnostics.csv",index=False); pd.DataFrame(projection_rows).to_csv(args.output/"per_projection.csv",index=False)
    aggregate[aggregate.configuration.str.contains("independent|sequential")].to_csv(args.output/"independent_vs_sequential.csv",index=False)
    aggregate[aggregate.beam>1].to_csv(args.output/"beam_ablation.csv",index=False)
    aggregate[aggregate.configuration.str.contains("sequential_k(4|8|16)_h0",regex=True)].to_csv(args.output/"k_ablation.csv",index=False)
    aggregate[aggregate.configuration.str.contains("sequential_k8_h[0124]_current",regex=True)].to_csv(args.output/"horizon_ablation.csv",index=False)
    make_plots(args.output,aggregate,drift,cand); build_report(args.output,aggregate,drift,pd.DataFrame(projection_rows))
    dump_json(args.output/"environment.json",{"args":vars(args),"model_layers":n_layers,"target_matrices":len(modules),"target_weights":total_weights,"gpu":torch.cuda.get_device_name(),"torch":torch.__version__,"platform":platform.platform(),"timestamp_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    with torch.no_grad():
        for name,module in modules.items(): module.weight.copy_(originals[name].to(device))
    print((args.output/"REPORT.md").read_text(),flush=True)


if __name__=="__main__": main()
