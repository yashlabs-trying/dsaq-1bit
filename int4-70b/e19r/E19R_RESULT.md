# E19-R — Correct Value-Only Functional QAT

## Outcome

The recovered E19 optimization failure was implementation-induced. A corrected
25-step value-only QAT run crossed the branch's quality target in a fresh decode:

| State | NLL | PPL | vs BF16 |
|---|---:|---:|---:|
| BF16 test | 3.32284 | 27.7389 | — |
| GPTQ 4-bit base | 3.89924 | 49.3648 | +78.0% |
| Factorized initialization | 3.59584 | 36.4462 | +31.4% |
| In-process selected state | 3.35585 | 28.6700 | +3.36% |
| **Fresh-process decode** | **3.35731** | **28.7120** | **+3.51%** |

The branch-specific `<10%` threshold is `27.7389 × 1.10 = 30.5128`, so the
freshly decoded checkpoint passes by 1.8008 PPL.

## Crucial bitrate correction

This is **not** a 1.20-BPW result. The recovered representation uses a 4-bit
GPTQ base plus a factorized residual and has theoretical accounting of:

`3.3884123897 BPW`

The saved `.pt` is a research checkpoint with unpacked tensors, not a production
bitstream. It is 395,944,879 bytes. The source therefore labels the run E19-R
at 3.39 BPW even though the old runner called it “1.20 BPW.”

## What was fixed

The old run had more faults than the original forensic report identified:

1. `C1/C2 → weight → loss` was disconnected by `.data.copy_()`, so prototype
   QAT did not update codebooks from language-model loss.
2. Teacher and student were the same mutable object.
3. CE divided by windows rather than prediction tokens, inflating its scale.
4. Assignment search ignored original weights and effectively targeted a zero
   residual.
5. Assignment IDs were treated as if numerical neighbors had semantic locality.
6. Tens of thousands of assignments moved on stale gradients.
7. Centroids were not refit after reassignment.
8. Best-state snapshots were shallow references and could be mutated later.
9. The stored FP16 scale/zero decode was not the exact base used during training.
10. The claimed 1.20-BPW label contradicted the measured 3.3884 BPW.

E19-R uses `torch.func.functional_call` so the decoded MLP weights remain in the
autograd graph, freezes every original model parameter, uses a separate cached
BF16 teacher distribution, normalizes CE/KL per token, applies FP16 fake-storage
in every forward, and accepts checkpoints only on a disjoint validation split.

## Reproducibility

- Model: `Qwen/Qwen2.5-0.5B`
- Training tokens: first 4,096 WikiText-2 train tokens
- Selection: first 1,024 WikiText-2 validation tokens
- Full validation: 4,096 WikiText-2 validation tokens
- Final report only: 4,096 WikiText-2 test tokens
- Sequence window: 128 tokens, boundaries preserved
- Seed: 42
- Steps: 25
- LR: `1e-4`
- KL weight: `0.20`
- Winning checkpoint SHA-256:
  `0e116a4d1a80e57e8c1607e764ecdd23184e206b6b16abdb902b0908a443ee8a`

The no-op functional-call check matched ordinary decoded-model CE exactly, and
all 144 MLP codebook tensors had nonzero gradients on every recorded step.

## Remaining serialization discrepancy

The in-process selected model produced PPL 28.6700 while a fresh decode produced
28.7120. Both pass the target. The 0.042-PPL difference comes from training with
the cached GPTQ reconstruction while the checkpoint decodes FP16 scale/zero
metadata. The next run must construct its training base through the same stored
decoder used by the fresh process.

## Files

- `e19_repaired.py`: recovered old GPU source retained for provenance.
- `e19r_value_sanity.py`: corrected value-only QAT experiment.
- `e19r_verify.py`: independent fresh-process evaluator.
- `recovered_gpu/e19_gpu_run_source.py`: exact source read back from the old pod.
- `recovered_gpu/report_e19_bpw1.20.json`: recovered old result JSON.
- `e19r_results/report.json`: complete new run metrics and step history.
- `e19r_results/fresh_verify.json`: fresh decode result.
- `e19r_results/e19r_value_best.pt`: local-only winning checkpoint (Git-ignored).

## Next decision

The optimizer hypothesis is validated. Do not spend time on discrete assignment
search yet: the quality target is already crossed at 3.39 BPW. First make the
stored decode bit-exact, then reduce the base/residual storage toward the true
1.20-BPW constraint while using this corrected QAT loop for recovery.

## Rate audit and next experiment

`e19r_bit_audit.py` decomposes the frozen checkpoint's proposed packed payload.
The 4-bit GPTQ IDs use 2.89719 model-BPW (85.5% of all bits). By contrast, the
two residual ID streams plus both residual codebooks together use only 0.12907
model-BPW. The high-value rate strategy is therefore to preserve the validated
factorized repair mechanism and reduce the GPTQ base through 3, 2, and 1 bit,
running the corrected QAT recovery after each step.

With the current FP16 scales and residual representation, a 1-bit base would
project to 1.21552 BPW if zero-points remained wastefully stored as FP16. Packing
the discrete zero-points at their natural one-bit width projects to 1.04576 BPW,
leaving about 0.154 BPW below the 1.20 target for format metadata or additional
repair capacity. These are storage projections, not quality results.

The runner now constructs its residual target and cached GPTQ base from the
exact FP16 scale/zero representation saved in the checkpoint. It also fails
closed unless that cached base is bit-identical to a decode from stored fields.
