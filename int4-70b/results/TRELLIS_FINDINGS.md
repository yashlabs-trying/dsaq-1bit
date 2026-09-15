# Controlled learned-trellis experiment

Status: implementation and correctness checks complete; GPU measurements pending.

This independent prototype tests the proposed combination of two-sided
DCT/Hadamard mixing, activation Hessians, K-means patterns, a bit-shift trellis,
and Block-LDLQ compensation. It is not an exact QTIP reproduction or a claim
that these established ingredients are novel.

## Fixed controls

- Qwen2.5-0.5B base, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- RTX 3090, one GPU; no original model finetuning.
- 8,192 WikiText-2 training calibration tokens, 256-token windows.
- Same two-sided transforms and 0.01 relative Hessian damping as the current E8P adaptation.
- One discrete refinement sweep after Block-LDLQ.
- Fixed and learned trellises use identical serialized storage: exactly 2 bits per compressed weight.

## Prototype and limits

The trellis has 256 states. Each four-bit symbol advances the state and selects
a two-dimensional node value. Four symbols represent an eight-weight block in
a uint16 path. The start state is fixed at zero for every block. This reset
limits search cost and long-range modeling; it is not QTIP's long-chain codec.

Each matrix stores 256 two-dimensional FP16 node values (1,024 bytes) and an
18-byte shape/seed/scale header. The final 521 predetermined blocks have
implicit path zero, paying exactly for the 1,042 bytes of metadata. These paths
are constrained during optimization. Fixed tables are stored too, so fixed and
learned node comparisons have identical byte and position budgets.

K-means fits transformed, normalized weight pairs. Node placement is fixed;
initial clustering does not optimize the trellis topology. The optional
sensitivity-aware search weights additive pair errors by the diagonal of the
conditional Block-LDLQ metric. Full correlations enter error compensation and
the acceptance test for refinement; Viterbi is not an exact full-Hessian solver.

The first gate compares attention and MLP matrices with fixed nodes, learned
nodes, learned nodes with weighted search, and the current E8P method. Its
Hessian error is a calibration proxy, not model perplexity or a speed benchmark.

## Reproduce

```bash
python -m unittest test_trellis_quant test_additive_vq test_lattice_experiment test_calibration
FLASHRUN_OUTPUT=/workspace/trellis_gate python -u trellis_experiment.py
FLASHRUN_TRELLIS_FULL=1 FLASHRUN_TRELLIS_VARIANT=learned_weighted \
FLASHRUN_OUTPUT=/workspace/trellis_full python -u trellis_experiment.py
```

Use the dependencies and external pinned QuIP# checkout documented in
[3090_LATTICE_FINDINGS.md](3090_LATTICE_FINDINGS.md). The E8P source is external,
GPL-3.0 licensed, and used only for the comparison gate. The trellis code is
an independent prototype of established bit-shift trellis/Viterbi ideas:
[QTIP](https://arxiv.org/abs/2406.11235),
[QuIP#](https://arxiv.org/abs/2402.04396).

Fourteen correctness tests passed on the pod, including exhaustive comparison
of Viterbi with all 65,536 paths and exact two-bit serialization round trips.
