# Controlled learned-trellis experiment

Status: basic full-model test complete; additional ablations in progress.

The basic learned-node, sensitivity-weighted trellis **regressed** to 193.0078
perplexity on the original slice and 434.0582 on the next slice. Our current
E8P adaptation scored 39.3299 and 55.9547 respectively in the preceding run.
BF16 reproduced exactly at 22.1719 and 27.5071 on this new pod. The basic
trellis's serialized transformer-linear payload is exactly 2.00 bpw.

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

## Initial measurements

| Matrix | E8P relative Hessian error | Fixed trellis | K-means trellis | K-means + weighted search |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 attention Q | 0.001848 | 0.004584 | 0.004200 | 0.004213 |
| Layer 0 MLP down | 0.010349 | 0.016009 | 0.016216 | 0.016237 |
| Layer 12 attention Q | 0.019060 | 0.032584 | 0.032500 | 0.032466 |

The first gate took 39.81 seconds; the basic full-model run took 125.65 seconds.
These are development encoding/evaluation durations, not compressed inference
speed results. See [trellis_gate_initial.json](trellis_gate_initial.json).

Path-aware node fitting alternates discrete Viterbi assignment with centroid
least-squares updates. It improves all three matrix proxies relative to the
basic learned trellis. This is quantizer fitting, not model finetuning. A
second ablation pins node zero to zero, so implicit paths reconstruct a neutral
vector. These upgrades are not yet accepted based on model perplexity.

A 16-weight reset variant is also supported through
`FLASHRUN_TRELLIS_WIDTH=16`. It stores uint32 paths; 261 omitted paths pay for
1,042 metadata bytes, leaving two additional bytes of savings per matrix.
Its total payload is slightly below 2.00 bpw. The reset width is an algorithm
format setting required when decoding these files; it is recorded in reports.

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

Seventeen correctness tests passed on the pod, including exhaustive comparison
of Viterbi with all 65,536 paths and exact two-bit serialization round trips.
