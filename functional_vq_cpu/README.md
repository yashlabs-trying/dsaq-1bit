# Layer-0 functional VQ CPU worker

This directory contains the memory-bounded Windows/CPU continuation of the
Qwen2.5-0.5B Layer-0 functional-VQ experiment.  It starts from the frozen E7a
same-byte codebook and changes only packed Gate/Up pattern IDs.  Candidate
assignments are selected on calibration A and accepted only when calibration B
does not worsen.

The CPU worker optimizes exact final MLP output SSE and deliberately does not
claim perplexity during search. A separate strict full-model evaluator measures
PPL from the authoritative packed decode.

## Reproducibility

- Model: `Qwen/Qwen2.5-0.5B`
- Revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`
- Fixed-token SHA-256: `70df7be65f81733de4ae37f80792ea0448d25061ef69b513882d60eae5c722e7`
- Packed size: 2,223,107 bytes
- Exact BPW: 1.1928587525755494
- CPU threads: 4

The 79.6 MB generated functional cache and resumable `.pt` state files are
excluded from Git because they are deterministically reproducible from the
pinned model, token IDs, and scripts.  Their manifest and tensor hashes are
tracked.

## Results

| Stage | Parent | Searched blocks | Changed IDs | Ez(B) | Ey(B) | Ly(B) | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|
| E7a-100 | E6 + adapted codebook | - | - | 0.415834 | 0.377260 | 0.142325 | - |
| E7b 10% | E7a-100 | 54,208 | 17,457 | 0.426988 | 0.359939 | 0.129555 | 283.5 s |
| E7b 20% | E7b 10% | 108,864 | 26,000 | 0.442888 | 0.340501 | 0.115941 | 587.1 s |
| E7c paired G/U 5% | E7b 20% | 13,440 block pairs | 4,116 | 0.443679 | 0.339306 | 0.115128 | 171.9 s |
| E7d Down 5% | E7c | 13,376 | 6,273 | 0.443679 | 0.331496 | 0.109889 | 31.7 s |
| E8a codebook EM | E7d | 6 accepted steps | 937 coordinates | 0.444531 | 0.331274 | 0.109742 | 7.8 s |
| E8b reassignment 10% | E8a | 54,208 | 4,085 | 0.445345 | 0.329233 | 0.108394 | 389.2 s |
| E8c Down round 1 | E8b | 13,376 | 6,255 | 0.445345 | 0.321489 | 0.103355 | 94.7 s |
| E8d Down round 2 | E8c | 13,376 | 5,985 | 0.445345 | 0.314460 | 0.098885 | 94.4 s |
| E8e Down round 3 | E8d | 13,376 | 4,173 | 0.445345 | 0.310231 | 0.096243 | 94.4 s |
| E8f Down round 4 | E8e | 13,376 | 3,318 | 0.445345 | 0.306895 | 0.094184 | 94.3 s |
| E8g Down round 5 | E8f | 13,376 | 2,845 | 0.445345 | **0.304083** | **0.092466** | 92.9 s |

E7b-20 reduced held-out squared output loss by 18.54% relative to E7a while
keeping the packed size unchanged.  Ez(B) worsened while Ey(B) improved, which
is expected: the objective preserves the function after the Down projection,
not the intermediate activation coordinate-by-coordinate.

E7b-20 packed SHA-256:
`97843deaf89ecd6ddcf0d5e555b36fb37e1a809987e23ab1b9dd0ca146fb85df`

E7c confirmed that paired Gate/Up search is valid but low-leverage after E7b:
498,892 candidate-pair evaluations reduced Ly(B) by only 0.70%.  E7d was more
efficient: Down reassignment against the actual student activation reduced
Ly(B) by another 4.55%.  Overall, E7a to E7d reduced held-out Ly(B) by 22.79%
with no added storage.  E7d packed SHA-256:
`de3ae63c998cf7ed201be633b9c7ad580f6a2f9d53f478a4f9ee04bae5706d57`

The second EM round showed that prototype adaptation and Gate/Up reassignment
were close to saturation, but repeated Down reassignment remained productive.
The predefined stop rule was reached on E8g when marginal Ly(B) improvement
fell below 2% (1.82%). E7a to E8g reduced held-out Ly(B) by 35.03% and Ey(B)
by 19.40%, with no added storage. E8g packed SHA-256:
`d6dabae1c6842d90cc634907a8a31801a1961145e56c9d5f03a508afccd1f92f`

## Strict E8g perplexity

The final E8g packed checkpoint was decoded into the pinned full model and
evaluated on all 4,096 fixed evaluation tokens in 16 sequences of length 256:

- NLL: 4.332956179976463
- PPL: **76.16912371426544**
- Reference PPL: 22.188270476805936
- Delta PPL: **+243.28553815804304%**
- CPU evaluation time: 62.84 seconds

This beats E7a's approximately 89.59 PPL but remains above the first target of
33.28 PPL (<50% degradation) and the breakthrough target of 24.41 PPL (<10%).

## Commands

Generate the deterministic cache:

```powershell
python functional_vq_cpu/scripts/export_layer0_cache.py
python functional_vq_cpu/scripts/validate_cache.py
```

Run a reassignment pass (PowerShell):

```powershell
$env:E7B_PARENT = (Resolve-Path 'artifacts/functional_vq_l0/packed/E7a_codebook_100.packed')
$env:E7B_COVERAGE = '.10'
$env:E7B_LABEL = 'E7b_reassign_10pct'
$env:E7B_THREADS = '4'
python functional_vq_cpu/scripts/e7b_cpu_reassign.py
```

The second pass uses `packed/E7b_reassign_10pct.packed` as its parent and
`E7B_COVERAGE=.20`.
