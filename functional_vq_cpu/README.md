# Layer-0 functional VQ CPU worker

This directory contains the memory-bounded Windows/CPU continuation of the
Qwen2.5-0.5B Layer-0 functional-VQ experiment.  It starts from the frozen E7a
same-byte codebook and changes only packed Gate/Up pattern IDs.  Candidate
assignments are selected on calibration A and accepted only when calibration B
does not worsen.

The CPU worker optimizes exact final MLP output SSE and deliberately does not
claim perplexity.  A strict full-model evaluation is required for PPL.

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

E7b-20 reduced held-out squared output loss by 18.54% relative to E7a while
keeping the packed size unchanged.  Ez(B) worsened while Ey(B) improved, which
is expected: the objective preserves the function after the Down projection,
not the intermediate activation coordinate-by-coordinate.

E7b-20 packed SHA-256:
`97843deaf89ecd6ddcf0d5e555b36fb37e1a809987e23ab1b9dd0ca146fb85df`

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
