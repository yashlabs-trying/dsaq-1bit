# E19-R 3-bit assignment refresh

## Parent

- Frozen parent: 131K-token, 200-step 3-bit value-QAT checkpoint.
- Test PPL: **35.45005197**.
- Theoretical payload: **2.664115 BPW** with FP16 zero-points, or a projected
  **2.516992 BPW** with natural-width packed zero-points.
- Parent SHA-256:
  `086b070742a7f5206f091197fc590a16d1172cf19d511a17a203044396b102d9`.

## Protocol

The parent used the first 200 windows in the deterministic shuffled order.
Assignment gradients used the next 128 windows, exact acceptance used the next
128, and sanity used the following 120. The official validation split was
reserved for a macro gate and test remained untouched until selection ended.

For every one of 6,537,240 MLP residual packages, the search exhaustively
scored all 16 C1 choices with C2 fixed and all 16 C2 choices with C1 fixed. It
used signed first-order LM-loss change, never numeric ID neighbors. Storage was
unchanged.

## Results

Baseline internal PPL was 33.256 on acceptance and 34.940 on sanity.

| Changed packages | Acceptance PPL | Sanity PPL | Accepted? |
|---:|---:|---:|:---:|
| 1 | 35.584 | 37.209 | No |
| 4 | 35.741 | 37.359 | No |
| 16 | 38.008 | 39.749 | No |
| 64 | 56.952 | 60.356 | No |
| 256 | 72.913 | 75.258 | No |
| 16,343 (0.25% diagnostic) | 51,503.8 | 49,104.5 | No |
| 32,686 (0.50% diagnostic) | 153,299.4 | 143,189.4 | No |

No candidate improved both internal splits, so the validation macro gate was
never opened. The original parent remained selected, final test PPL remained
35.45005197, and no value-QAT polish was run.

## Interpretation

This rejects **unregularized first-order assignment ranking**. It does not prove
that all assignment refreshes are useless. A stored-code substitution is a
large finite jump across 48 weights; `g dot delta_W` has no curvature or trust
penalty and selected destructive extreme moves. The failure of even the top one
move on both disjoint holdouts is strong enough that scaling this method further
would be scientifically unjustified.

The next assignment proposal must include finite-step curvature, for example a
diagonal-Fisher/Gauss-Newton penalty:

`score = g dot delta_W + 0.5 * lambda * sum(F * delta_W^2)`

and should first verify that its top individual moves improve the proposal set
under exact evaluation before coordinated batches are attempted. Until such a
proposal passes, the frozen 35.450 checkpoint remains the 3-bit champion.
