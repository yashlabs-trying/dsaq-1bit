# E19-R 3-bit base result

## Outcome

The exact-decoder 3-bit experiment completed on the RTX A5000. The selected
checkpoint was chosen only by the held-out validation slice; the WikiText-2
test split remained report-only.

| Model/state | Test PPL | Delta vs BF16 |
|---|---:|---:|
| BF16 | 27.7389 | - |
| 3-bit GPTQ base only | 1575.6512 | +5580.3% |
| 3-bit base + initialized factorized residual | 191.3785 | +590.0% |
| 25-step QAT | 48.3345 | +74.25% |
| **100-step run, validation-selected checkpoint** | **45.5440** | **+64.19%** |
| 131K-token QAT, 100 unique windows | 36.4101 | +31.26% |
| **131K-token QAT, 200 unique windows** | **35.4501** | **+27.80%** |

The final checkpoint was selected at step 50. The 1,016-token validation PPL
improved from 177.736 at initialization to 36.660 at step 50, then worsened to
46.729 by step 100 while training loss continued to fall. This is direct
evidence of calibration overfitting after step 50, not an optimization stall.

## Storage

- Current conservative accounting with FP16 zero-points: **2.664115 BPW**.
- Projected accounting with discrete zero-points packed at their natural
  three-bit width: **2.516992 BPW**.
- The research checkpoint is not a packed production file.

## Reproducibility checks

- Cached GPTQ base versus stored-field decode maximum difference: **0.0**.
- In-process test PPL: **45.5439719040**.
- Fresh-process decoded test PPL: **45.5439719040**.
- Winning checkpoint SHA-256:
  `f969ab752b7f67ea5fa3eb45b606da0afff0fe473f63e77f2be9b2ce96b902c8`.
- Winning checkpoint size: 395,944,879 bytes (research container).

## Interpretation and next decision

The corrected QAT remains powerful after dropping the base from four bits to
three: it reduced PPL from 191.38 to 45.54. It did not recover the required
less-than-10% degradation, however. We should not descend to two bits yet.

The immediate limiting factor is the tiny repeated calibration set: the
100-step run has only 32 distinct 128-token training windows, so it starts its
fourth pass through the same data after step 96. Validation began degrading much
earlier. The next controlled 3-bit run should use more distinct calibration
windows, full 4,096-token validation selection, and the same validated codec and
optimizer. If that does not close the gap, the next lever is assignment
re-optimization or larger residual capacity funded by the packed-zero savings.

## Large-data follow-up

The controlled follow-up changed only the QAT data regime: 131,072 unique
calibration tokens supplied 1,024 shuffled windows, the full 4,096-token
validation split selected checkpoints, and the test split remained report-only.
GPTQ/Hessian construction remained fixed at the original 32 windows.

At 100 unique training windows, test PPL reached 36.4101. Because validation was
still improving, the otherwise identical 200-step run continued to 200 unique
windows. Its best validation checkpoint occurred at step 190 and decoded to
test PPL **35.4501**, exactly reproduced in a fresh process. Its SHA-256 is
`086b070742a7f5206f091197fc590a16d1172cf19d511a17a203044396b102d9`.

This establishes that calibration diversity caused a large part of the earlier
stall: 45.5440 fell to 35.4501 without changing storage or representation. The
curve is now flattening, however. Full validation PPL moved only
26.6390 -> 26.4755 -> 26.4305 -> 26.3065 from steps 150 through 190, and step
200 was rejected. More value-only steps are now lower leverage. The next
controlled operation should be a discrete assignment refresh using the trained
3-bit codebooks, followed by another short value-QAT pass. Do not descend to a
2-bit base until the 3-bit representation either crosses 30.5128 test PPL or
demonstrates a clear assignment/capacity ceiling.
