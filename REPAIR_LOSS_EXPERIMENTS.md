# Repair-Loss Exploratory Experiments

Local SSSP AHR probes on seed 0, train4/test12, n=64. These are design diagnostics, not promoted paper results.

| Variant | node | graph | changed | copy-critical | gate recall | Note |
|---|---:|---:|---:|---:|---:|---|
| baseline scale5 margin0 seed0 | 0.975 | 0.517 | 0.368 | 0.373 | 0.962 | Original DeltaNAR AHR seed0. |
| copy-safe gate, 3 seeds | 0.967 +/- 0.009 | 0.417 +/- 0.040 | 0.477 +/- 0.041 | 0.330 +/- 0.047 | 0.968 +/- 0.030 | Best changed-output repair gain, but full-node and graph accuracy drop. |
| copy-safe gate seed0 | 0.971 +/- 0.000 | 0.448 +/- 0.000 | 0.534 +/- 0.000 | 0.377 +/- 0.000 | 0.977 +/- 0.000 | Strong single-seed changed gain; confirmed directionally by 3-seed run. |
| copy_margin=1, 3 seeds | 0.977 +/- 0.002 | 0.479 +/- 0.042 | 0.402 +/- 0.037 | 0.387 +/- 0.115 | 0.610 +/- 0.422 | Raises changed/copy-critical on average, but graph drops and gate recall is unstable. |
| copy_margin=1 seed0 | 0.980 +/- 0.000 | 0.537 +/- 0.000 | 0.454 +/- 0.000 | 0.550 +/- 0.000 | 0.013 +/- 0.000 | Best single-seed calibration result; not stable enough alone. |
| copy_margin=0.5 seed0 | 0.941 +/- 0.000 | 0.275 +/- 0.000 | 0.278 +/- 0.000 | 0.311 +/- 0.000 | 0.998 +/- 0.000 | Intermediate margin performs worse than baseline. |
| copy_margin=-1 seed0 | 0.976 +/- 0.000 | 0.480 +/- 0.000 | 0.391 +/- 0.000 | 0.299 +/- 0.000 | 0.972 +/- 0.000 | More aggressive copy gives mild changed gain but graph loss. |
| repair rw=1 cw=0.5 | 0.923 +/- 0.000 | 0.233 +/- 0.000 | 0.441 +/- 0.000 | 0.376 +/- 0.000 | 0.978 +/- 0.000 | Raises changed-output but damages full-node and graph accuracy. |
| repair rw=0.25 | 0.921 +/- 0.000 | 0.362 +/- 0.000 | 0.385 +/- 0.000 | 0.403 +/- 0.000 | 0.980 +/- 0.000 | Smaller repair weight still loses full-node/graph accuracy. |
| copy_scale=10 | 0.975 +/- 0.000 | 0.458 +/- 0.000 | 0.356 +/- 0.000 | 0.328 +/- 0.000 | 0.888 +/- 0.000 | Stronger copy bias does not improve changed-output repair. |

Conclusion: the independent copy-safe gate is the most promising local direction because it raises changed-output accuracy from the 3-seed baseline 0.376 to 0.477, but it still trades away full-node and graph accuracy. Naive repair-focused CE has the same trade-off more severely. A larger copy margin gives some average changed/copy-critical gain, but graph accuracy drops and gate recall becomes unstable. The next useful change should combine copy-safety calibration with a repair branch that does not damage the copied region.
