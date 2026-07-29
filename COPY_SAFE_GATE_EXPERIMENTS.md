# Copy-Safe Gate Experiments

Local AHR 3-seed experiments with train_delta_size=4, test_delta_size=12, train_n=16, n=64. The independent copy-safe gate was then validated by the C2 20-seed server rerun summarized in `C2_COPYSAFE_SERVER_SUMMARY.md`.

| Task | Variant | node | graph | changed | copy-critical | gate recall |
|---|---|---:|---:|---:|---:|---:|
| SSSP | baseline | 0.974 +/- 0.004 | 0.520 +/- 0.004 | 0.376 +/- 0.029 | 0.346 +/- 0.034 | 0.873 +/- 0.096 |
| SSSP | copy-safe gate | 0.967 +/- 0.009 | 0.417 +/- 0.040 | 0.477 +/- 0.041 | 0.330 +/- 0.047 | 0.968 +/- 0.030 |
| Bottleneck | baseline | 0.475 +/- 0.182 | 0.000 +/- 0.000 | 0.305 +/- 0.080 | 0.436 +/- 0.047 | 0.984 +/- 0.013 |
| Bottleneck | copy-safe gate | 0.782 +/- 0.227 | 0.156 +/- 0.206 | 0.514 +/- 0.163 | 0.469 +/- 0.073 | 0.993 +/- 0.006 |

Interpretation: copy-safe gating is the first structured local change that materially improves the hard-repair bottleneck setting. On SSSP it raises changed-output repair but hurts graph accuracy; the C2 full-budget 20-seed rerun keeps SSSP node accuracy stable and raises repair slices. On bottleneck, C2 reaches `0.916 +/- 0.092` node, `0.271 +/- 0.184` graph, `0.663 +/- 0.053` changed-output, and `0.746 +/- 0.039` copy-critical accuracy at `n=64`, making copy-safety calibration the strongest current architectural improvement.
