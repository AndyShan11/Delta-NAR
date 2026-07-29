# Day 2/3 Twenty-Seed Summary

## sssp

| Model | n | node_acc_v | graph_acc_v | changed fraction |
|---|---:|---:|---:|---:|
| DeltaNAR | 16 | 0.9910 +/- 0.0017 | 0.8945 +/- 0.0181 | 3.345e-02 |
| DeltaNAR | 64 | 0.9961 +/- 0.0106 | 0.9556 +/- 0.0156 | 2.222e-03 |
| DeltaNAR | 128 | 0.9839 +/- 0.0390 | 0.9109 +/- 0.2140 | 3.937e-04 |
| DeltaNAR | 256 | 0.9452 +/- 0.1259 | 0.8539 +/- 0.2979 | 1.255e-04 |
| DeltaNAR | 512 | 0.8948 +/- 0.2815 | 0.8415 +/- 0.3540 | 3.914e-06 |
| Naive-d | 16 | 0.9596 +/- 0.0227 | 0.6166 +/- 0.1180 | 3.345e-02 |
| Naive-d | 64 | 0.7717 +/- 0.1172 | 0.0016 +/- 0.0039 | 2.222e-03 |
| Naive-d | 128 | 0.5859 +/- 0.1429 | 0.0000 +/- 0.0000 | 3.937e-04 |
| Naive-d | 256 | 0.4224 +/- 0.1188 | 0.0000 +/- 0.0000 | 1.255e-04 |
| Naive-d | 512 | 0.2996 +/- 0.0790 | 0.0000 +/- 0.0000 | 3.914e-06 |
| Recompute | 16 | 0.4975 +/- 0.0164 | 0.0106 +/- 0.0013 | 3.345e-02 |
| Recompute | 64 | 0.1623 +/- 0.0192 | 0.0000 +/- 0.0000 | 2.222e-03 |
| Recompute | 128 | 0.0774 +/- 0.0123 | 0.0000 +/- 0.0000 | 3.937e-04 |
| Recompute | 256 | 0.0376 +/- 0.0041 | 0.0000 +/- 0.0000 | 1.255e-04 |
| Recompute | 512 | 0.0291 +/- 0.0018 | 0.0000 +/- 0.0000 | 3.914e-06 |

## bottleneck

| Model | n | node_acc_v | graph_acc_v | changed fraction |
|---|---:|---:|---:|---:|
| DeltaNAR | 256 | 0.8843 +/- 0.2102 | 0.0192 +/- 0.0188 | 1.647e-04 |
| DeltaNAR | 512 | 0.8571 +/- 0.2307 | 0.0110 +/- 0.0129 | 2.348e-05 |
| Naive-d | 256 | 0.3266 +/- 0.0619 | 0.0000 +/- 0.0000 | 1.647e-04 |
| Naive-d | 512 | 0.2366 +/- 0.0383 | 0.0000 +/- 0.0000 | 2.348e-05 |
| Recompute | 256 | 0.1392 +/- 0.0298 | 0.0000 +/- 0.0000 | 1.647e-04 |
| Recompute | 512 | 0.1216 +/- 0.0200 | 0.0000 +/- 0.0000 | 2.348e-05 |

## Paired bootstrap headline comparisons

| Task | n | Metric | Comparator | Difference | 95% CI | BH p |
|---|---:|---|---|---:|---:|---:|
| sssp | 16 | node_acc_v | Naive-d | 0.0314 | [0.0228, 0.0427] | 0 |
| sssp | 16 | graph_acc_v | Naive-d | 0.2779 | [0.2274, 0.3344] | 0 |
| sssp | 64 | node_acc_v | Naive-d | 0.2243 | [0.1772, 0.2801] | 0 |
| sssp | 64 | graph_acc_v | Naive-d | 0.9540 | [0.9464, 0.9600] | 0 |
| sssp | 128 | node_acc_v | Naive-d | 0.3980 | [0.3293, 0.4700] | 0 |
| sssp | 128 | graph_acc_v | Naive-d | 0.9109 | [0.8054, 0.9739] | 0 |
| sssp | 256 | node_acc_v | Naive-d | 0.5228 | [0.4276, 0.6081] | 0 |
| sssp | 256 | graph_acc_v | Naive-d | 0.8539 | [0.7076, 0.9664] | 0 |
| sssp | 512 | node_acc_v | Naive-d | 0.5952 | [0.4345, 0.7174] | 0 |
| sssp | 512 | graph_acc_v | Naive-d | 0.8415 | [0.6864, 0.9888] | 0 |
| sssp | 16 | node_acc_v | Recompute | 0.4935 | [0.4861, 0.5005] | 0 |
| sssp | 16 | graph_acc_v | Recompute | 0.8839 | [0.8758, 0.8917] | 0 |
| sssp | 64 | node_acc_v | Recompute | 0.8337 | [0.8252, 0.8431] | 0 |
| sssp | 64 | graph_acc_v | Recompute | 0.9556 | [0.9478, 0.9613] | 0 |
| sssp | 128 | node_acc_v | Recompute | 0.9065 | [0.8876, 0.9208] | 0 |
| sssp | 128 | graph_acc_v | Recompute | 0.9109 | [0.8039, 0.9737] | 0 |
| sssp | 256 | node_acc_v | Recompute | 0.9076 | [0.8438, 0.9559] | 0 |
| sssp | 256 | graph_acc_v | Recompute | 0.8539 | [0.7108, 0.9673] | 0 |
| sssp | 512 | node_acc_v | Recompute | 0.8657 | [0.7245, 0.9661] | 0 |
| sssp | 512 | graph_acc_v | Recompute | 0.8415 | [0.6885, 0.9886] | 0 |
| bottleneck | 256 | node_acc_v | Naive-d | 0.5577 | [0.4656, 0.6301] | 0 |
| bottleneck | 256 | graph_acc_v | Naive-d | 0.0192 | [0.0115, 0.0279] | 0 |
| bottleneck | 512 | node_acc_v | Naive-d | 0.6204 | [0.5169, 0.7024] | 0 |
| bottleneck | 512 | graph_acc_v | Naive-d | 0.0110 | [0.0057, 0.0168] | 0 |
| bottleneck | 256 | node_acc_v | Recompute | 0.7451 | [0.6398, 0.8267] | 0 |
| bottleneck | 256 | graph_acc_v | Recompute | 0.0192 | [0.0114, 0.0277] | 0 |
| bottleneck | 512 | node_acc_v | Recompute | 0.7355 | [0.6207, 0.8281] | 0 |
| bottleneck | 512 | graph_acc_v | Recompute | 0.0110 | [0.0057, 0.0170] | 0 |
