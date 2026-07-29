# A-line Local Diagnostic Metrics

Small local diagnostic run for targeted metric validation. Do not treat these as final server-scale paper numbers.

```json
{
  "tasks": [
    "sssp"
  ],
  "models": [
    "DeltaNAR"
  ],
  "seeds": [
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39
  ],
  "sizes": [
    64,
    128,
    256,
    512
  ],
  "train_n": 16,
  "train_episodes": 500,
  "test_episodes": 100,
  "epochs": 50,
  "batch": 64,
  "eval_batch": 1,
  "hidden": 128,
  "lr": 0.001,
  "out": "<REDACTED_RELATIVE_PATH>",
  "markdown": "<REDACTED_RELATIVE_PATH>",
  "test_topology": "er",
  "test_heavy_tail": false,
  "test_weight_hi": 10,
  "test_delta_kinds": null,
  "train_delta_kinds": null,
  "train_delta_size": 1,
  "test_delta_size": 1,
  "steps": 0,
  "factorized_eval": true,
  "cache_eval_data": true
}
```

## sssp

### node_acc_v

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.998 +/- 0.001 | 0.999 +/- 0.001 | 0.998 +/- 0.005 | 0.996 +/- 0.016 |

### graph_acc_v

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.949 +/- 0.023 | 0.960 +/- 0.026 | 0.916 +/- 0.201 | 0.926 +/- 0.216 |

### affected_node_acc_v

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.995 +/- 0.002 | 0.999 +/- 0.001 | 0.999 +/- 0.000 | 1.000 +/- 0.000 |

### changed_output_acc_v

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.994 +/- 0.002 | 0.997 +/- 0.001 | 0.999 +/- 0.001 | 1.000 +/- 0.000 |

### copy_critical_acc_v

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.997 +/- 0.001 | 0.999 +/- 0.001 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |

### pure_copy_node_acc_v

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.999 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 | 1.000 +/- 0.000 |

### pure_copy_changed_acc_v

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.992 +/- 0.000 | 0.995 +/- 0.000 | 0.998 +/- 0.000 | 1.000 +/- 0.000 |

### gate_recall

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.994 +/- 0.006 | 0.998 +/- 0.004 | 0.998 +/- 0.003 | 1.000 +/- 0.000 |

### copy_critical_gate_recall

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.993 +/- 0.007 | 0.997 +/- 0.005 | 0.999 +/- 0.002 | 1.000 +/- 0.000 |

### affected_fraction

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.002 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

### changed_output_fraction

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.002 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

### copy_critical_fraction

| Model | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.001 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |


