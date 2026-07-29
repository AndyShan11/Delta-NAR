# Code and Data Supplement

This anonymous artifact accompanies *DeltaNAR: Learning What to Reuse and What
to Recompute in Neural Algorithmic Reasoning*.

## Contents

- `src/`: graph generators, verified dynamic oracles, datasets, DeltaNAR, and
  training code.
- `tests/`: unit tests for dynamic SSSP, bottleneck paths, MST, and datasets.
- `results/`: seed-level JSON outputs used by the analyses and paper tables.
- `paper/scripts/` and `paper/tables/`: figure-generation scripts and exported
  numeric tables.
- root-level Python files: experiment runners and result-analysis scripts.
- `README.md` and `REPRODUCIBILITY.md`: overview and reproduction commands.

All benchmark instances are generated synthetically; no external dataset is
required. The archive intentionally excludes author-identifying server
runbooks, caches, LaTeX build products, and obsolete result archives.

## Quick Verification

From the artifact root, run:

```text
python tests/test_dynamic_sssp.py
python tests/test_bottleneck.py
python tests/test_dataset.py
python tests/test_mst.py
python tests/test_mst_dataset.py
```

See `REPRODUCIBILITY.md` for environment details, experiment commands, seeds,
metrics, and expected outputs.
