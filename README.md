# DeltaNAR / CLRS-Delta

[![Oracle tests](https://github.com/AndyShan11/Delta-NAR/actions/workflows/ci.yml/badge.svg)](https://github.com/AndyShan11/Delta-NAR/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Learn what to reuse and what to recompute after a local graph edit.**

This repository is the code, data, result, and reproducibility release for
*DeltaNAR: Learning What to Reuse and What to Recompute in Neural Algorithmic
Reasoning*.

Standard neural algorithmic reasoning (NAR) solves every instance from scratch.
CLRS-Delta studies the incremental alternative:

```text
(post-edit graph, previous solution, edit)
                         │
                         ▼
       affected-region and copy-safety gates
                         │
                         ▼
        (updated solution, affected region)
```

DeltaNAR explicitly separates safe reuse from necessary recomputation. It
combines previous-solution memory, edit features, an affected-region gate, a
copy-safety gate, and a copy-not-recompute decoder.

## What is included

- Dynamic graph generators and verified SSSP, widest-path, and MST oracles.
- CLRS-Delta datasets for local graph edits.
- DeltaNAR and matched ablation/baseline implementations.
- Multi-seed training, OOD evaluation, hard-slice, noisy-memory, efficiency,
  and self-correction runners.
- Raw result JSON files and scripts used to regenerate analyses and figures.
- Oracle/unit tests and artifact-verification utilities.
- The matched DEAR evaluation implementation used in the reported comparison.

## Headline findings

The included artifacts support the following paper-level observations:

- A model trained at graph size `n=16` reaches `0.987 ± 0.028` SSSP node
  validity at `n=256` over 20 seeds.
- On an edit-conditioned hard slice, DeltaNAR reaches `0.846` node validity,
  compared with `0.426` for previous-solution features without explicit
  copying and `0.042` for recomputation.
- Pure copying can achieve high node-average validity while having zero
  whole-graph validity, which motivates reporting both metrics.
- The node-level theory predicts scaling with the affected fraction under its
  stated local-edit assumptions; it is not a whole-graph exactness guarantee.

See [`RESULTS.md`](RESULTS.md) for the complete result record and
[`THEORY_FINAL.md`](THEORY_FINAL.md) for theorem statements and boundaries.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/AndyShan11/Delta-NAR.git
cd Delta-NAR
python -m venv .venv
```

Activate the environment:

```bash
# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA experiments, install the PyTorch build matching your local CUDA
runtime. The matched DEAR evaluation under `revision/code/DEAR_official/` has
additional PyTorch Geometric and DEQ dependencies documented in that subtree.

## Five-minute verification

The oracle tests do not require a GPU:

```bash
python tests/test_dynamic_sssp.py
python tests/test_bottleneck.py
python tests/test_dataset.py
python tests/test_mst.py
python tests/test_mst_dataset.py
```

Expected totals are 5 dynamic-SSSP tests, 2 bottleneck tests, 3 dataset tests,
3 MST tests, and 3 MST-dataset tests.

Run a small CPU/GPU training smoke test:

```bash
python experiments.py \
  --tasks sssp \
  --models DeltaNAR \
  --seeds 0 \
  --sizes 16 64 \
  --train_episodes 50 \
  --test_episodes 10 \
  --epochs 5 \
  --out results/_smoke.json
```

Run the local diagnostic grid:

```bash
python diagnose_a_metrics.py \
  --tasks sssp \
  --models DeltaNAR no-copy Naive-d Recomp+gate Recompute \
  --seeds 0 1 2 \
  --sizes 16 64 128 \
  --out results/A_local_diagnostics_sssp_3seed.json \
  --markdown RESULTS_AUDIT_sssp_3seed.md
```

## Repository map

| Path | Purpose |
|---|---|
| `src/graphs.py` | Graph generation and edit sampling |
| `src/sssp.py` | Dijkstra and max-min bottleneck oracles |
| `src/dynamic_sssp.py` | LPA*-style dynamic SSSP oracle |
| `src/mst.py` | MST generators and oracles |
| `src/dataset*.py` | Incremental SSSP, bottleneck, and MST datasets |
| `src/model.py` | DeltaNAR model, losses, batching, and metrics |
| `experiments.py` | Multi-seed size/OOD experiment runner |
| `diagnose_*.py` | Hard-slice, oracle-mask, and repair diagnostics |
| `results/` | Raw and merged result artifacts |
| `paper/scripts/` | Figure and theory-validation scripts |
| `revision/code/` | Matched revision and DEAR comparison code |
| `server_scripts/` | Recorded multi-GPU launch configurations |
| `tests/` | Deterministic oracle and dataset tests |

For a file-by-file inventory, see
[`ARTIFACT_MANIFEST.md`](ARTIFACT_MANIFEST.md). For full environment,
determinism, and command details, see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Reproducing different levels of evidence

1. **Logic verification:** run the five oracle/dataset tests.
2. **Smoke training:** run the one-seed, two-size command above.
3. **Core grid:** use `experiments.py` with the task, model, seed, and size
   grid documented in `REPRODUCIBILITY.md`.
4. **Paper audit:** run `verify_goal_artifacts.py` and inspect the raw JSON
   files referenced by `RESULTS.md`.
5. **Figure regeneration:** run `python paper/scripts/make_figures.py`.

Full paper-scale campaigns require substantially more compute than the smoke
tests. CUDA kernels may retain low-level nondeterminism, so the reported
evidence uses multiple seeds and stores seed-level outputs.

## Scope and limitations

- The main theorem is node-level and conditional on local edits.
- Low whole-graph exactness remains visible in the included reports.
- MST is included as a difficulty gradient, not presented as a solved headline
  task.
- The repository contains research code and recorded experiments rather than a
  production dynamic-graph service.

## Citation

Archival citation metadata will be added after publication. Until then, cite
the manuscript title and pin the repository commit used in your experiments.

## License

The original DeltaNAR code in this repository is released under the
[MIT License](LICENSE). Third-party code under
`revision/code/DEAR_official/` retains its own license and attribution.
