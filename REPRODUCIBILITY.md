# Reproducibility Notes

Date: 2026-06-02

This file is the artifact-facing reproduction guide for the local package.

## Environment

Local tested environment:

- Windows / PowerShell
- Python: 3.12 tested locally; set `$env:NAR_PYTHON` to the interpreter used
  for reproduction, or leave it unset to use `python`.
- PyTorch: 2.6.0+cu124
- CUDA available locally
- TeX: TeX Live 2024

For PowerShell snippets below:

```powershell
if (-not $env:NAR_PYTHON) { $env:NAR_PYTHON = "python" }
```

## Determinism

Training scripts set Python, NumPy, and Torch seeds through `set_seed`. CUDA
training can still have low-level nondeterminism. Paper results should therefore
be reported over multiple seeds and stored with raw seed-level values.

## Oracle Tests

Run:

```powershell
python tests\test_dynamic_sssp.py
python tests\test_bottleneck.py
python tests\test_dataset.py
python tests\test_mst.py
python tests\test_mst_dataset.py
```

Expected output:

- dynamic SSSP: 5 tests pass
- bottleneck: 2 tests pass
- dataset: 3 tests pass
- MST: 3 tests pass
- MST dataset: 3 tests pass

## Main Experiment Grid

Example:

```powershell
& $env:NAR_PYTHON experiments.py --tasks sssp bottleneck --models DeltaNAR Naive-d Recompute Recomp+gate --seeds 0 1 2 3 4 --sizes 16 32 64 128 256 --out results/main_grid.json
```

Existing primary artifacts:

- `results/SEED14_sssp.json`
- `results/SEED14_bot.json`
- `results/A_sssp.json`
- `results/A_bot.json`
- `results/ABL7.json`
- `results/BASE9_attn.json`
- `results/OOD10_failure.json`
- `results/TRANS11.json`
- `results/E_sssp_512.json`
- `results/EXT12_bot.json`

## A-Line Diagnostic Metrics

Run default single-edit diagnostics:

```powershell
& $env:NAR_PYTHON diagnose_a_metrics.py --tasks sssp --models DeltaNAR no-copy Naive-d Recomp+gate Recompute --seeds 0 1 2 --sizes 16 64 128 --out results/A_local_diagnostics_sssp_3seed.json --markdown RESULTS_AUDIT_sssp_3seed.md
```

Run harder multi-edit diagnostics:

```powershell
& $env:NAR_PYTHON diagnose_a_metrics.py --tasks sssp --models DeltaNAR no-copy Naive-d Recomp+gate Recompute --seeds 0 1 2 --sizes 16 64 128 --train_delta_size 8 --test_delta_size 8 --out results/A_local_diagnostics_sssp_k8_3seed.json --markdown RESULTS_AUDIT_sssp_k8_3seed.md
```

Run oracle-mask diagnostics:

```powershell
& $env:NAR_PYTHON diagnose_a_oracle.py --tasks sssp --seeds 0 --sizes 16 64 --out results/A_oracle_diagnostics_sssp_seed0.json --markdown RESULTS_A_ORACLE_sssp.md
```

## Theory Validation A-Suite

Run:

```powershell
& $env:NAR_PYTHON paper\scripts\run_a_suite.py
```

This produces `results/a_suite.json` and supports the theory-validation section.

## Figures

Run:

```powershell
& $env:NAR_PYTHON paper\scripts\make_figures.py
```

Figures are written to `paper/figures/`.

## Paper Build

From `paper/`:

```powershell
pdflatex main
bibtex main
pdflatex main
pdflatex main

pdflatex supplement
bibtex supplement
pdflatex supplement
pdflatex supplement
```

Expected artifacts:

- `paper/main.pdf`
- `paper/supplement.pdf`
- `paper/full_paper.pdf` if using the combined wrapper.

## Artifact Verification and Packaging

Run the local artifact verifier:

```powershell
& $env:NAR_PYTHON verify_goal_artifacts.py
```

Expected output:

- `GOAL_ARTIFACT_VERIFY.json`
- `GOAL_PROGRESS_AUDIT.md`
- terminal status `PASS`

Build a clean internal review/reproducibility bundle:

```powershell
& $env:NAR_PYTHON package_submission_artifacts.py
```

This writes the archive name configured in the packaging script and includes a
`SUBMISSION_MANIFEST.tsv`. The AAAI upload bundle is curated separately to
exclude local runbooks, temporary files, old archives, and caches.

## Anonymous Artifact Checklist

Before submission:

- Remove author names and affiliations.
- Do not include local server usernames or private paths in paper-visible files.
- Keep raw result JSONs and logs needed for tables.
- Include `README.md`, `REPRODUCIBILITY.md`, `RESULTS.md`, `THEORY_FINAL.md`,
  and diagnostic scripts.
- Make clear which local diagnostic numbers are pilot results and which server
  numbers are final paper results.
