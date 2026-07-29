# Artifact Manifest

Date: 2026-06-02

Latest update: 2026-06-16

## Paper Files

- `paper/main.tex`: main paper.
- `paper/supplement.tex`: supplementary material.
- `paper/references.bib`: bibliography.
- `paper/figures/*.pdf`: generated figures.
- `paper/PDF_PACKAGING.md`: notes on main/supplement/combined PDFs.

## Code Files

- `src/graphs.py`
- `src/sssp.py`
- `src/dynamic_sssp.py`
- `src/mst.py`
- `src/dataset.py`
- `src/dataset_bottleneck.py`
- `src/dataset_mst.py`
- `src/model.py`
- `src/train.py`
- `src/train_bottleneck.py`
- `src/train_mst.py`
- `src/train_generalist.py`
- `src/transfer.py`
- `experiments.py`
- `diagnose_repair_loss.py`: repair/copy-safe gate diagnostic runner.
- `analyze_c_results.py`: summarizes C-line weight-OOD results.
- `analyze_d_results.py`: parses D-line logs into structured JSON/Markdown.
- `analyze_c2_results.py`: merges split C2 copy-safe gate server results.
- `analyze_c43_results.py`: merges and summarizes C43 self-correction results.
- `verify_goal_artifacts.py`: checks the current A/B/C/D artifact evidence set.
- `write_goal_scope_status.py`: writes the local-complete versus external-pending goal matrix.
- `write_completion_audit.py`: maps each active-goal requirement to concrete evidence files.
- `package_submission_artifacts.py`: builds a clean reproducibility bundle.
- `check_artifact_hygiene.py`: scans the clean bundle for private paths and temporary files.

## Diagnostic / Analysis Scripts

- `diagnose_a_metrics.py`
- `diagnose_a_oracle.py`
- `analyze.py`
- `analyze2.py`
- `analyze3.py`
- `analyze4.py`
- `exp_a1.py`
- `exp_a2.py`
- `exp_a4.py`
- `exp_a5.py`
- `exp_a6.py`
- `paper/scripts/run_a_suite.py`
- `paper/scripts/make_figures.py`
- `analyze_b_results.py`
- `analyze_ahr_results.py`
- `analyze_copy_safe_gate.py`
- `analyze_repair_trials.py`

## Result Records

Primary:

- `RESULTS.md`
- `AAAI27_READINESS_AUDIT.md`
- `COMPLETION_AUDIT.md`
- `COMPLETION_AUDIT.json`
- `A_LOCAL_DIAGNOSTIC_SUMMARY.md`
- `BASELINE_CONFIG_AUDIT.md`
- `THEORY_FINAL.md`
- `RELATED_WORK_AUDIT.md`
- `NOVELTY_CLAIMS.md`
- `B_RESULTS_SERVER_SUMMARY.md`
- `C2_COPYSAFE_SERVER_SUMMARY.md`
- `C43_SELF_CORRECTION_SERVER_SUMMARY.md`
- `C43_INTEGRATION_SUMMARY.md`
- `A_HARD_REPAIR_AUDIT.md`
- `COPY_SAFE_GATE_EXPERIMENTS.md`
- `REPAIR_LOSS_EXPERIMENTS.md`
- `C_RESULTS_SUMMARY.md`
- `D_RESULTS_SUMMARY.md`
- `GOAL_PROGRESS_AUDIT.md`
- `GOAL_SCOPE_STATUS.md`
- `GOAL_SCOPE_STATUS.json`
- `GOAL_ARTIFACT_VERIFY.json`
- `ARTIFACT_HYGIENE_REPORT.json`
- `NAR_submission_artifact_20260616.tar.gz`

JSON/log artifacts:

- `results/SEED14_sssp.json`
- `results/SEED14_bot.json`
- `results/A_sssp.json`
- `results/A_bot.json`
- `results/A_sssp_triplet.json`
- `results/A_bot_triplet.json`
- `results/ABL7.json`
- `results/BASE9_attn.json`
- `results/OOD10_failure.json`
- `results/TRANS11.json`
- `results/E_sssp_512.json`
- `results/EXT12_bot.json`
- `results/a_suite.json`
- `results/A_local_diagnostics_sssp_3seed.json`
- `results/A_local_diagnostics_sssp_k8_3seed.json`
- `results/A_local_diagnostics_bottleneck_3seed.json`
- `results/A_oracle_diagnostics_sssp_seed0.json`
- `results/B01_*.json` through `results/B13_*.json`
- `results/C_weight_ood_summary.json`
- `results/C2_copysafe_20seed_merged.json`
- `results/C2_copysafe_gpu2_seed0_9.json`
- `results/C2_copysafe_gpu3_seed10_19.json`
- `results/C43_self_correction_20seed_merged.json`
- `results/C43_self_correction_gpu2_seed0_9.json`
- `results/C43_self_correction_gpu3_seed10_19.json`
- `results/AHR_*.json`
- `results/D_structured.json`
- `paper/tables/c_weight_ood_numbers.tex`
- `paper/tables/c2_copy_safe_numbers.tex`
- `paper/tables/d_structured_numbers.tex`

Server run scripts and packages:

- `server_scripts/B_SERVER_RUNBOOK.md`
- `server_scripts/run_B_*`
- `server_scripts/C2_SERVER_RUNBOOK.md`
- `server_scripts/run_C2_gpu2_copysafe.sh`
- `server_scripts/run_C2_gpu3_copysafe.sh`
- `server_scripts/C43_SELF_CORRECTION_RUNBOOK.md`
- `server_scripts/run_C43_gpu2_self_correction.sh`
- `server_scripts/run_C43_gpu3_self_correction.sh`
- `NAR_C2_copysafe_hotpatch_20260606.tar.gz`
- `NAR_C43_self_correction_hotpatch_20260614.tar.gz`
- `NAR_C43_self_correction_results_20260616_115903.tar.gz`
- `NAR_C2_results_20260609_193743.tar.gz` is the raw returned server package
  kept in the workspace; the clean submission package includes unpacked C2
  artifacts rather than this tarball.

## Tests

- `tests/test_dynamic_sssp.py`
- `tests/test_bottleneck.py`
- `tests/test_dataset.py`
- `tests/test_mst.py`
- `tests/test_mst_dataset.py`

## Submission Packaging Notes

Include:

- source code;
- tests;
- result JSONs needed to regenerate tables;
- paper source and figures;
- `README.md`;
- `REPRODUCIBILITY.md`;
- `ARTIFACT_MANIFEST.md`.

Exclude or clean:

- local `__pycache__` directories;
- temporary smoke-test artifacts whose names start with `_A_` or `_c2_`;
- author-identifying paths, hostnames, or usernames;
- obsolete tarballs unless they are intentionally part of the result archive.
