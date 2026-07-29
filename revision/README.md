# Confirmatory Revision Artifacts

This directory contains the confirmatory experiments added in July 2026.
All paths are relative and all artifacts are anonymous.

## Contents

- `code/dear_delta_train.py`: trains the official-architecture DEAR adaptation
  on matched post-edit SSSP graphs.
- `code/dear_delta_eval.py`: evaluates DEAR with tie-tolerant witness validity.
- `code/DEAR_official/`: the exact Apache-2.0-licensed DEAR source tree used by
  the adapters, excluding checkpoints, datasets, logs, version-control metadata,
  and run-specific files.
- `code/diagnose_conditional_hard_cases.py`: samples graphs with at least one
  invalid previous predecessor and reports aggregate and regional metrics.
- `results/dear_delta_matched_eval_seed*.json`: five DEAR evaluation seeds.
- `results/conditional_hard_{sssp,bottleneck}_10seed.json`: conditional
  invalid-copy slices.
- `results/stability_sssp_seeds20_39.json`: the second 20-seed stability batch.
- `results/day2_sssp_DeltaNAR_s*.json`: the first 20-seed stability batch.
- `results/day23_merged_bootstrap_10000.json`: paired bootstrap analysis.
- `METRIC_CORRECTION.md`: audit note for the pure-copy critical-slice guard.

The original 20-seed main grid remains the source for the primary table.
The stability batches are reported separately because they were run in a
refreshed environment with factorized/cached evaluation and should not be
silently pooled with the prerevision artifact.

The submission audit also corrected the empty-slice guard used for
`pure_copy_critical_acc_v`: the vacuous value 1 is now returned only when the
slice has zero elements, not when a nonempty Boolean slice is all false.  This
changes that diagnostic from 1 to 0 on the conditioned hard slices, as required
by the definition of a critical node.  No trained-model, graph-level, changed-
node, bootstrap, DEAR, or stability result is affected.

## Core Protocols

- Main and stability DeltaNAR: train at `n=16`, 500 training episodes,
  50 epochs, hidden width 128, learning rate 0.001.
- Conditional hard slice: ten training seeds and 100 accepted graphs per
  seed/size; every accepted graph contains at least one invalid old
  predecessor.
- DEAR: five seeds, 100 epochs, 10,000 matched training graphs per seed; the
  model sees the post-edit graph and source, but not the previous solution or
  edit metadata.
- Bootstrap: 10,000 paired resamples over 20 training seeds, percentile 95%
  intervals, and Benjamini--Hochberg correction over the headline family.

## Representative Commands

Run the hard slice from the packaged `DeltaNAR` root:

```bash
python revision/code/diagnose_conditional_hard_cases.py \
  --tasks sssp --seeds 0 1 2 3 4 5 6 7 8 9 \
  --sizes 64 128 256 --target-samples 100 --min-critical 1 \
  --out revision/results/conditional_hard_sssp_10seed.json \
  --markdown revision/results/conditional_hard_sssp_10seed.md
```

Run DEAR training from `revision/code/DEAR_official` so its original relative
imports resolve:

```bash
python ../dear_delta_train.py \
  --seed 0 --config configs/deq.yaml --delta-code-root ../../.. \
  --train-samples 10000 --val-samples 100 --test-samples 100 \
  --output ../../results/dear_delta_matched_train_seed0.json
```

After training produces the corresponding checkpoint:

```bash
python ../dear_delta_eval.py \
  --checkpoint serialised_models/best_dear_delta_matched_bellman_ford_seed0.ckpt \
  --delta-code-root ../../.. --sizes 16 64 128 256 \
  --episodes 100 --batch-size 1 \
  --output ../../results/dear_delta_matched_eval_seed0.json
```
