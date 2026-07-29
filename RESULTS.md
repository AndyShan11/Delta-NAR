# DeltaNAR / CLRS-Delta Results

This file is the honest full result record as of 2026-06-06. The primary metric
is `node_acc_v`, the tie-tolerant node-level validity metric. Whole-graph
`graph_acc_v` is reported as a hard secondary metric because it amplifies small
node errors and has high seed variance at large `n`.

Training size is `n=16` unless noted. Values are mean +/- population std over
seeds, as stored in `results/*.json` or parsed from `logs/*.log`.

## Sources Checked

- Server B-line return package: `NAR_B_results_20260606_112149.tar.gz`, parsed
  into `B_RESULTS_SERVER_SUMMARY.md` and `paper/tables/b_server_numbers.tex`.
  It contains `results/B01_*.json` through `results/B13_*.json`.
- Local hard-repair A-line audit: `results/AHR_sssp_train4_test12_3seed.json`
  and `results/AHR_oracle_sssp_train4_test12_3seed.json`, summarized in
  `A_HARD_REPAIR_AUDIT.md` and `paper/tables/a_hard_repair_numbers.tex`.
- Exploratory repair-loss probes: `REPAIR_LOSS_EXPERIMENTS.md`. These are
  negative local design diagnostics, not promoted paper results.
- Independent copy-safe gate probes: `COPY_SAFE_GATE_EXPERIMENTS.md`, followed
  by the C2 20-seed server rerun in `C2_COPYSAFE_SERVER_SUMMARY.md`. This is now
  the strongest model-improvement evidence for the hard-repair setting.
- Main 10-seed DeltaNAR: `results/SEED14_sssp.json`, `results/SEED14_bot.json`.
- Main 5-seed baselines: `results/A_sssp.json`, `results/A_sssp_triplet.json`,
  `results/A_bot.json`, `results/A_bot_triplet.json`.
- Ablations and baselines: `results/ABL7.json`, `results/BASE9_attn.json`.
- OOD: `results/B_*`, `results/C_*`, `results/OOD10_failure.json`.
  The C-line weight-OOD subset is also structured in
  `results/C_weight_ood_summary.json` and summarized in
  `C_RESULTS_SUMMARY.md`.
- Extreme size: `results/E_sssp_512.json`, `results/E_bot_512.json`,
  `results/EXT12_sssp.json`, `results/EXT12_bot.json`.
- Transfer: `results/TRANS11.json`.
- Generalist, rollout, intensity, transfer, MST: structured in
  `results/D_structured.json` and summarized in `D_RESULTS_SUMMARY.md`, parsed
  from `logs/D_intensity_s*.log`, `logs/D_mst_s*.log`, `results/TRANS11.json`,
  and the returned server `GEN8_s*.log` / `ROLL13_s*.log` records.
- Local theory-validation A-suite: `results/a_suite.json`, with full stdout in
  `logs/a_suite.log`.

## 2026-06-06 Server B-Line Update

The server return finished both balanced CUDA chains. Log scan found no
`Traceback`, `RuntimeError`, CUDA OOM, `Killed`, `Exception`, or NaN markers.
The updated paper now uses the B-line server grid for the headline size-OOD
figure and 20-seed main table.

Main 20-seed DeltaNAR summary:

| Task | n=64 node | n=256 node | n=256 graph |
|---|---:|---:|---:|
| SSSP | 0.997 +/- 0.002 | 0.987 +/- 0.028 | 0.774 +/- 0.364 |
| Bottleneck | 0.950 +/- 0.047 | 0.899 +/- 0.133 | 0.014 +/- 0.018 |

Matched B-line baseline collapse:

| Setting | DeltaNAR | Recompute | Recomp+gate | Prev+Delta | Triplet |
|---|---:|---:|---:|---:|---:|
| SSSP n=256 | 0.987 | 0.053 | 0.236 | 0.382 | 0.520 at n=64 |
| Bottleneck n=256 | 0.852 | 0.138 | 0.132 | 0.339 | 0.517 at n=64 |

New paper figures generated from the B-line package:

- `paper/figures/fig_b_server_overview.pdf`
- `paper/figures/fig_b_diagnostic_heatmap.pdf`
- `paper/figures/fig_b_oracle_copy.pdf`
- `paper/figures/fig_b_ood_extreme.pdf`

The key caveat is also now explicit: full-node accuracy must be defended with
changed-output and copy-critical slices because single-edit incremental
evaluation is copy-dominated. The B-line diagnostics support the copy mechanism,
but changed/copy-critical slices should be read as targeted metric audits rather
than a standalone ranking of repair ability.

## 2026-06-06 Local Hard-Repair A-Line Audit

This local GPU audit trains SSSP with four simultaneous edits and evaluates on
twelve simultaneous edits over three seeds. It is intentionally harder than the
default single-edit setting and is used as a repair stress test, not as a main
headline result.

At `n=64`, SSSP DeltaNAR remains best on full-node and graph metrics:

| Model | node | graph | changed | copy-critical |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.974 +/- 0.004 | 0.520 +/- 0.004 | 0.376 +/- 0.029 | 0.346 +/- 0.034 |
| no-copy | 0.334 +/- 0.048 | 0.000 +/- 0.000 | 0.250 +/- 0.040 | 0.291 +/- 0.030 |
| Prev+Delta | 0.626 +/- 0.034 | 0.000 +/- 0.000 | 0.338 +/- 0.016 | 0.299 +/- 0.007 |
| Recompute | 0.141 +/- 0.010 | 0.000 +/- 0.000 | 0.193 +/- 0.024 | 0.268 +/- 0.047 |

Bottleneck is much harder in this stress setting:

| Model | node | graph | changed | copy-critical |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.475 +/- 0.182 | 0.000 +/- 0.000 | 0.305 +/- 0.080 | 0.436 +/- 0.047 |
| no-copy | 0.330 +/- 0.026 | 0.000 +/- 0.000 | 0.217 +/- 0.016 | 0.430 +/- 0.094 |
| Prev+Delta | 0.416 +/- 0.115 | 0.000 +/- 0.000 | 0.267 +/- 0.063 | 0.482 +/- 0.046 |
| Recompute | 0.197 +/- 0.027 | 0.000 +/- 0.000 | 0.224 +/- 0.016 | 0.369 +/- 0.067 |

The oracle decomposition shows the remaining headroom:

| Variant | node | graph | changed |
|---|---:|---:|---:|
| raw no-copy | 0.208 +/- 0.010 | 0.000 +/- 0.000 | 0.286 +/- 0.024 |
| predicted copy | 0.974 +/- 0.004 | 0.520 +/- 0.004 | 0.376 +/- 0.029 |
| oracle affected copy | 0.977 +/- 0.001 | 0.514 +/- 0.005 | 0.348 +/- 0.030 |
| oracle copy-safe upper | 0.984 +/- 0.001 | 0.557 +/- 0.008 | 0.509 +/- 0.015 |
| pure copy | 0.980 +/- 0.000 | 0.525 +/- 0.000 | 0.441 +/- 0.000 |

Bottleneck oracle decomposition:

| Variant | node | graph | changed |
|---|---:|---:|---:|
| raw no-copy | 0.252 +/- 0.029 | 0.000 +/- 0.000 | 0.222 +/- 0.040 |
| predicted copy | 0.475 +/- 0.182 | 0.000 +/- 0.000 | 0.305 +/- 0.080 |
| oracle affected copy | 0.743 +/- 0.011 | 0.000 +/- 0.000 | 0.222 +/- 0.040 |
| oracle copy-safe upper | 0.988 +/- 0.001 | 0.587 +/- 0.045 | 0.734 +/- 0.009 |
| pure copy | 0.987 +/- 0.000 | 0.595 +/- 0.000 | 0.695 +/- 0.000 |

Matched-hard training check, DeltaNAR only:

| Task | Setting | node | graph | changed | copy-critical | gate recall |
|---|---|---:|---:|---:|---:|---:|
| SSSP | train4/test12 | 0.974 +/- 0.004 | 0.520 +/- 0.004 | 0.376 +/- 0.029 | 0.346 +/- 0.034 | 0.873 +/- 0.096 |
| SSSP | train12/test12 | 0.966 +/- 0.011 | 0.438 +/- 0.094 | 0.360 +/- 0.021 | 0.273 +/- 0.030 | 0.937 +/- 0.069 |
| Bottleneck | train4/test12 | 0.475 +/- 0.182 | 0.000 +/- 0.000 | 0.305 +/- 0.080 | 0.436 +/- 0.047 | 0.984 +/- 0.013 |
| Bottleneck | train12/test12 | 0.327 +/- 0.054 | 0.000 +/- 0.000 | 0.267 +/- 0.057 | 0.437 +/- 0.034 | 1.000 +/- 0.000 |

Interpretation: this improves the honesty of the paper. DeltaNAR is still the
best complete update model on SSSP, while bottleneck is the sharper failure
case. Hard-repair changed-output accuracy is not yet a strong standalone result, and
training directly on twelve-edit updates does not fix it. The next high-value
experiment is architectural: improve copy-safety calibration and the local
repair branch, then rerun this AHR audit.

Exploratory repair-loss probes support the same conclusion. On SSSP seed 0 at
`n=64`, a repair-focused CE term with `repair_weight=1` and
`critical_weight=0.5` improves changed-output accuracy from 0.368 to 0.441, but
drops graph accuracy from 0.517 to 0.233. A milder `repair_weight=0.25` still
drops graph accuracy to 0.362, and `copy_scale=10` does not improve changed
accuracy. A more conservative `copy_margin=1` improves the 3-seed changed
metric from 0.376 to 0.402 and copy-critical from 0.346 to 0.387, but graph
accuracy drops from 0.520 to 0.479 and gate recall becomes unstable
(`0.610 +/- 0.422`). Simple scalar loss reweighting or scalar copy calibration
is therefore not enough; the next model change should preserve safe copy while
adding a more structured local repair path.

An independent experimental copy-safe gate is a better direction and now has a
20-seed server validation. In the SSSP AHR 3-seed run, it improves changed-output
accuracy from 0.376 to 0.477, with stable gate recall `0.968 +/- 0.030`, but
node accuracy drops from 0.974 to 0.967 and graph accuracy from 0.520 to 0.417.
The C2 full-budget 20-seed rerun keeps SSSP node accuracy stable at
`0.966 +/- 0.027` at `n=64` and raises changed-output accuracy to
`0.686 +/- 0.032` and copy-critical accuracy to `0.722 +/- 0.025`.

On bottleneck AHR, the same copy-safe gate is much more promising: at `n=64`,
node improves from `0.475 +/- 0.182` to `0.782 +/- 0.227`, graph from
`0.000 +/- 0.000` to `0.156 +/- 0.206`, changed-output from `0.305 +/- 0.080` to
`0.514 +/- 0.163`, and copy-critical from `0.436 +/- 0.047` to
`0.469 +/- 0.073`. The C2 20-seed rerun is substantially stronger at `n=64`:
node `0.916 +/- 0.092`, graph `0.271 +/- 0.184`, changed-output
`0.663 +/- 0.053`, and copy-critical `0.746 +/- 0.039`. This makes copy-safety
calibration the clearest architectural improvement to promote for the
hard-repair story.

## 2026-06-14 C34/C35/C38/C42 Stress and Robustness Update

The C34/C35 matched hard-stress rerun strengthens the copy-safe conclusion.
C34 uses copy-safe repair; C35 uses the same repair loss and budget without the
independent copy-safe gate. At `n=512` in hard size extrapolation, copy-safe
reaches SSSP node validity `0.917 +/- 0.186` versus repair-only
`0.663 +/- 0.421`, and bottleneck node validity `0.811 +/- 0.295` versus
`0.599 +/- 0.336`. Bottleneck graph validity also improves from
`0.016 +/- 0.068` to `0.256 +/- 0.303`. This supports copy-safe calibration as
part of the method rather than a repair-loss side effect.

The wide-weight OOD C34/C35 stress remains a limitation in absolute terms.
At `n=256`, SSSP node validity is `0.348 +/- 0.331` for copy-safe versus
`0.151 +/- 0.232` for repair-only; bottleneck is `0.619 +/- 0.327` versus
`0.210 +/- 0.292`. The correct paper framing is therefore: copy-safe is more
robust under numeric shift, but wide weights are not solved.

C38 profiles the dense implementation. At `n=512`, peak allocated CUDA memory is
about 1.84GB for both copy-safe and repair-only, so the tested sizes are memory
feasible on 11GB RTX 2080 Ti cards. Runtime is the problem: evaluating 50
examples takes about 1023 seconds for SSSP and 987 seconds for bottleneck at
`n=512`.

C42 audits the exact-previous-solution assumption by corrupting the previous
predecessor and previous scalar value only at evaluation time. Under 10% noisy
previous-solution corruption at `n=128`, copy-safe keeps higher node validity
than repair-only on both SSSP (`0.810 +/- 0.125` versus `0.669 +/- 0.261`) and
bottleneck (`0.746 +/- 0.163` versus `0.539 +/- 0.233`). The decline is smooth
from 0% to 10% corruption, and affected-gate recall remains high, so the main
failure is not localization. Whole-graph validity, however, collapses under
corruption. This clarifies the empirical scope of the paper, but it
does not solve long-horizon self-correction.

## 2026-06-16 C43 Self-Correction Training Audit

C43 tests whether training with imperfect previous-solution memory improves
over the clean copy-safe model. The matched `self_correct` variant corrupts 5%
of the previous predecessor/value memory during training, while both variants
use the same copy-safe repair losses, hard multi-edit setting, and evaluation
budget. The returned server package is
`NAR_C43_self_correction_results_20260616_115903.tar.gz`; it merged 80 completed
settings across 20 seeds, two tasks, two variants, two sizes, and four
evaluation corruption rates. Log scans found no `Traceback`, CUDA OOM, `Killed`,
`Exception`, or NaN markers.

The fast completion is not evidence of an abnormal skip. The detailed logs show
that both GPU lanes started from `completed=0` and ran all assigned jobs.
Individual training/evaluation jobs took about 370--606 seconds. Summed across
both lanes, this is roughly 10.5 serial GPU-hours, so a same-evening finish on
two RTX 2080 Ti cards is plausible.

The main positive result is the calibrated copy-safe model's long-horizon
rollout stability. The earlier five-seed rollout of the uncalibrated model
ended at `0.598 +/- 0.307` SSSP node validity at step 59. In C43, clean
copy-safe training reaches:

| Variant | t=0 | t=1 | t=9 | t=19 | t=39 | t=59 |
|---|---:|---:|---:|---:|---:|---:|
| copy-safe | 0.993 +/- 0.010 | 0.981 +/- 0.027 | 0.937 +/- 0.090 | 0.915 +/- 0.104 | 0.888 +/- 0.121 | 0.871 +/- 0.125 |
| self-correct | 0.981 +/- 0.020 | 0.901 +/- 0.195 | 0.854 +/- 0.215 | 0.833 +/- 0.219 | 0.810 +/- 0.225 | 0.792 +/- 0.225 |

The noisy-memory training objective itself is mixed and should not be promoted
as the main method. It improves some 10% noisy-memory node metrics, but hurts
clean graph accuracy and underperforms clean copy-safe training in rollout:

| Task | n | Variant | clean node | 10% noisy node | clean graph | 10% noisy graph |
|---|---:|---|---:|---:|---:|---:|
| SSSP | 64 | copy-safe | 0.966 +/- 0.027 | 0.863 +/- 0.057 | 0.394 +/- 0.180 | 0.004 +/- 0.004 |
| SSSP | 64 | self-correct | 0.957 +/- 0.037 | 0.889 +/- 0.069 | 0.312 +/- 0.198 | 0.013 +/- 0.016 |
| SSSP | 128 | copy-safe | 0.958 +/- 0.068 | 0.809 +/- 0.126 | 0.394 +/- 0.299 | 0.000 +/- 0.000 |
| SSSP | 128 | self-correct | 0.905 +/- 0.212 | 0.826 +/- 0.199 | 0.274 +/- 0.270 | 0.000 +/- 0.000 |
| Bottleneck | 64 | copy-safe | 0.916 +/- 0.092 | 0.791 +/- 0.099 | 0.271 +/- 0.184 | 0.002 +/- 0.003 |
| Bottleneck | 64 | self-correct | 0.897 +/- 0.054 | 0.834 +/- 0.053 | 0.054 +/- 0.099 | 0.001 +/- 0.003 |
| Bottleneck | 128 | copy-safe | 0.896 +/- 0.140 | 0.747 +/- 0.162 | 0.288 +/- 0.272 | 0.000 +/- 0.000 |
| Bottleneck | 128 | self-correct | 0.832 +/- 0.132 | 0.757 +/- 0.133 | 0.012 +/- 0.032 | 0.000 +/- 0.000 |

Interpretation: C43 raises the paper's empirical status because the calibrated
copy-safe model no longer has a weak rollout result. It does not establish a
complete robustness case because the explicit self-correction objective is not yet a win and
whole-graph exactness under noisy memory remains unsolved. The right paper
framing is: copy-safe calibration is the method; naive noisy-memory training is
an honest negative/diagnostic result.

## 2026-06-23 C44 Completed and C45 Guarded-Deep Audit

The later package `NAR_C45_6d_extended_stress_results_20260623_160741.tar.gz`
supersedes the earlier partial C44 snapshot. It contains a completed C44 merge
(`results/C44_focused_stress18h_completed_20seed_merged.json`, 80/80 completed,
0 failures) and a partial C45 merge
(`results/C45_guarded_deep_20seed_merged.json`, 42 completed, 0 failures). The
C45 master logs enter Stage 2 but never print a final done line or Stage 3/C46
line; no C46 files are present. This is best interpreted as a checkpointed C45
snapshot rather than a crashed result. Log scans found no traceback, OOM, killed
process, or NaN signal.

C44 is now a complete fixed-target noisy-memory and unguarded certificate-repair
audit. Fixed-target noisy training is a mixed robustness result, not a clean
self-correction win. On SSSP at `n=256`, it slightly improves clean node validity
(`0.949 +/- 0.113` vs. copy-safe `0.939 +/- 0.126`) and improves 20% noisy-memory
node validity (`0.770 +/- 0.119` vs. `0.670 +/- 0.179`), but clean graph validity
falls (`0.376 +/- 0.325` vs. `0.427 +/- 0.381`) and graph validity remains 0
under 20% noise. On bottleneck at `n=256`, fixed-target noisy training improves
node validity (`0.891` vs. `0.847` clean; `0.752` vs. `0.602` at 20% noise) but
again does not solve graph exactness. Rollout is also not a clear win: at step
79, copy-safe/fixed-noisy are `0.854/0.840` for `n=64` and `0.842/0.847` for
`n=128`.

Unguarded certificate repair remains the strongest negative diagnostic. It fires
on nearly all vertices and catastrophically over-repairs SSSP: at `n=256`, clean
node validity falls from `0.939 +/- 0.126` to `0.042 +/- 0.017`; under 20% noisy
memory it falls from `0.670 +/- 0.179` to `0.037 +/- 0.008`. The same mechanism
is highly beneficial on bottleneck, where at `n=256` clean graph validity rises
from `0.274 +/- 0.278` to `0.751 +/- 0.346`, and under 20% noisy memory from
`0.000` to `0.583 +/- 0.432`. This confirms that local certificates are useful
but task-dependent; they are not safe enough to promote as a universal
self-correction method.

C45 tests a guarded certificate rule (`cert_guard=copy_unsafe`) on a broader
OOD suite with 120-step rollout. The completed part is strong for copy-safe and
copy-safe+guarded-cert over 20 seeds; fixed-target noisy has only two SSSP seeds
and no bottleneck rows, so it remains exploratory. The guarded rule fixes the
most extreme C44 over-repair but does not become a main method. On SSSP at
`n=256`, guarded cert is close on clean memory (`0.921` vs. copy-safe `0.939`)
but hurts at 30% noisy memory (`0.508` vs. `0.598`, changed-output `0.041` vs.
`0.199`). On bottleneck at `n=256`, guarded cert is clearly positive under 30%
noise (`0.730` vs. `0.520` node, `0.596` vs. `0.166` changed-output, `0.577`
vs. `0.115` copy-critical). OOD is similarly mixed: guarded cert helps bottleneck
heavy-tail at `n=256` and 30% noise (`0.553` vs. `0.438` node), is small-positive
on bottleneck wide-weight, and hurts BA and SSSP heavy-tail/BA slices.

Interpretation: C44/C45 strengthens the paper's honesty and robustness thickness
but does not establish a complete robustness case. The correct interpretation is:
copy-safe remains the main method, fixed-target noisy training is a mixed
diagnostic, and certificate repair should be framed as future learned/guarded
self-correction rather than as part of DeltaNAR.

## 2026-06-26 C48 Two-Day Focused Guard-Profile Audit

The two returned packages `NAR_C48_2d_focused_results_20260626_205319.tar.gz`
and `NAR_C48_2d_focused_results_20260626_205402.tar.gz` were both extracted.
The latter contains the merged result
`results/C48_2d_guard_profiles_8seed_merged.json` and
`C48_2D_GUARD_PROFILES_SERVER_SUMMARY.md`. C48 completed all 16 planned
`(train_mode, task, seed)` rows: 8 copy-safe SSSP seeds and 8 copy-safe
bottleneck seeds. Log scans found no traceback, OOM, killed-process, exception,
or NaN signal. The wall-clock time was about 29--30 hours per GPU lane, inside
the two-day budget.

C48 was designed to spend the limited server budget on the highest-value
self-correction question: can a guarded local certificate profile preserve SSSP
while retaining the bottleneck gains seen in C45? The answer is no for these
thresholded profiles. At `n=256` and 30% noisy previous memory, SSSP raw
copy-safe has `0.663 +/- 0.077` node validity and `0.193 +/- 0.098`
changed-output validity. The conservative `cu_m1` profile falls to
`0.611 +/- 0.100` node and `0.055 +/- 0.052` changed-output validity; the
default `cu_0` profile falls further to `0.575 +/- 0.123` and
`0.038 +/- 0.018`. SSSP wide-weight, heavy-tail, and BA slices show the same
direction: guarded repair hurts more than it helps under 30% noisy memory.

The same profiles are strongly positive for bottleneck. At `n=256` and 30%
noisy memory, raw copy-safe obtains `0.457 +/- 0.216` node validity and
`0.154 +/- 0.028` changed-output validity on ER. `cu_m1` improves these to
`0.632 +/- 0.191` and `0.544 +/- 0.260`; `cu_0` improves them to
`0.691 +/- 0.217` and `0.602 +/- 0.288`. The improvement also appears on
bottleneck wide-weight and heavy-tail, while BA is slightly hurt. Averaged over
ER, wide-weight, heavy-tail, and BA at `n=256`/30% noise, `cu_0` has a `-0.080`
SSSP node delta but a `+0.112` bottleneck node delta; changed-output deltas are
`-0.142` for SSSP and `+0.196` for bottleneck.

Rollout also favors raw copy-safe. At step 119, raw copy-safe reaches
`0.896 +/- 0.100` at `n=128` and `0.931 +/- 0.074` at `n=256`; the guarded
profiles reduce these values, with conservative profiles losing about 0.02--0.03
and default-threshold profiles losing about 0.06--0.07.

Interpretation: C48 strengthens the paper by making the self-correction boundary
unambiguous, but it does not establish a complete robustness result. The correct
claim is that local certificate repair contains a real bottleneck signal, but a
hand-thresholded guard is not safe enough for SSSP. Copy-safe remains the main
method; graph-level self-correction remains a future learned, task-aware
correction policy problem.

## 2026-06-28 C49 Task-Aware Certificate Routing Audit

The returned package `NAR_C49_2d_taskaware_results_20260628_201233.tar.gz` was
extracted and analyzed. It contains `results/C49_taskaware_12seed_merged.json`,
lane JSONs, logs, `C49_TASKAWARE_SERVER_SUMMARY.md`, and
`C49_TASKAWARE_POLICY_SUMMARY.md`. The run completed all 24 planned rows:
12 copy-safe SSSP seeds and 12 copy-safe bottleneck seeds. Log scans found no
traceback, CUDA OOM, killed-process, exception, or NaN signal. Each GPU lane ran
for about 26 hours, faster than the conservative two-day estimate.

C49 directly tests the most important remaining self-correction question after
C48: if thresholded certificate repair is harmful for SSSP but helpful for
bottleneck, can a task-aware routing policy keep raw copy-safe for SSSP and use
guarded repair only for bottleneck? The answer is a qualified yes. This is not a
new universal method, but it is a scoped positive diagnostic.

The SSSP side confirms that raw copy-safe should be preserved. At `n=256` and
30% noisy previous memory, raw copy-safe obtains `0.656 +/- 0.076` node validity,
`0.191 +/- 0.100` changed-output validity, and `0.256 +/- 0.223` copy-critical
validity. The conservative `cu_m1` guard falls to `0.606 +/- 0.097`,
`0.053 +/- 0.044`, and `0.111 +/- 0.233`; the default `cu_0` guard falls to
`0.578 +/- 0.113`, `0.040 +/- 0.018`, and `0.043 +/- 0.048`. SSSP wide-weight,
heavy-tail, and BA slices have the same negative direction under 30% noise.

The bottleneck side confirms a real certificate-correction signal. At `n=256`
and 30% noisy previous memory on ER, raw copy-safe obtains `0.526 +/- 0.204`
node validity, `0.164 +/- 0.029` changed-output validity, and
`0.114 +/- 0.037` copy-critical validity. `cu_m1` improves these to
`0.686 +/- 0.177`, `0.571 +/- 0.228`, and `0.556 +/- 0.253`; `cu_0` improves
them further to `0.733 +/- 0.190`, `0.630 +/- 0.249`, and `0.616 +/- 0.270`.
Wide-weight and heavy-tail bottleneck also improve; BA bottleneck is slightly
hurt, so the result is still scoped rather than universal.

The policy-level summary is the most useful paper takeaway. Averaged over ER,
wide-weight, heavy-tail, and BA at `n=256` and 30% noisy memory, the task-aware
policy "raw SSSP plus `cu_0` bottleneck" has zero SSSP penalty by construction
and improves bottleneck by `+0.098` node validity, `+0.192` changed-output
validity, and `+0.192` copy-critical validity. Under 10% noise the corresponding
bottleneck gains are `+0.066`, `+0.191`, and `+0.190`. Rollout still favors raw
copy-safe on SSSP: at step 119 and `n=256`, raw copy-safe reaches
`0.934 +/- 0.062`, while `cu_m1` and `cu_0` lose `0.028` and `0.066`.

Interpretation: C49 is better than expected because it turns the C48 negative
diagnostic into a scoped positive story. It does not solve graph-level
self-correction and should not be promoted as the core method. It does, however,
address the most serious methodological concern: certificate consistency is not merely
a failed idea; it is useful when routed by task. The paper's current status is
substantially stronger than after C48, but still not a complete robustness
result.

Missing or inconsistent sources:

- `logs/EXT12_1024.log` shows an incomplete old run and no
  `results/EXT12_1024.json` is present. Use `EXT12_sssp.json` for the steps=32
  SSSP 1024 numbers, but do not treat those as the SSSP headline because that
  steps cap is unstable.

## Main Size-OOD: DeltaNAR, 10 Seeds

SSSP, min-plus, `results/SEED14_sssp.json`:

| n | node_acc_v | graph_acc_v | aff_f1 |
|---:|---:|---:|---:|
| 16 | 0.990 +/- 0.002 | 0.881 +/- 0.024 | 0.809 +/- 0.143 |
| 32 | 0.995 +/- 0.002 | 0.913 +/- 0.019 | 0.751 +/- 0.180 |
| 64 | 0.997 +/- 0.002 | 0.926 +/- 0.051 | 0.699 +/- 0.224 |
| 128 | 0.994 +/- 0.008 | 0.884 +/- 0.196 | 0.598 +/- 0.312 |
| 256 | 0.987 +/- 0.024 | 0.760 +/- 0.335 | 0.565 +/- 0.370 |

Bottleneck / widest path, max-min, `results/SEED14_bot.json`:

| n | node_acc_v | graph_acc_v | aff_f1 |
|---:|---:|---:|---:|
| 16 | 0.978 +/- 0.007 | 0.760 +/- 0.069 | 0.930 +/- 0.030 |
| 32 | 0.963 +/- 0.014 | 0.389 +/- 0.114 | 0.855 +/- 0.084 |
| 64 | 0.943 +/- 0.054 | 0.111 +/- 0.066 | 0.785 +/- 0.186 |
| 128 | 0.904 +/- 0.133 | 0.025 +/- 0.019 | 0.744 +/- 0.256 |
| 256 | 0.852 +/- 0.163 | 0.014 +/- 0.022 | 0.696 +/- 0.260 |

Takeaway: SSSP node accuracy stays near 0.99 through 16x extrapolation, while
large-`n` graph accuracy has substantial variance. Bottleneck remains far above
from-scratch baselines, but whole-graph exactness collapses as expected from
roughly `node_acc^n`.

## Main Baseline Comparison, 5 Seeds

These are the broader baseline grids; DeltaNAR rows here are the original
5-seed runs, while the previous section supersedes them for the main headline.

SSSP `node_acc_v`:

| Model | n=16 | n=32 | n=64 | n=128 | n=256 |
|---|---:|---:|---:|---:|---:|
| DeltaNAR | 0.988 +/- 0.002 | 0.995 +/- 0.002 | 0.997 +/- 0.002 | 0.998 +/- 0.004 | 0.999 +/- 0.003 |
| Naive-d | 0.963 +/- 0.007 | 0.928 +/- 0.019 | 0.822 +/- 0.041 | 0.623 +/- 0.055 | 0.423 +/- 0.041 |
| Recomp+gate | 0.945 +/- 0.011 | 0.752 +/- 0.031 | 0.482 +/- 0.024 | 0.327 +/- 0.039 | 0.231 +/- 0.065 |
| Triplet | 0.925 +/- 0.041 | 0.781 +/- 0.064 | 0.544 +/- 0.062 | - | - |
| Recompute | 0.495 +/- 0.017 | 0.333 +/- 0.028 | 0.172 +/- 0.015 | 0.084 +/- 0.007 | 0.040 +/- 0.006 |

SSSP `graph_acc_v`:

| Model | n=16 | n=32 | n=64 | n=128 | n=256 |
|---|---:|---:|---:|---:|---:|
| DeltaNAR | 0.868 +/- 0.024 | 0.907 +/- 0.023 | 0.944 +/- 0.013 | 0.968 +/- 0.018 | 0.982 +/- 0.012 |
| Naive-d | 0.618 +/- 0.062 | 0.152 +/- 0.079 | 0.001 +/- 0.002 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Recomp+gate | 0.468 +/- 0.068 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Triplet | 0.400 +/- 0.199 | 0.008 +/- 0.013 | 0.000 +/- 0.000 | - | - |
| Recompute | 0.010 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

Bottleneck `node_acc_v`:

| Model | n=16 | n=32 | n=64 | n=128 | n=256 |
|---|---:|---:|---:|---:|---:|
| DeltaNAR | 0.980 +/- 0.005 | 0.968 +/- 0.013 | 0.961 +/- 0.017 | 0.951 +/- 0.026 | 0.918 +/- 0.047 |
| Naive-d | 0.960 +/- 0.009 | 0.879 +/- 0.054 | 0.678 +/- 0.103 | 0.471 +/- 0.116 | 0.326 +/- 0.096 |
| Recomp+gate | 0.828 +/- 0.194 | 0.513 +/- 0.211 | 0.267 +/- 0.146 | 0.155 +/- 0.065 | 0.126 +/- 0.037 |
| Triplet | 0.934 +/- 0.020 | 0.776 +/- 0.057 | 0.490 +/- 0.076 | - | - |
| Recompute | 0.802 +/- 0.019 | 0.615 +/- 0.025 | 0.351 +/- 0.040 | 0.210 +/- 0.033 | 0.139 +/- 0.009 |

Bottleneck `graph_acc_v`:

| Model | n=16 | n=32 | n=64 | n=128 | n=256 |
|---|---:|---:|---:|---:|---:|
| DeltaNAR | 0.790 +/- 0.051 | 0.462 +/- 0.103 | 0.148 +/- 0.063 | 0.030 +/- 0.017 | 0.007 +/- 0.009 |
| Naive-d | 0.608 +/- 0.071 | 0.081 +/- 0.078 | 0.000 +/- 0.001 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Recomp+gate | 0.280 +/- 0.144 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |
| Triplet | 0.423 +/- 0.106 | 0.003 +/- 0.004 | 0.000 +/- 0.000 | - | - |
| Recompute | 0.095 +/- 0.016 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 |

## Component Ablation

`results/ABL7.json`, `node_acc_v`.

SSSP:

| Model | n=16 | n=32 | n=64 | n=128 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.988 | 0.995 | 0.997 | 0.998 |
| no-copy | 0.978 | 0.888 | 0.628 | 0.375 |
| min-only | 0.988 | 0.994 | 0.998 | 1.000 |
| Naive-d | 0.985 | 0.995 | 0.998 | 0.999 |
| Recompute | 0.494 | 0.333 | 0.172 | 0.084 |

Bottleneck:

| Model | n=16 | n=32 | n=64 | n=128 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.980 | 0.968 | 0.961 | 0.951 |
| no-copy | 0.966 | 0.810 | 0.453 | 0.242 |
| min-only | 0.975 | 0.970 | 0.967 | 0.967 |
| Naive-d | 0.966 | 0.927 | 0.861 | 0.810 |
| Recompute | 0.801 | 0.615 | 0.351 | 0.209 |

Interpretation: removing copy from the gated DeltaNAR path causes size-OOD
collapse. Removing max aggregation does not collapse here. `Naive-d` in current
`experiments.py` is configured as `use_prev=True`, `use_delta=True`,
`use_gate=False`, `use_copy=False`; it behaves differently from the gated
`no-copy` ablation, so paper text should define it carefully.

Attention baseline, `results/BASE9_attn.json`, `node_acc_v`:

| Task | n=16 | n=32 | n=64 | n=128 |
|---|---:|---:|---:|---:|
| SSSP Attn | 0.861 | 0.699 | 0.467 | 0.286 |
| Bottleneck Attn | 0.791 | 0.622 | 0.405 | 0.238 |

## OOD Breadth

Topology OOD at `n=64`, train ER, metric `node_acc_v`:

| Task / Model | ER | BA | Grid |
|---|---:|---:|---:|
| SSSP DeltaNAR | 0.997 +/- 0.002 | 0.993 +/- 0.004 | 0.934 +/- 0.095 |
| SSSP Triplet | - | 0.960 +/- 0.006 | 0.691 +/- 0.037 |
| SSSP Recompute | 0.172 +/- 0.015 | 0.752 +/- 0.061 | 0.439 +/- 0.006 |
| Bottleneck DeltaNAR | 0.961 +/- 0.017 | 0.993 +/- 0.003 | 0.985 +/- 0.005 |
| Bottleneck Triplet | - | 0.969 +/- 0.006 | 0.805 +/- 0.014 |
| Bottleneck Recompute | 0.351 +/- 0.040 | 0.873 +/- 0.019 | 0.917 +/- 0.008 |

Weight OOD at `n=64`, SSSP, metric `node_acc_v`:

| Model | normal w<=10 | wide w<=100 | heavy-tail |
|---|---:|---:|---:|
| DeltaNAR | 0.997 +/- 0.002 | 0.843 +/- 0.088 | 0.972 +/- 0.028 |
| Recompute | 0.172 +/- 0.015 | 0.118 +/- 0.040 | 0.133 +/- 0.002 |

Failure OOD, test `edge_delete`, metric `node_acc_v`:

| Task / Model | n=16 | n=64 |
|---|---:|---:|
| SSSP DeltaNAR | 0.986 | 0.998 |
| SSSP Recompute | 0.507 | 0.173 |
| Bottleneck DeltaNAR | 0.982 | 0.977 |
| Bottleneck Recompute | 0.801 | 0.350 |

Dynamic-intensity OOD at `n=16`, train delta-size 1, test larger delta sizes,
metric `node_acc_v`, parsed from `logs/D_intensity_s*.log`:

| Model | k=1 | k=2 | k=4 | k=8 |
|---|---:|---:|---:|---:|
| DeltaNAR | 0.991 +/- 0.001 | 0.984 +/- 0.003 | 0.967 +/- 0.006 | 0.926 +/- 0.011 |
| Naive-d | 0.968 +/- 0.009 | 0.955 +/- 0.007 | 0.924 +/- 0.007 | 0.874 +/- 0.005 |
| Recompute | 0.513 +/- 0.022 | 0.506 +/- 0.023 | 0.499 +/- 0.026 | 0.499 +/- 0.023 |

The supplement's paper-facing dynamic-intensity table uses the stronger server
B07 matched `k=8` rerun for the `k=8` column; the structured source is
`results/D_structured.json` and the LaTeX macro is
`paper/tables/d_structured_numbers.tex`.

Delta-type OOD is reported in `THEORY.md` / `HANDOFF.md` from `exp_a6.py`:
weight-change train to structural add/delete test gives DeltaNAR 0.998 -> 0.998
and Recompute 0.17. No JSON/log artifact was found.

## Extreme Size

Reliable SSSP extreme-size result with `steps=n`, `results/E_sssp_512.json`:

| Task | Model | n | node_acc_v | graph_acc_v |
|---|---|---:|---:|---:|
| SSSP | DeltaNAR | 512 | 0.998 +/- 0.002 | 0.848 +/- 0.215 |
| SSSP | Recompute | 512 | 0.029 +/- 0.001 | 0.000 +/- 0.000 |

Bottleneck 512 from `results/E_bot_512.json` and 1024 from
`results/EXT12_bot.json`:

| Task | Model | n | node_acc_v | graph_acc_v |
|---|---|---:|---:|---:|
| Bottleneck | DeltaNAR | 512 | 0.937 +/- 0.004 | 0.006 |
| Bottleneck | Recompute | 512 | 0.123 +/- 0.006 | 0.000 |
| Bottleneck | DeltaNAR | 1024 | 0.956 +/- 0.047 | 0.013 |
| Bottleneck | Recompute | 1024 | 0.107 +/- 0.004 | 0.000 |

SSSP with `steps=32`, `results/EXT12_sssp.json`, is unstable and should be used
only as a caveat / steps-cap diagnostic:

| Task | Model | n | node_acc_v | graph_acc_v |
|---|---|---:|---:|---:|
| SSSP steps=32 | DeltaNAR | 512 | 0.675 +/- 0.460 | 0.667 |
| SSSP steps=32 | DeltaNAR | 1024 | 0.674 +/- 0.461 | 0.666 |
| SSSP steps=32 | Recompute | 512 | 0.028 +/- 0.000 | 0.000 |
| SSSP steps=32 | Recompute | 1024 | 0.027 +/- 0.000 | 0.000 |

## Temporal Rollout

60-step rollout at `n=64`, model feeds its own predictions forward, parsed from
`logs/ROLL13_s*.log`. This is seed-dependent but DeltaNAR remains much better
than the no-copy baseline.

| Model | t0 | t9 | t59 |
|---|---:|---:|---:|
| DeltaNAR | 0.971 +/- 0.026 | 0.729 +/- 0.232 | 0.598 +/- 0.307 |
| Naive-d | 0.843 +/- 0.045 | 0.101 +/- 0.012 | 0.097 +/- 0.009 |

Representative seeds:

| Seed | DeltaNAR t0 -> t59 | Naive-d t0 -> t59 |
|---:|---:|---:|
| 0 | 0.938 -> 0.596 | 0.858 -> 0.099 |
| 2 | 0.999 -> 0.918 | 0.847 -> 0.113 |

## Generalist

One processor, SSSP + bottleneck, parsed from `logs/GEN8_s*.log`,
metric `node_acc_v`:

| Task | n=16 | n=32 | n=64 | n=128 |
|---|---:|---:|---:|---:|
| SSSP | 0.978 +/- 0.015 | 0.986 +/- 0.014 | 0.993 +/- 0.010 | 0.997 +/- 0.005 |
| Bottleneck | 0.956 +/- 0.023 | 0.955 +/- 0.013 | 0.961 +/- 0.023 | 0.965 +/- 0.035 |

## Cross-Task Transfer

Train SSSP, evaluate bottleneck, `results/TRANS11.json`, metric `node_acc_v`:

| Seed | zero-shot n=16 | zero-shot n=64 | few-shot n=16 | few-shot n=64 |
|---:|---:|---:|---:|---:|
| 0 | 0.836 | 0.413 | 0.974 | 0.991 |
| 1 | 0.779 | 0.580 | 0.973 | 0.993 |
| 2 | 0.550 | 0.444 | 0.970 | 0.990 |

## MST Difficulty Gradient

MST is not a headline success; it is evidence of a harder task / whole-graph
difficulty gradient. Parsed from `logs/D_mst_s*.log`.

Graph accuracy:

| Model | n=16 | n=32 | n=64 |
|---|---:|---:|---:|
| DeltaNAR | 0.278 +/- 0.164 | 0.125 +/- 0.056 | 0.000 +/- 0.000 |
| Naive-d | 0.244 +/- 0.124 | 0.034 +/- 0.031 | 0.000 +/- 0.000 |
| Recompute | 0.051 +/- 0.031 | 0.003 +/- 0.003 | 0.000 +/- 0.000 |

Edge accuracy:

| Model | n=16 | n=32 | n=64 |
|---|---:|---:|---:|
| DeltaNAR | 0.983 +/- 0.004 | 0.989 +/- 0.005 | 0.938 +/- 0.056 |
| Naive-d | 0.976 +/- 0.013 | 0.961 +/- 0.038 | 0.959 +/- 0.032 |
| Recompute | 0.923 +/- 0.065 | 0.960 +/- 0.020 | 0.962 +/- 0.004 |

## Local Theory-Validation A-Suite

These numbers were rerun locally and archived in `results/a_suite.json`; full
stdout is in `logs/a_suite.log`.

| Experiment | Result |
|---|---|
| A1 affected-region size | ER p=0.3: mean `|A_t|` goes from 0.52 at n=16 to 0.01 at n=256 (`|A_t|/n` from 3.2% to ~0.0%); fixed average degree about 6 gives 1.9% at n=16 and about 0.1% at n=1024. |
| A2 accuracy vs `|A_t|` | node_acc: 0.998 for `|A|=0`, 0.970 for 1-2, 0.922 for 3-5, 0.882 for 6-10, 0.793 for 11+; gate recall rho=0.90. |
| A4 steps | DeltaNAR reaches 0.989/0.998 at 2 steps (n16/n64) and 0.991/0.999 at 16 steps; Recompute is more step-sensitive and weaker OOD. |
| A5 copy robustness | copy_scale saturates for >=5; copy_margin changes bottleneck node_acc@64 from 0.976 at margin 0 to 0.953-0.957 at margins 2-3. |
| A6 delta-type invariance | DeltaNAR 0.998 -> 0.998 from weight-change train to add/delete test; Recompute about 0.17. |

## Paper-Facing Summary

Use `node_acc_v` as the main metric. The clean headline is: train at `n=16`,
test up to 16x or 64x larger; DeltaNAR stays near-perfect on SSSP and remains
high on bottleneck, while from-scratch and strong triplet/attention baselines
collapse. Report graph accuracy honestly as secondary: SSSP graph accuracy is
strong but high-variance at large `n`; bottleneck and MST whole-graph exactness
collapse because per-node/edge errors compound over the whole structure.
