# Metric audit correction

During the submission audit, we found that the helper used by
`pure_copy_critical_acc_v` tested `x.any()` rather than `x.numel()` before
taking a mean.  Consequently, a nonempty all-false Boolean slice was assigned
the vacuous empty-slice value 1.0.

The guard now returns 1.0 only for an actually empty slice.  On the conditional
hard set, every accepted graph contains at least one critical node, defined as
a node whose old predecessor is invalid after the edit.  Pure copying must
therefore have zero accuracy on this slice, and the sanitized conditional-hard
artifacts record 0.0.

This correction affects only `pure_copy_critical_acc_v`.  It does not affect
DeltaNAR, Naive-d, or Recompute results; any graph-level, node-level,
changed-node, or trained-model critical accuracy; the paired bootstrap
analysis; the DEAR comparison; or the stability rerun.
