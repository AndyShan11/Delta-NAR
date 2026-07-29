# DeltaNAR Theory Package

Date: 2026-06-02

This is the cleaned proof package for the AAAI-27 draft. It is intentionally
node-level and conditional, matching the mechanism and metric used in the paper.

## Objects

At edit step `t`, the model receives the post-edit graph `G_t`, the edit
`Delta_t`, and a previous solution `y_{t-1}`. It predicts an updated solution
`hat y_t`.

For path tasks, let

```text
S_t = { reachable non-source vertices at time t }
```

be the evaluation domain used by `node_acc_v`. Let

```text
C_t = { v in S_t : y_t^*(v) != y_{t-1}^*(v) }
```

be the changed-output set. The proof uses a copy-critical region `A_t` satisfying
`C_t subset A_t`, where all vertices outside `A_t` can be safely copied from the
exact previous solution. The implementation's affected work-set may be a superset
of `C_t`; this is safe but gives a looser bound.

## Main Theorem

Assume:

1. Previous solution exactness: `y_{t-1}=y_{t-1}^*` on `S_t`.
2. Copy safety: all vertices in `S_t \ A_t` are correct when copied.
3. Average affected correctness: on `A_t`, the model is correct with average
   probability at least `gamma_t`.
4. False-positive recomputation outside `A_t` causes average error at most
   `eta_t`.

Then

```text
E[node_err_t]
  <= (|A_t| / |S_t|)(1 - gamma_t)
     + (|S_t \ A_t| / |S_t|) eta_t.
```

If `E|A_t| <= a`, `|S_t| >= c n`, `gamma_t >= gamma`, and `eta_t <= eta`, then

```text
E[node_acc_v] >= 1 - (a/(c n))(1 - gamma) - eta.
```

Thus, when edits are output-sensitive and false-positive recomputation is safe
or rare, node error scales with the affected fraction rather than graph size.

## Proof

For `v in S_t \ A_t`, copying is correct by exact previous solution and copy
safety. These vertices can only contribute error if they are falsely sent through
an imperfect recompute branch, which is captured by `eta_t`.

For `v in A_t`, no stronger structural claim is needed. In particular, if `A_t`
is a work-set superset, stale copying is not necessarily wrong for every vertex.
By definition, the average correctness on this region is at least `gamma_t`, so
average error there is at most `1 - gamma_t`.

Averaging over `S_t` gives

```text
E[node_err_t]
  <= (1/|S_t|) sum_{v in A_t} (1 - Pr[correct at v])
     + (1/|S_t|) sum_{v in S_t \ A_t} Pr[false-positive recompute error at v]
  <= (|A_t| / |S_t|)(1 - gamma_t)
     + (|S_t \ A_t| / |S_t|) eta_t.
```

Taking one minus the error bound gives the node-accuracy statement. Taking
expectations over edits and using `E|A_t| <= a` and `|S_t| >= c n` gives the
size-OOD corollary.

## Gate Recall and Recompute Accuracy

Let affected recall be

```text
rho_t = (1/|A_t|) sum_{v in A_t} Pr[hat a_t(v)=1].
```

Let `q_t` be recompute success on flagged affected vertices. If `A_t=C_t` and
the averaged gate/recompute events factorize, then the idealized mechanism gives
`gamma_t = rho_t q_t`. In the actual paper we use `gamma_t` directly because:

- work-set affected labels may be supersets of changed-output labels;
- copying can still be correct for some work-set vertices;
- no independence assumption between gate and recompute success is needed.

## Dependency Depth

For `v in A_t`, define the update dependency depth `d_t(v)` as the minimum number
of semiring relaxation steps needed to certify the new witness from exact copied
boundary values. Let `d_t(A_t)=max_{v in A_t} d_t(v)`.

For SSSP:

```text
D(v) = min_u D(u) + w(u,v).
```

For bottleneck:

```text
B(v) = max_u min(B(u), w(u,v)).
```

If a processor can represent one exact semiring relaxation per message-passing
step, then by induction on `d_t(v)`, all affected vertices with `d_t(v) <= T`
are exactly recomputable by a `T`-step aligned processor in the ideal case.
Learned approximation and training error are absorbed into `epsilon_T`, yielding

```text
q_t >= Pr[d_t(v) <= T | v in A_t, hat a_t(v)=1] (1 - epsilon_T).
```

The paper should use dependency depth rather than affected-subgraph diameter.
Diameter is only safe under extra graph-closure assumptions.

## Rollout

Without an additional self-correction or revisit assumption, copy-not-recompute
does not imply contraction. A wrong value on an unaffected vertex can be copied
forever.

Let `e_t` be node error after step `t` when predictions are fed forward. A safe
worst-case accumulation statement is

```text
E[e_t] <= E[e_{t-1}]
          + E[(|A_t|/|S_t|)(1 - gamma_t)]
          + E[eta_t]
          + domain-change terms.
```

If `E|A_t| <= a`, `|S_t| >= c n`, `gamma_t >= gamma`, and `eta_t <= eta`, then
over `T` rollout steps,

```text
E[e_T] <= e_0 + T((a/(c n))(1 - gamma) + eta)
          + domain-change terms.
```

This explains slow, seed-dependent drift and explicitly rules out an
unconditional contraction claim.

## Gate Sample Complexity

For a gate hypothesis class with VC dimension `d` and `m_A` affected training
vertices, a standard VC-style sufficient condition gives, with probability at
least `1-delta`,

```text
FN_A(h) <= hat_FN_A(h)
           + O(sqrt((d log(m_A/d) + log(1/delta)) / m_A)).
```

Since recall is `rho=1-FN_A`, high affected recall is learnable when affected
false negatives are low and

```text
m_A = Omega((d + log(1/delta)) / epsilon^2).
```

This is a sufficient condition only, not a tight neural-network generalization
claim.

## Whole-Graph Accuracy

The theorem is not a graph-accuracy theorem. A union bound gives

```text
Pr[any node error] <= |S_t| E[node_err_t].
```

If node errors were independent, graph accuracy would scale roughly like
`node_acc_v^{|S_t|}`. This explains why high node accuracy can coexist with low
bottleneck graph accuracy and why SSSP graph accuracy has high seed variance at
large size.
