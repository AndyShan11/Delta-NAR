# Incremental Neural Algorithmic Reasoning — Design of Record

**Working title:** *Learning to Update, Not Recompute: Neural Algorithmic Reasoning for Dynamic Graphs*
**Benchmark name (candidate):** CLRS-Δ / Dyn-CLRS (retire "DynaRoute/DARB" — too applied)
**Target:** a rigorous conference evaluation
**Status:** direction proposed 2026-05-29, narrowed from initial "三件套" idea. Under discussion.

---

## Thesis (one sentence)

Classical CS has a whole theory of **incremental/dynamic algorithms** that achieve sub-linear update by *localizing* work; Neural Algorithmic Reasoning has ignored it entirely (CLRS-30 is 100% compute-from-scratch). We introduce the incremental regime to NAR and show that **aligning a reasoner to an incremental algorithm — updating, not recomputing — yields better size/rollout OOD generalization at lower compute.**

## The gap (verified against literature, May 2026)

- CLRS-30 (Veličković et al., ICML 2022): 30 algorithms, **none incremental**. OOD protocol train n=16 / test n=64; MPNN 96.6% in-dist → ~45% OOD.
- Recent extensions stay static: GNARLy (arXiv:2509.18930, NP-hard via RL — explicitly "does not address dynamic/incremental"), KNARsack (arXiv:2509.15239, pseudo-poly).
- NAR × incremental algorithms (D*Lite / LPA* / Ramalingam–Reps): "an emerging research area" — i.e. unclaimed.
- Distinction to draw in related work: temporal-GNN benchmarks (TGB etc.) *predict* on dynamic graphs; they do **not** *execute* a known incremental algorithm with a supervised trajectory. That is the novelty.

## Foundations relied on

- NAR blueprint: encode–process–decode; algorithmic bottleneck; processor reuse (Veličković & Blundell, Patterns 2021, arXiv:2105.02761).
- Algorithmic alignment (Xu et al., ICLR 2020) → theory hook below.
- Min-plus/tropical MPNN ≈ Bellman-Ford is the CANONICAL alignment result (Veličković et al., "Neural Execution of Graph Algorithms", ICLR 2020). **Use as cited building block; do NOT claim as contribution.**
- Strong baseline = generalist Triplet-GMPNN (Ibarz et al. 2022).

---

## On scope & generality (rebuttal-ready: "why N tasks, not all dynamic graphs?")

Two distinct kinds of "breadth" — we narrow the first, strengthen the second:
- **Task-list breadth** (how many tasks in the suite) = engineering pile-up. More ≠ stronger; a big task pile is a poster/dataset-track signal.
- **Principle generality** (does the *mechanism* cover a whole class) is the core scientific axis. DeltaNAR's `(G_t,y_{t-1},h_{t-1},Δ_t)→y_t` + affected-region gate applies in principle to any problem with an incremental algorithm.

Scope rationale:
1. **CLRS-30 precedent.** CLRS-30 is <10% of textbook algorithms yet nobody calls it narrow — it samples representatively + defines an evaluation paradigm. We do the same for *incremental* NAR. First-mover defines the subfield → everyone after cites + adds tasks on top.
2. **Generality is proven by task-OOD transfer, not task count.** Train one processor on SSSP, test zero-/few-shot transfer to unseen incremental MST/connectivity. "Train 3, transfer to a 4th" is verifiable generality; "train 7 independent tasks" proves nothing (could be 7 overfits). This is processor-reuse, the soul of the NAR position paper.
3. **Inclusion bar is principled, not arbitrary:** a task must have a textbook incremental algorithm emitting a correct hint trajectory. Without a clean trajectory, evaluation either degrades to final-answer-only or uses an ill-defined alignment target. MAPF and min-cost flow are therefore outside the present benchmark scope.
4. **Design space is large + extensible** (Demetrescu–Italiano APSP, HDT connectivity, Pearce–Kelly topo-sort, incremental cycle detection, …). CLRS-Δ provides an extensible architecture and a protocol for adding incremental algorithms; the present evaluation uses three anchor tasks. Generality lives in the architecture, not the count.
5. **Scope = boundary vs temporal-GNN.** "Dynamic graph" ≠ "algorithmic reasoning on dynamic graphs." Predicting future links (TGB) is statistical, not NAR. Restricting to "tasks with an incremental algorithm to *execute*" is the necessary line that answers "how is this different from a temporal-GNN benchmark?"

**Trade-off:** a broad dynamic-graph benchmark would answer a different question. The present work prioritizes a deeper evaluation of incremental algorithm execution.

## Contributions (scoped to one paper)

### C1 — Benchmark: CLRS-Δ
Built ON the existing CLRS probe/hint infrastructure (do not reinvent the data format).
Each sample: `(G, solution_state, Δ) → (updated_solution, update_trajectory)`.
- **Anchor task: Dynamic SSSP** — ground truth from Ramalingam–Reps / LPA*.
- **Breadth (2): incremental MST, dynamic connectivity / transitive closure.**
- **Inclusion bar:** task must have a textbook incremental algorithm that emits a correct hint trace. (MAPF, min-cost-flow, congestion routing FAIL this bar → paper 2.)
- Three supervision signals (map 1:1 onto LPA*):
  1. final answer = updated g-values / predecessor pointers
  2. dynamic hints = affected / locally-inconsistent vertex set (the queued work-set)
  3. local recompute trajectory = order of key-extractions
- **Controlled axis ‖δ‖** (number of affected vertices/edges) is a first-class benchmark dimension.

### C2 — Model: DeltaNAR
`(G_t, y_{t-1}, h_{t-1}, Δ_t) → y_t`
- Event encoder (Δ = {edge update, node update, source/target update, failure}).
- Previous-solution memory (stateful processor — a twist on processor reuse).
- **★ Affected-Region Gate** = the novel mechanism. Predicts A_t ⊆ V = the algorithm's work-set; computation localized to A_t. Supervised by the classical algorithm's locally-inconsistent set (legitimate hint-supervision).
- Min-plus processor (cited building block).

### C3 — OOD generalization
| Axis | Role |
|---|---|
| Size (16→64→128→512) | table stakes |
| Weight / Topology (ER→scale-free/grid/road) | table stakes |
| **Dynamic-Intensity (train ‖δ‖=1, test k-edge/batched)** | ★ soul; tests theory |
| **Temporal/rollout (short→long update sequences)** | ★ soul; drift |
| Failure / cascade | stretch |

Metric: **graph-accuracy** (whole-graph correct), not per-node (GNARLy critique).

---

## Theory motivation

Incremental algorithms are output-sensitive: Ramalingam–Reps updates SSSP in time ~O(‖δ‖), not O(|V|+|E|).
**Claim (informal):** from-scratch NAR work scales with graph diameter/size → size-OOD collapse; DeltaNAR work localized to A_t scales with ‖δ‖ ≈ independent of n for local edits → degradation tracks the affected-region size distribution, not n.
**Falsifiable prediction:** degradation-vs-n flattens for DeltaNAR, steepens for from-scratch; both track ‖δ‖.
**Honest boundary:** edge-weight *decrease* can affect a Θ(n) downstream subtree → clean size-win lives in increase / non-tree-edge regime. Make this a controlled axis, not a hidden caveat.

## Decisive experiment
DeltaNAR vs (1) recompute-from-scratch Triplet-GMPNN every step [strong, fair], (2) Naïve-Δ (feed y_{t-1}, no gate). Win = accuracy parity-or-better + large size-OOD gap that tracks ‖δ‖ + rollout stability + sub-linear compute. The gate must earn accuracy, not just FLOPs.

## Risks
1. **Ground-truth generation correctness** (Ramalingam–Reps/LPA* + hint emission) — main engineering risk; local, testable.
2. Gate must improve accuracy, not only compute — designed into intensity/rollout axes.
3. Baseline fairness — adapt the generalist, never a vanilla-MPNN strawman.

## Cut from initial idea (→ paper 2 / future)
- LLM controller (natural-language constraints).
- MAPF, min-cost-flow, congestion-aware routing (no clean incremental ground-truth trace).

## Immediate next steps
1. Pull & read the CLRS codebase (google-deepmind/clrs) — understand probe/hint sampler internals.
2. Read Ramalingam–Reps (1996) + Koenig/Likhachev LPA*/D*Lite (AAAI 2002) closely.
3. Prototype Δ-SSSP ground-truth generator locally + unit-test against from-scratch Dijkstra on small graphs.
