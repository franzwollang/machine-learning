# Proteus Development Plan

This document lays out the suggested milestone order for taking Proteus (Paper 1) from
its current state to a submission-ready paper with a complete reference implementation.
`OPEN_ISSUES.md` tracks the individual defects and ambiguities; this file sequences them
and defines exit criteria. Issue references below use the stable numbering from that file.

## Status board

Statuses live only here (`not started` / `in progress` / `blocked` / `done`); milestone
bodies below stay static descriptions.

| Milestone | Status |
|---|---|
| M0 — Spec/implementation sync pass | done |
| M1 — Canonical clustering objective | blocked on #44 (soft×gabriel@τ* seed1 nested survives; denser kills that inflate + keep-band×persist moot at e2e; still ≠sample-ARI. No awaiting flip.) |
| M2 — Characteristic-scale selection rebuild | in progress (thr0.35 narrow; thr0.30 seed2 LW≠coarse via closest-to-unit load; densify×LW seed1 stays coarse. Do not flip default.) |
| M3 — Constant audit & calibration tier | done (c_{d,k} + C_Q(d) calibrated on the shared uniform-d-ball ensemble; S14.3 three-tier audit #37 complete; intrinsic-dim estimator #39 validated vs GT + Levina–Bickel cross-check shipped, SI S1.4.1. Operational estimator-wiring divergence surfaced as #40, deferred to the M5 junction-detection consumer.) |
| M4 — Stage 2 core (complex, evidence gate, dual flow) | in progress (#43 spectrum_safe×policy + fail_closed EvidenceGate matrix flags off; do not close. #41: densify384/cal/lifetime/nested mid ≠(1,2,1)/SI b1. Remaining: hollow×cal / #45) |
| M5 — Inference interface & diagnostics | not started |
| M6 — Evaluation, benchmarks, paper finalization | not started |

## Current state (July 2026)

- **Stage 1 is implemented and test-backed** (~3.3k LOC in `code/proteus/src/`): fixed-tau
  scaffold (EWMA moments, Oja, deferred nudges, two-tier Hebbian links, splits with shadow
  inheritance, pruning, CV stabilization), warm-started scale-grid search, AP + Q-gated
  clustering, T2 PCA transfer, and Q-gated recursion. Default test run (post M0–M4 stack
  merge, `main` @ 7fab454): **189 passed, 26 xfailed** (10 real-data deselected; strict
  "awaiting" markers for unimplemented modules).
- **Stage 2 core has begun**: `stage2/flag_complex.py` (Greedy Chaining, S4.1/S4.2/S4.5)
  and the evidence gate (`evidence/dm_score.py`, `gate.py`, `star_matrix.py`; S3.4–S3.6,
  S10.4) have landed as tested modules. **Inference, diagnostics, and preprocessing remain
  empty stubs** with typed contracts (`tests/contracts/`) and awaiting tests written against
  SI sections. The evidence gate is not yet wired into the Stage-1 runtime loop (see #27).
- **Paper + SI are drafted** with two load-bearing spec gaps (Q-score canonicalization,
  characteristic-scale selection) and a set of documentation-sync debts where the
  implementation has already made the right call.

## Guiding principle

Constants and rules split into two classes, and only one deserves derivation effort:

- **Acceptance-path** quantities (tau* selection, Q-score / partition acceptance, DM gate,
  stopping criteria) determine what the learned object *is*. These must be derived,
  calibrated with a declared protocol, or explicitly falsifier-tested.
- **Proposal-path** constants (torsion bands, nudge thresholds, prune floors, split offsets)
  only steer where the system looks; the evidence gate backstops correctness. These are
  operational defaults — logged, tunable, and not blockers.

Milestones M1–M3 fix the acceptance path. M4–M5 build the remaining machinery against an
already-canonical Stage 1. M6 is evaluation and paper hardening.

---

## M0 — Spec/implementation sync pass (documentation only)

Clear every issue where the decision is already made in code and only prose lags.

- Routing weights: Gaussian-relative vs rank-decay, with the dimension cutoff (#19).
- Two-tier Hebbian edge semantics: shadow create vs lift vs counter update (#20).
- Soft-mean / hard-variance asymmetry in paper §3 (#21).
- tau / dimensionality notation reconciliation across paper and SI (#23).
- Link pruning: directed floor + bilateral agreement replaces Wilson gauntlet in S3.1/S3.2 (#29).
- Demote `eta_GNG` to analysis-only or wire it in — one decision, documented (#34).
- Resolve vestigial `s_control` (#35).
- Pick the canonical stabilization criterion: CV threshold vs trend-exhaustion (#24).

**Exit:** paper/SI describe the implemented Stage 1 mechanics verbatim; no pure doc-sync
entries remain in `OPEN_ISSUES.md`. Estimated effort: days, not weeks — and it makes the
genuine gaps visible instead of buried.

## M1 — Canonical clustering objective (#27)

The Q-score is acceptance-path and everything downstream (recursion timing, tau*-via-
persistence in M2) depends on it. This lands before scale-selection work.

- **DONE (Part A):** the exact formulas for `W_v(i,j) = K_v(i,j) * A_sym(i,j)`,
  `LocalIntra`, `BoundaryInter`, `InterLocal` are promoted into SI S2.6.1 verbatim, with
  symbols in SI S0.1 + paper notation.
- Make the single-cluster null a first-class candidate: a partition is accepted only if it
  beats the null by a margin under the same criterion.
- Replace the residual cleanup passes in `clustering.py` with one Q-improving pairwise
  merge applied to fixpoint; delete dead helpers (4 dead helpers already removed).
- Falsifiers (already in the suite): circle -> exactly one cluster without collapse logic;
  hierarchical Gaussian -> six terminal leaves; swiss roll unchanged.

**FINDING (cross-family validated, #27):** the last two bullets are **not achievable at a
single scale** with the current Q primitives. Experiments + an independent GPT audit show
circle/swiss ring-arc clusters and hierarchy coarse-blob clusters have *overlapping*
single-scale `Q(C)` (and extent / conductance / modularity / spectral-gap) distributions,
and the partition-`Q` null is degenerate (whole-component boundary = 0 → `Q → +∞`). It is
**not** a missing kernel term — `K_v` is already in `W_v`. The constant-free acceptance
rule and the null test are therefore **coupled to M2**: the arbiter is Q-partition
persistence across adjacent τ grid points (SI S2.6.2), or the M4 S3.4 DM evidence gate.
The `clustering.py` heuristic purge lands **with M2**, not before.

**Exit (revised):** *(Part A, done)* SI S2.6.1 pins the Q primitives verbatim and documents
the single-scale scope. *(Part B, deferred into M2)* `clustering.py` becomes heuristic-free
once the persistence signal is available; all clustering/recursion regressions still pass.

## M2 — Characteristic-scale selection rebuild (#28, #32, #31)

- Replace the variance-load band selector with the compensated node-count / support-trace
  signal (`N(tau) * tau^{d/2}` knees) as primary, and Q-partition persistence across
  adjacent grid points as the structural cross-check (this is S2.6.2 made operational).
- Write the `c_{d,k}` calibration protocol as an SI section (uniform d-ball ensemble,
  median `r_k / sqrt(tau)` at equilibrium, tabulated over (d, k)) and ship the lookup table.
- State the minimal tau <-> heat-kernel bridge: equilibrium lemma + declared
  `Sigma_smooth = tau * I` convention; anisotropic scale-space deferred (#32).
- Update the hierarchy-recovery harness to per-level tau and tighten its gates (#31).
- Keep the legacy load-band selector behind a flag during transition so scenario tests can
  bisect regressions.

**Exit:** `band_lo = 0.65` and the one-step-coarser patch are gone; scale-search tests pass
with materially tighter tau* tolerance (target: within one grid step of geometric truth on
the synthetic suite); SI S2.5/S2.5.1 rewritten to match.

## M3 — Constant audit and calibration tier (#36, #37, #39)

- Derive or calibrate `C_Q(d)` (needed by the S3.3 guards and the S12 edit-budget bound).
- Complete the S14.3 three-tier classification (derived / calibrated / free) covering every
  constant that exists in `src/`, with calibration protocols referenced where applicable.
- Validate the degree-proxy intrinsic-dimension estimator against synthetic ground truth;
  calibrate or replace with Levina–Bickel behind the same interface (#39).

**Exit:** every constant in code appears in S14.3 with a status; no acceptance-path
quantity is labeled "empirical" without a falsifier or calibration protocol.

## M4 — Stage 2 core: complex, evidence gate, dual flow

Build against the now-canonical Stage 1 output. The spec here is in good shape (closed-form
DM gate, concrete quadratic dual-flow solve, explicit density formula); this is mostly
disciplined implementation of S3–S6 and S10.

Suggested internal order (each step flips its strict-xfail "awaiting" tests):

1. **Flag complex + Greedy Chaining** (S4.1–S4.2), T3 transfer (S4.4), simplex data
   structures (S4.5). Unblocks the circle 1-ring topology test (#25).
2. **DM evidence gate** (S3.4–S3.6): closed-form score, affected-region localization,
   rerouting, Bayes-factor margin, cadence/hysteresis/edit budgets; star-matrix
   conditioning check (S10.4). Reduction tests: DM consistency (S3.5).
   **Also (post-#44 hollow-edge):** wire this gate into the Stage-1 split loop as the
   split arbiter so `max_nodes` can become a safety assert — diagnosis and remedy order
   in `reference/open_loop_growth_and_node_cap.md` (A1 opens a numbered issue when #44
   frees capacity; same wiring effort as the runtime scaffold loop).
3. **Dual flow + density** (S6.1–S6.4): online face-pressure tallies, conservative solve
   (loopy Gaussian BP), simplex-local density, mass-conservation and flux health checks.
   Reduction tests: simplex–node correspondence (S9.3), mass conservation properties.

**Exit:** `stage2.flag_complex`, `evidence.dm_score/gate/star_matrix`, and
`stage2.dual_flow/density` awaiting tests pass; circle and swiss-roll Stage 2 scenario
assertions (topology, reconstruction, held-out likelihood) pass.

## M5 — Inference interface and diagnostics

1. **Memberships** (S7): Gaussian summaries `(mu_C, Sigma_C)` via law of total covariance,
   closed-form membership, PL reference membership, multiscale trajectory. Flips
   `inference.membership` tests (circle, hierarchy).
2. **Torsion audit + ladder** (S5.1–S5.3): diagnostic, split placement, edge-ratio fallback.
3. **Junction detection + freeze** (S8.4), with the manifold-zoo dataset generator (#26)
   landing first as a fixture. Flips `diagnostics.junction` tests.
4. **Optional warps** (S5.5–S5.6) last, strictly evidence-gated: patch identification,
   mini-NSF training, identity-on-boundary, composition rules.

**Exit:** nested-spheres, linked-tori, and dimension-junction scenarios pass; membership
trajectory stability metrics run end to end.

## M6 — Evaluation, benchmarks, paper finalization

- Real-data loaders (COIL-20, CMU MoCap, OpenFace CK+, PBMC3k — currently
  `NotImplementedError`) and their scenario tests.
- Baselines (VAE, UMAP+KDE, VR reference) and metrics already scaffolded in
  `tests/metrics/` (persistent homology, MMD, held-out log-likelihood) wired into the
  S14 evaluation protocols.
- Ablations named in paper §8: fixed scale, heuristic acceptance, no torsion audit,
  no warps, uniform simplex density, leaf-only complexes.
- Paper hardening: architecture figure (#17), remaining figures from
  `paper_author_notes.md`, citations (#18), title decision (#16), final
  notation/cross-reference pass.

**Exit:** paper + SI submission-ready with real experimental sections; benchmark suite
green including the currently-awaiting performance envelopes.

---

## Dependencies and risks

- **M1 and M2 are coupled (finding, #27)**: the theory iteration budgeted below has now run.
  The Q primitives are trustworthy (Part A done), but the *cluster-count acceptance* — the
  single-cluster null and the constant-free heuristic purge — provably cannot be settled at
  a single scale: circle/swiss ring-arc clusters and hierarchy coarse blobs have overlapping
  single-scale `Q` (and extent / conductance / modularity / spectral) distributions, and the
  partition-`Q` null is degenerate. The arbiter is cross-scale Q-partition persistence, which
  is M2's secondary signal (S2.6.2). So M1-Part-B and M2's persistence machinery should be
  built **together**: implement persistence in M2, then complete the `clustering.py` purge
  against it. tau* selection (M2 primary) still uses the trustworthy Q, so the ordering risk
  the original note worried about is resolved in Part A.
- **M2 is the riskiest milestone**: replacing the selector can destabilize every recursion
  and scenario test simultaneously. Mitigation: keep the legacy selector behind a flag,
  migrate tests one scenario at a time, and require the new selector to dominate the old
  one on tau* accuracy before deleting it.
- **Q definition risk (M1) — RESOLVED as a spec/plan finding**: the merge-to-fixpoint rule
  cannot reproduce circle/hierarchy at a single scale, but this is *not* a missing `K_v`
  term (the kernel is already in `W_v`). The information needed is cross-scale (persistence),
  so no single-scale Q redefinition fixes it; see the coupling note above.
- **M4 step 2 (evidence gate) is the heart of the architecture**: everything after it
  ("geometry proposes, evidence decides") assumes it works. Its reduction tests
  (DM consistency) should be treated as blocking, not advisory.
- Packaging hygiene (#38 — move contract types out of `tests/`) **done**: canonical shapes
  now live in `proteus/types.py`; `tests/contracts/*` are re-export shims. `src/` no longer
  imports from the test tree, so M4 can add new types directly in the package.
