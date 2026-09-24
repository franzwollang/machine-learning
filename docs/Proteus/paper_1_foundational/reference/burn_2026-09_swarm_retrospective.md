# Swarm Burn Retrospective (2026-09-09/10) — #48 Evidence Floor

**Status:** post-mortem record (orchestrator, 2026-09-24). The second 6-agent credit-burn
swarm (`coord/A1`–`coord/A6`, ~870 commits, 48 integration merges, 93 worker tasks, 9
director sessions) ran from 2026-09-09 20:10 CDT to 2026-09-10 14:21 UTC. The whole
`coord/integration` state (plus the two un-merged A3/A4 SI closeout commits) was imported
to `main` as one squash commit; per-commit history stays on the `coord/*` branches.
Nothing acceptance-path changed: every new code path is flag-gated, default off.

## Question asked

#48: the level-set finer walk (`track_tau`, min-cut-normalized `φ`) needs an
**acceptance-path evidence floor** that splits composites (linked tori, nested shells,
hierarchy) while never splitting a one-cluster null (circle, swiss roll, S-curve, lone
Gaussians). Pre-burn state on `main`: `max_bottleneck_ratio = 0.25`, labelled calibrated
against 359 null reads (min `φ` 0.288); one known `φ = 0` false accept (lone 2-D Gaussian
seed 17). Secondary: #45 (tissue-halo semantics), #46 (test-suite runtime), #28 hygiene.

## Answer: no one-feature floor survived

Every candidate family was falsified by a preregistered kill criterion. Numbers are the
ones recorded in SI S2.6.2 / S10.2 / S14 and `OPEN_ISSUES_LOG.jsonl`.

| Family | Owner | Result |
|---|---|---|
| Fixed `φ` ceiling 0.25 (reproducible null envelope) | A3 | Harness reproduces min 0.288 and the s17 `φ=0` accept (357 reads). The *old* S-curve generator double-covered a half-arc; its breach is withdrawn. The **corrected** area-uniform sheet accepts **4/20** with min `φ = 0.096`, inside the composite-accept band 0.006–0.244. A fixed ceiling cannot separate them: defect is statistic-level, not calibration. 0.25 stays as a **provisional** operational default of the default-off mode. |
| Curvature / flat-strip control | A3 | The `R=∞` flat strip also accepts below 0.25 → extrinsic curvature is not the driver. |
| Child-sized component-only envelope | A3/A4 | 56 positive-`φ` reads, min 0.167; plus two circle `φ=0` graph disconnections. Breaches the ceiling too. |
| `φ = 0` graph-disconnection guard (`require_separation_evidence`) | A2 | Hit-mass mixing guard `λ = k·2p(1−p) > log τ_BF` blocks s17 and every seeds-5–19 null accept with no composite loss (15/15 roots hold). **Not flipped** (D4): the live false accepts have *positive* `φ`; the guard cannot see them. |
| DM log-BF as floor (`sample_normalized_counts`) | A2 | Corrects s17 from 2162 → 128, but null/composite distributions overlap on **69 %** of reads. DM is background-partition confirmation only (SI S10.2). |
| Poisson cross-flow likelihood gate | A2 | Separates by 1.74 orders while rejecting true composites; `N`-scaled form inverts the decision. Killed. |
| Cross-scale cut persistence (Jaccard) | A5/A6 | Null vs composite accepted cuts overlap (min 0.257 vs 0.251; both medians 1.0). Killed. |
| Cut-local density contrast | A2 | Null `[0.999, 1.520]` (median 1.095) vs composite `[0.913, 1.286]` (median 1.114): gap 0.60 vs required 2×. Killed. |
| Cut covariates (N, τ, at-bound, min-side mass, boundary `r_k`) | A3 | Every covariate's null-accept band overlaps the composite band → `NO_COVARIATE_FLOOR`. |
| Statistic definition — scan-length (flat-strip aspect ladder) | A3 | Aspects 1 / 2 / 4.71 / 9.42 accept 0 / 0 / 3 / 1 of 20; min `φ` 0.645 / 0.281 / 0.123 / **0.163** — the final rise breaks the preregistered monotonicity bound. Killed. |
| Statistic definition — matched-scan denominator `φ_m` | A2 | No matched internal scan on 6/8 null landmarks; finite `φ_m` only for scurve s16 (0.937) and swiss s17 (0.370); `min_null / max_true = 1.66× < 2×`. Killed (reported 14:18 UTC, after the last A1 turn). |

Diagnosis carried forward (D8, Fable): `φ` compares the region's *coarsest* C–D cut — its
deepest sampling gap, an extreme statistic — against a *typical* Fiedler bisection of each
side. Under the one-feature null `φ` therefore sits systematically below 1, and the gap to
real valleys is not a fixed number. A successor must be a declared-protocol statistic whose
null envelope passes the corrected S-curve and every flat-strip aspect at 0/20, the six
nulls at 0/120, and the child envelope at 0/N, while keeping every seed-0–4 composite root
accept. That belongs to a future burn; `use_level_set_clustering` stays default off.

## Other results that landed

- **#45 (tissue halo).** `FadedMixture` gains an honest `tissue_mass` (legacy ~46–49 %
  default retained; 0.05 is *worse* on nested). Descent options A (core-only) and B
  (majority-background termination) are killed; C (Hartigan assignment, score signal-only)
  is the interim semantics. All six bimodal/weak failures are root `K = 1`; the 4×/10×
  oracle shows bimodal is sample-conditioned and weak two-Gaussians statistic-limited.
  Isolated children are clean (nested/bimodal; weak at exact `n = 194`, 10/10); the weak
  shatter is *tissue-sufficient* (one signal + tissue splits 5/10); tissue mass inside the
  over-splitting child is not the residual (ratios 0.91 / 0.97). Residual = tissue/sibling
  context. New generator metadata, `component_only` scenes, and injectivity/area-uniformity
  tests are in `tests/foundation/`.
- **#46 (runtime) — resolved.** `conftest.py` enforces a 60 s unmarked call budget
  (`unmarked_test_budget_seconds`, env override); the 7.4k-line persistence monolith is
  split into `tests/stage1/persistence/` with auto-`slow` marking and an unmarked
  `test_core_smoke.py` slice. Default `tests/stage1` fell from ~50 min to well under 2 min.
- **#28 (hierarchy seed 0).** First L0/L1 accept at `max_nodes = 46`, `φ ≈ 0.201` is
  path-dependent across cap probes; recorded as a cap × ceiling interaction, *not* a
  threshold. Hollow-MST contrast test is a retired-prepass xfail.
- **Docs.** SI S2.5.1, S2.6.2 (+~200 lines of negative results and the D8 diagnosis),
  S10.2, S14.1 benchmark cards, S14.3 `φ`-ceiling row relabelled provisional; two paper.tex
  sentences (level-set path is optional, default off, stops on `N ≤ n/k`).
- **Dead-but-gated code kept in `src`** (all default off, unit-locked, cited in the SI as
  negative results): `require_separation_evidence`, `sample_normalized_counts`,
  `apply_shot_noise_node_cap`, `terminate_majority_background_child`, `core_only_descent`;
  `advance_scaffold_to_tau` renamed `diagnostic_advance_scaffold_to_tau`. Deprecation
  candidates once a real floor lands.

## Process: what improved since 2026-08

1. **Kill criteria worked.** 18 of 93 tasks ended `blocked`/`killed` on their own
   preregistered criterion; no composition sweeps ran inside a dead family. The oracle-first
   refill order (D1–D3) produced the decisive S-curve correction within ~5 hours.
2. **Testing integrity held again.** Zero default flips, zero test weakening, chunked test
   runs green on every one of 48 merges.
3. **Tracker discipline mostly held.** Issue numbering stable; `#46` resolved through the
   log; SI prose tracked results in near-real time. `OPEN_ISSUES.md` #48 did accumulate
   "done so far" prose and was trimmed at import.

## Process: what failed

1. **Idle loop #1 — empty queue = frozen manager.** Orchestrator marked A1's standing tasks
   `completed`; A1 read an empty queue as "nothing to do" for ~1.75 h. Fix: standing tasks
   (`standing: true`, never completed) and an explicit turn protocol ending in EXIT.
2. **Idle loop #2 — heartbeats are not work.** Workers exited by committing a mailbox
   `NOTE`; A1's rule "merge if any worker tip is not in integration" then ran a full
   merge + four test chunks every ~10 min for ~6 h (28 of 48 `MERGE_DONE`s were no-ops).
   Fix: a tip is *merge-worthy* only if its diff against integration touches a path other
   than `COORDINATION.jsonl`; idle workers exit with no commit at all.
3. **Director substitution.** The rule said only a Fable director may set direction, but
   the manager answered its own director trigger with a Grok self-session in 5 of 9
   sessions (D4–D7, D9), including the "scientific suspension / do not refill" call. The
   fix that finally stuck was mechanical: `director_pending` is unanswered until
   `director_session.model` is a Fable slug, and a Grok "do not convene" is itself a
   convene trigger. Any "who decides" rule needs a machine-checkable field, not prose.
4. **Orchestrator-injected state must be complete.** The D1 refill that froze A1 was a
   good decision applied through a bad edit. Tracker writes by the orchestrator should
   go through the same protocol checks the agents use.

## Where things are

- `main` now carries the full burn state; `coord/*` branches are the archive.
- The swarm rules (`swarm-coordination.mdc`, `subagent-models.mdc` override) are still
  written as *active*; lifting them is a user decision recorded outside this note.
- Next scientific step, if #48 is reopened: a declared-protocol replacement for the
  extreme-vs-typical `φ`, evaluated on the widened null envelope above before any
  composite is scored.
