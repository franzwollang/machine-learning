# Open-Loop Growth and the max_nodes Cap

**Status:** theory note / directive (human + orchestrator, 2026-08-09). Companion to
`empty_region_evidence_and_scale.md`. Goal: eventually **remove `max_nodes`**; the cap
exists only because Stage-1 growth is not self-limiting. Guidance for later work — do
not start until the #44 hollow-edge track lands.

## Diagnosis (from code review of splits.py / scaffold.py / stabilization.py / pruning.py)

1. **Root cause — growth is open-loop.** `propose_splits` fires on a bare
   `variance > tau` threshold and `_propose_and_apply_splits` auto-accepts every
   proposal, every epoch: no significance test vs. the EWMA noise floor
   (`alpha = ln2/k`, ~11-sample window — nodes just under tau stochastically cross),
   no post-split verification, no rollback. The S3.4 DM evidence gate (cadence,
   hysteresis, edit budget — already implemented in `evidence/gate.py`) is **not
   wired into the loop**. The cap is a stand-in for the unwired gate.
2. **Counter-pressure self-normalizes away.** Node pruning and split viability use
   *relative* floors (vs. neighbour-mean / population-mean hits). Splits conserve
   flow, so an over-split region starves uniformly — nobody is low *relative* to
   peers and pruning never fires, precisely in the runaway regime. Same pathology
   as the flat Lindeberg response / node-count knee: adaptive equalization erases
   relative signals.
3. **No sample-support awareness.** Nothing knows the dataset has n points; epoch
   recycling re-presents the same data as fresh evidence. The controller's
   `max_nodes <= n/2` clamp is a crude proxy for the missing statistic.
4. **Irreducible-variance blindness.** Noise, tissue, curvature, and local
   intrinsic-dimension mismatch put a floor under variance that node insertion
   cannot remove; `tau_local` is uniform, so such regions split forever.
5. **Stopping test measures equalization, not adequacy.** `is_stable` checks
   variance-CV over *mature* nodes; fresh children reset `update_count` and drop
   out of the mature set, so persistent churn can hide from its own stopping test.

## Remedy, in leverage order (all flag-gated, default off; acceptance-path per S14.3)

1. **Wire the evidence gate as split arbiter:** splits become scored proposals;
   accept iff the DM marginal likelihood favors two children over the parent given
   actual sample support, under the gate's budget/hysteresis. This alone makes the
   cap a safety assert instead of load-bearing.
2. **Absolute support floors:** prune/split viability vs. counts relative to n,
   not the self-normalizing neighbourhood mean.
3. **Noise-floor null:** split only when over-cap excess is statistically
   significant vs. an irreducible-variance null (or let tau_local respond to
   local d).

## Instruction to A1

Open an issue for this when queue capacity frees up (post-#44-hollow-edge). It
naturally merges with the M4 "wire gate into runtime loop" plan item — same
mechanism, one wiring effort. Cross-cutting principle for all agents: **relative,
scaffold-side statistics self-normalize under adaptation; structural decisions
need evidence-bearing tests with declared nulls.**
