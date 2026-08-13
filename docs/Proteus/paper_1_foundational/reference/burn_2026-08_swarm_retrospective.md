# Swarm Burn Retrospective (2026-08-09/10) — Lessons Learned

**Status:** post-mortem record (orchestrator, 2026-08-12). The 6-agent credit-burn swarm
(`coord/A1`–`coord/A6`, ~90 integration merges, ~850 commits) is over. Durable artifacts
were selectively imported to `main`; the ~100 one-off `test_ph_*` probe files (seed- and
schedule-specific negative results) were left on `coord/integration` — their findings
live in the SI, `OPEN_ISSUES.md`, and the logs.

## What was imported to main

- Flag-gated src modules (all defaults off, no acceptance-path changes):
  `stage1/edge_evidence.py` (hollow-edge/Gabriel/soft-capacity + ROC/Youden/Poisson
  calibrators), `stage2/dual_flow.py` (#43 dual-adjacency/BP-sketch family),
  gate/persistence/controller/recursion flag extensions.
- Core test locks for those flags (`test_edge_evidence.py`, `test_recursion.py`,
  `test_scale_search_persistence.py`, evidence tests), the PH per-region harness core,
  and the hollow-edge adversarial-null/ROC infrastructure (reusable for any future
  edge- or density-classifier calibration).
- SI/paper updates (experiments recorded as *proposed*), the two theory notes, and
  current tracking state (`OPEN_ISSUES.md`, `PLANNING.md`, logs).

## Scientific lessons

1. **The graph-cut separation family is falsified for tissue-bearing scenes.** The
   decisive result was the oracle experiment: cutting *every* cross-label edge with
   ground-truth labels still leaves one connected component — tissue nodes bridge
   everything. With ~20% uniform clutter, the support is genuinely connected;
   "disconnected components" exist only for the signal subset. No edge statistic can
   fix this, regardless of calibration quality.
2. **Emptiness is the wrong statistic under clutter; density contrast is the right
   one.** Hollowness (H-ratio, Gabriel) measures dip-to-zero; tissue guarantees density
   never dips to zero. Separation must come from *upper level sets* of density —
   Hartigan's cluster tree (Chaudhuri–Dasgupta robust single linkage; ToMATo).
3. **Midpoint tests fail on interlocked geometry.** Cross-tori edges had H median ≈ 1:
   straight segments between interlocked tubes pass near the other tube's data.
4. **Edge-local counting statistics starve on coarse scaffolds** (64 nodes → endpoint
   balls with ~0 samples), and lifted-edge pruning is not a cut-set.
5. **"Majors K=2" is not recovery.** Component counts over major clusters repeatedly
   looked like wins while sample-ARI stayed at chance. Sample-level, background-aware
   ARI is the only honest metric.
6. **Seed-fragile effect windows are noise.** A parameter band that works at seed 0 and
   vanishes at seeds 1–2 (the recurring "keep band") is fixture-fitting, not a finding.

## Process lessons

1. **Directives need kill criteria.** The oracle falsification arrived early; the swarm
   then ran hundreds of composition sweeps (soft × gabriel × persist × denser × seed)
   inside the dead family because refill pressure (throughput mode) had no
   "if X fails, abandon the family and escalate" rule.
2. **Throughput mode optimizes tasks completed, not information gained.** Queue refills
   should price experiments by expected information (decisive oracles first), not by
   availability of parameter grids.
3. **What worked:** testing integrity held perfectly (zero `@awaiting` flips over ~850
   commits, defaults all off); the mailbox/turn-boundary merge protocol ran ~90 cycles
   without a lost update; SI prose tracked experiments in near-real time.

## The pivot (see OPEN_ISSUES #44)

Target object changes from "components of the support" to **components of density
upper level sets with an explicit background class**, read from the equalized
scaffold's node-spacing field (magnification: node density ∝ p^gamma, and the cluster
tree is invariant to monotone transforms, so gamma need not be known). Validation =
batch Chaudhuri–Dasgupta oracle on raw samples vs. scaffold-spacing version, scored by
background-aware sample-ARI. The prepass-flag zoo in `recursion.py`/`edge_evidence.py`
becomes deprecation candidate once the level-set layer validates.
