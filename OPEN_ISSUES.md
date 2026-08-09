# Proteus Paper 1 — Open Issues

Current, active issues only — resolution history lives in `OPEN_ISSUES_LOG.jsonl`, never
here. Numbering is historical and stable: resolved issues are deleted rather than
renumbered, so gaps in the sequence are expected. Each entry lists only the work that
actually remains. See `PLANNING.md` for the suggested order of attack.

Next issue number: 45

## 16. Fuzzy title decision

- Title emphasizes "fuzzy manifold memberships." §6.3 has an operational anchor, but this is not central.
- Revisit only when the paper moves toward submission.

## 17. Architectural overview figure

- The paper has several owned objects and stages without a unifying diagram (placeholder figures exist in `paper.tex`).
- Produce once the reference implementation stabilizes, so the figure reflects the real pipeline.

## 18. Formal citations

- §1.5 is prose-only; the intended citation list is tracked in §13 "References Prep."
- Does not affect implementation.

## 25. Circle mesh topology test

- The circle scaffold passes node-count and reconstruction-error assertions but lacks an explicit topology check that the lifted-edge graph is a single connected 1-ring (Betti_0 = 1, Betti_1 = 1).
- Options: (a) Vietoris–Rips persistent homology on node positions (`giotto-tda` / `ripser`), (b) flag-complex Betti numbers via `gudhi`, (c) simple graph checks (connected components + cycle rank) on the lifted graph directly.
- **FINDING (empirical, turn 19): option (c) is insufficient and even the bare flag complex over-reports `b1`.** On the standard circle fixture at `tau*` (64 nodes) the lifted graph is a single connected component with no isolates (so `b0 = 1` is recoverable), but it is a *triangulated band*, not a clean 1-ring: raw undirected cycle rank `E - V + 1 = 50` (E=113, max degree 7). Building the flag/clique complex on that same graph (55 triangles + 12 tetrahedra) collapses most but not all spurious loops, leaving `b1 = 6` — still not 1. So a correct `b1 = 1` requires either a *persistence-filtered* PH (take the single most-persistent H1 feature; needs `gudhi`/`ripser`, neither currently installed) or scaffold-mesh cleanup, not a fixed-threshold graph/clique-complex Betti count. `b0 = 1` (single connected 1-skeleton) is the only topology invariant robustly available at Stage 1.
- Blocked on / naturally lands with Stage 2 flag-complex construction **plus a persistence filtration**; the naive Stage-1 graph check (c) cannot deliver the loop invariant. If a Stage-1 check lands, scope it to `b0 = 1`.
- **UPDATE (turn 20): flag-complex construction has landed** (`stage2/flag_complex.py`; SI S4.1/S4.2/S13.4), and it confirms the finding. The *sparse lifted-graph* flag complex of the fitted circle scaffold (built to `d_final`, expanded to a clique complex) retains **6 essential (never-filled) `H1` loops** — the triangulated-band holes are not closed by any lifted clique, so no persistence threshold recovers `b1 = 1` from the lifted graph alone. Vietoris--Rips PH on the node *positions* (SI S14.2, dense pairwise) is the route that can fill the band holes, but on the tissue-polluted circle scaffold it also births spurious loops and does not cleanly separate `b1 = 1` at the fixed `1.5 sigma_star` filtration. The residual topology-recovery work (choosing the filtration/persistence reading that robustly recovers `b1 = 1` on real scaffolds) is tracked in **#41** (per-region harness + lifetime reading now scaffolded; do not flip circle `b1 = 1` until fitted-region evidence is green); this issue keeps the sharpened `b0 = 1`-only Stage-1 scope.

## 26. Manifold-zoo junction test (circle + line + plane + box)

- Classic GNG benchmark: 1D circle, 1D segment, 2D plane patch, and 3D box meeting at dimensional junctions.
- **Generator has landed** (`tests/datasets/synthetic/manifold_zoo.py`, `make_manifold_zoo`): a connected R^3 scene of intrinsic dims {1,1,2,3} meeting at 1<->1 / 1<->2 / 2<->3 junctions, with full per-component intrinsic-dim ground truth, per-component topology (circle carries `b1 = 1`), and three `JunctionExpectation`s. Backed by a new `AxisAlignedBoxFadedComponent` (solid k-box signal). Diagnostic tests (`tests/scenarios/synthetic/test_manifold_zoo.py`, generator coverage in `test_dataset_scales.py`) pass now.
- **Remaining (deferred scenario assertions, blocked on later milestones):** mesh quality per patch, `d_final` accuracy at junctions, junction detection (S8.4, M5) and Stage 2 heterogeneous simplex dimension (S4.2, M4). Placeholders wired as `@awaiting("diagnostics.junction", si="S8.4")` and `@awaiting("stage2.flag_complex", si="S4.2")`; flip them when those modules land.
- **UPDATE (turn 20):** the flag-complex constructor now exists (`stage2/flag_complex.py`) and handles heterogeneous per-star `d_final` correctly (unit-tested). But `test_manifold_zoo_heterogeneous_simplex_dimension` cannot be flipped yet: the operational `d_final` is seeded to the working dimension and never refreshed (#40), so a fitted zoo scaffold carries uniform `d_final = 3` and the constructor produces uniform 3-simplices, not the ground-truth per-patch `{1,1,2,3}`. This test therefore additionally blocks on the #40 `d_final` refresh landing at its S8.4 junction-detection consumer (M5).

## 27. Clustering: canonicalize the Q-score and remove cleanup heuristics

The AP -> Q-merge -> refine pipeline is implemented and passes the circle, swiss-roll, and hierarchical-Gaussian regressions (six terminal leaves). Recursion is Q-gated (`recursion.py`: leaf when `n_clusters <= 1` or `partition_q_score <= 0`). The Q primitives are now pinned down in SI S2.6.1 (`K_v`, `A_sym`, `W_v = K_v * A_sym`, `LocalIntra`, `BoundaryInter`, `InterLocal` promoted verbatim from `reference/stage1_clustering_and_resolution.md`). What remains is making the implementation heuristic-free:

- **FINDING (cross-family validated): the constant-free single-scale null is under-determined.** Direct experiments plus an independent GPT audit (agent `0b462607`) establish that at a single scale the graph-local `Q(C)` and `InterLocal/LocalIntra` do **not** carry enough information to separate a uniform manifold that must be one cluster (circle/swiss ring arcs) from genuinely multi-modal structure that must stay split (hierarchical-Gaussian coarse blobs). Their per-cluster `Q` distributions overlap (circle arcs `Q≈0.55–1.18`; hierarchy coarse-blob clusters `Q≈0.58–0.88`), and the same overlap holds for extent ratios `v̂/v`, conductance, modularity, full-graph (shadow∪lifted) variants, and spectral gap. Separately, the partition-`Q` null is mathematically degenerate: a whole connected component has empty boundary → `BoundaryInter = 0` → `Q(null) → +∞`, so a same-criterion comparison trivially favours one cluster. The currently-passing pipeline distinguishes the cases only via **side channels** (lifted-component count, size imbalance). Conclusion: "one Q-improving merge-to-fixpoint + a single-cluster-null test, with no constants" is *not achievable with the present single-scale primitives* — and it is not a missing `K_v` term (the kernel is already in `W_v`).
- **Canonical arbiter deferred to persistence (M2) / DM gate (M4).** The intrinsic-vs-composite distinction is a cross-scale statement (SI S2.6.2): a real partition persists across ≥ 2 adjacent τ grid points; a uniform manifold's arc-partition should not. The persistence arbiter now **exists** (`stage1/persistence.py`; #28). *With the S2.6.1 stand-ins present* it reproduces the qualitative discrimination (circle → no persistent split; hierarchical → persistent 3-way split), because the recorded per-scale partitions are already heuristic-collapsed. SI S2.6.1/S2.6.2 document the scope and operational signal: single-scale `Q` is a proposal screen and the cleanup passes are operational stand-ins (empirically still **load-bearing** — see the corrected finding below). The alternative M4 S3.4 Dirichlet–multinomial gate supplies the complementary non-degenerate likelihood-ratio null.
- **Persistence accept-gate is wired into recursion (`RecursionConfig.require_persistent_split`, SI S2.6.2), default off.** A region's split is accepted only if a multi-cluster partition persists across adjacent `tau` grid points (`persistence_result.tau_star_index is not None`); non-persistent fragmentation makes the region terminal. Two integration tests lock in the *gate-with-stand-ins-present* behaviour (`test_persistence_gate_circle_is_single_feature` → single leaf; `test_persistence_gate_hierarchy_matches_gt` → six leaves, fine ARI 1.0).
- **CORRECTED FINDING (cross-family validated): the persistence gate does NOT replace the stand-ins; they are load-bearing and deletion is BLOCKED.** A *full* ablation — monkeypatch `_refine_boundaries`, `_absorb_*`, **and** `_q_merge_any_improving` (which also disables the `<= 3` collapse and the `>= 4` re-merge) to identities — with `require_persistent_split=True` gives **circle → 37 leaves** (want 1) and **hierarchy → 12 leaves** (want 6). This *refutes* the earlier turn-7 ablation note ("gate alone yields circle→1"), which almost certainly left the `<= 3` collapse intact and so silently kept the very heuristic under test. GPT audit (agent `320a28ae`) reproduced circle→37 exactly and confirmed the diagnosis and every point below.
  - *Mechanism (warm-start false positive):* on the warm-started sweep the circle's arc-partitions mostly do NOT agree across adjacent scales (matched-Jaccard 0.17–0.33, as the theory predicts), but an **isolated fine-end pair coincides at 0.609** — enough to satisfy `P_persist=2` at `theta_ovl=0.5`, so the gate accepts a spurious split and recursion explodes. An independent **cold-start** refit of the same scales removes that block entirely (overlaps drop to 0.475/0.419), confirming it is a warm-start/path-dependence artifact, not a real feature. A genuine feature (hierarchy) instead persists from the **coarsest** grid point with high, stable overlaps (0.68–0.93) over a run of length ≥ 3.
  - *Hardening direction (ii) — coarse-anchoring — has LANDED (`PersistenceConfig.coarse_anchored=True`, default).* The characteristic split must be anchored at the coarsest multi-cluster grid point: letting `j0` = coarsest index with `K >= 2`, accept iff `run_length[j0] >= P_persist`, else terminal. This rejects the isolated fine-end warm-start block. Full-strip recursion ablation (all stand-ins → identities, gate on): **circle 37 → 1 leaf, hierarchy 12 → 6 leaves** (both correct); **swiss roll 32 → 12 leaves** (still over-fragments). Suite stays green because every current test runs *with* stand-ins present, where coarse-anchoring reproduces the identical `tau*` as the legacy rule. Cross-family GPT audit (`gpt-5.4-high`, agent `e8eef21f`): implementation correct, swiss-still-fragments conclusion correct; verdict LAND WITH CAVEAT — the scale-space justification is *motivational, not a theorem* (non-enhancement is about smoothed-density extrema, not the warm-started scaffold partition sweep), so it carries a transient-coarse-blip false-negative and grid-sensitivity (both now documented in SI S2.6.2 as operational trade-offs).
  - *Residual (narrowed):* the remaining blocker is **marginal coarse-scale arc-persistence on developable manifolds** — the swiss roll's coarsest partition is 3 arcs whose adjacent overlap (~0.568) sits just above `theta_ovl=0.5`, so coarse-anchoring admits it and the region fragments without the stand-ins.
  - *Hardening (ii′) — cold-start path-independence recheck — IMPLEMENTED and REFUTED as a gate.* `PersistenceConfig.cold_start_recheck` (default **off**) + `controller._cold_start_recheck` + `persistence.interval_is_persistent` re-fit the candidate coarse-anchored interval from independently cold-started scaffolds and keep it only if it still persists. It does **not** work: cold single-`tau` fits have high **resolution-level** variance, so a genuine multi-level feature's interval fails the overlap test. On the hierarchical Gaussian the warm coarse anchor is a stable 3-way partition but independent cold refits of the two anchor grid points return 6-way vs 3-way (matched overlap ≈0.27 < 0.5) → interval rejected → **full-strip recursion ablation with the recheck on collapses hierarchy to 1 leaf** (want 6; circle→1, swiss→1). The matched-Jaccard overlap cannot separate true absence-of-structure from ordinary cross-scale resolution variance — exactly the discrimination the S3.4 Bayes-factor *margin* provides. This refutes the specific overlap-based recheck (not every conceivable path-independence diagnostic), leaving path (i) as the only currently validated route. Mechanism retained behind the default-off flag as a reproducible diagnostic (SI S2.6.2). Independently reproduced + implementation-audited by cross-family GPT (`gpt-5.4-high`, agent `f49edf2a`): warm anchor 3/3 overlap 1.0 persist; cold 6/3 overlap 0.266 reject; verdict LGTM. NOT recommended: raising `theta_ovl` alone (brittle) or `min_persistence >= 3` alone (overfit risk).
  - *DM cluster-acceptance reduction — IMPLEMENTED behind a flag, validated, finding below.* The S3.4 gate is written for node/star edits and does not state the partition-into-K reduction; that reduction is now specified (proposed SI S2.6.3) and implemented as `stage1/dm_cluster.py` (`block_flow_matrix`, `dm_partition_logbf`, `dm_gated_merge`, `dm_partition_verdict`, `run_clustering_dm`) plus `RecursionConfig.require_dm_split` (default off). The reduction models a candidate K-block partition as a Dirichlet–multinomial homogeneity test of the K block-to-block routing rows (`log BF = Σ_k log m(N_k) − log m(ΣN_k)`), which reduces term-for-term to the audited `evaluate_edit`/S3.5 gate for K=2 (locked by `test_dm_cluster.py`).
    - *FINDING (measured; matches the S2.6.1 cross-scale argument): single-scale DM alone over-fragments at every recursion level.* Recursion leaf counts (gt circle=1, hierarchy=6, swiss=1): `dm` alone → circle **56**, hierarchy **32**, swiss **76**; `persist` alone → 1/6/1; `default`(stand-ins) → 1/6/1; **`persist+dm` → 1/6/1**. A developable manifold's arcs have band-concentrated block rows the homogeneity test reads as heterogeneous, so no single-scale statistic (Q, conductance, block-BF) rejects them — the discrimination is inherently cross-scale. The DM margin is therefore *complementary* to persistence, not a standalone replacement.
    - *Consequence:* the validated heuristic-free path is **DM ∘ persistence** — persistence rejects uniform-manifold arc-partitions cross-scale; the DM-gated merge does the within-region partition with **no** S2.6.1 stand-ins, matching the default leaf counts on all three scenarios. The load-bearing stand-ins (`_refine_boundaries` eta=0.3, `_absorb_tiny_clusters_into_dominant`, `_absorb_one_tiny_satellite`, `_absorb_full_graph_isolates`, the `<= 3`-fragment collapse) are **retained**; deletion is still BLOCKED pending the fuller scenario suite (nested spheres, linked tori, manifold zoo) under `persist+dm` with stand-ins ablated. Only then: make `persist+dm` default, re-scope the single-scale tests (`test_circle_clustering_produces_one_cluster` ==1, `test_swiss_roll_stage1_diagnostics_at_tau_star` ≤3), and delete the stand-ins.
    - *Cross-family audit (gpt-5.4-high, agent 0419cc8a) reconciled.* Region-level BF confirmed exact (= `evaluate_edit` for K=2). Fixes applied: (a) `recursion.py` child configs now propagate `require_dm_split`/`dm_cluster` (were dropped below root — this is why dm-alone leaf counts rose after the fix); (b) `dm_gated_merge` now holds the outcome space **fixed** (flow matrix computed once over AP fragments, rows pooled on merge, columns never contract) so the pairwise homogeneity BF equals the exact `F_DM` partition-edit delta (locked by `test_dm_merge_pairwise_equals_exact_fixed_outcome_delta`); (c) merge adjacency now uses the full shadow+lifted graph, matching the tiers scored by `block_flow_matrix`. Residual design ambiguity noted: the block-level accept gate (`dm_partition_verdict`, J=K, exact homogeneity) and the fragment-level fixed-outcome merge use different (but each internally exact) outcome resolutions — acceptable since the merge is proposal-path with the verdict + persistence as acceptance-path backstops.
    - *FINDING (fuller scenario suite, turn 25): persist+dm does NOT generalize to disconnected multi-component scenes, and the bottleneck is UPSTREAM of the acceptance gate.* Recursion leaves on the fuller suite (`/tmp/dm_validate_fuller.py`): nested_spheres (gt cc=2) → default **1**, persist 7, dm 66, **persist+dm 12**; linked_tori (gt cc=2) → default 1, persist 1, dm 68, **persist+dm 1**; manifold_zoo (gt cc=1 / 4 patches) → default 1, persist 1, dm 76, **persist+dm 1**. No path recovers the ground-truth component count. Direct scale probes (`/tmp/dm_probe.py`, `/tmp/dm_probe2.py`) show why: the `L=1` load-crossover picks **tau\*=0.81** (spheres) / **0.50** (tori), where single-scale connectivity clustering already returns **K=1** (the whole scene is one cluster); the two components only separate at **tau≈0.004** (spheres) / **≈0.006** (tori), ~80× finer than tau\* *and* ~80× finer than the ground-truth `expected_tau` (0.31 / 0.48, which themselves give K=1). So the root region is declared single-cluster before any gate runs, and recursion terminates — the DM/persistence gates are handed a scaffold whose structure the selected scale has already dissolved. This is a scale-selection defect, filed as **#44**, not a defect in the DM reduction. `persist+dm` remains validated on the canonical suite (circle=1, hierarchy=6, swiss=1); deletion of the S2.6.1 stand-ins stays **BLOCKED** — now gated on #44, since the fuller suite cannot fairly test a stand-in replacement while its structure never reaches the clustering stage.
- **Paper/SI prose** should describe the implemented AP -> Q-merge -> refine pipeline (the Leiden detour is obsolete). SI S2.6.1/S2.6.2 now document the scope, the persistence signal, and its warm-start limitation; paper §3 prose still needs a one-line pointer to S2.6.2 persistence as the cluster-count arbiter.

## 28. Scale selection: remaining calibration and cleanup

The load-band heuristic and most of the original exit criterion are resolved (see log).
The default selector is now the variance-load `L = 1` up-crossing
(`controller._select_load_crossover`), which carries no `band_lo` / one-step-coarser
constant and lands tau* within one grid step of geometric truth (circle 8.0x -> 1.6x,
swiss 3.9x -> 0.9x). Scale-search test tolerance tightened `10x -> 3x` (plus a swiss-roll
analog); SI S2.5.1 and the S14.3 table rewritten to match. The legacy load-band selector **has been deleted** from `controller.py` (unknown
`selector` values raise; suite is `load_crossover` / `persistence` only). Paper §3
points to SI S2.6.2 persistence as the cluster-count arbiter. Q-partition persistence
(`selector="persistence"`, `stage1/persistence.py`) remains the structural arbiter for
recursion timing (`P_persist=2`, `theta_ovl=0.5`, SI S2.6.2).

**Finding (cross-family audited, cold-start validated):** the proposed *primary* signal —
knees/plateaus in the compensated node count `N(tau) * tau^{d/2}` (equivalently `V_C(tau)`)
— is **not usable as an operational selector**. Warm-started it is path-dependent and its
"peak" tracks the node budget `N_max`; cold-started with a high cap it is noisy (node count
even goes non-monotone) and its log-log slope never settles at the theoretical `d/2`. This
compounds the earlier self-normalization diagnosis (the raw Lindeberg response is flat at
equilibrium). The knee proposal is therefore demoted to a diagnostic; persistence, not the
compensated count, is the structural signal, and `L = 1` fixes each feature's resolution.

Remaining work:
- **Persistence tau* is coarse-end (hybrid prototyped, default off; diagnose closed).**
  Flag `PersistenceConfig.resolve_within_interval` (`"none"` | `"load_crossover"`,
  default `"none"`) is wired. A6-T16..T18: within-block loads on hierarchy persistent
  subgrid are all >1 → LC picks coarsest stabilized `tau*=0.199` (~9×
  `expected_tau=0.0225`); root cause is category mismatch (`expected_tau` = fine-leaf
  packing vs persistence coarse 3-way / `fine_cluster_tau=0.36`). Circle/swiss: hybrid
  is a no-op (no persist split; LC fallback already near expected). Fine-end-of-block
  rejected as anti-SI. Paper §scale synced to L=1 + coarse-end; hierarchy hybrid≫expected
  regression locked. Do **not** flip the default; persistence stays structural arbiter
  with coarse-end resolution until a SI-justified within-interval signal exists.
- **Landed (A3-T31 SI A+C):** SI S2.6.2 + S14.3 document
  `PersistenceConfig.resolve_within_interval` (`none` | `load_crossover`, default
  `none`; hybrid ≤ fine-leaf).
- **Landed (A6-T28..T30 experimental):** `resolve_within_interval="mid_interval"`
  midpoint probe (default still `"none"`); `load_band` deprecated alias →
  `load_crossover` + DeprecationWarning; paper §scale notes experimental
  mid-interval / hybrid default-off. Do **not** flip default.
- **Landed (A3-T39 SI):** S2.6.2/S14.3 `mid_interval` experimental row.
- **Landed (A6-T31..T33 experimental):** `fine_end_of_block` (default still
  `"none"`); Phi table on hierarchy: `none`~`fine_cluster`, `load_crossover`~16×
  E[τ], `mid_interval`~2.7×, `fine_end` undershoots (~0.25×).
- **Landed (A6-T34..T36 + A3-T44 SI):** `three_quarter_interval` experimental
  (default `"none"`); Phi: three_quarter ~0.82× E[τ] closest probe but slight
  undershoot; mid≤3/4≤fine ordering locked; paper + SI S2.6.2/S14.3 rows.
- **Landed (A6-T37..T39):** circle/swiss Phi tables — no persist split ⇒ all
  within-interval modes identical (LC fallback); experimental
  `three_quarter_load_screened` (reject if load≪1; default `"none"`) — on
  hierarchy 3/4 load≫1 so screened==raw (undershoot is not a low-load
  artifact). Paper notes ~0.82× closest. Do **not** flip default.
- **Landed (A6-T40..T42 + A3-T47 SI):** `mid_interval_load_screened` + shared
  `_WITHIN_INTERVAL_LOAD_SCREEN_MIN=0.5`; hierarchy mid/3q screened==raw
  (load≫1). SI S2.6.2/S14.3 rows present.
- **Landed (A6-T43..T45):** `two_thirds_interval` + `two_thirds_load_screened`
  (default `"none"`); hierarchy Phi seed0: mid~2.69× / two_thirds~1.49× /
  3q~0.82× (still closest) / fine~0.25×; screened==raw. Paper lists ratios.
  Remaining: A3 SI rows for two_thirds modes; optional denser-grid /
  load-weighted within-block pick. Do **not** flip default.

## 44. Recursion terminates at a single coarse feature instead of descending to finer scales

Surfaced by the M1-Part-B fuller-suite validation (#27). **The coarse `tau*` is
not the bug** — disconnected components *should* unify at a coarse root scale;
the defect is that recursion treated `K=1`-at-`tau*` as terminal and never
re-searched *finer* scales inside a single feature.

- **Evidence:** nested_spheres shells separate only near `tau≈0.004` (vs
  `tau*≈0.81`); linked_tori near `tau≈0.006` (vs `tau*≈0.50`).
- **Design settled (A2):** (a) geometric multi-step cap strictly below parent
  `tau*` (`finer_tau_cap_ratio`); (b) one finer walk per region bounded by
  `max_finer_scale_steps` + gate reject + `min_samples`/`max_depth`; (c)
  `prefer_disconnected_prepass` major-lifted-component short-circuit. Acceptance
  gate owns stop/descent; pair with `require_persistent_split` so uniform manifolds
  do not shatter (flag alone ~circle 21 leaves; persist+flag+steps≤4+min_samples=80
  → circle=1 / swiss=1).
- **Landed (flag-gated, default off):** `RecursionConfig.allow_finer_research` +
  `prefer_disconnected_prepass` / `finer_prepass_min_frac=0.2` in
  `stage1/recursion.py`; unit coverage in `tests/stage1/test_recursion.py`.
  RecursionConfig docstring documents the recommended persist pairing.
- **FINDING (A2-T4):** under recommended pairing
  (persist+allow_finer_research+steps≤4+min_samples), nested_spheres and
  linked_tori still yield **1 leaf** — aspiration not recovered; do not flip
  awaiting tests.
- **FINDING (A2-T7..T9 pairing studies):** with unit-test harness `n_seeds=8`,
  `persist=True` keeps circle = 1 leaf across prepass on/off, `finer_prepass_min_frac`
  ∈ {0.15,0.2,0.3,0.4}, and `max_finer_scale_steps` ∈ {4,8,12}; dropping persist
  false-hits (~16–21 leaves) even with prepass. On nested_spheres, none of
  persist±prepass±`require_dm_split` recovered gt cc=2 with ARI>0.5 (steps≤8 → 1
  leaf; deeper / dm-without-persist → 5–9 leaves, ARI≲0.09). Blockers: lifted-CC
  prepass misses concentric shells (same radius-connected graph at recurse caps);
  persist rejects shell-scale splits; dm over-clusters. Need radius-aware /
  signal-band / tissue-filtered split — not more pairing knobs. SI A+C still held.
- **FINDING (A2 radial-gap):** lifted graph **radius-bridges** shells (usually
  `n_cc=1`; rare splits are noise fragments). Flag-gated
  `prefer_radial_gap_prepass` (default off) recovers shell membership on clean
  unit scaffolds / GT-signal-filtered radial gap (ARI_shell=1.0), but e2e
  tissue-filled nested_spheres still unrecovered (persist+radial+steps≤4 →
  circle=1 nested=1; deeper shatters circle).
- **Landed (A2, integrator-hardened):** `prefer_radial_band_prepass` (default
  off) — histogram-trough / contiguous peak-support mask before radial gap,
  with Q-maximizing mid-band assignment; unit fixture with mid-band bridges
  green.
- **FINDING (A2-T10..T12):** band e2e still unrecovered — persist+band
  `steps<=8` → 1 leaf; deeper over-fragments (not shells). Hold awaiting + SI A+C.
- **Landed (A2-T13/T14/T15):** `prefer_noncentroid_radial_band_prepass`
  (default off) + `finer_radial_min_trough_rel` (noncentroid default 0.35);
  `prefer_signal_density_band_prepass` (default off) with knn×radial-hist keep
  (`finer_signal_density_keep_frac=0.55`). Units green (integrator-hardened
  plain-fail/dens-recover + mean-fail/median-recover contrasts).
- **FINDING (A2-T15 RECOVERY):** under unit harness (`n_per_sphere=64`,
  `n_seeds=8`, persist, `steps<=12`), `prefer_signal_density_band_prepass`
  recovers nested shells **2 leaves / ARI=1.0**; circle stays 1 leaf at
  `steps<=8`. Band/noncentroid alone still unrecovered. Hold awaiting flip
  until linked_tori/swiss guards confirmed.
- **Landed (A3-T28 SI A+C):** SI S2.6.2 + S14.3 document `allow_finer_research`
  and radial-band / noncentroid / signal-density flags (proposal-path, default
  off; cites unit nested ARI=1.0). Score is `rho_knn * rho_radial` (A2 restored
  multiply after divide regression).
- **FINDING (A2-T18..T20 guards):** linked_tori under persist+signal_density
  stays **1 leaf** (radial origin unsuitable for offset rings; deeper steps
  over-fragment, ARI≈0). Swiss: steps≤4 → 1 leaf; steps=8 **shatters** (~17
  leaves) — keep recommended pairing `max_finer_scale_steps≤4`.
  `finer_signal_density_keep_frac=0.55` confirmed sweet spot (no default change).
  **Do not flip awaiting** (tori unrecovered).
- **Landed (A2-T21..T23):** `prefer_pca_axis_gap_prepass` (default off) +
  centroid-separation gate; unit offset rings recover ARI=1.0; concentric
  rejected. Linked_tori e2e under persist+pca(+sd) steps 4/8/12 still **1 leaf**.
  Docstring documents recommended pairing (uniforms steps≤4; nested sd+steps≥8;
  tori PCA prototype).
- **Landed (A3-T37 SI A+C):** SI S2.6.2 + S14.3 document
  `prefer_pca_axis_gap_prepass` (proposal-path, default off; interlocking
  unrecovered).
- **Landed (A2-T24..T26):** `prefer_tube_major_radius_prepass` (Hopf tube
  residual, default off) — unit interlocking rings ARI=1.0; concentric rejected.
  `prefer_spectral_gap_prepass` (Fiedler/lifted+kNN, default off) — unit offset
  rings ok; nested spectral steps=8 shatters (~8 leaves). Linked_tori e2e under
  persist+tube/pca/spectral/sd+pca+tube steps 4/8/12 still **1 leaf**. Harness
  guards circle/swiss persist+sd+pca steps=4 → 1. Recursion tests 18 passed.
  **Do not flip awaiting** (tori unrecovered).
- **DIRECTIVE (human, 2026-08-09): reframe — stop geometry-specific prepasses.**
  Disconnection is a scale-free topological property; tau descent + per-geometry
  coordinate cues (radial/PCA/tube/spectral/linking) is the wrong hypothesis
  family (probe evidence: same 64 nodes at tau=0.27 and tau=0.004 for
  nested_spheres — descent buys no resolution). Replace with **hollow-edge
  (empty-region) evidence**: data-side mid-segment occupancy test per lifted
  edge (Gabriel/lens ratio `H = n_mid/n_end`, Poisson null), cut hollow edges
  before clustering at the region's own tau*. Coordinate-free; predicted to
  cover nested/tori/zoo/swiss in one statistic. Full derivation, literature
  (Gabriel 1969, Toussaint 1980, Chaudhuri–Dasgupta 2010, ToMATo), and ordered
  experiment protocol:
  `docs/Proteus/paper_1_foundational/reference/empty_region_evidence_and_scale.md`.
- **Landed (A2-T27..T29 + A3-T38 + A4-T18):** frozen-scaffold probe +
  `stage1/edge_evidence.py` + `prefer_hollow_edge_prepass` (default off; Q on
  pruned edges) + SI S2.6.1/S2.6.2/S14.3 hollow prose + adversarial-null ROC
  harness (`hollow_edge_nulls` / `test_hollow_edge_roc`).
- **FINDING (A2-T27 probe, loud):** note `L/4` mid-ball alone mass-false-hollows
  (`n_end~0` ⇒ `H=0`, prune shatters). Operational hit cfg `mid_frac=0.35`,
  `h0=0.35`, `min_end=0.5` + Gabriel-empty fallback recovers nested+tori majors=2
  at seed0 but is **multi-seed fragile**. Cross-tori interlocking H50 med~1;
  oracle cut-label-cross still 1 CC via tissue. Zoo/swiss/circle stay 1 under
  hit cfg. Theory direction supported; **universal `h_0` not calibrated**.
- **FINDING (A2-T29/T30 LOUD):** persist+hollow e2e unrecovered. Fixed-τ
  majors=2 at nested@0.27 / tori@0.5 is **not recovery** — sample ARI~chance.
  `mid_frac=0.35` is empty-ball (H nondiscriminative); at `0.5` H separates
  but lifted prune is not a cut-set. Default `H|Gabriel` drives spurious K=2
  via Gabriel at low `n_end`. **Never treat major-CC count as recovery.**
- **Landed (A2-T30..T32):** multi-τ prune→CC harness; `require_gabriel_and_h`
  + `hollow_require_persistent_agree` (both default off). Conjunction
  suppresses probe K=2 seed-stable; conj+agree keeps uniforms/nested/tori at
  1 leaf under harness. Raising `min_end` alone *increases* Gabriel usage —
  prefer conjunction or `gabriel_fallback=False` + calibrated h0/mid.
- **Landed (A4-T24 ROC export):** `recommend_hollow_edge_configs` primary
  **mid=0.5, h0=0.7, gabriel=False, min_end=0.5** (sheet FPR=0, TPR=0.9,
  q01≈0.82, AUC≈0.999); A2 `(0.35,0.35)` kept as alt. Sheet-null safety ≠
  nested ARI recovery.
- **Landed (A3-T45/T48 SI):** ROC mid_frac table + A4 primary +
  `require_gabriel_and_h` prose (proposed; no default flip).
- **Landed (A2-T33..T35 + A4-T27 multi-τ ROC):** `a4_roc_primary_config` /
  `hollow_use_a4_primary` (def off); `mst_critical_only` (def off); sample-ARI
  harness (K=2≠recovery). Uniforms/zoo ok; **nested/tori unrecovered**.
  Multi-τ ROC: primary dens1 sheet-safe; mid>0.5 TPR collapses; thinning
  raises FPR.
- **Landed (A2 bridge follow-on):** `bridge_critical_only` /
  `hollow_bridge_critical_only` (def off) — true cut-set beyond MST;
  nested@0.27 majors≤1; multi-seed A4+bridge ARI still unrecovered.
- **Remaining:** soft-capacity / denser-scaffold hollow ARI; Poisson-null
  h0 calibration export; fuller suite green with **sample-ARI** → retire
  radial/PCA family + awaiting-flip review (A1 sign-off). **Do not flip
  awaiting.** Distinct from #28. Post-track: open #45 open-loop /
  `max_nodes` (`reference/open_loop_growth_and_node_cap.md`) into M4.

## 41. Stage 2 topology recovery: persistent-homology Betti validation on fitted regions

Flag-complex *construction* has landed (`stage2/flag_complex.py`, SI S4.1/S4.2). Recovery
assertions that validate learned objects against ground-truth Betti numbers remain
`@awaiting`: `test_nested_spheres_topology`, `test_linked_tori_betti_numbers`, and the
circle `b1 = 1` target of #25.

- **Canonical tool:** Vietoris--Rips PH on node positions (SI S14.2;
  `tests/metrics/persistent_homology.py`). Sparse lifted-graph flag complexes are *not*
  the right input — band holes are essential (#25).
- **Landed (A4 harness):** `per_region_topology` / lifetime helpers with
  `FILTRATION_MULT=1.5` (SI S14.2) and `DEFAULT_LIFETIME_FRAC=0.5` (proposal-path
  operational; now logged in SI S14.2/S14.3);
  `tests/scenarios/synthetic/test_ph_harness_scaffold.py` clean-geometry smokes green;
  recovery xfails unchanged.
- **FINDING (A4 diagnostics):** on tissue-polluted whole clouds (`tissue_fraction~0.2`),
  fixed_threshold over-reports `b1` and lifetime alone inflates `b0` without restoring
  clean Betti; **signal-label / per-region filtering is load-bearing** and restores
  `(b0,b1)=(1,1)` under both readings. Clean torus grid can recover `b1=2` at
  `lifetime_frac=0.5`. Do not flip awaiting recovery tests yet.
- **Landed (helper):** `topology_from_accepted_regions` feeds accepted-region node
  positions into `per_region_topology` (recovery tests still awaiting).
- **FINDING (A4 fitted-circle probe):** on `scaffold_at_star`, SI `1.5*sigma*`
  yields `b1=0` (whole and accepted-region); lifetime inflates `b0` still with
  `b1=0`. NN signal-label filter recovers `b1=1` only near `~8*sigma*` (existence
  proof, not SI default). Do not flip awaiting tests.
- **Landed (SI log):** S14.2 lifetime-reading clause + S14.3 `lifetime_frac=0.5`
  operational row (proposal-path; tissue/per-region caveat; no fitted-circle recovery
  claim).
- **Landed (A4-T8 stepping stone):** Fibonacci nested-sphere clean-shell PH via
  `topology_from_accepted_regions` + signal filter green (`test_ph_nested_spheres_clean_shells.py`);
  fixed_threshold per shell `(1,0,1)`; lifetime needs `frac≈0.75` at modest n;
  tissue whole-cloud polluted, `include_labels` restores. Prefer Fibonacci S2
  (lat/lon grids birth spurious `b1`). Recovery awaiting unchanged.
- **FINDING (A4-T9 calib):** fitted-circle signal-filter (`seed=21`):
  `fixed_threshold` recovers `(1,1)` at min `filtration_mult=6` (window ~[6,10],
  fills by 12); `lifetime` needs `filtration_mult≥6` **and** `lifetime_frac≥4`
  (default 0.5 leaves `b0≫1`). SI `(1.5, 0.5)` still fails — do not flip defaults
  or awaiting. Probe: `test_ph_fitted_circle_calibration.py`.
- **Landed (A4-T10):** `nearest_data_labels` NN helper in
  `tests/metrics/persistent_homology.py`; fitted-circle probes refactored.
- **FINDING (A4-T12 reading path):** keep acceptance = SI S14.2 fixed_threshold
  at `1.5σ*`; fitted-circle `b1=0` is loop-unborn (coverage/scale), not a license
  to raise `filtration_mult`. Prefer denser accepted-region coverage so true H1
  births ≤`1.5σ*`. Optional fallback: declared calibration protocol → S14.3 log —
  do **not** silently adopt mult=6 / frac≥4. Lifetime stays proposal-path;
  clean-shell modest-n window frac≥~0.75 (A4-T13 sweep).
- **Landed (A3-T35 SI draft):** S14.2 proposed reading-path (coverage-first;
  calibrated `filtration_mult` fallback labeled proposed) + S14.3 rows.
- **Landed (A4-T15..T17):** `sweep_lifetime_frac` harness; `run_per_region_ph`
  prototype (nested clean shells / linked_tori grids); denser-coverage probe
  recovers SI `1.5σ*` on clean circles (prefer coverage over raising mult).
  PH synthetic 32/32 green. Recovery `@awaiting` unchanged.
- **Landed (A4-T19..T20):** denser fitted `max_nodes≥128` recovers SI `1.5σ*`
  on circle `scaffold_at_star` (prefer over mult=6); `run_per_region_ph` +
  diagnostics wired into nested/tori `@awaiting` scaffolding (clean harness
  green; fitted still xfail).
- **FINDING (A4 nested fitted denser):** max_nodes 64/128/256 raises n_sig
  but SI `(1,0,1)` **not** recovered — betti worsens (spurious b1). Denser
  alone insufficient.
- **FINDING (A4-T25 recipes):** nested max_nodes=128 signal+lifetime
  frac∈{0.25..4} + hollow-prune(mid0.5/h0.7) — SI `(1,0,1)` still not
  recovered; hollow kept all signal nodes; shell1 spurious b1 persists.
- **FINDING (A4-T26 tori denser):** linked_tori max_nodes 64/128/256
  (labels 0/1) — SI `(1,2,1)` not recovered (closer on b1 only). Keep
  `@awaiting`.
- **FINDING (A4-T28..T32):** tori lifetime+hollow recipes fail SI; nested
  dual-scale coarse=3 recovers shell1 only; tori dual-scale fails both
  scales; per-shell local σ no outer gain; circle cal mult=6 recovers
  nested shell2 only (inverse of coarse=3). No single global mult hits
  both shells. Keep `@awaiting`.
- **Remaining before flipping recovery tests:** per-shell mult schedule;
  tori local-σ; hollow+cal-mult combo; keep recovery `@awaiting` until
  SI-default fitted evidence is green.
- **Dependency note:** heterogeneous per-patch simplex *dimension* (manifold-zoo S4.2)
  still blocks on #40; pure topology (b-numbers) does not.

## 43. Evidence gate: wire the affected dual-subgraph connectivity check (SI S10.4)

- S10.4's dynamic-preservation rule requires an edit to be *evidence-bearing* only if
  (a) every affected post-edit star is well-conditioned **and** (b) the affected dual
  subgraph stays connected. `evidence/gate.py::score_edit` enforces (a) all-or-nothing and
  exposes a `dual_connected` hook for (b).
- **Landed (stub):** `affected_dual_subgraph_connected` in `gate.py` is the pure BFS
  induced-subgraph hook; `tests/evidence/test_dual_subgraph_connectivity.py` locks
  disconnect ⇒ evidence-path reject. When adjacency is `None`, the helper conservatively
  returns `True` (same default as `score_edit(..., dual_connected=True)`).
- **Landed (A5-T31..T33 experimental):** `stage2/dual_flow.py` builds facet-sharing
  `DualAdjacency` behind `DualFlowConfig.enable_dual_adjacency` (default off);
  `GateConfig.apply_dual_adjacency` wires into `score_edit`/`evaluate`
  (disconnect ⇒ reject). Wiring test flipped green; evidence subset 10 passed.
  A3-T36 drafted SI S6.6 DualAdjacency stub. Mass/density/benchmark remain
  `@awaiting("stage2.dual_flow")`.
- **Landed (A5-T34..T36 + A3-T40):** `dry_run_dual_from_edit` helper; experimental
  `ConservativeBPResult` / `enable_conservative_bp` sketch (not real loopy BP);
  expanded synthetic dual graphs; SI S6.6 expanded to match producer + gate flag.
  Evidence subset 20 passed.
- **Landed (A5-EXP-S61 + A3-T46 SI):** `accumulate_face_pressure_tally` (S6.1)
  + `classify_boundary_facets` (S6.3) behind flags (default off); SI stubs
  match.
- **Landed (A5-T40..T42 + A3-T49 SI):** dry-run `face_tallies` demo via
  `samples=`; `simplex_local_density` S6.4 sketch (`enable_simplex_density`,
  default off); acceptance-path plan docstring. SI S6.4 stub present.
- **Landed (A5-T43..T45 + A3 seam SI):** live BMU face-tally harness;
  `build_divergence_stencil` / `solve_as_message_pass` (A_S residual sketch);
  `stitch_orientation_seam_pressures` + `apply_ghost_reservoir`
  (`enable_seam_ghost`, default off). Evidence 36→42 with μ/ε follow-ons.
- **Landed (A5-EXP-mu/flux):** whitened λ_f / μ_S soft solve + `epsilon_flux`
  / spectrum step-shrink (flags off). Gaps remain: Stage-1 BMU wiring;
  real loopy BP; face-registry seam/ghost; live S6.4 density.
  Mass/density/benchmark stay `@awaiting`. **Do not close #43** until
  acceptance-path default replaces the conservative open default / fuller S6.
