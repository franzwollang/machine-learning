# Proteus Paper 1 — Open Issues

Current, active issues only — resolution history lives in `OPEN_ISSUES_LOG.jsonl`, never
here. Numbering is historical and stable: resolved issues are deleted rather than
renumbered, so gaps in the sequence are expected. Each entry lists only the work that
actually remains. See `PLANNING.md` for the suggested order of attack.

Next issue number: 49

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
- **Hierarchy seed-0 5/6 residual:** A6-T16 confirms that the accepted split is
  the true L0/L1 cut (ARI 1), but its `N` and `phi` are path-dependent: the
  first accept shifted from 46 to 45 nodes, and `phi` changed from 0.197 to
  0.061 between caps 46 and 64. A6-T17 is canceled. Do not propose a cap rule
  or treat 45/46 or the observed `phi` values as thresholds. A6-T19/T20/T21
  landed the SI S2.5.1 retraction and paper audit: the first accept is not
  stable, density is not a remaining floor, and no usable cap number follows.
  Persistence, within-interval, and scale-search defaults stay unchanged.
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
- **Landed (A6-T43..T45 + A3-T50 SI):** `two_thirds_interval` +
  `two_thirds_load_screened` (default `"none"`); hierarchy Phi seed0: mid~2.69× /
  two_thirds~1.49× / 3q~0.82× (still closest on **standard** grid) / fine~0.25×;
  screened==raw. SI S2.6.2/S14.3 rows present.
- **Landed (A6-T46..T49 + A3-EXP-si53/T55 SI):** experimental
  `ScaleSearchConfig.halve_grid_steps` (half log-step) +
  `resolve_within_interval="load_weighted_interval"` (argmin `|log L|` among
  `L≥0.5`). FINDING: densify **flips** ranking on seed-0 (dense two_thirds~1.00×
  beats 3q~0.76× — quantization); seed-4 dense **rejects** persistence (LC
  fallback); seeds 1–2 never accept a multi-cluster split; load_weighted
  systematically reproduces coarse-end on hierarchy (`L(i_lo)~0.6–0.7`). Do
  **not** flip default.
- **Landed (A6-T50/T51 + A3-T55):** multi-seed Phi hierarchy export
  (seeds 0..4; std+dense) + `load_weighted×halve_grid` combo probe; SI notes
  densify seed-fragility. Paper pins load_weighted≡coarse. Do **not** flip
  default.
- **Landed (A6-T53/T54):** circle/swiss under `halve_grid_steps` stay
  LC-fallback identity across within-interval modes (densify moves LC peak
  only). Seed-4 densified persist-reject is mechanical under coarse-anchored:
  first half-step neighbor Jaccard drops below `overlap_threshold`
  (`0.39 < 0.5`) so `run_lengths[0]=1`. Do **not** flip default.
- **Landed (A6-T55..T57):** seed3 short persist-block (`len=3`) forces
  mid≡two_thirds≡three_quarter (~8.83×); seed4 Jaccard half-step export
  table for SI; circle densify×`load_weighted` stays LC identity. Do **not**
  flip default.
- **Landed (A6-T58..T60):** experimental `densify_overlap_recover` /
  lower-threshold floor `0.35` recovers seed4 densified `run0=16` but
  collaterally flips seed1 — **keep default none**. Paper pins seed3
  short-block mechanism; multi-seed densify Jaccard first-step table
  seeds0..4 locked (accept both `{0,3}`, std-only `{4}`, reject both
  `{1,2}` + dense `{4}`). Do **not** flip default.
- **Landed (A6-T61..T66):** densify-recover collateral map flips seed1
  std+dense + seed4 dense; seed2 stays reject; seed3 std `run0` 3→5;
  accept `{0,1,3,4}` under thr0.35. Seed3 densify restores seed0
  fractional landing (mid~2.30× / 2/3~1.00× / 3q~0.76×); std short-block
  mid≡2/3≡3q@8.83× is quantization. thr sensitivity: `0.35` is the narrow
  band; `0.30` over-accepts densified seed2; `0.40` loses seed1/seed4
  dense recovers. densify×`load_weighted` on seed3 stays coarse.
  `densify_overlap_recover_threshold` probe override (default none).
  Paper pins. Do **not** flip default.
- **Landed (A6-T67..T69):** thr×Phi export — threshold is Jaccard
  accept/reject gate only; shared densified accepts keep `Phi_C`
  identical across `0.30/0.35/0.40`; seed2 Phi row only at `0.30`.
  densify×LW seed0/4: LW stays coarse 16× alias when accept; seed0
  recover-invariant; densified seed4 needs recover to match seed0
  hierarchy.   Formal densify×LW×thr combo + thr0.30 seed2 ov0 pin +
  paper pins. Keep `densify_overlap_recover` default none. Do **not**
  flip default.
- **Landed (A6-T67..T72 formal+followon):** densify×LW×recover-thr combo —
  LW≡coarse 16× on accepted cells **except** thr0.30 densified seed2 where
  LW picks idx1 (~12×) vs none idx0 (first LW≠coarse under recover-thr);
  mechanism is closest-to-unit load (`L0≈0.61` vs `L1≈1.56`). densified
  seed2 `ov0≈0.340` only accepts at thr0.30. seed1 densify×LW stays
  coarse across thr; thr0.40 dense reject `ov0≈0.364`. Paper pins. Keep
  default none. Do **not** flip default.
- **Landed (A6-T73..T77):** thr0.30 densified load-vector export — only
  seed2 flips `|log L1|<|log L0|` (LW=1); seeds 0/1/3/4 LW≡coarse.
  Fractional densify hierarchy mid~2.30 / 2/3~1.00 / 3/4~0.76 / fine~0.25
  vs LW one-step ~12.1×. Phi at LW idx1 **rises** vs coarse and mid (not
  Phi-descent); load-screened mid/2/3/3q ≡ raw. In-block argmax Phi lands
  at unstabilized idx1 ≡ LW; `load_crossover` hybrid stays coarse (stab
  filter skips idx1). Keep default none. Do **not** flip default.
- **Landed (A6-T78..T80):** thr0.30 densified multi-seed — in-block
  argmax Phi = idx1 on **every** accept (seeds0–4), but LW≡Phi-peak
  **only seed2** (seeds0/1/3/4 keep LW≡coarse≠peak); LC hybrid≡coarse
  on all five. Seed2 LC-eligible idx2 has Phi2/Phi1≈0.78 (near-peak,
  Phi2≫Phi0) yet LC stays coarse because `|L0−1|<|L2−1|` after stab
  skips the peak. Paper pins. Keep `densify_overlap_recover` default
  none. Do **not** flip default.
- **Landed (A6-T81..T86):** thr0.30 densify seeds0–4 first-stab-after-peak
  =idx2 always (Phi(fsa)/Phi(peak)≈0.95/0.85/0.78/0.93/0.90); stab-only
  Phi-argmax **never** equals LW (sArg=1≠0 seeds0/1/3/4; sArg=2≠1 seed2).
  Post-peak Phi decay +1..+4 monotonic (seed2 deepest
  0.78/0.62/0.47/0.33); stab-skip×thr 0.30/0.35/0.40 — thr gates accept
  only, LC≡coarse always, near-peak skip only seed2@0.30. Paper pins.
  Keep `densify_overlap_recover` default none. Do **not** flip default.
- **Landed (A6-T87..T89):** thr0.30 densify Phi decay-to-0.5 half-life
  idx 5/5/4/6/5 (frac~3.91/3.49/2.77/4.04/3.72; seed2 fastest);
  `|L0−1|≪|Lfsa−1|` all accepts (LC≡coarse); paper pins. Keep
  `densify_overlap_recover` default none. Do **not** flip default.
- **Landed (A6-T90..T95):** half-life thr-invariant across
  `0.30/0.35/0.40` on shared accepts (floor=Jaccard gate only);
  `|Lpeak−1|>|L0−1|` all seeds; only seed2 `|log Lpeak|<|log L0|`
  ⇒ sole LW≠coarse. half-life×`halve_grid`: std collapses to peak+1
  (frac≲1, `tau_r≈0.55`; seed2 reject) vs densify multi-step T87
  pins + deeper `tau_r`. Near-peak stab-skip ≡ seed2 ≡ fastest
  half-life. Paper pins. Keep default none. Do **not** flip default.
- **Landed (A6-T96..T102):** half-life×halve×thr floors — std accept
  `{0,1,3,4}` thr-invariant; densify keeps T64; seed2 unique
  near-peak∧`|log L|`favors peak∧LW≠coarse. thr0.30 densify half-life
  uniquely closest to **mid** (gaps 2/2/3/1/2; half always coarser
  than mid; densify-flip `2/3` farther) — proximity ≠ fine-leaf
  fraction. Circle/swiss half-life without persist (circle 4/6→8/12,
  swiss 4/5→8/10 under densify). Paper pins. Keep default none.
  Do **not** flip default.
- **Landed (A6-T103..T105):** thr0.30 densify half-life×LW — d_LW
  5/5/3/6/5 vs d_mid 2/2/3/1/2 (LW farther except seed2 tie); half-life
  proximity ≠ LW preference. Seed3 std short-block: mid≡tt≡tq@1
  (~8.83×) but half≡fine@2 (~4.88×) ≢ mid; densify half@6/mid@7. Paper
  pins T99..T104 + circle/swiss. Keep `densify_overlap_recover` default
  none. Do **not** flip default.
- **Landed (A6-T106..T108):** thr0.30 densify half-life×LC-hybrid —
  LC≡coarse on all accepts; d_LC 5/5/4/6/5 always > d_mid; seed2
  d_LC=d_LW+1 (LC stays 0 while LW=1) — half-life proximity ≠ LC
  preference. Std multi-seed half≡peak+1 thr-invariant on accepts
  `{0,1,3,4}` across `{0.30,0.35,0.40}`; seed2 reject. Paper pins
  T103..T107. Keep default none. Do **not** flip default.
- **Landed (A6-T109..T111):** thr0.30 densify half×stab-only Phi-argmax
  — mid closer on seeds 0/1/3/4 (d 2/2/1/2 vs sArg 4/4/5/4); only seed2
  d_sArg=2<d_mid=3 — half-life ≠ sArg preference. LC straddle: fine
  always closer to half than coarse, yet `|L-1|` keeps LC≡0; fine beats
  mid only on seed2. Paper pins T109/T110. Keep default none. Do **not**
  flip default.

## 44. Stage-1 separation: density level-set clustering layer (validated pivot)

Original defect: recursion treats `K=1` at the coarse `tau*` as terminal, so
multi-component scenes (nested spheres, linked tori) end as one feature. Two hypothesis
families were run to exhaustion in the 2026-08 swarm burn and are **falsified**:
(1) finer-tau descent + geometry-specific prepasses (radial/PCA/tube/spectral/linking),
and (2) hollow-edge (empty-region) edge statistics. Decisive falsifier: with tissue the
support is *connected* (an oracle cut of every cross-label edge still leaves 1 CC), so
separation is a density-contrast question, not an emptiness question. Experiment history
lives in `OPEN_ISSUES_LOG.jsonl`, SI S2.6.1/S2.6.2 (flags stay proposal-path, default
off), and `reference/burn_2026-08_swarm_retrospective.md`.

**Validated pivot (orchestrator probes, 2026-08-12):** the right object is the Hartigan
density cluster tree — components of upper level sets `{p >= lambda}` with an explicit
background class — estimated by Chaudhuri–Dasgupta robust single linkage; on the
scaffold, per-node density is read from **node spacing** (equalized code: node density is
a monotone transform of `p`; the cluster tree is invariant to monotone transforms, so the
magnification exponent need not be known). Probe:
`code/proteus/tests/scenarios/synthetic/cd_level_set_probe.py`; scoring is signal-only
ARI + background recall (full-cloud ARI is capped by fade-halo labels, see #45).

- Batch C–D oracle on raw samples: circle/swiss sig-ARI **1.0/1.0**; nested spheres
  **0.973**; hierarchical Gaussian **0.99** at K=6 with the coarse K=3 level in the same
  tree (0.57 vs fine labels); zoo single component as expected; gap-corrected uniform
  tori **0.99** across seeds 0–2 (at n_per=8000, k=12).
- Scaffold-native read (fine tau, raised cap, k=8 node-spacing density, C–D linking at
  alpha=1.0, BMU transfer): circle/swiss **1.0/1.0**; hierarchy **0.93–0.99** at K=6;
  corrected tori **0.989/0.986/0.998** (seeds 0–2, 768 nodes); nested spheres
  **0.82/0.92/0.55** (1024 nodes).
- Measured caveats: (a) hit-equalization compresses density contrast (magnification
  `gamma < 1`), so the **node budget sets the finest resolvable separation valley**
  (node `r_k` must sit inside the valley) — nested seed-2 weakness is halo mass stealing
  node resolution; (b) HDBSCAN condensed-tree (eom) extraction collapses on the
  quasi-lattice node set — level-sweep reading works, principled automatic extraction is
  open; (c) the repo linked_tori generator is unseparable by construction (#45), so the
  corrected uniform-sampled tori scene is the fair benchmark.

Remaining work:
- **LANDED (2026-08-13; default off):** `stage1/level_set.py` implements the node-spacing
  C–D tree (`k=8`, `alpha=1`, four-node floor, 120 quantile levels) plus an explicit
  merge DAG (overlap matching, ToMATo survivor), rank persistence, normalized excess
  mass, sequential geometric screens, and a background-aware DM sibling collapse.
  `RecursionConfig.use_level_set_clustering` returns the coarsest
  relative-mass-filtered `K>=2` cut that clears DM, preserves label `-1` as
  terminal `RecursionNode.is_background`, and propagates through finer
  research / child recursion. `dm_partition_background_logbf`
  keeps background as a fixed additional outcome (its unchanged row cancels), with an
  exact-edit unit lock. Legacy Q/AP persistence and no-background DM flags are explicitly
  rejected when combined rather than silently mixing outcome spaces; retired geometry /
  hollow prepass flags are likewise rejected instead of ignored. Defaults unchanged.
- **LANDED (mass-filtered coarse anchor + flow-bottleneck guard, still default
  off; 2026-08-13):** the single candidate is the coarsest level whose cut keeps
  `K>=2` after `min_cluster_frac=0.15` of the region node budget; satellite-only
  levels are skipped, so tissue-bridged balanced mid-tree cuts (nested shells,
  tori) are reachable. Position statistics cannot make that visit safe — rank
  persistence, excess mass, subsample stability, and k-perturbation stability
  were all measured inseparable between circle arc cuts and nested shell cuts —
  so the guard is `max_bottleneck_ratio=0.25`: cross-cut max-flow over the weaker
  block's internal (max-variance bisection) max-flow, on the fitted Hebbian
  flows. Measured on fitted scaffolds: arcs 0.60–1.29, true splits 0.000–0.070
  (~8x gap). DM confirms cuts that pass the guard; failure rejects the region
  outright. Unit-locked with a paired arc-cut/weak-bridge fixture.
- **Measured (2026-08-13, fitted scaffolds, no expected K):**
  `level_set_fitted_sweep.py` passes **25/25** scene-seeds (seeds 0--4):
  circle and swiss-roll connected nulls reject 10/10; hierarchy returns `K=3`
  with ARI 0.568--0.582; repaired canonical linked tori return `K=2` with ARI
  0.981--0.998 and coverage 0.991--0.999; nested spheres return `K=2` with ARI
  0.727--0.914 and coverage 0.671--0.955. Nested requires
  `n_per_sphere=3000`, `max_nodes=1536`; fits stabilize at 1238--1392 nodes.
  The former 1024 cap truncates the read and only recovers 2/5 seeds. kNN-graph
  probes remain smoke checks: position-derived flows cannot emulate fitted-flow
  physics for sampling-gap arcs.
- **LANDED (2026-08-16; still default off):** SI S2.6.2 now states that
  recursion splits Hartigan children on connected supports (uniform-manifold
  null = no valley). Level-set replaces AP/Q/prepasses as the Stage-1
  structural proposal; background-aware DM stays inside extraction;
  persistence may later compose on level-set snapshots (flag exclusions
  remain mixed-outcome-space guards). Valley-resolvability trichotomy
  (`resolved_split` / `resolved_null` / `under_resolved`) is on
  `LevelSetSelection` and unit-locked. Under-resolved capped scaffolds raise
  `max_nodes` at the current `tau` (`grow_nodes_when_underresolved`, factor
  2, at most 5 steps, ceiling `n/2`); a bottleneck-rejected arc cut at the
  cap is `resolved_null` and does not grow. Trichotomy does not skip
  finer-`tau` (composites at coarse `L=1`). Operational stand-in for #47.
- **Measured (2026-09-07, normal path, seed 0 unless noted):**
  `level_set_normal_path_sweep.py` — `load_crossover` +
  `use_level_set_clustering` + `allow_finer_research` + cap-growth
  (lean `max_epochs=12`, 8-point grid, `max_finer_scale_steps=16` to span
  the measured ~80× `tau_sep` gap; production default is 8 steps / 16×).
  Root recovery: hierarchy 5/5 seeds `K=3` ARI 0.568–0.582; linked tori
  s0 `K=2` ARI 1.000 cover 1.000; nested s0 `K=2` ARI 0.728 cover 0.666
  (meets fitted-sweep thresholds). Uniform nulls fail at 16 finer steps
  (circle s0 `rootK=2` / 9 signal leaves; swiss s0 `rootK=2` / 14 leaves)
  because LC on the wide `[1e-5,10]` grid lands circle/swiss at
  `tau*=0.027` and the long finer walk overshoots into sampling-gap arcs.
  Ablation: circle stays 1 leaf with growth-only or `max_finer_scale_steps<=4`.
  Child recursion over-fragments even when the root split is correct
  (tori 21 signal leaves; hierarchy 6). Nested/tori multi-seed not rerun
  (25–30 min/seed).
- **ACCEPTANCE BLOCKER:** awaiting-flip is still blocked. Seed-0
  normal-path with #48 (2026-09-09): circle/swiss 1 leaf; hierarchy
  root K=3 ARI 0.582 and 6 signal leaves (the generator's 3×2 fine
  pairs); linked tori / nested root K=2 with 2 signal leaves.
  Bimodal-circle root K=2 ARI 0.842. Do not flip
  `use_level_set_clustering`. Do not delete S2.6.1 stand-ins.
- Evidence-gated insertion so `max_nodes` becomes a safety assert is #47
  (M4 / `reference/open_loop_growth_and_node_cap.md`).
- Geometry / hollow prepass zoo is **quarantined**
  (`FALSIFIED_PREPASS_FLAGS` in `recursion.py`; SI S2.6.2). Flags stay
  default-off for ROC / adversarial-null calibration. Do not enable them
  on the acceptance path. Awaiting-flip review remains after #48 + #45.

## 45. Synthetic fade-halo labels need benchmark-wide semantics

The repaired linked-tori generator now exposes the underlying remaining
benchmark issue: faded-density generators label all samples below
`lambda=0.5` as background, often about half the cloud. Full-cloud metrics are
therefore dominated by halo labels even when signal topology/clustering is
correct.

Mechanism (2026-09-09): the half-cloud tissue is structural, not a knob.
`FadedMixture.density = Σ w_c [λ_c f_c + (1-λ_c) u]` with `u` the uniform on
the support box, so signal mass ≈ tissue mass ≈ 1 before normalisation
and every faded scene samples 46–49% tissue (circle 0.48, swiss 0.46,
tori 0.49, nested 0.47, bimodal 0.48). `tissue_fraction` only sets the
box padding; metadata reports `requested=0.03, actual=0.46`. The SI
S14 calibration text speaks of "tissue fractions 0/0.05" as if they were
mass fractions — check what those runs actually contained.

Consequence for the reader (nested seed 2 under `track_tau`, tree
composition probe): the root accepts at the coarsest resolving τ
(0.12, N=283); at that τ the outer shell (1722 pts, 4× the inner area)
is only ~3× above `u` while the inner is ~11×, so 57% of the outer
shell's nodes fall below the C-D level and go to background
(cover 0.618). The surviving outer fragment is 60% tissue and then
shatters Hartigan-faithfully (child splits at ρ 0.018, φ = 0) into the
8 signal leaves seen on seeds 0 and 2. Seeds 1, 3, 4 keep the whole
outer shell (cover ≥ 0.886) and give 2 leaves. Not a growth-policy
defect: the accepted cut is right, its background partition of the
lower-density component is crude at the first-accept scale.

Remaining work: weak two-Gaussian shatter is tissue-sufficient (A5-T11):
one-signal+tissue splits 5/10, while both-signals/no-tissue and isolated
`n=194` children stay clean. Tissue mass in the over-splitting child is not the
residual (A4-T13: bimodal ratio 0.91, nested ratio 0.97). The current-tip
census (A5-T16) found nested clean on all five seeds (`sigLeaves=rootK=2`) but
bimodal seed 2 again over-split (`sigLeaves=3>rootK=2`); the effect is
tip-sensitive rather than a standing nested-child failure. A5-T15 therefore
skipped the nested ablation. Only benchmark-semantics documentation remains
this burn. Keep signal-only ARI + background recall. Do not resolve #45, tune
`tissue_mass`, reopen descent A/B/D, or resume tissue-fraction sweeps.

Level-set consequence (2026-09-09, from #48): the root split assigns about
half the tissue to signal children (nested seed-0 bg recall 0.51; nested
child 3 is 51% tissue, bimodal-circle child 3 is 37%). The child reader
then declares most of the region background and accepts small
shell+tissue chunks (nested: 5 signal leaves; bimodal: 4 with the `N≤n/k`
bound). This is the remaining child over-split once the legacy fallthrough
and `τ_shot` are fixed.

Mechanism (`level_set_root_tissue_probe.py`, seed 0): node label `-1` is
exactly C–D inactivity (`core_radius > r`) at the selected level; runt and
relative-mass filters add nothing and DM only accepts/rejects. At the
selected radius (bimodal r=0.63, nested r=0.69) the low-density halo nodes
are active and connected to the shell components — 13/55 and 102/307
signal-labelled nodes have ≥50% tissue catchments, at ~0.64× the hit count
of ≥90%-signal nodes (2528 vs 3959; 619 vs 972). Samples follow BMU Voronoi
(`assign_samples_to_clusters`), so 0% of leaked tissue is nearer a
background node. The reader is Hartigan-faithful: halo belongs to the
coarse cluster. The generator's λ=0.5 fade defines it as background.
The descent decision is now recorded in SI S2.6.2: (A) branch-core descent
failed its linked-tori/nested coverage gate; (B) majority-background
termination did not improve leaf counts; (C) retain Hartigan assignment
and score signal-only is the adopted interim semantics. Nested also loses
755 outer-shell points to the root background child at the selected level
(coverage 0.76): the outer shell is only fully connected at radii where it
also connects to the inner shell through tissue. Remaining work is the
tissue/sibling-context residual only: A5-T11/T13 and A4-T13 isolate sibling
versus tissue sufficiency. Do not reopen A/B, tune a descent fraction, or
resume tissue-mass sweeps.

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
- **FINDING (A4-T33..T36 + A3 SI):** per-shell mult schedule `{1:3,2:6}`
  recovers **both** nested shells on fitted scaffold (128 and denser 256);
  first full nested fitted Betti on this harness — **proposal-path** only
  (not SI single-mult default). Tori local-σ / crossed / denser / mult-sweep
  (`mult∈{1..8}`, max b1=1) never reach `(1,2,1)`. Hollow+cal=6 no gain.
  schedule×local-σ **regresses** nested shell2 — keep global σ. Keep
  `@awaiting`.
- **FINDING (A4-T37..T39):** tori lifetime×mult grid max_b1=1; denser
  clean-grid (24×12..40×20) all `(1,2,1)` but fitted n=500/max_nodes=256
  yields **partial** torus0 `(1,2,0)` — first interlocking fitted b1=2
  (other torus / b2 still fail; n=1000 still max_b1=1); nested
  schedule×lifetime recovers both shells only at `frac≥4` (SI 0.5
  inflates). Keep `@awaiting` (not SI single-default).
- **FINDING (A4-T40..T43):** denser max_nodes 384/512 **REGRESS** (b0
  inflate as σ↓; partial b1=2 only at 256 fine or 384 fine on other
  torus). cal-mult=6 on denser fitted **erases** 256 partial b1=2.
  Multi-seed denser256 partial is **seed-fragile** (only seed2 of 0..2
  gets both tori `(1,2,0)`; still b2=0). lifetime_frac on seed2 denser256
  never unlocks b2 (low frac inflates b0; frac≥2 stays `(1,2,0)`). Keep
  `@awaiting`.
- **FINDING (A4-T44/T45):** Stage-1 seed sweep denser256 is fragile —
  seed77 both-tori `(1,2,0)`; seed7 sporadic dirty b2 `(2,1,1)` on
  torus0; no full `(1,2,1)`. hollow+lifetime on seed2 denser256 keeps
  most signal but torus1 dirty b2 / inflated b0 across fracs — still no
  `(1,2,1)`. Keep `@awaiting`.
- **FINDING (A4-T46..T49):** lifetime×cal-mult on seed2 denser256
  `max_b2=0`. seed77 lifetime vs hollow: signal stays `(1,2,0)` /
  `max_b2=0`; hollow dirty torus1 b2 only. seed7 filtration/lifetime
  cleanup: fixed dirty `(2,1,1)/(1,2,0)`; `n_clean=0`. seed77
  hollow×lifetime×cal-mult: dirty b2 only at SI fine mult=1.5; cal≥3
  kills dirty without cleaning to `(1,2,1)`. Keep `@awaiting`.
- **FINDING (A4-T50..T55):** densify ladder seed77: 256+512 both-partial
  `(1,2,0)`; 384 regresses `(1,1,0)`; `max_b2=0`. seed7+hollow: dirty
  `(2,1,1)` persists (`n_clean=0`). hollow mid×h0: dirty only
  mid=0.5×h0∈{0.5,0.7}; mid=0.65 preserves both-partial no dirty.
  mid65×life×cal: `max_b2=0` / no dirty. densify512×hollow: inflates
  torus0 b1→3 still `max_b2=0` (no dirty lever unlike denser256).
  seed7×mid65: dirty persists. Clean `(1,2,1)` still unreachable under
  A4 owns_files probes. Keep `@awaiting`.
- **FINDING (A4-T56..T61):** hollow×sigma-scale dirty only at scale=1.0
  (`n_clean=0`). Stage1×mid65: only seed77 both-partial; seed7 dirty
  persists. tissue×noise: `max_b2=0`; both-partial only noise=0.02×
  tissue∈{0,0.03}. densify384×hollow stays `(1,1,0)` / `max_b2=0`
  (hollow ≠ restore both-partial). circle tissue×mult: SI `b1=0` all
  tissue; recover min_mult≥3. nested sigma×hollow ≈ no-op
  (`any_all_either=false`). Clean `(1,2,1)` / SI circle b1 / nested
  voids still unreachable. Keep `@awaiting`.
- **FINDING (A4-T62..T64):** densify384×lifetime×cal-mult: signal stays
  `(1,1,0)`; cal raises `b1` but `max_b2=0` / `n_clean=0`. circle
  tissue×lifetime: SI mult never recovers; cal mult=6 recovers only at
  high frac. nested hollow mid-sweep: mid0.35 prunes; mid≥0.5 no-op;
  `any_all_either=false`. Keep `@awaiting`.
- **FINDING (A4-T65..T73):** densify384×hollow×cal `max_b2=0`. circle
  lifetime×noise SI dead (cal frac≥2..4). nested densify256/512×hollow
  no shell unlock (`any_b2` false at 512). seed2 densify256 both-partial
  ×hollow×cal / high-frac≥4: preserves both-partial, never introduces
  b2. circle tissue×noise×frac: SI dead; cal clean floors mostly frac≥4
  (tissue0.08×noise0 never). nested schedule{1:3,2:6}@densify512×hollow
  fails (densify512 kills T33 schedule recovery). Keep `@awaiting`.
- **FINDING (A4-T74..T79):** schedule{1:3,2:6}@densify128/256×hollow
  recovers `(1,0,1)` (cliff is **densify512**, not 256). circle
  tissue0.08×noise0: SI/cal6 never jointly clean; **first** clean
  `(1,1)` at proposal-path cal mult=4×frac=3 only. seed7 densify256
  highfrac + hollow-cfg×lifetime: `any_clean_b2=false`; mid0.35×h0≥0.5
  erases dirty-b2 without unlocking `(1,2,1)`. Keep `@awaiting`.
- **FINDING (A4-T80..T88):** densify384/320 fail; densify288 schedule×
  hollow recovers `(1,0,1)` (signal+primary+mild); 304 mild-only
  transitional; fail by 320 — cliff onset after **288**. tissue0.08
  cal4×frac3 pin survives noise through **0.20** (26/27; proposal-path
  ONLY). seed7 erase×cal/sigma/gabriel: void or dirty-b2 reintro — never
  clean `(1,2,1)`. Keep `@awaiting`.
- **FINDING (A4-T89..T94):** densify cliff **NON-MONOTONIC** — 296 full
  recover; 300 mild-only; 308 full recover; 312 hard-fail. tissue0.08
  cal4×frac3 pin collapses for all noise>0.20; tissue0.12 preserves
  cal4≤0.20 + mult3@0.22 residual (proposal-path ONLY). seed7
  erase×lifetime soft / mst / soft_capacity: dirty-b2 only — never
  clean `(1,2,1)`. Keep `@awaiting`.
- **FINDING (A4-T95..T100):** densify304/306 mild-only; 310 full recover
  — cliff map 296full/300mild/304mild/306mild/308full/310full/312fail.
  tissue0.12 mult3 residual dies by noise=0.25 (survives only@0.22);
  mult3@0.22 transfers to tissue{0.10,0.15} (proposal-path ONLY).
  erase×mid×h0 / bridge_critical / bridge_mass: dirty-b2 reintro on
  baseline — never clean `(1,2,1)`. Keep `@awaiting`.
- **FINDING (A4-T101..T103):** densify302 mild-only (signal/primary fail;
  mild all_match); densify314 hard-fail all arms — cliff continues
  non-mono (+302mild/+314fail). mult3 residual noise edge **NON-MONO**:
  survive@0.22 / die@0.23 / survive@0.24 / die@0.25 (cal4 never;
  proposal-path ONLY). gabriel∧H×bridge/bridge_mass: any_clean=false
  any_full=false max_b2=1 dirty-only; conj_bridge never clean;
  persist_agree N/A at HollowEdgeConfig. Keep `@awaiting`.
- **FINDING (A4-T104..T106):** densify298 **full** recover / densify316
  **hard-fail** — cliff continues (fills 296full↔300mild / past 314fail).
  mult3 residual fine noise **NON-MONO**: 0.22✓/0.225✓/0.23✗/0.235✓/
  0.24✓/0.25✗ (cal4 pin dead; proposal-path ONLY). denser384
  soft×gabriel∧H×bridge: signal (1,2,0) void-absent; max_b2=0 (no
  dirty reintro vs denser256 dirty max_b2=1). Keep `@awaiting`.
- **FINDING (A4-T107..T109):** densify294 **and** 318 **both hard-fail**
  (no full/mild) — NON-MONO cliff (294 fails between 288full/296full;
  318 past 316fail). mult3 residual dip **narrowly@0.23**
  (0.2275✓/0.2325✓; cal4 never; proposal-path ONLY). denser512
  soft×gab∧H×bridge: max_b2=0 void-absent; signal (1,1,0)/(1,3,0) not
  (1,2,0)@384 — densify≠void unlock. Keep `@awaiting`.
- **Remaining before flipping recovery tests:** densify290/292/320 pin;
  tissue0.10/0.15@0.235; erase×persist_agree RecursionConfig leaf PH or
  denser640/seed-vary; keep recovery `@awaiting` until SI-default fitted
  evidence is green.
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
- **Landed (A5-EXP-mu/flux + T46..T48 + A3 SI):** whitened λ_f / μ_S soft
  solve + `epsilon_flux` / spectrum damp; count-aware `λ_f`; patch `Σμ_S`;
  Stage-1 BMU wiring sketch (flags off).
- **Landed (A5-EXP-glue + ann-inc + A3 SI):** `enable_shared_face_glue` +
  Complex→`node_to_simplices` / ANN BMU bridge (`enable_complex_ann_incidence`,
  flags off).
- **Landed (A5-T49..T51 + A3-T54 SI):** `enable_global_face_solve` stub;
  `enable_live_density` / `route_live_density_from_complex`; dry_run
  `DualDryRunResult.stage1_route` wires Complex ANN when flagged (all
  default off).
- **Landed (A5-T52..T54 + A3-EXP-si63 SI):** `enable_loopy_bp_schedule` /
  `solve_loopy_bp_schedule` (cavity msgs); `enable_mass_normalization` +
  `epsilon_mass`; `probe_acceptance_none_open_default` documents current
  open-default matrix (flag-on detects disconnect; defaults unchanged).
- **Landed (A5-T55..T57):** BP spectrum damping probe; online→offline
  schedule sketch; `probe_fail_closed_dual_adjacency_plan` documents path
  to replace None=>True (defaults unchanged).
- **Landed (A5-T58..T60):** `enable_bp_damping_policy` /
  `propose_bp_damping_policy` (cond>cap ⇒ ridge + raised damping); 
  `enable_online_offline_loopy_compose` /
  `run_online_offline_loopy_compose` (live BMU→loopy BP); 
  `GateConfig.fail_closed_dual_adjacency` default `False` +
  `probe_gate_fail_closed_switch` (score_edit None⇒reject only when
  apply+fail_closed). Flags/defaults unchanged.
- **Landed (A5-T61..T63):** `enable_bp_policy_in_loopy` wires policy into
  `solve_loopy_bp_schedule`; `enable_loopy_bp_convergence_probe` /
  `probe_loopy_bp_convergence` residual trajectory; compose forwards
  policy flag. Defaults off.
- **Landed (A5-T64..T66):** `enable_loopy_bp_residual_stop` /
  `propose_loopy_bp_residual_stop` (plateau/tol sketch — not a production
  certificate); `probe_fail_closed_score_edit_matrix` 9-cell accept/reject
  matrix (`GateConfig` defaults unchanged); `enable_mass_loopy_compose_probe`
  / `probe_mass_loopy_compose`. Flags off.
- **Landed (A5-T67..T69):** residual-stop early-exits
  `solve_loopy_bp_schedule` (`residual_stop_reason`/`iters`);
  `enable_loopy_bp_spectrum_safe_cert` /
  `probe_loopy_bp_spectrum_safe_cert` (no-ridge+stop harness — not
  production cert); `enable_policy_residual_compose_probe` /
  `probe_policy_residual_compose` (policy pin + compose residual-stop
  forward). Flags off.
- **Landed (A5-T70..T71):** `enable_spectrum_safe_policy_pin_probe` /
  `probe_spectrum_safe_policy_pin` multi-cond grid; 
  `probe_fail_closed_evidence_gate_matrix` live `EvidenceGate.evaluate`
  parity vs `score_edit`. Flags/defaults unchanged.
- **Landed (A5-T72..T77):** `enable_spectrum_safe_policy_traj_probe` /
  `probe_spectrum_safe_policy_traj` cap-sweep residual traj;
  `probe_fail_closed_dry_run_evidence_gate` live dry_run×fail_closed×
  EvidenceGate; `probe_residual_mass_loopy_compose` early-exit pin;
  `probe_fail_closed_dry_run_reconnect_bridge` disconnect→reconnect;
  `enable_spectrum_safe_policy_mass_compose_probe` /
  `probe_spectrum_safe_policy_mass_compose`; 
  `enable_residual_mass_patience_sweep_probe` /
  `probe_residual_mass_patience_sweep`. Flags/defaults unchanged.
- **Landed (A5-T78..T80):** `enable_spectrum_safe_policy_mass_traj_probe`
  / `probe_spectrum_safe_policy_mass_traj`; 
  `enable_residual_mass_policy_patience_probe` /
  `probe_residual_mass_policy_patience`; 
  `enable_spectrum_policy_mass_fail_closed_bridge_probe` /
  `probe_spectrum_policy_mass_fail_closed_bridge`. Flags/defaults
  unchanged.
- **Landed (A5-T81..T87):** spectrum/residual patience(+cap) compose
  probes + traj×fail_closed + patience×cap grids +
  patience×cap×fail_closed bridge + residual patience×cap×traj
  (`enable_*` / `probe_*`; flags off).
- **Landed (A5-T88..T93):** patience×cap×fail_closed EvidenceGate
  matrix + residual traj×fail_closed bridge + spectrum dry_run×EG +
  spectrum patience×cap traj + residual patience×cap×fail_closed
  matrix + spectrum traj×fail_closed reconnect (`enable_*` /
  `probe_*`; flags off; dual 147p). Gaps remain: fail-closed
  acceptance flip. Mass/density/benchmark stay `@awaiting`. **Do not
  close #43** until acceptance-path default replaces the conservative
  open default / fuller S6.
- **Landed (A5-T94..T99):** residual traj×fail_closed dry_run EG +
  spectrum traj×EG matrix + residual patience×cap×dry_run EG +
  spectrum traj×dry_run EG + residual traj×EG matrix + residual
  patience×cap×fail_closed reconnect (`enable_*` / `probe_*`; flags
  off; dual 153→159p). Do **not** flip spectrum-safe/policy/mass/
  enable_dual defaults; mass/density/benchmark stay `@awaiting`.
  **Do not close #43.**
- **Landed (A5-T100..T102):** residual×spectrum dual-path patience×cap×
  fail_closed compose + spectrum traj×fail_closed×dry_run×reconnect +
  residual patience×cap traj×matrix×dry_run EG triple (`enable_*` /
  `probe_*`; flags off; dual 165p). Do **not** flip spectrum-safe/
  policy/mass/enable_dual defaults; mass/density/benchmark stay
  `@awaiting`. **Do not close #43.**
- **Landed (A5-T103..T105):** residual×spectrum dual-path patience×cap×
  traj compose + spectrum traj×fail_closed×dry_run×matrix×reconnect +
  residual×spectrum dual-path×fail_closed dry_run EG compose
  (`enable_*` / `probe_*`; flags off; dual 171p). Do **not** flip
  spectrum-safe/policy/mass/enable_dual defaults; mass/density/
  benchmark stay `@awaiting`. **Do not close #43.**
- **Landed (A5-T106..T108):** residual×spectrum traj×fail_closed
  dry_run/reconnect/matrix compose (`enable_*` / `probe_*`; flags off;
  dual 177p). Do **not** flip spectrum-safe/policy/mass/enable_dual
  defaults; mass/density/benchmark stay `@awaiting`. **Do not close #43.**

## 47. Evidence-gated Stage-1 insertion so max_nodes is a safety assert

Operational cap-doubling on under-resolved level-set scaffolds (#44
trichotomy) is a stand-in, not equilibrium `N*`. Open-loop growth
(`propose_splits` auto-accepts `variance > tau`; relative prune floors
self-normalize) is diagnosed in
`docs/Proteus/paper_1_foundational/reference/open_loop_growth_and_node_cap.md`.
Remaining: DM-gated insertion (and split acceptance) in the Stage-1
runtime loop so the mesh grows only while a valley remains unresolved,
and `max_nodes` becomes a safety ceiling rather than the resolution
control. Scheduled with M4 evidence-gate wiring. Do not treat
cap-doubling as the final `N*` story.

## 48. Finer-walk evidence floor vs the local one-feature null

The #44 finer walk that reaches composite `tau_sep` (~80× below coarse
`L=1`) shatters uniform manifolds, because any finite sample is
multimodal at a fine enough bandwidth. The stop must be an
**acceptance-path evidence floor**, not another `max_finer_scale_steps`
budget (SI S2.6.2).

Null: this region is one Hartigan cluster. Accept only when the
min-cut-normalized flow bottleneck `φ`, augmented by hit-conditioned
cross-flow evidence, clears the deepest shot-noise valley expected at
the current `(n, τ, k, N, geometry)`. Background-aware DM confirms a
background partition but does not supply this valley-existence floor.
Fade/tissue semantics remain separate (#45). Raw persistence / excess
mass are already measured inseparable.

**LANDED (level-set mode; `use_level_set_clustering` still default
off):** one-feature statistic = min-cut-normalized `φ` (cross-cut
max-flow over the minimum intrinsic internal cut of either side;
Fiedler / hop-geodesic median splits, largest component per half)
against the calibrated ceiling `max_bottleneck_ratio = 0.25`, then the
background-aware DM verdict; resolution bound `N ≤ n/k` (clamps scale
search and the walk); `track_tau` finer walk — fresh
`fit_scaffold_at_tau` at every finer τ with N free up to `n/k`, stop
when the bound binds; no γ, no `no_cut` trigger, no step-budget
dependence, no studentized ρ. Full normal-path sweep, seeds 0–4: every
null 30/30 (circle, swiss, lone torus / inner shell / 2-D / 4-D
Gaussian), hierarchy root K=3 5/5 (5–6 of 6 fine leaves), linked tori
root K=2 5/5 (ARI ≥ 0.999), nested shells root K=2 5/5 with exactly 2
signal leaves; bimodal circle 2/5 and weak two-Gaussians 2/5 are the
only failures. SI S2.6.2 / S14.3 updated.

Remaining (ordered by severity):
- **One-feature statistic: min-cut-normalized φ (landed 2026-09-09).**
  The studentized ρ = φ/φ₀ floor was ill-posed — the candidate is a
  disconnection of the superlevel set, so the signal-induced graph is
  already the candidate's components and no connected disagreeing
  bisection of it exists (pool empty 91/91 null reads) — and is
  retired (`null_bottleneck_ratio`, `studentized_bottleneck`,
  `one_feature_null` removed; walk stops on the `n/k` bound only). The
  statistic is φ = cross-cut flow / min(intrinsic min cut of A, of B):
  a valley must be weaker than any cut inside the pieces it separates.
  `max_bottleneck_ratio = 0.25` is **calibrated** by the declared
  null-ensemble protocol (six null scenes × seeds 0–19, root reads with
  a candidate, `level_set_root_accept_probe.py --max-depth 1`): 359
  reads, φ min 0.288 (swiss s17), p1 0.487, p5 0.63, median 1.30;
  composite accepts 0.006–0.244. SI S2.6.2 / S14.3 rewritten;
  `test_max_bottleneck_ratio_is_calibrated_null_envelope` pins the
  value to the protocol. The reproducible harness rerun found 357 reads,
  min 0.2876, p1 0.521, p5 0.642, and median 1.365: the minimum and false
  accept reproduce, but the count and percentiles drift slightly from the
  published table. The old S-curve generator double-covered a half-arc,
  so its seed-8 read is withdrawn. The corrected injective, area-uniform
  sheet (`θ ∈ [-1.5π, 1.5π]`,
  `z = sign(θ)(cos(θ) - 1)`, width 2, `n=800`) accepts on seeds
  1/16/17/18 of 0--19 at φ 0.190/0.242/0.202/0.096 (60 reads).
  Its null minimum 0.096 overlaps the composite-accept band 0.006--0.244,
  so no fixed ceiling separates them: the defect is statistic-level, not
  calibration (director D3). Keep 0.25 unchanged as the
  calibrated-provisional operational ceiling of the default-off mode.
  A3-T9's flat-strip control also accepts below 0.25, so extrinsic curvature
  is not the driver and the dependent R=2/min-side follow-ons are killed.
  The component-only child envelope likewise breaches 0.25: 56 positive-phi
  reads, min 0.167 (plus two circle graph disconnections at phi=0).
  The cross-scale cut-persistence oracle is killed: null and composite
  accepted cuts overlap (minimum Jaccard 0.257 versus 0.251; both medians 1.0).
  Cut-local density contrast is also killed (A2-T10): null
  `[0.999,1.520]`, median 1.095, overlaps composite accepts
  `[0.913,1.286]`, median 1.114, for a 0.60 gap rather than the required 2x.
  The final record-only A3-T11 covariate check and A3-T14 classification of
  the two child-envelope phi=0 cases are now complete:
  every measured covariate overlaps the composite-accept band, and both circle
  cases are ordinary graph disconnections in the same class as lone-Gaussian
  seed 17. No additive floor candidate remains. D8 (Fable) reopens the
  statistic's definition, not the ceiling: phi compares the coarsest C--D cut
  (the region's deepest sampling gap, hence an extreme) against a single
  typical Fiedler bisection. Under the one-feature null, phi may therefore sit
  systematically below one and deepen with the number of independent gap
  sites; the open elongated sheets breach while closed or compact nulls do
  not. The report-only falsifiers are a matched-scan denominator (A2-T16) and
  a flat-strip aspect ladder (A3-T16). The 0.25 ceiling,
  `require_separation_evidence`, and `use_level_set_clustering` remain
  unchanged pending D9. Any accepted successor must use a declared protocol
  and null envelope, pass the corrected S-curve and every flat-strip aspect at
  0/20, the six nulls at 0/120, and the child envelope at 0/N, while retaining
  every seed-0--4 composite root accept.
- **Graph-disconnection false accept (acceptance path, owns the DM
  overconfidence item).** lone 2-D Gaussian seed 17 accepts at N=50,
  τ=0.012: a 72-point clump in the shoulder has no Hebbian link outside
  itself, so cross flow is exactly 0 (φ = 0) and DM confirms with logBF
  2162. A2-T1 found that nearest-saddle/Poisson support is stronger on
  this false accept than on true two-Gaussian accepts, so a cut-local
  link-absence guard is not valid. A default-off
  `require_separation_evidence` guard now requires zero-cross cuts to
  clear the derived hit-mass mixing expectation
  `λ = k·2p(1-p) > log(tau_bf)`. It blocks the seed-17 false accept while
  preserving two-Gaussian, linked-tori, and nested-sphere root accepts
  on seeds 0--4. On the six nulls over seeds 5--19, the unguarded path
  accepted only lone-Gaussian seed 17 (1/266 candidate reads), while the
  guard accepted none; all 15 composite roots still accept. The generalized
  cross-flow Poisson gate is falsified: it separates by only 1.74 orders while
  rejecting true composites, and its $N$-scaled form inverts the decision.
  Keep the zero-cross guard off unless the A2-T7 flip-readiness sweep is
  metric-identical to the default path and the conditional A2-T8 null check
  remains clean. Repro:
  `level_set_root_accept_probe.py --seed 17 --scenes
  lone_gauss2d_null --max-depth 1`.
  A2-T7 is now metric-identical to both A6 OFF references on all 60 lines,
  and the guard is a no-op on all corrected S-curve reads because none has
  zero cross flow. D4 killed the standalone A2-T8 flip: the guard remains
  default off because it cannot see the live positive-phi sheet/strip/child
  false accepts. Keep the guard and passthrough; review only as part of a
  later complete acceptance package.
- **Connected-support valleys (bimodal circle 2/5, weak two-Gaussians
  2/5).** Failures terminate at the root with no accepted candidate,
  usually at the `n/k` bound; neither child recursion nor DM rejection is
  implicated. Analytic valley depth is seed-invariant and valley-band
  sample counts overlap PASS/FAIL seeds. The `4x/10x` oracle separates the
  cases: all three failing bimodal seeds recover by `4x` (sample-conditioned),
  while weak two-Gaussians stays `K=1` on at least half the failures even at
  `10x` (statistic-limited). Further scene scaling is stopped; do not change
  expectations or lower the ceiling. A4-T8's root-only separation curve is
  monotone in analytic valley depth: `K=2` rates rise 2/5, 3/5, 5/5, 5/5,
  5/5 across separations 2.5, 3, 3.5, 4, 6. At child-sized total
  `n in {300,600}`, clear two-Gaussian remains 5/5 but nested/tori are 0/5
  versus 5/5 at full suite size, confirming a power boundary rather than
  grounds to lower the ceiling. Weak seed-2 matched-tau reads change
  bottleneck/no-cut regime with `n`; n-stratification is a director item,
  not permission to resume the killed scene-scaling family.
- **Tissue-heavy children (→ #45).** Root splits leak tissue into
  signal children (bimodal child 37% tissue); the reader then marks
  most of the child background and accepts shell+tissue chunks. Nested
  no longer over-splits under the connected null (2 leaves on all
  seeds). A4's component-only child-sized null generators have landed;
  A5-T6 found nested+bimodal pure children clean, and A5-T7 found both
  weak components clean on all 10 exact `n=194` reads. This rules out an
  intrinsic small-`n` null failure for the isolated children; the residual is
  tissue/sibling context. A3-T7's broader child-sized envelope has landed
  and itself breaches the provisional ceiling, but that statistic-level
  result does not change the tissue/sibling diagnosis for the exact children.
- Weak two-Gaussians (sep 2.5σ) children in the full tissue/sibling context
  can still split at the `N≤24` bound; isolated `n=194` components do not.
  Revisit with #45 semantics rather than lowering the #48 ceiling.
- **DM log-BF is not a valley-existence floor.** The default-off
  sample-normalized correction reduces the Gaussian false accept from
  `2162` to `128`, but corrected null/composite distributions overlap
  on 69% of reads and true two-Gaussian accepts sit inside the null band.
  SI S10.2 now records the negative result and limits DM to
  background-partition confirmation. Do not tune its margin or flip
  `sample_normalized_counts`.
- **Paper sync and hierarchy cross-check (A6-T13/T14/T16):** the D4 paper update
  has landed (the separation guard stays default off; no A2-T8 follow-up
  sentence). The hierarchy seed-0 true L0/L1 accept has path-dependent `N`
  and `phi` across cap probes (A6-T16), so the earlier `N=46`, `phi=0.201`
  observation is not stable and is not grounds to lower 0.25 or raise the
  cap. A5-T12 records the matched-tau `phi` instability without reviving an
  n-indexed ceiling. Cut persistence and cut-local density contrast are killed;
  no evidence-floor candidate remains this burn.
- Do not flip `use_level_set_clustering` until the corrected S-curve
  null is 0/20 under a new acceptance-path statistic with a declared protocol
  and null envelope. Keep 0.25 unchanged, do not open #47 this burn, do not
  delete S2.6.1 stand-ins, and do not retune frozen suite numbers.
