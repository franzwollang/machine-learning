# Proteus Paper 1 — Open Issues

Current, active issues only — resolution history lives in `OPEN_ISSUES_LOG.jsonl`, never
here. Numbering is historical and stable: resolved issues are deleted rather than
renumbered, so gaps in the sequence are expected. Each entry lists only the work that
actually remains. See `PLANNING.md` for the suggested order of attack.

Next issue number: 48

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
- **ACCEPTANCE BLOCKER:** awaiting-flip is still blocked. The finer-step
  budget that recovers nested/tori shatters uniforms; the budget that
  preserves uniforms (`<=4`) is the pairing already known not to reach
  `tau_sep`. Child descent needs a stop that is not “always finer-tau on
  a resolved null.” Do not flip `use_level_set_clustering`. Do not delete
  S2.6.1 stand-ins. Sample-level background-aware ARI remains the recovery
  metric.
- Evidence-gated insertion so `max_nodes` becomes a safety assert is #47
  (M4 / `reference/open_loop_growth_and_node_cap.md`).
- After the layer validates on repaired benchmarks (#45): deprecate the prepass-flag zoo
  in `recursion.py` / `edge_evidence.py` (all proposal-path, default off; keep the
  ROC/adversarial-null calibration infrastructure) and run the awaiting-flip review.

## 45. Synthetic fade-halo labels need benchmark-wide semantics

The repaired linked-tori generator now exposes the underlying remaining
benchmark issue: faded-density generators label all samples below
`lambda=0.5` as background, often about half the cloud. Full-cloud metrics are
therefore dominated by halo labels even when signal topology/clustering is
correct.

Remaining work: define benchmark-wide halo/background semantics; retain
signal-only ARI + background recall until then; re-baseline the `@awaiting`
recovery expectations (#41, #26) on repaired scenes without weakening their
topology criteria.

## 46. Pytest runtime hygiene: simulations are misclassified as tests

The Stage-1 test layout mixes fast correctness checks with full stochastic experiments:
`tests/stage1/test_recursion.py` currently takes ~55 minutes (69 tests), and a combined
AP/scale-search/persistence subset was still running after ~54 minutes. This defeats the
feedback role of pytest and makes routine regression validation impractical.

Work:
- Profile individual cases with bounded per-test timing; identify any accidental
  performance regressions separately from intentionally large simulations.
- Keep deterministic unit/integration tests within a few seconds and the routine Stage-1
  regression target within roughly two minutes.
- Move long seed sweeps, calibration studies, and large synthetic reconstructions to
  explicit diagnostic/benchmark scripts (or a separately selected `slow` suite that is
  never part of routine pytest).
- Add a runtime guard in the routine test command so newly misclassified simulations fail
  visibly instead of silently expanding suite time.


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
