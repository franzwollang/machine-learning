# Proteus Paper 1 — Open Issues

Current, active issues only — resolution history lives in `OPEN_ISSUES_LOG.jsonl`, never
here. Numbering is historical and stable: resolved issues are deleted rather than
renumbered, so gaps in the sequence are expected. Each entry lists only the work that
actually remains. See `PLANNING.md` for the suggested order of attack.

Next issue number: 40

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
- Blocked on / naturally lands with Stage 2 flag-complex construction; option (c) could land earlier as a Stage 1 test.

## 26. Manifold-zoo junction test (circle + line + plane + box)

- Classic GNG benchmark: 1D circle, 1D segment, 2D plane patch, and 3D box meeting at dimensional junctions.
- Tests mesh quality per patch, `d_final` accuracy at junctions, junction detection (S8.4), and Stage 2 heterogeneous simplex dimension (S4.2).
- Needs a dataset generator in `tests/datasets/synthetic/` with per-component intrinsic-dim ground truth. Generator can land early as a diagnostic fixture; scenario assertions are deferred until S8.4 is implemented.

## 27. Clustering: canonicalize the Q-score and remove cleanup heuristics

The AP -> Q-merge -> refine pipeline is implemented and passes the circle, swiss-roll, and hierarchical-Gaussian regressions (six terminal leaves). Recursion is Q-gated (`recursion.py`: leaf when `n_clusters <= 1` or `partition_q_score <= 0`). The Q primitives are now pinned down in SI S2.6.1 (`K_v`, `A_sym`, `W_v = K_v * A_sym`, `LocalIntra`, `BoundaryInter`, `InterLocal` promoted verbatim from `reference/stage1_clustering_and_resolution.md`). What remains is making the implementation heuristic-free:

- **FINDING (cross-family validated): the constant-free single-scale null is under-determined.** Direct experiments plus an independent GPT audit (agent `0b462607`) establish that at a single scale the graph-local `Q(C)` and `InterLocal/LocalIntra` do **not** carry enough information to separate a uniform manifold that must be one cluster (circle/swiss ring arcs) from genuinely multi-modal structure that must stay split (hierarchical-Gaussian coarse blobs). Their per-cluster `Q` distributions overlap (circle arcs `Q≈0.55–1.18`; hierarchy coarse-blob clusters `Q≈0.58–0.88`), and the same overlap holds for extent ratios `v̂/v`, conductance, modularity, full-graph (shadow∪lifted) variants, and spectral gap. Separately, the partition-`Q` null is mathematically degenerate: a whole connected component has empty boundary → `BoundaryInter = 0` → `Q(null) → +∞`, so a same-criterion comparison trivially favours one cluster. The currently-passing pipeline distinguishes the cases only via **side channels** (lifted-component count, size imbalance). Conclusion: "one Q-improving merge-to-fixpoint + a single-cluster-null test, with no constants" is *not achievable with the present single-scale primitives* — and it is not a missing `K_v` term (the kernel is already in `W_v`).
- **Canonical arbiter deferred to persistence (M2) / DM gate (M4).** The intrinsic-vs-composite distinction is a cross-scale statement (SI S2.6.2): a real partition persists across ≥ 2 adjacent τ grid points; a uniform manifold's arc-partition should not. The persistence arbiter now **exists** (`stage1/persistence.py`; #28). *With the S2.6.1 stand-ins present* it reproduces the qualitative discrimination (circle → no persistent split; hierarchical → persistent 3-way split), because the recorded per-scale partitions are already heuristic-collapsed. SI S2.6.1/S2.6.2 document the scope and operational signal: single-scale `Q` is a proposal screen and the cleanup passes are operational stand-ins (empirically still **load-bearing** — see the corrected finding below). The alternative M4 S3.4 Dirichlet–multinomial gate supplies the complementary non-degenerate likelihood-ratio null.
- **Persistence accept-gate is wired into recursion (`RecursionConfig.require_persistent_split`, SI S2.6.2), default off.** A region's split is accepted only if a multi-cluster partition persists across adjacent `tau` grid points (`persistence_result.tau_star_index is not None`); non-persistent fragmentation makes the region terminal. Two integration tests lock in the *gate-with-stand-ins-present* behaviour (`test_persistence_gate_circle_is_single_feature` → single leaf; `test_persistence_gate_hierarchy_matches_gt` → six leaves, fine ARI 1.0).
- **CORRECTED FINDING (cross-family validated): the persistence gate does NOT replace the stand-ins; they are load-bearing and deletion is BLOCKED.** A *full* ablation — monkeypatch `_refine_boundaries`, `_absorb_*`, **and** `_q_merge_any_improving` (which also disables the `<= 3` collapse and the `>= 4` re-merge) to identities — with `require_persistent_split=True` gives **circle → 37 leaves** (want 1) and **hierarchy → 12 leaves** (want 6). This *refutes* the earlier turn-7 ablation note ("gate alone yields circle→1"), which almost certainly left the `<= 3` collapse intact and so silently kept the very heuristic under test. GPT audit (agent `320a28ae`) reproduced circle→37 exactly and confirmed the diagnosis and every point below.
  - *Mechanism (warm-start false positive):* on the warm-started sweep the circle's arc-partitions mostly do NOT agree across adjacent scales (matched-Jaccard 0.17–0.33, as the theory predicts), but an **isolated fine-end pair coincides at 0.609** — enough to satisfy `P_persist=2` at `theta_ovl=0.5`, so the gate accepts a spurious split and recursion explodes. An independent **cold-start** refit of the same scales removes that block entirely (overlaps drop to 0.475/0.419), confirming it is a warm-start/path-dependence artifact, not a real feature. A genuine feature (hierarchy) instead persists from the **coarsest** grid point with high, stable overlaps (0.68–0.93) over a run of length ≥ 3.
  - *Hardening direction (ii) — coarse-anchoring — has LANDED (`PersistenceConfig.coarse_anchored=True`, default).* The characteristic split must be anchored at the coarsest multi-cluster grid point: letting `j0` = coarsest index with `K >= 2`, accept iff `run_length[j0] >= P_persist`, else terminal. This rejects the isolated fine-end warm-start block. Full-strip recursion ablation (all stand-ins → identities, gate on): **circle 37 → 1 leaf, hierarchy 12 → 6 leaves** (both correct); **swiss roll 32 → 12 leaves** (still over-fragments). Suite stays green because every current test runs *with* stand-ins present, where coarse-anchoring reproduces the identical `tau*` as the legacy rule. Cross-family GPT audit (`gpt-5.4-high`, agent `e8eef21f`): implementation correct, swiss-still-fragments conclusion correct; verdict LAND WITH CAVEAT — the scale-space justification is *motivational, not a theorem* (non-enhancement is about smoothed-density extrema, not the warm-started scaffold partition sweep), so it carries a transient-coarse-blip false-negative and grid-sensitivity (both now documented in SI S2.6.2 as operational trade-offs).
  - *Residual (narrowed):* the remaining blocker is **marginal coarse-scale arc-persistence on developable manifolds** — the swiss roll's coarsest partition is 3 arcs whose adjacent overlap (~0.568) sits just above `theta_ovl=0.5`, so coarse-anchoring admits it and the region fragments without the stand-ins. Two viable hardening paths (not just one): (i) the Stage-2 S3.4 DM evidence gate as the acceptance null [blocks on M4]; (ii′) require the coarse-anchored interval to survive an independent **cold-start / reverse-sweep** recomputation of the same grid interval — a self-contained, path-independence check that targets the marginal-overlap artifact without a new threshold (per the GPT audit). NOT recommended: raising `theta_ovl` alone (brittle) or `min_persistence >= 3` alone (overfit risk).
  - *Blocked deletion scope:* the load-bearing stand-ins are `_refine_boundaries` (eta=0.3), `_absorb_tiny_clusters_into_dominant`, `_absorb_one_tiny_satellite`, `_absorb_full_graph_isolates`, and the `<= 3`-fragment collapse in `run_clustering` (the four previously-dead helpers are already removed). Only once (i) or (ii′) lands and is validated across ALL scenarios (circle, swiss, hierarchy, and — when available — nested spheres / linked tori) should the gate be made default, the single-scale tests re-scoped (chiefly `test_circle_clustering_produces_one_cluster` ==1 and `test_swiss_roll_stage1_diagnostics_at_tau_star` ≤3), and the stand-ins deleted.
- **Paper/SI prose** should describe the implemented AP -> Q-merge -> refine pipeline (the Leiden detour is obsolete). SI S2.6.1/S2.6.2 now document the scope, the persistence signal, and its warm-start limitation; paper §3 prose still needs a one-line pointer to S2.6.2 persistence as the cluster-count arbiter.

## 28. Scale selection: remaining calibration and cleanup

The load-band heuristic and most of the original exit criterion are resolved (see log).
The default selector is now the variance-load `L = 1` up-crossing
(`controller._select_load_crossover`), which carries no `band_lo` / one-step-coarser
constant and lands tau* within one grid step of geometric truth (circle 8.0x -> 1.6x,
swiss 3.9x -> 0.9x). Scale-search test tolerance tightened `10x -> 3x` (plus a swiss-roll
analog); SI S2.5.1 and the S14.3 table rewritten to match; the legacy load-band selector
is retained behind `ScaleSearchConfig.selector="load_band"` only for regression bisection.
Q-partition persistence (`selector="persistence"`, `stage1/persistence.py`) remains the
structural arbiter for recursion timing (`P_persist=2`, `theta_ovl=0.5`, SI S2.6.2).

**Finding (cross-family audited, cold-start validated):** the proposed *primary* signal —
knees/plateaus in the compensated node count `N(tau) * tau^{d/2}` (equivalently `V_C(tau)`)
— is **not usable as an operational selector**. Warm-started it is path-dependent and its
"peak" tracks the node budget `N_max`; cold-started with a high cap it is noisy (node count
even goes non-monotone) and its log-log slope never settles at the theoretical `d/2`. This
compounds the earlier self-normalization diagnosis (the raw Lindeberg response is flat at
equilibrium). The knee proposal is therefore demoted to a diagnostic; persistence, not the
compensated count, is the structural signal, and `L = 1` fixes each feature's resolution.

Remaining work:
- **`c_{d,k}` calibration protocol.** Declare it as a calibration (uniform d-ball ensemble,
  median `r_k / sqrt(tau)` at equilibrium, tabulated over (d, k)) rather than an analytic
  constant, and ship the lookup table (couples to `C_Q(d)` #36 under the same ensemble).
- **Persistence tau* is coarse-end.** The persistence selector lands tau* at the coarse end
  of the persistent interval (hierarchical tau*=0.36 vs expected 0.0225); refine toward the
  within-interval characteristic scale before making persistence the default for structured
  regions.
- **Delete the legacy load-band selector** (and now-dormant `_legacy_slope_selector`,
  `_detect_peak` in `controller.py`) once `load_crossover` is validated to dominate across
  every scenario/recursion regression — kept behind the flag until then (M2 mitigation).

## 31. Recursion vs hierarchical GT: remaining harness follow-ups

- The structural bar and moment-matching harness are in place (`tests/harness/hierarchy_recovery.py`: Hungarian matching, Hotelling mean gate, Frobenius covariance gate; six-leaf regression passes).
- Remaining: (a) use per-level tau from each recursion frame (not only the root) when comparing deeper trees; (b) tighten gates now that SI S2.5.4 declares `Sigma_smooth = tau * I` the canonical (no longer provisional) isotropic map — the harness's `tau * I` is now spec-backed, so the gate tolerances can be tightened against it.

## 36. C_Q(d) is referenced but never defined

- SI S3.3 uses `C_Q(d)` ("variance-cap star-radius constant in the regular interior") in the prune-radius guard and merge guard, and the S12 edit-budget argument leans on the resulting `B_p` jump bound — but no formula, derivation, or calibration is given anywhere.
- Derive it (expected star radius of a cap-equilibrated Voronoi cell under local isotropy) or define it via the same uniform-d-ball calibration ensemble as `c_{d,k}` (#28), and add it to S14.3 with the appropriate status label.

## 37. Constant-status audit of S14.3

- Extend the S14.3 defaults table into a complete three-tier classification: **derived** (follows from a derivation; e.g. `alpha = ln2/k`, grid ratio, BDeu `alpha_0`), **calibrated** (measured on a declared reference ensemble with a written protocol; e.g. `c_{d,k}`, `C_Q(d)`, equilibrium load target), and **free operational default** (tunable, logged, backstopped by the evidence gate; e.g. torsion ladder bands, `kappa = 0.5`, `rho_max = 10`, prune floors).
- Every constant in `src/` should appear in the table with its status; constants that exist only in code (e.g. `gaussian_cutoff_dim = 8`, split budget `2 * h_prune`, neonatal `link_protection`) currently do not.

## 38. Promote canonical types out of tests/

- `src/proteus/` imports canonical dataclasses (`NodeState`, `Link`, and the SI-contract shapes) from `tests/contracts/`. Production code depending on the test tree is a packaging smell and blocks eventual distribution.
- Move the contract types into the package (e.g. `proteus/types.py` or per-module homes) and have the test contracts import from the package, not the reverse.

## 39. Intrinsic-dimension estimator is a degree proxy

- `intrinsic_dim.py` estimates `d_final` from graph degree (degree − 1, neighbor-median smoothed); Levina–Bickel is deferred by design. The proxy feeds AP preferences, PMI smoothing, T2 rank selection, and (later) simplex dimension and junction detection — a lot of load for an uncalibrated proxy.
- Before Stage 2 lands: validate the proxy against ground truth on the synthetic suite (swiss roll d=2, circle d=1, mixed-dim and junction datasets), and either calibrate a correction or implement Levina–Bickel behind the same interface. S8.4 junction detection should not inherit silent bias from the estimator.
