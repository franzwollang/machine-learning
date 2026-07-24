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
- **Canonical arbiter deferred to persistence (M2) / DM gate (M4).** The intrinsic-vs-composite distinction is a cross-scale statement (SI S2.6.2): a real partition persists across ≥ 2 adjacent τ grid points; a uniform manifold's arc-partition does not. The persistence arbiter now **exists** (`stage1/persistence.py`; #28), and independently reproduces the qualitative discrimination (circle → no persistent split; hierarchical → persistent 3-way split). SI S2.6.1 documents the scope and S2.6.2 the operational signal: single-scale `Q` is a proposal screen and the cleanup passes are provisional operational stand-ins for persistence. The alternative M4 S3.4 Dirichlet–multinomial gate supplies the complementary non-degenerate likelihood-ratio null.
- **Remaining code work (blocked on wiring persistence into the accept path).** `clustering.py` still carries the provisional heuristics with dataset-motivated constants: boundary-refine `eta = 0.3` (`_refine_boundaries`), the tiny-satellite absorbers (`_absorb_tiny_clusters_into_dominant`, `_absorb_one_tiny_satellite`, `_absorb_full_graph_isolates`), and the `<= 3`-fragment collapse in `run_clustering`. The persistence signal exists but is not yet the acceptance arbiter inside `run_recursive_discovery` / `run_clustering`. Next step: gate a proposed split on persistence (accept only if it persists ≥ `P_persist` grid points), confirm the circle=1 / swiss≤3 / hierarchy(6-leaf) regressions hold under that gate, *then* delete the stand-in heuristics. Deleting them now (without the gate wired in) regresses circle→4 and swiss→4 (measured). The four previously-dead helpers are already removed.
- **Paper/SI prose** should describe the implemented AP -> Q-merge -> refine pipeline (the Leiden detour is obsolete). The S2.6.1 scope paragraph now covers why uniform manifolds and bump-on-background cases need different cleanup passes; paper §3 prose still needs a one-line pointer to S2.6.2 persistence as the cluster-count arbiter.

## 28. Scale selection: retire the load-band heuristic (default flip remaining)

Deeper diagnosis than the original `c_{d,k}` framing: the raw Lindeberg response `R_i(tau) = (sqrt(tau)/c_{d,k})^d * rho_hat_i` is **self-normalizing at equilibrium** (a converged scaffold locks `r_k ~ c_{d,k} * sqrt(tau)`, cancelling the tau-dependence), so the characteristic-scale signal must come from quantities equilibration cannot normalize away.

Landed behind the `selector` flag (SI S2.5.1 / S2.6.2 / S14.3 documented; default still `"load_band"`):
- **Primary — variance-load crossing** (`selector="load_crossing"`): the scale where mean load `E[sigma^2/tau]` crosses the cap (`load = 1`, i.e. residual variance equals the cap), log-interpolated on the coarsest stabilized bracket. Target `1` is derived (it is the cap). Empirically dominates the band selector on tau* accuracy across circle/swiss/variable_density/nested_spheres (ratios ~0.7–1.9 vs the band's frequent ~8x overshoot); dedicated tests assert the dominance.
- **Combined arbiter** (`selector="combined"`): persistence scale for a splittable multi-modal region, else the load-crossing operating scale, else the legacy fallback — unifying tau* selection with recursion timing (S2.6.2).
- **Persistence** (`selector="persistence"`, `stage1/persistence.py`): sample-space partition tracking, Hungarian-matched cluster Jaccard, `P_persist=2`, `theta_ovl=0.5`. Discriminates uniform (circle → no persistent split) from multi-modal (hierarchical → persistent 3-way).
- The compensated node-count / support trace `N(tau)*tau^{d/2}` is documented as a **diagnostic, not the selector**: it is confounded once node count saturates the operational budget `N_max` (observed: hierarchical caps immediately, swiss caps mid-grid), so it does not dominate universally — this is why the load-crossing is the primary signal.

Remaining:
1. Migrate the recursion (`test_recursion.py`) and scenario (`test_circle.py`, `test_swiss_roll.py`, `test_hierarchy.py`) configs to `selector="combined"` one at a time, confirming tau*/leaf-count stability, then flip the `ScaleSearchConfig.selector` default from `"load_band"` to `"combined"`.
2. Once nothing selects `load_band`, delete `_select_characteristic_scale` (`band_lo = 0.65` + one-step-coarser patch) and the dormant `_legacy_slope_selector` / `_detect_peak` helpers.
3. `c_{d,k}` becomes a declared **calibration protocol** (uniform d-ball ensemble, median `r_k / sqrt(tau)` at equilibrium, tabulated over (d, k)) rather than an analytic constant.
- Exit: `band_lo` and the one-step-coarser patch removed from the production path; scale-search/scenario tests pass with materially tighter tau* tolerance (target within one grid step of geometric truth).

## 31. Recursion vs hierarchical GT: remaining harness follow-ups

- The structural bar and moment-matching harness are in place (`tests/harness/hierarchy_recovery.py`: Hungarian matching, Hotelling mean gate, Frobenius covariance gate; six-leaf regression passes).
- Remaining: (a) use per-level tau from each recursion frame (not only the root) when comparing deeper trees; (b) tighten gates once #32 fixes the canonical `tau -> Sigma_smooth` map (the harness currently uses provisional isotropic `tau * I`).

## 32. tau and Gaussian scale-space: the variance-cap / heat-kernel bridge

- Stage 1 `tau` is operationally a variance cap; the resolution theory (S2.8 and both paper propositions) treats it as a Gaussian convolution bandwidth on the latent density. The bridge is asserted, not derived, and the test harness uses provisional `Sigma_smooth = tau * I`.
- Recommended minimal resolution for Paper 1: state an equilibrium lemma — each settled catchment-conditional density approximates the latent density smoothed at bandwidth ~tau, up to the `c_{d,k}` calibration factor and curvature terms (S2.5.2 has the expansion machinery) — and declare `Sigma_smooth = tau * I` as the convention. Anisotropic / intrinsic scale-space (PCA or tangent-space metric from T2, semigroup on covariances) is explicitly future work.

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
