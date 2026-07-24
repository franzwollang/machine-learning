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

## 21. Paper prose for soft-mean / hard-variance asymmetry

- The SI is updated: S2.3.1 introduces the hard-Voronoi variance vs soft-routing mean asymmetry, S2.3.2 has the Steiner shift and the hard-Voronoi variance conservation law, and S4.4 T0 / S12.6 reference the shadow mechanism.
- Remaining: paper §3.2–3.3 still describes moment updates without the asymmetry. Add a brief statement (or an explicit deferral to SI S2.3.1) so the paper does not imply the variance is soft-kernel-weighted.

## 23. Reconcile tau / dimensionality notation across Paper 1 and SI

- One reconciliation pass remains: confirm paper prose, notation tables, SI S2.4, S2.5, S2.6, S4.4, and S8.4 all use the same convention — region-level `tau` set at region entry, uniform `tau_local` within a scaffold run, per-node `d_final` as diagnostic only, cluster/recursion-level scale selection via scale response + AP.
- None of these sections should imply per-node `d_final` rescales the split cap.

## 25. Circle mesh topology test

- The circle scaffold passes node-count and reconstruction-error assertions but lacks an explicit topology check that the lifted-edge graph is a single connected 1-ring (Betti_0 = 1, Betti_1 = 1).
- Options: (a) Vietoris–Rips persistent homology on node positions (`giotto-tda` / `ripser`), (b) flag-complex Betti numbers via `gudhi`, (c) simple graph checks (connected components + cycle rank) on the lifted graph directly.
- Blocked on / naturally lands with Stage 2 flag-complex construction; option (c) could land earlier as a Stage 1 test.

## 26. Manifold-zoo junction test (circle + line + plane + box)

- Classic GNG benchmark: 1D circle, 1D segment, 2D plane patch, and 3D box meeting at dimensional junctions.
- Tests mesh quality per patch, `d_final` accuracy at junctions, junction detection (S8.4), and Stage 2 heterogeneous simplex dimension (S4.2).
- Needs a dataset generator in `tests/datasets/synthetic/` with per-component intrinsic-dim ground truth. Generator can land early as a diagnostic fixture; scenario assertions are deferred until S8.4 is implemented.

## 27. Clustering: canonicalize the Q-score and remove cleanup heuristics

The AP -> Q-merge -> refine pipeline is implemented and passes the circle, swiss-roll, and hierarchical-Gaussian regressions (six terminal leaves). Recursion is Q-gated (`recursion.py`: leaf when `n_clusters <= 1` or `partition_q_score <= 0`). What remains is making the spec canonical and the implementation heuristic-free:

- **SI S2.6.1 must pin down the Q primitives.** `LocalIntra`, `BoundaryInter`, and the scale-conditioned edge evidence `W_v(i,j) = K_v(i,j) * A_sym(i,j)` are currently prose-level; the exact formulas live only in `reference/stage1_clustering_and_resolution.md` and in code. Promote them into the SI so the implementation is verbatim-checkable.
- **Single-cluster null as first-class candidate.** Uniform single-component manifolds (circle) currently rely on isolate absorption plus a final "collapse if <= 3 fragments" pass. The principled form: any proposed partition must beat the one-cluster null by a margin under the same Q (or DM-evidence) criterion.
- **Replace residual cleanup passes with one rule.** `clustering.py` carries dataset-motivated constants (`tiny_max = 14`, boundary-refine `eta = 0.3`, imbalance `r >= 0.22`, final collapse when `<= 3` clusters) plus several apparently dead helpers (`_merge_tiny_lifted_into_full_graph_neighbor`, `_coalesce_if_marginal_k_way_split`, `_maybe_rebalance_two_cluster_partition`, `_legacy_slope_selector`, `_detect_peak`). Target: a single Q-improving pairwise merge applied to fixpoint plus the null test. If the existing regressions cannot pass under that single rule, the Q definition (likely the missing kernel term `K_v`) is what needs fixing — not more cleanup.
- **Paper/SI prose** should describe the implemented AP -> Q-merge -> refine pipeline (the Leiden detour is obsolete), and explain why uniform manifolds and bump-on-background cases previously needed different cleanup passes.

## 28. Scale selection: response degeneracy and the load-band heuristic

- Deeper diagnosis than the original `c_{d,k}` framing: the raw Lindeberg response `R_i(tau) = (sqrt(tau)/c_{d,k})^d * rho_hat_i` is **self-normalizing at equilibrium**. A converged scaffold equalizes hit masses and locks the k-NN radius to the cap (`r_k ~ c_{d,k} * sqrt(tau)`), so the tau-dependence cancels and the trace is flat/monotone regardless of how well `c_{d,k}` is calibrated. The characteristic-scale signal must come from quantities equilibration cannot normalize away.
- Current selector (`stage1/controller.py`) is an empirical proxy: coarsest stabilized grid point with variance load in `[0.65, 1.0]`, plus a "one step coarser" patch when the band has a single member. Tests only require `0.5 < tau*/expected < 10`.
- Rework direction:
  1. Primary signal: the compensated node-count / support trace — on a d-dimensional support, equilibrium node count scales as `N(tau) ~ V_supp / tau^{d/2}`, so knees/plateaus in `N(tau) * tau^{d/2}` (equivalently `V_C(tau)`) mark characteristic scales. The dormant `_legacy_slope_selector` and the S2.5 support trace were both circling this.
  2. Secondary signal: Q-partition persistence — the coarsest scale at which the AP+Q partition yields a coherent multi-cluster split that persists across >= 2 adjacent grid points. This operationalizes the S2.6.2 persistence theory and unifies tau* selection with recursion timing.
  3. `c_{d,k}` becomes a declared **calibration protocol** (uniform d-ball ensemble, measure median `r_k / sqrt(tau)` at equilibrium, tabulate over (d, k)) rather than an analytic constant.
- Exit criterion: `band_lo = 0.65` and the one-step-coarser patch removed; scale-search tests pass with materially tighter tau* tolerance bands; SI S2.5/S2.5.1 rewritten to match.

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
