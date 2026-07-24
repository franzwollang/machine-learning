# Proteus Codebase TODOs (Stage 1 focus; Stage 2 noted)

This document tracks missing features and current simplifications relative to the draft paper and SI. For each item:

- Current: what the code does now
- Should: what we want (concrete behavior)
- Notes: implementation guidance (APIs, data flows, edge cases)

---

## 1) Neighbour search backend (ANN)

- Current:
  - `_query_neighbors` does a naive O(N) scan per query.
  - Works for tests; slows on larger N.
- Should:
  - Add an optional HNSW (hnswlib) backend for k-NN queries.
  - Keep the naive path as a fallback (no external dep; parity in tests).
- Notes:
  - Build index once per Stage 1 `fit` and refresh only after deferred moves or splits that change positions.
  - Typical HNSW params: `M∈[16,32]`, `efConstruction∈[100,400]`, `efSearch∈[50,200]` (increase near convergence).
  - API sketch:
    - Stage1 `__init__`: `use_ann: bool=False`, `ann_params: dict|None`.
    - On `_initialise_nodes`: build/populate index if `use_ann`.
    - On `_apply_deferred_nudges` and `_split_node`: update index for the affected node(s).

---

## 2) CV convergence with neighbour-normalised incoherence (ρ̃)

- Current:
  - `_compute_cv` uses ρ/mean(ρ) (global normalisation).
- Should:
  - Use neighbour-normalised ρ̃ per SI: divide each node’s ρ by the mean ρ of its neighbours (1-hop) with ε guard. Compute CV over ρ̃.
- Notes:
  - Use `neighbour_graph()` to get adjacency. For isolated nodes, fall back to global mean.
  - Keep tolerance at 0.01 (configurable `cv_tolerance`).

---

## 3) Link accounting granularity (rank-weighted, k-neighbour directed counters)

- Current:
  - Only the BMU pair gets a link hit per sample; increments by 1.0.
- Should:
  - Accumulate directed counters to the full k-neighbourhood from the BMU: for BMU index `i` and its ranked neighbour `j` at rank `r`, increment `C(i→j) += 2^{-(r-1)}`. Maintain link creation/protection as today.
- Notes:
  - Keep counters float; maintain `protected_until` to avoid premature pruning.
  - This enables later Wilson-based pruning to have informative proportions.

---

## 4) Link pruning gauntlet (power shield + Wilson + bilateral vote)

- Current:
  - Simple: prune links with `total_hits < min_link_strength` after protection window.
- Should:
  - Stage 1 gauntlet:
    1. Power shield: require effective sample size `n_eff = C(i→j)+C(j→i)` ≥ `n_req` (config) before a link is eligible.
    2. Relative weakness: per node `i`, compute Wilson one-sided upper bound for each outgoing link proportion `p_ij = C(i→j)/Σ_k C(i→k)`. Vote to prune links with `ν⁺ < median_peer`.
    3. Bilateral agreement: remove link only if both endpoints vote prune.
- Notes:
  - Wilson upper bound helper:
    ```python
    def wilson_upper(p: float, n: float, z: float = 1.2816) -> float:
        denom = 1.0 + (z*z)/n
        centre = p + (z*z)/(2.0*n)
        rad = z * ((p*(1.0-p)/n + (z*z)/(4.0*n*n)) ** 0.5)
        return (centre + rad) / denom
    ```
  - Keep `min_link_strength` as a coarse safety net; gate decisions via power shield.

---

## 5) Split budget guard (2× prune rule)

- Current:
  - A node splits when its mean variance exceeds `tau_local`; no hit-budget check.
- Should:
  - Permit split only if parent hits satisfy `h_parent ≥ 2 * h_prune`, where `h_prune = c_p * mean_hits` over active nodes; default `c_p = 0.20`.
- Notes:
  - Compute `mean_hits` over nodes with `update_count ≥ prune_after` to avoid early bias.
  - If the budget is not met, defer split (node continues updating statistics).

---

## 6) τ / τ_local mapping with local dimension estimate (optional)

- Current:
  - `tau_local = tau / d_subspace` (single value).
- Should:
  - Allow optional per-node `d_final_i` (smoothed) and set `tau_local_i` proportionally (e.g., `tau / max(d_final_i,1)`).
- Notes:
  - Approximate `d_final_i` via node degree minus 1, optionally smoothed with neighbour median; keep simple for Stage 1.
  - Expose a flag `use_local_dim: bool=False`.

---

## 7) Coarse scale search over τ (grid; optional GP refine later)

- Current:
  - Run Stage 1 once at `tau = mean var(data)`.
- Should:
  - In `ProteusAlgorithm`, evaluate a geometric grid of `tau_i = tau_0 * r^j` with `r = 1/√2` (7–9 points/decade); for each, run Stage 1 to equilibrium and record a response Φ(τ).
  - Define Φ(τ):
    - Default: `Φ(τ) = - mean_min_distance(subset, stage1_nodes)` (higher is better). This reuses the quantisation error already computed in the wrapper.
    - Optional (later): swap to the γ=1 normalised response from SI to align with theory without changing the grid logic.
  - Bracket choice: detect the first local maximum by sign flip of the second difference in log-τ space: `Δ²Φ(τ_j) = Φ(τ_{j+1}) - 2Φ(τ_j) + Φ(τ_{j-1})`; choose `τ* = τ_j` at the first negative→positive flip.
  - Apply per-node/partition before attempting a split; pass `tau=τ*` into `Stage1.fit` for that subset and cache `τ*` on the hierarchy node for reuse.
- Notes:
- Keep it coarse; add GP refinement later only within the first bracket.
- Cache `(τ, Φ(τ))` per subset signature to avoid duplicate runs during recursion.
- Guard wall-time: retain `max_epochs` and CV tolerance to exit quickly on unhelpful τ.

---

## 8) Masking for missing modalities / structured zeros (optional)

- Current:
  - Residuals and distances computed on all dimensions; zeros treated as valid values.
- Should:
  - Accept an optional `mask: np.ndarray[bool]` per sample (or infer `x != 0`); compute errors/distances only on active dims; zero out updates on inactive dims. Optionally normalise updates by active count.
- Notes:
  - Minimal invasive changes:
    - In `_apply_rank_weighted_updates`, compute `error = (x - position)`; if a mask is provided for this sample, set `error[~mask] = 0` and scale by `1/max(1, mask.sum())`.
  - For distance queries, apply mask when computing deltas; if masks vary per sample, fall back to unmasked distances unless an explicit mask is passed in.

---

## 9) CV tolerance self-checks & metrics

- Current:
  - Tests assert CV decreases and ends below coarse thresholds.
- Should:
  - After switching to neighbour-normalised CV, update tests to check that `last_cv` decreases and is below `cv_tolerance` (default 0.01–0.02) on synthetic datasets.
- Notes:
  - Keep legacy assertions with relaxed bounds behind a flag during migration.

---

## 10) Stage 2 (not implemented here)

- Current:
  - No Stage 2: no simplices, torsion ladder, dual flow, or shape-quality checks.
- Should:
  - Out of scope for this TODO sprint. Keep the wrapper’s `run_stage2` disabled.
- Notes:
  - When implemented, promote Stage 1 stats with warm start; seed simplices (Greedy Chaining); add torsion audit and optional mini-flows.

---

## 11) Link creation scope (clarification)

- Current:
  - Link created between BMU1 and BMU2 only.
- Should:
  - This is acceptable for Stage 1. No change required.
- Notes:
  - With rank-weighted link counters (Item 3), the BMU-centred directed statistics become richer without densifying the graph.

---

## Test additions (after features land)

- Add unit tests for:
  - Neighbour-normalised CV path (monotone decrease; threshold pass on circle/swiss-roll).
  - Link pruning gauntlet: power shield gating and Wilson-based prune decisions (use synthetic traffic patterns to assert bilateral pruning only on weak links).
  - Split budget guard (2× prune rule): verify deferred split when parent hits are insufficient.
  - Scale grid search: on mixtures, τ\* chosen near the best quantisation response among grid points.
  - Leiden split: verify communities satisfy `min_cluster_size` and `split_balance_fraction`; data→community mapping via nearest Stage 1 node is consistent.

---

## 13) Analytic node-count estimates in gaussian_efficiency (diagnostics only)

- Current:
  - `test_gaussian_efficiency.py` logs K (node count), coverage ratio r̄/S, and a crude geometric lower bound N_min; it does not contextualize K against principled target ranges.
- Should:
  - Add two principled, analytical estimates for the “ideal” number of nodes to achieve a target RMS quantisation error fraction `e` on synthetic Gaussians:
    1. High‑rate vector quantization lower bound (optimistic codebook):
       - Distortion D ≈ G_d · N^(−2/d) ⇒ r_rms ≈ √G_d · σ · N^(−1/d)
       - For target RMS fraction e = r_rms/σ: `N_low ≈ (√G_d / e)^d`
       - Use tabulated `G_2≈0.0802`, `G_3≈0.0785`; for d>3, use `G_d≈0.08` as a proxy.
    2. Conservative geometric covering bound (finite coverage inside a Kσ mass ball):
       - Let K_eff ≈ 1.5–2.0 for 95–99% mass. With packing/covering inefficiency η_d (2D hex packing η_2≈0.9069),
         `N_high ≳ (K_eff/e)^d / η_d`. (Increase 10–30% if treating covering vs packing strictly.)
  - Compute both bounds for the test’s chosen `e` (use the observed r̄/S or a configured `e≈0.08–0.10`).
  - Report (and optionally warn) when observed K lies far below N_low (underfitting) or far above N_high (overgrowth at selected τ).
  - Do not change τ or set τ_center from these analytics; they are diagnostics for synthetic data to surface easy issues and guide fixes.
- Notes:
  - Implementation in `test_gaussian_efficiency.py`:
    - Add helpers to compute N_low, N_high per (d, e, K_eff, η_d).
    - Log both bounds alongside existing rows; append a simple ratio `K / midpoint(N_low,N_high)` in log.
    - Start with warnings only (no hard asserts) to avoid flakiness; later, consider soft thresholds (e.g., expect `K` within [0.5·N_low, 2.0·N_high]) for stable d=2 cases.
  - Rationale:
    - These bounds provide a sanity window for Stage 1 growth on controlled Gaussians, helping detect mis‑tuned τ, missing split guards, or a non‑convergent CV policy—without introducing new runtime hyperparameters or relying on unavailable real‑world quantities.

---

## 12) Replace K-means split with Leiden community detection on Stage 1 graph

- Current:
  - Hierarchy splits use 2-means (`_kmeans2`) on Stage 1 node positions, then refine on data.
- Should:
  - Build a weighted, undirected Stage 1 node graph with edge weight `w_ij = C(i→j)+C(j→i)` for links that survive protection/pruning.
  - Run Leiden to obtain 2+ communities; auto-tune resolution to satisfy `min_cluster_size` and `split_balance_fraction` (existing constraints). If libraries unavailable, fall back to 2-means with a clear TODO.
  - Map data to communities via nearest Stage 1 node; derive `child_indices` from assignments; keep existing guards.
- Notes:
  - Resolution auto-tune: start at 1.0; if the largest community violates balance/size, increase (×2); if too fragmented, decrease (÷2); cap at ≤ 6 tries.
  - Ties to Item 3: richer rank-weighted counters improve community evidence but are not required to ship this change.
  - Keep the number of free knobs fixed by reusing existing constraints; do not introduce new thresholds.
