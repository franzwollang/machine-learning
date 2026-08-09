"""Recursive scale-discovery orchestrator for Stage 1 (SI S2.5, S4.4, S2.6.2)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional

import numpy as np

from proteus.stage1.clustering import (
    ClusterResult,
    _lifted_components_covering_all_nodes,
    compute_edge_weights,
    partition_q_score,
    run_clustering,
)
from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.dm_cluster import (
    DMClusterConfig,
    dm_partition_verdict,
    run_clustering_dm,
)
from proteus.stage1.edge_evidence import (
    HollowEdgeConfig,
    a4_roc_primary_config,
    prune_hollow_edges,
)
from proteus.stage1.pruning import demote_lifted_by_cluster
from proteus.stage1.transfer import apply_t2_transfer


@dataclass
class RecursionConfig:
    """Configuration for the recursive scale-discovery loop.

    ``require_persistent_split`` turns on the SI S2.6.2 **feature-persistence
    accept-gate**: a region's proposed multi-cluster split is accepted only if a
    multi-cluster partition *persists* across adjacent ``tau`` grid points (the
    cross-scale arbiter), rather than trusting a single-scale partition alone.
    When enabled the scale search is forced to record partitions and select the
    persistence scale (``selector="persistence"``); a region whose split does not
    persist is treated as terminal (a single intrinsic feature).  This is the
    *intended* canonical replacement for the single-scale cleanup heuristics of
    S2.6.1 (OPEN_ISSUES #27).  The persistence rule is **coarse-anchored**
    (S2.6.2), which removes the warm-start fine-end false positive: with the
    S2.6.1 stand-ins removed and this gate on, the circle collapses to a single
    leaf and the hierarchical Gaussian keeps its six leaves.  It is **not yet safe
    to delete those stand-ins**, however: a developable manifold (swiss roll)
    whose coarsest partition is a few arcs still over-fragments under the gate
    alone, pending the Stage 2 DM evidence gate (S3.4, M4).  See the "residual
    limitation" note in SI S2.6.2.  Default off during the M2 transition.

    ``require_dm_split`` turns on the **DM cluster-acceptance path** (SI S3.4
    reduction, proposed S2.6.3, OPEN_ISSUES #27): AP proposals are merged by
    the Dirichlet--multinomial block-homogeneity Bayes factor
    (:func:`proteus.stage1.dm_cluster.run_clustering_dm`) instead of the
    S2.6.1 single-scale cleanup stand-ins, and a region's split is retained
    only if the K-way block partition clears the ``log(tau_bf)`` margin
    against the one-feature null -- the non-degenerate likelihood-ratio null
    the graph-local ``Q`` cannot provide.  Independent of and complementary to
    persistence (the two flags may be combined).  It is a **proposed /
    operational** path pending validation, and is **not** a licence to delete
    the S2.6.1 stand-ins.  Default off.

    ``allow_finer_research`` (OPEN_ISSUES #44, **proposed / operational,
    default off**) enables a single finer-than-``tau*`` re-search when the
    region's characteristic-scale partition is ``K<=1`` (or ``Q<=0``), or when
    persistence finds no split at the coarse grid.  Without this flag the
    orchestrator treats such a region as terminal and never probes scales where
    disconnected sub-structure (nested shells, linked tori) becomes visible.
    When enabled, the re-search caps ``tau_max`` at
    ``tau_star * finer_tau_cap_ratio`` (strictly ``< tau_star``) and reuses the
    existing persistence / DM accept gates on the finer proposal.  **Stop
    guarantee:** at most one finer attempt per region invocation; if the
    capped search still yields ``K<=1`` / ``Q<=0`` or the gate rejects, the
    region is terminal; ``min_samples`` / ``max_depth`` still bound the tree.

    **Recommended pairing for uniform manifolds (required):** enable
    ``require_persistent_split`` together with ``allow_finer_research``, keep
    ``max_finer_scale_steps <= 4``, and ``min_samples >= 80``.  A2-T3 measured
    circle/swiss = 1 leaf under that pairing; flag alone over-fragments the
    circle (~21 leaves).  A2-T4: the same recommended pairing still yields
    **1 leaf** on nested_spheres / linked_tori (gt cc=2) at n≈160–240 — shell
    recovery is **not** solved by this pairing; deeper walks / prepass tend to
    over-fragment with near-zero ARI.  Do not flip awaiting component tests.

    **A2-T7/T8 pairing studies (measurement only):** with the unit-test scale
    harness (``n_seeds=8``), ``persist=True`` keeps circle = 1 leaf across
    ``prefer_disconnected_prepass`` on/off, ``finer_prepass_min_frac`` in
    ``{0.15,0.2,0.3,0.4}``, and ``max_finer_scale_steps`` in ``{4,8,12}``.
    Dropping persist still false-hits (~16–21 leaves) even with prepass.
    Harness caveat: leaner ``n_seeds=6`` can yield circle 2–7 leaves under the
    same persist pairing — match the unit-test envelope when judging shatter.
    On nested_spheres none of persist±prepass±``require_dm_split`` recovered
    gt cc=2 with ARI>0.5 (steps≤8 → 1 leaf; deeper / dm-without-persist →
    5–9 leaves, ARI≲0.09) until ``prefer_signal_density_band_prepass``
    (knn×radial keep; A2-T15) recovered nested unit harness at steps≥8
    (2 leaves ARI=1.0).  A2-T18: linked_tori still 1 leaf under that path
    (radial origin unsuitable for offset rings).  A2-T19: swiss stays 1 leaf
    at ``max_finer_scale_steps<=4`` with signal-density; steps=8 shatters.
    ``finer_signal_density_keep_frac=0.55`` is the measured sweet spot
    (0.4 misses nested; 0.8 over-fragments).  Hold awaiting flips until A1
    confirms + a non-radial tori path exists.
    Optional SI prose for gate-owns-stop + persist pairing: see A3 mailbox
    Opt A+C sketch (REQUEST_TRACKER).

    ``max_finer_scale_steps`` bounds how many successive geometric shrinks of
    the ``tau_max`` cap are attempted inside one region (each step multiplies
    the cap by ``finer_tau_cap_ratio``).  Nested multi-component scenes can
    require many grid steps below the coarse ``tau*`` (OPEN_ISSUES #44
    evidence: ~80x).  Default ``8`` is an operational budget; the flag remains
    off so the default acceptance path is unchanged.

    ``prefer_disconnected_prepass`` (OPEN_ISSUES #44c, **proposed /
    operational, default off**) short-circuits the finer re-search walk when
    the lifted Hebbian graph at a capped scale has **≥2 major connected
    components** (each at least ``finer_prepass_min_frac`` of the scaffold
    nodes, and at least 3 nodes).  Tiny components are absorbed into the
    nearest major by Euclidean position.  Zero inter-component lifted edges
    imply block-diagonal flow, so this is the cheap obvious-disconnect path
    before the general AP/DM finer search.  Pair with a modest
    ``max_finer_scale_steps`` (and preferably ``require_persistent_split``)
    so uniform manifolds that only fracture at extreme fine scales do not
    trigger a false prepass hit.

    **A2 diagnostic (lifted-CC vs shells):** on nested_spheres the major-lifted
    prepass usually misses concentric shells because the lifted graph stays
    **one CC** (or splits into tiny noise fragments) across the finer walk —
    shells remain radius-bridged.  A radial gap in distance-from-centroid on
    scaffold positions recovers shell membership on clean two-ring unit
    scaffolds (``prefer_radial_gap_prepass``, default off).  E2E (n_seeds=8):
    persist+radial+``steps<=4`` keeps circle=1 but nested stays 1 leaf;
    deeper steps (8–12) shatter circle (10–22 leaves) and yield nested 3–4
    leaves with ARI≈0 — still not shell recovery.  Hold awaiting / SI A+C.

    **A2-T10 scaffold probe (n=160 nested, 12 recurse caps):** ``n_cc`` is 1
    at most caps; when ``n_cc>=2`` sizes are noise fragments (e.g. 60+4), never
    two majors.  Cross-shell lifted edges persist (≈7–38) and mid-radius
    bridge nodes (≈3–9) keep shells in one component.  Plain radial-gap never
    fires: the largest balanced gap sits in a tissue/mid-band continuum
    (``gap_ratio`` typically < ``finer_radial_min_gap_ratio``) or is an
    unbalanced tail (sides ≈62+2).  Dropping mid-band / shell-band-only nodes
    before the gap search recovers shell ARI≈0.93–1.0 with ``Q>0`` on the
    same scaffolds — hence ``prefer_radial_band_prepass`` (histogram-trough
    exclusion, default off).  ``hit_count`` alone is a weak tissue filter.

    **A2-T13 strengthen (flag-gated, default off):**
    ``prefer_noncentroid_radial_band_prepass`` re-runs the band prepass with a
    coordinate-median origin (resists dense one-sided tissue that collapses
    Weiszfeld) and applies a relative trough-depth / bimodality gate
    (``finer_radial_min_trough_rel``, default ``0.35`` on that path).  The
    plain ``prefer_radial_band_prepass`` path stays unchanged unless
    ``finer_radial_min_trough_rel`` is set ``> 0`` explicitly.

    **A2-T14 (flag-gated, default off):**
    ``prefer_signal_density_band_prepass`` keeps the top
    ``finer_signal_density_keep_frac`` of nodes by knn-density × radial-hist
    signal score, then runs the band gap on the kept subset (apply cut to
    full scaffold).  Do **not** divide by ``rho_radial`` — that upweights
    sparse mid-continuum tissue and regresses nested-shell recovery.

    **A2-T21 (flag-gated, default off):**
    ``prefer_pca_axis_gap_prepass`` is the non-radial dual of the plain
    radial gap: project scaffold positions onto the leading principal
    component and take the largest balanced 1-D gap (same size /
    ``min_gap_ratio`` / ``Q>0`` gates), plus a centroid-separation gate
    ``||c0-c1||/(rms0+rms1) >= 1`` that rejects PC1 diameter cuts on
    concentric scaffolds.  Recovers laterally **offset** rings on unit
    scaffolds where radial-from-origin is the wrong cue; interlocking
    linked_tori still unrecovered under e2e persist+pca (geometry
    interpenetration) — do not flip awaiting.

    **A2-T24 (flag-gated, default off):**
    ``prefer_tube_major_radius_prepass`` assigns nodes by tube residual to a
    Hopf-linked major-circle pair (axis-aligned template over coordinate
    permutations): circle A in a coordinate plane about the origin, circle B
    translated by ``R`` into the orthogonal plane.  Recovers interlocking
    thin rings / thick tori on unit scaffolds in the synthetic linked_tori
    pose; concentric shells and laterally offset rings are the wrong cue
    (should miss or fail ``Q``).  E2e linked_tori recovery is **not**
    claimed — do not flip awaiting.

    **A2-T25 (flag-gated, default off):**
    ``prefer_spectral_gap_prepass`` bipartitions via the Fiedler vector of
    the normalised Laplacian of the lifted neighbour graph (fallback:
    position kNN).  Alternate linking-structure cue; report circle / swiss /
    nested regressions under the flag — default remains off.

    **A2-T28 (flag-gated, default off):**
    ``prefer_hollow_edge_prepass`` cuts lifted edges whose data-side
    hollowness ratio ``H = n_mid / n_end`` is below ``hollow_h0`` (Gabriel
    empty-diameter fallback when endpoint mass is below
    ``hollow_min_end_count``), then takes major connected components of the
    pruned graph.  This is the #44 empty-region / support-topology path
    (theory note ``empty_region_evidence_and_scale.md``): disconnection is
    scale-free, so the prepass runs at the region's own ``tau*`` and does
    **not** require finer-scale descent.  A2-T27 probe: seed-0 nested+tori
    major-CC hit near ``mid_radius_frac=0.35`` / ``h0=0.35``; multi-seed
    fragile and ``h0`` uncalibrated — do **not** flip awaiting.  A2-T30:
    multi-tau scan + fixed-tau ``K=2`` majors have sample ARI≈chance
    (Gabriel-driven at probe taus; empty-ball / non-cut-set); treat as
    diagnostic only.  A2-T31: ``hollow_require_gabriel_and_h`` (default
    False) cuts only when ``H < h0`` ∧ Gabriel-empty — suppresses
    Gabriel-only spurious majors on the probe grid (A4 sheet q01≈0.57).
    A2-T32: ``hollow_require_persistent_agree`` (default False) additionally
    requires a persistent multi-cluster at the region's scale-search result
    before accepting a hollow prepass split.
    A2-T33: ``hollow_use_a4_primary`` (default False) applies A4 ROC primary
    ``(mid=0.5, h0=0.7, gabriel=False, min_end=0.5)`` instead of the
    operational hollow knobs — sheet-null safe ≠ sample-ARI recovery.
    A2-T34: ``hollow_mst_critical_only`` (default False) intersects hollow
    cuts with the Euclidean MST edge set (conservative bridge proxy).
    Capacity/flow follow-on: ``hollow_bridge_critical_only`` (default False)
    intersects with graph-theoretic bridges (true cut-set; stricter than
    MST).      A2-T37: ``hollow_soft_capacity_only`` (default False) intersects
    with high Brandes-betweenness edges (``hollow_soft_capacity_frac`` of
    max; continuous soft capacity / flow proxy).  A2-T39:
    ``hollow_soft_capacity_method`` selects ``"betweenness"`` (default) or
    ``"bridge_mass"`` (min-cut mass on bridges).  A2-T40: soft×persist_agree
    + ``hollow_soft_capacity_frac`` sweep — collapse / uniform-safe ≠
    sample-ARI recovery; defaults off.  A2-T41: soft×
    ``hollow_require_gabriel_and_h`` conjunction collapses nested+tori
    majors≤1 (soft alone still leaves tori chance-ARI K=2); not recovery.
    A2-T42: multi-seed soft_capacity_frac sweep (seeds 0..2) — tori
    chance-ARI K=2 seed-fragile; nested≤1; defaults off.
   
    **Recommended pairing (A2-T19/T20/T23):**
    - Uniforms (circle/swiss): ``require_persistent_split`` +
      ``allow_finer_research`` + ``max_finer_scale_steps<=4`` +
      ``min_samples>=80``; optional ``prefer_signal_density_band_prepass``
      with ``finer_signal_density_keep_frac=0.55`` (steps=8 + sd shatters
      swiss).
    - Nested shells (unit harness): same persist + ``allow_finer_research``
      + ``prefer_signal_density_band_prepass`` + ``steps>=8`` +
      ``keep_frac=0.55`` + ``min_samples=20`` → 2 leaves ARI=1.0.
    - Linked tori: still open; try ``prefer_pca_axis_gap_prepass`` (offset
      cue) or ``prefer_tube_major_radius_prepass`` (interlock cue) —
      e2e recovery not claimed.  Hollow-edge is the intended general
      replacement (flag off until calibrated).
    """

    scale_search: ScaleSearchConfig = field(default_factory=ScaleSearchConfig)
    min_samples: int = 100
    max_depth: int = 5
    r_min: int = 3
    explained_energy: float = 0.999
    require_persistent_split: bool = False
    require_dm_split: bool = False
    dm_cluster: DMClusterConfig = field(default_factory=DMClusterConfig)
    allow_finer_research: bool = False
    finer_tau_cap_ratio: float = 1.0 / np.sqrt(2.0)
    max_finer_scale_steps: int = 8
    prefer_disconnected_prepass: bool = False
    finer_prepass_min_frac: float = 0.2
    prefer_radial_gap_prepass: bool = False
    finer_radial_min_gap_ratio: float = 0.25
    prefer_radial_band_prepass: bool = False
    finer_radial_hist_bins: int = 16
    prefer_noncentroid_radial_band_prepass: bool = False
    finer_radial_min_trough_rel: float = 0.0
    prefer_signal_density_band_prepass: bool = False
    finer_signal_density_keep_frac: float = 0.55
    prefer_pca_axis_gap_prepass: bool = False
    prefer_tube_major_radius_prepass: bool = False
    finer_tube_min_residual_ratio: float = 0.15
    prefer_spectral_gap_prepass: bool = False
    finer_spectral_knn: int = 8
    prefer_hollow_edge_prepass: bool = False
    hollow_mid_radius_frac: float = 0.35
    hollow_h0: float = 0.35
    hollow_min_end_count: float = 0.5
    hollow_gabriel_fallback: bool = True
    hollow_require_gabriel_and_h: bool = False
    hollow_require_persistent_agree: bool = False
    hollow_use_a4_primary: bool = False
    hollow_mst_critical_only: bool = False
    hollow_bridge_critical_only: bool = False
    hollow_soft_capacity_only: bool = False
    hollow_soft_capacity_frac: float = 0.25
    hollow_soft_capacity_method: str = "betweenness"
    seed: int = 42


@dataclass
class RecursionNode:
    """A node in the recursion tree.

    ``sample_indices`` lists original dataset row indices belonging to this
    region (same order as rows of ``data`` passed into the frame that created
    the node); leaves partition ``0 .. n-1`` at the root.
    """

    region_id: int
    level: int
    parent_id: Optional[int]
    tau_star: Optional[float]
    n_samples: int
    dim: int
    n_clusters: int
    children: list[int] = field(default_factory=list)
    is_leaf: bool = True
    sample_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=int),
    )


@dataclass
class RecursionTree:
    """The full recursion tree produced by recursive discovery."""

    nodes: list[RecursionNode] = field(default_factory=list)

    @property
    def depth(self) -> int:
        if not self.nodes:
            return 0
        return max(n.level for n in self.nodes) + 1

    @property
    def leaves(self) -> list[RecursionNode]:
        return [n for n in self.nodes if n.is_leaf]


def _clusters_from_labels(labels: np.ndarray) -> list[set[int]]:
    """Group node ids by non-negative cluster label into member sets."""

    groups: dict[int, set[int]] = {}
    for node_id, lbl in enumerate(np.asarray(labels, dtype=int)):
        if lbl < 0:
            continue
        groups.setdefault(int(lbl), set()).add(int(node_id))
    return [groups[k] for k in sorted(groups)]


def assign_samples_to_clusters(
    data: np.ndarray,
    scaffold: "Stage1Scaffold",  # noqa: F821
    labels: np.ndarray,
) -> dict[int, np.ndarray]:
    """Map each data sample to its BMU's cluster label.

    Returns a dict mapping cluster label -> array of sample indices.
    """

    n_samples = data.shape[0]
    cluster_indices: dict[int, list[int]] = {}
    for i in range(n_samples):
        bmu_ids, _ = scaffold.ann.query_knn(data[i], k=1)
        bmu = int(bmu_ids[0])
        label = int(labels[bmu])
        cluster_indices.setdefault(label, []).append(i)
    return {k: np.array(v, dtype=int) for k, v in cluster_indices.items()}


def _cluster_scaffold(
    scaffold: "Stage1Scaffold",  # noqa: F821
    config: RecursionConfig,
):
    """Run AP or DM clustering on ``scaffold`` per config flags."""

    if config.require_dm_split:
        return run_clustering_dm(scaffold, config.dm_cluster)
    return run_clustering(scaffold)


def _major_lifted_component_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
) -> ClusterResult | None:
    """Return a partition from ≥2 major lifted components, else ``None``.

    OPEN_ISSUES #44c cheap prepass: when the lifted graph already separates
    into multiple large components, skip AP and treat those components as the
    candidate split.  Components smaller than
    ``max(min_abs, ceil(n * min_frac))`` are absorbed into the nearest major
    component by Euclidean node position.  Requires ``Q > 0`` on the resulting
    partition.
    """

    n = len(scaffold.nodes)
    if n < 2:
        return None
    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))

    graph_lifted = scaffold.links.neighbour_graph(n)
    comps = _lifted_components_covering_all_nodes(n, graph_lifted)
    majors = [c for c in comps if len(c) >= threshold]
    if len(majors) < 2:
        return None

    positions = np.asarray(
        [scaffold.nodes[i].position for i in range(n)], dtype=float,
    )
    major_centroids = [
        positions[sorted(c)].mean(axis=0) for c in majors
    ]

    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(majors):
        for m in members:
            labels[int(m)] = cid

    # Absorb non-major nodes into nearest major by centroid distance.
    leftovers = [c for c in comps if len(c) < threshold]
    for tiny in leftovers:
        for m in tiny:
            dists = [
                float(np.sum((positions[int(m)] - cen) ** 2))
                for cen in major_centroids
            ]
            labels[int(m)] = int(np.argmin(dists))

    clusters: list[set[int]] = [
        set(np.where(labels == cid)[0].tolist()) for cid in range(len(majors))
    ]
    # Drop empties (should not happen after absorb).
    clusters = [c for c in clusters if c]
    if len(clusters) < 2:
        return None

    W = compute_edge_weights(scaffold)
    pq = partition_q_score(clusters, n, W, graph_lifted)
    if pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters]
    # Relabel densely 0..K-1
    dense = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            dense[int(m)] = cid

    return ClusterResult(
        labels=dense,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=len(clusters),
        partition_q_score=float(pq),
    )


def _resolve_hollow_edge_config(
    config: RecursionConfig | None = None,
    *,
    mid_radius_frac: float = 0.35,
    h0: float = 0.35,
    min_end_count: float = 0.5,
    gabriel_fallback: bool = True,
    require_gabriel_and_h: bool = False,
    mst_critical_only: bool = False,
    bridge_critical_only: bool = False,
    soft_capacity_only: bool = False,
    soft_capacity_frac: float = 0.25,
    soft_capacity_method: str = "betweenness",
    use_a4_primary: bool = False,
) -> HollowEdgeConfig:
    """Build ``HollowEdgeConfig`` from recursion knobs (A2-T33/T34/T37/T39)."""

    if config is not None:
        use_a4_primary = bool(config.hollow_use_a4_primary)
        mst_critical_only = bool(config.hollow_mst_critical_only)
        bridge_critical_only = bool(config.hollow_bridge_critical_only)
        soft_capacity_only = bool(config.hollow_soft_capacity_only)
        soft_capacity_frac = float(config.hollow_soft_capacity_frac)
        soft_capacity_method = str(config.hollow_soft_capacity_method)
        mid_radius_frac = float(config.hollow_mid_radius_frac)
        h0 = float(config.hollow_h0)
        min_end_count = float(config.hollow_min_end_count)
        gabriel_fallback = bool(config.hollow_gabriel_fallback)
        require_gabriel_and_h = bool(config.hollow_require_gabriel_and_h)
    if use_a4_primary:
        return a4_roc_primary_config(
            require_gabriel_and_h=bool(require_gabriel_and_h),
            mst_critical_only=bool(mst_critical_only),
            bridge_critical_only=bool(bridge_critical_only),
            soft_capacity_only=bool(soft_capacity_only),
            soft_capacity_frac=float(soft_capacity_frac),
            soft_capacity_method=str(soft_capacity_method),
        )
    return HollowEdgeConfig(
        mid_radius_frac=float(mid_radius_frac),
        h0=float(h0),
        min_end_count=float(min_end_count),
        gabriel_fallback=bool(gabriel_fallback),
        require_gabriel_and_h=bool(require_gabriel_and_h),
        mst_critical_only=bool(mst_critical_only),
        bridge_critical_only=bool(bridge_critical_only),
        soft_capacity_only=bool(soft_capacity_only),
        soft_capacity_frac=float(soft_capacity_frac),
        soft_capacity_method=str(soft_capacity_method),
    )


def _hollow_edge_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    data: np.ndarray,
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
    mid_radius_frac: float = 0.35,
    h0: float = 0.35,
    min_end_count: float = 0.5,
    gabriel_fallback: bool = True,
    require_gabriel_and_h: bool = False,
    mst_critical_only: bool = False,
    bridge_critical_only: bool = False,
    soft_capacity_only: bool = False,
    soft_capacity_frac: float = 0.25,
    soft_capacity_method: str = "betweenness",
    use_a4_primary: bool = False,
    hollow_config: HollowEdgeConfig | None = None,
) -> ClusterResult | None:
    """Partition via hollow-edge pruning + major CCs (OPEN_ISSUES #44).

    Cuts lifted edges with data-side hollowness ``H < h0`` (Gabriel
    empty-diameter fallback when endpoint mass is low; optional
    ``require_gabriel_and_h`` conjunction), then applies the same
    major-component absorption / ``Q > 0`` gate as
    :func:`_major_lifted_component_partition`.  Returns ``None`` unless ≥2
    majors survive.

    Recovery claims must assert **sample ARI**, not major-CC count alone
    (A2-T30/T35): empty-ball Gabriel and non-cut-set redundant paths can
    yield ``K=2`` with ARI≈chance.
    """

    n = len(scaffold.nodes)
    if n < 2:
        return None
    data_arr = np.asarray(data, dtype=float)
    if data_arr.ndim != 2 or data_arr.shape[0] < 1:
        return None

    positions = np.asarray(
        [scaffold.nodes[i].position for i in range(n)], dtype=float,
    )
    edges = [
        (int(link.i), int(link.j)) for link in scaffold.links.lifted_links()
    ]
    cfg = hollow_config if hollow_config is not None else _resolve_hollow_edge_config(
        mid_radius_frac=mid_radius_frac,
        h0=h0,
        min_end_count=min_end_count,
        gabriel_fallback=gabriel_fallback,
        require_gabriel_and_h=require_gabriel_and_h,
        mst_critical_only=mst_critical_only,
        bridge_critical_only=bridge_critical_only,
        soft_capacity_only=soft_capacity_only,
        soft_capacity_frac=soft_capacity_frac,
        soft_capacity_method=soft_capacity_method,
        use_a4_primary=use_a4_primary,
    )
    kept = prune_hollow_edges(positions, edges, data_arr, config=cfg)

    graph_pruned: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, j in kept:
        if i == j:
            continue
        graph_pruned[i].append(j)
        graph_pruned[j].append(i)

    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))

    comps = _lifted_components_covering_all_nodes(n, graph_pruned)
    majors = [c for c in comps if len(c) >= threshold]
    if len(majors) < 2:
        return None

    major_centroids = [positions[sorted(c)].mean(axis=0) for c in majors]
    labels = np.full(n, -1, dtype=int)
    for cid, members in enumerate(majors):
        for m in members:
            labels[int(m)] = cid

    leftovers = [c for c in comps if len(c) < threshold]
    for tiny in leftovers:
        for m in tiny:
            dists = [
                float(np.sum((positions[int(m)] - cen) ** 2))
                for cen in major_centroids
            ]
            labels[int(m)] = int(np.argmin(dists))

    clusters: list[set[int]] = [
        set(np.where(labels == cid)[0].tolist()) for cid in range(len(majors))
    ]
    clusters = [c for c in clusters if c]
    if len(clusters) < 2:
        return None

    # Score Q on the *pruned* edge set: hollow bridges are exactly the
    # cross terms that would drive Q ≤ 0 on the raw lifted weights, even
    # though the data-side cut is the intended support split (A2-T27/T29).
    W_full = compute_edge_weights(scaffold)
    kept_set = {tuple(sorted(e)) for e in kept}
    W_pruned = {
        (i, j): w for (i, j), w in W_full.items()
        if tuple(sorted((i, j))) in kept_set
    }
    pq = partition_q_score(clusters, n, W_pruned, graph_pruned)
    if pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters]
    dense = np.full(n, -1, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            dense[int(m)] = cid

    return ClusterResult(
        labels=dense,
        exemplar_indices=np.asarray(exemplars, dtype=int),
        n_clusters=len(clusters),
        partition_q_score=float(pq),
    )


def _hollow_prepass_accepted(
    config: RecursionConfig,
    scale_result: "ScaleSearchResult | None",  # noqa: F821
    hollow: ClusterResult | None,
) -> bool:
    """Whether a hollow partition clears Q (+ optional persistence) gates."""

    if hollow is None or hollow.n_clusters <= 1 or hollow.partition_q_score <= 0.0:
        return False
    if not config.hollow_require_persistent_agree:
        return True
    if scale_result is None:
        return False
    persistence = getattr(scale_result, "persistence_result", None)
    return persistence is not None and persistence.tau_star_index is not None


def _radial_gap_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
    min_gap_ratio: float = 0.25,
) -> ClusterResult | None:
    """Return a 2-way partition from a large radial gap, else ``None``.

    OPEN_ISSUES #44 proposed prepass: concentric shells can remain a single
    lifted connected component while still separating in
    distance-from-centroid.  Sort scaffold node radii about the position
    centroid, take the largest gap that leaves both sides with at least
    ``max(min_abs, ceil(n * min_frac))`` nodes, and require
    ``gap / median(radius) >= min_gap_ratio`` plus ``Q > 0``.
    """

    n = len(scaffold.nodes)
    if n < 2 * max(int(min_abs), 1):
        return None
    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))
    if 2 * threshold > n:
        return None

    positions = np.asarray(
        [scaffold.nodes[i].position for i in range(n)], dtype=float,
    )
    centroid = positions.mean(axis=0)
    radii = np.linalg.norm(positions - centroid, axis=1)
    order = np.argsort(radii)
    gaps = np.diff(radii[order])
    best: tuple[float, int] | None = None
    for i, gap in enumerate(gaps):
        left = i + 1
        right = n - left
        if left < threshold or right < threshold:
            continue
        g = float(gap)
        if best is None or g > best[0]:
            best = (g, i)
    if best is None:
        return None

    med_r = float(np.median(radii)) + 1e-12
    if best[0] / med_r < float(min_gap_ratio):
        return None

    thr = 0.5 * (
        float(radii[order[best[1]]]) + float(radii[order[best[1] + 1]])
    )
    labels = (radii > thr).astype(int)
    clusters: list[set[int]] = [
        set(np.where(labels == cid)[0].tolist()) for cid in (0, 1)
    ]
    if min(len(c) for c in clusters) < threshold:
        return None

    graph_lifted = scaffold.links.neighbour_graph(n)
    W = compute_edge_weights(scaffold)
    pq = partition_q_score(clusters, n, W, graph_lifted)
    if pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters]
    return ClusterResult(
        labels=labels,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=2,
        partition_q_score=float(pq),
    )


def _pca_axis_gap_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
    min_gap_ratio: float = 0.25,
    min_centroid_sep_ratio: float = 1.0,
) -> ClusterResult | None:
    """Return a 2-way partition from a large PCA-axis gap, else ``None``.

    OPEN_ISSUES #44 / A2-T21 non-radial prepass: laterally offset rings
    (linked-tori geometry cue) do not separate in distance-from-centroid,
    but can separate along the leading principal axis of node positions.
    Center the scaffold, project onto PC1, take the largest balanced gap
    with the same size / ``gap / median(|proj|)`` / ``Q > 0`` gates as
    :func:`_radial_gap_partition`.  An additional **centroid-separation**
    gate ``||c0-c1|| / (rms0+rms1) >= min_centroid_sep_ratio`` (operational
    default ``1.0``) rejects PC1 diameter cuts on concentric / isotropic
    scaffolds that otherwise clear the 1-D gap ratio.
    """

    n = len(scaffold.nodes)
    if n < 2 * max(int(min_abs), 1):
        return None
    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))
    if 2 * threshold > n:
        return None

    positions = np.asarray(
        [scaffold.nodes[i].position for i in range(n)], dtype=float,
    )
    centered = positions - positions.mean(axis=0)
    # Leading principal component via thin SVD (n × d, d small).
    try:
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.size == 0:
        return None
    axis = vt[0]
    proj = centered @ axis
    spread = float(np.median(np.abs(proj))) + 1e-12
    order = np.argsort(proj)
    gaps = np.diff(proj[order])
    best: tuple[float, int] | None = None
    for i, gap in enumerate(gaps):
        left = i + 1
        right = n - left
        if left < threshold or right < threshold:
            continue
        g = float(gap)
        if best is None or g > best[0]:
            best = (g, i)
    if best is None:
        return None
    if best[0] / spread < float(min_gap_ratio):
        return None

    thr = 0.5 * (
        float(proj[order[best[1]]]) + float(proj[order[best[1] + 1]])
    )
    labels = (proj > thr).astype(int)
    clusters: list[set[int]] = [
        set(np.where(labels == cid)[0].tolist()) for cid in (0, 1)
    ]
    if min(len(c) for c in clusters) < threshold:
        return None

    # Offset vs diameter: require cluster centroids to be well separated
    # relative to within-cluster RMS radii (rejects concentric PC1 halves).
    c0 = positions[labels == 0].mean(axis=0)
    c1 = positions[labels == 1].mean(axis=0)
    rms0 = float(np.sqrt(((positions[labels == 0] - c0) ** 2).sum(axis=1).mean()))
    rms1 = float(np.sqrt(((positions[labels == 1] - c1) ** 2).sum(axis=1).mean()))
    sep = float(np.linalg.norm(c0 - c1))
    if sep / (rms0 + rms1 + 1e-12) < float(min_centroid_sep_ratio):
        return None

    graph_lifted = scaffold.links.neighbour_graph(n)
    W = compute_edge_weights(scaffold)
    pq = partition_q_score(clusters, n, W, graph_lifted)
    if pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters]
    return ClusterResult(
        labels=labels,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=2,
        partition_q_score=float(pq),
    )


def _tube_residuals_hopf_xy_yz(
    positions: np.ndarray, major_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tube distances to Hopf-linked major circles in xy and yz planes."""

    R = float(major_radius)
    xy = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    xy = np.maximum(xy, 1e-12)
    nearest_xy = np.column_stack(
        [R * positions[:, 0] / xy, R * positions[:, 1] / xy, np.zeros(len(positions))],
    )
    d_xy = np.linalg.norm(positions - nearest_xy, axis=1)

    yz = np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)
    yz = np.maximum(yz, 1e-12)
    nearest_yz = np.column_stack(
        [
            np.full(len(positions), R),
            R * positions[:, 1] / yz,
            R * positions[:, 2] / yz,
        ],
    )
    d_yz = np.linalg.norm(positions - nearest_yz, axis=1)
    return d_xy, d_yz


def _tube_major_radius_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
    min_residual_ratio: float = 0.15,
) -> ClusterResult | None:
    """Return a 2-way partition from Hopf-linked tube residuals, else ``None``.

    OPEN_ISSUES #44 / A2-T24 interlocking-tori cue: linked rings do not
    separate along a single radial or PC1 axis when they interpenetrate.
    Assign each scaffold node to the nearer of two major circles in the
    synthetic linked_tori pose (circle in a coordinate plane about the
    origin; sister circle translated by ``R`` into the orthogonal plane).
    Search coordinate permutations and translations that place a candidate
    ring centre at the origin (do **not** mean-centre the whole cloud —
    that breaks the Hopf template).  Requires balanced sizes, mean residual
    separation ``>= min_residual_ratio * R``, and ``Q>0``.  Ambient ``d>3``
    uses the leading three principal components (then re-translated).
    """

    n = len(scaffold.nodes)
    if n < 2 * max(int(min_abs), 1):
        return None
    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))
    if 2 * threshold > n:
        return None

    positions = np.asarray(
        [scaffold.nodes[i].position for i in range(n)], dtype=float,
    )
    if positions.ndim != 2 or positions.shape[0] != n:
        return None
    d = int(positions.shape[1])
    if d < 3:
        return None
    if d > 3:
        centered = positions - positions.mean(axis=0)
        try:
            _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        if vt.shape[0] < 3:
            return None
        # PCA coords are mean-centred; translation search below restores pose.
        base = centered @ vt[:3].T
    else:
        base = positions.copy()

    perms = (
        (0, 1, 2),
        (0, 2, 1),
        (1, 2, 0),
        (1, 0, 2),
        (2, 0, 1),
        (2, 1, 0),
    )
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for perm in perms:
        P0 = base[:, list(perm)]
        # Candidate origin = mean of points near the first plane (small |z|).
        z_scale = float(np.std(P0[:, 2])) + 1e-9
        near = np.abs(P0[:, 2]) <= max(z_scale, 1e-6)
        if int(np.sum(near)) < threshold:
            near = np.ones(n, dtype=bool)
        origin = P0[near].mean(axis=0)
        # Also try raw (no shift) and half-shift along x (Hopf second centre).
        cyl0 = np.sqrt(P0[:, 0] ** 2 + P0[:, 1] ** 2)
        R_guess = float(np.median(cyl0[near])) if np.any(near) else float(np.median(cyl0))
        origins = (
            np.zeros(3),
            origin,
            np.array([origin[0] - 0.5 * R_guess, origin[1], origin[2]]),
            np.array([0.5 * R_guess, 0.0, 0.0]),
        )
        for o in origins:
            P = P0 - o
            cyl = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
            z_s = float(np.std(P[:, 2])) + 1e-9
            w = np.exp(-((P[:, 2] / z_s) ** 2))
            R_w = (
                float(np.average(cyl, weights=w))
                if float(w.sum()) > 0
                else float(np.median(cyl))
            )
            R_med = float(np.median(cyl))
            for R in (R_w, R_med, 0.5 * (R_w + R_med)):
                if not (R > 1e-9):
                    continue
                d_xy, d_yz = _tube_residuals_hopf_xy_yz(P, R)
                labels = (d_yz < d_xy).astype(int)
                n0 = int(np.sum(labels == 0))
                n1 = int(np.sum(labels == 1))
                if n0 < threshold or n1 < threshold:
                    continue
                sep = float(np.mean(np.abs(d_xy - d_yz)))
                if sep / (R + 1e-12) < float(min_residual_ratio):
                    continue
                # Plane-thinness + cross-plane contrast: each tube's members
                # hug its plane, and the two groups must prefer *different*
                # planes (rejects concentric coplanar shells).
                z0 = float(np.mean(np.abs(P[labels == 0, 2])))
                z1 = float(np.mean(np.abs(P[labels == 1, 2])))
                x0 = float(np.mean(np.abs(P[labels == 0, 0] - R)))
                x1 = float(np.mean(np.abs(P[labels == 1, 0] - R)))
                # label0 → xy tube, label1 → yz tube
                ok = (
                    z0 <= 0.35 * R
                    and x1 <= 0.35 * R
                    and z0 < z1 - 0.05 * R
                    and x1 < x0 - 0.05 * R
                )
                if not ok:
                    # Swapped polarity.
                    ok_s = (
                        z1 <= 0.35 * R
                        and x0 <= 0.35 * R
                        and z1 < z0 - 0.05 * R
                        and x0 < x1 - 0.05 * R
                    )
                    if not ok_s:
                        continue
                    labels = 1 - labels
                bal = min(n0, n1) / float(n)
                score = sep * bal
                if best is None or score > best[0]:
                    best = (score, labels.copy(), P.copy(), float(R))

    if best is None:
        return None
    _score, labels, _P_best, _R_best = best
    clusters: list[set[int]] = [
        set(np.where(labels == cid)[0].tolist()) for cid in (0, 1)
    ]
    if min(len(c) for c in clusters) < threshold:
        return None

    graph_lifted = scaffold.links.neighbour_graph(n)
    W = compute_edge_weights(scaffold)
    pq = partition_q_score(clusters, n, W, graph_lifted)
    if pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters]
    return ClusterResult(
        labels=labels,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=2,
        partition_q_score=float(pq),
    )


def _spectral_gap_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
    knn: int = 8,
) -> ClusterResult | None:
    """Return a 2-way Fiedler bipartition of the lifted (or kNN) graph.

    OPEN_ISSUES #44 / A2-T25 spectral / linking cue: take the second-smallest
    eigenvector of the normalised Laplacian of the lifted neighbour graph
    (fallback: symmetrised position kNN if the lifted graph is too sparse),
    threshold at the median, and require balanced sizes plus ``Q>0``.
    """

    n = len(scaffold.nodes)
    if n < 2 * max(int(min_abs), 1):
        return None
    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))
    if 2 * threshold > n:
        return None

    # Dense adjacency from lifted neighbour lists.
    adj = np.zeros((n, n), dtype=float)
    graph = scaffold.links.neighbour_graph(n)
    for i, nbrs in graph.items():
        ii = int(i)
        for j in nbrs:
            jj = int(j)
            if 0 <= ii < n and 0 <= jj < n and ii != jj:
                adj[ii, jj] = 1.0
                adj[jj, ii] = 1.0

    # Fallback / blend: position kNN if lifted degree is too low.
    degrees = adj.sum(axis=1)
    if float(np.median(degrees)) < 2.0:
        positions = np.asarray(
            [scaffold.nodes[i].position for i in range(n)], dtype=float,
        )
        k = max(2, min(int(knn), n - 1))
        # Squared distances; exclude self.
        d2 = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        nn = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        adj_knn = np.zeros_like(adj)
        rows = np.repeat(np.arange(n), k)
        adj_knn[rows, nn.ravel()] = 1.0
        adj_knn = np.maximum(adj_knn, adj_knn.T)
        adj = np.maximum(adj, adj_knn)

    deg = adj.sum(axis=1)
    # Isolated nodes: spectral bipartition is undefined.
    if np.any(deg <= 0.0):
        return None
    # Normalised Laplacian L = I - D^{-1/2} A D^{-1/2}.
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    scaled = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    lap = np.eye(n) - scaled
    try:
        vals, vecs = np.linalg.eigh(lap)
    except np.linalg.LinAlgError:
        return None
    # Fiedler: second-smallest eigenvector (vals ascending).
    if vals.shape[0] < 2:
        return None
    fiedler = vecs[:, 1]
    med = float(np.median(fiedler))
    labels = (fiedler > med).astype(int)
    # Degenerate median cut (all equal).
    if len(set(int(x) for x in labels)) < 2:
        labels = (fiedler >= 0.0).astype(int)
    if len(set(int(x) for x in labels)) < 2:
        return None

    clusters: list[set[int]] = [
        set(np.where(labels == cid)[0].tolist()) for cid in (0, 1)
    ]
    if min(len(c) for c in clusters) < threshold:
        return None

    graph_lifted = scaffold.links.neighbour_graph(n)
    W = compute_edge_weights(scaffold)
    pq = partition_q_score(clusters, n, W, graph_lifted)
    if pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters]
    return ClusterResult(
        labels=labels,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=2,
        partition_q_score=float(pq),
    )


def _geometric_median(points: np.ndarray, *, max_iter: int = 32) -> np.ndarray:
    """Weiszfeld geometric median (kept for experiments; prefer coord median)."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("points must be a non-empty (n, d) array")
    if pts.shape[0] == 1:
        return pts[0].copy()
    x = pts.mean(axis=0)
    for _ in range(max_iter):
        diffs = pts - x
        dists = np.linalg.norm(diffs, axis=1)
        # Coincident points: stay put / skip zero-weight rows.
        mask = dists > 1e-12
        if not np.any(mask):
            return x
        weights = 1.0 / dists[mask]
        x_new = (pts[mask] * weights[:, None]).sum(axis=0) / weights.sum()
        if float(np.linalg.norm(x_new - x)) < 1e-9:
            return x_new
        x = x_new
    return x


def _coordinate_median(points: np.ndarray) -> np.ndarray:
    """Per-axis median origin — resists dense one-sided tissue clumps (#44)."""

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("points must be a non-empty (n, d) array")
    return np.median(pts, axis=0)


def _signal_density_residual_keep_mask(
    positions: np.ndarray,
    radii: np.ndarray,
    *,
    keep_frac: float = 0.55,
    hist_bins: int = 16,
    knn: int = 8,
) -> np.ndarray:
    """Keep nodes with high knn-density × radial-hist signal (#44 T14).

    Mid-radius continuum tissue often sits in low-count radius bins while
    shell arcs land in dense modes.  Operational score:
    ``rho_knn(i) * (rho_radial(r_i) + eps)``; keep the top ``keep_frac``.
    (Dividing by ``rho_radial`` falsely upweights sparse mid bins and
    regresses nested-shell e2e recovery — see A2-T15 / A2-T18.)
    """

    pts = np.asarray(positions, dtype=float)
    r = np.asarray(radii, dtype=float)
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    frac = float(keep_frac)
    if not (0.0 < frac <= 1.0):
        frac = 0.55
    k = max(1, min(int(knn), n - 1)) if n > 1 else 1
    if n == 1:
        return np.ones(1, dtype=bool)

    # Pairwise distances for small scaffolds (prepass operates on node graphs).
    dmat = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(dmat, np.inf)
    knn_d = np.partition(dmat, kth=k - 1, axis=1)[:, :k]
    rho_knn = 1.0 / (knn_d.mean(axis=1) + 1e-12)

    bins = max(8, int(hist_bins))
    hist, edges = np.histogram(r, bins=bins)
    # Convert counts → per-bin density proxy (count / width), then lookup.
    widths = np.maximum(np.diff(edges), 1e-12)
    rho_bin = hist.astype(float) / widths
    # Assign each radius to a bin.
    bin_idx = np.clip(np.digitize(r, edges[1:-1], right=False), 0, bins - 1)
    rho_radial = rho_bin[bin_idx]
    # Prefer locally dense nodes that also sit in dense radius modes (shells).
    residual = rho_knn * (rho_radial + 1e-12)
    n_keep = max(2, int(np.ceil(n * frac)))
    order = np.argsort(-residual)
    mask = np.zeros(n, dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def _radial_band_gap_partition(
    scaffold: "Stage1Scaffold",  # noqa: F821
    *,
    min_frac: float = 0.2,
    min_abs: int = 3,
    min_gap_ratio: float = 0.25,
    hist_bins: int = 16,
    origin: str = "centroid",
    min_trough_rel: float = 0.0,
    signal_density_keep_frac: float | None = None,
) -> ClusterResult | None:
    """Radial-gap split after excluding the radius-histogram trough (#44).

    Plain centroid radial gap fails on tissue-filled nested shells because
    mid-radius bridge nodes fill the continuum between shell modes, so the
    largest *balanced* gap is too small (or the max gap is an unbalanced
    tail).  Operational remedy (flag-gated): build a 1-D radius histogram,
    take the two tallest local peaks, drop nodes in the lowest-density bin
    between them, then run the same gap rule on the remaining nodes and
    apply the cut to the full scaffold.  Requires ``Q > 0``.

    ``origin="coord_median"`` (alias ``geom_median``) uses the per-axis
    median instead of the mean centroid — dense one-sided tissue clumps
    collapse Weiszfeld, so the L1 / axis median is the operational
    non-centroid default.  ``min_trough_rel > 0`` requires relative trough
    depth ``(min(peak)-valley)/min(peak) >= min_trough_rel`` so weak /
    unimodal histograms do not fake a two-shell cut.
    """

    n = len(scaffold.nodes)
    if n < 2 * max(int(min_abs), 1):
        return None
    frac = float(min_frac)
    if not (0.0 < frac <= 0.5):
        frac = 0.2
    threshold = max(int(min_abs), int(np.ceil(n * frac)))
    if 2 * threshold > n:
        return None

    bins = max(8, int(hist_bins))
    positions = np.asarray(
        [scaffold.nodes[i].position for i in range(n)], dtype=float,
    )
    origin_key = str(origin).lower().strip()
    if origin_key in (
        "coord_median", "coordinate_median", "geom_median",
        "geometric_median", "median",
    ):
        center = _coordinate_median(positions)
    elif origin_key in ("weiszfeld",):
        center = _geometric_median(positions)
    else:
        center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    signal_keep = np.ones(n, dtype=bool)
    if signal_density_keep_frac is not None:
        signal_keep = _signal_density_residual_keep_mask(
            positions,
            radii,
            keep_frac=float(signal_density_keep_frac),
            hist_bins=bins,
        )
        if int(signal_keep.sum()) < 2 * threshold:
            return None
    r_lo = float(np.min(radii))
    r_hi = float(np.max(radii))
    if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_hi <= r_lo:
        return None
    edges = np.linspace(r_lo, r_hi, bins + 1)
    hist, _ = np.histogram(radii[signal_keep], bins=edges)
    # Local maxima including edge bins (shell modes often land at ends).
    peaks: list[int] = []
    for i in range(len(hist)):
        left = int(hist[i - 1]) if i > 0 else -1
        right = int(hist[i + 1]) if i < len(hist) - 1 else -1
        if int(hist[i]) > 0 and int(hist[i]) >= left and int(hist[i]) >= right:
            peaks.append(i)
    if len(peaks) < 2:
        # Fallback: two densest bins at least 2 bins apart.
        order = list(np.argsort(-hist))
        primary = int(order[0])
        secondary = next(
            (int(j) for j in order[1:] if abs(int(j) - primary) >= 2 and int(hist[j]) > 0),
            None,
        )
        if secondary is None:
            return None
        peaks = [primary, secondary]
    peaks = sorted(peaks, key=lambda i: int(hist[i]), reverse=True)[:2]
    lo, hi = sorted(peaks)
    if hi - lo < 2:
        return None
    valley = lo + int(np.argmin(hist[lo: hi + 1]))
    # Optional bimodality / trough-depth gate (A2-T13).
    trough_rel = float(min_trough_rel)
    if trough_rel > 0.0:
        peak_h = float(min(int(hist[lo]), int(hist[hi])))
        if peak_h <= 0.0:
            return None
        depth = (peak_h - float(int(hist[valley]))) / peak_h
        if depth < trough_rel:
            return None

    def _grow(peak: int, step: int) -> list[int]:
        """Contiguous hist>0 support from ``peak`` toward the valley."""
        out = [peak]
        j = peak + step
        while 0 <= j < bins and j != valley and int(hist[j]) > 0:
            out.append(j)
            j += step
        return out

    left_bins = set(_grow(lo, +1))
    right_bins = set(_grow(hi, -1))
    peak_bins = left_bins | right_bins
    if valley not in peak_bins and not (set(range(bins)) - peak_bins):
        peak_bins = peak_bins  # noqa: keep peak support

    def _in_bins(r: float, bset: set[int]) -> bool:
        for b in bset:
            lo_e, hi_e = float(edges[b]), float(edges[b + 1])
            if b == bins - 1:
                if lo_e <= r <= hi_e:
                    return True
            elif lo_e <= r < hi_e:
                return True
        return False

    peak_mask = np.array([_in_bins(float(radii[i]), peak_bins) for i in range(n)])
    mask = peak_mask & signal_keep  # peak-mode support ∩ signal-density keep
    idx = np.where(mask)[0]
    if len(idx) < 2 * threshold:
        return None

    peak_idx = idx
    r = radii[peak_idx]
    thr_masked = max(int(min_abs), int(np.ceil(len(peak_idx) * frac)))
    order = np.argsort(r)
    gaps = np.diff(r[order])
    best: tuple[float, int] | None = None
    for i, gap in enumerate(gaps):
        left = i + 1
        right = len(peak_idx) - left
        if left < thr_masked or right < thr_masked:
            continue
        g = float(gap)
        if best is None or g > best[0]:
            best = (g, i)
    if best is None:
        return None

    med_r = float(np.median(radii[idx])) + 1e-12
    if best[0] / med_r < float(min_gap_ratio):
        return None

    thr_cut = 0.5 * (
        float(r[order[best[1]]]) + float(r[order[best[1] + 1]])
    )
    base_labels = (radii > thr_cut).astype(int)
    # Mid-band (non-peak-support) nodes: try both shell assignments for Q > 0.
    midband = np.where(~mask)[0]
    best_pq = -np.inf
    best_labels: np.ndarray | None = None
    candidates: list[np.ndarray] = [base_labels]
    if len(midband) > 0:
        for side in (0, 1):
            lab = base_labels.copy()
            lab[midband] = side
            candidates.append(lab)
    graph_lifted = scaffold.links.neighbour_graph(n)
    W = compute_edge_weights(scaffold)
    for lab in candidates:
        clusters = [
            set(np.where(lab == cid)[0].tolist()) for cid in (0, 1)
        ]
        if min(len(c) for c in clusters) < threshold:
            continue
        pq = float(partition_q_score(clusters, n, W, graph_lifted))
        if pq > best_pq:
            best_pq = pq
            best_labels = lab
    if best_labels is None or best_pq <= 0.0:
        return None

    hits = np.array([node.hit_count for node in scaffold.nodes], dtype=float)
    clusters_final: list[set[int]] = [
        set(np.where(best_labels == cid)[0].tolist()) for cid in (0, 1)
    ]
    exemplars = [int(max(c, key=lambda g: hits[g])) for c in clusters_final]
    return ClusterResult(
        labels=best_labels,
        exemplar_indices=np.array(exemplars, dtype=int),
        n_clusters=2,
        partition_q_score=float(best_pq),
    )


def _research_finer_split(
    data: np.ndarray,
    dim: int,
    config: RecursionConfig,
    parent_tau: float,
):
    """Capped multi-step scale re-search strictly finer than ``parent_tau`` (#44).

    Walks ``tau_max`` downward geometrically (``finer_tau_cap_ratio`` per step,
    at most ``max_finer_scale_steps`` times) until a multi-cluster proposal with
    ``Q > 0`` appears, else returns ``None``.  Does not recurse into children;
    the caller applies accept gates and child descent.  **Stop guarantee:**
    step budget + ``tau_min`` bound the walk; a failed / gated-out proposal
    leaves the region terminal.

    When ``prefer_disconnected_prepass`` is on, each step first tries the
    major-lifted-component short-circuit (#44c) before the general AP/DM path.
    When ``prefer_hollow_edge_prepass`` is on, hollow-edge pruning + major CCs
    are tried next (support-topology / empty-region evidence).  When
    ``prefer_radial_gap_prepass`` is on, a centroid-radial gap split is
    tried next (concentric shells that stay lifted-connected).  When
    ``prefer_radial_band_prepass`` is on, a histogram-trough-masked radial
    gap is tried (mid-band / tissue continuum exclusion).  When
    ``prefer_noncentroid_radial_band_prepass`` is on, the same band rule runs
    with a coordinate-median origin and a trough-depth bimodality gate.
    When ``prefer_signal_density_band_prepass`` is on, knn-density residual
    masking precedes the band gap.  When ``prefer_pca_axis_gap_prepass`` is
    on, a leading-PC 1-D gap is tried (offset / non-radial geometry cue).
    When ``prefer_tube_major_radius_prepass`` is on, a Hopf-linked tube
    residual assignment is tried (interlocking rings).  When
    ``prefer_spectral_gap_prepass`` is on, a Fiedler bipartition of the
    lifted / kNN graph is tried.
    """

    ratio = float(config.finer_tau_cap_ratio)
    if not (0.0 < ratio < 1.0):
        ratio = float(config.scale_search.grid_ratio)
    tau_min = float(config.scale_search.tau_min)
    tau_cap = float(parent_tau) * ratio
    max_steps = max(1, int(config.max_finer_scale_steps))

    for _step in range(max_steps):
        if not (tau_min < tau_cap < float(parent_tau)):
            return None

        scale_search_config = replace(config.scale_search, tau_max=tau_cap)
        if config.require_persistent_split:
            scale_search_config = replace(
                scale_search_config,
                selector="persistence",
                record_partitions=True,
            )

        result = run_scale_search(data, dim, scale_search_config)
        scaffold = result.scaffold_at_star
        if scaffold is None or len(scaffold.nodes) < 2:
            tau_cap *= ratio
            continue

        if config.require_persistent_split:
            persistence = result.persistence_result
            if persistence is None or persistence.tau_star_index is None:
                tau_cap *= ratio
                continue

        # #44c: cheap disconnected-lifted prepass before general clustering.
        if config.prefer_disconnected_prepass:
            pre = _major_lifted_component_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
            )
            if (
                pre is not None
                and pre.n_clusters > 1
                and pre.partition_q_score > 0.0
            ):
                return result, scaffold, pre

        # #44: hollow-edge (empty-region) prepass — data-side bridge cut.
        if config.prefer_hollow_edge_prepass:
            hollow = _hollow_edge_partition(
                scaffold,
                data,
                min_frac=float(config.finer_prepass_min_frac),
                hollow_config=_resolve_hollow_edge_config(config),
            )
            if _hollow_prepass_accepted(config, result, hollow):
                return result, scaffold, hollow

        # #44: radial-gap prepass for concentric shells still lifted-connected.
        if config.prefer_radial_gap_prepass:
            radial = _radial_gap_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
                min_gap_ratio=float(config.finer_radial_min_gap_ratio),
            )
            if (
                radial is not None
                and radial.n_clusters > 1
                and radial.partition_q_score > 0.0
            ):
                return result, scaffold, radial

        # #44: trough-masked radial-band prepass (mid-band continuum exclusion).
        if config.prefer_radial_band_prepass:
            band = _radial_band_gap_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
                min_gap_ratio=float(config.finer_radial_min_gap_ratio),
                hist_bins=int(config.finer_radial_hist_bins),
                origin="centroid",
                min_trough_rel=float(config.finer_radial_min_trough_rel),
            )
            if (
                band is not None
                and band.n_clusters > 1
                and band.partition_q_score > 0.0
            ):
                return result, scaffold, band

        # #44 / A2-T13: non-centroid (coord-median) + trough-depth band prepass.
        if config.prefer_noncentroid_radial_band_prepass:
            trough = float(config.finer_radial_min_trough_rel)
            if trough <= 0.0:
                trough = 0.35  # operational default on the noncentroid path
            band_nc = _radial_band_gap_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
                min_gap_ratio=float(config.finer_radial_min_gap_ratio),
                hist_bins=int(config.finer_radial_hist_bins),
                origin="coord_median",
                min_trough_rel=trough,
            )
            if (
                band_nc is not None
                and band_nc.n_clusters > 1
                and band_nc.partition_q_score > 0.0
            ):
                return result, scaffold, band_nc

        # #44 / A2-T14: signal-density residual mask + band prepass.
        if config.prefer_signal_density_band_prepass:
            band_sd = _radial_band_gap_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
                min_gap_ratio=float(config.finer_radial_min_gap_ratio),
                hist_bins=int(config.finer_radial_hist_bins),
                origin="centroid",
                min_trough_rel=float(config.finer_radial_min_trough_rel),
                signal_density_keep_frac=float(
                    config.finer_signal_density_keep_frac
                ),
            )
            if (
                band_sd is not None
                and band_sd.n_clusters > 1
                and band_sd.partition_q_score > 0.0
            ):
                return result, scaffold, band_sd

        # #44 / A2-T21: non-radial PCA-axis gap (offset rings / linked_tori cue).
        if config.prefer_pca_axis_gap_prepass:
            pca_gap = _pca_axis_gap_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
                min_gap_ratio=float(config.finer_radial_min_gap_ratio),
            )
            if (
                pca_gap is not None
                and pca_gap.n_clusters > 1
                and pca_gap.partition_q_score > 0.0
            ):
                return result, scaffold, pca_gap

        # #44 / A2-T24: Hopf-linked tube major-radius residual (interlock cue).
        if config.prefer_tube_major_radius_prepass:
            tube = _tube_major_radius_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
                min_residual_ratio=float(config.finer_tube_min_residual_ratio),
            )
            if (
                tube is not None
                and tube.n_clusters > 1
                and tube.partition_q_score > 0.0
            ):
                return result, scaffold, tube

        # #44 / A2-T25: spectral Fiedler bipartition (linking / graph cue).
        if config.prefer_spectral_gap_prepass:
            spectral = _spectral_gap_partition(
                scaffold,
                min_frac=float(config.finer_prepass_min_frac),
                knn=int(config.finer_spectral_knn),
            )
            if (
                spectral is not None
                and spectral.n_clusters > 1
                and spectral.partition_q_score > 0.0
            ):
                return result, scaffold, spectral

        cluster_result = _cluster_scaffold(scaffold, config)
        if (
            cluster_result.n_clusters > 1
            and cluster_result.partition_q_score > 0.0
        ):
            return result, scaffold, cluster_result

        tau_cap *= ratio

    return None


def _dm_accepts_split(
    scaffold: "Stage1Scaffold",  # noqa: F821
    cluster_result: Any,
    config: RecursionConfig,
) -> bool:
    """True if DM gate is off, or the K-way partition clears ``log(tau_bf)``."""

    if not config.require_dm_split:
        return True
    clusters = _clusters_from_labels(cluster_result.labels)
    _log_bf, accepted = dm_partition_verdict(
        scaffold, clusters, config.dm_cluster,
    )
    return bool(accepted)


def _descend_into_clusters(
    *,
    data_arr: np.ndarray,
    dim: int,
    config: RecursionConfig,
    tree: RecursionTree,
    node: RecursionNode,
    region_id: int,
    _level: int,
    orig_rows: np.ndarray,
    scaffold: "Stage1Scaffold",  # noqa: F821
    cluster_result: Any,
) -> RecursionTree:
    """Demote, map samples, create children, and recurse."""

    demote_lifted_by_cluster(
        scaffold, cluster_result.labels,
        beta=float(getattr(scaffold, "prune_beta", 0.5)),
    )

    sample_map = assign_samples_to_clusters(
        data_arr, scaffold, cluster_result.labels,
    )

    children_created: list[int] = []
    for label, child_indices in sorted(sample_map.items()):
        if len(child_indices) < config.min_samples:
            child_id = len(tree.nodes)
            global_child = orig_rows[np.asarray(child_indices, dtype=int)]
            tree.nodes.append(RecursionNode(
                region_id=child_id,
                level=_level + 1,
                parent_id=region_id,
                tau_star=None,
                n_samples=len(child_indices),
                dim=dim,
                n_clusters=0,
                is_leaf=True,
                sample_indices=global_child.copy(),
            ))
            children_created.append(child_id)
            continue

        cluster_node_ids = np.where(cluster_result.labels == label)[0]
        # Operationally d_final == working dim (SI S1.4.1 refresh semantics),
        # so d_hat reduces to the region working dim unless refreshed.
        d_finals = [scaffold.nodes[int(i)].d_final for i in cluster_node_ids]
        d_hat = int(np.median(d_finals)) if d_finals else dim

        t2 = apply_t2_transfer(
            data_arr, child_indices, dim, d_hat,
            r_min=config.r_min,
            explained_energy=config.explained_energy,
        )

        child_config = RecursionConfig(
            scale_search=config.scale_search,
            min_samples=config.min_samples,
            max_depth=config.max_depth,
            r_min=config.r_min,
            explained_energy=config.explained_energy,
            require_persistent_split=config.require_persistent_split,
            require_dm_split=config.require_dm_split,
            dm_cluster=config.dm_cluster,
            allow_finer_research=config.allow_finer_research,
            finer_tau_cap_ratio=config.finer_tau_cap_ratio,
            max_finer_scale_steps=config.max_finer_scale_steps,
            prefer_disconnected_prepass=config.prefer_disconnected_prepass,
            finer_prepass_min_frac=config.finer_prepass_min_frac,
            prefer_radial_gap_prepass=config.prefer_radial_gap_prepass,
            finer_radial_min_gap_ratio=config.finer_radial_min_gap_ratio,
            prefer_radial_band_prepass=config.prefer_radial_band_prepass,
            finer_radial_hist_bins=config.finer_radial_hist_bins,
            prefer_noncentroid_radial_band_prepass=(
                config.prefer_noncentroid_radial_band_prepass
            ),
            finer_radial_min_trough_rel=config.finer_radial_min_trough_rel,
            prefer_signal_density_band_prepass=(
                config.prefer_signal_density_band_prepass
            ),
            finer_signal_density_keep_frac=config.finer_signal_density_keep_frac,
            prefer_pca_axis_gap_prepass=config.prefer_pca_axis_gap_prepass,
            prefer_tube_major_radius_prepass=(
                config.prefer_tube_major_radius_prepass
            ),
            finer_tube_min_residual_ratio=config.finer_tube_min_residual_ratio,
            prefer_spectral_gap_prepass=config.prefer_spectral_gap_prepass,
            finer_spectral_knn=config.finer_spectral_knn,
            prefer_hollow_edge_prepass=config.prefer_hollow_edge_prepass,
            hollow_mid_radius_frac=config.hollow_mid_radius_frac,
            hollow_h0=config.hollow_h0,
            hollow_min_end_count=config.hollow_min_end_count,
            hollow_gabriel_fallback=config.hollow_gabriel_fallback,
            hollow_require_gabriel_and_h=config.hollow_require_gabriel_and_h,
            hollow_require_persistent_agree=config.hollow_require_persistent_agree,
            hollow_use_a4_primary=config.hollow_use_a4_primary,
            hollow_mst_critical_only=config.hollow_mst_critical_only,
            hollow_bridge_critical_only=config.hollow_bridge_critical_only,
            hollow_soft_capacity_only=config.hollow_soft_capacity_only,
            hollow_soft_capacity_frac=config.hollow_soft_capacity_frac,
            hollow_soft_capacity_method=config.hollow_soft_capacity_method,
            seed=config.seed + region_id + label,
        )

        child_id = len(tree.nodes)
        children_created.append(child_id)

        child_orig_globals = orig_rows[np.asarray(child_indices, dtype=int)]

        run_recursive_discovery(
            t2.child_data,
            t2.child_dim,
            child_config,
            _level=_level + 1,
            _parent_id=region_id,
            _tree=tree,
            _sample_indices=child_orig_globals,
        )

    node.children = children_created
    if children_created:
        node.is_leaf = False

    return tree


def run_recursive_discovery(
    data: np.ndarray,
    dim: int,
    config: RecursionConfig | None = None,
    *,
    _level: int = 0,
    _parent_id: int | None = None,
    _tree: RecursionTree | None = None,
    _sample_indices: np.ndarray | None = None,
) -> RecursionTree:
    """Recursively discover scale structure via scale search + clustering + T2.

    At each level: run scale search to find tau_star, cluster the
    converged scaffold via Q-score seed merging, and for each cluster
    with enough samples apply the T2 PCA transfer and recurse into the
    child.  Recursion is gated by Q(P; v) > 0 on the proposed partition.
    """

    config = config if config is not None else RecursionConfig()
    tree = _tree if _tree is not None else RecursionTree()
    data_arr = np.asarray(data, dtype=float)
    n_samples = data_arr.shape[0]

    if _sample_indices is None:
        orig_rows = np.arange(n_samples, dtype=int)
    else:
        orig_rows = np.asarray(_sample_indices, dtype=int, copy=True)
    if orig_rows.shape[0] != n_samples:
        raise ValueError(
            "_sample_indices length must match data row count "
            f"({orig_rows.shape[0]} vs {n_samples})",
        )

    region_id = len(tree.nodes)
    node = RecursionNode(
        region_id=region_id,
        level=_level,
        parent_id=_parent_id,
        tau_star=None,
        n_samples=n_samples,
        dim=dim,
        n_clusters=0,
        is_leaf=True,
        sample_indices=orig_rows.copy(),
    )
    tree.nodes.append(node)

    if n_samples < config.min_samples:
        return tree

    if _level >= config.max_depth:
        return tree

    scale_search_config = config.scale_search
    if config.require_persistent_split:
        # The persistence accept-gate (SI S2.6.2) needs the per-grid-point
        # partitions and the persistence-selected characteristic scale.
        scale_search_config = replace(
            scale_search_config,
            selector="persistence",
            record_partitions=True,
        )

    result = run_scale_search(data_arr, dim, scale_search_config)
    node.tau_star = result.tau_star
    scaffold = result.scaffold_at_star

    if scaffold is None or len(scaffold.nodes) < 2:
        return tree

    need_finer = False
    cluster_result = None

    # #44 hollow-edge at the region's own tau*: support disconnection is
    # scale-free, so try before persistence / AP and before finer descent.
    if config.prefer_hollow_edge_prepass:
        hollow = _hollow_edge_partition(
            scaffold,
            data_arr,
            min_frac=float(config.finer_prepass_min_frac),
            hollow_config=_resolve_hollow_edge_config(config),
        )
        if _hollow_prepass_accepted(config, result, hollow):
            cluster_result = hollow
            node.n_clusters = hollow.n_clusters

    if cluster_result is None and config.require_persistent_split:
        # Accept a split only if a multi-cluster partition persists across
        # adjacent scales; a region with no persistent split is a single
        # intrinsic feature (terminal leaf), regardless of any transient
        # single-scale fragmentation (SI S2.6.2, OPEN_ISSUES #27).
        # OPEN_ISSUES #44: when ``allow_finer_research`` is on, defer the
        # terminal decision so a capped finer re-search can still run.
        persistence = result.persistence_result
        if persistence is None or persistence.tau_star_index is None:
            need_finer = True
        else:
            cluster_result = _cluster_scaffold(scaffold, config)
            node.n_clusters = cluster_result.n_clusters
            if (
                cluster_result.n_clusters <= 1
                or cluster_result.partition_q_score <= 0.0
            ):
                need_finer = True
    elif cluster_result is None:
        cluster_result = _cluster_scaffold(scaffold, config)
        node.n_clusters = cluster_result.n_clusters
        if (
            cluster_result.n_clusters <= 1
            or cluster_result.partition_q_score <= 0.0
        ):
            need_finer = True

    if need_finer:
        # OPEN_ISSUES #44: optionally re-search strictly finer than this
        # region's characteristic tau* before declaring the region terminal.
        if not config.allow_finer_research or node.tau_star is None:
            return tree
        researched = _research_finer_split(
            data_arr, dim, config, float(node.tau_star),
        )
        if researched is None:
            return tree
        # Keep node.tau_star as the coarse characteristic scale (hierarchy
        # root); use the finer scaffold only for the accepted split.
        _finer_result, scaffold, cluster_result = researched
        node.n_clusters = cluster_result.n_clusters

    assert cluster_result is not None

    if not _dm_accepts_split(scaffold, cluster_result, config):
        # Region-level accept gate: retain the multi-cluster split only if it
        # clears the Bayes-factor margin against the one-feature null. A region
        # whose split is not evidence-bearing is a single intrinsic feature
        # (terminal leaf), the non-degenerate likelihood-ratio null that the
        # graph-local Q cannot supply (SI S2.6.1 / S3.4, OPEN_ISSUES #27).
        node.n_clusters = 1
        return tree

    return _descend_into_clusters(
        data_arr=data_arr,
        dim=dim,
        config=config,
        tree=tree,
        node=node,
        region_id=region_id,
        _level=_level,
        orig_rows=orig_rows,
        scaffold=scaffold,
        cluster_result=cluster_result,
    )
