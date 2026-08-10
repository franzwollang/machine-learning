"""Stage-1 seed=77 denser256 lifetime vs hollow+lifetime hunt (#41 / A4-T47).

A4-T44: denser seed2 / max_nodes=256 / Stage-1 seed=77 is the best
both-tori partial scaffold ``(1,2,0)`` / ``(1,2,0)`` (b2 missing).
A4-T43/T45 probed lifetime alone and hollow+lifetime separately; this
harness **jointly** contrasts signal-lifetime vs hollow-pruned-lifetime on
that seed77 denser256 scaffold toward full ``(1,2,1)``.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.edge_evidence import HollowEdgeConfig, prune_hollow_edges
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.metrics.persistent_homology import (
    DEFAULT_LIFETIME_FRAC,
    FILTRATION_MULTIPLIER,
    format_lifetime_frac_sweep_table,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
    sweep_lifetime_frac_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 77
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
HOLLOW_CFG = HollowEdgeConfig(
    mid_radius_frac=0.5, h0=0.7, min_end_count=0.5, gabriel_fallback=False,
)


def _undirected_edges_from_adj(adj: dict) -> list[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, nbrs in adj.items():
        for j in nbrs:
            a, b = (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            if a != b:
                edges.add((a, b))
    return sorted(edges)


def _hollow_pruned_node_mask(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    *,
    config: HollowEdgeConfig | None = None,
) -> np.ndarray:
    """True for nodes that retain ≥1 non-hollow neighbour edge."""
    n = int(positions.shape[0])
    keep = np.zeros(n, dtype=bool)
    if not edges:
        return keep
    surviving = prune_hollow_edges(positions, edges, data, config=config)
    for i, j in surviving:
        keep[int(i)] = True
        keep[int(j)] = True
    return keep


def _summarize_lifetime_rows(
    rows,
) -> tuple[
    bool,
    bool,
    bool,
    int,
    int,
    tuple[tuple[int, float, tuple[int, ...]], ...],
    tuple[tuple[int, float, tuple[int, ...]], ...],
]:
    recovering: list[tuple[int, float, tuple[int, ...]]] = []
    b2_cells: list[tuple[int, float, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_full = False
    any_b1 = False
    for row in rows:
        betti = tuple(int(x) for x in row.betti)
        rid = int(row.region_id) if row.region_id is not None else -1
        max_b1 = max(max_b1, int(betti[1]) if len(betti) > 1 else 0)
        max_b2 = max(max_b2, int(betti[2]) if len(betti) > 2 else 0)
        if betti == EXPECTED_TORI:
            any_full = True
            recovering.append((rid, float(row.lifetime_frac), betti))
        if len(betti) > 1 and int(betti[1]) >= 2:
            any_b1 = True
        if len(betti) > 2 and int(betti[2]) >= 1:
            b2_cells.append((rid, float(row.lifetime_frac), betti))
    return (
        any_full,
        any_b1,
        bool(b2_cells),
        max_b1,
        max_b2,
        tuple(recovering),
        tuple(b2_cells),
    )


@dataclass(frozen=True)
class Seed77LifetimeHollowBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    n_after_hollow: int
    hollow_fallback: bool
    sigma_star: float
    signal_fixed_betti: dict[int, tuple[int, ...]]
    hollow_fixed_betti: dict[int, tuple[int, ...]]
    both_tori_partial_fixed: bool
    fracs: tuple[float, ...]
    signal_table: str
    hollow_table: str
    signal_any_full: bool
    signal_any_b2: bool
    signal_any_b1_ge_2: bool
    signal_max_b1: int
    signal_max_b2: int
    signal_recovering: tuple[tuple[int, float, tuple[int, ...]], ...]
    signal_b2_cells: tuple[tuple[int, float, tuple[int, ...]], ...]
    hollow_any_full: bool
    hollow_any_b2: bool
    hollow_any_b1_ge_2: bool
    hollow_max_b1: int
    hollow_max_b2: int
    hollow_recovering: tuple[tuple[int, float, tuple[int, ...]], ...]
    hollow_b2_cells: tuple[tuple[int, float, tuple[int, ...]], ...]
    any_full_match: bool
    any_b2: bool
    max_b2: int


@pytest.fixture(scope="module")
def linked_tori_seed77_lifetime_hollow_bundle() -> Seed77LifetimeHollowBundle:
    """Fit seed77 denser256; contrast signal vs hollow lifetime sweeps."""
    dataset = make_linked_tori(
        n_per_torus=N_PER_TORUS,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=DATASET_SEED,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=MAX_NODES,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=STAGE1_SEED,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    scaffold = result.scaffold_at_star
    pos = scaffold.node_positions()
    sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [0, 1])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    adj = scaffold.links.neighbour_graph(pos.shape[0])
    edges = _undirected_edges_from_adj(adj)
    hollow_keep = _hollow_pruned_node_mask(
        pos, edges, dataset.points, config=HOLLOW_CFG,
    )
    hollow_signal = signal_mask & hollow_keep
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not (
        np.any(node_labels[hollow_signal] == 0)
        and np.any(node_labels[hollow_signal] == 1)
    ):
        hollow_pos, hollow_labs = signal_pos, signal_labs
        hollow_fallback = True
    else:
        hollow_pos = pos[hollow_signal]
        hollow_labs = node_labels[hollow_signal]
        hollow_fallback = False

    signal_fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_seed77_signal_fixed",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=EXPECTED_TORI,
    )
    hollow_fixed = run_per_region_ph(
        hollow_pos,
        hollow_labs,
        sigma,
        scenario="linked_tori_seed77_hollow_fixed",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=EXPECTED_TORI,
    )
    signal_fixed_betti = {
        int(r.region_id): tuple(int(x) for x in r.betti)
        for r in signal_fixed.reports
    }
    hollow_fixed_betti = {
        int(r.region_id): tuple(int(x) for x in r.betti)
        for r in hollow_fixed.reports
    }
    both_partial = (
        signal_fixed_betti.get(0) == (1, 2, 0)
        and signal_fixed_betti.get(1) == (1, 2, 0)
    )

    signal_rows = sweep_lifetime_frac_per_region(
        signal_pos,
        signal_labs,
        sigma,
        fracs=FRAC_GRID,
        include_labels=[0, 1],
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        target_betti=EXPECTED_TORI,
    )
    hollow_rows = sweep_lifetime_frac_per_region(
        hollow_pos,
        hollow_labs,
        sigma,
        fracs=FRAC_GRID,
        include_labels=[0, 1],
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        target_betti=EXPECTED_TORI,
    )
    (
        sig_full,
        sig_b1,
        sig_b2,
        sig_max_b1,
        sig_max_b2,
        sig_rec,
        sig_b2_cells,
    ) = _summarize_lifetime_rows(signal_rows)
    (
        hol_full,
        hol_b1,
        hol_b2,
        hol_max_b1,
        hol_max_b2,
        hol_rec,
        hol_b2_cells,
    ) = _summarize_lifetime_rows(hollow_rows)

    any_full = sig_full or hol_full
    any_b2 = sig_b2 or hol_b2
    max_b2 = max(sig_max_b2, hol_max_b2)

    return Seed77LifetimeHollowBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        n_after_hollow=n_after,
        hollow_fallback=bool(hollow_fallback),
        sigma_star=float(sigma),
        signal_fixed_betti=signal_fixed_betti,
        hollow_fixed_betti=hollow_fixed_betti,
        both_tori_partial_fixed=both_partial,
        fracs=FRAC_GRID,
        signal_table=format_lifetime_frac_sweep_table(signal_rows),
        hollow_table=format_lifetime_frac_sweep_table(hollow_rows),
        signal_any_full=sig_full,
        signal_any_b2=sig_b2,
        signal_any_b1_ge_2=sig_b1,
        signal_max_b1=sig_max_b1,
        signal_max_b2=sig_max_b2,
        signal_recovering=sig_rec,
        signal_b2_cells=sig_b2_cells,
        hollow_any_full=hol_full,
        hollow_any_b2=hol_b2,
        hollow_any_b1_ge_2=hol_b1,
        hollow_max_b1=hol_max_b1,
        hollow_max_b2=hol_max_b2,
        hollow_recovering=hol_rec,
        hollow_b2_cells=hol_b2_cells,
        any_full_match=any_full,
        any_b2=any_b2,
        max_b2=max_b2,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_lifetime_hollow_harness_lands(
    linked_tori_seed77_lifetime_hollow_bundle,
) -> None:
    """Seed77 denser lifetime/hollow contrast lands; SI defaults untouched."""
    bundle = linked_tori_seed77_lifetime_hollow_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.n_after_hollow >= 0
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert len(bundle.signal_table) > 0
    assert len(bundle.hollow_table) > 0
    # T44 baseline: Stage-1 seed77 both-tori partial interlocking.
    assert bundle.both_tori_partial_fixed


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_lifetime_hollow_documents_gap(
    linked_tori_seed77_lifetime_hollow_bundle,
) -> None:
    """Document seed77 lifetime vs hollow+lifetime vs b2; never flip awaiting.

    Soft: any full ``(1,2,1)`` or any ``b2≥1`` on either arm is proposal-path
    evidence. Otherwise keep explicit ``max_b2 == 0``. Joint hunt ≠ SI recovery.
    """
    bundle = linked_tori_seed77_lifetime_hollow_bundle
    if bundle.any_full_match or bundle.any_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert (
                len(bundle.signal_recovering) >= 1
                or len(bundle.hollow_recovering) >= 1
            )
            assert bundle.max_b2 >= 1
        if bundle.any_b2:
            assert (
                len(bundle.signal_b2_cells) >= 1
                or len(bundle.hollow_b2_cells) >= 1
            )
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_b2 is False
        assert bundle.max_b2 == 0
        assert bundle.signal_any_full is False
        assert bundle.hollow_any_full is False
        assert bundle.signal_any_b2 is False
        assert bundle.hollow_any_b2 is False
        assert bundle.signal_recovering == ()
        assert bundle.hollow_recovering == ()
        assert bundle.signal_b2_cells == ()
        assert bundle.hollow_b2_cells == ()
        # Fixed seed77 baseline still documents partial b1.
        assert bundle.both_tori_partial_fixed
