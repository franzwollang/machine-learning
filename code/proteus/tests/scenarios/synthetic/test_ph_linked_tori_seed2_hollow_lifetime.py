"""Hollow-signal + lifetime on seed=2 denser fitted linked-tori (#41 / A4-T45).

A4-T42/T43: denser seed2 / max_nodes=256 / Stage-1 seed=77 yields both tori
``(1,2,0)`` under fixed_threshold; lifetime_frac alone never unlocks ``b2``.
This harness applies the A4 primary hollow prune (mid=0.5, h0=0.7, no Gabriel)
to signal nodes, then sweeps ``lifetime_frac`` on the hollow-pruned cloud
toward ``b2`` / full ``(1,2,1)``.

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


@dataclass(frozen=True)
class Seed2HollowLifetimeBundle:
    dataset_seed: int
    max_nodes: int
    n_signal: int
    n_after_hollow: int
    hollow_fallback: bool
    sigma_star: float
    signal_fixed_betti: dict[int, tuple[int, ...]]
    hollow_fixed_betti: dict[int, tuple[int, ...]]
    fracs: tuple[float, ...]
    table: str
    any_full_match: bool
    any_b2: bool
    any_b1_ge_2: bool
    max_b1: int
    max_b2: int
    recovering_cells: tuple[tuple[int, float, tuple[int, ...]], ...]
    b2_cells: tuple[tuple[int, float, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_seed2_hollow_lifetime_bundle() -> Seed2HollowLifetimeBundle:
    """Fit denser seed=2; hollow-prune signal; sweep lifetime_frac."""
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
        scenario="linked_tori_seed2_signal_fixed",
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
        scenario="linked_tori_seed2_hollow_fixed",
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

    rows = sweep_lifetime_frac_per_region(
        hollow_pos,
        hollow_labs,
        sigma,
        fracs=FRAC_GRID,
        include_labels=[0, 1],
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        target_betti=EXPECTED_TORI,
    )
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

    return Seed2HollowLifetimeBundle(
        dataset_seed=DATASET_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        n_after_hollow=n_after,
        hollow_fallback=bool(hollow_fallback),
        sigma_star=float(sigma),
        signal_fixed_betti=signal_fixed_betti,
        hollow_fixed_betti=hollow_fixed_betti,
        fracs=FRAC_GRID,
        table=format_lifetime_frac_sweep_table(rows),
        any_full_match=any_full,
        any_b2=bool(b2_cells),
        any_b1_ge_2=any_b1,
        max_b1=max_b1,
        max_b2=max_b2,
        recovering_cells=tuple(recovering),
        b2_cells=tuple(b2_cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed2_hollow_lifetime_harness_lands(
    linked_tori_seed2_hollow_lifetime_bundle,
) -> None:
    """Hollow+lifetime denser seed2 ladder lands; SI defaults untouched."""
    bundle = linked_tori_seed2_hollow_lifetime_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.n_after_hollow >= 0
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert len(bundle.table) > 0
    # Signal fixed baseline still shows T42 partial interlocking.
    assert any(
        len(b) > 1 and int(b[1]) >= 2 for b in bundle.signal_fixed_betti.values()
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed2_hollow_lifetime_documents_gap(
    linked_tori_seed2_hollow_lifetime_bundle,
) -> None:
    """Document hollow+lifetime vs b2 on seed2 denser; never flip awaiting.

    Soft: any full ``(1,2,1)`` or any ``b2≥1`` cell is proposal-path evidence.
    Otherwise keep explicit ``max_b2 == 0``. Hollow+lifetime ≠ SI recovery.
    """
    bundle = linked_tori_seed2_hollow_lifetime_bundle
    if bundle.any_full_match or bundle.any_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert len(bundle.recovering_cells) >= 1
            assert bundle.max_b2 >= 1
        if bundle.any_b2:
            assert len(bundle.b2_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_b2 is False
        assert bundle.max_b2 == 0
        assert bundle.recovering_cells == ()
        assert bundle.b2_cells == ()
        # Signal baseline still documents partial b1 (T42).
        assert any(
            len(b) > 1 and int(b[1]) >= 2
            for b in bundle.signal_fixed_betti.values()
        )
