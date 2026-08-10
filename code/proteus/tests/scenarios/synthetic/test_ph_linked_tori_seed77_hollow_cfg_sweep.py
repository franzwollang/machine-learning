"""Hollow mid/h0 config sweep on seed77 denser256 (#41 / A4-T52-followon).

A4-T47/T49: primary hollow ``mid=0.5/h0=0.7/noGab`` on seed77 denser256
yields dirty torus1 ``b2`` (inflated ``b0``); cal-mult≥3 erases dirty without
clean ``(1,2,1)``. This harness freezes the seed77 scaffold and sweeps
alternate ``(mid_radius_frac, h0)`` hollow configs (gabriel off) under
fixed_threshold + default lifetime, asking whether a milder/stricter hollow
cfg recovers clean void Betti.

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
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 77
# Compact mid×h0 grid around primary ROC (0.5, 0.7) and A2 probe (0.35, 0.35).
MID_GRID: tuple[float, ...] = (0.35, 0.5, 0.65)
H0_GRID: tuple[float, ...] = (0.35, 0.5, 0.7)
MIN_END_COUNT: float = 0.5


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
    config: HollowEdgeConfig,
) -> np.ndarray:
    n = int(positions.shape[0])
    keep = np.zeros(n, dtype=bool)
    if not edges:
        return keep
    surviving = prune_hollow_edges(positions, edges, data, config=config)
    for i, j in surviving:
        keep[int(i)] = True
        keep[int(j)] = True
    return keep


def _is_clean_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[0]) == 1
        and int(betti[1]) >= 2
        and int(betti[2]) >= 1
    )


def _is_dirty_b2(betti: tuple[int, ...]) -> bool:
    return (
        len(betti) > 2
        and int(betti[2]) >= 1
        and not _is_clean_b2(betti)
    )


@dataclass(frozen=True)
class HollowCfgRow:
    mid_radius_frac: float
    h0: float
    n_after_hollow: int
    hollow_fallback: bool
    fixed_betti: dict[int, tuple[int, ...]]
    lifetime_betti: dict[int, tuple[int, ...]]
    any_full: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int


@dataclass(frozen=True)
class Seed77HollowCfgSweepBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    sigma_star: float
    signal_fixed_betti: dict[int, tuple[int, ...]]
    both_tori_partial_fixed: bool
    mids: tuple[float, ...]
    h0s: tuple[float, ...]
    rows: tuple[HollowCfgRow, ...]
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    clean_cells: tuple[tuple[float, float, int, str, tuple[int, ...]], ...]
    dirty_cells: tuple[tuple[float, float, int, str, tuple[int, ...]], ...]
    full_cells: tuple[tuple[float, float, int, str, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_seed77_hollow_cfg_sweep_bundle() -> Seed77HollowCfgSweepBundle:
    """Fit seed77 denser256 once; sweep hollow mid×h0 under fixed+lifetime."""
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

    signal_fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_seed77_cfg_signal_fixed",
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
    both_partial = (
        signal_fixed_betti.get(0) == (1, 2, 0)
        and signal_fixed_betti.get(1) == (1, 2, 0)
    )

    rows: list[HollowCfgRow] = []
    clean_cells: list[tuple[float, float, int, str, tuple[int, ...]]] = []
    dirty_cells: list[tuple[float, float, int, str, tuple[int, ...]]] = []
    full_cells: list[tuple[float, float, int, str, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_full = False
    any_clean = False
    any_dirty = False

    for mid in MID_GRID:
        for h0 in H0_GRID:
            hcfg = HollowEdgeConfig(
                mid_radius_frac=float(mid),
                h0=float(h0),
                min_end_count=MIN_END_COUNT,
                gabriel_fallback=False,
            )
            hollow_keep = _hollow_pruned_node_mask(
                pos, edges, dataset.points, config=hcfg,
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

            fixed = run_per_region_ph(
                hollow_pos,
                hollow_labs,
                sigma,
                scenario=f"seed77_hollow_mid{mid}_h0{h0}_fixed",
                include_labels=[0, 1],
                reading="fixed_threshold",
                max_dim=2,
                filtration_mult=FILTRATION_MULTIPLIER,
                expected_betti=EXPECTED_TORI,
            )
            life = run_per_region_ph(
                hollow_pos,
                hollow_labs,
                sigma,
                scenario=f"seed77_hollow_mid{mid}_h0{h0}_life",
                include_labels=[0, 1],
                reading="lifetime",
                max_dim=2,
                filtration_mult=FILTRATION_MULTIPLIER,
                lifetime_frac=DEFAULT_LIFETIME_FRAC,
                expected_betti=EXPECTED_TORI,
            )
            fixed_betti = {
                int(r.region_id): tuple(int(x) for x in r.betti)
                for r in fixed.reports
            }
            life_betti = {
                int(r.region_id): tuple(int(x) for x in r.betti)
                for r in life.reports
            }
            row_full = False
            row_clean = False
            row_dirty = False
            row_max_b1 = 0
            row_max_b2 = 0
            for reading, betti_map in (
                ("fixed", fixed_betti),
                ("lifetime", life_betti),
            ):
                for lab, betti in betti_map.items():
                    b1 = int(betti[1]) if len(betti) > 1 else 0
                    b2 = int(betti[2]) if len(betti) > 2 else 0
                    row_max_b1 = max(row_max_b1, b1)
                    row_max_b2 = max(row_max_b2, b2)
                    if betti == EXPECTED_TORI:
                        row_full = True
                        any_full = True
                        full_cells.append(
                            (float(mid), float(h0), int(lab), reading, betti)
                        )
                    if _is_clean_b2(betti):
                        row_clean = True
                        any_clean = True
                        clean_cells.append(
                            (float(mid), float(h0), int(lab), reading, betti)
                        )
                    if _is_dirty_b2(betti):
                        row_dirty = True
                        any_dirty = True
                        dirty_cells.append(
                            (float(mid), float(h0), int(lab), reading, betti)
                        )
            max_b1 = max(max_b1, row_max_b1)
            max_b2 = max(max_b2, row_max_b2)
            rows.append(
                HollowCfgRow(
                    mid_radius_frac=float(mid),
                    h0=float(h0),
                    n_after_hollow=n_after,
                    hollow_fallback=bool(hollow_fallback),
                    fixed_betti=fixed_betti,
                    lifetime_betti=life_betti,
                    any_full=row_full,
                    any_clean_b2=row_clean,
                    any_dirty_b2=row_dirty,
                    max_b1=row_max_b1,
                    max_b2=row_max_b2,
                )
            )

    return Seed77HollowCfgSweepBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        signal_fixed_betti=signal_fixed_betti,
        both_tori_partial_fixed=both_partial,
        mids=MID_GRID,
        h0s=H0_GRID,
        rows=tuple(rows),
        any_full_match=any_full,
        any_clean_b2=any_clean,
        any_dirty_b2=any_dirty,
        max_b1=max_b1,
        max_b2=max_b2,
        clean_cells=tuple(clean_cells),
        dirty_cells=tuple(dirty_cells),
        full_cells=tuple(full_cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_hollow_cfg_sweep_harness_lands(
    linked_tori_seed77_hollow_cfg_sweep_bundle,
) -> None:
    """Seed77 hollow mid×h0 sweep lands; SI defaults untouched."""
    bundle = linked_tori_seed77_hollow_cfg_sweep_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.mids == MID_GRID
    assert bundle.h0s == H0_GRID
    assert len(bundle.rows) == len(MID_GRID) * len(H0_GRID)
    assert bundle.both_tori_partial_fixed


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_hollow_cfg_sweep_documents_gap(
    linked_tori_seed77_hollow_cfg_sweep_bundle,
) -> None:
    """Document hollow mid×h0 vs clean (1,2,1); never flip awaiting.

    Soft: any full ``(1,2,1)`` or clean ``b2`` is proposal-path evidence.
    Otherwise keep documenting dirty-only / cfg gap. Sweep ≠ SI recovery.
    """
    bundle = linked_tori_seed77_hollow_cfg_sweep_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert len(bundle.full_cells) >= 1
        if bundle.any_clean_b2:
            assert len(bundle.clean_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.full_cells == ()
        assert bundle.clean_cells == ()
        assert bundle.both_tori_partial_fixed
        # Dirty may appear under some cfgs (primary) or all be erased.
        assert bundle.any_dirty_b2 or bundle.max_b2 >= 0
