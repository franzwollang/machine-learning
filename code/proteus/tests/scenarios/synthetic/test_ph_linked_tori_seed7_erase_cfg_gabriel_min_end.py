"""Seed7 erase-cfg × gabriel / min_end soft (#41 / A4-T88).

A4-T79/T82/T85: erase cfgs ``mid=0.35 × h0∈{0.5,0.7}`` with
``gabriel_fallback=False`` and ``min_end_count=0.5`` keep ``max_b2=0`` —
erase≠recover under SI, cal-mult, sigma-scale, or soft lifetime. This
harness freezes those erase mid/h0 pairs and crosses a soft
``gabriel_fallback∈{False,True}`` × ``min_end_count∈{0.25,0.5,1.0,2.0}``
grid (plus optional ``require_gabriel_and_h`` conjunction when gabriel
is on), asking whether Gabriel/min_end retuning unlocks clean void Betti
after erase.

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
    sweep_lifetime_mult_grid_per_region,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES: int = 256
DATASET_SEED: int = 2
STAGE1_SEED: int = 7
# T79/T82/T85 erase band.
ERASE_CFGS: tuple[tuple[float, float], ...] = ((0.35, 0.5), (0.35, 0.7))
GABRIEL_GRID: tuple[bool, ...] = (False, True)
MIN_END_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
# Soft lifetime at SI mult only (proposal-path recover probe).
SOFT_FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
SOFT_MULT_ARMS: tuple[float, ...] = (FILTRATION_MULTIPLIER,)


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
class Seed7EraseGabrielMinEndRow:
    mid_radius_frac: float
    h0: float
    gabriel_fallback: bool
    min_end_count: float
    require_gabriel_and_h: bool
    n_after_hollow: int
    hollow_fallback: bool
    fixed_betti: dict[int, tuple[int, ...]]
    fixed_max_b2: int
    soft_any_full: bool
    soft_any_clean_b2: bool
    soft_any_dirty_b2: bool
    soft_max_b2: int
    any_full: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    soft_clean_cells: tuple[tuple[float, float, int, tuple[int, ...]], ...]


@dataclass(frozen=True)
class Seed7EraseGabrielMinEndBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    sigma_star: float
    signal_fixed_betti: dict[int, tuple[int, ...]]
    signal_fixed_dirty_b2: bool
    erase_cfgs: tuple[tuple[float, float], ...]
    gabriel_grid: tuple[bool, ...]
    min_end_grid: tuple[float, ...]
    soft_fracs: tuple[float, ...]
    soft_mults: tuple[float, ...]
    rows: tuple[Seed7EraseGabrielMinEndRow, ...]
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    best_cfg: tuple[float, float, bool, float, bool] | None
    erase_fixed_max_b2: int
    table: str


@pytest.fixture(scope="module")
def linked_tori_seed7_erase_cfg_gabriel_min_end_bundle() -> (
    Seed7EraseGabrielMinEndBundle
):
    """Fit seed7 denser256; erase × gabriel/min_end × soft lifetime."""
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
    sigma_star = float(sigma_star_from_tau(result.tau_star))
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [0, 1])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]
    adj = scaffold.links.neighbour_graph(pos.shape[0])
    edges = _undirected_edges_from_adj(adj)

    signal_fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma_star,
        scenario="linked_tori_seed7_erase_gab_signal_fixed",
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
    signal_fixed_dirty = any(
        _is_dirty_b2(b) for b in signal_fixed_betti.values()
    )

    rows: list[Seed7EraseGabrielMinEndRow] = []
    any_full = False
    any_clean = False
    any_dirty = False
    max_b1 = 0
    max_b2 = 0
    erase_fixed_max_b2 = 0
    best_cfg: tuple[float, float, bool, float, bool] | None = None
    best_score = -1
    table_lines = [
        "mid\th0\tgab\tmin_end\tconj\taxis\tkey\tregion\tbetti\tclean\tdirty"
    ]

    # When gabriel is on, also try require_gabriel_and_h conjunction once.
    conj_arms: tuple[bool, ...] = (False, True)

    for mid, h0 in ERASE_CFGS:
        for gab in GABRIEL_GRID:
            for min_end in MIN_END_GRID:
                for conj in conj_arms:
                    if conj and not gab:
                        # Conjunction without gabriel is a no-op; skip.
                        continue
                    hcfg = HollowEdgeConfig(
                        mid_radius_frac=float(mid),
                        h0=float(h0),
                        min_end_count=float(min_end),
                        gabriel_fallback=bool(gab),
                        require_gabriel_and_h=bool(conj),
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
                        sigma_star,
                        scenario=(
                            f"seed7_erase_gab_mid{mid}_h0{h0}_"
                            f"g{int(gab)}_me{min_end:g}_c{int(conj)}_fixed"
                        ),
                        include_labels=[0, 1],
                        reading="fixed_threshold",
                        max_dim=2,
                        filtration_mult=FILTRATION_MULTIPLIER,
                        expected_betti=EXPECTED_TORI,
                    )
                    fixed_betti = {
                        int(r.region_id): tuple(int(x) for x in r.betti)
                        for r in fixed.reports
                    }
                    fixed_max_b2 = max(
                        (int(b[2]) if len(b) > 2 else 0)
                        for b in fixed_betti.values()
                    ) if fixed_betti else 0
                    erase_fixed_max_b2 = max(erase_fixed_max_b2, fixed_max_b2)

                    soft_clean_cells: list[
                        tuple[float, float, int, tuple[int, ...]]
                    ] = []
                    soft_full = False
                    soft_clean = False
                    soft_dirty = False
                    soft_max_b2 = 0
                    fixed_full = False
                    fixed_clean = False
                    fixed_dirty = False
                    row_max_b1 = 0
                    row_max_b2 = 0

                    for betti in fixed_betti.values():
                        b1 = int(betti[1]) if len(betti) > 1 else 0
                        b2 = int(betti[2]) if len(betti) > 2 else 0
                        row_max_b1 = max(row_max_b1, b1)
                        row_max_b2 = max(row_max_b2, b2)
                        if _is_dirty_b2(betti):
                            fixed_dirty = True
                            any_dirty = True
                        if _is_clean_b2(betti):
                            fixed_clean = True
                            any_clean = True
                        if betti == EXPECTED_TORI:
                            fixed_full = True
                            any_full = True
                        table_lines.append(
                            f"{mid:g}\t{h0:g}\t{int(gab)}\t{min_end:g}\t"
                            f"{int(conj)}\tfixed\tSI\t-\t{betti}\t"
                            f"{int(_is_clean_b2(betti))}\t"
                            f"{int(_is_dirty_b2(betti))}"
                        )

                    soft_grid = sweep_lifetime_mult_grid_per_region(
                        hollow_pos,
                        hollow_labs,
                        sigma_star,
                        fracs=SOFT_FRAC_GRID,
                        mults=SOFT_MULT_ARMS,
                        scenario=(
                            f"seed7_erase_gab_soft_mid{mid}_h0{h0}_"
                            f"g{int(gab)}_me{min_end:g}_c{int(conj)}"
                        ),
                        include_labels=[0, 1],
                        max_dim=2,
                        expected_betti=EXPECTED_TORI,
                        b1_target=2,
                    )
                    for cell in soft_grid.rows:
                        betti = tuple(int(x) for x in cell.betti)
                        rid = int(cell.region_id)
                        frac = float(cell.lifetime_frac)
                        mult = float(cell.filtration_mult)
                        b1 = int(betti[1]) if len(betti) > 1 else 0
                        b2 = int(betti[2]) if len(betti) > 2 else 0
                        row_max_b1 = max(row_max_b1, b1)
                        row_max_b2 = max(row_max_b2, b2)
                        soft_max_b2 = max(soft_max_b2, b2)
                        is_clean = _is_clean_b2(betti)
                        is_dirty = _is_dirty_b2(betti)
                        if betti == EXPECTED_TORI:
                            soft_full = True
                            any_full = True
                        if is_clean:
                            soft_clean = True
                            any_clean = True
                            soft_clean_cells.append((frac, mult, rid, betti))
                        if is_dirty:
                            soft_dirty = True
                            any_dirty = True
                        table_lines.append(
                            f"{mid:g}\t{h0:g}\t{int(gab)}\t{min_end:g}\t"
                            f"{int(conj)}\tsoft\t{frac:g}x{mult:g}\t{rid}\t"
                            f"{betti}\t{int(is_clean)}\t{int(is_dirty)}"
                        )

                    row_full = soft_full or fixed_full
                    row_clean = soft_clean or fixed_clean
                    row_dirty = soft_dirty or fixed_dirty
                    score = (
                        (1000 if row_full else 0)
                        + (100 if row_clean else 0)
                        + 10 * row_max_b2
                        + row_max_b1
                        - (5 if row_dirty else 0)
                    )
                    if score > best_score:
                        best_score = score
                        best_cfg = (
                            float(mid),
                            float(h0),
                            bool(gab),
                            float(min_end),
                            bool(conj),
                        )

                    max_b1 = max(max_b1, row_max_b1)
                    max_b2 = max(max_b2, row_max_b2)
                    rows.append(
                        Seed7EraseGabrielMinEndRow(
                            mid_radius_frac=float(mid),
                            h0=float(h0),
                            gabriel_fallback=bool(gab),
                            min_end_count=float(min_end),
                            require_gabriel_and_h=bool(conj),
                            n_after_hollow=n_after,
                            hollow_fallback=bool(hollow_fallback),
                            fixed_betti=fixed_betti,
                            fixed_max_b2=int(fixed_max_b2),
                            soft_any_full=soft_full,
                            soft_any_clean_b2=soft_clean,
                            soft_any_dirty_b2=soft_dirty,
                            soft_max_b2=int(soft_max_b2),
                            any_full=row_full,
                            any_clean_b2=row_clean,
                            any_dirty_b2=row_dirty,
                            max_b1=row_max_b1,
                            max_b2=row_max_b2,
                            soft_clean_cells=tuple(soft_clean_cells),
                        )
                    )

    return Seed7EraseGabrielMinEndBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma_star),
        signal_fixed_betti=signal_fixed_betti,
        signal_fixed_dirty_b2=bool(signal_fixed_dirty),
        erase_cfgs=ERASE_CFGS,
        gabriel_grid=GABRIEL_GRID,
        min_end_grid=MIN_END_GRID,
        soft_fracs=SOFT_FRAC_GRID,
        soft_mults=SOFT_MULT_ARMS,
        rows=tuple(rows),
        any_full_match=any_full,
        any_clean_b2=any_clean,
        any_dirty_b2=any_dirty,
        max_b1=max_b1,
        max_b2=max_b2,
        best_cfg=best_cfg,
        erase_fixed_max_b2=int(erase_fixed_max_b2),
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed7_erase_cfg_gabriel_min_end_harness_lands(
    linked_tori_seed7_erase_cfg_gabriel_min_end_bundle,
) -> None:
    """Seed7 erase×gabriel/min_end lands; SI defaults untouched."""
    bundle = linked_tori_seed7_erase_cfg_gabriel_min_end_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.erase_cfgs == ERASE_CFGS
    assert bundle.gabriel_grid == GABRIEL_GRID
    assert bundle.min_end_grid == MIN_END_GRID
    assert bundle.soft_fracs == SOFT_FRAC_GRID
    assert bundle.soft_mults == SOFT_MULT_ARMS
    # 2 erase × (4 min_end × gab=False + 4 min_end × gab=True × 2 conj)
    # = 2 × (4 + 8) = 24
    assert len(bundle.rows) == 24
    assert bundle.signal_fixed_dirty_b2 is True
    assert bundle.best_cfg is not None
    header = bundle.table.splitlines()[0]
    assert "gab" in header and "min_end" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed7_erase_cfg_gabriel_min_end_documents_gap(
    linked_tori_seed7_erase_cfg_gabriel_min_end_bundle,
) -> None:
    """Document erase×gabriel/min_end vs clean (1,2,1); never flip awaiting.

    Soft: any full ``(1,2,1)`` or clean ``b2`` is proposal-path evidence.
    Otherwise keep documenting erase≠recover under gabriel/min_end soft.
    """
    bundle = linked_tori_seed7_erase_cfg_gabriel_min_end_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        assert bundle.max_b2 >= 1
        assert any(
            r.any_clean_b2 or r.any_full for r in bundle.rows
        )
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.signal_fixed_dirty_b2 is True
        assert all(not r.soft_clean_cells for r in bundle.rows)
        assert bundle.any_dirty_b2 or bundle.max_b2 >= 0
        assert bundle.erase_fixed_max_b2 >= 0
