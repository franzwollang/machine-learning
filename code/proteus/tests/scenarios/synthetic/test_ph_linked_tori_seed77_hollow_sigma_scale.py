"""Primary hollow × sigma-scale on seed77 denser256 (#41 / A4-T56-followon).

A4-T52: primary ``mid=0.5×h0=0.7`` hollow invents dirty torus ``b2`` on
seed77 denser256. A4-T34: per-region local sigma (no hollow) did not unlock
``(1,2,1)``. This harness freezes the primary hollow prune and multiplies
``sigma_star`` by a scale grid under fixed_threshold, asking whether
filtration-radius retuning converts dirty/partial into clean void Betti.

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
SIGMA_SCALES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
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
class SigmaScaleRow:
    scale: float
    sigma: float
    betti: dict[int, tuple[int, ...]]
    any_full: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    both_partial: bool


@dataclass(frozen=True)
class Seed77HollowSigmaScaleBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    n_after_hollow: int
    hollow_fallback: bool
    sigma_star: float
    mid_radius_frac: float
    h0: float
    scales: tuple[float, ...]
    signal_fixed_betti: dict[int, tuple[int, ...]]
    hollow_fixed_betti: dict[int, tuple[int, ...]]
    rows: tuple[SigmaScaleRow, ...]
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    clean_cells: tuple[tuple[float, int, tuple[int, ...]], ...]
    dirty_cells: tuple[tuple[float, int, tuple[int, ...]], ...]
    table: str


@pytest.fixture(scope="module")
def linked_tori_seed77_hollow_sigma_scale_bundle() -> Seed77HollowSigmaScaleBundle:
    """Fit seed77 denser256; primary hollow; sweep sigma scales."""
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
    hollow_keep = _hollow_pruned_node_mask(
        pos, edges, dataset.points, config=HOLLOW_CFG,
    )
    hollow_signal = signal_mask & hollow_keep
    n_signal = int(np.sum(signal_mask))
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
        sigma_star,
        scenario="linked_tori_seed77_sigma_scale_signal_fixed",
        include_labels=[0, 1],
        reading="fixed_threshold",
        max_dim=2,
        filtration_mult=FILTRATION_MULTIPLIER,
        expected_betti=EXPECTED_TORI,
    )
    hollow_fixed = run_per_region_ph(
        hollow_pos,
        hollow_labs,
        sigma_star,
        scenario="linked_tori_seed77_sigma_scale_hollow_fixed",
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

    rows: list[SigmaScaleRow] = []
    clean_cells: list[tuple[float, int, tuple[int, ...]]] = []
    dirty_cells: list[tuple[float, int, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_full = False
    any_clean = False
    any_dirty = False
    table_lines = ["scale\tsigma\tregion\tbetti\tclean\tdirty"]

    for scale in SIGMA_SCALES:
        sigma = float(sigma_star * scale)
        ph = run_per_region_ph(
            hollow_pos,
            hollow_labs,
            sigma,
            scenario=f"linked_tori_seed77_hollow_sigma_scale_{scale:g}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti)
            for r in ph.reports
        }
        row_full = False
        row_clean = False
        row_dirty = False
        row_b1 = 0
        row_b2 = 0
        for rid, b in betti.items():
            b1 = int(b[1]) if len(b) > 1 else 0
            b2 = int(b[2]) if len(b) > 2 else 0
            row_b1 = max(row_b1, b1)
            row_b2 = max(row_b2, b2)
            is_clean = _is_clean_b2(b)
            is_dirty = _is_dirty_b2(b)
            if b == EXPECTED_TORI:
                row_full = True
                any_full = True
            if is_clean:
                row_clean = True
                any_clean = True
                clean_cells.append((float(scale), int(rid), b))
            if is_dirty:
                row_dirty = True
                any_dirty = True
                dirty_cells.append((float(scale), int(rid), b))
            table_lines.append(
                f"{scale:g}\t{sigma:.6g}\t{rid}\t{b}\t{is_clean}\t{is_dirty}"
            )
        max_b1 = max(max_b1, row_b1)
        max_b2 = max(max_b2, row_b2)
        both_partial = betti.get(0) == (1, 2, 0) and betti.get(1) == (1, 2, 0)
        rows.append(
            SigmaScaleRow(
                scale=float(scale),
                sigma=sigma,
                betti=betti,
                any_full=row_full,
                any_clean_b2=row_clean,
                any_dirty_b2=row_dirty,
                max_b1=row_b1,
                max_b2=row_b2,
                both_partial=both_partial,
            )
        )

    return Seed77HollowSigmaScaleBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=n_signal,
        n_after_hollow=n_after,
        hollow_fallback=bool(hollow_fallback),
        sigma_star=sigma_star,
        mid_radius_frac=float(HOLLOW_CFG.mid_radius_frac),
        h0=float(HOLLOW_CFG.h0),
        scales=SIGMA_SCALES,
        signal_fixed_betti=signal_fixed_betti,
        hollow_fixed_betti=hollow_fixed_betti,
        rows=tuple(rows),
        any_full_match=any_full,
        any_clean_b2=any_clean,
        any_dirty_b2=any_dirty,
        max_b1=max_b1,
        max_b2=max_b2,
        clean_cells=tuple(clean_cells),
        dirty_cells=tuple(dirty_cells),
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_hollow_sigma_scale_harness_lands(
    linked_tori_seed77_hollow_sigma_scale_bundle,
) -> None:
    """Primary hollow × sigma-scale harness lands; SI defaults untouched."""
    bundle = linked_tori_seed77_hollow_sigma_scale_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.mid_radius_frac == 0.5
    assert bundle.h0 == 0.7
    assert bundle.n_signal > 0
    assert bundle.n_after_hollow >= 0
    assert bundle.sigma_star > 0.0
    assert bundle.scales == SIGMA_SCALES
    assert len(bundle.rows) == len(SIGMA_SCALES)
    header = bundle.table.splitlines()[0]
    assert "scale" in header and "sigma" in header and "betti" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_hollow_sigma_scale_documents_gap(
    linked_tori_seed77_hollow_sigma_scale_bundle,
) -> None:
    """Document sigma-scale vs clean (1,2,1); never flip awaiting.

    Soft: any full/clean cell is proposal-path evidence. Otherwise keep
    documenting dirty/partial gap under primary hollow.
    """
    bundle = linked_tori_seed77_hollow_sigma_scale_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_full_match:
            assert any(r.any_full for r in bundle.rows)
        if bundle.any_clean_b2:
            assert len(bundle.clean_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.clean_cells == ()
        assert bundle.n_after_hollow <= bundle.n_signal
        # Baseline scale=1.0 should still be present in the grid.
        assert any(abs(r.scale - 1.0) < 1e-12 for r in bundle.rows)
        assert bundle.max_b2 >= 0
