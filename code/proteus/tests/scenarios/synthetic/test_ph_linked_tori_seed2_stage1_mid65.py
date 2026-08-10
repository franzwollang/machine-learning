"""Stage-1 seed × mild mid=0.65 hollow on denser seed2 (#41 / A4-T57-followon).

A4-T44: Stage-1 seed sweep on denser seed2 (no hollow) never unlocked ``b2``.
A4-T52/T53: mild ``mid=0.65`` preserves both-partial without dirty on seed77.
This harness crosses a compact Stage-1 seed grid with mid=0.65 hollow on the
seed2 denser256 recipe, asking whether Stage-1 stochasticity + mild hollow
unlocks clean ``(1,2,1)``.

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
# Compact vs T44's 10-seed grid — keep turn runtime bounded.
STAGE1_SEEDS: tuple[int, ...] = (2, 7, 21, 42, 77, 99)
HOLLOW_CFG = HollowEdgeConfig(
    mid_radius_frac=0.65, h0=0.7, min_end_count=0.5, gabriel_fallback=False,
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
class Stage1Mid65Row:
    stage1_seed: int
    n_signal: int
    n_after_hollow: int
    hollow_fallback: bool
    sigma_star: float
    signal_betti: dict[int, tuple[int, ...]]
    hollow_betti: dict[int, tuple[int, ...]]
    both_partial_signal: bool
    both_partial_hollow: bool
    any_full: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int


@dataclass(frozen=True)
class Seed2Stage1Mid65Bundle:
    dataset_seed: int
    max_nodes: int
    mid_radius_frac: float
    h0: float
    stage1_seeds: tuple[int, ...]
    rows: tuple[Stage1Mid65Row, ...]
    n_seeds_both_partial_hollow: int
    n_seeds_with_b2: int
    n_seeds_full: int
    n_seeds_clean: int
    n_seeds_dirty: int
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    clean_cells: tuple[tuple[int, int, tuple[int, ...]], ...]
    dirty_cells: tuple[tuple[int, int, tuple[int, ...]], ...]
    table: str


@pytest.fixture(scope="module")
def linked_tori_seed2_stage1_mid65_bundle() -> Seed2Stage1Mid65Bundle:
    """Fit denser seed2 across Stage-1 seeds; mild mid=0.65 hollow PH."""
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

    rows: list[Stage1Mid65Row] = []
    clean_cells: list[tuple[int, int, tuple[int, ...]]] = []
    dirty_cells: list[tuple[int, int, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    n_both = 0
    n_b2 = 0
    n_full = 0
    n_clean = 0
    n_dirty = 0
    any_full = False
    any_clean = False
    any_dirty = False
    table_lines = [
        "stage1_seed\tn_after\tsignal_betti\thollow_betti\tclean\tdirty"
    ]

    for stage1_seed in STAGE1_SEEDS:
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
            seed=int(stage1_seed),
        )
        result = run_scale_search(
            dataset.points, dim=gt.ambient_dim, config=config,
        )
        scaffold = result.scaffold_at_star
        pos = scaffold.node_positions()
        sigma = float(sigma_star_from_tau(result.tau_star))
        node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
        signal_mask = np.isin(node_labels, [0, 1])
        signal_pos = pos[signal_mask]
        signal_labs = node_labels[signal_mask]
        n_signal = int(np.sum(signal_mask))

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

        signal_ph = run_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            scenario=f"linked_tori_seed2_s{stage1_seed}_signal",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        hollow_ph = run_per_region_ph(
            hollow_pos,
            hollow_labs,
            sigma,
            scenario=f"linked_tori_seed2_s{stage1_seed}_mid65",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        signal_betti = {
            int(r.region_id): tuple(int(x) for x in r.betti)
            for r in signal_ph.reports
        }
        hollow_betti = {
            int(r.region_id): tuple(int(x) for x in r.betti)
            for r in hollow_ph.reports
        }
        both_sig = (
            signal_betti.get(0) == (1, 2, 0)
            and signal_betti.get(1) == (1, 2, 0)
        )
        both_hol = (
            hollow_betti.get(0) == (1, 2, 0)
            and hollow_betti.get(1) == (1, 2, 0)
        )
        if both_hol:
            n_both += 1

        row_full = False
        row_clean = False
        row_dirty = False
        row_b1 = 0
        row_b2 = 0
        for rid, b in hollow_betti.items():
            b1 = int(b[1]) if len(b) > 1 else 0
            b2 = int(b[2]) if len(b) > 2 else 0
            row_b1 = max(row_b1, b1)
            row_b2 = max(row_b2, b2)
            if b == EXPECTED_TORI:
                row_full = True
                any_full = True
            if _is_clean_b2(b):
                row_clean = True
                any_clean = True
                clean_cells.append((int(stage1_seed), int(rid), b))
            if _is_dirty_b2(b):
                row_dirty = True
                any_dirty = True
                dirty_cells.append((int(stage1_seed), int(rid), b))

        if row_b2 >= 1:
            n_b2 += 1
        if row_full:
            n_full += 1
        if row_clean:
            n_clean += 1
        if row_dirty:
            n_dirty += 1
        max_b1 = max(max_b1, row_b1)
        max_b2 = max(max_b2, row_b2)

        table_lines.append(
            f"{stage1_seed}\t{n_after}\t{signal_betti}\t{hollow_betti}\t"
            f"{row_clean}\t{row_dirty}"
        )
        rows.append(
            Stage1Mid65Row(
                stage1_seed=int(stage1_seed),
                n_signal=n_signal,
                n_after_hollow=n_after,
                hollow_fallback=bool(hollow_fallback),
                sigma_star=sigma,
                signal_betti=signal_betti,
                hollow_betti=hollow_betti,
                both_partial_signal=both_sig,
                both_partial_hollow=both_hol,
                any_full=row_full,
                any_clean_b2=row_clean,
                any_dirty_b2=row_dirty,
                max_b1=row_b1,
                max_b2=row_b2,
            )
        )

    return Seed2Stage1Mid65Bundle(
        dataset_seed=DATASET_SEED,
        max_nodes=MAX_NODES,
        mid_radius_frac=float(HOLLOW_CFG.mid_radius_frac),
        h0=float(HOLLOW_CFG.h0),
        stage1_seeds=STAGE1_SEEDS,
        rows=tuple(rows),
        n_seeds_both_partial_hollow=n_both,
        n_seeds_with_b2=n_b2,
        n_seeds_full=n_full,
        n_seeds_clean=n_clean,
        n_seeds_dirty=n_dirty,
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
def test_linked_tori_seed2_stage1_mid65_harness_lands(
    linked_tori_seed2_stage1_mid65_bundle,
) -> None:
    """Stage-1 × mid65 harness lands; SI defaults untouched."""
    bundle = linked_tori_seed2_stage1_mid65_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.mid_radius_frac == 0.65
    assert bundle.h0 == 0.7
    assert bundle.stage1_seeds == STAGE1_SEEDS
    assert len(bundle.rows) == len(STAGE1_SEEDS)
    header = bundle.table.splitlines()[0]
    assert "stage1_seed" in header and "hollow_betti" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed2_stage1_mid65_documents_gap(
    linked_tori_seed2_stage1_mid65_bundle,
) -> None:
    """Document Stage-1×mid65 vs clean (1,2,1); never flip awaiting."""
    bundle = linked_tori_seed2_stage1_mid65_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_clean_b2:
            assert len(bundle.clean_cells) >= 1
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        assert bundle.clean_cells == ()
        assert bundle.n_seeds_full == 0
        assert bundle.n_seeds_clean == 0
        # Seed77 baseline row should exist in the compact grid.
        assert any(r.stage1_seed == 77 for r in bundle.rows)
        assert bundle.max_b2 >= 0
