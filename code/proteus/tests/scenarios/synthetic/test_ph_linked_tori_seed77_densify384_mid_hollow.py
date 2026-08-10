"""Densify max_nodes=384 mid-regime × hollow on seed77 (#41 / A4-T59-followon).

A4-T50: seed77 densify ladder keeps both-partial at 256/512 but **384 regresses**
to ``(1,1,0)`` (max_b2=0). A4-T52/T54: primary hollow unlocks dirty ``b2`` at
256 but not at 512. This harness freezes the mid-regime ``max_nodes=384``
scaffold and contrasts signal vs primary (``mid=0.5``) vs mild (``mid=0.65``)
hollow under fixed_threshold + lifetime, asking whether mid-density + hollow
recovers both-partial / unlocks void Betti.

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
MAX_NODES: int = 384
DATASET_SEED: int = 2
STAGE1_SEED: int = 77
FRAC_GRID: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
PRIMARY_HOLLOW = HollowEdgeConfig(
    mid_radius_frac=0.5, h0=0.7, min_end_count=0.5, gabriel_fallback=False,
)
MILD_HOLLOW = HollowEdgeConfig(
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


def _summarize_lifetime_rows(rows):
    recovering: list[tuple[int, float, tuple[int, ...]]] = []
    clean_cells: list[tuple[int, float, tuple[int, ...]]] = []
    dirty_cells: list[tuple[int, float, tuple[int, ...]]] = []
    max_b1 = 0
    max_b2 = 0
    any_full = False
    for row in rows:
        betti = tuple(int(x) for x in row.betti)
        rid = int(row.region_id) if row.region_id is not None else -1
        max_b1 = max(max_b1, int(betti[1]) if len(betti) > 1 else 0)
        max_b2 = max(max_b2, int(betti[2]) if len(betti) > 2 else 0)
        if betti == EXPECTED_TORI:
            any_full = True
            recovering.append((rid, float(row.lifetime_frac), betti))
        if _is_clean_b2(betti):
            clean_cells.append((rid, float(row.lifetime_frac), betti))
        if _is_dirty_b2(betti):
            dirty_cells.append((rid, float(row.lifetime_frac), betti))
    return (
        any_full,
        max_b1,
        max_b2,
        tuple(recovering),
        tuple(clean_cells),
        tuple(dirty_cells),
    )


def _apply_hollow(
    pos: np.ndarray,
    signal_mask: np.ndarray,
    node_labels: np.ndarray,
    signal_pos: np.ndarray,
    signal_labs: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    cfg: HollowEdgeConfig,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    hollow_keep = _hollow_pruned_node_mask(pos, edges, data, config=cfg)
    hollow_signal = signal_mask & hollow_keep
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not (
        np.any(node_labels[hollow_signal] == 0)
        and np.any(node_labels[hollow_signal] == 1)
    ):
        return signal_pos, signal_labs, n_after, True
    return pos[hollow_signal], node_labels[hollow_signal], n_after, False


@dataclass(frozen=True)
class HollowArm:
    name: str
    mid_radius_frac: float
    n_after: int
    fallback: bool
    fixed_betti: dict[int, tuple[int, ...]]
    both_partial: bool
    table: str
    any_full: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b1: int
    max_b2: int
    recovering: tuple[tuple[int, float, tuple[int, ...]], ...]
    clean_cells: tuple[tuple[int, float, tuple[int, ...]], ...]
    dirty_cells: tuple[tuple[int, float, tuple[int, ...]], ...]


@dataclass(frozen=True)
class Seed77Densify384MidHollowBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    sigma_star: float
    signal_fixed_betti: dict[int, tuple[int, ...]]
    signal_both_partial: bool
    signal_max_b1: int
    signal_max_b2: int
    fracs: tuple[float, ...]
    arms: tuple[HollowArm, ...]
    any_full_match: bool
    any_clean_b2: bool
    any_dirty_b2: bool
    max_b2: int
    any_both_partial_after_hollow: bool


@pytest.fixture(scope="module")
def linked_tori_seed77_densify384_mid_hollow_bundle() -> (
    Seed77Densify384MidHollowBundle
):
    """Fit seed77 max_nodes=384; contrast signal vs primary/mild hollow."""
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
    edges = _undirected_edges_from_adj(
        scaffold.links.neighbour_graph(pos.shape[0])
    )

    signal_fixed = run_per_region_ph(
        signal_pos,
        signal_labs,
        sigma,
        scenario="linked_tori_seed77_m384_signal_fixed",
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
    signal_both = (
        signal_fixed_betti.get(0) == (1, 2, 0)
        and signal_fixed_betti.get(1) == (1, 2, 0)
    )
    signal_max_b1 = max(
        (int(b[1]) for b in signal_fixed_betti.values() if len(b) > 1),
        default=0,
    )
    signal_max_b2 = max(
        (int(b[2]) for b in signal_fixed_betti.values() if len(b) > 2),
        default=0,
    )

    arms: list[HollowArm] = []
    any_full = False
    any_clean = False
    any_dirty = False
    max_b2 = signal_max_b2
    any_both_hollow = False

    for name, cfg in (("primary", PRIMARY_HOLLOW), ("mild", MILD_HOLLOW)):
        h_pos, h_labs, n_after, fallback = _apply_hollow(
            pos,
            signal_mask,
            node_labels,
            signal_pos,
            signal_labs,
            edges,
            dataset.points,
            cfg,
        )
        fixed = run_per_region_ph(
            h_pos,
            h_labs,
            sigma,
            scenario=f"linked_tori_seed77_m384_{name}_hollow_fixed",
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
        both = (
            fixed_betti.get(0) == (1, 2, 0)
            and fixed_betti.get(1) == (1, 2, 0)
        )
        if both:
            any_both_hollow = True
        rows = sweep_lifetime_frac_per_region(
            h_pos,
            h_labs,
            sigma,
            fracs=FRAC_GRID,
            include_labels=[0, 1],
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            target_betti=EXPECTED_TORI,
        )
        hol_full, hol_b1, hol_b2, hol_rec, hol_clean, hol_dirty = (
            _summarize_lifetime_rows(rows)
        )
        # Also count fixed-threshold dirty/clean.
        for b in fixed_betti.values():
            if _is_clean_b2(b):
                any_clean = True
            if _is_dirty_b2(b):
                any_dirty = True
            if b == EXPECTED_TORI:
                any_full = True
        any_full = any_full or hol_full
        any_clean = any_clean or bool(hol_clean)
        any_dirty = any_dirty or bool(hol_dirty)
        max_b2 = max(max_b2, hol_b2)
        for b in fixed_betti.values():
            if len(b) > 2:
                max_b2 = max(max_b2, int(b[2]))

        arms.append(
            HollowArm(
                name=name,
                mid_radius_frac=float(cfg.mid_radius_frac),
                n_after=n_after,
                fallback=bool(fallback),
                fixed_betti=fixed_betti,
                both_partial=both,
                table=format_lifetime_frac_sweep_table(rows),
                any_full=hol_full,
                any_clean_b2=bool(hol_clean),
                any_dirty_b2=bool(hol_dirty),
                max_b1=hol_b1,
                max_b2=hol_b2,
                recovering=hol_rec,
                clean_cells=hol_clean,
                dirty_cells=hol_dirty,
            )
        )

    return Seed77Densify384MidHollowBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=int(signal_pos.shape[0]),
        sigma_star=float(sigma),
        signal_fixed_betti=signal_fixed_betti,
        signal_both_partial=signal_both,
        signal_max_b1=signal_max_b1,
        signal_max_b2=signal_max_b2,
        fracs=FRAC_GRID,
        arms=tuple(arms),
        any_full_match=any_full,
        any_clean_b2=any_clean,
        any_dirty_b2=any_dirty,
        max_b2=max_b2,
        any_both_partial_after_hollow=any_both_hollow,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_densify384_mid_hollow_harness_lands(
    linked_tori_seed77_densify384_mid_hollow_bundle,
) -> None:
    """Seed77 densify384×hollow lands; SI defaults untouched."""
    bundle = linked_tori_seed77_densify384_mid_hollow_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert DEFAULT_LIFETIME_FRAC == 0.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.fracs == FRAC_GRID
    assert len(bundle.arms) == 2
    assert {a.name for a in bundle.arms} == {"primary", "mild"}
    assert all(len(a.table) > 0 for a in bundle.arms)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_seed77_densify384_mid_hollow_documents_gap(
    linked_tori_seed77_densify384_mid_hollow_bundle,
) -> None:
    """Document densify384 mid-regime×hollow vs clean (1,2,1); never flip awaiting.

    Soft: any full ``(1,2,1)`` or clean ``b2`` is proposal-path evidence.
    Otherwise keep documenting mid-regime regression / hollow gap.
    """
    bundle = linked_tori_seed77_densify384_mid_hollow_bundle
    if bundle.any_full_match or bundle.any_clean_b2:
        assert FILTRATION_MULTIPLIER == 1.5
        if bundle.any_clean_b2:
            assert bundle.max_b2 >= 1
    else:
        assert bundle.any_full_match is False
        assert bundle.any_clean_b2 is False
        # T50: 384 signal alone typically regresses both-partial; record either way.
        assert bundle.signal_max_b1 >= 0
        assert bundle.any_dirty_b2 or bundle.max_b2 >= 0
        assert all(a.n_after <= bundle.n_signal for a in bundle.arms)
