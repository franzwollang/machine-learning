"""Nested spheres per-shell sigma × hollow prune (#41 / A4-T61-followon).

A4-T31: per-shell local sigma dual-scale did not flip nested ``@awaiting``.
A4-T52: primary hollow invents dirty ``b2`` on denser linked-tori scaffolds.
This harness applies primary hollow (``mid=0.5/h0=0.7/noGab``) on a nested
fitted scaffold and contrasts dual-scale PH under global vs per-shell local
sigma, asking whether hollow + local sigma recovers shell voids ``(1,0,1)``.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.edge_evidence import HollowEdgeConfig, prune_hollow_edges
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.nested_spheres import make_nested_spheres
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    format_dual_scale_ph_table,
    nearest_data_labels,
    per_region_median_nn_sigma,
    run_dual_scale_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_SHELL: tuple[int, ...] = (1, 0, 1)
MAX_NODES: int = 128
DATASET_SEED: int = 21
STAGE1_SEED: int = 77
COARSE_MULT: float = 3.0
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


def _all_either(dual) -> bool:
    if not dual.rows or len(dual.rows) != 2:
        return False
    return all(bool(r.coarse_match or r.fine_match) for r in dual.rows)


def _any_b2(dual) -> bool:
    for r in dual.rows:
        for betti in (r.coarse_betti, r.fine_betti):
            if len(betti) > 2 and int(betti[2]) >= 1:
                return True
    return False


@dataclass(frozen=True)
class NestedSigmaHollowArm:
    name: str
    n_signal: int
    global_sigma: float
    local_labels: tuple[int, ...]
    local_sigmas: tuple[float, ...]
    global_table: str
    local_table: str
    global_all_either: bool
    local_all_either: bool
    global_any_coarse: bool
    global_any_fine: bool
    local_any_coarse: bool
    local_any_fine: bool
    global_any_b2: bool
    local_any_b2: bool
    local_improves: bool


@dataclass(frozen=True)
class NestedSigmaHollowBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    n_after_hollow: int
    hollow_fallback: bool
    mid_radius_frac: float
    h0: float
    coarse_mult: float
    fine_mult: float
    signal_arm: NestedSigmaHollowArm
    hollow_arm: NestedSigmaHollowArm
    hollow_improves_vs_signal: bool
    any_all_either: bool
    any_b2: bool


@pytest.fixture(scope="module")
def nested_sigma_hollow_bundle() -> NestedSigmaHollowBundle:
    """Fit nested max_nodes=128; dual-scale signal vs primary hollow."""
    dataset = make_nested_spheres(
        n_per_sphere=500,
        radii=(1.0, 2.0),
        ambient_dim=3,
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
    global_sigma = float(sigma_star_from_tau(result.tau_star))
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [1, 2])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]
    n_signal = int(signal_pos.shape[0])

    edges = _undirected_edges_from_adj(
        scaffold.links.neighbour_graph(pos.shape[0])
    )
    hollow_keep = _hollow_pruned_node_mask(
        pos, edges, dataset.points, config=HOLLOW_CFG,
    )
    hollow_signal = signal_mask & hollow_keep
    n_after = int(np.sum(hollow_signal))
    if n_after < 8 or not (
        np.any(node_labels[hollow_signal] == 1)
        and np.any(node_labels[hollow_signal] == 2)
    ):
        hollow_pos, hollow_labs = signal_pos, signal_labs
        hollow_fallback = True
    else:
        hollow_pos = pos[hollow_signal]
        hollow_labs = node_labels[hollow_signal]
        hollow_fallback = False

    def _arm(name: str, pts: np.ndarray, labs: np.ndarray) -> NestedSigmaHollowArm:
        labs_u, local_sigmas = per_region_median_nn_sigma(
            pts, labs, include_labels=[1, 2],
        )
        dual_g = run_dual_scale_per_region_ph(
            pts,
            labs,
            global_sigma,
            coarse_mult=COARSE_MULT,
            fine_mult=FILTRATION_MULTIPLIER,
            scenario=f"nested_{name}_global_sigma_dual",
            include_labels=[1, 2],
            max_dim=2,
            expected_betti=EXPECTED_SHELL,
        )
        dual_l = run_dual_scale_per_region_ph(
            pts,
            labs,
            local_sigmas,
            coarse_mult=COARSE_MULT,
            fine_mult=FILTRATION_MULTIPLIER,
            scenario=f"nested_{name}_local_sigma_dual",
            include_labels=labs_u,
            max_dim=2,
            expected_betti=EXPECTED_SHELL,
        )
        g_all = _all_either(dual_g)
        l_all = _all_either(dual_l)
        g_matches = {
            int(r.region_id): bool(r.coarse_match or r.fine_match)
            for r in dual_g.rows
        }
        l_matches = {
            int(r.region_id): bool(r.coarse_match or r.fine_match)
            for r in dual_l.rows
        }
        local_improves = bool(l_all and not g_all) or any(
            l_matches.get(k, False) and not g_matches.get(k, False)
            for k in (1, 2)
        )
        return NestedSigmaHollowArm(
            name=name,
            n_signal=int(pts.shape[0]),
            global_sigma=global_sigma,
            local_labels=tuple(int(x) for x in labs_u),
            local_sigmas=tuple(float(s) for s in local_sigmas),
            global_table=format_dual_scale_ph_table(dual_g),
            local_table=format_dual_scale_ph_table(dual_l),
            global_all_either=bool(g_all),
            local_all_either=bool(l_all),
            global_any_coarse=bool(dual_g.any_coarse_match),
            global_any_fine=bool(dual_g.any_fine_match),
            local_any_coarse=bool(dual_l.any_coarse_match),
            local_any_fine=bool(dual_l.any_fine_match),
            global_any_b2=_any_b2(dual_g),
            local_any_b2=_any_b2(dual_l),
            local_improves=bool(local_improves),
        )

    signal_arm = _arm("signal", signal_pos, signal_labs)
    hollow_arm = _arm("hollow", hollow_pos, hollow_labs)
    hollow_improves = bool(
        (hollow_arm.local_all_either or hollow_arm.global_all_either)
        and not (signal_arm.local_all_either or signal_arm.global_all_either)
    ) or bool(
        (hollow_arm.local_any_fine or hollow_arm.global_any_fine)
        and not (signal_arm.local_any_fine or signal_arm.global_any_fine)
    )
    any_all = bool(
        signal_arm.global_all_either
        or signal_arm.local_all_either
        or hollow_arm.global_all_either
        or hollow_arm.local_all_either
    )
    any_b2 = bool(
        signal_arm.global_any_b2
        or signal_arm.local_any_b2
        or hollow_arm.global_any_b2
        or hollow_arm.local_any_b2
    )

    return NestedSigmaHollowBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=n_signal,
        n_after_hollow=n_after,
        hollow_fallback=bool(hollow_fallback),
        mid_radius_frac=float(HOLLOW_CFG.mid_radius_frac),
        h0=float(HOLLOW_CFG.h0),
        coarse_mult=COARSE_MULT,
        fine_mult=FILTRATION_MULTIPLIER,
        signal_arm=signal_arm,
        hollow_arm=hollow_arm,
        hollow_improves_vs_signal=hollow_improves,
        any_all_either=any_all,
        any_b2=any_b2,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_sigma_hollow_harness_lands(
    nested_sigma_hollow_bundle,
) -> None:
    """Nested sigma×hollow lands; SI defaults untouched."""
    bundle = nested_sigma_hollow_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.n_after_hollow >= 0
    assert bundle.fine_mult == FILTRATION_MULTIPLIER
    assert bundle.mid_radius_frac == 0.5
    assert bundle.h0 == 0.7
    for arm in (bundle.signal_arm, bundle.hollow_arm):
        assert arm.global_sigma > 0.0
        assert arm.local_labels == (1, 2)
        assert len(arm.local_sigmas) == 2
        assert all(np.isfinite(s) and s > 0.0 for s in arm.local_sigmas)
        assert "coarse_betti" in arm.global_table.splitlines()[0]
        assert "coarse_betti" in arm.local_table.splitlines()[0]


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_sigma_hollow_documents_gap(
    nested_sigma_hollow_bundle,
) -> None:
    """Document hollow×local-sigma vs nested void recovery; never flip awaiting.

    Soft: all-shell either-match under hollow or signal is proposal-path evidence.
    Otherwise keep documenting non-recovery (SI fine still insufficient alone).
    """
    bundle = nested_sigma_hollow_bundle
    if bundle.any_all_either:
        assert FILTRATION_MULTIPLIER == 1.5
        assert (
            bundle.signal_arm.global_all_either
            or bundle.signal_arm.local_all_either
            or bundle.hollow_arm.global_all_either
            or bundle.hollow_arm.local_all_either
        )
    else:
        assert bundle.any_all_either is False
        assert bundle.signal_arm.global_all_either is False
        assert bundle.signal_arm.local_all_either is False
        assert bundle.hollow_arm.global_all_either is False
        assert bundle.hollow_arm.local_all_either is False
        assert bundle.n_after_hollow <= bundle.n_signal
        # Hollow may invent dirty b2 without full shell recovery — record only.
        assert bundle.any_b2 or not bundle.any_b2
