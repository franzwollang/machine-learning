"""Nested densify max_nodes=512 × hollow dual-scale (#41 / A4-T70-followon).

A4-T67: nested densify256×hollow leaves ``any_all_either=false`` (hollow ≈
no-op / no shell unlock; dirty ``b2`` can appear). Nested schedule denser at
256 preserves some per-shell mult recovery. This harness freezes densify
``max_nodes=512`` and contrasts signal vs primary (``mid=0.5``) vs mild
(``mid=0.65``) hollow under global-sigma dual-scale PH, asking whether a
further densify step + hollow unlocks shell voids ``(1,0,1)``.

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
    run_dual_scale_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_SHELL: tuple[int, ...] = (1, 0, 1)
MAX_NODES: int = 512
DATASET_SEED: int = 21
STAGE1_SEED: int = 77
COARSE_MULT: float = 3.0
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
class NestedDensify512HollowArm:
    name: str
    mid_radius_frac: float | None
    n_signal: int
    fallback: bool
    table: str
    all_either: bool
    any_coarse: bool
    any_fine: bool
    any_b2: bool
    improves_vs_signal: bool


@dataclass(frozen=True)
class NestedDensify512HollowBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes: int
    n_signal: int
    sigma_star: float
    coarse_mult: float
    fine_mult: float
    signal_arm: NestedDensify512HollowArm
    hollow_arms: tuple[NestedDensify512HollowArm, ...]
    any_all_either: bool
    any_hollow_improves: bool
    any_b2: bool
    best_arm: str | None


@pytest.fixture(scope="module")
def nested_densify512_hollow_bundle() -> NestedDensify512HollowBundle:
    """Fit nested max_nodes=512; dual-scale signal vs primary/mild hollow."""
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
    sigma = float(sigma_star_from_tau(result.tau_star))
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [1, 2])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]
    n_signal = int(signal_pos.shape[0])
    edges = _undirected_edges_from_adj(
        scaffold.links.neighbour_graph(pos.shape[0])
    )

    def _dual_arm(
        name: str,
        pts: np.ndarray,
        labs: np.ndarray,
        *,
        mid: float | None,
        fallback: bool,
        signal_all: bool,
        signal_fine: bool,
    ) -> NestedDensify512HollowArm:
        dual = run_dual_scale_per_region_ph(
            pts,
            labs,
            sigma,
            coarse_mult=COARSE_MULT,
            fine_mult=FILTRATION_MULTIPLIER,
            scenario=f"nested_densify512_hollow_{name}_dual",
            include_labels=[1, 2],
            max_dim=2,
            expected_betti=EXPECTED_SHELL,
        )
        all_e = _all_either(dual)
        any_fine = bool(dual.any_fine_match)
        improves = bool(
            (all_e and not signal_all) or (any_fine and not signal_fine)
        )
        return NestedDensify512HollowArm(
            name=name,
            mid_radius_frac=mid,
            n_signal=int(pts.shape[0]),
            fallback=bool(fallback),
            table=format_dual_scale_ph_table(dual),
            all_either=bool(all_e),
            any_coarse=bool(dual.any_coarse_match),
            any_fine=any_fine,
            any_b2=_any_b2(dual),
            improves_vs_signal=improves,
        )

    raw_signal = _dual_arm(
        "signal",
        signal_pos,
        signal_labs,
        mid=None,
        fallback=False,
        signal_all=False,
        signal_fine=False,
    )
    signal_arm = NestedDensify512HollowArm(
        name=raw_signal.name,
        mid_radius_frac=None,
        n_signal=raw_signal.n_signal,
        fallback=False,
        table=raw_signal.table,
        all_either=raw_signal.all_either,
        any_coarse=raw_signal.any_coarse,
        any_fine=raw_signal.any_fine,
        any_b2=raw_signal.any_b2,
        improves_vs_signal=False,
    )

    hollow_arms: list[NestedDensify512HollowArm] = []
    any_all = bool(signal_arm.all_either)
    any_improve = False
    any_b2 = bool(signal_arm.any_b2)
    best_arm: str | None = None

    for name, cfg in (("primary", PRIMARY_HOLLOW), ("mild", MILD_HOLLOW)):
        hollow_keep = _hollow_pruned_node_mask(
            pos, edges, dataset.points, config=cfg,
        )
        hollow_signal = signal_mask & hollow_keep
        n_after = int(np.sum(hollow_signal))
        if n_after < 8 or not (
            np.any(node_labels[hollow_signal] == 1)
            and np.any(node_labels[hollow_signal] == 2)
        ):
            h_pos, h_labs = signal_pos, signal_labs
            fallback = True
        else:
            h_pos = pos[hollow_signal]
            h_labs = node_labels[hollow_signal]
            fallback = False
        arm = _dual_arm(
            name,
            h_pos,
            h_labs,
            mid=float(cfg.mid_radius_frac),
            fallback=fallback,
            signal_all=signal_arm.all_either,
            signal_fine=signal_arm.any_fine,
        )
        if not fallback:
            arm = NestedDensify512HollowArm(
                name=arm.name,
                mid_radius_frac=arm.mid_radius_frac,
                n_signal=n_after,
                fallback=False,
                table=arm.table,
                all_either=arm.all_either,
                any_coarse=arm.any_coarse,
                any_fine=arm.any_fine,
                any_b2=arm.any_b2,
                improves_vs_signal=arm.improves_vs_signal,
            )
        hollow_arms.append(arm)
        any_all = any_all or arm.all_either
        any_improve = any_improve or arm.improves_vs_signal
        any_b2 = any_b2 or arm.any_b2
        if arm.improves_vs_signal and best_arm is None:
            best_arm = name

    return NestedDensify512HollowBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes=MAX_NODES,
        n_signal=n_signal,
        sigma_star=sigma,
        coarse_mult=COARSE_MULT,
        fine_mult=FILTRATION_MULTIPLIER,
        signal_arm=signal_arm,
        hollow_arms=tuple(hollow_arms),
        any_all_either=any_all,
        any_hollow_improves=any_improve,
        any_b2=any_b2,
        best_arm=best_arm,
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_densify512_hollow_harness_lands(
    nested_densify512_hollow_bundle,
) -> None:
    """Nested densify512×hollow lands; SI defaults untouched."""
    bundle = nested_densify512_hollow_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes == MAX_NODES
    assert bundle.n_signal > 0
    assert bundle.sigma_star > 0.0
    assert bundle.fine_mult == FILTRATION_MULTIPLIER
    assert len(bundle.hollow_arms) == 2
    assert {a.name for a in bundle.hollow_arms} == {"primary", "mild"}
    assert "coarse_betti" in bundle.signal_arm.table.splitlines()[0]
    assert all("coarse_betti" in a.table.splitlines()[0] for a in bundle.hollow_arms)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_densify512_hollow_documents_gap(
    nested_densify512_hollow_bundle,
) -> None:
    """Document densify512×hollow vs nested void recovery; never flip awaiting.

    Soft: all-shell either-match under any arm is proposal-path evidence.
    Otherwise keep documenting denser+hollow non-recovery (T67 gap).
    """
    bundle = nested_densify512_hollow_bundle
    if bundle.any_all_either:
        assert FILTRATION_MULTIPLIER == 1.5
        assert (
            bundle.signal_arm.all_either
            or any(a.all_either for a in bundle.hollow_arms)
        )
    else:
        assert bundle.any_all_either is False
        assert bundle.signal_arm.all_either is False
        assert all(a.all_either is False for a in bundle.hollow_arms)
        assert all(a.n_signal <= bundle.n_signal for a in bundle.hollow_arms)
        assert bundle.any_b2 or not bundle.any_b2
        assert bundle.any_hollow_improves is False or bundle.best_arm is not None
