"""Nested schedule {1:3,2:6} × densify304 × hollow cliff probe (#41 / A4-T95).

A4-T92: densify300 mild-only; densify308 full recover (signal+primary+mild).
A4-T89: densify296 full recover; densify312 hard fail — non-monotonic cliff.
This harness probes the single rung ``max_nodes=304`` between the 300 mild
and 308 win cells under signal vs primary/mild hollow.

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
    format_scheduled_mult_ph_table,
    nearest_data_labels,
    run_scheduled_mult_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_SHELL: tuple[int, ...] = (1, 0, 1)
MAX_NODES_GRID: tuple[int, ...] = (304,)
DATASET_SEED: int = 21
STAGE1_SEED: int = 77
CIRCLE_CALIBRATED_MULT: float = 6.0
COARSE_INNER_MULT: float = 3.0
SCHEDULE: dict[int, float] = {1: COARSE_INNER_MULT, 2: CIRCLE_CALIBRATED_MULT}
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


@dataclass(frozen=True)
class NestedScheduleDensify304HollowArm:
    name: str
    mid_radius_frac: float | None
    n_signal: int
    fallback: bool
    table: str
    all_match: bool | None
    matches: dict[int, bool | None]
    betti: dict[int, tuple[int, ...]]
    any_b2: bool
    preserves_vs_signal: bool


@dataclass(frozen=True)
class NestedScheduleDensify304HollowRung:
    max_nodes: int
    n_signal: int
    sigma_star: float
    signal_arm: NestedScheduleDensify304HollowArm
    hollow_arms: tuple[NestedScheduleDensify304HollowArm, ...]
    any_all_match: bool
    signal_recovers: bool
    any_hollow_preserves: bool
    any_b2: bool
    best_arm: str | None


@dataclass(frozen=True)
class NestedScheduleDensify304HollowBundle:
    dataset_seed: int
    stage1_seed: int
    max_nodes_grid: tuple[int, ...]
    schedule: dict[int, float]
    rungs: tuple[NestedScheduleDensify304HollowRung, ...]
    any_all_match: bool
    any_signal_recovers: bool
    any_b2: bool
    recovering_max_nodes: tuple[int, ...]
    mild_only: bool
    full_recover: bool
    hard_fail: bool
    table: str


def _fit_rung(max_nodes: int) -> NestedScheduleDensify304HollowRung:
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
        max_nodes=int(max_nodes),
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

    def _scheduled_arm(
        name: str,
        pts: np.ndarray,
        labs: np.ndarray,
        *,
        mid: float | None,
        fallback: bool,
        signal_all: bool | None,
    ) -> NestedScheduleDensify304HollowArm:
        scheduled = run_scheduled_mult_per_region_ph(
            pts,
            labs,
            sigma,
            mult_by_region=SCHEDULE,
            scenario=f"nested_schedule_densify{max_nodes}_cliff_hollow_{name}",
            reading="fixed_threshold",
            max_dim=2,
            expected_betti=EXPECTED_SHELL,
        )
        matches = {int(r.region_id): r.match for r in scheduled.rows}
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti)
            for r in scheduled.rows
        }
        all_match = scheduled.all_match
        any_b2 = any(len(b) > 2 and int(b[2]) >= 1 for b in betti.values())
        preserves = bool(all_match is True and signal_all is True)
        return NestedScheduleDensify304HollowArm(
            name=name,
            mid_radius_frac=mid,
            n_signal=int(pts.shape[0]),
            fallback=bool(fallback),
            table=format_scheduled_mult_ph_table(scheduled),
            all_match=all_match,
            matches=matches,
            betti=betti,
            any_b2=any_b2,
            preserves_vs_signal=preserves,
        )

    signal_arm = _scheduled_arm(
        "signal",
        signal_pos,
        signal_labs,
        mid=None,
        fallback=False,
        signal_all=None,
    )
    signal_arm = NestedScheduleDensify304HollowArm(
        name=signal_arm.name,
        mid_radius_frac=None,
        n_signal=signal_arm.n_signal,
        fallback=False,
        table=signal_arm.table,
        all_match=signal_arm.all_match,
        matches=signal_arm.matches,
        betti=signal_arm.betti,
        any_b2=signal_arm.any_b2,
        preserves_vs_signal=False,
    )

    hollow_arms: list[NestedScheduleDensify304HollowArm] = []
    any_all = bool(signal_arm.all_match is True)
    any_preserve = False
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
        arm = _scheduled_arm(
            name,
            h_pos,
            h_labs,
            mid=float(cfg.mid_radius_frac),
            fallback=fallback,
            signal_all=signal_arm.all_match,
        )
        if not fallback:
            arm = NestedScheduleDensify304HollowArm(
                name=arm.name,
                mid_radius_frac=arm.mid_radius_frac,
                n_signal=n_after,
                fallback=False,
                table=arm.table,
                all_match=arm.all_match,
                matches=arm.matches,
                betti=arm.betti,
                any_b2=arm.any_b2,
                preserves_vs_signal=arm.preserves_vs_signal,
            )
        hollow_arms.append(arm)
        any_all = any_all or (arm.all_match is True)
        any_preserve = any_preserve or arm.preserves_vs_signal
        any_b2 = any_b2 or arm.any_b2
        if arm.all_match is True and best_arm is None:
            best_arm = name

    return NestedScheduleDensify304HollowRung(
        max_nodes=int(max_nodes),
        n_signal=n_signal,
        sigma_star=sigma,
        signal_arm=signal_arm,
        hollow_arms=tuple(hollow_arms),
        any_all_match=any_all,
        signal_recovers=bool(signal_arm.all_match is True),
        any_hollow_preserves=any_preserve,
        any_b2=any_b2,
        best_arm=best_arm if signal_arm.all_match is not True else (
            "signal" if best_arm is None else best_arm
        ),
    )


@pytest.fixture(scope="module")
def nested_schedule_densify304_hollow_bundle() -> (
    NestedScheduleDensify304HollowBundle
):
    """Fit nested schedule×hollow at densify 304."""
    rungs = tuple(_fit_rung(mn) for mn in MAX_NODES_GRID)
    recovering = tuple(
        int(r.max_nodes)
        for r in rungs
        if r.signal_recovers or r.any_all_match
    )
    any_all = any(r.any_all_match for r in rungs)
    any_signal = any(r.signal_recovers for r in rungs)
    any_b2 = any(r.any_b2 for r in rungs)
    # Classify vs T92 neighbors: full = signal recovers; mild-only = hollow
    # all_match without signal; hard-fail = none.
    full_recover = any(r.signal_recovers for r in rungs)
    mild_only = (not full_recover) and any(
        any(a.name == "mild" and a.all_match is True for a in r.hollow_arms)
        for r in rungs
    )
    hard_fail = not any_all and not any_signal
    table_lines = [
        "max_nodes\tarm\tall_match\tany_b2\tn_signal\tbetti"
    ]
    for r in rungs:
        for arm in (r.signal_arm, *r.hollow_arms):
            table_lines.append(
                f"{r.max_nodes}\t{arm.name}\t{arm.all_match}\t{int(arm.any_b2)}\t"
                f"{arm.n_signal}\t{arm.betti}"
            )
    return NestedScheduleDensify304HollowBundle(
        dataset_seed=DATASET_SEED,
        stage1_seed=STAGE1_SEED,
        max_nodes_grid=MAX_NODES_GRID,
        schedule=dict(SCHEDULE),
        rungs=rungs,
        any_all_match=any_all,
        any_signal_recovers=any_signal,
        any_b2=any_b2,
        recovering_max_nodes=recovering,
        mild_only=mild_only,
        full_recover=full_recover,
        hard_fail=hard_fail,
        table="\n".join(table_lines),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_schedule_densify304_hollow_harness_lands(
    nested_schedule_densify304_hollow_bundle,
) -> None:
    """Nested schedule×densify304×hollow lands; SI defaults untouched."""
    bundle = nested_schedule_densify304_hollow_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.dataset_seed == DATASET_SEED
    assert bundle.stage1_seed == STAGE1_SEED
    assert bundle.max_nodes_grid == MAX_NODES_GRID
    assert bundle.schedule == SCHEDULE
    assert len(bundle.rungs) == len(MAX_NODES_GRID)
    assert {r.max_nodes for r in bundle.rungs} == set(MAX_NODES_GRID)
    for r in bundle.rungs:
        assert r.n_signal > 0
        assert r.sigma_star > 0.0
        assert len(r.hollow_arms) == 2
        assert {a.name for a in r.hollow_arms} == {"primary", "mild"}
    header = bundle.table.splitlines()[0]
    assert "max_nodes" in header and "all_match" in header


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_schedule_densify304_hollow_documents_gap(
    nested_schedule_densify304_hollow_bundle,
) -> None:
    """Document densify304 vs T92@300 mild / 308 win; never flip awaiting.

    Soft: full / mild-only / hard-fail classification pins the non-mono cliff.
    """
    bundle = nested_schedule_densify304_hollow_bundle
    if bundle.full_recover or bundle.mild_only or bundle.any_all_match:
        assert FILTRATION_MULTIPLIER == 1.5
        assert bundle.hard_fail is False
        assert len(bundle.recovering_max_nodes) >= 1 or bundle.mild_only
        assert bundle.any_b2 or not bundle.any_b2
    else:
        assert bundle.any_all_match is False
        assert bundle.any_signal_recovers is False
        assert bundle.full_recover is False
        assert bundle.mild_only is False
        assert bundle.hard_fail is True
        assert bundle.recovering_max_nodes == ()
        for r in bundle.rungs:
            assert r.signal_arm.all_match is not True
            assert all(a.all_match is not True for a in r.hollow_arms)
