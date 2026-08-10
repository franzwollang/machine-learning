"""Denser max_nodes hunt for fitted linked-tori (1,2,1) (#41 follow-on).

A4-T38 found partial ``b1=2`` on interlocking torus0 at ``n_per_torus=500``,
``max_nodes=256`` with Betti ``(1,2,0)`` (b2 missing; other torus fails).
This harness asks whether ``max_nodes∈{384,512}`` recovers full ``(1,2,1)``
on both tori under SI fine mult + fixed_threshold.

Evidence-gathering only — does **not** flip ``@awaiting`` or SI defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
from proteus.stage1.stabilization import StabilizationConfig
from tests.datasets.synthetic.linked_tori import make_linked_tori
from tests.metrics.persistent_homology import (
    FILTRATION_MULTIPLIER,
    nearest_data_labels,
    run_per_region_ph,
    sigma_star_from_tau,
)

EXPECTED_TORI: tuple[int, ...] = (1, 2, 1)
N_PER_TORUS: int = 500
MAX_NODES_LADDER: tuple[int, ...] = (256, 384, 512)


@dataclass(frozen=True)
class MaxNodesHuntRow:
    max_nodes: int
    n_signal: int
    n_per_torus: dict[int, int]
    sigma_star: float
    all_match: bool | None
    betti: dict[int, tuple[int, ...]]
    max_b1: int
    any_b1_ge_2: bool
    any_full_torus: bool


@dataclass(frozen=True)
class LinkedToriMaxNodesHuntBundle:
    n_per_torus_data: int
    rows: tuple[MaxNodesHuntRow, ...]
    any_full_recover: bool
    any_b1_ge_2: bool
    max_b1: int
    b1_ge_2_cells: tuple[tuple[int, int, tuple[int, ...]], ...]


@pytest.fixture(scope="module")
def linked_tori_max_nodes_hunt_bundle() -> LinkedToriMaxNodesHuntBundle:
    """Fit n=500 interlocking tori across denser max_nodes ladder."""
    dataset = make_linked_tori(
        n_per_torus=N_PER_TORUS,
        major_radius=2.0,
        minor_radius=0.5,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint

    rows: list[MaxNodesHuntRow] = []
    for max_nodes in MAX_NODES_LADDER:
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
            seed=77,
        )
        result = run_scale_search(
            dataset.points, dim=gt.ambient_dim, config=config,
        )
        pos = result.scaffold_at_star.node_positions()
        sigma = sigma_star_from_tau(result.tau_star)
        node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
        signal_mask = np.isin(node_labels, [0, 1])
        signal_pos = pos[signal_mask]
        signal_labs = node_labels[signal_mask]
        ph = run_per_region_ph(
            signal_pos,
            signal_labs,
            sigma,
            scenario=f"linked_tori_max_nodes_hunt_m{max_nodes}",
            include_labels=[0, 1],
            reading="fixed_threshold",
            max_dim=2,
            filtration_mult=FILTRATION_MULTIPLIER,
            expected_betti=EXPECTED_TORI,
        )
        betti = {
            int(r.region_id): tuple(int(x) for x in r.betti) for r in ph.reports
        }
        max_b1 = max((int(b[1]) for b in betti.values() if len(b) > 1), default=0)
        any_b1 = any(len(b) > 1 and int(b[1]) >= 2 for b in betti.values())
        any_full = any(b == EXPECTED_TORI for b in betti.values())
        rows.append(
            MaxNodesHuntRow(
                max_nodes=int(max_nodes),
                n_signal=int(signal_pos.shape[0]),
                n_per_torus={
                    int(lab): int(np.sum(signal_labs == lab)) for lab in (0, 1)
                },
                sigma_star=float(sigma),
                all_match=ph.all_match,
                betti=betti,
                max_b1=max_b1,
                any_b1_ge_2=any_b1,
                any_full_torus=any_full,
            )
        )

    cells: list[tuple[int, int, tuple[int, ...]]] = []
    for row in rows:
        for lab, betti in row.betti.items():
            if len(betti) > 1 and int(betti[1]) >= 2:
                cells.append((int(row.max_nodes), int(lab), tuple(betti)))

    return LinkedToriMaxNodesHuntBundle(
        n_per_torus_data=N_PER_TORUS,
        rows=tuple(rows),
        any_full_recover=any(r.all_match is True for r in rows),
        any_b1_ge_2=bool(cells),
        max_b1=max((r.max_b1 for r in rows), default=0),
        b1_ge_2_cells=tuple(cells),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_max_nodes_hunt_harness_lands(
    linked_tori_max_nodes_hunt_bundle,
) -> None:
    """max_nodes ladder lands; SI fine mult untouched."""
    bundle = linked_tori_max_nodes_hunt_bundle
    assert FILTRATION_MULTIPLIER == 1.5
    assert bundle.n_per_torus_data == N_PER_TORUS
    assert len(bundle.rows) == len(MAX_NODES_LADDER)
    assert [r.max_nodes for r in bundle.rows] == list(MAX_NODES_LADDER)
    assert all(r.n_signal > 0 for r in bundle.rows)
    assert all(r.sigma_star > 0.0 for r in bundle.rows)


@pytest.mark.scenario
@pytest.mark.synthetic
def test_linked_tori_max_nodes_hunt_documents_gap(
    linked_tori_max_nodes_hunt_bundle,
) -> None:
    """Document denser max_nodes fitted gap; never flip awaiting on partial b1.

    Soft: full ``(1,2,1)`` both tori or partial ``b1≥2`` are proposal-path
    evidence. Otherwise keep explicit ``max_b1 < 2``. Partial ≠ SI recovery.
    """
    bundle = linked_tori_max_nodes_hunt_bundle
    if bundle.any_full_recover or bundle.any_b1_ge_2:
        assert FILTRATION_MULTIPLIER == 1.5
        assert bundle.max_b1 >= 2
        if bundle.any_b1_ge_2:
            assert len(bundle.b1_ge_2_cells) >= 1
        if not bundle.any_full_recover:
            assert all(r.all_match is not True for r in bundle.rows)
    else:
        assert bundle.any_full_recover is False
        assert bundle.any_b1_ge_2 is False
        assert bundle.max_b1 < 2
        assert bundle.b1_ge_2_cells == ()
        for row in bundle.rows:
            assert row.all_match is not True
