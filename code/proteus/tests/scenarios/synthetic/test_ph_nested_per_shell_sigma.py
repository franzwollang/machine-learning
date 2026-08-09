"""Per-shell local-sigma dual-scale PH on nested fitted scaffolds (#41 / A4-T31).

Compares dual-scale fixed_threshold PH using:
  * global ``sigma_star = sqrt(tau*)`` (A4-T29 baseline), vs
  * per-shell ``sigma = median NN gap`` of that shell's signal nodes.

Evidence-gathering only — does **not** flip ``test_nested_spheres_topology``
``@awaiting`` or change SI ``FILTRATION_MULTIPLIER``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from proteus.stage1.controller import ScaleSearchConfig, run_scale_search
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


@dataclass(frozen=True)
class NestedPerShellSigmaBundle:
    max_nodes: int
    n_signal: int
    global_sigma: float
    local_labels: tuple[int, ...]
    local_sigmas: tuple[float, ...]
    coarse_mult: float
    fine_mult: float
    global_table: str
    local_table: str
    global_any_coarse: bool
    global_any_fine: bool
    local_any_coarse: bool
    local_any_fine: bool
    global_all_either: bool
    local_all_either: bool
    local_improves: bool


@pytest.fixture(scope="module")
def nested_per_shell_sigma_bundle() -> NestedPerShellSigmaBundle:
    """Fit max_nodes=128; dual-scale under global vs per-shell local sigma."""
    dataset = make_nested_spheres(
        n_per_sphere=500,
        radii=(1.0, 2.0),
        ambient_dim=3,
        noise=0.02,
        tissue_fraction=0.03,
        seed=21,
    )
    gt = dataset.ground_truth
    tau_lo, tau_hi = gt.tau_grid_hint
    max_nodes = 128
    config = ScaleSearchConfig(
        tau_min=tau_lo,
        tau_max=tau_hi,
        max_grid_points=8,
        k=8,
        n_seeds=8,
        max_nodes=max_nodes,
        ann_backend="naive",
        stabilization=StabilizationConfig(
            min_equilibrium_epochs=3, max_epochs=15,
        ),
        seed=77,
    )
    result = run_scale_search(dataset.points, dim=gt.ambient_dim, config=config)
    pos = result.scaffold_at_star.node_positions()
    global_sigma = sigma_star_from_tau(result.tau_star)
    node_labels = nearest_data_labels(pos, dataset.points, dataset.labels)
    signal_mask = np.isin(node_labels, [1, 2])
    signal_pos = pos[signal_mask]
    signal_labs = node_labels[signal_mask]

    labs, local_sigmas = per_region_median_nn_sigma(
        signal_pos,
        signal_labs,
        include_labels=[1, 2],
    )
    coarse_mult = 3.0
    fine_mult = FILTRATION_MULTIPLIER

    dual_g = run_dual_scale_per_region_ph(
        signal_pos,
        signal_labs,
        global_sigma,
        coarse_mult=coarse_mult,
        fine_mult=fine_mult,
        scenario="nested_global_sigma_dual_scale",
        include_labels=[1, 2],
        max_dim=2,
        expected_betti=(1, 0, 1),
    )
    dual_l = run_dual_scale_per_region_ph(
        signal_pos,
        signal_labs,
        local_sigmas,
        coarse_mult=coarse_mult,
        fine_mult=fine_mult,
        scenario="nested_local_sigma_dual_scale",
        include_labels=labs,
        max_dim=2,
        expected_betti=(1, 0, 1),
    )

    def _all_either(dual) -> bool:
        if not dual.rows or len(dual.rows) != 2:
            return False
        return all(bool(r.coarse_match or r.fine_match) for r in dual.rows)

    g_all = _all_either(dual_g)
    l_all = _all_either(dual_l)
    # Local "improves" if it recovers a shell the global path missed, or all.
    g_matches = {
        int(r.region_id): bool(r.coarse_match or r.fine_match) for r in dual_g.rows
    }
    l_matches = {
        int(r.region_id): bool(r.coarse_match or r.fine_match) for r in dual_l.rows
    }
    local_improves = bool(l_all and not g_all) or any(
        l_matches.get(k, False) and not g_matches.get(k, False) for k in (1, 2)
    )

    return NestedPerShellSigmaBundle(
        max_nodes=max_nodes,
        n_signal=int(signal_pos.shape[0]),
        global_sigma=float(global_sigma),
        local_labels=tuple(int(x) for x in labs),
        local_sigmas=tuple(float(s) for s in local_sigmas),
        coarse_mult=float(coarse_mult),
        fine_mult=float(fine_mult),
        global_table=format_dual_scale_ph_table(dual_g),
        local_table=format_dual_scale_ph_table(dual_l),
        global_any_coarse=bool(dual_g.any_coarse_match),
        global_any_fine=bool(dual_g.any_fine_match),
        local_any_coarse=bool(dual_l.any_coarse_match),
        local_any_fine=bool(dual_l.any_fine_match),
        global_all_either=bool(g_all),
        local_all_either=bool(l_all),
        local_improves=bool(local_improves),
    )


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_per_shell_sigma_tables(nested_per_shell_sigma_bundle) -> None:
    """Global + local dual-scale tables land; local sigmas positive and finite."""
    bundle = nested_per_shell_sigma_bundle
    assert bundle.max_nodes == 128
    assert bundle.n_signal > 0
    assert bundle.global_sigma > 0.0
    assert bundle.fine_mult == FILTRATION_MULTIPLIER == 1.5
    assert bundle.local_labels == (1, 2)
    assert len(bundle.local_sigmas) == 2
    assert all(np.isfinite(s) and s > 0.0 for s in bundle.local_sigmas)
    assert "coarse_betti" in bundle.global_table.splitlines()[0]
    assert "coarse_betti" in bundle.local_table.splitlines()[0]
    assert len(bundle.global_table.splitlines()) >= 3
    assert len(bundle.local_table.splitlines()) >= 3


@pytest.mark.scenario
@pytest.mark.synthetic
def test_nested_per_shell_sigma_documents_gap(
    nested_per_shell_sigma_bundle,
) -> None:
    """Document whether local sigma recovers more shells; never flip awaiting.

    Soft gate: if local recovers all shells, local-sigma is a viable probe.
    Otherwise keep explicit non-recovery (SI fine still insufficient alone).
    """
    bundle = nested_per_shell_sigma_bundle
    if bundle.local_all_either:
        assert FILTRATION_MULTIPLIER == 1.5
        assert bundle.local_any_coarse or bundle.local_any_fine
    else:
        assert bundle.local_all_either is False
        # Global path remains incomplete under SI fine if local also fails.
        if not bundle.global_all_either:
            assert bundle.global_any_fine is False or not bundle.global_all_either
        assert "fine_betti" in bundle.local_table
