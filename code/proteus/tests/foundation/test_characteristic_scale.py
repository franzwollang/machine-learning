"""Unit tests for the compensated node-count knee selector (SI S2.5, S2.5.1).

These exercise the selector logic on controlled traces so the behavior is
deterministic and independent of the scaffold's estimation noise.  The
integration behavior on real scaffolds (where the compensated trace does not yet
plateau cleanly --- see OPEN_ISSUES #28) is covered by the controller tests.
"""

from __future__ import annotations

import numpy as np

from proteus.stage1.characteristic_scale import (
    CharacteristicScaleConfig,
    compensated_node_count,
    select_characteristic_scale_by_node_count,
)


def test_compensated_node_count_recovers_constant_on_ideal_scaling() -> None:
    # Ideal resolved regime: N(tau) = V / tau^{d/2} makes C(tau) constant.
    d = 2.0
    tau = np.array([1.0, 0.5, 0.25, 0.125], dtype=float)
    V = 40.0
    N = V / np.power(tau, d / 2.0)
    C = compensated_node_count(N, tau, np.full(len(tau), d))
    assert np.allclose(C, V)


def test_knee_at_coarse_edge_of_plateau() -> None:
    # Descending (coarse-to-fine) grid.  N is floored while the support is
    # under-resolved (indices 0,1), then tracks V/tau^{d/2} (indices 1..4) so
    # the compensated count is constant from index 1 onward.
    d = 2.0
    tau = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=float)
    V = 32.0
    ideal = V / np.power(tau, d / 2.0)
    N = ideal.copy()
    N[0] = 8.0  # coarse point under-resolved: fewer nodes than the power law
    stabilized = [True] * len(tau)
    result = select_characteristic_scale_by_node_count(
        N, tau, np.full(len(tau), d), stabilized,
    )
    # Plateau begins at index 1; its coarse edge is the characteristic scale.
    assert result.knee_index == 1
    assert result.knee_tau == tau[1]


def test_no_plateau_returns_none() -> None:
    # A strictly geometric decay in C (no flat region) has no knee.
    d = 1.0
    tau = np.array([1.0, 0.5, 0.25, 0.125, 0.0625], dtype=float)
    N = np.array([4.0, 4.0, 4.0, 4.0, 4.0], dtype=float)  # capped node count
    C = compensated_node_count(N, tau, np.full(len(tau), d))
    # C halves-ish every step -> relative change well above the tolerance.
    assert np.all(np.abs(np.diff(C)) > 0)
    result = select_characteristic_scale_by_node_count(
        N, tau, np.full(len(tau), d), [True] * len(tau),
    )
    assert result.knee_index is None
    assert result.knee_tau is None


def test_unstable_grid_points_are_skipped() -> None:
    d = 2.0
    tau = np.array([1.0, 0.5, 0.25, 0.125], dtype=float)
    V = 20.0
    N = V / np.power(tau, d / 2.0)
    # The coarsest point did not stabilize; the plateau's usable coarse edge
    # is the first stabilized index that begins a flat run.
    stabilized = [False, True, True, True]
    result = select_characteristic_scale_by_node_count(
        N, tau, np.full(len(tau), d), stabilized,
    )
    assert result.knee_index == 1


def test_min_plateau_requires_longer_flat_run() -> None:
    d = 2.0
    tau = np.array([1.0, 0.5, 0.25, 0.125], dtype=float)
    # Flat only between indices 2 and 3; a single flat pair.
    N = np.array([5.0, 13.0, 40.0, 40.0 * 2.0 ** (d / 2.0)], dtype=float)
    cfg = CharacteristicScaleConfig(plateau_tol=0.05, min_plateau=3)
    result = select_characteristic_scale_by_node_count(
        N, tau, np.full(len(tau), d), [True] * len(tau), cfg,
    )
    # No run of 3 flat points -> no knee.
    assert result.knee_index is None
