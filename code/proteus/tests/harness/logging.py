"""Structured logging schema for the SI S14.4 logging checklist.

Provides a ``ProteusRunLog`` that scenario tests populate during a run
and assert against afterwards.  The schema is implementation-independent:
it describes *what* must be logged, not how the implementation produces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pytest


@dataclass
class ScaleDiscoveryLog:
    """Per-region scale-discovery entries (SI S14.4 bullet 1)."""
    region_id: int
    phi_trace: Optional[np.ndarray] = None
    v_trace: Optional[np.ndarray] = None
    grid_points_evaluated: int = 0
    peak_brackets: int = 0
    optimizer_refinements: int = 0
    tau_star: Optional[float] = None
    stabilization_failures: int = 0


@dataclass
class ScaffoldStatsLog:
    """Per-region scaffold statistics (SI S14.4 bullet 2)."""
    region_id: int
    hit_cv: float = float("inf")
    variance_cap_violations: int = 0
    deferred_nudge_rate: float = 0.0
    ann_recall: Optional[float] = None
    intrinsic_dim_estimates: Optional[np.ndarray] = None
    ap_cluster_sizes: Optional[np.ndarray] = None


@dataclass
class EvidenceGateLog:
    """Per-gate-firing entry (SI S14.4 bullet 3)."""
    region_id: int
    proposals_queued: int = 0
    proposals_scored: int = 0
    proposals_accepted: int = 0
    proposals_rejected: int = 0
    proposals_stale: int = 0
    hysteresis_blocks: int = 0
    edit_budget_used: int = 0
    edit_budget_max: int = 0


@dataclass
class TorsionWarpLog:
    """Per-region torsion and warp statistics (SI S14.4 bullet 4)."""
    region_id: int
    r_s_distribution: Optional[np.ndarray] = None
    ladder_band_fractions: Optional[dict[str, float]] = None
    edge_ratio_fallback_count: int = 0
    p_kappa: float = 0.0
    mini_nsf_patch_sizes: Optional[list[int]] = None
    warp_train_steps: Optional[list[int]] = None
    held_out_ll_deltas: Optional[list[float]] = None
    rollback_count: int = 0


@dataclass
class DualFlowDensityLog:
    """Per-region dual-flow and density statistics (SI S14.4 bullet 5)."""
    region_id: int
    r_data: float = float("inf")
    r_cons: float = float("inf")
    epsilon_mass: float = float("inf")
    epsilon_flux: float = float("inf")
    boundary_classifications: Optional[dict[str, int]] = None
    message_convergence_iters: int = 0
    volume_floor_uses: int = 0


@dataclass
class MembershipStoppingLog:
    """Per-region membership and stopping diagnostics (SI S14.4 bullet 6)."""
    region_id: int
    mu_residuals: Optional[np.ndarray] = None
    junction_scores: Optional[np.ndarray] = None
    freeze_events: int = 0
    lift_events: int = 0
    mass_cv: float = float("inf")
    eta_s_plus_beta_max: float = float("inf")
    held_out_ll_improvement: float = 0.0
    stopping_reason: str = ""


@dataclass
class ProteusRunLog:
    """Complete structured log for one Proteus training run.

    Scenario tests populate this incrementally during a run and then
    assert against its contents.
    """
    scale_discovery: list[ScaleDiscoveryLog] = field(default_factory=list)
    scaffold_stats: list[ScaffoldStatsLog] = field(default_factory=list)
    evidence_gate: list[EvidenceGateLog] = field(default_factory=list)
    torsion_warp: list[TorsionWarpLog] = field(default_factory=list)
    dual_flow_density: list[DualFlowDensityLog] = field(default_factory=list)
    membership_stopping: list[MembershipStoppingLog] = field(default_factory=list)

    def assert_edit_budget(self, max_per_epoch: int) -> None:
        """Assert no region exceeded the per-epoch edit budget."""
        for entry in self.evidence_gate:
            assert entry.edit_budget_used <= max_per_epoch, (
                f"Region {entry.region_id}: {entry.edit_budget_used} edits "
                f"exceeds budget {max_per_epoch}"
            )

    def assert_mass_conservation(self, threshold: float = 1e-6) -> None:
        """Assert epsilon_mass is within tolerance after every dual-flow solve."""
        for entry in self.dual_flow_density:
            assert entry.epsilon_mass <= threshold, (
                f"Region {entry.region_id}: epsilon_mass={entry.epsilon_mass} "
                f"exceeds threshold {threshold}"
            )


@pytest.fixture
def proteus_run_log() -> ProteusRunLog:
    """Pytest fixture providing a fresh run log for scenario tests."""
    return ProteusRunLog()
