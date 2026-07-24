"""Canonical Proteus data types (SI S2-S10).

These dataclasses define the public shapes of Proteus objects. They were
originally authored under ``tests/contracts/`` so that tests could be written
before the implementation modules existed; they now live in the package proper
(OPEN_ISSUES #38) so that production code no longer depends on the test tree.

The ``tests/contracts/*`` modules re-export the relevant types from here, so the
SI-section-annotated contract surface is preserved for test authors while the
canonical definitions live in one place. Each block below cites the owning SI
section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Node, link, scaffold, and recursion-tree state (SI S2.3, S2.4, S4.4)
# ---------------------------------------------------------------------------


@dataclass
class NodeState:
    """Per-node state maintained by Stage 1 and Stage 2 (SI S2.3)."""

    position: np.ndarray             # w_i in R^d
    residual_mean: np.ndarray        # m_i
    residual_sq: np.ndarray          # s_i
    nudge: np.ndarray                # a_i (deferred)
    principal_dir: np.ndarray        # u_i (Oja direction)
    hit_count: float = 0.0           # h_i
    variance: float = 0.0            # sigma_i^2 = tr(s_i - m_i * m_i)
    d_final: int = 1                 # smoothed local intrinsic dimension
    update_count: int = 0            # number of routed statistical updates
    # Partition-aligned shadow moments (SI S2.3.2)
    m_pos: np.ndarray = field(default_factory=lambda: np.empty(0))
    s_pos: np.ndarray = field(default_factory=lambda: np.empty(0))
    h_pos: float = 0.0
    update_count_pos: int = 0
    m_neg: np.ndarray = field(default_factory=lambda: np.empty(0))
    s_neg: np.ndarray = field(default_factory=lambda: np.empty(0))
    h_neg: float = 0.0
    update_count_neg: int = 0


@dataclass
class Link:
    """Directed link with transition counters (SI S2, S3.1)."""

    i: int
    j: int
    count_ij: float = 0.0            # C(i -> j)
    count_ji: float = 0.0            # C(j -> i)
    protected_until: int = 0
    lifted: bool = False             # True after a BMU_1->BMU_2 co-activation


@dataclass
class RegionScaffold:
    """A stabilized Stage 1 scaffold for a single region (SI S2.5, S4.4)."""

    nodes: list[NodeState]
    links: list[Link]
    tau_star: float                   # selected characteristic scale
    stabilized: bool = False
    cv_history: list[float] = field(default_factory=list)


@dataclass
class RecursionTreeNode:
    """A node in the Stage 1 recursion tree (SI S4.4)."""

    region_id: int
    level: int
    parent_id: Optional[int]
    scaffold: Optional[RegionScaffold] = None
    children: list[int] = field(default_factory=list)
    sample_indices: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Scale parameters, response traces, and selection (SI S2.4, S2.5)
# ---------------------------------------------------------------------------


@dataclass
class ScaleParameter:
    """A single candidate or selected scale (SI S2.4).

    ``tau_global`` is the operational scale primitive. ``s_control`` is a
    reporting/normalization convention only -- ``1 - exp(-tau / d_subspace)``
    -- with no operational consumer (OPEN_ISSUES #35).
    """

    tau_global: float                # operational scale primitive
    d_subspace: int
    s_control: float | None = None   # reporting only; 1 - exp(-tau / d_subspace)


@dataclass
class ScaleResponseTrace:
    """Phi_C(tau) and V_C(tau) evaluated on a geometric grid (SI S2.5)."""

    tau_values: np.ndarray           # (J,) grid of tau
    phi_values: np.ndarray           # (J,) cluster response
    support_values: np.ndarray       # (J,) support trace V_C


@dataclass
class ScaleSelection:
    """Result of characteristic-scale selection (SI S2.5.1)."""

    tau_star: float
    response_at_star: float
    bracket_indices: tuple[int, int]
    trace: ScaleResponseTrace


# ---------------------------------------------------------------------------
# AP cluster assignment and summaries (SI S2.6)
# ---------------------------------------------------------------------------


@dataclass
class APClusterAssignment:
    """Result of Affinity Propagation on the Hebbian graph (SI S2.6)."""

    node_labels: np.ndarray          # (N_nodes,) cluster label per node
    exemplar_indices: np.ndarray     # (K,) exemplar node index per cluster
    similarities: np.ndarray         # (N_edges,) smoothed-PMI similarities used


@dataclass
class ClusterSummary:
    """Per-cluster summary after AP and data routing (SI S2.6)."""

    cluster_id: int
    node_indices: np.ndarray
    sample_count: int
    centroid: np.ndarray
    scale_at_selection: float


# ---------------------------------------------------------------------------
# Simplicial complex, mass field, and face pressure (SI S4, S6, S10)
# ---------------------------------------------------------------------------


@dataclass
class Simplex:
    """A top-dimensional simplex in the flag complex (SI S4.1)."""

    vertex_ids: tuple[int, ...]
    volume: float = 0.0              # |S|_d
    mass: float = 0.0                # m_S


@dataclass
class Complex:
    """A simplicial complex for a single region (SI S4)."""

    simplices: list[Simplex]
    vertex_positions: np.ndarray     # (N_nodes, d)
    intrinsic_dim: int


@dataclass
class MassField:
    """Settled simplex mass field for a region (SI S10)."""

    masses: np.ndarray               # (M,) one per simplex
    region_id: int


@dataclass
class FacePressureField:
    """Conservative face-pressure field for a region (SI S6)."""

    empirical_tallies: np.ndarray    # hat{p}_f, (F,)
    conservative_pressures: np.ndarray  # p_f, (F,)
    region_id: int


# ---------------------------------------------------------------------------
# Evidence gate (SI S3.4, S3.5, S3.6)
# ---------------------------------------------------------------------------


class EditType(Enum):
    SPLIT = "split"
    PRUNE = "prune"
    MERGE = "merge"
    WARP = "warp"


@dataclass
class EditProposal:
    """A candidate structural edit before evidence scoring (SI S3.4)."""

    edit_type: EditType
    affected_node_ids: list[int]
    diagnostic_strength: float       # priority-queue key


@dataclass
class EvidenceRegion:
    """The localized region scored by the evidence gate (SI S3.4)."""

    core_node_ids: list[int]         # V_core
    ring_node_ids: list[int]         # neighbor ring
    transition_counts: np.ndarray    # (|V_aff|, max_J) counts n_{i->j}


@dataclass
class EvidenceVerdict:
    """Result of evidence scoring for a single edit (SI S3.4)."""

    accepted: bool
    f_dm_edit: float                 # F_DM(R; M_edit)
    f_dm_keep: float                 # F_DM(R; M_keep)
    log_bayes_factor: float          # f_dm_keep - f_dm_edit
    margin: float                    # log(tau_BF) threshold used
    proposal: EditProposal


# ---------------------------------------------------------------------------
# Torsion, edge-ratio, and warp geometry (SI S5)
# ---------------------------------------------------------------------------


@dataclass
class TorsionState:
    """Torsion diagnostic for a single simplex (SI S5.1, S5.2)."""

    simplex_id: int
    omega_S: np.ndarray              # antisymmetric 2-form
    kappa_S: float                   # ||Omega_S||_F
    R_S: float                       # kappa_S / tau*
    ladder_band: str                 # "ignore", "monitor", "geometric_fix", "warp"


@dataclass
class EdgeRatioCheck:
    """Split-placement edge-ratio fallback result (SI S5.3)."""

    simplex_id: int
    r_LS: float                      # ell_max / ell_min
    passed: bool                     # r_LS <= 5


@dataclass
class WarpAttachment:
    """A local warp attached to a patch (SI S5.5, S5.6)."""

    patch_simplex_ids: list[int]
    warp_type: str                   # "mini_nsf" or "global_glow"
    pre_R_S_median: float
    post_R_S_median: float
    held_out_ll_delta: float
    accepted: bool


# ---------------------------------------------------------------------------
# Gaussian summaries, memberships, and trajectories (SI S7)
# ---------------------------------------------------------------------------


@dataclass
class GaussianSummary:
    """Per-region Gaussian fit used for canonical membership (SI S7.1, S7.2)."""

    region_id: int
    mu_C: np.ndarray                 # cluster mode in y-space
    Sigma_C: np.ndarray              # covariance in y-space
    has_warp: bool = False


@dataclass
class Membership:
    """A single-level membership score (SI S7.2)."""

    region_id: int
    score: float                     # mu_C(x) in (0, 1]


@dataclass
class MembershipTrajectory:
    """Multiscale fuzzy membership trajectory (SI S7.4)."""

    path: list[Membership]           # root to leaf

    @property
    def scores(self) -> np.ndarray:
        return np.array([m.score for m in self.path])

    @property
    def region_ids(self) -> list[int]:
        return [m.region_id for m in self.path]

    @property
    def depth(self) -> int:
        return len(self.path)


# ---------------------------------------------------------------------------
# Junction, ledger, and boundary diagnostics (SI S8.4, S9.2, S6.3)
# ---------------------------------------------------------------------------


@dataclass
class JunctionScore:
    """Per-vertex junction score (SI S8.4)."""

    vertex_id: int
    J_i: int                         # 0..5
    frozen: bool = False


class BoundaryType(Enum):
    TRUE_MANIFOLD = "true_manifold"
    COMPUTATIONAL = "computational"
    ORIENTATION_SEAM = "orientation_seam"


@dataclass
class BoundaryClassification:
    """Per-facet boundary classification (SI S6.3)."""

    facet_id: int
    boundary_type: BoundaryType


@dataclass
class ExpressivityLedger:
    """Snapshot of the operational error ledger (SI S9.2)."""

    mass_cv: float
    junction_residual: float         # eta_s + beta_max
    torsion_q95: float
    held_out_ll_delta: float
    stat_scale: float                # Delta_stat
    e_tot: float
    saturated: bool
