"""Node, link, scaffold, and recursion-tree contracts (SI S2.3, S2.4, S4.4)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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
