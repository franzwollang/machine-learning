"""Variance-cap split proposals for fixed-tau Stage 1 scaffolds.

Splits always commit to the Oja direction u_i (SI S2.3.2).  Children
inherit partition-aligned shadow moments rather than zeroed or copied
aggregates, recovering identifiability while remaining streaming-bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from proteus.moments import variance_from_moments
from proteus.nodes import make_node


@dataclass(frozen=True)
class SplitProposal:
    """A local variance-cap split candidate."""

    node_id: int
    axis_unit_vector: np.ndarray
    child_offset_norm: float


def propose_splits(scaffold: Any) -> list[SplitProposal]:
    """Return split proposals for over-cap nodes with enough warmup."""

    proposals: list[SplitProposal] = []
    tau_local = np.asarray(scaffold.tau_local, dtype=float)
    for node_id, node in enumerate(scaffold.nodes):
        if getattr(node, "update_count", 0) < scaffold.prune_after:
            continue
        if node.variance <= tau_local[node_id]:
            continue
        axis = _unit(node.principal_dir)
        proposals.append(
            SplitProposal(
                node_id=node_id,
                axis_unit_vector=axis,
                child_offset_norm=0.5 * float(np.sqrt(tau_local[node_id])),
            )
        )
    return proposals


def apply_split(scaffold: Any, proposal: SplitProposal) -> bool:
    """Apply a split along the Oja direction with shadow-moment inheritance.

    State classification (SI S2.3.1 refinement-admissibility):

    * Flow-conserved: hit counts distributed by shadow masses h_pos / h_neg.
    * Frame-relative: children inherit their shadow pair as initial aggregate
      moments; their own shadow pairs, nudge, and update_count reset to zero.
    * Structural: positions placed symmetrically along +/- u_i; principal_dir
      and d_final inherited.
    """

    parent = scaffold.nodes[proposal.node_id]
    active_hits = [
        node.hit_count
        for node in scaffold.nodes
        if getattr(node, "update_count", 0) >= scaffold.prune_after
    ]
    mean_active_hits = float(np.mean(active_hits)) if active_hits else 0.0
    if mean_active_hits <= 0.0:
        return False
    h_prune = scaffold.prune_hit_fraction * mean_active_hits
    if parent.hit_count < 2.0 * h_prune:
        return False

    offset = float(proposal.child_offset_norm) * proposal.axis_unit_vector

    # Positive-side child: position at w + offset, shadows referenced to w
    # Steiner shift d = +offset  =>  m' = m - d,  s' = s - 2d*m + d^2
    child = make_node(
        parent.position + offset,
        scaffold.dim,
        principal_dir=parent.principal_dir,
        d_final=parent.d_final,
    )
    child.residual_mean, child.residual_sq = _steiner_shift(
        parent.m_pos, parent.s_pos, offset,
    )
    child.variance = variance_from_moments(child.residual_mean, child.residual_sq)
    child.hit_count = parent.h_pos

    # Parent becomes negative-side child: position at w - offset
    # Steiner shift d = -offset  =>  m' = m + offset,  s' = s + 2*offset*m + offset^2
    parent.position = parent.position - offset
    neg_m, neg_s = _steiner_shift(parent.m_neg, parent.s_neg, -offset)
    parent.residual_mean = neg_m
    parent.residual_sq = neg_s
    parent.variance = variance_from_moments(neg_m, neg_s)
    parent.hit_count = parent.h_neg

    _reset_shadow_and_nudge(parent)
    _reset_shadow_and_nudge(child)

    scaffold.nodes.append(child)
    scaffold.tau_local = np.append(
        scaffold.tau_local, scaffold.tau_local[proposal.node_id]
    )
    scaffold.ann.add(child.position)
    scaffold.ann.update(proposal.node_id, parent.position)
    _bisect_incident_links(scaffold, proposal.node_id)
    return True


def _reset_shadow_and_nudge(node: Any) -> None:
    """Reset shadow pairs, nudge, and update_count after a split."""

    node.m_pos = np.zeros_like(node.residual_mean)
    node.s_pos = np.zeros_like(node.residual_mean)
    node.h_pos = 0.0
    node.update_count_pos = 0
    node.m_neg = np.zeros_like(node.residual_mean)
    node.s_neg = np.zeros_like(node.residual_mean)
    node.h_neg = 0.0
    node.update_count_neg = 0
    node.nudge = np.zeros_like(node.nudge)
    node.update_count = 0


def _steiner_shift(
    m: np.ndarray, s: np.ndarray, d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a parallel-axis (Steiner) shift to EWMA moments.

    When the reference point moves by ``d``, the residual ``e' = e - d``
    transforms the running moments as ``m' = m - d``, ``s' = s - 2d*m + d^2``.
    Variance ``tr(s' - m'*m') = tr(s - m*m)`` is shift-invariant.
    """

    d_arr = np.asarray(d, dtype=float)
    m_new = m - d_arr
    s_new = s - 2.0 * d_arr * m + d_arr * d_arr
    return m_new, np.maximum(s_new, 0.0)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        fallback = np.zeros_like(vector)
        fallback[0] = 1.0
        return fallback
    return vector / norm


def _bisect_incident_links(scaffold: Any, node_id: int) -> None:
    """Halve counters on all links incident to ``node_id``."""

    for link in scaffold.links.as_list():
        if link.i == node_id or link.j == node_id:
            link.count_ij *= 0.5
            link.count_ji *= 0.5
