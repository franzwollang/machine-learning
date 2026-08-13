"""Star-matrix identifiability / conditioning check (SI S10.4).

The evidence gate (S3.4) may only claim an ``F_DM`` improvement where the local
transition router actually identifies the mass field. Identifiability is checked
locally by *star matrices*: for node ``i`` the star matrix maps the masses of
the simplices in ``Star(i)`` to the outgoing transition probabilities from
``i``.

**Theory (Fisher / Jacobian).** Up to normalization the theoretical ``K_i`` is
the Jacobian of ``q(.|i; m)`` with respect to ``m^{(i)}`` at the canonical
``kappa`` (SI S10.1 / S10.4). The local Fisher information is
``I_i(m) = n_i K_i^T diag(1/q) K_i``; local rank condition (A1) is nonsingularity
of ``I_i`` on the tangent of the simplex (full rank modulo the one-dimensional
global scaling direction), together with dual-graph connectivity (A2).

**Operational runtime form (blessed in SI S10.4).** The dry-run conditioning
check uses the topology-only *edge--simplex incidence* proxy
``K_i^{inc}[j,S] = 1[{i,j} subset S]`` (rows = outgoing neighbors, columns =
incident simplices) rather than evaluating the mass-/kappa-dependent Jacobian.
A star is router-ill-conditioned when this map is near-degenerate --- e.g.
several simplices routing through an identical set of edges --- so their masses
cannot be told apart from transition counts. The combinatorial stand-in for
full rank modulo scaling is ``n_outcomes >= n_simplices`` (single-simplex stars
are trivially identifiable). Both the incidence proxy and the count guard are
labeled *operational* in S10.4; they may later be upgraded to the weighted
Jacobian without changing the gate contract.

The runtime test uses ``rho_i = sigma_min(K_i^{inc}) / sigma_max(K_i^{inc})``
with default ``rho_min = 1e-4`` (conservative ``1e-3``; SI S14.3). Stars below
``rho_min`` or failing the count guard are quarantined and contribute no
likelihood term to ``F_DM`` (S10.4 dynamic preservation rule).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "RHO_MIN_DEFAULT",
    "RHO_MIN_CONSERVATIVE",
    "star_incidence_matrix",
    "condition_ratio",
    "is_evidence_bearing",
    "quarantined_nodes",
]

# SI S10.4: runtime conditioning flag thresholds (operational).
RHO_MIN_DEFAULT: float = 1e-4
RHO_MIN_CONSERVATIVE: float = 1e-3


def star_incidence_matrix(
    out_edges: Sequence[int],
    star_simplices: Sequence[Sequence[int]],
    node_id: int,
) -> np.ndarray:
    """Operational edge--simplex incidence ``K_i^{inc}`` for ``node_id`` (S10.4).

    This is the SI-blessed first-implementation runtime matrix used by the
    dry-run ``rho*`` test --- not the theoretical router Jacobian.

    Parameters
    ----------
    out_edges:
        Neighbour node ids ``j`` reachable from ``node_id`` (the outgoing
        categorical outcomes).
    star_simplices:
        The simplices (vertex-id sequences) incident to ``node_id``.
    node_id:
        The centre node ``i``.

    Returns
    -------
    ``K`` of shape ``(len(out_edges), len(star_simplices))`` with ``K[r, c] = 1``
    when edge ``(node_id, out_edges[r])`` is contained in ``star_simplices[c]``.
    """

    edge_index = {int(j): r for r, j in enumerate(out_edges)}
    K = np.zeros((len(out_edges), len(star_simplices)), dtype=float)
    for c, simplex in enumerate(star_simplices):
        verts = set(int(v) for v in simplex)
        if int(node_id) not in verts:
            continue
        for j in verts:
            if j == int(node_id):
                continue
            r = edge_index.get(j)
            if r is not None:
                K[r, c] = 1.0
    return K


def condition_ratio(K: np.ndarray) -> float:
    """Return the S10.4 runtime conditioning ratio ``sigma_min(K)/sigma_max(K)``.

    This is the literal runtime ratio of the S10.4 conditioning paragraph over the
    ``min(n_outcomes, n_simplices)`` singular values of ``K``. Empty or all-zero
    matrices are maximally ill-conditioned (``rho = 0``). Full-rank-modulo-scaling
    (the ``n_outcomes >= n_simplices`` requirement) is enforced separately in
    :func:`is_evidence_bearing`, since a wide ``K`` can have a well-conditioned
    ratio while still leaving some simplex masses unidentifiable.
    """

    K = np.asarray(K, dtype=float)
    if K.size == 0:
        return 0.0
    sv = np.linalg.svd(K, compute_uv=False)
    if sv.size == 0:
        return 0.0
    sigma_max = float(sv[0])
    sigma_min = float(sv[-1])
    if sigma_max <= 0.0:
        return 0.0
    return sigma_min / sigma_max


def is_evidence_bearing(K: np.ndarray, rho_min: float = RHO_MIN_DEFAULT) -> bool:
    """True iff the star can carry likelihood evidence (SI S10.4).

    Requires both operational checks from the S10.4 runtime paragraph:

    * *Count guard* (combinatorial stand-in for full rank modulo scaling): at
      least as many outgoing outcomes as incident simplices
      (``n_outcomes >= n_simplices``); otherwise the simplex masses are
      under-determined by the transition counts. A single-simplex star is
      trivially identifiable for its lone mass.
    * *Conditioning*: ``sigma_min(K)/sigma_max(K) >= rho_min`` on the incidence
      proxy ``K = K_i^{inc}``.
    """

    K = np.asarray(K, dtype=float)
    if K.size == 0:
        return False
    n_outcomes, n_simplices = K.shape
    if n_simplices == 0 or n_outcomes == 0:
        return False
    if not np.any(K != 0.0):
        return False
    if n_simplices == 1:
        return True
    if n_simplices > n_outcomes:
        return False
    return condition_ratio(K) >= rho_min


def quarantined_nodes(
    stars: Mapping[int, np.ndarray],
    rho_min: float = RHO_MIN_DEFAULT,
) -> set[int]:
    """Return the ids of stars that must be quarantined from ``F_DM`` (S10.4)."""

    return {
        node_id
        for node_id, K in stars.items()
        if not is_evidence_bearing(K, rho_min)
    }
