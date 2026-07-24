"""Star-matrix identifiability / conditioning check (SI S10.4).

The evidence gate (S3.4) may only claim an ``F_DM`` improvement where the local
transition router actually identifies the mass field. Identifiability is checked
locally by *star matrices*: for node ``i`` the star matrix ``K_i`` maps the
masses of the simplices in ``Star(i)`` to the outgoing transition probabilities
from ``i``. Up to normalization ``K_i`` is the Jacobian of ``q(.|i; m)`` with
respect to ``m^{(i)}`` (SI S10.4), and the local Fisher information is
``I_i(m) = n_i K_i^T diag(1/q) K_i``; the local rank condition is exactly
nonsingularity of ``I_i`` on the tangent of the simplex, i.e. that ``K_i`` has
full rank modulo the one-dimensional global scaling direction.

Operationally, an outgoing transition ``i -> j`` is supported by exactly the
simplices in ``Star(i)`` that contain the edge ``(i, j)``, so the natural
(unnormalized) star matrix is the *edge--simplex incidence matrix* of the star:
rows are the outgoing edges from ``i``, columns are the simplices containing
``i``, with a unit entry where the edge lies in the simplex. A star is
router-ill-conditioned when this map is near-degenerate --- e.g. several
simplices routing through an identical set of edges --- so their masses cannot be
told apart from transition counts.

The runtime test uses the conditioning ratio ``rho_i = sigma_min(K_i) /
sigma_max(K_i)`` with default flag threshold ``rho_min = 1e-4`` (conservative
``1e-3`` for noise-sensitive runs); stars below ``rho_min`` are quarantined and
contribute no likelihood term to ``F_DM`` (SI S10.4 dynamic preservation rule).

S10.4 also requires each star to have "full rank modulo the one-dimensional
scaling direction". Because the router ``q(.|i;m)`` is normalized (homogeneous of
degree zero in ``m``), a star with more incident simplices than outgoing outcomes
is under-determined: there are more simplex masses to identify than independent
transition equations (``n_outcomes - 1`` free probabilities vs ``n_simplices - 1``
free masses modulo scaling). Such stars are flagged non-evidence-bearing directly,
in addition to the ``rho_min`` conditioning ratio.

The exact runtime realization of ``K_i`` (the normalized-router Jacobian at the
canonical ``kappa``) is under-specified by S10.4; the edge--simplex incidence
matrix plus the ``n_outcomes >= n_simplices`` rank guard is the operational
first-implementation proxy (OPEN_ISSUES #42).
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
    """Edge--simplex incidence matrix ``K_i`` for the star of ``node_id`` (S10.4).

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

    Requires both:

    * *Full rank modulo scaling*: at least as many outgoing outcomes as incident
      simplices (``n_outcomes >= n_simplices``); otherwise the simplex masses are
      under-determined by the transition counts. A single-simplex star is
      trivially identifiable for its lone mass.
    * *Conditioning*: ``sigma_min(K)/sigma_max(K) >= rho_min``.
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
