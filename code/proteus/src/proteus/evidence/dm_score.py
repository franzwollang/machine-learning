"""Dirichlet--Multinomial evidence score for structure selection (SI S3.4, S3.5).

Stage 2 follows the principle "geometry proposes, evidence decides": a proposed
structural edit (split / prune / merge / warp) is accepted only if it improves a
closed-form Bayesian evidence score on the localized *affected region* by more
than a fixed log-Bayes-factor margin.

The score is the node-local Dirichlet--multinomial (BDeu) marginal of the
outgoing transition counts, with the categorical probabilities integrated out
analytically (SI S3.4). For a node ``i`` with ``J_i`` outgoing categorical
outcomes under candidate topology ``M``, BDeu concentration ``alpha_{0,i}`` and
observed counts ``n_{i->j}`` (total ``n_i``),

    log p_i(M) = lgamma(J_i * a0) - lgamma(J_i * a0 + n_i)
                 + sum_j [ lgamma(a0 + n_{i->j}) - lgamma(a0) ].

The affected-region score is ``F_DM(R; M) = - sum_{i in V_aff} log p_i(M)`` and a
candidate is accepted iff

    F_DM(R; M_edit) < F_DM(R; M_keep) - log(tau_BF).

Because ``m`` is integrated out analytically, the score is *closed form*: it has
no optimizer iteration count and is invariant to the order in which the routed
transition events were observed (the marginal is exchangeable). This prevents the
iteration-budget artifacts that an iterative per-candidate MAP fit would create
(SI S10.2).

The BDeu concentration is derived, not free: ``alpha_{0,i} = 1/(d_final_i + 1)``
(SI S2.7). The acceptance margin is ``log(tau_BF)``; SI S14.3 lists **``log
tau_BF``** in ``[1, 3]`` for splits (empirical). ``tau_BF`` itself is therefore in
``[e, e^3] ~ [2.7, 20]``.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from math import log

import numpy as np
from scipy.special import gammaln

from proteus.types import EditProposal, EvidenceVerdict

__all__ = [
    "NodeTransition",
    "bdeu_alpha",
    "node_log_marginal",
    "f_dm",
    "evaluate_edit",
]


def bdeu_alpha(d_final: int) -> float:
    """BDeu concentration ``alpha_{0,i} = 1/(d_final_i + 1)`` (SI S2.7).

    One equivalent pseudo-observation spread uniformly over the local branching
    factor ``J_i ~ d_final_i + 1``.
    """

    return 1.0 / (float(d_final) + 1.0)


@dataclass(frozen=True)
class NodeTransition:
    """Outgoing transition evidence at a single node under one topology (SI S3.4).

    Attributes
    ----------
    counts:
        Observed outgoing transition counts ``n_{i->j}`` for the routed
        outcomes. Outcomes with zero count may be omitted (they contribute
        nothing to the product term) but are still counted in ``j_outcomes``.
    j_outcomes:
        The number of outgoing categorical outcomes ``J_i`` permitted under this
        topology (the branching factor). Must be at least ``len(counts)``.
    alpha_0:
        BDeu concentration ``alpha_{0,i}`` (see :func:`bdeu_alpha`).
    node_id:
        Optional node identifier, used only for bookkeeping / quarantine.
    """

    counts: np.ndarray
    j_outcomes: int
    alpha_0: float
    node_id: int = -1

    def __post_init__(self) -> None:
        counts = np.asarray(self.counts, dtype=float).ravel()
        if np.any(counts < 0.0):
            raise ValueError("transition counts must be non-negative")
        if self.j_outcomes < counts.size:
            raise ValueError(
                "j_outcomes must be at least the number of observed outcomes"
            )
        if self.j_outcomes < 1:
            raise ValueError("j_outcomes must be >= 1")
        if self.alpha_0 <= 0.0:
            raise ValueError("alpha_0 must be positive")
        object.__setattr__(self, "counts", counts)


def node_log_marginal(
    counts: Sequence[float] | np.ndarray,
    j_outcomes: int,
    alpha_0: float,
) -> float:
    """Closed-form node-local Dirichlet--multinomial log-marginal ``log p_i(M)``.

    Implements the SI S3.4 marginal. The result depends only on the multiset of
    counts, the branching factor ``J_i`` and the concentration; it is exact
    (no optimizer) and exchangeable in event order.
    """

    counts_arr = np.asarray(counts, dtype=float).ravel()
    n_i = float(counts_arr.sum())
    j = int(j_outcomes)
    ja0 = j * alpha_0
    log_p = gammaln(ja0) - gammaln(ja0 + n_i)
    # Outcomes with zero count contribute lgamma(a0) - lgamma(a0) = 0, so summing
    # over only the observed counts is exact for the full J_i-way product.
    if counts_arr.size:
        log_p += float(
            np.sum(gammaln(alpha_0 + counts_arr) - gammaln(alpha_0))
        )
    return float(log_p)


def f_dm(
    region: Iterable[NodeTransition],
    *,
    quarantined: set[int] | None = None,
) -> float:
    """Affected-region evidence score ``F_DM(R; M) = -sum_i log p_i(M)`` (SI S3.4).

    Nodes in ``quarantined`` (router-ill-conditioned stars, SI S10.4) contribute
    no likelihood term and are skipped in both the keep and edit baselines by the
    caller (:func:`evaluate_edit`).
    """

    quarantined = quarantined or set()
    total = 0.0
    for node in region:
        if node.node_id in quarantined:
            continue
        total -= node_log_marginal(node.counts, node.j_outcomes, node.alpha_0)
    return float(total)


def evaluate_edit(
    keep_region: Sequence[NodeTransition],
    edit_region: Sequence[NodeTransition],
    proposal: EditProposal,
    *,
    tau_bf: float,
    quarantined: set[int] | None = None,
) -> EvidenceVerdict:
    """Score a candidate edit against the keep baseline (SI S3.4).

    The edit is accepted iff
    ``F_DM(R; M_edit) < F_DM(R; M_keep) - log(tau_BF)``, equivalently the
    log-Bayes-factor ``F_DM_keep - F_DM_edit`` exceeds ``log(tau_BF)``.

    ``quarantined`` node ids (SI S10.4 ill-conditioned stars) are excluded from
    *both* F_DM baselines so their likelihood terms cannot create a spurious
    evidence improvement.
    """

    if tau_bf < 1.0:
        raise ValueError("tau_bf must be >= 1 (a non-negative log-margin)")
    margin = float(log(tau_bf))
    f_keep = f_dm(keep_region, quarantined=quarantined)
    f_edit = f_dm(edit_region, quarantined=quarantined)
    log_bf = f_keep - f_edit
    return EvidenceVerdict(
        accepted=bool(log_bf > margin),
        f_dm_edit=f_edit,
        f_dm_keep=f_keep,
        log_bayes_factor=log_bf,
        margin=margin,
        proposal=proposal,
    )
