"""Evidence-gate cadence, hysteresis, and edit budgets (SI S3.6).

The DM score of S3.4 decides a *single* candidate edit. This module owns the
runtime bookkeeping around it: a per-region proposal queue keyed by diagnostic
strength, an equilibration window before scoring, hysteresis against immediate
reversals, and a per-epoch positive-jump edit budget. It also composes the DM
score (:mod:`proteus.evidence.dm_score`) with the star-matrix conditioning check
(:mod:`proteus.evidence.star_matrix`) so that ill-conditioned stars are
quarantined from the likelihood terms (S10.4 dynamic preservation rule).

Window and budgets (SI S3.6):

    W          = max(N_nodes_C, 4 k |Q_C|)          equilibration window
    T_hyst     = 2 W                                 reversal lockout
    N_edit^+   = N_nodes_C / log N_nodes_C           accepted prune/merge budget

Cadence config values (``tau_bf``, ``k``, ``rho_min``) are operational defaults
backstopped by the gate itself (SI S14.3): ``tau_bf`` in ``[1, 3]`` for splits,
``rho_min = 1e-4``.
"""
from __future__ import annotations

import heapq
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import log

import numpy as np

from proteus.types import EditProposal, EditType, EvidenceVerdict
from proteus.evidence.dm_score import NodeTransition, evaluate_edit
from proteus.evidence.star_matrix import RHO_MIN_DEFAULT, quarantined_nodes

__all__ = [
    "GateConfig",
    "gate_window",
    "hysteresis_window",
    "edit_budget",
    "score_edit",
    "EvidenceGate",
]


@dataclass(frozen=True)
class GateConfig:
    """Evidence-gate cadence parameters (SI S3.6, S14.3).

    Attributes
    ----------
    tau_bf:
        Bayes-factor margin base; the acceptance margin is ``log(tau_bf)``
        (SI S3.4, S14.3 empirical default, ``[1, 3]`` for splits).
    k:
        Neighbour count used in the equilibration-window formula (SI S3.6).
    rho_min:
        Star-matrix conditioning flag threshold (SI S10.4).
    """

    tau_bf: float = 3.0
    k: int = 10
    rho_min: float = RHO_MIN_DEFAULT


def gate_window(n_nodes: int, queue_len: int, k: int) -> int:
    """Equilibration window ``W = max(N_nodes, 4 k |Q_C|)`` (SI S3.6, eq. gate-window)."""

    return int(max(int(n_nodes), 4 * int(k) * int(queue_len)))


def hysteresis_window(window: int) -> int:
    """Reversal lockout ``T_hyst = 2 W`` (SI S3.6)."""

    return int(2 * int(window))


def edit_budget(n_nodes: int) -> int:
    """Per-epoch accepted prune/merge budget ``N_nodes / log N_nodes`` (SI S3.6).

    Below ``e`` nodes the log is not usable; the budget is at least one edit so a
    tiny region can still be repaired.
    """

    n = int(n_nodes)
    if n <= 2:
        return 1
    return max(1, int(n / log(n)))


def score_edit(
    keep_region: Sequence[NodeTransition],
    edit_region: Sequence[NodeTransition],
    proposal: EditProposal,
    *,
    config: GateConfig | None = None,
    edit_stars: Mapping[int, np.ndarray] | None = None,
    keep_stars: Mapping[int, np.ndarray] | None = None,
) -> EvidenceVerdict:
    """Score one edit with S10.4 conditioning quarantine applied (SI S3.4, S10.4).

    Any node whose pre- or post-edit star is router-ill-conditioned is quarantined
    from *both* F_DM baselines, so it cannot manufacture an evidence improvement.
    """

    config = config or GateConfig()
    quarantined: set[int] = set()
    if edit_stars is not None:
        quarantined |= quarantined_nodes(edit_stars, config.rho_min)
    if keep_stars is not None:
        quarantined |= quarantined_nodes(keep_stars, config.rho_min)
    return evaluate_edit(
        keep_region,
        edit_region,
        proposal,
        tau_bf=config.tau_bf,
        quarantined=quarantined,
    )


@dataclass(order=True)
class _QueueItem:
    priority: float
    seq: int = field(compare=True)
    proposal: EditProposal = field(compare=False)


class EvidenceGate:
    """Stateful per-region evidence gate (SI S3.6).

    Owns the proposal queue, the sample clock, hysteresis lockouts, and the
    per-epoch edit budget for one region ``C`` of ``n_nodes`` nodes. Scoring is
    delegated to :func:`score_edit`.
    """

    def __init__(self, n_nodes: int, config: GateConfig | None = None) -> None:
        self.n_nodes = int(n_nodes)
        self.config = config or GateConfig()
        self._heap: list[_QueueItem] = []
        self._seq = 0
        self.clock = 0
        self.accepted_this_epoch = 0
        self._locked_until: dict[int, int] = {}

    @property
    def queue_len(self) -> int:
        return len(self._heap)

    def window(self) -> int:
        """Current equilibration window ``W`` given the queue length (SI S3.6)."""

        return gate_window(self.n_nodes, self.queue_len, self.config.k)

    def budget(self) -> int:
        """Remaining accepted prune/merge edits this epoch (SI S3.6)."""

        return max(0, edit_budget(self.n_nodes) - self.accepted_this_epoch)

    def start_epoch(self) -> None:
        self.accepted_this_epoch = 0

    def advance(self, samples: int = 1) -> None:
        self.clock += int(samples)

    def propose(self, proposal: EditProposal) -> None:
        """Append a proposal, ordered by descending diagnostic strength (SI S3.6)."""

        # Negate priority so the max diagnostic strength pops first from the min-heap.
        heapq.heappush(
            self._heap,
            _QueueItem(-float(proposal.diagnostic_strength), self._seq, proposal),
        )
        self._seq += 1

    def pop(self) -> EditProposal | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap).proposal

    def _consumes_budget(self, proposal: EditProposal) -> bool:
        return proposal.edit_type in (EditType.PRUNE, EditType.MERGE)

    def can_accept(self, proposal: EditProposal) -> bool:
        """Whether accepting ``proposal`` now respects hysteresis and budget (S3.6)."""

        for node_id in proposal.affected_node_ids:
            if self.clock < self._locked_until.get(int(node_id), 0):
                return False
        if self._consumes_budget(proposal) and self.budget() <= 0:
            return False
        return True

    def commit(self, verdict: EvidenceVerdict) -> None:
        """Record an accepted edit: set hysteresis lockouts and consume budget (S3.6)."""

        if not verdict.accepted:
            return
        lock = self.clock + hysteresis_window(self.window())
        for node_id in verdict.proposal.affected_node_ids:
            self._locked_until[int(node_id)] = lock
        if self._consumes_budget(verdict.proposal):
            self.accepted_this_epoch += 1

    def evaluate(
        self,
        keep_region: Sequence[NodeTransition],
        edit_region: Sequence[NodeTransition],
        proposal: EditProposal,
        *,
        edit_stars: Mapping[int, np.ndarray] | None = None,
        keep_stars: Mapping[int, np.ndarray] | None = None,
    ) -> EvidenceVerdict:
        """Score ``proposal``; force-reject if cadence/budget forbids it now (S3.6)."""

        verdict = score_edit(
            keep_region,
            edit_region,
            proposal,
            config=self.config,
            edit_stars=edit_stars,
            keep_stars=keep_stars,
        )
        if verdict.accepted and not self.can_accept(proposal):
            verdict = EvidenceVerdict(
                accepted=False,
                f_dm_edit=verdict.f_dm_edit,
                f_dm_keep=verdict.f_dm_keep,
                log_bayes_factor=verdict.log_bayes_factor,
                margin=verdict.margin,
                proposal=proposal,
            )
        return verdict
