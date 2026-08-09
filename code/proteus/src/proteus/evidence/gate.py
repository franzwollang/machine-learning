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
from collections import deque
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from math import log
from typing import Protocol, runtime_checkable

import numpy as np

from proteus.types import EditProposal, EditType, EvidenceVerdict
from proteus.evidence.dm_score import NodeTransition, evaluate_edit, f_dm
from proteus.evidence.star_matrix import RHO_MIN_DEFAULT, quarantined_nodes

__all__ = [
    "DualAdjacency",
    "GateConfig",
    "gate_window",
    "hysteresis_window",
    "edit_budget",
    "affected_dual_subgraph_connected",
    "score_edit",
    "EvidenceGate",
]


@runtime_checkable
class DualAdjacency(Protocol):
    """S6 dual / face-graph adjacency list shape (SI S6.2 / S10.4; #43).

    SI anchors (read-only cross-links; no SI prose change this turn):

    * Producer home — SI **S6 Dual Flow** ``\\label{sec:si-dual-flow}``;
      subsection **S6.2 Conservative Reconstruction** (face/factor graph that
      yields simplex adjacency once dual-flow lands as ``stage2.dual_flow``).
    * Consumer home — SI **S10.4 Star Matrix Identifiability**, paragraph
      **Dynamic preservation rule** (condition A2: affected dual subgraph
      remains connected for an edit to be evidence-bearing). Operational
      star-matrix proxy labeled ``\\label{par:si-star-runtime-matrix}``.

    Contract for values passed as ``dual_adjacency`` (not invented here):

    * **Vertices** — simplex ids of the *post-edit dry-run* complex.
    * **Edges** — undirected; two simplices are adjacent iff they share a facet
      (codim-1 face). Producer should keep the list symmetric.
    * **Representation** — adjacency list ``Mapping``-like: ``adj.get(u, ())``
      yields neighbor ids. A *missing* key is an isolated vertex (empty nbrs).
    * **Affected set** — separate ``affected_simplices`` arg: ids touched by the
      edit dry-run. Connectivity is of the *induced* subgraph on that set (BFS).
    * **``None``** — S6 producer unavailable; :func:`affected_dual_subgraph_connected`
      returns ``True`` (same default as ``score_edit(..., dual_connected=True)``).
    * **Producer** — :mod:`proteus.stage2.dual_flow` (proposal-path stub;
      ``DualFlowConfig.enable_dual_adjacency``, default off). Full S6 pressure
      solve / density still outstanding (#43). Call path: dry-run adj →
      ``affected_dual_subgraph_connected(adj, affected)`` →
      ``score_edit(..., dual_connected=bool)`` or gated
      ``apply_dual_adjacency`` kwargs on :func:`score_edit`.

    Vacuous ``True`` for empty / singleton affected sets. No behavior change vs
    plain ``Mapping[Hashable, Sequence[Hashable]]`` — this Protocol documents the
    S6 shape for typed call sites.
    """

    def get(
        self,
        key: Hashable,
        default: Sequence[Hashable] = (),
        /,
    ) -> Sequence[Hashable]:
        """Neighbors of ``key``, or ``default`` when missing / isolated."""
        ...


@dataclass(frozen=True)
class GateConfig:
    """Evidence-gate cadence parameters (SI S3.6, S14.3).

    Attributes
    ----------
    tau_bf:
        Bayes-factor threshold; the acceptance margin is ``log(tau_bf)``. SI S14.3
        lists ``log tau_BF`` in ``[1, 3]`` for splits (empirical), so the default
        ``tau_bf = 3.0`` gives ``log 3 ~ 1.10``, at the low end of that range.
    k:
        Neighbour count used in the equilibration-window formula (SI S3.6).
    rho_min:
        Star-matrix conditioning flag threshold (SI S10.4).
    apply_dual_adjacency:
        Proposal-path flag (#43 / SI S10.4 A2). When ``False`` (default),
        ``score_edit`` / :meth:`EvidenceGate.evaluate` honour the caller-supplied
        ``dual_connected`` bool and ignore any ``dual_adjacency`` kwarg —
        acceptance path unchanged. When ``True`` and both ``dual_adjacency`` and
        ``affected_simplices`` are provided, connectivity is computed via
        :func:`affected_dual_subgraph_connected` (Stage-2 dual-flow stub).
        Operational default off until full S6 dual-flow is acceptance-ready.
    """

    tau_bf: float = 3.0
    k: int = 10
    rho_min: float = RHO_MIN_DEFAULT
    apply_dual_adjacency: bool = False


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


def affected_dual_subgraph_connected(
    dual_adjacency: DualAdjacency | None,
    affected_simplices: Sequence[Hashable],
) -> bool:
    """Whether the induced dual subgraph on ``affected_simplices`` is connected.

    SI S10.4's dynamic-preservation rule (A2) requires the *affected* dual
    subgraph to stay connected for an edit to be evidence-bearing. Vertices of
    the dual graph are simplices; an edge joins two simplices that share a
    facet (codim-1 face). That adjacency is a Stage-2 dual-flow / face-graph
    artifact (SI S6 / S10.4). Experimental producer:
    :mod:`proteus.stage2.dual_flow` (proposal-path, flag-gated; full S6 pressure
    solve still outstanding — OPEN_ISSUES #43).

    ``dual_adjacency`` shape: see :class:`DualAdjacency` (S6 adjacency contract).

    Behavior of this stub:

    * ``dual_adjacency is None`` — S6 graph unavailable; conservatively return
      ``True`` so callers without a dual graph assert connectivity (same default
      as ``score_edit(..., dual_connected=True)``).
    * Empty or singleton ``affected_simplices`` — vacuously connected.
    * Otherwise — BFS on the *induced* subgraph (only walk neighbors that are
      themselves in the affected set). Returns ``False`` if any affected
      simplex is unreachable from the first.

    Call sites should compute this on the post-edit dry-run complex and pass the
    boolean into :func:`score_edit` / :meth:`EvidenceGate.evaluate` as
    ``dual_connected=...``, or (proposal-path) pass ``dual_adjacency`` +
    ``affected_simplices`` with ``GateConfig.apply_dual_adjacency=True`` so the
    gate computes connectivity from the ``stage2.dual_flow`` stub.
    """

    if dual_adjacency is None:
        return True

    affected = list(dict.fromkeys(affected_simplices))
    if len(affected) <= 1:
        return True

    affected_set = set(affected)
    start = affected[0]
    seen: set[Hashable] = {start}
    queue: deque[Hashable] = deque([start])
    while queue:
        u = queue.popleft()
        for v in dual_adjacency.get(u, ()):
            if v in affected_set and v not in seen:
                seen.add(v)
                queue.append(v)
    return seen == affected_set


def score_edit(
    keep_region: Sequence[NodeTransition],
    edit_region: Sequence[NodeTransition],
    proposal: EditProposal,
    *,
    config: GateConfig | None = None,
    edit_stars: Mapping[int, np.ndarray] | None = None,
    keep_stars: Mapping[int, np.ndarray] | None = None,
    dual_connected: bool = True,
    dual_adjacency: DualAdjacency | None = None,
    affected_simplices: Sequence[Hashable] | None = None,
) -> EvidenceVerdict:
    """Score one edit under the S10.4 dynamic-preservation rule (SI S3.4, S10.4).

    An edit is *evidence-bearing* only if **every** affected pre- and post-edit
    star is well-conditioned (``rho_i >= rho_min``) **and** the affected dual
    subgraph stays connected (``dual_connected``). If either fails, the edit
    cannot claim an ``F_DM`` improvement: it is rejected on the evidence path
    (a geometry-only remediation path, deferred, may still accept it). This is
    all-or-nothing per S10.4 -- partial evidence from the still-conditioned nodes
    may **not** be used to accept the edit.

    ``edit_stars`` / ``keep_stars`` map affected node ids to their star matrices
    (see :mod:`proteus.evidence.star_matrix`); when omitted the caller asserts the
    stars are conditioned. ``dual_connected`` is the affected dual-subgraph
    connectivity result from the dry run (Stage-2 dual graph; OPEN_ISSUES #43).

    When ``config.apply_dual_adjacency`` is true and both ``dual_adjacency`` and
    ``affected_simplices`` are provided, ``dual_connected`` is overwritten by
    :func:`affected_dual_subgraph_connected` (proposal-path wiring for the
    ``stage2.dual_flow`` stub). Flag off ⇒ kwargs ignored; acceptance path
    unchanged.
    """

    config = config or GateConfig()
    if (
        config.apply_dual_adjacency
        and dual_adjacency is not None
        and affected_simplices is not None
    ):
        dual_connected = affected_dual_subgraph_connected(
            dual_adjacency, affected_simplices
        )

    ill: set[int] = set()
    if edit_stars is not None:
        ill |= quarantined_nodes(edit_stars, config.rho_min)
    if keep_stars is not None:
        ill |= quarantined_nodes(keep_stars, config.rho_min)

    if ill or not dual_connected:
        # Not evidence-bearing (S10.4): F_DM is reported for diagnostics but the
        # edit cannot be accepted on the evidence path.
        f_keep = f_dm(keep_region)
        f_edit = f_dm(edit_region)
        return EvidenceVerdict(
            accepted=False,
            f_dm_edit=f_edit,
            f_dm_keep=f_keep,
            log_bayes_factor=f_keep - f_edit,
            margin=float(log(config.tau_bf)),
            proposal=proposal,
        )

    return evaluate_edit(
        keep_region,
        edit_region,
        proposal,
        tau_bf=config.tau_bf,
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
        # Window at the moment the gate last fired, used for T_hyst = 2 W so the
        # lockout reflects the firing queue, not the post-pop queue (SI S3.6).
        self._firing_window = self.window()

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
        lock = self.clock + hysteresis_window(self._firing_window)
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
        dual_connected: bool = True,
        dual_adjacency: DualAdjacency | None = None,
        affected_simplices: Sequence[Hashable] | None = None,
    ) -> EvidenceVerdict:
        """Score ``proposal``; force-reject if cadence/budget forbids it now (S3.6)."""

        # Freeze the firing window (SI S3.6) before the queue mutates, so a later
        # commit() derives T_hyst = 2 W from the queue that actually fired.
        self._firing_window = self.window()
        verdict = score_edit(
            keep_region,
            edit_region,
            proposal,
            config=self.config,
            edit_stars=edit_stars,
            keep_stars=keep_stars,
            dual_connected=dual_connected,
            dual_adjacency=dual_adjacency,
            affected_simplices=affected_simplices,
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
