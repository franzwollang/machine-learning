"""Evidence-gate cadence, hysteresis, and edit-budget invariants (SI S3.6)."""
from __future__ import annotations

import numpy as np

from proteus.evidence import (
    EvidenceGate,
    edit_budget,
    gate_window,
    hysteresis_window,
)
from proteus.types import EditProposal, EditType
from tests.harness.dm_fixtures import split_regions


def test_window_hysteresis_budget_formulas():
    """W = max(N_nodes, 4k|Q_C|); T_hyst = 2W; N_edit^+ = N_nodes/log N_nodes."""

    assert gate_window(10, 3, 10) == max(10, 4 * 10 * 3)  # 120
    assert gate_window(500, 1, 8) == 500                   # node-count dominates
    assert hysteresis_window(120) == 240
    assert edit_budget(100) == int(100 / np.log(100))
    assert edit_budget(4) == 2
    assert edit_budget(2) == 1   # floor for tiny regions
    assert edit_budget(1) == 1


def test_queue_pops_in_priority_order():
    """The proposal queue is keyed by descending diagnostic strength (SI S3.6)."""

    gate = EvidenceGate(n_nodes=10)
    gate.propose(EditProposal(EditType.SPLIT, [0], diagnostic_strength=0.2))
    gate.propose(EditProposal(EditType.SPLIT, [1], diagnostic_strength=0.9))
    gate.propose(EditProposal(EditType.SPLIT, [2], diagnostic_strength=0.5))

    assert gate.queue_len == 3
    assert gate.window() == max(10, 4 * 10 * 3)  # queue length enters W
    popped = [gate.pop().diagnostic_strength for _ in range(3)]
    assert popped == [0.9, 0.5, 0.2]


def _favored(rng, edit_type: EditType) -> tuple:
    keep, edit = split_regions(4000, sep=0.97, rng=rng)
    return keep, edit, EditProposal(edit_type, [0, 1], diagnostic_strength=1.0)


def test_edit_budget_caps_accepted_prune_merge():
    """At most N_nodes/log N_nodes accepted prune/merge edits per epoch (S3.6)."""

    gate = EvidenceGate(n_nodes=4)          # edit_budget(4) == 2
    rng = np.random.default_rng(1)
    accepted = 0
    for _ in range(5):
        keep, edit, prop = _favored(rng, EditType.PRUNE)
        verdict = gate.evaluate(keep, edit, prop)
        if verdict.accepted:
            gate.commit(verdict)
            accepted += 1
        gate.advance(10_000)                # clear hysteresis lockout each round
    assert accepted == 2
    assert gate.budget() == 0

    # A fresh epoch restores the budget.
    gate.start_epoch()
    assert gate.budget() == 2


def test_hysteresis_blocks_immediate_reversal():
    """An edit cannot be reverted within T_hyst = 2W samples of acceptance (S3.6)."""

    gate = EvidenceGate(n_nodes=100)        # ample budget
    rng = np.random.default_rng(2)

    keep, edit, prop = _favored(rng, EditType.MERGE)
    v1 = gate.evaluate(keep, edit, prop)
    assert v1.accepted
    gate.commit(v1)

    # Overlapping edit immediately after -> locked out.
    keep2, edit2, prop2 = _favored(rng, EditType.MERGE)
    assert not gate.evaluate(keep2, edit2, prop2).accepted

    # After the full T_hyst window it is allowed again.
    gate.advance(hysteresis_window(gate.window()) + 1)
    keep3, edit3, prop3 = _favored(rng, EditType.MERGE)
    assert gate.evaluate(keep3, edit3, prop3).accepted
