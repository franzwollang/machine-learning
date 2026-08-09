"""Affected dual-subgraph connectivity stub (SI S10.4 / OPEN_ISSUES #43).

Green tests below lock the pure BFS hook and the ``dual_connected=False`` ⇒
evidence-path reject property. They intentionally use hand-built adjacency
dicts — **not** a Stage-2 dual/face graph.

S6 / M4 dual-flow wiring (do **not** flip without ``stage2.dual_flow``):

* Module key: ``stage2.dual_flow`` (SI S6.2 / S10.4).
* When dual-flow lands, replace the conservative ``None`` → ``True`` default by
  supplying a real post-edit dual adjacency into
  ``affected_dual_subgraph_connected`` (or call sites) and pass the bool into
  ``score_edit`` / ``EvidenceGate.evaluate``.
* Keep the disconnect⇒reject property test; add integration coverage under an
  ``@awaiting("stage2.dual_flow", ...)`` marker until the producer exists, then
  remove that marker (never weaken thresholds).
"""
from __future__ import annotations

import numpy as np
import pytest

from proteus.evidence import (
    EvidenceGate,
    affected_dual_subgraph_connected,
    bdeu_alpha,
    score_edit,
    star_incidence_matrix,
)
from proteus.evidence.dm_score import NodeTransition
from proteus.types import EditProposal, EditType
from tests.harness.markers import awaiting


def test_stub_defaults_true_without_dual_graph():
    """No S6 dual adjacency yet: stub asserts connectivity (OPEN_ISSUES #43)."""

    assert affected_dual_subgraph_connected(None, ["S0", "S1", "S2"]) is True
    assert affected_dual_subgraph_connected(None, []) is True


def test_induced_dual_subgraph_connectivity():
    """BFS on the induced dual subgraph: path-connected vs two components."""

    # Path S0—S1—S2 (facet-adjacency stand-in until S6 lands).
    path = {"S0": ["S1"], "S1": ["S0", "S2"], "S2": ["S1"]}
    assert affected_dual_subgraph_connected(path, ["S0", "S1", "S2"]) is True
    assert affected_dual_subgraph_connected(path, ["S0", "S2"]) is False  # induced drops S1

    # Two disjoint edges: affected set spanning both components is disconnected.
    disjoint = {"A": ["B"], "B": ["A"], "C": ["D"], "D": ["C"]}
    assert affected_dual_subgraph_connected(disjoint, ["A", "B"]) is True
    assert affected_dual_subgraph_connected(disjoint, ["A", "B", "C"]) is False


def test_disconnect_rejects_evidence_path():
    """Property (SI S10.4): dual_connected=False forces reject even when F_DM would accept."""

    a0 = bdeu_alpha(2)
    keep = [NodeTransition(np.array([15.0, 16.0]), 2, a0, node_id=0)]
    edit = [NodeTransition(np.array([30.0, 1.0]), 2, a0, node_id=0)]
    proposal = EditProposal(EditType.SPLIT, [0], diagnostic_strength=1.0)
    good_stars = {0: star_incidence_matrix([1, 2, 3, 4], [[0, 1, 2], [0, 3, 4]], 0)}

    # Dual graph of three simplices; edit "affects" endpoints only -> disconnected.
    dual = {"S0": ["S1"], "S1": ["S0", "S2"], "S2": ["S1"]}
    connected = affected_dual_subgraph_connected(dual, ["S0", "S1", "S2"])
    disconnected = affected_dual_subgraph_connected(dual, ["S0", "S2"])
    assert connected is True
    assert disconnected is False

    v_ok = score_edit(
        keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars,
        dual_connected=connected,
    )
    assert v_ok.accepted

    v_disc = score_edit(
        keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars,
        dual_connected=disconnected,
    )
    assert not v_disc.accepted

    gate = EvidenceGate(n_nodes=4)
    v_gate = gate.evaluate(
        keep, edit, proposal, edit_stars=good_stars, keep_stars=good_stars,
        dual_connected=disconnected,
    )
    assert not v_gate.accepted


@awaiting("stage2.dual_flow", si="S6.2")
def test_s6_dual_adjacency_wires_into_evidence_gate():
    """Integration placeholder: real dual/face graph → score_edit (OPEN_ISSUES #43).

    When ``stage2.dual_flow`` lands, this should:
    1. Build post-edit dry-run dual adjacency (simplices as verts; facet edges).
    2. Compute ``affected_dual_subgraph_connected(adj, affected_ids)``.
    3. Pass that bool into ``score_edit`` / ``EvidenceGate.evaluate``.
    4. Assert disconnect rejects on the evidence path (same property as the
       hand-built test above) — do not flip until the producer exists.
    """
    pytest.fail("awaiting stage2.dual_flow dual/face adjacency producer")
