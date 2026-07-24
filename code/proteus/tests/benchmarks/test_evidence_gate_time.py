"""Evidence gate scoring wall-time benchmark (SI S3.6)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from proteus.evidence import NodeTransition, bdeu_alpha, score_edit
from proteus.types import EditProposal, EditType
from tests.harness.budgets import load_budgets


@pytest.mark.benchmark
def test_evidence_gate_time_small() -> None:
    """F_DM scoring on a small affected region must complete within budget (S3.6).

    The gate is local: cost scales with the number of changed stars plus a
    neighbour ring, not with the global mesh (SI S4.6)."""

    budget = load_budgets("small")["evidence_gate_time"]
    rng = np.random.default_rng(3)

    n_nodes = 64          # affected region: changed stars + neighbour ring
    max_out = 12          # outgoing outcomes per node
    a0 = bdeu_alpha(4)

    def make_region(offset: float) -> list[NodeTransition]:
        region = []
        for i in range(n_nodes):
            counts = rng.integers(0, 50, size=max_out).astype(float) + offset
            region.append(NodeTransition(counts, max_out, a0, node_id=i))
        return region

    keep_region = make_region(0.0)
    edit_region = make_region(1.0)
    proposal = EditProposal(EditType.SPLIT, list(range(n_nodes)), diagnostic_strength=1.0)

    start = time.perf_counter()
    for _ in range(100):  # amortize timer noise over repeated scorings
        score_edit(keep_region, edit_region, proposal)
    elapsed = time.perf_counter() - start

    assert elapsed < float(budget), (
        f"evidence-gate scoring {elapsed:.3f}s exceeds budget {budget}s"
    )
