"""Dirichlet-multinomial evidence gate invariants (SI S3.4, S3.5)."""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from proteus.evidence import (
    NodeTransition,
    bdeu_alpha,
    evaluate_edit,
    f_dm,
    node_log_marginal,
)
from proteus.types import EditProposal, EditType
from tests.harness.dm_fixtures import split_regions


def _bf(n_events: int, sep: float) -> float:
    """Log-Bayes-factor favouring the true (split) topology at ``n_events``."""
    keep, split = split_regions(
        n_events, sep=sep, rng=np.random.default_rng(7)
    )
    proposal = EditProposal(EditType.SPLIT, [0, 1], diagnostic_strength=1.0)
    verdict = evaluate_edit(keep, split, proposal, tau_bf=3.0)
    return verdict.log_bayes_factor


def test_fdm_closed_form():
    """F_DM is closed-form: it equals the analytic Dirichlet-multinomial marginal
    exactly and is invariant to the order in which events were observed (there is
    no optimizer iteration count, SI S3.4/S10.2)."""

    a0 = bdeu_alpha(2)
    counts = np.array([5.0, 0.0, 12.0, 3.0])
    j = 4

    # Matches the S3.4 closed form computed independently with gammaln.
    n_i = counts.sum()
    ref = (
        gammaln(j * a0)
        - gammaln(j * a0 + n_i)
        + np.sum(gammaln(a0 + counts) - gammaln(a0))
    )
    got = node_log_marginal(counts, j, a0)
    assert abs(got - float(ref)) < 1e-12

    # Exchangeable: permuting the per-outcome event order cannot change the score.
    rng = np.random.default_rng(0)
    for _ in range(8):
        perm = rng.permutation(counts.size)
        assert abs(node_log_marginal(counts[perm], j, a0) - got) < 1e-12

    # Region F_DM is order-invariant over nodes as well.
    region = [
        NodeTransition(counts, j, a0, node_id=0),
        NodeTransition(np.array([1.0, 2.0]), 2, bdeu_alpha(1), node_id=1),
    ]
    assert abs(f_dm(region) - f_dm(list(reversed(region)))) < 1e-12


def test_fdm_monotone_in_evidence():
    """The evidence for the true topology strengthens monotonically with H_R:
    relative to the competing (merged) topology, F_DM(true) decreases without
    bound, i.e. the log-Bayes-factor favouring the true split grows with the
    affected-region event count (SI S3.5)."""

    sizes = [200, 800, 3200, 12800]
    bfs = [_bf(n, sep=0.98) for n in sizes]

    # Strictly increasing evidence for the true topology as H_R grows.
    for prev, nxt in zip(bfs, bfs[1:]):
        assert nxt > prev + 1.0
    # And genuinely favouring the split (positive) once evidence accrues.
    assert bfs[0] > 0.0
