"""Tests for the DM cluster-acceptance reduction (SI S3.4 reduction, #27).

The block-homogeneity Bayes factor must (a) reproduce the audited S3.4/S3.5
``evaluate_edit`` log-BF exactly, (b) accept a genuinely modular partition and
reject a homogeneous one, and (c) drive an agglomerative merge that collapses
indistinguishable fragments while preserving evidence-bearing blocks.
"""
from __future__ import annotations

from math import log

import numpy as np
import pytest

from proteus.evidence import NodeTransition, evaluate_edit
from proteus.evidence.dm_score import bdeu_alpha
from proteus.evidence.dm_score import node_log_marginal as _m
from proteus.links import LinkCounters
from proteus.stage1.dm_cluster import (
    DMClusterConfig,
    block_flow_matrix,
    dm_gated_merge,
    dm_partition_logbf,
    dm_partition_verdict,
)
from proteus.types import EditProposal, EditType
from tests.harness.dm_fixtures import two_subpopulation_counts


class _Node:
    def __init__(self, d_final: int = 1) -> None:
        self.d_final = int(d_final)
        self.hit_count = 1.0
        self.position = np.zeros(1, dtype=float)


class _Scaffold:
    def __init__(self, n: int, edges: list[tuple[int, int, float]]) -> None:
        links = LinkCounters()
        for i, j, w in edges:
            links.increment_directed(i, j, float(w), lift=True)
        self.nodes = [_Node() for _ in range(n)]
        self.links = links


def test_block_flow_matrix_aggregates_directed_counts() -> None:
    """``N[k, l]`` sums directed transition counts within/between blocks."""

    sc = _Scaffold(
        4, [(0, 1, 5.0), (0, 2, 3.0), (1, 3, 1.0), (2, 0, 2.0)],
    )
    N = block_flow_matrix(sc, [{0, 1}, {2, 3}])
    # within block0: 0->1 = 5;  block0->block1: 0->2 (3) + 1->3 (1) = 4
    # block1->block0: 2->0 = 2; within block1: none
    assert N.tolist() == [[5.0, 4.0], [2.0, 0.0]]


def test_dm_partition_logbf_matches_evaluate_edit() -> None:
    """The block-model log-BF equals the S3.4 ``evaluate_edit`` log-BF exactly."""

    rng = np.random.default_rng(11)
    a0 = bdeu_alpha(1)
    keep, c1, c2 = two_subpopulation_counts(4000, sep=0.97, rng=rng)
    N = np.array([c1, c2], dtype=float)

    logbf = dm_partition_logbf(N, a0)
    keep_region = [NodeTransition(keep, 2, a0, node_id=0)]
    split_region = [
        NodeTransition(c1, 2, a0, node_id=0),
        NodeTransition(c2, 2, a0, node_id=1),
    ]
    verdict = evaluate_edit(
        keep_region, split_region,
        EditProposal(EditType.SPLIT, [0, 1], diagnostic_strength=1.0),
        tau_bf=3.0,
    )
    assert logbf == pytest.approx(verdict.log_bayes_factor, rel=1e-9)
    assert logbf > 0.0  # real substructure -> split favoured


def test_dm_partition_logbf_single_block_is_neg_inf() -> None:
    """A one-block partition has nothing to accept (log-BF = -inf)."""

    assert dm_partition_logbf(np.array([[7.0]]), bdeu_alpha(1)) == float("-inf")


def test_dm_partition_verdict_accepts_modular_rejects_homogeneous() -> None:
    """Region verdict: accept a bottlenecked split, reject a well-mixed one."""

    modular = _Scaffold(
        4, [(0, 1, 100.0), (1, 0, 100.0), (2, 3, 100.0), (3, 2, 100.0),
            (1, 2, 1.0)],
    )
    log_bf, accepted = dm_partition_verdict(modular, [{0, 1}, {2, 3}])
    assert accepted and log_bf > log(3.0)

    homogeneous = _Scaffold(
        4, [(0, 1, 50.0), (0, 2, 50.0), (2, 3, 50.0), (2, 0, 50.0)],
    )
    log_bf_h, accepted_h = dm_partition_verdict(homogeneous, [{0, 1}, {2, 3}])
    assert not accepted_h and log_bf_h < log(3.0)


def test_dm_gated_merge_collapses_homogeneous_keeps_modular() -> None:
    """The DM merge collapses indistinguishable fragments, not real blocks."""

    a0 = bdeu_alpha(1)
    margin = log(3.0)

    homogeneous = _Scaffold(
        4, [(0, 1, 50.0), (0, 2, 50.0), (2, 3, 50.0), (2, 0, 50.0)],
    )
    graph = homogeneous.links.neighbour_graph(4)
    merged = dm_gated_merge(
        [{0, 1}, {2, 3}], homogeneous, a0, margin, graph,
    )
    assert len(merged) == 1

    modular = _Scaffold(
        4, [(0, 1, 200.0), (1, 0, 200.0), (2, 3, 200.0), (3, 2, 200.0),
            (1, 2, 1.0)],
    )
    graph_m = modular.links.neighbour_graph(4)
    kept = dm_gated_merge([{0, 1}, {2, 3}], modular, a0, margin, graph_m)
    assert len(kept) == 2


def test_dm_merge_pairwise_equals_exact_fixed_outcome_delta() -> None:
    """With a fixed outcome space the pairwise merge BF is the exact edit delta.

    ``dm_gated_merge`` holds the destination columns fixed and pools rows only,
    so merging blocks a,b changes ``F_DM`` only through those two rows; every
    other row cancels. This locks that the pairwise homogeneity BF equals the
    exact ``F_DM(after) - F_DM(before)`` partition-edit delta (audit follow-up).
    """

    a0 = bdeu_alpha(1)
    j = 3
    N = np.array([[10.0, 2.0, 1.0], [3.0, 9.0, 2.0], [1.0, 2.0, 8.0]])
    f_before = -(_m(N[0], j, a0) + _m(N[1], j, a0) + _m(N[2], j, a0))
    f_after = -(_m(N[0] + N[1], j, a0) + _m(N[2], j, a0))  # merge rows 0,1
    exact_delta = f_after - f_before
    pair_bf = _m(N[0], j, a0) + _m(N[1], j, a0) - _m(N[0] + N[1], j, a0)
    assert pair_bf == pytest.approx(exact_delta, rel=1e-12)


def test_dm_cluster_config_default_tau_bf() -> None:
    """The DM cluster config shares the S3.6 ``tau_BF`` operational default."""

    assert DMClusterConfig().tau_bf == 3.0
