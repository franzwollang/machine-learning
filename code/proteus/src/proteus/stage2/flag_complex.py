"""Stage 2 flag-complex construction and topology summary (SI S4.1, S4.2, S4.5, S13.4).

Stage 2 represents the topology of a region as the *top-dimensional flag complex*
of its stabilized *lifted* Hebbian graph (SI S4.1). A ``d``-simplex
``sigma = {v_0, ..., v_d}`` is present exactly when every pair of its vertices is
a lifted edge (i.e. the vertices form a clique) and the local intrinsic-dimension
rule permits a ``d``-simplex in that star. The graph owns topology; the complex is
the closure of that topology at the locally estimated dimension, so link/node
edits induce simplex edits automatically and no separate Delaunay / alpha-complex
maintenance is needed.

Construction follows Greedy Chaining (SI S4.2, algorithm box S13.4): a
degree-descending breadth-first traversal seeds each node ``i`` at its local
intrinsic dimension ``d_final[i]``, enumerates rank-ordered ``d``-subsets of its
lifted neighbourhood, verifies clique completion by adjacency lookup, and inserts
the canonical sorted simplex into a deduplication set. The per-node simplex
incidence count is retained as a junction signature (SI S4.2): interior nodes of a
homogeneous ``d``-manifold reach the expected incidence, whereas junction and
orphan nodes under-fill and are flagged.

Topology *validation* (recovering Betti numbers from a fitted region) is a
separate evaluation concern (SI S14.2, Vietoris--Rips persistent homology on node
positions, ``tests/metrics/persistent_homology.py``) and is deliberately not part
of this construction module. Note two current limitations that block end-to-end
topology-recovery scenarios (tracked in OPEN_ISSUES): the operational ``d_final``
is seeded to the working dimension and never refreshed (OPEN_ISSUES #40), so a
scaffold does not yet carry the heterogeneous per-patch intrinsic dimensions the
S4.2 junction mesh needs; and the sparse lifted-graph flag complex of a
tissue-polluted scaffold retains essential spurious loops (OPEN_ISSUES #25).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import combinations
from math import factorial
from typing import Optional, Sequence

import numpy as np

from proteus.types import Complex, Simplex

__all__ = [
    "FlagComplexConfig",
    "FlagComplexResult",
    "build_flag_complex",
    "flag_complex_from_scaffold",
    "simplex_volume",
]


@dataclass(frozen=True)
class FlagComplexConfig:
    """Greedy-Chaining construction parameters (SI S4.2, S13.4).

    Attributes
    ----------
    max_simplex_dim:
        Optional operational cap on the enumerated simplex dimension. ``None``
        uses each node's ``d_final`` verbatim (the SI default). A finite cap
        bounds the ``C(|N(i)|, d)`` subset enumeration on very high-dimensional
        stars; it is an *operational* guard, not a theoretical constant.
    final_extension:
        If ``True`` (SI S4.2), nodes with zero simplex incidence after the main
        pass receive one final node-seeded extension attempt at the largest
        clique dimension their neighbourhood admits (bounded by ``d_final``).
    """

    max_simplex_dim: Optional[int] = None
    final_extension: bool = True


@dataclass
class FlagComplexResult:
    """Constructed flag complex plus junction/orphan diagnostics (SI S4.2)."""

    complex: Complex
    simplices: list[tuple[int, ...]]
    incidence_counts: np.ndarray          # (N,) simplex incidence per node
    orphan_ids: list[int]                 # nodes with zero simplex incidence


def simplex_volume(vertex_positions: np.ndarray) -> float:
    """Unsigned ``d``-volume of the simplex on ``d+1`` vertices (SI S4.5).

    ``vertex_positions`` is ``(d+1, D)``. The volume is
    ``sqrt(det(E^T E)) / d!`` where ``E`` stacks the ``d`` edge vectors from the
    first vertex. A 0-simplex (single vertex) has zero ``d``-volume.
    """

    pts = np.asarray(vertex_positions, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return 0.0
    d = pts.shape[0] - 1
    edges = pts[1:] - pts[0]            # (d, D)
    gram = edges @ edges.T             # (d, d)
    det = float(np.linalg.det(gram))
    if det <= 0.0:
        return 0.0
    return float(np.sqrt(det) / factorial(d))


def _symmetric_adjacency(
    neighbour_graph: dict[int, Sequence[int]], n_nodes: int
) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for i, nbrs in neighbour_graph.items():
        if i < 0 or i >= n_nodes:
            continue
        for j in nbrs:
            if j == i or j < 0 or j >= n_nodes:
                continue
            adj[i].add(j)
            adj[j].add(i)
    return adj


def build_flag_complex(
    neighbour_graph: dict[int, Sequence[int]],
    d_final: Sequence[int],
    positions: np.ndarray,
    config: Optional[FlagComplexConfig] = None,
) -> FlagComplexResult:
    """Build the top-dimensional flag complex via Greedy Chaining (SI S4.2, S13.4).

    Parameters
    ----------
    neighbour_graph:
        Undirected adjacency of the *lifted* Hebbian edges
        (``Stage1Scaffold.neighbour_graph()``). Need not be symmetric on input;
        it is symmetrized internally.
    d_final:
        Per-node local intrinsic dimension (``NodeState.d_final``).
    positions:
        ``(N, D)`` node positions for simplex volumes.
    config:
        :class:`FlagComplexConfig`; defaults used when ``None``.
    """

    if config is None:
        config = FlagComplexConfig()
    positions = np.asarray(positions, dtype=float)
    n_nodes = len(d_final)
    adj = _symmetric_adjacency(neighbour_graph, n_nodes)

    def _target_dim(i: int) -> int:
        d = int(d_final[i])
        if config.max_simplex_dim is not None:
            d = min(d, int(config.max_simplex_dim))
        return max(d, 0)

    def _is_clique(vertices: tuple[int, ...]) -> bool:
        for a, b in combinations(vertices, 2):
            if b not in adj[a]:
                return False
        return True

    simplices: set[tuple[int, ...]] = set()
    incidence = np.zeros(n_nodes, dtype=int)

    # Degree-descending priority queue with breadth-first amortization (S13.4).
    heap: list[tuple[int, int]] = [(-len(adj[v]), v) for v in range(n_nodes)]
    heapq.heapify(heap)
    visited = np.zeros(n_nodes, dtype=bool)

    def _seed_node(i: int) -> None:
        d = _target_dim(i)
        neighbours = sorted(adj[i])
        if d <= 0 or len(neighbours) < d:
            return
        if incidence[i] >= d + 1:
            return
        for subset in combinations(neighbours, d):
            if not _is_clique(subset):
                continue
            sigma = tuple(sorted((i, *subset)))
            if sigma in simplices:
                continue
            simplices.add(sigma)
            for v in sigma:
                incidence[v] += 1
            if incidence[i] >= d + 1:
                break

    while heap:
        _, i = heapq.heappop(heap)
        if visited[i]:
            continue
        visited[i] = True
        _seed_node(i)
        for j in sorted(adj[i]):
            if not visited[j]:
                heapq.heappush(heap, (-len(adj[j]), j))

    # Final node-seeded extension for zero-incidence nodes (SI S4.2).
    if config.final_extension:
        for i in range(n_nodes):
            if incidence[i] > 0:
                continue
            _seed_largest_clique(i, adj, positions, _target_dim, simplices, incidence)

    orphan_ids = [int(i) for i in range(n_nodes) if incidence[i] == 0]

    simplex_list = sorted(simplices)
    simplex_objs: list[Simplex] = []
    top_dim = 0
    for sigma in simplex_list:
        vol = simplex_volume(positions[list(sigma)]) if positions.size else 0.0
        simplex_objs.append(Simplex(vertex_ids=sigma, volume=vol, mass=0.0))
        top_dim = max(top_dim, len(sigma) - 1)

    complex_obj = Complex(
        simplices=simplex_objs,
        vertex_positions=positions,
        intrinsic_dim=top_dim,
    )
    return FlagComplexResult(
        complex=complex_obj,
        simplices=simplex_list,
        incidence_counts=incidence,
        orphan_ids=orphan_ids,
    )


def _seed_largest_clique(
    i: int,
    adj: list[set[int]],
    positions: np.ndarray,
    target_dim,
    simplices: set[tuple[int, ...]],
    incidence: np.ndarray,
) -> None:
    """One extension attempt: add the largest admissible clique through ``i``."""

    neighbours = sorted(adj[i])
    d_cap = target_dim(i)
    for d in range(min(d_cap, len(neighbours)), 0, -1):
        for subset in combinations(neighbours, d):
            ok = True
            for a, b in combinations(subset, 2):
                if b not in adj[a]:
                    ok = False
                    break
            if not ok:
                continue
            sigma = tuple(sorted((i, *subset)))
            # A zero-incidence node cannot already belong to a stored simplex,
            # so this insert is always new.
            simplices.add(sigma)
            for v in sigma:
                incidence[v] += 1
            return


def flag_complex_from_scaffold(
    scaffold,
    config: Optional[FlagComplexConfig] = None,
) -> FlagComplexResult:
    """Convenience wrapper: build the flag complex from a ``Stage1Scaffold``.

    Uses the lifted-edge neighbour graph, per-node ``d_final``, and node
    positions (SI S4.1 T3 transfer inherits the stabilized Stage 1 graph).
    """

    graph = scaffold.neighbour_graph()
    d_final = [int(node.d_final) for node in scaffold.nodes]
    positions = np.array([np.asarray(node.position, dtype=float) for node in scaffold.nodes])
    return build_flag_complex(graph, d_final, positions, config=config)
