"""Hebbian directed-link bookkeeping with shadow/lifted two-tier edges.

All k-NN co-activations create or update edges (shadow by default).
Only BMU_1 -> BMU_2 co-activations promote an edge to "lifted" status.
The structural graph (used by d_final, simplex construction, pruning
peer comparisons, and CV computation) sees only lifted edges.  The full
graph including shadows is available for AP similarity computation.
"""

from __future__ import annotations

from collections.abc import Sequence

from tests.contracts.state import Link


class LinkCounters:
    """Directed counters on an undirected Hebbian graph."""

    def __init__(self) -> None:
        self._links: dict[tuple[int, int], Link] = {}

    def increment_directed(
        self,
        i: int,
        j: int,
        weight: float,
        *,
        lift: bool = False,
        protected_until: int | None = None,
    ) -> None:
        """Increment ``C(i -> j)`` by ``weight``, creating if absent.

        If ``lift`` is True the edge is promoted to structural status.
        When ``protected_until`` is set on **new** link creation or on first
        lift to structural, it is written to ``Link.protected_until`` (and
        extended on lift if the new deadline is larger).
        """

        if i == j:
            return
        weight_f = float(weight)
        if weight_f < 0.0:
            raise ValueError("link-counter weight must be non-negative")
        key = _ordered_key(i, j)
        link = self._links.get(key)
        if link is None:
            link = Link(
                i=key[0],
                j=key[1],
                lifted=lift,
                protected_until=int(protected_until) if protected_until is not None else 0,
            )
            self._links[key] = link
        elif lift and not link.lifted:
            link.lifted = True
            if protected_until is not None:
                link.protected_until = max(
                    int(link.protected_until), int(protected_until),
                )
        _add_weight(link, i, j, weight_f)

    def record_neighborhood(
        self,
        neighbors: Sequence[int],
        weights: Sequence[float],
        *,
        protected_until: int | None = None,
    ) -> None:
        """Record Hebbian co-activation evidence for one routed sample.

        BMU_1 -> BMU_2 (rank 0 -> rank 1) creates/lifts a structural edge.
        All other k-NN neighbors create or update shadow edges.
        """

        if len(neighbors) != len(weights):
            raise ValueError("neighbors and weights must have equal length")
        if len(neighbors) < 2:
            return
        bmu1 = int(neighbors[0])
        bmu2 = int(neighbors[1])
        self.increment_directed(
            bmu1, bmu2, float(weights[1]), lift=True, protected_until=protected_until,
        )
        for target, weight in zip(neighbors[2:], weights[2:]):
            self.increment_directed(
                bmu1, int(target), float(weight), lift=False, protected_until=protected_until,
            )

    def total_count(self, i: int, j: int) -> float:
        """Return ``C(i -> j) + C(j -> i)`` for the unordered edge."""

        link = self._links.get(_ordered_key(i, j))
        if link is None:
            return 0.0
        return float(link.count_ij + link.count_ji)

    def as_list(self) -> list[Link]:
        """Return all links (shadow + lifted) in stable sorted order."""

        return [self._links[key] for key in sorted(self._links)]

    def lifted_links(self) -> list[Link]:
        """Return only structurally lifted links."""

        return [link for link in self.as_list() if link.lifted]

    def remove(self, i: int, j: int) -> None:
        """Remove an unordered link if present."""

        self._links.pop(_ordered_key(i, j), None)

    def neighbour_graph(self, n_nodes: int) -> dict[int, list[int]]:
        """Return undirected adjacency of lifted (structural) edges only."""

        graph: dict[int, list[int]] = {idx: [] for idx in range(n_nodes)}
        for link in self.as_list():
            if link.lifted:
                graph[link.i].append(link.j)
                graph[link.j].append(link.i)
        return graph

    def full_graph(self, n_nodes: int) -> dict[int, list[int]]:
        """Return undirected adjacency of all edges (shadow + lifted)."""

        graph: dict[int, list[int]] = {idx: [] for idx in range(n_nodes)}
        for link in self.as_list():
            graph[link.i].append(link.j)
            graph[link.j].append(link.i)
        return graph

    def rebuild_from_links(self, links: list[Link]) -> None:
        """Replace contents with ``links`` in their current orientations."""

        self._links.clear()
        for link in links:
            self._links[_ordered_key(link.i, link.j)] = link


def _ordered_key(i: int, j: int) -> tuple[int, int]:
    a, b = int(i), int(j)
    if a <= b:
        return a, b
    return b, a


def _add_weight(link: Link, i: int, j: int, weight: float) -> None:
    if i == link.i and j == link.j:
        link.count_ij += weight
    elif i == link.j and j == link.i:
        link.count_ji += weight
