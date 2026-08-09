"""Experimental Stage-2 dual / face-graph adjacency stub (SI S6.2 / S10.4; #43).

This module is a *proposal-path* producer for the evidence gate's affected
dual-subgraph connectivity check (SI S10.4 dynamic preservation A2). It builds
an undirected dual adjacency whose vertices are simplex ids and whose edges
join simplices that share a facet (codim-1 face) — the S6 face/factor graph
shape documented on :class:`proteus.evidence.gate.DualAdjacency`.

**What this stub is not.** Full SI S6 dual-flow (online face-pressure tallies,
conservative reconstruction via loopy Gaussian BP, boundary taxonomy, simplex-
local density) is **not** implemented here. Those remain M4 / OPEN_ISSUES #43
follow-ons; mass-conservation and density ``@awaiting("stage2.dual_flow")``
tests stay xfail until that producer lands. This file only unblocks the
adjacency → gate wiring path.

Flag (proposal-path, SI S14.3 operational default):

* ``DualFlowConfig.enable_dual_adjacency`` defaults **off**. When off, builders
  return ``None`` so :func:`proteus.evidence.gate.affected_dual_subgraph_connected`
  keeps its conservative ``True`` default (acceptance path unchanged).
* Call sites that opt in (tests / experimental dry-runs) pass
  ``enable_dual_adjacency=True`` and feed the returned adjacency into the gate.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from proteus.evidence.gate import (
    DualAdjacency,
    affected_dual_subgraph_connected,
)
from proteus.types import Complex

__all__ = [
    "DualFlowConfig",
    "DualAdjacencyDict",
    "build_dual_adjacency",
    "build_dual_adjacency_from_complex",
    "affected_subgraph_connected",
    "resolve_dual_connected",
]

# Concrete DualAdjacency realization used by this stub (SI S6.2 contract).
DualAdjacencyDict: TypeAlias = dict[Hashable, tuple[Hashable, ...]]


@dataclass(frozen=True)
class DualFlowConfig:
    """Proposal-path flags for the dual-adjacency stub (SI S6.2 / S14.3).

    Attributes
    ----------
    enable_dual_adjacency:
        When ``False`` (default), builders return ``None`` and the evidence gate
        keeps its conservative connectivity default. When ``True``, builders
        emit a facet-sharing dual adjacency for experimental / dry-run wiring.
        Operational default — not derived or calibrated; acceptance-path code
        must leave this off until full S6 dual-flow is ready (#43).
    """

    enable_dual_adjacency: bool = False


def _as_vertex_frozenset(vertices: Sequence[Hashable]) -> frozenset[Hashable]:
    verts = frozenset(vertices)
    if len(verts) != len(list(vertices)):
        raise ValueError("simplex vertex ids must be unique")
    if not verts:
        raise ValueError("simplex must have at least one vertex")
    return verts


def _facets(vertices: frozenset[Hashable]) -> list[frozenset[Hashable]]:
    """Codim-1 faces of ``vertices`` (drop one vertex each)."""

    if len(vertices) <= 1:
        return []
    return [vertices - {v} for v in vertices]


def build_dual_adjacency(
    simplices: Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]],
    *,
    config: DualFlowConfig | None = None,
) -> DualAdjacencyDict | None:
    """Build undirected dual adjacency from simplex vertex lists (SI S6.2).

    Parameters
    ----------
    simplices:
        Either a sequence of vertex-id sequences (simplex id = enumeration
        index ``0..n-1``) or a mapping ``simplex_id -> vertex_ids``.
    config:
        When ``enable_dual_adjacency`` is false, returns ``None`` immediately.

    Returns
    -------
    DualAdjacencyDict | None
        Symmetric adjacency list suitable for
        :func:`affected_dual_subgraph_connected`, or ``None`` when the flag is
        off. Isolated simplices appear with an empty neighbor tuple.
    """

    cfg = config or DualFlowConfig()
    if not cfg.enable_dual_adjacency:
        return None

    if isinstance(simplices, Mapping):
        items: list[tuple[Hashable, frozenset[Hashable]]] = [
            (sid, _as_vertex_frozenset(verts)) for sid, verts in simplices.items()
        ]
    else:
        items = [
            (i, _as_vertex_frozenset(verts)) for i, verts in enumerate(simplices)
        ]

    # facet -> simplex ids that own that facet (codim-1 face).
    facet_owners: dict[frozenset[Hashable], list[Hashable]] = defaultdict(list)
    for sid, verts in items:
        for facet in _facets(verts):
            facet_owners[facet].append(sid)

    nbrs: dict[Hashable, set[Hashable]] = {sid: set() for sid, _ in items}
    for owners in facet_owners.values():
        if len(owners) < 2:
            continue
        # Every pair of simplices sharing this facet is dual-adjacent.
        for i, a in enumerate(owners):
            for b in owners[i + 1 :]:
                if a == b:
                    continue
                nbrs[a].add(b)
                nbrs[b].add(a)

    return {sid: tuple(sorted(nbrs[sid], key=repr)) for sid, _ in items}


def build_dual_adjacency_from_complex(
    complex: Complex,
    *,
    config: DualFlowConfig | None = None,
    id_fn: None = None,
) -> DualAdjacencyDict | None:
    """Build dual adjacency from a :class:`~proteus.types.Complex` (SI S6.2).

    Simplex ids default to enumeration indices into ``complex.simplices``.
    """

    del id_fn  # reserved for stable external ids; unused in this stub
    verts_by_index = [tuple(s.vertex_ids) for s in complex.simplices]
    return build_dual_adjacency(verts_by_index, config=config)


def affected_subgraph_connected(
    dual_adjacency: DualAdjacency | None,
    affected_simplices: Sequence[Hashable],
) -> bool:
    """Induced dual-subgraph connectivity (SI S10.4 A2).

    Thin alias of :func:`proteus.evidence.gate.affected_dual_subgraph_connected`
    so Stage-2 call sites can import the producer+check from one module.
    """

    return affected_dual_subgraph_connected(dual_adjacency, affected_simplices)


def resolve_dual_connected(
    simplices: Sequence[Sequence[Hashable]] | Mapping[Hashable, Sequence[Hashable]],
    affected_simplices: Sequence[Hashable],
    *,
    config: DualFlowConfig | None = None,
) -> bool:
    """Build dual adjacency (if enabled) and return affected-subgraph connectivity.

    When the dual-adjacency flag is off, returns ``True`` (same conservative
    default as ``score_edit(..., dual_connected=True)`` / ``adj is None``).
    """

    adj = build_dual_adjacency(simplices, config=config)
    return affected_subgraph_connected(adj, affected_simplices)
