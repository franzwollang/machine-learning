"""Node, link, scaffold, and recursion-tree contracts (SI S2.3, S2.4, S4.4).

Canonical definitions live in :mod:`proteus.types`; this module re-exports them
so the SI-section-annotated contract surface is preserved for test authors
(OPEN_ISSUES #38).
"""
from __future__ import annotations

from proteus.types import Link, NodeState, RecursionTreeNode, RegionScaffold

__all__ = ["NodeState", "Link", "RegionScaffold", "RecursionTreeNode"]
