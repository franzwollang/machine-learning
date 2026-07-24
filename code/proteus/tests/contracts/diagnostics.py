"""Junction, ledger, and boundary diagnostics contracts (SI S8.4, S9.2, S6.3).

Canonical definitions live in :mod:`proteus.types` (OPEN_ISSUES #38).
"""
from __future__ import annotations

from proteus.types import (
    BoundaryClassification,
    BoundaryType,
    ExpressivityLedger,
    JunctionScore,
)

__all__ = [
    "JunctionScore",
    "BoundaryType",
    "BoundaryClassification",
    "ExpressivityLedger",
]
