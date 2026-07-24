"""xfail markers tagged with SI section references.

Usage in test files::

    from tests.harness.markers import awaiting

    @awaiting("stage1.scaffold", si="S2.3")
    def test_variance_correction():
        ...
"""
from __future__ import annotations

import pytest


def awaiting(module: str, *, si: str = ""):
    """Mark a test as xfail until its implementation module lands.

    Parameters
    ----------
    module:
        Dotted module name within ``proteus`` that the test requires,
        e.g. ``"stage1.scaffold"`` or ``"evidence.dm_score"``.
    si:
        SI section reference for traceability, e.g. ``"S2.3"``.
    """
    reason = f"awaiting implementation: {module}"
    if si:
        reason += f" (SI {si})"
    def decorator(func):
        return pytest.mark.awaiting(
            pytest.mark.xfail(strict=True, reason=reason)(func)
        )

    return decorator


real_data = pytest.mark.real_data
slow = pytest.mark.slow
