"""Test configuration for the rebuilt Proteus implementation."""

from __future__ import annotations

import pathlib
import sys

import pytest

SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


_DIR_MARKERS = {
    "foundation": ("foundation",),
    "properties": ("property",),
    "reductions": ("reduction",),
    "benchmarks": ("benchmark",),
    "stage1": ("stage1",),
    "stage2": ("stage2",),
    "evidence": ("evidence",),
    "inference": ("inference",),
    "diagnostics": ("diagnostics",),
}

_SCENARIO_MARKERS = {
    "synthetic": ("scenario", "synthetic"),
    "real": ("scenario", "real_data"),
}

_TARGET_MARKERS = {
    "stage1.": "stage1",
    "stage2.": "stage2",
    "evidence.": "evidence",
    "inference.": "inference",
    "diagnostics.": "diagnostics",
}


def pytest_collection_modifyitems(config, items):
    """Attach stable semantic markers to every collected test.

    The project keeps many future-facing xfail tests in the tree.  These
    markers make it easy to select and summarize tests by layer, while xfail
    reasons preserve the exact implementation module that is still pending.
    """

    for item in items:
        rel_parts = item.path.relative_to(pathlib.Path(__file__).parent).parts
        markers: set[str] = set()

        if rel_parts:
            top = rel_parts[0]
            markers.update(_DIR_MARKERS.get(top, ()))
            if top == "scenarios" and len(rel_parts) > 1:
                markers.update(_SCENARIO_MARKERS.get(rel_parts[1], ("scenario",)))

        for existing in item.iter_markers():
            if existing.name == "xfail" and existing.kwargs.get("reason"):
                reason = str(existing.kwargs["reason"])
                if reason.startswith("awaiting implementation: "):
                    markers.add("awaiting")
                    target = reason.split("awaiting implementation: ", 1)[1]
                    target = target.split(" ", 1)[0]
                    for prefix, marker in _TARGET_MARKERS.items():
                        if target.startswith(prefix):
                            markers.add(marker)

        for marker in sorted(markers):
            item.add_marker(getattr(pytest.mark, marker))


# ---------------------------------------------------------------------------
# Marker-grouped terminal summary
# ---------------------------------------------------------------------------

_SUMMARY_MARKERS = [
    "foundation",
    "property",
    "reduction",
    "stage1",
    "stage2",
    "evidence",
    "inference",
    "diagnostics",
    "scenario",
    "synthetic",
    "real_data",
    "benchmark",
    "awaiting",
]

# Stash marker names on each TestReport so the terminal summary can read them
# without needing access to the original Item (which is gone by that point).

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    report._marker_names = {
        m.name for m in item.iter_markers() if m.name in _SUMMARY_MARKERS
    }


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print a marker-grouped outcome table at the end of the run."""

    marker_counts: dict[str, dict[str, int]] = {
        m: {} for m in _SUMMARY_MARKERS
    }

    outcome_map = {
        "passed": "passed",
        "failed": "failed",
        "error": "error",
        "xfailed": "xfailed",
        "xpassed": "xpassed",
        "skipped": "skipped",
    }

    for outcome_key, display in outcome_map.items():
        for report in terminalreporter.stats.get(outcome_key, []):
            names = getattr(report, "_marker_names", set())
            for marker in names:
                if marker in marker_counts:
                    marker_counts[marker][display] = (
                        marker_counts[marker].get(display, 0) + 1
                    )

    # Build the table only for markers that have at least one test.
    rows: list[tuple[str, str]] = []
    for marker in _SUMMARY_MARKERS:
        counts = marker_counts[marker]
        if not counts:
            continue
        parts = []
        for col in ["passed", "xfailed", "failed", "error", "skipped", "xpassed"]:
            n = counts.get(col, 0)
            if n:
                parts.append(f"{n} {col}")
        rows.append((marker, ", ".join(parts)))

    if not rows:
        return

    terminalreporter.write_sep("=", "marker summary")
    width = max(len(name) for name, _ in rows)
    for name, summary in rows:
        terminalreporter.write_line(f"  {name:<{width}}  {summary}")
