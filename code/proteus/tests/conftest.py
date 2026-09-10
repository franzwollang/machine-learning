"""Test configuration for the rebuilt Proteus implementation."""

from __future__ import annotations

import os
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

# OPEN_ISSUES #46: seed sweeps / hollow-prepass calibration in
# test_recursion.py are simulations, not routine unit tests.
_SLOW_NAME_PARTS = (
    "harness",
    "ari",
    "denser",
    "youden",
    "multiseed",
    "a4_primary",
    "soft_capacity",
    "soft_x",
    "soft_frac",
    "soft_keep",
    "proposed_h0",
    "bridge_mass",
    "finer_research_circle",
    "finer_research_swiss",
    "finer_research_nested",
    "finer_research_persist",
    "hierarchical_gaussian_recursion",
    "persistence_gate_circle",
    "persistence_gate_hierarchy",
    "hollow_persist",
    "hollow_gabriel",
    "hollow_recovery",
    "multi_tau_hollow",
)

# OPEN_ISSUES #46: whole-module simulations that blow the default Stage-1
# slice (~45 min). Director backlog may later split these into slow-marked
# submodules with a small unmarked smoke set.
_SLOW_STAGE1_MODULES = frozenset(
    {
        "test_scale_search_persistence.py",
    }
)

# Markers that opt a test out of the unmarked call-budget guard (#46).
_BUDGET_EXEMPT_MARKERS = frozenset({"slow", "real_data", "benchmark"})

# Declared default: any unmarked test whose call phase exceeds this fails
# the default suite instead of silently expanding runtime.
_DEFAULT_UNMARKED_CALL_BUDGET_SECONDS = 60.0


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "unmarked_test_budget_seconds",
        "Fail unmarked (not slow/real_data/benchmark) tests whose call "
        "phase exceeds this many seconds. 0 disables. OPEN_ISSUES #46.",
        default=str(int(_DEFAULT_UNMARKED_CALL_BUDGET_SECONDS)),
    )


def unmarked_call_budget_seconds(config: pytest.Config | None = None) -> float | None:
    """Return the unmarked call-phase budget in seconds, or None if disabled.

    Precedence: ``PROTEUS_UNMARKED_TEST_BUDGET_SECONDS`` env (``0`` disables),
    then pytest.ini ``unmarked_test_budget_seconds``, then the #46 default.
    """

    raw_env = os.environ.get("PROTEUS_UNMARKED_TEST_BUDGET_SECONDS")
    if raw_env is not None:
        value = float(raw_env)
        return None if value <= 0 else value
    if config is not None:
        raw_ini = str(config.getini("unmarked_test_budget_seconds")).strip()
        if raw_ini:
            value = float(raw_ini)
            return None if value <= 0 else value
    return _DEFAULT_UNMARKED_CALL_BUDGET_SECONDS


def _is_budget_exempt(item: pytest.Item) -> bool:
    return any(m.name in _BUDGET_EXEMPT_MARKERS for m in item.iter_markers())


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
            if top == "stage1" and rel_parts[-1] in _SLOW_STAGE1_MODULES:
                markers.add("slow")
            if top == "stage1" and rel_parts[-1] == "test_recursion.py":
                name = item.name.lower()
                if any(part in name for part in _SLOW_NAME_PARTS):
                    markers.add("slow")

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
# Marker-grouped terminal summary + #46 unmarked runtime guard
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
    "slow",
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

    # OPEN_ISSUES #46: fail newly misclassified multi-minute simulations in
    # the default suite instead of silently expanding wall time.
    if call.when != "call" or _is_budget_exempt(item):
        return
    budget = unmarked_call_budget_seconds(item.config)
    if budget is None or report.duration <= budget:
        return
    report.outcome = "failed"
    budget_txt = f"{budget:.0f}" if budget >= 10 else f"{budget:g}"
    report.longrepr = (
        f"OPEN_ISSUES #46: unmarked test exceeded call budget "
        f"({report.duration:.1f}s > {budget_txt}s). "
        f"Mark @pytest.mark.slow (or real_data/benchmark) if this is a "
        f"simulation; set PROTEUS_UNMARKED_TEST_BUDGET_SECONDS=0 to disable."
    )


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
