# Proteus Implementation

This folder is the clean implementation home for the current Proteus Paper 1 specification.

The implementation should follow:

- `docs/Proteus/paper_1_foundational/paper.tex`
- `docs/Proteus/paper_1_foundational/SI.tex`

The previous prototype has been copied to `code/legacy/proteus_v1/` for reference. Treat it as stale: useful for visual diagnostics and early Stage 1 experiments, but not as the canonical implementation.

## Install

```bash
# Core dependencies only:
pip install -e .

# With test dependencies:
pip install -e ".[test]"

# With real-data benchmark dependencies:
pip install -e ".[real_data]"

# CPU-only torch (if you don't need GPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
```

## Running Tests

```bash
# Default: runs all tests except real-data benchmarks
pytest

# Include real-data benchmarks (requires cached datasets):
pytest -m real_data

# All tests including real data:
pytest --real-data-cache=/path/to/cache

# Skip slow tests:
pytest -m "not slow"
```

**Stage 1 vs Stage 2 reconstruction:** scenario tests measure Stage 1 quality as mean / max min-distance from data to learned prototype positions (`tests/metrics/reconstruction.py`), normalized per dataset in `tests/harness/stage1_scenario_metrics.py`. Stage 2 will add density-based reconstruction for comparison on the same fixtures.

## Real-Data Cache

Real-data benchmarks download datasets on first use into `tests/.data_cache/` (gitignored). To use a pre-populated cache:

```bash
export PROTEUS_DATA_CACHE=/path/to/shared/cache
# or:
pytest --real-data-cache=/path/to/shared/cache
```

Set `PROTEUS_OFFLINE=1` to skip datasets that are not yet cached instead of downloading them.

## Initial Priorities

1. Build the SI-aligned fixed-scale Stage 1 scaffold.
2. Port and update the synthetic dataset and visualization diagnostics from `legacy/proteus_v1/tests/`.
3. Add the scale-response controller and evidence gate.
4. Add Stage 2 complex construction, torsion diagnostics, dual-flow density, and inference.
