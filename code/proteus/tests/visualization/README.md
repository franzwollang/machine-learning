# Visualization Diagnostics

The legacy visualization diagnostics in `code/legacy/proteus_v1/tests/` are the reference material for this folder.

Useful files to port first:

- `test_visualize_hierarchy_comparison.py`
- `test_gaussian_efficiency.py`

When ported, keep generated files under `tests/artifacts/` and avoid relying on the legacy `ProteusAlgorithm` API.
