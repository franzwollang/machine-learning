"""Frozen level-set diagnostic suite (OPEN_ISSUES #44).

Do **not** retune ``n``, ARI, or coverage to make a path pass.  New scenes
go in ``density_valleys.py`` and are added here with their own frozen
defaults.  Thresholds below are the 2026-08-13 fitted-sweep cuts.
"""

from __future__ import annotations

# Uniform / one-feature nulls
CIRCLE_N = 1500
SWISS_N = 2000

# Split scenes
HIERARCHY_N = 600
HIERARCHY_K = 3
HIERARCHY_MIN_ARI = 0.50
HIERARCHY_MIN_COVERAGE = 0.95

TORI_N_PER = 4000
TORI_K = 2
TORI_MIN_ARI = 0.95
TORI_MIN_COVERAGE = 0.95
TORI_MAX_NODES_FITTED = 768

NESTED_N_PER = 3000
NESTED_K = 2
NESTED_MIN_ARI = 0.70
NESTED_MIN_COVERAGE = 0.60
NESTED_MAX_NODES_FITTED = 1536

# Connected-support valley scenes (added 2026-09-09; not yet in the
# fitted 25/25 matrix). Parameters are frozen at introduction.
BIMODAL_CIRCLE_N = 1500
BIMODAL_CIRCLE_KAPPA = 3.0
BIMODAL_CIRCLE_K = 2

TWO_GAUSSIANS_N = 800
TWO_GAUSSIANS_SIGMA = 0.25
TWO_GAUSSIANS_WEAK_SEP = 2.5
TWO_GAUSSIANS_CLEAR_SEP = 6.0

# Lone-uniform root nulls (added 2026-09-09, #48 gate test)
LONE_TORUS_N_PER = TORI_N_PER
LONE_SHELL_N_PER = NESTED_N_PER
LONE_GAUSS2D_N = 800
LONE_GAUSS2D_SIGMA = 0.25
LONE_GAUSS4D_N = 800
LONE_TISSUE_RADIUS = 1.0

# Widened null geometries for φ-ceiling envelope (A3-T2 / #48)
WIDEN_NULL_N = 800
UNIFORM_DISC_RADIUS = 1.0
UNIFORM_CUBE_HALF = 1.0
LONE_GAUSS3D_SIGMA = 0.25
SCURVE_SHEET_NOISE = 0.0
FILLED_BALL_RADIUS = 1.0
# Existing-null n×k grid for the widen protocol
WIDEN_N_GRID: tuple[int, ...] = (400, 800, 1600)
WIDEN_K_DEFAULT = 8
WIDEN_K_GRID: tuple[int, ...] = (8, 16)
