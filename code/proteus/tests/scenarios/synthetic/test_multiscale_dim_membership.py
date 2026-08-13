"""Baseline tests: multi-scale features, dim chains, and soft membership.

The paper's headline object is a *scale-space cluster hierarchy*: a coarse
blurred feature at some average intrinsic dimension resolves, at finer
scales, into sub-features whose intrinsic dimension is at or below the
parent's — and points carry *fuzzy membership* along the root-to-leaf path
(SI S7). Neither the hierarchical-Gaussian scene (multi-level but
dimension-flat) nor the manifold zoo (dimension-diverse but single-level)
exercises that conjunction. These tests provide the minimal baseline.

Fixture: "rods-in-sheets-in-slab" — a three-level density hierarchy in R^3
whose levels have strictly ordered intrinsic dimensions:

* level 2 (finest / highest density): four 1-D rods, two lying on each sheet;
* level 1: two parallel 2-D sheets (each contains its two rods);
* level 0 (coarsest): one 3-D slab feature — the two sheets plus diffuse
  slab tissue filling the gap between them — against sparse outer tissue.

What passes today (baseline, via the level-set sweep of
``cd_level_set_probe`` and the validated ``estimate_d_final_mle``):
the C-D cluster tree contains all three levels with correct memberships,
and the measured per-cluster mean intrinsic dim is non-increasing under
refinement. The generators also already carry graded ground-truth
membership (``fade_weight``), which the hard ``lambda >= 0.5`` labels
throw away — locked here so the fixture is ready for the algorithm-side
membership tests.

What is ``@awaiting``: algorithm-side canonical membership (SI S7.2) and
multiscale membership trajectories (SI S7.4) — only the dataclasses exist
in ``proteus.types`` — and end-to-end recursion recovering the three-level
hierarchy (blocked on the OPEN_ISSUES #44 level-set layer).
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from proteus.intrinsic_dim import estimate_d_final_mle
from tests.harness.markers import awaiting
from tests.datasets.synthetic.faded_density import GaussianFadedComponent
from tests.scenarios.synthetic.cd_level_set_probe import cd_sweep

pytestmark = [pytest.mark.scenario, pytest.mark.synthetic]

# Noise must sit below the along-rod sample spacing (4/350 ~ 0.011), or the
# local MLE dimension reads the noise ball instead of the rod (measured:
# noise 0.02 inflates rod dim to ~2.4; noise 0.005 reads ~1.3-1.5).
_NOISE = 0.005


def make_rods_sheets_slab(seed: int = 0):
    """Three-level scene; returns (points, rod_labels, sheet_labels).

    Labels are per-level ground truth with -1 for non-members: rods 0..3 at
    the finest level, sheets 0..1 at the middle level (a sheet's rods belong
    to that sheet). Slab tissue and outer tissue are -1 at both levels.
    """
    rng = np.random.default_rng(seed)
    pts: list[np.ndarray] = []
    rod: list[int] = []
    sheet: list[int] = []
    for sheet_id, z in enumerate((-0.5, 0.5)):
        n_sheet = 900
        p = np.c_[
            rng.uniform(0, 4, n_sheet),
            rng.uniform(0, 4, n_sheet),
            np.full(n_sheet, z),
        ]
        pts.append(p + rng.normal(0, _NOISE, p.shape))
        rod.extend([-1] * n_sheet)
        sheet.extend([sheet_id] * n_sheet)
        for rod_slot, y in enumerate((1.0, 3.0)):
            n_rod = 350
            p = np.c_[
                rng.uniform(0, 4, n_rod),
                np.full(n_rod, y),
                np.full(n_rod, z),
            ]
            pts.append(p + rng.normal(0, _NOISE, p.shape))
            rod.extend([2 * sheet_id + rod_slot] * n_rod)
            sheet.extend([sheet_id] * n_rod)
    n_slab = 250
    pts.append(
        np.c_[
            rng.uniform(0, 4, n_slab),
            rng.uniform(0, 4, n_slab),
            rng.uniform(-0.5, 0.5, n_slab),
        ]
    )
    rod.extend([-1] * n_slab)
    sheet.extend([-1] * n_slab)
    n_outer = 150
    pts.append(rng.uniform(-1, 5, (n_outer, 3)))
    rod.extend([-1] * n_outer)
    sheet.extend([-1] * n_outer)
    return np.vstack(pts), np.array(rod), np.array(sheet)


def _best_level_at_k(levels, expect_k, y_true):
    """Best-ARI level with exactly ``expect_k`` clusters (ARI on members)."""
    best = None
    mask = y_true >= 0
    for r, k_found, lab in levels:
        if k_found != expect_k:
            continue
        ari = adjusted_rand_score(y_true[mask], lab[mask])
        if best is None or ari > best[0]:
            best = (ari, r, lab)
    return best


@pytest.fixture(scope="module")
def scene():
    points, rod, sheet = make_rods_sheets_slab(seed=0)
    levels = cd_sweep(points, k=8, alpha=1.0, n_levels=120, min_size=20)
    return points, rod, sheet, levels


def test_level_set_tree_contains_three_level_hierarchy(scene):
    """One density tree holds rods (K=4), sheets (K=2), and the slab (K=1)."""
    _, rod, sheet, levels = scene

    best_rods = _best_level_at_k(levels, 4, rod)
    assert best_rods is not None, "no level with K=4 (rods) in the tree"
    assert best_rods[0] >= 0.95

    best_sheets = _best_level_at_k(levels, 2, sheet)
    assert best_sheets is not None, "no level with K=2 (sheets) in the tree"
    assert best_sheets[0] >= 0.90

    # The scale ordering must be right: rods resolve at a finer level than
    # sheets, and a single coarse feature exists beyond the sheet level.
    assert best_rods[1] < best_sheets[1]
    assert any(
        k_found == 1 and r > best_sheets[1] for r, k_found, _ in levels
    ), "no coarse K=1 level beyond the sheet level"


def test_avg_intrinsic_dim_non_increasing_under_refinement(scene):
    """Per-cluster mean intrinsic dim: rods < sheets <= coarse slab.

    Children resolve at or below the parent's dimension. Bounds were
    calibrated on seeds 0-2 (rod clusters 1.29-1.45, sheet clusters
    1.71-1.80, coarse 1.73-1.78); the coarse comparison is weak because the
    coarse mean averages sheet and slab members.
    """
    points, rod, sheet, levels = scene
    dims = estimate_d_final_mle(points, ambient_dim=3)

    _, r_rods, lab_rods = _best_level_at_k(levels, 4, rod)
    _, r_sheets, lab_sheets = _best_level_at_k(levels, 2, sheet)
    lab_coarse = next(
        lab for r, k_found, lab in levels if k_found == 1 and r > r_sheets
    )

    def cluster_dims(lab):
        return [
            float(dims[lab == c].mean()) for c in sorted(set(lab[lab >= 0]))
        ]

    rod_dims = cluster_dims(lab_rods)
    sheet_dims = cluster_dims(lab_sheets)
    coarse_dims = cluster_dims(lab_coarse)
    assert len(rod_dims) == 4 and len(sheet_dims) == 2
    assert len(coarse_dims) == 1

    # Finest level reads ~1-D, middle level ~2-D leaning: strict separation.
    assert max(rod_dims) <= min(sheet_dims) - 0.1
    assert all(1.0 <= d <= 1.7 for d in rod_dims)
    assert all(1.5 <= d <= 2.3 for d in sheet_dims)
    # Children at or below the parent (small slack: coarse averages mix).
    assert max(rod_dims) <= max(sheet_dims) + 0.1
    assert max(sheet_dims) <= coarse_dims[0] + 0.15
    assert 1.5 <= coarse_dims[0] <= 2.5


def test_fade_weights_provide_graded_membership_ground_truth():
    """Generators carry soft-membership GT that hard labels discard.

    ``fade_weight`` is the per-component lambda used for labeling; the
    ``lambda >= 0.5`` threshold reduces it to hard labels. This locks the
    graded signal itself: interior points read ~1 for their own component,
    the overlap region reads strictly graded values for *both* components,
    and the profile is monotone along the between-centers axis.
    """
    dim = 3
    center_a = np.zeros(dim)
    center_b = np.zeros(dim)
    center_b[0] = 4.0
    comp_a = GaussianFadedComponent(
        center=center_a, sigma=1.0, transition_radius=1.5,
    )
    comp_b = GaussianFadedComponent(
        center=center_b, sigma=1.0, transition_radius=1.5,
    )

    t = np.linspace(0.0, 4.0, 41)
    line = np.zeros((t.size, dim))
    line[:, 0] = t
    w_a = comp_a.fade_weight(line)
    w_b = comp_b.fade_weight(line)

    # Interior: own membership ~1, sibling membership small.
    assert w_a[0] > 0.95 and w_b[0] < 0.05
    assert w_b[-1] > 0.95 and w_a[-1] < 0.05
    # Monotone gradation along the axis.
    assert np.all(np.diff(w_a) < 0.0) and np.all(np.diff(w_b) > 0.0)
    # Overlap midpoint: genuinely fuzzy in both components, and symmetric.
    mid = t.size // 2
    assert 0.1 < w_a[mid] < 0.9 and 0.1 < w_b[mid] < 0.9
    assert abs(w_a[mid] - w_b[mid]) < 1e-9


@awaiting("inference.membership", si="S7.2")
def test_boundary_point_membership_is_graded():
    """Canonical membership mu_C(x) grades boundary points across siblings.

    Intended: fit the rods-sheets-slab scene, build per-region
    ``GaussianSummary`` fits, and assert a point midway between the two
    rods of one sheet receives ``Membership.score`` in (0.1, 0.9) for both
    rod regions while a rod-core point reads > 0.9 for its own region.
    Only the S7 dataclasses exist in ``proteus.types``; no membership
    computation is implemented yet.
    """
    pytest.fail("Not implemented")


@awaiting("inference.membership", si="S7.4")
def test_multiscale_membership_trajectory_root_to_leaf():
    """MembershipTrajectory spans the slab -> sheet -> rod hierarchy.

    Intended: for a rod-core point, ``MembershipTrajectory.path`` has depth
    3 (slab, its sheet, its rod) with every score > 0.5; for a slab-tissue
    point the trajectory terminates at the coarse region with graded
    (sub-0.5) scores at finer levels rather than a hard assignment.
    """
    pytest.fail("Not implemented")


@awaiting("stage1.level_set.branch_extraction", si="S2.6")
def test_recursion_recovers_three_level_hierarchy():
    """End-to-end recursion should emit the 3-level dim-decreasing tree.

    Intended (OPEN_ISSUES #44 level-set layer): ``run_recursive_discovery``
    on the rods-sheets-slab scene yields a root slab feature that splits
    into the two sheets and then into the four rods (background tier
    excluded), with per-level sample ARI matching the baseline
    ``cd_sweep`` results above and per-feature mean intrinsic dim
    non-increasing along every root-to-leaf path.
    """
    pytest.fail("Not implemented")
