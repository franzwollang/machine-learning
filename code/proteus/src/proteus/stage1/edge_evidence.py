"""Empty-region (hollow-edge) evidence for Stage 1 (OPEN_ISSUES #44).

Batch hollowness for a lifted edge ``(i, j)`` with endpoints ``x_i, x_j`` and
length ``L`` (theory note ``reference/empty_region_evidence_and_scale.md``):

- ``n_mid`` = data count in ball of radius ``r = mid_radius_frac * L`` about
  the midpoint;
- ``n_end`` = mean of the same counts about ``x_i`` and ``x_j``;
- ``H = n_mid / (n_end + eps)``.

Within-support edges have ``H = O(1)``; bridges over a void have ``H ≈ 0``.
At low endpoint mass, fall back to the Gabriel empty-diameter test (cut when
the open diameter ball contains no data).  Proposal-path defaults; the cut
threshold ``h_0`` is acceptance-path and needs Poisson-null calibration
before any awaiting flip (S14.3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-9


@dataclass(frozen=True)
class HollowEdgeConfig:
    """Operational defaults for batch hollow-edge scoring / pruning.

    ``mid_radius_frac`` and ``h0`` are proposal-path / operational until a
    Poisson lens-null calibration lands (OPEN_ISSUES #44).  Probe A2-T27 found
    joint nested+tori major-CC recovery on seed 0 near
    ``mid_radius_frac=0.35``, ``h0=0.35`` (note's ``L/4`` alone false-hollows
    when ``n_end≈0``); multi-seed fragile — do not flip awaiting.

    A2-T30 audit (adapted nested/tori scaffolds): at ``mid_radius_frac=0.35``
    mid-balls are typically smaller than the node→data gap, so ``H≈0`` on
    *both* cross-shell and intra-shell lifted edges (non-discriminative).
    Around ``mid_radius_frac=0.5`` cross vs intra ``H`` separates on nested
    lifted edges, but hollow-pruning still fails as a cut-set (redundant
    Hebbian paths) and fixed-tau ``K=2`` majors have sample ARI≈chance —
    do **not** treat major-CC count as recovery.  Gabriel fallback at low
    ``n_end`` amplifies spurious cuts.  Keep flag default-off.

    A2-T30 multi-tau scan + A4 sheet null (q01≈0.57 > h0=0.35): default
    ``H-or-Gabriel`` yields spurious majors=2 at probe taus (nested@0.27,
    tori@0.5) with sample ARI≈chance, driven by Gabriel at low ``n_end``.
    ``require_gabriel_and_h=True`` (cut iff ``H < h0`` ∧ Gabriel-empty)
    suppresses those spurious K=2 hits on the probe grid while keeping
    ``prefer_hollow_edge_prepass`` default-off.  Raising ``min_end_count``
    alone *increases* Gabriel usage; prefer conjunction or
    ``gabriel_fallback=False`` with a calibrated ``h0`` / mid_frac.

    A2-T33 / A4-T24 primary ROC export (sheet FPR≈0, bridge TPR≈0.9):
    ``mid_radius_frac=0.5``, ``h0=0.7``, ``gabriel_fallback=False``,
    ``min_end_count=0.5`` — see :func:`a4_roc_primary_config`.  Sheet-null
    safe ≠ nested cut-set / sample-ARI recovery; keep default-off.

    A2-T34: ``mst_critical_only=True`` restricts cuts to hollow edges that
    also lie on a Euclidean MST of the lifted graph (conservative
    capacity/bridge proxy).  Contrast vs H-only and Gabriel∧H; default off.

    A2 capacity/flow follow-on: ``bridge_critical_only=True`` intersects
    hollow cuts with *graph-theoretic bridges* of the lifted undirected
    graph (edges whose removal increases the CC count).  Bridges ⊆ every
    spanning tree, so this is a stricter true cut-set than MST-critical;
    default off.  Mutually independent of ``mst_critical_only`` (both may
    apply as successive intersections).

    A2-T37 soft-capacity: ``soft_capacity_only=True`` intersects hollow
    cuts with edges whose Brandes betweenness is at least
    ``soft_capacity_frac * max(betweenness)`` (operational default
    ``0.25``).  Continuous capacity/flow proxy between hard bridges and
    unrestricted hollow; default off. Independent of MST/bridge flags
    (successive intersections when combined).

    A2-T39 follow-on: ``soft_capacity_method`` selects the score —
    ``"betweenness"`` (default Brandes) or ``"bridge_mass"`` (min-cut
    mass on bridges: ``min(|comp_u|,|comp_v|)`` after removing a bridge;
    non-bridges score 0).  Operational / proposal-path; default method
    remains betweenness.

    A2-T40 follow-on: ``soft_capacity_frac`` sweep (nested@0.27 / tori@0.5
    under A4 primary+soft betweenness) — see
    :data:`SOFT_CAPACITY_FRAC_SWEEP_*` exports.  Soft×persist-agree leaf
    harness stays uniform-safe and unrecovered on nested/tori.  Defaults
    remain off; sheet-null / collapse ≠ sample-ARI recovery.

    A2-T41: soft×``require_gabriel_and_h`` conjunction (successive
    intersections) — see :data:`SOFT_X_GABRIEL_CONJ_*`.  Soft alone keeps
    tori@0.5 chance-ARI K=2; conj alone and soft×conj collapse both
    nested@0.27 and tori@0.5 to ≤1 major (still not sample-ARI recovery).
    Flags remain default-off.

    A2-T42: multi-seed ``soft_capacity_frac`` sweep (seeds 0..2) — see
    :data:`SOFT_CAPACITY_FRAC_MULTISEED_*`.  Nested collapses across
    seeds; tori chance-ARI K=2 is seed-fragile.  Defaults off.

    A2-T43: proposed Youden / Poisson-LR ``h0`` calibration from the
    sheet-null export + mid=0.5 adversarial ROC — see
    :data:`PROPOSED_H0_CALIBRATION_*` / :func:`proposed_h0_calibrated_config`.
    Proposed only; never the HollowEdgeConfig / RecursionConfig default.

    A2-T44-followon: soft×proposed ``h0`` combo + denser scaffold under
    proposed ``h0`` — see :data:`SOFT_X_PROPOSED_H0_*` /
    :data:`DENSER_PROPOSED_H0_*`. Soft×youden collapses nested like soft
    alone but keeps tori chance-ARI K=2; denser soft×youden collapses
    both to ≤1. Still not sample-ARI recovery; defaults off.

    A2-T44: multi-seed soft×Youden ``h0≈0.73`` (seeds 0..2) — see
    :data:`SOFT_X_YOUDEN_MULTISEED_*`. Soft×youden is seed-fragile
    (seed0: nested≤1 / tori K=2; seed1: soft *inflates* nested K=2
    ARI≈0.08 while youden alone ≤1; seed2: both soft collapses). Still
    not sample-ARI recovery; defaults off.

    A2-T45: denser-scaffold × ``proposed_h0_calibrated_config`` ARI —
    see :data:`DENSER_PROPOSED_H0_*` (retag of follow-on denser table).

    A2-T46: soft×poisson_lr vs Youden vs A4 majors+ARI contrast — see
    :data:`SOFT_H0_METHOD_CONTRAST_*` / :func:`format_soft_h0_method_contrast_table`.
    Under soft, h0∈{0.7,0.73,0.76} is near-null; soft drives the pattern.

    A2-T47: soft_frac × Youden seed1 nested-inflate mechanism — see
    :data:`SOFT_FRAC_X_YOUDEN_SEED_INFLATE_*`. Seed1 inflate is
    frac-windowed (soft_frac∈{0.1,0.25,0.5} → nested K=2 ARI≈0.05–0.08;
    frac≥0.75 collapses ≤1); seed0/2 never inflate. Defaults off.

    A2-T48 / A2-T49: denser soft×Youden multi-seed + h0-only contrast —
    see :data:`DENSER_SOFT_X_YOUDEN_MULTISEED_*`. Seed0 youden alone keeps
    tori chance-ARI K=2; soft×* and seeds1–2 collapse both ≤1. Seed1
    baseline inflate does **not** reproduce on denser. Defaults off.

    A2-T50: denser soft_frac × seed1 inflate window — see
    :data:`DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_*`. Across soft_frac∈
    {0.1,0.25,0.5,0.75,0.9} denser seed1 never inflates (both ≤1);
    denser kills the baseline frac-window. Seed0 soft_0.1 still keeps
    tori chance-ARI K=2; soft≥0.25 collapses. Defaults off.

    A2-T51: bridge_mass vs betweenness soft×Youden seed1 inflate —
    see :data:`BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_*`. Betweenness seed1
    inflate is method-specific; ``bridge_mass`` never inflates nested
    on seed1 across the frac window. Defaults off.

    A2-T52: soft×Youden at operational scale-search ``tau*`` (not fixed
    probe tau) — see :data:`SOFT_X_YOUDEN_TAU_STAR_*`. Seed1 probe-tau
    soft inflate is **absent** at ``tau*``; seed0 tori keeps chance-ARI
    K≥2 under both modes. Still not sample-ARI recovery; defaults off.

    A2-T53: denser × bridge_mass soft×Youden seed1 inflate — see
    :data:`DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_*`. On denser scaffolds
    both betweenness and bridge_mass never inflate seed1 (method contrast
    from baseline is denser-killed). Defaults off.

    A2-T54: soft×persist_agree at operational ``tau*`` e2e leaf table —
    see :data:`SOFT_X_PERSIST_TAU_STAR_*`. Seed1 nested K=2 chance-ARI
    survives soft×persist; majors-absent (T52) ≠ e2e leaf recovery.
    Defaults off.

    A2-T55: denser soft×Youden seed0 tori ARI window — see
    :data:`DENSER_SOFT_SEED0_TORI_ARI_WINDOW_*`. Fine soft_frac grid on
    denser scaffolds: soft_frac≤0.12 keeps tori K=2 chance-ARI≈0.16–0.18;
    soft≥0.15 collapses to 1 (tighter than T50's soft≥0.25 coarse grid).
    Nested stays ≤1. Still not sample-ARI recovery; defaults off.

    A2-T56: denser soft seed0 tori window × bridge_mass — see
    :data:`DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_*`. Betweenness keep band
    (soft≤0.12) is **method-specific**: bridge_mass collapses tori to 1
    across soft_frac∈{0.05..0.25}. Defaults off.

    A2-T57: denser soft×persist_agree at operational ``tau*`` e2e — see
    :data:`DENSER_SOFT_X_PERSIST_TAU_STAR_*`. Denser kills baseline T54
    seed1 nested e2e inflate; denser-youden seed0 nested K=2 chance-ARI
    is killed by soft/persist; uniforms stay 1. Defaults off.

    A2-T58: soft×``require_gabriel_and_h`` at operational ``tau*`` e2e —
    see :data:`SOFT_X_GABRIEL_TAU_STAR_*`. Seed1 nested K=2 chance-ARI
    survives youden/soft/conj/soft×conj (contrast T41 fixed-tau majors
    collapse under conj); circle youden shatters, soft/conj keep
    uniforms at 1. Soft≠sample-ARI recovery; defaults off.

    A2-T59: denser soft seed0 keep-band under ``persist_agree`` (bet vs
    bridge_mass) — see :data:`DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_*`.
    T55 majors keep-band soft≤0.12 does **not** survive denser e2e
    soft/soft×persist for either method (all ≤1 leaf); youden alone
    keeps seed0 nested K=2 chance-ARI≈0.01. Defaults off.

    A2-T60-followon: denser soft×``require_gabriel_and_h`` at operational
    ``tau*`` e2e — see :data:`DENSER_SOFT_X_GABRIEL_TAU_STAR_*`. Denser
    kills baseline T58 seed1 nested e2e inflate; denser-youden seed0
    nested K=2 chance-ARI is killed by soft/conj; circle youden does
    not shatter on denser. Defaults off.

    A2-T61-followon: denser soft×gabriel×persist compose at operational
    ``tau*`` e2e — see :data:`DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_*`.
    Triple compose does not unlock beyond T57/T60 pairwise denser
    collapse; denser-youden seed0 nested K=2 chance-ARI≈0.01 killed by
    soft×conj / soft×persist / soft×conj×persist. Defaults off.

    A2-T61: non-denser soft keep-band × persist majors baseline — see
    :data:`SOFT_KEEP_BAND_X_PERSIST_MAJORS_*`. Baseline n=80/120 majors
    keep-band soft≤0.5 → tori K=2 chance-ARI≈0.26 (wider than denser
    T55 ≤0.12); soft≥0.75 collapses; soft×persist e2e all ≤1 — keep-band
    is majors-only. Defaults off.

    A2-T63: soft×gabriel×persist majors non-denser seed1 inflate window —
    see :data:`SOFT_X_GABRIEL_X_PERSIST_MAJORS_*`. Seed1 majors soft alone
    inflates nested K=2 chance-ARI≈0.08 (killed by gabriel conj); e2e
    seed1 nested K=2 chance-ARI≈0 survives soft×conj×persist (majors≠e2e).
    Defaults off.

    A2-T64-followon: denser soft keep-band × ``require_gabriel_and_h``
    majors — see :data:`DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_*`. T55
    denser majors keep-band soft≤0.12 → tori K=2 is killed by gabriel
    conj at majors; lean tau* e2e soft/soft×conj all ≤1 (youden alone
    keeps seed0 nested K=2 chance-ARI≈0.01). Defaults off.

    A2-T65-followon / A2-T65: denser soft keep-band × gabriel × persist
    e2e frac grid — see
    :data:`DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_*`. Across
    soft_frac∈{0.05,0.12,0.15,0.25}, soft×persist and soft×conj×persist
    collapse nested+tori to ≤1 (youden alone keeps seed0 nested K=2
    chance-ARI≈0.01). Defaults off.

    A2-T66: denser soft keep-band × gabriel multi-seed majors/e2e —
    see :data:`DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_*`. T55/T64
    denser majors keep-band soft≤0.12 → tori K=2 is seed0-only; seeds
    1–2 stay ≤1 at majors (incl. soft keep fracs); gabriel conj kills
    seed0 keep. Lean e2e: only seed0 youden nested K=2 chance-ARI≈0.01;
    soft/soft×conj and seeds 1–2 all ≤1. Defaults off.

    A2-T67: denser soft×gabriel×persist compose seed1 inflate window —
    see :data:`DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_*`. Denser
    kills T63 seed1 majors soft nested inflate and T63 seed1 e2e nested
    inflate; seed1 soft/soft×conj/soft×persist/soft×conj×persist all ≤1
    on majors+e2e. Only seed0 youden remains (majors tori K=2
    chance-ARI≈0.14; e2e nested K=2 chance-ARI≈0.01). Defaults off.

    A2-T68: denser soft keep-band × gabriel × persist multi-seed e2e —
    see :data:`DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_*`.
    Extends T65 seed0 persist frac grid across seeds 0..2 with lean
    keep fracs {0.05,0.12,0.15}: soft×persist / soft×conj×persist and
    seeds 1–2 youden all ≤1 nested+tori; only seed0 youden nested K=2
    chance-ARI≈0.01 remains (T65/T66/T67 singleton). Defaults off.

    A2-T69: denser soft keep-band × gabriel seed0-only keep ×
    soft×persist majors pin — see
    :data:`DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_*`.
    Seed0 denser majors soft≤0.12 → tori K=2 chance-ARI (gabriel kills;
    soft≥0.15 collapses); soft×persist / soft×conj×persist e2e at lean
    keep fracs all ≤1 nested+tori — keep-band is majors-only under
    persist compose (T61 denser+gabriel pin). Defaults off.
    """

    mid_radius_frac: float = 0.35
    h0: float = 0.35
    min_end_count: float = 0.5
    gabriel_fallback: bool = True
    require_gabriel_and_h: bool = False
    mst_critical_only: bool = False
    bridge_critical_only: bool = False
    soft_capacity_only: bool = False
    soft_capacity_frac: float = 0.25
    soft_capacity_method: str = "betweenness"
    eps: float = _EPS


# A4-T24 → A2-T33 primary HollowEdgeConfig (flag-gated; do not flip defaults).
A4_PRIMARY_MID_RADIUS_FRAC: float = 0.5
A4_PRIMARY_H0: float = 0.7
A4_PRIMARY_MIN_END_COUNT: float = 0.5
A4_PRIMARY_GABRIEL_FALLBACK: bool = False


# ---------------------------------------------------------------------------
# Poisson-null ``h0`` calibration export (A2-T38 → A3/A4 SI sync)
# ---------------------------------------------------------------------------
# Snapshot of sheet-null H quantiles (connected density-gradient sheet,
# seed=0, n=49 edges) from ``tests.scenarios.synthetic.hollow_edge_nulls``.
# Under a locally homogeneous Poisson field ``E[H]≈1``; the lower tail is
# the practical null for choosing acceptance-path ``h0`` without fixture
# seed-tuning.  Live harness may re-check within tolerance; do not flip
# RecursionConfig / HollowEdgeConfig defaults from these numbers alone.

POISSON_NULL_SHEET_SEED: int = 0
POISSON_NULL_SHEET_N_EDGES: int = 49

# mid_radius_frac → {quantile_label: H}
POISSON_NULL_SHEET_H_QUANTILES: dict[float, dict[str, float]] = {
    0.25: {
        "q0.01": 0.15,
        "q0.05": 0.4087,
        "q0.1": 0.6925,
        "q0.25": 0.8077,
        "q0.5": 1.0,
        "mean_h": 1.0177,
    },
    0.35: {
        "q0.01": 0.4265,
        "q0.05": 0.6596,
        "q0.1": 0.7438,
        "q0.25": 0.8913,
        "q0.5": 1.0164,
        "mean_h": 1.0328,
    },
    0.5: {
        "q0.01": 0.7571,
        "q0.05": 0.8164,
        "q0.1": 0.8621,
        "q0.25": 0.9362,
        "q0.5": 1.0087,
        "mean_h": 1.0177,
    },
}

# A4 recommend_hollow_edge_configs primary (sheet FPR≈0, bridge TPR≈0.9):
# h0=0.7 ≤ sheet q01≈0.82 at mid=0.5 with gabriel off.  SI should note
# sheet-null safe ≠ nested/tori sample-ARI recovery.
POISSON_NULL_PRIMARY_MID: float = A4_PRIMARY_MID_RADIUS_FRAC
POISSON_NULL_PRIMARY_H0: float = A4_PRIMARY_H0
POISSON_NULL_PRIMARY_SHEET_Q01: float = 0.82
POISSON_NULL_SI_NOTE: str = (
    "Poisson-null sheet H: mid=0.25/0.35/0.5 q01≈0.15/0.43/0.76 (meanH≈1); "
    "A4 primary mid=0.5 h0=0.7≤q01≈0.82 gabriel=False (sheet FPR≈0, bridge "
    "TPR≈0.9). Sheet-null safe ≠ nested cut-set / sample-ARI recovery; "
    "keep HollowEdgeConfig / RecursionConfig defaults off."
)


def format_poisson_null_h0_table(
    quantiles: dict[float, dict[str, float]] | None = None,
) -> str:
    """Compact TSV of sheet-null H quantiles for A3/A4 SI handoff (A2-T38)."""

    qmap = POISSON_NULL_SHEET_H_QUANTILES if quantiles is None else quantiles
    header = "mid\tq01\tq05\tq10\tq25\tq50\tmeanH"
    lines = [header]
    for mid in sorted(qmap):
        row = qmap[mid]
        lines.append(
            f"{mid:g}\t{row['q0.01']:.4f}\t{row['q0.05']:.4f}\t"
            f"{row['q0.1']:.4f}\t{row['q0.25']:.4f}\t{row['q0.5']:.4f}\t"
            f"{row['mean_h']:.4f}"
        )
    lines.append(
        f"# primary mid={POISSON_NULL_PRIMARY_MID:g} "
        f"h0={POISSON_NULL_PRIMARY_H0:g} "
        f"sheet_q01≈{POISSON_NULL_PRIMARY_SHEET_Q01:g} "
        f"gabriel={A4_PRIMARY_GABRIEL_FALLBACK}"
    )
    lines.append(f"# {POISSON_NULL_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Proposed Youden / Poisson-LR h0 calibration (A2-T43 → A3 SI)
# ---------------------------------------------------------------------------
# Derived from the frozen sheet-null export (mid=0.5) and the mid=0.5
# adversarial ROC (sheet negatives vs empty-gap bridges).  Proposed
# acceptance-path candidates only — never wired as HollowEdgeConfig /
# RecursionConfig defaults.  Sheet/bridge ROC safe ≠ nested/tori
# sample-ARI recovery.

PROPOSED_H0_CALIBRATION_MID: float = A4_PRIMARY_MID_RADIUS_FRAC  # 0.5
PROPOSED_H0_OPERATIONAL: float = 0.35  # current HollowEdgeConfig default
# Max Youden J = TPR−FPR on mid=0.5 pooled adversarial ROC (~h0≈0.734).
PROPOSED_H0_YOUDEN: float = 0.73
PROPOSED_H0_YOUDEN_TPR: float = 1.0
PROPOSED_H0_YOUDEN_FPR: float = 0.029
# A4 primary (conservative Youden-family): sheet FPR≈0, bridge TPR≈0.9.
PROPOSED_H0_YOUDEN_A4: float = A4_PRIMARY_H0  # 0.7
# Poisson-LR / null lower-tail: sheet export q01 at mid=0.5 (snap).
PROPOSED_H0_POISSON_LR: float = 0.76
PROPOSED_H0_POISSON_LR_SHEET_Q01: float = POISSON_NULL_SHEET_H_QUANTILES[0.5]["q0.01"]

# method → (h0, sheet_role_note)
PROPOSED_H0_CALIBRATION_TABLE: dict[str, tuple[float, str]] = {
    "operational": (PROPOSED_H0_OPERATIONAL, "HollowEdgeConfig default; weak Youden"),
    "youden": (PROPOSED_H0_YOUDEN, "max TPR-FPR mid=0.5 adversarial ROC"),
    "youden_a4": (PROPOSED_H0_YOUDEN_A4, "A4 primary FPR≈0 TPR≈0.9"),
    "poisson_lr": (PROPOSED_H0_POISSON_LR, "sheet-null q01 lower-tail snap"),
}

PROPOSED_H0_CALIBRATION_SI_NOTE: str = (
    "A2-T43 proposed h0 calibration (mid=0.5, gabriel=False): Youden max "
    "J≈0.97 at h0≈0.73 (TPR=1 FPR≈0.03); A4 primary h0=0.7 (FPR≈0 TPR≈0.9); "
    f"Poisson-LR sheet q01≈{PROPOSED_H0_POISSON_LR_SHEET_Q01:.2f}→h0=0.76; "
    "operational h0=0.35 weak on Youden. Proposed only — defaults off; "
    "sheet/bridge ROC ≠ nested/tori sample-ARI recovery; no awaiting flip."
)


def proposed_h0_calibrated_config(
    method: str = "youden_a4",
    **overrides: object,
) -> HollowEdgeConfig:
    """Proposed calibrated HollowEdgeConfig (A2-T43); never the default.

    ``method`` is one of ``operational`` / ``youden`` / ``youden_a4`` /
    ``poisson_lr``.  Base knobs match A4 primary (mid=0.5, gabriel off)
    except ``operational``, which keeps mid=0.35 + gabriel fallback.
    """

    if method not in PROPOSED_H0_CALIBRATION_TABLE:
        raise ValueError(
            f"unknown h0 calibration method {method!r}; "
            f"expected one of {sorted(PROPOSED_H0_CALIBRATION_TABLE)}"
        )
    h0, _ = PROPOSED_H0_CALIBRATION_TABLE[method]
    if method == "operational":
        base = dict(
            mid_radius_frac=0.35,
            h0=float(h0),
            min_end_count=0.5,
            gabriel_fallback=True,
            require_gabriel_and_h=False,
            soft_capacity_only=False,
        )
    else:
        base = dict(
            mid_radius_frac=PROPOSED_H0_CALIBRATION_MID,
            h0=float(h0),
            min_end_count=A4_PRIMARY_MIN_END_COUNT,
            gabriel_fallback=A4_PRIMARY_GABRIEL_FALLBACK,
            require_gabriel_and_h=False,
            soft_capacity_only=False,
        )
    base.update(overrides)
    return HollowEdgeConfig(**base)  # type: ignore[arg-type]


def format_proposed_h0_calibration_table() -> str:
    """TSV export of proposed Youden / Poisson-LR h0 candidates (A2-T43)."""

    lines = [
        "# proposed Youden / Poisson-LR h0 calibration (mid=0.5 adversarial)",
        "method\th0\tnote",
    ]
    for method, (h0, note) in PROPOSED_H0_CALIBRATION_TABLE.items():
        lines.append(f"{method}\t{h0:g}\t{note}")
    lines.append(
        f"# youden_raw TPR={PROPOSED_H0_YOUDEN_TPR:g} "
        f"FPR={PROPOSED_H0_YOUDEN_FPR:g}; "
        f"poisson_lr_sheet_q01={PROPOSED_H0_POISSON_LR_SHEET_Q01:.4f}"
    )
    lines.append(f"# {PROPOSED_H0_CALIBRATION_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft-capacity frac sweep export (A2-T40 → A3/A4 SI sync)
# ---------------------------------------------------------------------------
# Snapshot majors under A4 primary + soft_capacity_only (betweenness) on
# baseline scaffolds (seed=0, max_nodes=64, k=8).  Higher ``frac`` is a
# stricter high-betweenness gate → fewer hollow cuts → more edges kept.
# Nested collapses spurious A4 K=2 across the operational frac grid;
# tori retains chance-ARI K=2 until frac≳0.9.  Not acceptance-path.

SOFT_CAPACITY_FRAC_SWEEP_NESTED_TAU: float = 0.27
SOFT_CAPACITY_FRAC_SWEEP_TORI_TAU: float = 0.5
SOFT_CAPACITY_FRAC_SWEEP_METHOD: str = "betweenness"

# frac → majors (nested@0.27 A4+soft; A4-alone majors=2 ARI≈0.12)
SOFT_CAPACITY_FRAC_SWEEP_NESTED_MAJORS: dict[float, int] = {
    0.1: 1,
    0.25: 1,
    0.5: 1,
    0.75: 1,
    0.9: 1,
}

# frac → (majors, sample_ARI_or_None); ARI only when majors≥2
SOFT_CAPACITY_FRAC_SWEEP_TORI: dict[float, tuple[int, float | None]] = {
    0.1: (2, 0.26),
    0.25: (2, 0.26),
    0.5: (2, 0.26),
    0.9: (1, None),
}

SOFT_CAPACITY_FRAC_SWEEP_SI_NOTE: str = (
    "A2-T40 soft_capacity_frac sweep (A4 primary+betweenness): nested@0.27 "
    "collapses majors≤1 for frac∈{0.1,0.25,0.5,0.75,0.9}; tori@0.5 keeps "
    "K=2 ARI≈0.26 until frac=0.9→1 major. Soft×persist_agree leaf harness "
    "uniform-safe; nested/tori unrecovered. Defaults off; no awaiting flip."
)


def format_soft_capacity_frac_sweep_table() -> str:
    """TSV export of soft-capacity frac sweep for A3/A4 SI sync (A2-T40)."""

    lines = [
        "# soft_capacity_frac sweep (A4 primary + soft betweenness)",
        f"# method={SOFT_CAPACITY_FRAC_SWEEP_METHOD}",
        "dataset\ttau\tfrac\tmajors\tsample_ari",
    ]
    for frac, maj in SOFT_CAPACITY_FRAC_SWEEP_NESTED_MAJORS.items():
        lines.append(
            f"nested\t{SOFT_CAPACITY_FRAC_SWEEP_NESTED_TAU:g}\t"
            f"{frac:g}\t{maj}\t"
        )
    for frac, (maj, ari) in SOFT_CAPACITY_FRAC_SWEEP_TORI.items():
        ari_s = "" if ari is None else f"{ari:.2f}"
        lines.append(
            f"tori\t{SOFT_CAPACITY_FRAC_SWEEP_TORI_TAU:g}\t"
            f"{frac:g}\t{maj}\t{ari_s}"
        )
    lines.append(f"# {SOFT_CAPACITY_FRAC_SWEEP_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft × Gabriel∧H conjunction export (A2-T41 → A3/A4 SI sync)
# ---------------------------------------------------------------------------
# Snapshot majors+sample-ARI under A4 primary with soft_capacity_only,
# require_gabriel_and_h, and their successive intersection (soft×conj).
# Soft alone collapses nested spurious K=2 but keeps tori chance-ARI K=2;
# conj alone and soft×conj collapse both scaffolds to ≤1 major.  Collapse
# ≠ sample-ARI recovery; keep flags default-off.

SOFT_X_GABRIEL_CONJ_NESTED_TAU: float = 0.27
SOFT_X_GABRIEL_CONJ_TORI_TAU: float = 0.5
SOFT_X_GABRIEL_CONJ_SOFT_FRAC: float = 0.25
SOFT_X_GABRIEL_CONJ_SOFT_METHOD: str = "betweenness"

# mode → (nested_majors, nested_ari, tori_majors, tori_ari)
SOFT_X_GABRIEL_CONJ_TABLE: dict[str, tuple[int, float | None, int, float | None]] = {
    "a4": (2, 0.12, 2, 0.26),
    "soft": (1, None, 2, 0.26),
    "conj": (1, None, 1, None),
    "soft_x_conj": (1, None, 1, None),
}

SOFT_X_GABRIEL_CONJ_SI_NOTE: str = (
    "A2-T41 soft×require_gabriel_and_h (A4 primary+betweenness frac=0.25): "
    "soft alone nested@0.27→≤1 major, tori@0.5 keeps K=2 ARI≈0.26; "
    "conj and soft×conj collapse nested+tori to ≤1 major. Collapse ≠ "
    "sample-ARI recovery; HollowEdgeConfig / RecursionConfig defaults off; "
    "no awaiting flip."
)


def format_soft_x_gabriel_conj_table() -> str:
    """TSV export of soft×Gabriel∧H conjunction majors+ARI (A2-T41)."""

    lines = [
        "# soft × require_gabriel_and_h conjunction (A4 primary)",
        f"# soft_frac={SOFT_X_GABRIEL_CONJ_SOFT_FRAC:g} "
        f"method={SOFT_X_GABRIEL_CONJ_SOFT_METHOD}",
        "mode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for mode, (nm, na, tm, ta) in SOFT_X_GABRIEL_CONJ_TABLE.items():
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"{mode}\tnested\t{SOFT_X_GABRIEL_CONJ_NESTED_TAU:g}\t{nm}\t{na_s}"
        )
        lines.append(
            f"{mode}\ttori\t{SOFT_X_GABRIEL_CONJ_TORI_TAU:g}\t{tm}\t{ta_s}"
        )
    lines.append(f"# {SOFT_X_GABRIEL_CONJ_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-seed soft-capacity frac sweep (A2-T42 → A3/A4 SI sync)
# ---------------------------------------------------------------------------
# Extends A2-T40 seed-0 frac sweep across dataset seeds 0..2 (scaffold RNG
# matched to dataset seed).  Nested stays ≤1 major under soft for all
# seeds/fracs; tori chance-ARI K=2 is seed-fragile (seed0 until frac=0.9;
# seed2 only at frac=0.1; seed1 already ≤1).  No sample-ARI recovery.

SOFT_CAPACITY_FRAC_MULTISEED_SEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_CAPACITY_FRAC_MULTISEED_FRACS: tuple[float, ...] = (0.1, 0.25, 0.5, 0.9)
SOFT_CAPACITY_FRAC_MULTISEED_NESTED_TAU: float = 0.27
SOFT_CAPACITY_FRAC_MULTISEED_TORI_TAU: float = 0.5
SOFT_CAPACITY_FRAC_MULTISEED_METHOD: str = "betweenness"

# seed → frac → nested majors
SOFT_CAPACITY_FRAC_MULTISEED_NESTED: dict[int, dict[float, int]] = {
    0: {0.1: 1, 0.25: 1, 0.5: 1, 0.9: 1},
    1: {0.1: 1, 0.25: 1, 0.5: 1, 0.9: 1},
    2: {0.1: 1, 0.25: 1, 0.5: 1, 0.9: 1},
}

# seed → frac → (tori majors, sample_ARI_or_None)
SOFT_CAPACITY_FRAC_MULTISEED_TORI: dict[int, dict[float, tuple[int, float | None]]] = {
    0: {
        0.1: (2, 0.26),
        0.25: (2, 0.26),
        0.5: (2, 0.26),
        0.9: (1, None),
    },
    1: {
        0.1: (1, None),
        0.25: (1, None),
        0.5: (1, None),
        0.9: (1, None),
    },
    2: {
        0.1: (2, 0.22),
        0.25: (1, None),
        0.5: (1, None),
        0.9: (1, None),
    },
}

SOFT_CAPACITY_FRAC_MULTISEED_SI_NOTE: str = (
    "A2-T42 multi-seed soft_capacity_frac sweep (A4 primary+betweenness, "
    "seeds 0..2): nested@0.27 majors≤1 all seeds/fracs; tori@0.5 chance-ARI "
    "K=2 is seed-fragile (seed0 until frac=0.9; seed2 only frac=0.1 ARI≈0.22; "
    "seed1 already ≤1). No sample-ARI recovery; defaults off; no awaiting flip."
)


def format_soft_capacity_frac_multiseed_table() -> str:
    """TSV export of multi-seed soft-capacity frac sweep (A2-T42)."""

    lines = [
        "# multi-seed soft_capacity_frac sweep (A4 primary + soft betweenness)",
        f"# method={SOFT_CAPACITY_FRAC_MULTISEED_METHOD} "
        f"seeds={list(SOFT_CAPACITY_FRAC_MULTISEED_SEEDS)}",
        "seed\tdataset\ttau\tfrac\tmajors\tsample_ari",
    ]
    for seed in SOFT_CAPACITY_FRAC_MULTISEED_SEEDS:
        for frac, maj in SOFT_CAPACITY_FRAC_MULTISEED_NESTED[seed].items():
            lines.append(
                f"{seed}\tnested\t{SOFT_CAPACITY_FRAC_MULTISEED_NESTED_TAU:g}\t"
                f"{frac:g}\t{maj}\t"
            )
        for frac, (maj, ari) in SOFT_CAPACITY_FRAC_MULTISEED_TORI[seed].items():
            ari_s = "" if ari is None else f"{ari:.2f}"
            lines.append(
                f"{seed}\ttori\t{SOFT_CAPACITY_FRAC_MULTISEED_TORI_TAU:g}\t"
                f"{frac:g}\t{maj}\t{ari_s}"
            )
    lines.append(f"# {SOFT_CAPACITY_FRAC_MULTISEED_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft × proposed h0 combo (A2-T44-followon → A3 SI)
# ---------------------------------------------------------------------------
# Proposed Youden/Poisson-LR h0 alone mirrors A4 primary chance-ARI K=2 on
# baseline nested@0.27 / tori@0.5. Soft×proposed collapses nested spurious
# K=2 (like soft alone) but retains tori chance-ARI K=2. Soft alone is the
# collapse driver; calibrated h0 does not unlock sample-ARI recovery.

SOFT_X_PROPOSED_H0_NESTED_TAU: float = 0.27
SOFT_X_PROPOSED_H0_TORI_TAU: float = 0.5
SOFT_X_PROPOSED_H0_SOFT_FRAC: float = 0.25
SOFT_X_PROPOSED_H0_SOFT_METHOD: str = "betweenness"

# mode → (nested_majors, nested_ari, tori_majors, tori_ari)
SOFT_X_PROPOSED_H0_TABLE: dict[str, tuple[int, float | None, int, float | None]] = {
    "youden": (2, 0.12, 2, 0.26),
    "youden_a4": (2, 0.12, 2, 0.26),
    "poisson_lr": (2, 0.08, 2, 0.26),
    "soft_x_youden": (1, None, 2, 0.26),
    "soft_x_youden_a4": (1, None, 2, 0.26),
    "soft_x_poisson_lr": (1, None, 2, 0.26),
}

SOFT_X_PROPOSED_H0_SI_NOTE: str = (
    "A2-T44-followon / A2-T46 soft×proposed h0 (mid=0.5 gabriel=False + "
    "betweenness frac=0.25): youden/youden_a4/poisson_lr alone keep "
    "nested+tori chance-ARI K=2 (ARI≈0.12/0.26; poisson nested≈0.08); "
    "soft×youden / soft×youden_a4 / soft×poisson_lr all collapse nested≤1 "
    "but retain tori K=2 ARI≈0.26 — h0 method contrast is near-null under "
    "soft. Soft drives collapse; calibrated h0 ≠ sample-ARI recovery; "
    "defaults off; no awaiting flip."
)

# Focused soft×h0 method contrast keys (A2-T46).
SOFT_H0_METHOD_CONTRAST_MODES: tuple[str, ...] = (
    "youden",
    "youden_a4",
    "poisson_lr",
    "soft_x_youden",
    "soft_x_youden_a4",
    "soft_x_poisson_lr",
)

SOFT_H0_METHOD_CONTRAST_SI_NOTE: str = (
    "A2-T46 soft×poisson_lr(h0=0.76) vs Youden(0.73) vs A4(0.7) majors+ARI "
    "contrast (seed0 baseline): alone all chance-ARI K=2; soft×* identical "
    "collapse pattern (nested≤1, tori K=2 ARI≈0.26). Soft dominates; h0 "
    "choice among {0.7,0.73,0.76} does not change majors/ARI under soft. "
    "Defaults off; no awaiting flip."
)


def format_soft_x_proposed_h0_table() -> str:
    """TSV export of soft×proposed Youden/Poisson-LR h0 combo (A2-T44-followon)."""

    lines = [
        "# soft × proposed Youden/Poisson-LR h0 (baseline scaffold)",
        f"# soft_frac={SOFT_X_PROPOSED_H0_SOFT_FRAC:g} "
        f"method={SOFT_X_PROPOSED_H0_SOFT_METHOD}",
        "mode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for mode, (nm, na, tm, ta) in SOFT_X_PROPOSED_H0_TABLE.items():
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"{mode}\tnested\t{SOFT_X_PROPOSED_H0_NESTED_TAU:g}\t{nm}\t{na_s}"
        )
        lines.append(
            f"{mode}\ttori\t{SOFT_X_PROPOSED_H0_TORI_TAU:g}\t{tm}\t{ta_s}"
        )
    lines.append(f"# {SOFT_X_PROPOSED_H0_SI_NOTE}")
    return "\n".join(lines)


def format_soft_h0_method_contrast_table() -> str:
    """TSV export of soft×poisson_lr vs Youden vs A4 contrast (A2-T46)."""

    lines = [
        "# soft × h0 method contrast (Youden 0.73 / A4 0.7 / poisson_lr 0.76)",
        f"# soft_frac={SOFT_X_PROPOSED_H0_SOFT_FRAC:g} "
        f"method={SOFT_X_PROPOSED_H0_SOFT_METHOD}",
        "mode\th0\tdataset\ttau\tmajors\tsample_ari",
    ]
    h0_of = {
        "youden": PROPOSED_H0_YOUDEN,
        "youden_a4": PROPOSED_H0_YOUDEN_A4,
        "poisson_lr": PROPOSED_H0_POISSON_LR,
        "soft_x_youden": PROPOSED_H0_YOUDEN,
        "soft_x_youden_a4": PROPOSED_H0_YOUDEN_A4,
        "soft_x_poisson_lr": PROPOSED_H0_POISSON_LR,
    }
    for mode in SOFT_H0_METHOD_CONTRAST_MODES:
        nm, na, tm, ta = SOFT_X_PROPOSED_H0_TABLE[mode]
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        h0 = h0_of[mode]
        lines.append(
            f"{mode}\t{h0:g}\tnested\t{SOFT_X_PROPOSED_H0_NESTED_TAU:g}\t"
            f"{nm}\t{na_s}"
        )
        lines.append(
            f"{mode}\t{h0:g}\ttori\t{SOFT_X_PROPOSED_H0_TORI_TAU:g}\t"
            f"{tm}\t{ta_s}"
        )
    lines.append(f"# {SOFT_H0_METHOD_CONTRAST_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Multi-seed soft × Youden h0 (A2-T44 → A3 SI)
# ---------------------------------------------------------------------------
# Extends seed-0 soft×youden across dataset seeds 0..2 (scaffold RNG matched).
# Soft×youden is seed-fragile: seed0 collapses nested / keeps tori chance-ARI
# K=2; seed1 soft *inflates* nested spurious K=2 (ARI≈0.08) while youden alone
# was ≤1; seed2 soft collapses both.  No sample-ARI recovery.

SOFT_X_YOUDEN_MULTISEED_SEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_X_YOUDEN_MULTISEED_NESTED_TAU: float = 0.27
SOFT_X_YOUDEN_MULTISEED_TORI_TAU: float = 0.5
SOFT_X_YOUDEN_MULTISEED_SOFT_FRAC: float = 0.25
SOFT_X_YOUDEN_MULTISEED_SOFT_METHOD: str = "betweenness"
SOFT_X_YOUDEN_MULTISEED_H0: float = PROPOSED_H0_YOUDEN

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
SOFT_X_YOUDEN_MULTISEED_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.12, 2, 0.26),
        "soft_x_youden": (1, None, 2, 0.26),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_x_youden": (2, 0.08, 1, None),
    },
    2: {
        "youden": (1, None, 2, 0.22),
        "soft_x_youden": (1, None, 1, None),
    },
}

SOFT_X_YOUDEN_MULTISEED_SI_NOTE: str = (
    "A2-T44 multi-seed soft×Youden h0≈0.73 (mid=0.5 gabriel=False + "
    "betweenness frac=0.25, seeds 0..2): seed0 soft collapses nested≤1 / "
    "keeps tori K=2 ARI≈0.26; seed1 soft *inflates* nested K=2 ARI≈0.08 "
    "while youden alone ≤1; seed2 soft collapses both ≤1. Soft×youden "
    "seed-fragile; calibrated h0 ≠ sample-ARI recovery; defaults off; "
    "no awaiting flip."
)


def format_soft_x_youden_multiseed_table() -> str:
    """TSV export of multi-seed soft×Youden h0 majors+ARI (A2-T44)."""

    lines = [
        "# multi-seed soft × Youden h0≈0.73 (baseline scaffold)",
        f"# h0={SOFT_X_YOUDEN_MULTISEED_H0:g} "
        f"soft_frac={SOFT_X_YOUDEN_MULTISEED_SOFT_FRAC:g} "
        f"method={SOFT_X_YOUDEN_MULTISEED_SOFT_METHOD} "
        f"seeds={list(SOFT_X_YOUDEN_MULTISEED_SEEDS)}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for seed in SOFT_X_YOUDEN_MULTISEED_SEEDS:
        for mode, (nm, na, tm, ta) in SOFT_X_YOUDEN_MULTISEED_TABLE[seed].items():
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"{seed}\t{mode}\tnested\t"
                f"{SOFT_X_YOUDEN_MULTISEED_NESTED_TAU:g}\t{nm}\t{na_s}"
            )
            lines.append(
                f"{seed}\t{mode}\ttori\t"
                f"{SOFT_X_YOUDEN_MULTISEED_TORI_TAU:g}\t{tm}\t{ta_s}"
            )
    lines.append(f"# {SOFT_X_YOUDEN_MULTISEED_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser scaffold × proposed h0 (A2-T44-followon / A2-T45 → A3 SI)
# ---------------------------------------------------------------------------
# denser: n_per_sphere=160 / n_per_torus=240, max_nodes=128, k=8, seed=0.
# Youden alone collapses nested≤1 but keeps tori chance-ARI K=2 (ARI≈0.14);
# soft×youden / soft×poisson_lr collapse both scaffolds to ≤1 major.

DENSER_PROPOSED_H0_NESTED_N: int = 160
DENSER_PROPOSED_H0_TORI_N: int = 240
DENSER_PROPOSED_H0_MAX_NODES: int = 128
DENSER_PROPOSED_H0_NESTED_TAU: float = 0.27
DENSER_PROPOSED_H0_TORI_TAU: float = 0.5
DENSER_PROPOSED_H0_SOFT_FRAC: float = 0.25

# mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_PROPOSED_H0_TABLE: dict[str, tuple[int, float | None, int, float | None]] = {
    "youden": (1, None, 2, 0.14),
    "soft_x_youden": (1, None, 1, None),
    "soft_x_poisson_lr": (1, None, 1, None),
}

DENSER_PROPOSED_H0_SI_NOTE: str = (
    "A2-T45 denser×proposed h0 (n=160/240, max_nodes=128, mid=0.5 "
    "gabriel=False): youden alone nested@0.27→≤1, tori@0.5 K=2 ARI≈0.14; "
    "soft×youden / soft×poisson_lr collapse both ≤1 major. Collapse ≠ "
    "sample-ARI recovery; defaults off; no awaiting flip."
)


def format_denser_proposed_h0_table() -> str:
    """TSV export of denser scaffold × proposed h0 (A2-T45)."""

    lines = [
        "# denser scaffold × proposed Youden/Poisson-LR h0",
        f"# nested_n={DENSER_PROPOSED_H0_NESTED_N} "
        f"tori_n={DENSER_PROPOSED_H0_TORI_N} "
        f"max_nodes={DENSER_PROPOSED_H0_MAX_NODES} "
        f"soft_frac={DENSER_PROPOSED_H0_SOFT_FRAC:g}",
        "mode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for mode, (nm, na, tm, ta) in DENSER_PROPOSED_H0_TABLE.items():
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"{mode}\tnested\t{DENSER_PROPOSED_H0_NESTED_TAU:g}\t{nm}\t{na_s}"
        )
        lines.append(
            f"{mode}\ttori\t{DENSER_PROPOSED_H0_TORI_TAU:g}\t{tm}\t{ta_s}"
        )
    lines.append(f"# {DENSER_PROPOSED_H0_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft_frac × Youden seed1 nested-inflate mechanism (A2-T47 → A3 SI)
# ---------------------------------------------------------------------------
# Baseline scaffold (n=80/120, max_nodes=64). Soft×Youden nested inflate on
# seed1 is frac-windowed: soft_frac∈{0.1,0.25,0.5} yields nested K=2
# (ARI≈0.05–0.08) while youden alone ≤1 and frac≥0.75 collapses ≤1.
# Seed0 soft collapses nested at all fracs (tori K=2 until frac≥0.75);
# seed2 never inflates. Selective soft cuts ≠ sample-ARI recovery.

SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_FRAC_X_YOUDEN_SEED_INFLATE_FRACS: tuple[float, ...] = (
    0.1, 0.25, 0.5, 0.75, 0.9,
)
SOFT_FRAC_X_YOUDEN_SEED_INFLATE_NESTED_TAU: float = 0.27
SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TORI_TAU: float = 0.5
SOFT_FRAC_X_YOUDEN_SEED_INFLATE_METHOD: str = "betweenness"
SOFT_FRAC_X_YOUDEN_SEED_INFLATE_H0: float = PROPOSED_H0_YOUDEN

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
# mode "youden" = h0-only; "soft_{frac}" = soft_capacity_only at that frac
SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.12, 2, 0.26),
        "soft_0.1": (1, None, 2, 0.26),
        "soft_0.25": (1, None, 2, 0.26),
        "soft_0.5": (1, None, 2, 0.26),
        "soft_0.75": (1, None, 1, None),
        "soft_0.9": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_0.1": (2, 0.05, 1, None),
        "soft_0.25": (2, 0.08, 1, None),
        "soft_0.5": (2, 0.08, 1, None),
        "soft_0.75": (1, None, 1, None),
        "soft_0.9": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 2, 0.22),
        "soft_0.1": (1, None, 2, 0.22),
        "soft_0.25": (1, None, 1, None),
        "soft_0.5": (1, None, 1, None),
        "soft_0.75": (1, None, 1, None),
        "soft_0.9": (1, None, 1, None),
    },
}

SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SI_NOTE: str = (
    "A2-T47 soft_frac×Youden h0≈0.73 seed1 nested-inflate mechanism "
    "(mid=0.5 gabriel=False + betweenness, seeds 0..2): seed1 inflate is "
    "frac-windowed — soft_frac∈{0.1,0.25,0.5} → nested K=2 ARI≈0.05–0.08 "
    "while youden alone ≤1; frac≥0.75 collapses ≤1. Seed0 soft collapses "
    "nested at all fracs (tori K=2 until frac≥0.75); seed2 never inflates. "
    "Selective soft cuts ≠ sample-ARI recovery; defaults off; no awaiting "
    "flip."
)


def format_soft_frac_x_youden_seed_inflate_table() -> str:
    """TSV export of soft_frac×Youden seed1 inflate mechanism (A2-T47)."""

    lines = [
        "# soft_frac × Youden seed1 nested-inflate mechanism (baseline)",
        f"# h0={SOFT_FRAC_X_YOUDEN_SEED_INFLATE_H0:g} "
        f"method={SOFT_FRAC_X_YOUDEN_SEED_INFLATE_METHOD} "
        f"seeds={list(SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SEEDS)} "
        f"fracs={list(SOFT_FRAC_X_YOUDEN_SEED_INFLATE_FRACS)}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for seed in SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SEEDS:
        for mode, (nm, na, tm, ta) in (
            SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"{seed}\t{mode}\tnested\t"
                f"{SOFT_FRAC_X_YOUDEN_SEED_INFLATE_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"{seed}\t{mode}\ttori\t"
                f"{SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
    lines.append(f"# {SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft×Youden multi-seed + h0-only contrast (A2-T48 / A2-T49 → A3)
# ---------------------------------------------------------------------------
# denser: n_per_sphere=160 / n_per_torus=240, max_nodes=128, k=8, seeds 0..2.
# Seed0 youden alone: nested≤1 / tori K=2 ARI≈0.14; soft×youden collapses
# both ≤1. Seeds 1–2: youden and soft×youden both ≤1 on nested+tori.
# Baseline seed1 soft inflate does **not** reproduce on denser scaffold.

DENSER_SOFT_X_YOUDEN_MULTISEED_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_SOFT_X_YOUDEN_MULTISEED_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_X_YOUDEN_MULTISEED_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_X_YOUDEN_MULTISEED_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES
DENSER_SOFT_X_YOUDEN_MULTISEED_NESTED_TAU: float = 0.27
DENSER_SOFT_X_YOUDEN_MULTISEED_TORI_TAU: float = 0.5
DENSER_SOFT_X_YOUDEN_MULTISEED_SOFT_FRAC: float = 0.25
DENSER_SOFT_X_YOUDEN_MULTISEED_H0: float = PROPOSED_H0_YOUDEN

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_X_YOUDEN_MULTISEED_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 2, 0.14),
        "soft_x_youden": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_x_youden": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_x_youden": (1, None, 1, None),
    },
}

DENSER_SOFT_X_YOUDEN_MULTISEED_SI_NOTE: str = (
    "A2-T48/T49 denser soft×Youden multi-seed + h0-only contrast "
    "(n=160/240, max_nodes=128, mid=0.5 gabriel=False, seeds 0..2): "
    "seed0 youden alone nested≤1 / tori K=2 ARI≈0.14; soft×youden "
    "collapses both ≤1. Seeds1–2: h0-only and soft×* both ≤1 on "
    "nested+tori — baseline seed1 soft inflate does not reproduce on "
    "denser. Collapse ≠ sample-ARI recovery; defaults off; no awaiting "
    "flip."
)


def format_denser_soft_x_youden_multiseed_table() -> str:
    """TSV export of denser soft×Youden multi-seed / h0-only (A2-T48/T49)."""

    lines = [
        "# denser soft × Youden multi-seed + h0-only contrast",
        f"# nested_n={DENSER_SOFT_X_YOUDEN_MULTISEED_NESTED_N} "
        f"tori_n={DENSER_SOFT_X_YOUDEN_MULTISEED_TORI_N} "
        f"max_nodes={DENSER_SOFT_X_YOUDEN_MULTISEED_MAX_NODES} "
        f"h0={DENSER_SOFT_X_YOUDEN_MULTISEED_H0:g} "
        f"soft_frac={DENSER_SOFT_X_YOUDEN_MULTISEED_SOFT_FRAC:g} "
        f"seeds={list(DENSER_SOFT_X_YOUDEN_MULTISEED_SEEDS)}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for seed in DENSER_SOFT_X_YOUDEN_MULTISEED_SEEDS:
        for mode, (nm, na, tm, ta) in (
            DENSER_SOFT_X_YOUDEN_MULTISEED_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"{seed}\t{mode}\tnested\t"
                f"{DENSER_SOFT_X_YOUDEN_MULTISEED_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"{seed}\t{mode}\ttori\t"
                f"{DENSER_SOFT_X_YOUDEN_MULTISEED_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
    lines.append(f"# {DENSER_SOFT_X_YOUDEN_MULTISEED_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft_frac × Youden seed1 inflate window (A2-T50 → A3 SI)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128. Baseline seed1 frac-window inflate does
# **not** reproduce: soft_frac∈{0.1..0.9} → nested+tori ≤1 on seed1.
# Seed0: youden / soft_0.1 keep tori K=2 chance-ARI; soft≥0.25 collapses.
# Seed2: all modes ≤1. Denser kills inflate; still not sample-ARI recovery.

DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_FRACS: tuple[float, ...] = (
    0.1, 0.25, 0.5, 0.75, 0.9,
)
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_NESTED_TAU: float = 0.27
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TORI_TAU: float = 0.5
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_METHOD: str = "betweenness"
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_H0: float = PROPOSED_H0_YOUDEN

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 2, 0.14),
        "soft_0.1": (1, None, 2, 0.18),
        "soft_0.25": (1, None, 1, None),
        "soft_0.5": (1, None, 1, None),
        "soft_0.75": (1, None, 1, None),
        "soft_0.9": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_0.1": (1, None, 1, None),
        "soft_0.25": (1, None, 1, None),
        "soft_0.5": (1, None, 1, None),
        "soft_0.75": (1, None, 1, None),
        "soft_0.9": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_0.1": (1, None, 1, None),
        "soft_0.25": (1, None, 1, None),
        "soft_0.5": (1, None, 1, None),
        "soft_0.75": (1, None, 1, None),
        "soft_0.9": (1, None, 1, None),
    },
}

DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SI_NOTE: str = (
    "A2-T50 denser soft_frac×Youden seed1 inflate window "
    "(n=160/240, max_nodes=128, mid=0.5 gabriel=False + betweenness): "
    "across soft_frac∈{0.1,0.25,0.5,0.75,0.9} denser seed1 never "
    "inflates (nested+tori ≤1) — denser kills the baseline frac-window. "
    "Seed0 soft_0.1 keeps tori K=2 ARI≈0.18; soft≥0.25 collapses; seed2 "
    "all ≤1. Collapse ≠ sample-ARI recovery; defaults off; no awaiting "
    "flip."
)


def format_denser_soft_frac_x_youden_seed_inflate_table() -> str:
    """TSV export of denser soft_frac×Youden seed1 inflate window (A2-T50)."""

    lines = [
        "# denser soft_frac × Youden seed1 inflate window",
        f"# nested_n={DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_NESTED_N} "
        f"tori_n={DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TORI_N} "
        f"max_nodes={DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_MAX_NODES} "
        f"h0={DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_H0:g} "
        f"method={DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_METHOD} "
        f"seeds={list(DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SEEDS)} "
        f"fracs={list(DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_FRACS)}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for seed in DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SEEDS:
        for mode, (nm, na, tm, ta) in (
            DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"{seed}\t{mode}\tnested\t"
                f"{DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"{seed}\t{mode}\ttori\t"
                f"{DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
    lines.append(f"# {DENSER_SOFT_FRAC_X_YOUDEN_SEED_INFLATE_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Bridge_mass vs betweenness soft×Youden seed1 inflate (A2-T51 → A3 SI)
# ---------------------------------------------------------------------------
# Baseline n=80/120, max_nodes=64, Youden h0≈0.73. Seed1 betweenness soft
# inflate (frac∈{0.1,0.25,0.5} → nested K=2 ARI≈0.05–0.08) does **not**
# reproduce under soft_capacity_method=bridge_mass (all fracs ≤1).
# Multi-seed@frac=0.25: seed0 soft_bet keeps tori K=2; soft_mass collapses
# both; seed2 both soft methods collapse. Method-specific ≠ recovery.

BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SEEDS: tuple[int, ...] = (0, 1, 2)
BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRACS: tuple[float, ...] = (
    0.1, 0.25, 0.5, 0.75, 0.9,
)
BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_TAU: float = 0.27
BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_TAU: float = 0.5
BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_H0: float = PROPOSED_H0_YOUDEN
BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRAC: float = 0.25

# Multi-seed@frac=0.25: mode → (nested_majors, nested_ari, tori_majors, tori_ari)
BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.12, 2, 0.26),
        "soft_betweenness": (1, None, 2, 0.26),
        "soft_bridge_mass": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_betweenness": (2, 0.08, 1, None),
        "soft_bridge_mass": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 2, 0.22),
        "soft_betweenness": (1, None, 1, None),
        "soft_bridge_mass": (1, None, 1, None),
    },
}

# Seed1 frac-window: method → frac → (nested_majors, nested_ari, tori_majors, tori_ari)
BRIDGE_MASS_X_YOUDEN_SEED1_FRAC_TABLE: dict[
    str, dict[float, tuple[int, float | None, int, float | None]]
] = {
    "betweenness": {
        0.1: (2, 0.05, 1, None),
        0.25: (2, 0.08, 1, None),
        0.5: (2, 0.08, 1, None),
        0.75: (1, None, 1, None),
        0.9: (1, None, 1, None),
    },
    "bridge_mass": {
        0.1: (1, None, 1, None),
        0.25: (1, None, 1, None),
        0.5: (1, None, 1, None),
        0.75: (1, None, 1, None),
        0.9: (1, None, 1, None),
    },
}

BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SI_NOTE: str = (
    "A2-T51 bridge_mass vs betweenness soft×Youden seed1 inflate "
    "(baseline n=80/120, mid=0.5 gabriel=False, h0≈0.73): betweenness "
    "seed1 inflate is method-specific (frac∈{0.1,0.25,0.5} → nested K=2 "
    "ARI≈0.05–0.08); bridge_mass never inflates seed1 across the frac "
    "window. Multi-seed@0.25: seed0 soft_bet keeps tori K=2 / soft_mass "
    "collapses; seed2 both soft ≤1. Method-specific ≠ sample-ARI "
    "recovery; defaults off; no awaiting flip."
)


def format_bridge_mass_x_youden_seed_inflate_table() -> str:
    """TSV export of bridge_mass vs betweenness seed1 inflate (A2-T51)."""

    lines = [
        "# bridge_mass vs betweenness soft × Youden seed1 inflate",
        f"# h0={BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_H0:g} "
        f"frac={BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRAC:g} "
        f"seeds={list(BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SEEDS)} "
        f"seed1_fracs={list(BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRACS)}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for seed in BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SEEDS:
        for mode, (nm, na, tm, ta) in (
            BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"{seed}\t{mode}\tnested\t"
                f"{BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"{seed}\t{mode}\ttori\t"
                f"{BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
    lines.append("seed\tmethod\tfrac\tdataset\ttau\tmajors\tsample_ari")
    for method, frac_table in BRIDGE_MASS_X_YOUDEN_SEED1_FRAC_TABLE.items():
        for frac, (nm, na, tm, ta) in frac_table.items():
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"1\t{method}\t{frac:g}\tnested\t"
                f"{BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"1\t{method}\t{frac:g}\ttori\t"
                f"{BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
    lines.append(f"# {BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft×Youden at operational scale-search tau* (A2-T52 → A3 SI)
# ---------------------------------------------------------------------------
# Baseline n=80/120, max_nodes=64. Scale-search lean grid max_grid_points=12
# (seed=42+dataset_seed); evaluate hollow majors+ARI at tau* (not fixed
# probe 0.27/0.5). Seed1 soft inflate @probe is absent at tau*; seed0 tori
# keeps chance-ARI K≥2 under youden (K=3 ARI≈0.22) and soft (K=2 ARI≈0.27).
# Operational tau* ≠ sample-ARI recovery; defaults off.

SOFT_X_YOUDEN_TAU_STAR_SEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_X_YOUDEN_TAU_STAR_SOFT_FRAC: float = 0.25
SOFT_X_YOUDEN_TAU_STAR_SOFT_METHOD: str = "betweenness"
SOFT_X_YOUDEN_TAU_STAR_H0: float = PROPOSED_H0_YOUDEN
SOFT_X_YOUDEN_TAU_STAR_MAX_GRID_POINTS: int = 12
SOFT_X_YOUDEN_TAU_STAR_SCALE_SEED_BASE: int = 42

# seed → (nested_tau_star, tori_tau_star) recorded under lean n_grid=12
SOFT_X_YOUDEN_TAU_STAR_VALUES: dict[int, tuple[float, float]] = {
    0: (0.5021607031056003, 0.5021607031056003),
    1: (0.5021607031056003, 1.0021583738168336),
    2: (1.0021583738168336, 1.0021583738168336),
}

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
SOFT_X_YOUDEN_TAU_STAR_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 3, 0.22),
        "soft_x_youden": (1, None, 2, 0.27),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_x_youden": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_x_youden": (1, None, 1, None),
    },
}

SOFT_X_YOUDEN_TAU_STAR_SI_NOTE: str = (
    "A2-T52 soft×Youden at operational scale-search tau* "
    "(baseline n=80/120, mid=0.5 gabriel=False, lean max_grid_points=12, "
    "scale_seed=42+dataset_seed): seed1 probe-tau soft inflate is absent "
    "at tau* (nested+tori ≤1); seed0 tori keeps chance-ARI K≥2 under "
    "youden (K=3 ARI≈0.22) and soft (K=2 ARI≈0.27); seeds1–2 both ≤1. "
    "Operational tau* ≠ sample-ARI recovery; defaults off; no awaiting "
    "flip."
)


def format_soft_x_youden_tau_star_table() -> str:
    """TSV export of soft×Youden at operational tau* (A2-T52)."""

    lines = [
        "# soft × Youden at operational scale-search tau*",
        f"# h0={SOFT_X_YOUDEN_TAU_STAR_H0:g} "
        f"soft_frac={SOFT_X_YOUDEN_TAU_STAR_SOFT_FRAC:g} "
        f"method={SOFT_X_YOUDEN_TAU_STAR_SOFT_METHOD} "
        f"max_grid_points={SOFT_X_YOUDEN_TAU_STAR_MAX_GRID_POINTS} "
        f"scale_seed_base={SOFT_X_YOUDEN_TAU_STAR_SCALE_SEED_BASE} "
        f"seeds={list(SOFT_X_YOUDEN_TAU_STAR_SEEDS)}",
        "seed\tmode\tdataset\ttau_star\tmajors\tsample_ari",
    ]
    for seed in SOFT_X_YOUDEN_TAU_STAR_SEEDS:
        n_tau, t_tau = SOFT_X_YOUDEN_TAU_STAR_VALUES[seed]
        for mode, (nm, na, tm, ta) in SOFT_X_YOUDEN_TAU_STAR_TABLE[seed].items():
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"{seed}\t{mode}\tnested\t{n_tau:.4f}\t{nm}\t{na_s}"
            )
            lines.append(
                f"{seed}\t{mode}\ttori\t{t_tau:.4f}\t{tm}\t{ta_s}"
            )
    lines.append(f"# {SOFT_X_YOUDEN_TAU_STAR_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser × bridge_mass soft×Youden seed1 inflate (A2-T53 → A3 SI)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128. Baseline betweenness seed1 inflate
# (T51) and denser-betweenness kill (T50) both imply denser×bridge_mass
# never inflates; method contrast is denser-killed. Seed0 youden keeps
# tori K=2 chance-ARI; both soft methods collapse. Not recovery.

DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRACS: tuple[float, ...] = (
    0.1, 0.25, 0.5, 0.75, 0.9,
)
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_TAU: float = 0.27
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_TAU: float = 0.5
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_H0: float = PROPOSED_H0_YOUDEN
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRAC: float = 0.25

# Multi-seed@frac=0.25: mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 2, 0.14),
        "soft_betweenness": (1, None, 1, None),
        "soft_bridge_mass": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_betweenness": (1, None, 1, None),
        "soft_bridge_mass": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_betweenness": (1, None, 1, None),
        "soft_bridge_mass": (1, None, 1, None),
    },
}

# Seed1 denser frac-window: method → frac → (...)
DENSER_BRIDGE_MASS_X_YOUDEN_SEED1_FRAC_TABLE: dict[
    str, dict[float, tuple[int, float | None, int, float | None]]
] = {
    "betweenness": {
        0.1: (1, None, 1, None),
        0.25: (1, None, 1, None),
        0.5: (1, None, 1, None),
        0.75: (1, None, 1, None),
        0.9: (1, None, 1, None),
    },
    "bridge_mass": {
        0.1: (1, None, 1, None),
        0.25: (1, None, 1, None),
        0.5: (1, None, 1, None),
        0.75: (1, None, 1, None),
        0.9: (1, None, 1, None),
    },
}

DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SI_NOTE: str = (
    "A2-T53 denser×bridge_mass soft×Youden seed1 inflate "
    "(n=160/240, max_nodes=128, mid=0.5 gabriel=False, h0≈0.73): on denser "
    "scaffolds both betweenness and bridge_mass never inflate seed1 across "
    "soft_frac∈{0.1..0.9} — denser kills the baseline betweenness method "
    "contrast (T51). Multi-seed@0.25: seed0 youden keeps tori K=2 ARI≈0.14; "
    "both soft methods collapse; seeds1–2 all ≤1. Denser-killed ≠ "
    "sample-ARI recovery; defaults off; no awaiting flip."
)


def format_denser_bridge_mass_x_youden_seed_inflate_table() -> str:
    """TSV export of denser×bridge_mass soft×Youden seed1 inflate (A2-T53)."""

    lines = [
        "# denser × bridge_mass soft × Youden seed1 inflate",
        f"# nested_n={DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_N} "
        f"tori_n={DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_N} "
        f"max_nodes={DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_MAX_NODES} "
        f"h0={DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_H0:g} "
        f"frac={DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRAC:g} "
        f"seeds={list(DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SEEDS)} "
        f"seed1_fracs={list(DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_FRACS)}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    for seed in DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SEEDS:
        for mode, (nm, na, tm, ta) in (
            DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"{seed}\t{mode}\tnested\t"
                f"{DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"{seed}\t{mode}\ttori\t"
                f"{DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
    lines.append("seed\tmethod\tfrac\tdataset\ttau\tmajors\tsample_ari")
    for method, frac_table in (
        DENSER_BRIDGE_MASS_X_YOUDEN_SEED1_FRAC_TABLE.items()
    ):
        for frac, (nm, na, tm, ta) in frac_table.items():
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"1\t{method}\t{frac:g}\tnested\t"
                f"{DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"1\t{method}\t{frac:g}\ttori\t"
                f"{DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
    lines.append(f"# {DENSER_BRIDGE_MASS_X_YOUDEN_SEED_INFLATE_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft×persist_agree at operational tau* e2e leaves (A2-T54 → A3 SI)
# ---------------------------------------------------------------------------
# Baseline n=80/120, Youden h0≈0.73, lean max_grid_points=12,
# scale_seed=42+dataset_seed. Recursive discovery leaf counts (not majors):
# seed1 nested K=2 chance-ARI≈0 survives youden/soft/persist/soft×persist;
# seeds0/2 + all tori stay 1 leaf. Circle youden alone shatters (2 leaves);
# soft/persist/soft×persist keep uniforms at 1. T52 majors-absent ≠ e2e
# leaf kill; soft×persist ≠ sample-ARI recovery; defaults off.

SOFT_X_PERSIST_TAU_STAR_SEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_X_PERSIST_TAU_STAR_SOFT_FRAC: float = 0.25
SOFT_X_PERSIST_TAU_STAR_SOFT_METHOD: str = "betweenness"
SOFT_X_PERSIST_TAU_STAR_H0: float = PROPOSED_H0_YOUDEN
SOFT_X_PERSIST_TAU_STAR_MAX_GRID_POINTS: int = 12
SOFT_X_PERSIST_TAU_STAR_SCALE_SEED_BASE: int = 42

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
SOFT_X_PERSIST_TAU_STAR_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
    },
    1: {
        "youden": (2, 0.0, 1, None),
        "soft": (2, 0.0, 1, None),
        "persist": (2, 0.0, 1, None),
        "soft_x_persist": (2, 0.0, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
    },
}

# Uniform leaf counts under the same lean tau* / Youden knobs (seed=0).
SOFT_X_PERSIST_TAU_STAR_UNIFORMS: dict[str, dict[str, int]] = {
    "circle": {
        "youden": 2,
        "soft": 1,
        "persist": 1,
        "soft_x_persist": 1,
    },
    "swiss": {
        "youden": 1,
        "soft": 1,
        "persist": 1,
        "soft_x_persist": 1,
    },
}

SOFT_X_PERSIST_TAU_STAR_SI_NOTE: str = (
    "A2-T54 soft×persist_agree at operational scale-search tau* e2e "
    "(baseline n=80/120, Youden h0≈0.73, mid=0.5 gabriel=False, lean "
    "max_grid_points=12, scale_seed=42+dataset_seed): seed1 nested K=2 "
    "chance-ARI≈0 survives youden/soft/persist/soft×persist — soft×persist "
    "does not kill e2e seed1 inflate (contrast T52 majors-absent at "
    "tau*); seeds0/2 + all tori stay 1 leaf. Circle youden alone "
    "shatters (2 leaves); soft/persist/soft×persist keep uniforms at 1. "
    "E2e leaf ≠ sample-ARI recovery; defaults off; no awaiting flip."
)


def format_soft_x_persist_tau_star_table() -> str:
    """TSV export of soft×persist_agree at operational tau* e2e (A2-T54)."""

    lines = [
        "# soft × persist_agree at operational scale-search tau* e2e leaves",
        f"# h0={SOFT_X_PERSIST_TAU_STAR_H0:g} "
        f"soft_frac={SOFT_X_PERSIST_TAU_STAR_SOFT_FRAC:g} "
        f"method={SOFT_X_PERSIST_TAU_STAR_SOFT_METHOD} "
        f"max_grid_points={SOFT_X_PERSIST_TAU_STAR_MAX_GRID_POINTS} "
        f"scale_seed_base={SOFT_X_PERSIST_TAU_STAR_SCALE_SEED_BASE} "
        f"seeds={list(SOFT_X_PERSIST_TAU_STAR_SEEDS)}",
        "seed\tmode\tdataset\tleaves\tsample_ari",
    ]
    for seed in SOFT_X_PERSIST_TAU_STAR_SEEDS:
        for mode, (nl, na, tl, ta) in SOFT_X_PERSIST_TAU_STAR_TABLE[seed].items():
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"{seed}\t{mode}\tnested\t{nl}\t{na_s}")
            lines.append(f"{seed}\t{mode}\ttori\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in SOFT_X_PERSIST_TAU_STAR_UNIFORMS.items():
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {SOFT_X_PERSIST_TAU_STAR_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft×Youden seed0 tori ARI window (A2-T55 → A3 SI)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, seed0 only. Fine soft_frac grid around
# the T50 soft_0.1 keep / soft≥0.25 collapse coarse boundary. Measured:
# soft_frac∈{0.05,0.08,0.10,0.12} → tori K=2 chance-ARI≈0.16–0.18;
# soft≥0.15 → collapse K=1. Nested ≤1 throughout. Tighter window than
# T50's 0.1-vs-0.25 coarse grid; still not sample-ARI recovery.

DENSER_SOFT_SEED0_TORI_ARI_WINDOW_SEED: int = 0
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_FRACS: tuple[float, ...] = (
    0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25,
)
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_NESTED_TAU: float = 0.27
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TORI_TAU: float = 0.5
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_METHOD: str = "betweenness"
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC: float = 0.12
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC: float = 0.15

# mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (1, None, 2, 0.14),
    "soft_0.05": (1, None, 2, 0.16),
    "soft_0.08": (1, None, 2, 0.16),
    "soft_0.1": (1, None, 2, 0.18),
    "soft_0.12": (1, None, 2, 0.18),
    "soft_0.15": (1, None, 1, None),
    "soft_0.18": (1, None, 1, None),
    "soft_0.2": (1, None, 1, None),
    "soft_0.25": (1, None, 1, None),
}

DENSER_SOFT_SEED0_TORI_ARI_WINDOW_SI_NOTE: str = (
    "A2-T55 denser soft×Youden seed0 tori ARI window "
    "(n=160/240, max_nodes=128, mid=0.5 gabriel=False + betweenness): "
    "fine soft_frac grid shows keep band soft_frac≤0.12 → tori K=2 "
    "chance-ARI≈0.16–0.18; soft≥0.15 collapses to 1 (tighter than T50 "
    "soft≥0.25 coarse claim). Nested ≤1 throughout. Chance-ARI ≠ "
    "sample-ARI recovery; defaults off; no awaiting flip."
)


def format_denser_soft_seed0_tori_ari_window_table() -> str:
    """TSV export of denser soft×Youden seed0 tori ARI window (A2-T55)."""

    lines = [
        "# denser soft × Youden seed0 tori ARI window",
        f"# nested_n={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_NESTED_N} "
        f"tori_n={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TORI_N} "
        f"max_nodes={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_MAX_NODES} "
        f"h0={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_H0:g} "
        f"method={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_METHOD} "
        f"seed={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_SEED} "
        f"fracs={list(DENSER_SOFT_SEED0_TORI_ARI_WINDOW_FRACS)} "
        f"keep_max={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC:g} "
        f"collapse_min={DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC:g}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    seed = DENSER_SOFT_SEED0_TORI_ARI_WINDOW_SEED
    for mode, (nm, na, tm, ta) in (
        DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"{seed}\t{mode}\tnested\t"
            f"{DENSER_SOFT_SEED0_TORI_ARI_WINDOW_NESTED_TAU:g}\t"
            f"{nm}\t{na_s}"
        )
        lines.append(
            f"{seed}\t{mode}\ttori\t"
            f"{DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TORI_TAU:g}\t"
            f"{tm}\t{ta_s}"
        )
    lines.append(f"# {DENSER_SOFT_SEED0_TORI_ARI_WINDOW_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft seed0 tori window × bridge_mass (A2-T56 → A3 SI)
# ---------------------------------------------------------------------------
# Same denser seed0 grid as T55. Betweenness keep band (soft≤0.12 → tori
# K=2) does **not** reproduce under soft_capacity_method=bridge_mass:
# soft_frac∈{0.05..0.25} → tori+nested ≤1. Keep band is method-specific.

DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_SEED: int = 0
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_FRACS: tuple[float, ...] = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_FRACS
)
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_NESTED_N: int = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_NESTED_N
)
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_TORI_N: int = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TORI_N
)
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_MAX_NODES: int = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_MAX_NODES
)
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_NESTED_TAU: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_NESTED_TAU
)
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_TORI_TAU: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_TORI_TAU
)
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_METHOD: str = "bridge_mass"

# mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (1, None, 2, 0.14),
    "soft_0.05": (1, None, 1, None),
    "soft_0.08": (1, None, 1, None),
    "soft_0.1": (1, None, 1, None),
    "soft_0.12": (1, None, 1, None),
    "soft_0.15": (1, None, 1, None),
    "soft_0.18": (1, None, 1, None),
    "soft_0.2": (1, None, 1, None),
    "soft_0.25": (1, None, 1, None),
}

DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_SI_NOTE: str = (
    "A2-T56 denser soft seed0 tori ARI window × bridge_mass "
    "(n=160/240, max_nodes=128, mid=0.5 gabriel=False): T55 betweenness "
    "keep band soft_frac≤0.12 (tori K=2 chance-ARI) is method-specific — "
    "bridge_mass collapses tori to 1 across soft_frac∈{0.05..0.25}. "
    "Nested ≤1. Soft≠sample-ARI recovery; defaults off; no awaiting flip."
)


def format_denser_soft_seed0_bridge_mass_window_table() -> str:
    """TSV export of denser soft seed0 window × bridge_mass (A2-T56)."""

    lines = [
        "# denser soft seed0 tori ARI window × bridge_mass",
        f"# nested_n={DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_NESTED_N} "
        f"tori_n={DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_TORI_N} "
        f"max_nodes={DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_MAX_NODES} "
        f"h0={DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_H0:g} "
        f"method={DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_METHOD} "
        f"seed={DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_SEED} "
        f"fracs={list(DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_FRACS)}",
        "seed\tmode\tdataset\ttau\tmajors\tsample_ari",
    ]
    seed = DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_SEED
    for mode, (nm, na, tm, ta) in (
        DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"{seed}\t{mode}\tnested\t"
            f"{DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_NESTED_TAU:g}\t"
            f"{nm}\t{na_s}"
        )
        lines.append(
            f"{seed}\t{mode}\ttori\t"
            f"{DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_TORI_TAU:g}\t"
            f"{tm}\t{ta_s}"
        )
    lines.append(f"# {DENSER_SOFT_SEED0_BRIDGE_MASS_WINDOW_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft×persist_agree at operational tau* e2e leaves (A2-T57 → A3 SI)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, lean max_grid_points=12,
# scale_seed=42+dataset_seed, soft_frac=0.25 betweenness. Contrast T54
# baseline (n=80/120): denser kills seed1 nested e2e inflate (all modes
# ≤1 leaf); denser-youden alone leaves seed0 nested K=2 chance-ARI≈0.01,
# but soft/persist/soft×persist collapse that inflate. Circle youden does
# not shatter on denser (T54 circle youden=2). Soft≠sample-ARI recovery.

DENSER_SOFT_X_PERSIST_TAU_STAR_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_SOFT_X_PERSIST_TAU_STAR_SOFT_FRAC: float = 0.25
DENSER_SOFT_X_PERSIST_TAU_STAR_SOFT_METHOD: str = "betweenness"
DENSER_SOFT_X_PERSIST_TAU_STAR_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_X_PERSIST_TAU_STAR_MAX_GRID_POINTS: int = 12
DENSER_SOFT_X_PERSIST_TAU_STAR_SCALE_SEED_BASE: int = 42
DENSER_SOFT_X_PERSIST_TAU_STAR_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_X_PERSIST_TAU_STAR_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_X_PERSIST_TAU_STAR_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_X_PERSIST_TAU_STAR_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.01, 1, None),
        "soft": (1, None, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
    },
}

# Uniform leaf counts under the same denser lean tau* / Youden knobs (seed=0).
DENSER_SOFT_X_PERSIST_TAU_STAR_UNIFORMS: dict[str, dict[str, int]] = {
    "circle": {
        "youden": 1,
        "soft": 1,
        "persist": 1,
        "soft_x_persist": 1,
    },
    "swiss": {
        "youden": 1,
        "soft": 1,
        "persist": 1,
        "soft_x_persist": 1,
    },
}

DENSER_SOFT_X_PERSIST_TAU_STAR_SI_NOTE: str = (
    "A2-T57 denser soft×persist_agree at operational scale-search tau* e2e "
    "(n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False, "
    "lean max_grid_points=12, scale_seed=42+dataset_seed, soft_frac=0.25 "
    "betweenness): denser kills baseline T54 seed1 nested e2e inflate "
    "(all modes ≤1 leaf); denser-youden alone leaves seed0 nested K=2 "
    "chance-ARI≈0.01, killed by soft/persist/soft×persist. Circle youden "
    "does not shatter on denser (contrast T54 circle youden=2). Soft≠"
    "sample-ARI recovery; defaults off; no awaiting flip."
)


def format_denser_soft_x_persist_tau_star_table() -> str:
    """TSV export of denser soft×persist at operational tau* e2e (A2-T57)."""

    lines = [
        "# denser soft × persist_agree at operational scale-search tau* e2e",
        f"# nested_n={DENSER_SOFT_X_PERSIST_TAU_STAR_NESTED_N} "
        f"tori_n={DENSER_SOFT_X_PERSIST_TAU_STAR_TORI_N} "
        f"max_nodes={DENSER_SOFT_X_PERSIST_TAU_STAR_MAX_NODES} "
        f"h0={DENSER_SOFT_X_PERSIST_TAU_STAR_H0:g} "
        f"soft_frac={DENSER_SOFT_X_PERSIST_TAU_STAR_SOFT_FRAC:g} "
        f"method={DENSER_SOFT_X_PERSIST_TAU_STAR_SOFT_METHOD} "
        f"max_grid_points={DENSER_SOFT_X_PERSIST_TAU_STAR_MAX_GRID_POINTS} "
        f"scale_seed_base={DENSER_SOFT_X_PERSIST_TAU_STAR_SCALE_SEED_BASE} "
        f"seeds={list(DENSER_SOFT_X_PERSIST_TAU_STAR_SEEDS)}",
        "seed\tmode\tdataset\tleaves\tsample_ari",
    ]
    for seed in DENSER_SOFT_X_PERSIST_TAU_STAR_SEEDS:
        for mode, (nl, na, tl, ta) in (
            DENSER_SOFT_X_PERSIST_TAU_STAR_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"{seed}\t{mode}\tnested\t{nl}\t{na_s}")
            lines.append(f"{seed}\t{mode}\ttori\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in DENSER_SOFT_X_PERSIST_TAU_STAR_UNIFORMS.items():
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {DENSER_SOFT_X_PERSIST_TAU_STAR_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft×require_gabriel_and_h at operational tau* e2e leaves (A2-T58 → A3 SI)
# ---------------------------------------------------------------------------
# Baseline n=80/120, Youden h0≈0.73, lean max_grid_points=12,
# scale_seed=42+dataset_seed, soft_frac=0.25 betweenness. Contrast T41
# (fixed probe tau majors): at operational tau* seed1 nested K=2
# chance-ARI≈0 survives youden/soft/conj/soft×conj — conj does not kill
# e2e seed1 inflate. Seeds0/2 + all tori stay 1 leaf. Circle youden
# alone shatters (2 leaves); soft/conj/soft×conj keep uniforms at 1.
# Soft≠sample-ARI recovery; defaults off.

SOFT_X_GABRIEL_TAU_STAR_SEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_X_GABRIEL_TAU_STAR_SOFT_FRAC: float = 0.25
SOFT_X_GABRIEL_TAU_STAR_SOFT_METHOD: str = "betweenness"
SOFT_X_GABRIEL_TAU_STAR_H0: float = PROPOSED_H0_YOUDEN
SOFT_X_GABRIEL_TAU_STAR_MAX_GRID_POINTS: int = 12
SOFT_X_GABRIEL_TAU_STAR_SCALE_SEED_BASE: int = 42

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
SOFT_X_GABRIEL_TAU_STAR_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
    1: {
        "youden": (2, 0.0, 1, None),
        "soft": (2, 0.0, 1, None),
        "conj": (2, 0.0, 1, None),
        "soft_x_conj": (2, 0.0, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
}

# Uniform leaf counts under the same lean tau* / Youden knobs (seed=0).
SOFT_X_GABRIEL_TAU_STAR_UNIFORMS: dict[str, dict[str, int]] = {
    "circle": {
        "youden": 2,
        "soft": 1,
        "conj": 1,
        "soft_x_conj": 1,
    },
    "swiss": {
        "youden": 1,
        "soft": 1,
        "conj": 1,
        "soft_x_conj": 1,
    },
}

SOFT_X_GABRIEL_TAU_STAR_SI_NOTE: str = (
    "A2-T58 soft×require_gabriel_and_h at operational scale-search tau* "
    "e2e (baseline n=80/120, Youden h0≈0.73, mid=0.5 gabriel=False, lean "
    "max_grid_points=12, scale_seed=42+dataset_seed, soft_frac=0.25 "
    "betweenness): seed1 nested K=2 chance-ARI≈0 survives "
    "youden/soft/conj/soft×conj — conj does not kill e2e seed1 inflate "
    "(contrast T41 fixed-tau majors collapse under conj); seeds0/2 + all "
    "tori stay 1 leaf. Circle youden alone shatters (2 leaves); "
    "soft/conj/soft×conj keep uniforms at 1. E2e leaf ≠ sample-ARI "
    "recovery; defaults off; no awaiting flip."
)


def format_soft_x_gabriel_tau_star_table() -> str:
    """TSV export of soft×gabriel_and_h at operational tau* e2e (A2-T58)."""

    lines = [
        "# soft × require_gabriel_and_h at operational scale-search tau* e2e",
        f"# h0={SOFT_X_GABRIEL_TAU_STAR_H0:g} "
        f"soft_frac={SOFT_X_GABRIEL_TAU_STAR_SOFT_FRAC:g} "
        f"method={SOFT_X_GABRIEL_TAU_STAR_SOFT_METHOD} "
        f"max_grid_points={SOFT_X_GABRIEL_TAU_STAR_MAX_GRID_POINTS} "
        f"scale_seed_base={SOFT_X_GABRIEL_TAU_STAR_SCALE_SEED_BASE} "
        f"seeds={list(SOFT_X_GABRIEL_TAU_STAR_SEEDS)}",
        "seed\tmode\tdataset\tleaves\tsample_ari",
    ]
    for seed in SOFT_X_GABRIEL_TAU_STAR_SEEDS:
        for mode, (nl, na, tl, ta) in SOFT_X_GABRIEL_TAU_STAR_TABLE[seed].items():
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"{seed}\t{mode}\tnested\t{nl}\t{na_s}")
            lines.append(f"{seed}\t{mode}\ttori\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in SOFT_X_GABRIEL_TAU_STAR_UNIFORMS.items():
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {SOFT_X_GABRIEL_TAU_STAR_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft seed0 keep-band × persist_agree bet vs bridge_mass (A2-T59)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, seed0, Youden h0≈0.73, lean
# max_grid_points=12, scale_seed=42. Contrast T55 majors keep-band
# (soft≤0.12 → tori K=2 betweenness): at denser e2e tau*, soft and
# soft×persist collapse nested+tori to ≤1 across keep/collapse fracs
# for BOTH betweenness and bridge_mass — T55 keep-band is majors-only;
# T56 method contrast is moot at e2e. Youden alone still leaves seed0
# nested K=2 chance-ARI≈0.01 (T57); persist kills it. Soft≠recovery.

DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_SEED: int = 0
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_FRACS: tuple[float, ...] = (
    0.05, 0.12, 0.15, 0.25,
)
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_MAX_GRID_POINTS: int = 12
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_SCALE_SEED_BASE: int = 42
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_KEEP_MAX_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC
)
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_COLLAPSE_MIN_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC
)

# mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (2, 0.01, 1, None),
    "persist": (1, None, 1, None),
    "soft_bet_0.05": (1, None, 1, None),
    "soft_x_persist_bet_0.05": (1, None, 1, None),
    "soft_bet_0.12": (1, None, 1, None),
    "soft_x_persist_bet_0.12": (1, None, 1, None),
    "soft_bet_0.15": (1, None, 1, None),
    "soft_x_persist_bet_0.15": (1, None, 1, None),
    "soft_bet_0.25": (1, None, 1, None),
    "soft_x_persist_bet_0.25": (1, None, 1, None),
    "soft_bridge_0.05": (1, None, 1, None),
    "soft_x_persist_bridge_0.05": (1, None, 1, None),
    "soft_bridge_0.12": (1, None, 1, None),
    "soft_x_persist_bridge_0.12": (1, None, 1, None),
    "soft_bridge_0.15": (1, None, 1, None),
    "soft_x_persist_bridge_0.15": (1, None, 1, None),
    "soft_bridge_0.25": (1, None, 1, None),
    "soft_x_persist_bridge_0.25": (1, None, 1, None),
}

DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_SI_NOTE: str = (
    "A2-T59 denser soft seed0 keep-band under persist_agree "
    "(bet vs bridge_mass) at operational tau* e2e (n=160/240, "
    "max_nodes=128, Youden h0≈0.73, lean max_grid_points=12, "
    "scale_seed=42): T55 majors keep-band soft_frac≤0.12 (tori K=2 "
    "betweenness) does **not** survive denser e2e soft or soft×persist "
    "— both betweenness and bridge_mass collapse nested+tori to ≤1 "
    "across soft_frac∈{0.05,0.12,0.15,0.25}; T56 method contrast is "
    "moot at e2e. Youden alone leaves seed0 nested K=2 chance-ARI≈0.01 "
    "(T57), killed by persist/soft. Soft≠sample-ARI recovery; defaults "
    "off; no awaiting flip."
)


def format_denser_soft_seed0_keep_band_x_persist_table() -> str:
    """TSV export of denser soft seed0 keep-band × persist e2e (A2-T59)."""

    lines = [
        "# denser soft seed0 keep-band × persist_agree bet vs bridge_mass e2e",
        f"# nested_n={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_NESTED_N} "
        f"tori_n={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_TORI_N} "
        f"max_nodes={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_MAX_NODES} "
        f"h0={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_H0:g} "
        f"max_grid_points={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_MAX_GRID_POINTS} "
        f"scale_seed_base={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_SCALE_SEED_BASE} "
        f"seed={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_SEED} "
        f"fracs={list(DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_FRACS)} "
        f"keep_max={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_KEEP_MAX_FRAC:g} "
        f"collapse_min={DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_COLLAPSE_MIN_FRAC:g}",
        "seed\tmode\tdataset\tleaves\tsample_ari",
    ]
    seed = DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_SEED
    for mode, (nl, na, tl, ta) in (
        DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(f"{seed}\t{mode}\tnested\t{nl}\t{na_s}")
        lines.append(f"{seed}\t{mode}\ttori\t{tl}\t{ta_s}")
    lines.append(f"# {DENSER_SOFT_SEED0_KEEP_BAND_X_PERSIST_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft×require_gabriel_and_h at operational tau* e2e (A2-T60-followon)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, lean max_grid_points=12,
# scale_seed=42+dataset_seed, soft_frac=0.25 betweenness. Contrast T58
# baseline (n=80/120): denser kills seed1 nested e2e inflate (all modes
# ≤1 leaf); denser-youden alone leaves seed0 nested K=2 chance-ARI≈0.01,
# but soft/conj/soft×conj collapse that inflate. Circle youden does not
# shatter on denser (T58 circle youden=2). Soft≠sample-ARI recovery.

DENSER_SOFT_X_GABRIEL_TAU_STAR_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_SOFT_X_GABRIEL_TAU_STAR_SOFT_FRAC: float = 0.25
DENSER_SOFT_X_GABRIEL_TAU_STAR_SOFT_METHOD: str = "betweenness"
DENSER_SOFT_X_GABRIEL_TAU_STAR_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_X_GABRIEL_TAU_STAR_MAX_GRID_POINTS: int = 12
DENSER_SOFT_X_GABRIEL_TAU_STAR_SCALE_SEED_BASE: int = 42
DENSER_SOFT_X_GABRIEL_TAU_STAR_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_X_GABRIEL_TAU_STAR_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_X_GABRIEL_TAU_STAR_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_X_GABRIEL_TAU_STAR_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.01, 1, None),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
}

# Uniform leaf counts under the same denser lean tau* / Youden knobs (seed=0).
DENSER_SOFT_X_GABRIEL_TAU_STAR_UNIFORMS: dict[str, dict[str, int]] = {
    "circle": {
        "youden": 1,
        "soft": 1,
        "conj": 1,
        "soft_x_conj": 1,
    },
    "swiss": {
        "youden": 1,
        "soft": 1,
        "conj": 1,
        "soft_x_conj": 1,
    },
}

DENSER_SOFT_X_GABRIEL_TAU_STAR_SI_NOTE: str = (
    "A2-T60-followon denser soft×require_gabriel_and_h at operational "
    "scale-search tau* e2e (n=160/240, max_nodes=128, Youden h0≈0.73, "
    "mid=0.5 gabriel=False, lean max_grid_points=12, "
    "scale_seed=42+dataset_seed, soft_frac=0.25 betweenness): denser "
    "kills baseline T58 seed1 nested e2e inflate (all modes ≤1 leaf); "
    "denser-youden alone leaves seed0 nested K=2 chance-ARI≈0.01, "
    "killed by soft/conj/soft×conj. Circle youden does not shatter on "
    "denser (contrast T58 circle youden=2). Soft≠sample-ARI recovery; "
    "defaults off; no awaiting flip."
)


def format_denser_soft_x_gabriel_tau_star_table() -> str:
    """TSV export of denser soft×gabriel_and_h at operational tau* (A2-T60)."""

    lines = [
        "# denser soft × require_gabriel_and_h at operational scale-search tau* e2e",
        f"# nested_n={DENSER_SOFT_X_GABRIEL_TAU_STAR_NESTED_N} "
        f"tori_n={DENSER_SOFT_X_GABRIEL_TAU_STAR_TORI_N} "
        f"max_nodes={DENSER_SOFT_X_GABRIEL_TAU_STAR_MAX_NODES} "
        f"h0={DENSER_SOFT_X_GABRIEL_TAU_STAR_H0:g} "
        f"soft_frac={DENSER_SOFT_X_GABRIEL_TAU_STAR_SOFT_FRAC:g} "
        f"method={DENSER_SOFT_X_GABRIEL_TAU_STAR_SOFT_METHOD} "
        f"max_grid_points={DENSER_SOFT_X_GABRIEL_TAU_STAR_MAX_GRID_POINTS} "
        f"scale_seed_base={DENSER_SOFT_X_GABRIEL_TAU_STAR_SCALE_SEED_BASE} "
        f"seeds={list(DENSER_SOFT_X_GABRIEL_TAU_STAR_SEEDS)}",
        "seed\tmode\tdataset\tleaves\tsample_ari",
    ]
    for seed in DENSER_SOFT_X_GABRIEL_TAU_STAR_SEEDS:
        for mode, (nl, na, tl, ta) in (
            DENSER_SOFT_X_GABRIEL_TAU_STAR_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"{seed}\t{mode}\tnested\t{nl}\t{na_s}")
            lines.append(f"{seed}\t{mode}\ttori\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in DENSER_SOFT_X_GABRIEL_TAU_STAR_UNIFORMS.items():
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {DENSER_SOFT_X_GABRIEL_TAU_STAR_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft×gabriel×persist compose at operational tau* e2e (A2-T61-followon)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, lean max_grid_points=12,
# scale_seed=42+dataset_seed, soft_frac=0.25 betweenness. Modes contrast
# T57 soft×persist and T60 soft×conj pairwise denser baselines against the
# triple soft×conj×persist compose. Compose does not unlock beyond pairwise
# denser collapse; denser-youden alone leaves seed0 nested K=2
# chance-ARI≈0.01. Soft≠sample-ARI recovery.

DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SOFT_FRAC: float = 0.25
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SOFT_METHOD: str = "betweenness"
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_MAX_GRID_POINTS: int = 12
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SCALE_SEED_BASE: int = 42
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_MAX_NODES: int = DENSER_PROPOSED_H0_MAX_NODES

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.01, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
}

# Uniform leaf counts under the same denser lean tau* / Youden knobs (seed=0).
DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_UNIFORMS: dict[str, dict[str, int]] = {
    "circle": {
        "youden": 1,
        "soft_x_conj": 1,
        "soft_x_persist": 1,
        "soft_x_conj_x_persist": 1,
    },
    "swiss": {
        "youden": 1,
        "soft_x_conj": 1,
        "soft_x_persist": 1,
        "soft_x_conj_x_persist": 1,
    },
}

DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SI_NOTE: str = (
    "A2-T61-followon denser soft×require_gabriel_and_h×persist_agree "
    "compose at operational scale-search tau* e2e (n=160/240, "
    "max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False, lean "
    "max_grid_points=12, scale_seed=42+dataset_seed, soft_frac=0.25 "
    "betweenness): triple compose does not unlock beyond T57/T60 "
    "pairwise denser collapse (soft×conj / soft×persist / "
    "soft×conj×persist all ≤1 leaf); denser-youden alone leaves seed0 "
    "nested K=2 chance-ARI≈0.01. Circle/swiss stay 1. Soft≠sample-ARI "
    "recovery; defaults off; no awaiting flip."
)


def format_denser_soft_x_gabriel_x_persist_tau_star_table() -> str:
    """TSV export of denser soft×gabriel×persist compose@tau* (A2-T61)."""

    lines = [
        "# denser soft × require_gabriel_and_h × persist_agree compose "
        "at operational scale-search tau* e2e",
        f"# nested_n={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_NESTED_N} "
        f"tori_n={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_TORI_N} "
        f"max_nodes={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_MAX_NODES} "
        f"h0={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_H0:g} "
        f"soft_frac={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SOFT_FRAC:g} "
        f"method={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SOFT_METHOD} "
        f"max_grid_points={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_MAX_GRID_POINTS} "
        f"scale_seed_base={DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SCALE_SEED_BASE} "
        f"seeds={list(DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SEEDS)}",
        "seed\tmode\tdataset\tleaves\tsample_ari",
    ]
    for seed in DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SEEDS:
        for mode, (nl, na, tl, ta) in (
            DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"{seed}\t{mode}\tnested\t{nl}\t{na_s}")
            lines.append(f"{seed}\t{mode}\ttori\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in (
        DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_UNIFORMS.items()
    ):
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {DENSER_SOFT_X_GABRIEL_X_PERSIST_TAU_STAR_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Non-denser soft keep-band × persist majors baseline (A2-T61 → A3 SI)
# ---------------------------------------------------------------------------
# Baseline n=80/120, max_nodes=64, seed0, Youden h0≈0.73, mid=0.5
# gabriel=False, betweenness soft_capacity. Fixed-tau majors (0.27/0.5)
# keep-band soft_frac≤0.5 → tori K=2 chance-ARI≈0.26 (wider than denser
# T55 ≤0.12); soft≥0.75 collapses to 1. Nested soft → ≤1. Soft and
# soft×persist_agree at operational tau* e2e (lean max_grid_points=12,
# scale_seed=42) collapse nested+tori to ≤1 across keep/collapse fracs —
# keep-band is majors-only (same lesson as denser T59). Soft≠recovery.

SOFT_KEEP_BAND_X_PERSIST_MAJORS_SEED: int = 0
SOFT_KEEP_BAND_X_PERSIST_MAJORS_FRACS: tuple[float, ...] = (
    0.12, 0.25, 0.5, 0.75, 0.9,
)
SOFT_KEEP_BAND_X_PERSIST_MAJORS_NESTED_N: int = 80
SOFT_KEEP_BAND_X_PERSIST_MAJORS_TORI_N: int = 120
SOFT_KEEP_BAND_X_PERSIST_MAJORS_MAX_NODES: int = 64
SOFT_KEEP_BAND_X_PERSIST_MAJORS_NESTED_TAU: float = 0.27
SOFT_KEEP_BAND_X_PERSIST_MAJORS_TORI_TAU: float = 0.5
SOFT_KEEP_BAND_X_PERSIST_MAJORS_H0: float = PROPOSED_H0_YOUDEN
SOFT_KEEP_BAND_X_PERSIST_MAJORS_METHOD: str = "betweenness"
SOFT_KEEP_BAND_X_PERSIST_MAJORS_MAX_GRID_POINTS: int = 12
SOFT_KEEP_BAND_X_PERSIST_MAJORS_SCALE_SEED_BASE: int = 42
SOFT_KEEP_BAND_X_PERSIST_MAJORS_KEEP_MAX_FRAC: float = 0.5
SOFT_KEEP_BAND_X_PERSIST_MAJORS_COLLAPSE_MIN_FRAC: float = 0.75

# Fixed-tau majors: mode → (nested_majors, nested_ari, tori_majors, tori_ari)
SOFT_KEEP_BAND_X_PERSIST_MAJORS_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (2, 0.12, 2, 0.26),
    "soft_0.12": (1, None, 2, 0.26),
    "soft_0.25": (1, None, 2, 0.26),
    "soft_0.5": (1, None, 2, 0.26),
    "soft_0.75": (1, None, 1, None),
    "soft_0.9": (1, None, 1, None),
}

# E2E soft / soft×persist at keep/collapse fracs (seed0): all ≤1 leaf.
# mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
SOFT_KEEP_BAND_X_PERSIST_MAJORS_E2E_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (1, None, 1, None),
    "persist": (1, None, 1, None),
    "soft_0.12": (1, None, 1, None),
    "soft_x_persist_0.12": (1, None, 1, None),
    "soft_0.25": (1, None, 1, None),
    "soft_x_persist_0.25": (1, None, 1, None),
    "soft_0.5": (1, None, 1, None),
    "soft_x_persist_0.5": (1, None, 1, None),
    "soft_0.75": (1, None, 1, None),
    "soft_x_persist_0.75": (1, None, 1, None),
}

SOFT_KEEP_BAND_X_PERSIST_MAJORS_SI_NOTE: str = (
    "A2-T61 non-denser soft keep-band × persist majors baseline "
    "(n=80/120, max_nodes=64, Youden h0≈0.73, mid=0.5 gabriel=False, "
    "betweenness, fixed-tau majors 0.27/0.5 + lean tau* e2e "
    "max_grid_points=12 scale_seed=42): majors keep-band soft_frac≤0.5 "
    "→ tori K=2 chance-ARI≈0.26 (wider than denser T55 ≤0.12); "
    "soft≥0.75 collapses to 1; nested soft → ≤1. Soft and soft×persist "
    "e2e collapse nested+tori to ≤1 across soft_frac∈{0.12,0.25,0.5,0.75} "
    "— keep-band is majors-only (same lesson as denser T59). "
    "Chance-ARI ≠ sample-ARI recovery; defaults off; no awaiting flip."
)


def format_soft_keep_band_x_persist_majors_table() -> str:
    """TSV export of non-denser soft keep-band × persist majors (A2-T61)."""

    lines = [
        "# non-denser soft keep-band × persist majors baseline",
        f"# nested_n={SOFT_KEEP_BAND_X_PERSIST_MAJORS_NESTED_N} "
        f"tori_n={SOFT_KEEP_BAND_X_PERSIST_MAJORS_TORI_N} "
        f"max_nodes={SOFT_KEEP_BAND_X_PERSIST_MAJORS_MAX_NODES} "
        f"h0={SOFT_KEEP_BAND_X_PERSIST_MAJORS_H0:g} "
        f"method={SOFT_KEEP_BAND_X_PERSIST_MAJORS_METHOD} "
        f"max_grid_points={SOFT_KEEP_BAND_X_PERSIST_MAJORS_MAX_GRID_POINTS} "
        f"scale_seed_base={SOFT_KEEP_BAND_X_PERSIST_MAJORS_SCALE_SEED_BASE} "
        f"seed={SOFT_KEEP_BAND_X_PERSIST_MAJORS_SEED} "
        f"fracs={list(SOFT_KEEP_BAND_X_PERSIST_MAJORS_FRACS)} "
        f"keep_max={SOFT_KEEP_BAND_X_PERSIST_MAJORS_KEEP_MAX_FRAC:g} "
        f"collapse_min={SOFT_KEEP_BAND_X_PERSIST_MAJORS_COLLAPSE_MIN_FRAC:g}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tmajors_or_leaves\tsample_ari",
    ]
    seed = SOFT_KEEP_BAND_X_PERSIST_MAJORS_SEED
    for mode, (nm, na, tm, ta) in (
        SOFT_KEEP_BAND_X_PERSIST_MAJORS_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"majors\t{seed}\t{mode}\tnested\t"
            f"{SOFT_KEEP_BAND_X_PERSIST_MAJORS_NESTED_TAU:g}\t{nm}\t{na_s}"
        )
        lines.append(
            f"majors\t{seed}\t{mode}\ttori\t"
            f"{SOFT_KEEP_BAND_X_PERSIST_MAJORS_TORI_TAU:g}\t{tm}\t{ta_s}"
        )
    for mode, (nl, na, tl, ta) in (
        SOFT_KEEP_BAND_X_PERSIST_MAJORS_E2E_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
        lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append(f"# {SOFT_KEEP_BAND_X_PERSIST_MAJORS_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Soft×gabriel×persist majors non-denser seed1 inflate window (A2-T63)
# ---------------------------------------------------------------------------
# Baseline n=80/120, max_nodes=64, Youden h0≈0.73, mid=0.5 gabriel=False,
# soft_frac=0.25 betweenness, seeds 0..2. Fixed-tau majors (0.27/0.5):
# seed1 soft alone inflates nested K=2 chance-ARI≈0.08; gabriel conj /
# soft×conj kill that majors inflate. Lean tau* e2e (max_grid_points=12,
# scale_seed=42+seed): seed1 nested K=2 chance-ARI≈0 survives youden /
# soft / conj / persist / soft×conj / soft×persist / conj×persist /
# soft×conj×persist (same T54/T58 seed1 window) — majors≠e2e. Circle
# youden shatters; soft/conj/persist keep uniforms at 1. Soft≠recovery.

SOFT_X_GABRIEL_X_PERSIST_MAJORS_SEEDS: tuple[int, ...] = (0, 1, 2)
SOFT_X_GABRIEL_X_PERSIST_MAJORS_SOFT_FRAC: float = 0.25
SOFT_X_GABRIEL_X_PERSIST_MAJORS_SOFT_METHOD: str = "betweenness"
SOFT_X_GABRIEL_X_PERSIST_MAJORS_NESTED_N: int = 80
SOFT_X_GABRIEL_X_PERSIST_MAJORS_TORI_N: int = 120
SOFT_X_GABRIEL_X_PERSIST_MAJORS_MAX_NODES: int = 64
SOFT_X_GABRIEL_X_PERSIST_MAJORS_NESTED_TAU: float = 0.27
SOFT_X_GABRIEL_X_PERSIST_MAJORS_TORI_TAU: float = 0.5
SOFT_X_GABRIEL_X_PERSIST_MAJORS_H0: float = PROPOSED_H0_YOUDEN
SOFT_X_GABRIEL_X_PERSIST_MAJORS_MAX_GRID_POINTS: int = 12
SOFT_X_GABRIEL_X_PERSIST_MAJORS_SCALE_SEED_BASE: int = 42

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
# majors surface: soft / conj / soft_x_conj (persist N/A at majors prune)
SOFT_X_GABRIEL_X_PERSIST_MAJORS_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.12, 2, 0.26),
        "soft": (1, None, 2, 0.26),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft": (2, 0.08, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 2, 0.22),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
}

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
SOFT_X_GABRIEL_X_PERSIST_MAJORS_E2E_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
    1: {
        "youden": (2, 0.0, 1, None),
        "soft_x_conj": (2, 0.0, 1, None),
        "soft_x_persist": (2, 0.0, 1, None),
        "soft_x_conj_x_persist": (2, 0.0, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
}

SOFT_X_GABRIEL_X_PERSIST_MAJORS_UNIFORMS: dict[str, dict[str, int]] = {
    "circle": {
        "youden": 2,
        "soft_x_conj": 1,
        "soft_x_persist": 1,
        "soft_x_conj_x_persist": 1,
    },
    "swiss": {
        "youden": 1,
        "soft_x_conj": 1,
        "soft_x_persist": 1,
        "soft_x_conj_x_persist": 1,
    },
}

SOFT_X_GABRIEL_X_PERSIST_MAJORS_SI_NOTE: str = (
    "A2-T63 soft×require_gabriel_and_h×persist_agree majors non-denser "
    "seed1 inflate window (n=80/120, max_nodes=64, Youden h0≈0.73, "
    "mid=0.5 gabriel=False, soft_frac=0.25 betweenness, fixed-tau majors "
    "0.27/0.5 + lean tau* e2e max_grid_points=12 scale_seed=42+seed): "
    "seed1 majors soft alone inflates nested K=2 chance-ARI≈0.08 — "
    "killed by gabriel conj / soft×conj; e2e seed1 nested K=2 "
    "chance-ARI≈0 survives soft×conj / soft×persist / soft×conj×persist "
    "(majors≠e2e; same T54/T58 seed1 window). Circle youden shatters; "
    "soft/conj/persist keep uniforms at 1. Soft≠sample-ARI recovery; "
    "defaults off; no awaiting flip."
)


def format_soft_x_gabriel_x_persist_majors_table() -> str:
    """TSV export of soft×gabriel×persist majors seed1 window (A2-T63)."""

    lines = [
        "# soft × require_gabriel_and_h × persist_agree majors "
        "non-denser seed1 inflate window",
        f"# nested_n={SOFT_X_GABRIEL_X_PERSIST_MAJORS_NESTED_N} "
        f"tori_n={SOFT_X_GABRIEL_X_PERSIST_MAJORS_TORI_N} "
        f"max_nodes={SOFT_X_GABRIEL_X_PERSIST_MAJORS_MAX_NODES} "
        f"h0={SOFT_X_GABRIEL_X_PERSIST_MAJORS_H0:g} "
        f"soft_frac={SOFT_X_GABRIEL_X_PERSIST_MAJORS_SOFT_FRAC:g} "
        f"method={SOFT_X_GABRIEL_X_PERSIST_MAJORS_SOFT_METHOD} "
        f"max_grid_points={SOFT_X_GABRIEL_X_PERSIST_MAJORS_MAX_GRID_POINTS} "
        f"scale_seed_base={SOFT_X_GABRIEL_X_PERSIST_MAJORS_SCALE_SEED_BASE} "
        f"seeds={list(SOFT_X_GABRIEL_X_PERSIST_MAJORS_SEEDS)}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tmajors_or_leaves\tsample_ari",
    ]
    for seed in SOFT_X_GABRIEL_X_PERSIST_MAJORS_SEEDS:
        for mode, (nm, na, tm, ta) in (
            SOFT_X_GABRIEL_X_PERSIST_MAJORS_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"majors\t{seed}\t{mode}\tnested\t"
                f"{SOFT_X_GABRIEL_X_PERSIST_MAJORS_NESTED_TAU:g}\t{nm}\t{na_s}"
            )
            lines.append(
                f"majors\t{seed}\t{mode}\ttori\t"
                f"{SOFT_X_GABRIEL_X_PERSIST_MAJORS_TORI_TAU:g}\t{tm}\t{ta_s}"
            )
        for mode, (nl, na, tl, ta) in (
            SOFT_X_GABRIEL_X_PERSIST_MAJORS_E2E_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
            lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in SOFT_X_GABRIEL_X_PERSIST_MAJORS_UNIFORMS.items():
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {SOFT_X_GABRIEL_X_PERSIST_MAJORS_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft keep-band × require_gabriel_and_h majors (A2-T64-followon)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, seed0, Youden h0≈0.73, mid=0.5
# gabriel=False, betweenness soft_capacity. Fixed-tau majors (0.27/0.5)
# reproduce T55 keep-band soft_frac≤0.12 → tori K=2 chance-ARI≈0.16–0.18;
# soft≥0.15 collapses. Gabriel conj alone and soft×conj kill that majors
# keep-band (all ≤1). Lean tau* e2e (max_grid_points=12, scale_seed=42):
# youden alone leaves seed0 nested K=2 chance-ARI≈0.01 (T57/T60); conj /
# soft / soft×conj collapse nested+tori to ≤1 across keep/collapse fracs
# — keep-band is majors-only and gabriel-fragile. Soft≠recovery.

DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_SEED: int = 0
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_FRACS: tuple[float, ...] = (
    0.05, 0.08, 0.10, 0.12, 0.15, 0.25,
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_NESTED_N: int = DENSER_PROPOSED_H0_NESTED_N
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_TORI_N: int = DENSER_PROPOSED_H0_TORI_N
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_MAX_NODES: int = (
    DENSER_PROPOSED_H0_MAX_NODES
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_NESTED_TAU: float = 0.27
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_TORI_TAU: float = 0.5
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_METHOD: str = "betweenness"
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_MAX_GRID_POINTS: int = 12
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_SCALE_SEED_BASE: int = 42
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_KEEP_MAX_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_COLLAPSE_MIN_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC
)

# Fixed-tau majors: mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (1, None, 2, 0.14),
    "conj": (1, None, 1, None),
    "soft_0.05": (1, None, 2, 0.16),
    "soft_x_conj_0.05": (1, None, 1, None),
    "soft_0.08": (1, None, 2, 0.16),
    "soft_x_conj_0.08": (1, None, 1, None),
    "soft_0.1": (1, None, 2, 0.18),
    "soft_x_conj_0.1": (1, None, 1, None),
    "soft_0.12": (1, None, 2, 0.18),
    "soft_x_conj_0.12": (1, None, 1, None),
    "soft_0.15": (1, None, 1, None),
    "soft_x_conj_0.15": (1, None, 1, None),
    "soft_0.25": (1, None, 1, None),
    "soft_x_conj_0.25": (1, None, 1, None),
}

# E2E soft / soft×conj at keep/collapse fracs (seed0): soft/conj ≤1;
# youden alone keeps nested K=2 chance-ARI≈0.01.
# mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_E2E_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (2, 0.01, 1, None),
    "conj": (1, None, 1, None),
    "soft_0.05": (1, None, 1, None),
    "soft_x_conj_0.05": (1, None, 1, None),
    "soft_0.12": (1, None, 1, None),
    "soft_x_conj_0.12": (1, None, 1, None),
    "soft_0.15": (1, None, 1, None),
    "soft_x_conj_0.15": (1, None, 1, None),
    "soft_0.25": (1, None, 1, None),
    "soft_x_conj_0.25": (1, None, 1, None),
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_UNIFORMS: dict[str, dict[str, int]] = {
    "circle": {
        "youden": 1,
        "conj": 1,
        "soft_0.05": 1,
        "soft_x_conj_0.05": 1,
        "soft_0.12": 1,
        "soft_x_conj_0.12": 1,
    },
    "swiss": {
        "youden": 1,
        "conj": 1,
        "soft_0.05": 1,
        "soft_x_conj_0.05": 1,
        "soft_0.12": 1,
        "soft_x_conj_0.12": 1,
    },
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_SI_NOTE: str = (
    "A2-T64-followon denser soft keep-band × require_gabriel_and_h majors "
    "(n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False, "
    "betweenness, fixed-tau majors 0.27/0.5 + lean tau* e2e "
    "max_grid_points=12 scale_seed=42): T55 majors keep-band "
    "soft_frac≤0.12 → tori K=2 chance-ARI≈0.16–0.18 is killed by gabriel "
    "conj / soft×conj at majors (all ≤1); soft≥0.15 collapses alone. "
    "Lean e2e youden alone leaves seed0 nested K=2 chance-ARI≈0.01; "
    "conj/soft/soft×conj collapse nested+tori to ≤1 across "
    "soft_frac∈{0.05,0.12,0.15,0.25} — keep-band is majors-only and "
    "gabriel-fragile. Chance-ARI ≠ sample-ARI recovery; defaults off; "
    "no awaiting flip."
)


def format_denser_soft_keep_band_x_gabriel_majors_table() -> str:
    """TSV export of denser soft keep-band × gabriel majors (A2-T64-followon)."""

    lines = [
        "# denser soft keep-band × require_gabriel_and_h majors",
        f"# nested_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_NESTED_N} "
        f"tori_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_TORI_N} "
        f"max_nodes={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_MAX_NODES} "
        f"h0={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_H0:g} "
        f"method={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_METHOD} "
        f"max_grid_points="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_MAX_GRID_POINTS} "
        f"scale_seed_base="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_SCALE_SEED_BASE} "
        f"seed={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_SEED} "
        f"fracs={list(DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_FRACS)} "
        f"keep_max={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_KEEP_MAX_FRAC:g} "
        f"collapse_min="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_COLLAPSE_MIN_FRAC:g}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tmajors_or_leaves\tsample_ari",
    ]
    seed = DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_SEED
    for mode, (nm, na, tm, ta) in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"majors\t{seed}\t{mode}\tnested\t"
            f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_NESTED_TAU:g}\t"
            f"{nm}\t{na_s}"
        )
        lines.append(
            f"majors\t{seed}\t{mode}\ttori\t"
            f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_TORI_TAU:g}\t"
            f"{tm}\t{ta_s}"
        )
    for mode, (nl, na, tl, ta) in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_E2E_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
        lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_UNIFORMS.items()
    ):
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {DENSER_SOFT_KEEP_BAND_X_GABRIEL_MAJORS_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft keep-band × gabriel × persist e2e frac grid (A2-T65-followon)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False,
# betweenness soft_capacity, lean tau* e2e (max_grid_points=12,
# scale_seed=42). Extends T64 soft/soft×conj keep-band e2e with persist
# and soft×conj×persist across denser keep/collapse fracs. Youden alone
# leaves seed0 nested K=2 chance-ARI≈0.01; soft×persist and
# soft×conj×persist collapse nested+tori to ≤1 for every
# soft_frac∈{0.05,0.12,0.15,0.25} — denser keep-band does not survive
# e2e under persist compose. Soft≠sample-ARI recovery.

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_SEED: int = 0
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_FRACS: tuple[float, ...] = (
    0.05, 0.12, 0.15, 0.25,
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_NESTED_N: int = (
    DENSER_PROPOSED_H0_NESTED_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_TORI_N: int = (
    DENSER_PROPOSED_H0_TORI_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_MAX_NODES: int = (
    DENSER_PROPOSED_H0_MAX_NODES
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_METHOD: str = "betweenness"
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_MAX_GRID_POINTS: int = 12
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_SCALE_SEED_BASE: int = 42
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_KEEP_MAX_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_COLLAPSE_MIN_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC
)

# mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (2, 0.01, 1, None),
    "persist": (1, None, 1, None),
    "soft_x_persist_0.05": (1, None, 1, None),
    "soft_x_conj_x_persist_0.05": (1, None, 1, None),
    "soft_x_persist_0.12": (1, None, 1, None),
    "soft_x_conj_x_persist_0.12": (1, None, 1, None),
    "soft_x_persist_0.15": (1, None, 1, None),
    "soft_x_conj_x_persist_0.15": (1, None, 1, None),
    "soft_x_persist_0.25": (1, None, 1, None),
    "soft_x_conj_x_persist_0.25": (1, None, 1, None),
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_UNIFORMS: dict[
    str, dict[str, int]
] = {
    "circle": {
        "youden": 1,
        "persist": 1,
        "soft_x_persist_0.12": 1,
        "soft_x_conj_x_persist_0.12": 1,
    },
    "swiss": {
        "youden": 1,
        "persist": 1,
        "soft_x_persist_0.12": 1,
        "soft_x_conj_x_persist_0.12": 1,
    },
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_SI_NOTE: str = (
    "A2-T65-followon denser soft keep-band × gabriel × persist e2e frac "
    "grid (n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 "
    "gabriel=False, betweenness, lean tau* max_grid_points=12 "
    "scale_seed=42): youden alone leaves seed0 nested K=2 "
    "chance-ARI≈0.01; soft×persist and soft×conj×persist collapse "
    "nested+tori to ≤1 across soft_frac∈{0.05,0.12,0.15,0.25} — denser "
    "T55 keep-band does not survive e2e under persist compose (extends "
    "T64 soft/soft×conj and T61-followon frac=0.25 compose). Chance-ARI "
    "≠ sample-ARI recovery; defaults off; no awaiting flip."
)


def format_denser_soft_keep_band_x_gabriel_x_persist_e2e_table() -> str:
    """TSV export of denser soft keep×gabriel×persist e2e (A2-T65-followon)."""

    lines = [
        "# denser soft keep-band × gabriel × persist e2e frac grid",
        f"# nested_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_NESTED_N} "
        f"tori_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_TORI_N} "
        f"max_nodes={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_MAX_NODES} "
        f"h0={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_H0:g} "
        f"method={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_METHOD} "
        f"max_grid_points="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_MAX_GRID_POINTS} "
        f"scale_seed_base="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_SCALE_SEED_BASE} "
        f"seed={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_SEED} "
        f"fracs={list(DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_FRACS)} "
        f"keep_max="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_KEEP_MAX_FRAC:g} "
        f"collapse_min="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_COLLAPSE_MIN_FRAC:g}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tleaves\tsample_ari",
    ]
    seed = DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_SEED
    for mode, (nl, na, tl, ta) in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
        lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_UNIFORMS.items()
    ):
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(
        f"# {DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_E2E_SI_NOTE}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft keep-band × gabriel multi-seed majors/e2e (A2-T66)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False,
# betweenness soft_capacity, seeds 0..2. Lean keep/collapse fracs
# {0.05,0.12,0.15}. Fixed-tau majors (0.27/0.5): T55/T64 seed0 keep-band
# soft_frac≤0.12 → tori K=2 chance-ARI≈0.16–0.18 is seed0-singleton;
# seeds 1–2 stay ≤1 under youden/soft/conj (no multi-seed keep). Gabriel
# conj kills seed0 keep. Lean tau* e2e (max_grid_points=12,
# scale_seed=42+seed): only seed0 youden nested K=2 chance-ARI≈0.01;
# soft/soft×conj and seeds 1–2 collapse nested+tori to ≤1. Soft≠recovery.

DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_FRACS: tuple[float, ...] = (
    0.05, 0.12, 0.15,
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_NESTED_N: int = (
    DENSER_PROPOSED_H0_NESTED_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_TORI_N: int = (
    DENSER_PROPOSED_H0_TORI_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_MAX_NODES: int = (
    DENSER_PROPOSED_H0_MAX_NODES
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_NESTED_TAU: float = 0.27
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_TORI_TAU: float = 0.5
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_METHOD: str = "betweenness"
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_MAX_GRID_POINTS: int = 12
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_SCALE_SEED_BASE: int = 42
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_KEEP_MAX_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_COLLAPSE_MIN_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC
)

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 2, 0.14),
        "conj": (1, None, 1, None),
        "soft_0.05": (1, None, 2, 0.16),
        "soft_x_conj_0.05": (1, None, 1, None),
        "soft_0.12": (1, None, 2, 0.18),
        "soft_x_conj_0.12": (1, None, 1, None),
        "soft_0.15": (1, None, 1, None),
        "soft_x_conj_0.15": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_0.05": (1, None, 1, None),
        "soft_x_conj_0.05": (1, None, 1, None),
        "soft_0.12": (1, None, 1, None),
        "soft_x_conj_0.12": (1, None, 1, None),
        "soft_0.15": (1, None, 1, None),
        "soft_x_conj_0.15": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_0.05": (1, None, 1, None),
        "soft_x_conj_0.05": (1, None, 1, None),
        "soft_0.12": (1, None, 1, None),
        "soft_x_conj_0.12": (1, None, 1, None),
        "soft_0.15": (1, None, 1, None),
        "soft_x_conj_0.15": (1, None, 1, None),
    },
}

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_E2E_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.01, 1, None),
        "soft_0.12": (1, None, 1, None),
        "soft_x_conj_0.12": (1, None, 1, None),
        "soft_0.15": (1, None, 1, None),
        "soft_x_conj_0.15": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_0.12": (1, None, 1, None),
        "soft_x_conj_0.12": (1, None, 1, None),
        "soft_0.15": (1, None, 1, None),
        "soft_x_conj_0.15": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_0.12": (1, None, 1, None),
        "soft_x_conj_0.12": (1, None, 1, None),
        "soft_0.15": (1, None, 1, None),
        "soft_x_conj_0.15": (1, None, 1, None),
    },
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_UNIFORMS: dict[
    str, dict[str, int]
] = {
    "circle": {
        "youden": 1,
        "soft_0.12": 1,
        "soft_x_conj_0.12": 1,
    },
    "swiss": {
        "youden": 1,
        "soft_0.12": 1,
        "soft_x_conj_0.12": 1,
    },
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_SI_NOTE: str = (
    "A2-T66 denser soft keep-band × require_gabriel_and_h multi-seed "
    "majors/e2e (n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 "
    "gabriel=False, betweenness, seeds 0..2, lean fracs "
    "{0.05,0.12,0.15}, fixed-tau majors 0.27/0.5 + lean tau* e2e "
    "max_grid_points=12 scale_seed=42+seed): T55/T64 denser majors "
    "keep-band soft_frac≤0.12 → tori K=2 chance-ARI≈0.16–0.18 is "
    "seed0-only; seeds 1–2 stay ≤1 under youden/soft/conj (no "
    "multi-seed keep); gabriel conj kills seed0 keep. Lean e2e only "
    "seed0 youden nested K=2 chance-ARI≈0.01; soft/soft×conj and "
    "seeds 1–2 collapse nested+tori to ≤1. Chance-ARI ≠ sample-ARI "
    "recovery; defaults off; no awaiting flip."
)


def format_denser_soft_keep_band_x_gabriel_multiseed_table() -> str:
    """TSV export of denser soft keep×gabriel multi-seed (A2-T66)."""

    lines = [
        "# denser soft keep-band × require_gabriel_and_h multi-seed "
        "majors/e2e",
        f"# nested_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_NESTED_N} "
        f"tori_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_TORI_N} "
        f"max_nodes={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_MAX_NODES} "
        f"h0={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_H0:g} "
        f"method={DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_METHOD} "
        f"max_grid_points="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_MAX_GRID_POINTS} "
        f"scale_seed_base="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_SCALE_SEED_BASE} "
        f"seeds={list(DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_SEEDS)} "
        f"fracs={list(DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_FRACS)} "
        f"keep_max="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_KEEP_MAX_FRAC:g} "
        f"collapse_min="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_COLLAPSE_MIN_FRAC:g}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tmajors_or_leaves\tsample_ari",
    ]
    for seed in DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_SEEDS:
        for mode, (nm, na, tm, ta) in (
            DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"majors\t{seed}\t{mode}\tnested\t"
                f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"majors\t{seed}\t{mode}\ttori\t"
                f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
        for mode, (nl, na, tl, ta) in (
            DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_E2E_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
            lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_UNIFORMS.items()
    ):
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(f"# {DENSER_SOFT_KEEP_BAND_X_GABRIEL_MULTISEED_SI_NOTE}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft×gabriel×persist compose seed1 inflate window (A2-T67)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False,
# soft_frac=0.25 betweenness, seeds 0..2. Fixed-tau majors (0.27/0.5):
# denser kills T63 seed1 soft nested majors inflate (baseline nested K=2
# chance-ARI≈0.08 → denser seed1 soft ≤1); soft_frac=0.25 also collapses
# seed0 denser youden tori K=2 keep. Lean tau* e2e (max_grid_points=12,
# scale_seed=42+seed): denser kills T63 seed1 nested e2e inflate
# (baseline nested K=2 chance-ARI≈0 survives soft×conj×persist → denser
# seed1 all ≤1); only seed0 youden nested K=2 chance-ARI≈0.01 remains
# (same T61/T66 singleton). Circle/swiss stay 1. Soft≠recovery.

DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SEEDS: tuple[int, ...] = (0, 1, 2)
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SOFT_FRAC: float = 0.25
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SOFT_METHOD: str = "betweenness"
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_NESTED_N: int = (
    DENSER_PROPOSED_H0_NESTED_N
)
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_TORI_N: int = (
    DENSER_PROPOSED_H0_TORI_N
)
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_MAX_NODES: int = (
    DENSER_PROPOSED_H0_MAX_NODES
)
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_NESTED_TAU: float = 0.27
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_TORI_TAU: float = 0.5
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_H0: float = PROPOSED_H0_YOUDEN
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_MAX_GRID_POINTS: int = 12
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SCALE_SEED_BASE: int = 42

# seed → mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (1, None, 2, 0.14),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft": (1, None, 1, None),
        "conj": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
    },
}

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_E2E_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.01, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "soft_x_conj": (1, None, 1, None),
        "soft_x_persist": (1, None, 1, None),
        "soft_x_conj_x_persist": (1, None, 1, None),
    },
}

DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_UNIFORMS: dict[
    str, dict[str, int]
] = {
    "circle": {
        "youden": 1,
        "soft_x_conj": 1,
        "soft_x_persist": 1,
        "soft_x_conj_x_persist": 1,
    },
    "swiss": {
        "youden": 1,
        "soft_x_conj": 1,
        "soft_x_persist": 1,
        "soft_x_conj_x_persist": 1,
    },
}

DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SI_NOTE: str = (
    "A2-T67 denser soft×require_gabriel_and_h×persist_agree compose "
    "seed1 inflate window (n=160/240, max_nodes=128, Youden h0≈0.73, "
    "mid=0.5 gabriel=False, soft_frac=0.25 betweenness, fixed-tau majors "
    "0.27/0.5 + lean tau* e2e max_grid_points=12 scale_seed=42+seed): "
    "denser kills T63 seed1 majors soft nested inflate (baseline nested "
    "K=2 chance-ARI≈0.08 → denser seed1 soft ≤1) and T63 seed1 e2e "
    "nested inflate (baseline nested K=2 chance-ARI≈0 survives "
    "soft×conj×persist → denser seed1 all ≤1); soft_frac=0.25 also "
    "collapses denser seed0 youden tori majors keep. Only seed0 youden "
    "remains (majors tori K=2 chance-ARI≈0.14; e2e nested K=2 "
    "chance-ARI≈0.01). Circle/swiss stay 1. Soft≠sample-ARI recovery; "
    "defaults off; no awaiting flip."
)


def format_denser_soft_x_gabriel_x_persist_seed_inflate_table() -> str:
    """TSV export of denser soft×gabriel×persist seed1 inflate (A2-T67)."""

    lines = [
        "# denser soft × require_gabriel_and_h × persist_agree compose "
        "seed1 inflate window",
        f"# nested_n={DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_NESTED_N} "
        f"tori_n={DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_TORI_N} "
        f"max_nodes={DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_MAX_NODES} "
        f"h0={DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_H0:g} "
        f"soft_frac="
        f"{DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SOFT_FRAC:g} "
        f"method={DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SOFT_METHOD} "
        f"max_grid_points="
        f"{DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_MAX_GRID_POINTS} "
        f"scale_seed_base="
        f"{DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SCALE_SEED_BASE} "
        f"seeds={list(DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SEEDS)}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tmajors_or_leaves\tsample_ari",
    ]
    for seed in DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SEEDS:
        for mode, (nm, na, tm, ta) in (
            DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_TABLE[seed].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(
                f"majors\t{seed}\t{mode}\tnested\t"
                f"{DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_NESTED_TAU:g}\t"
                f"{nm}\t{na_s}"
            )
            lines.append(
                f"majors\t{seed}\t{mode}\ttori\t"
                f"{DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_TORI_TAU:g}\t"
                f"{tm}\t{ta_s}"
            )
        for mode, (nl, na, tl, ta) in (
            DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_E2E_TABLE[
                seed
            ].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
            lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in (
        DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_UNIFORMS.items()
    ):
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(
        f"# {DENSER_SOFT_X_GABRIEL_X_PERSIST_SEED_INFLATE_SI_NOTE}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft keep-band × gabriel × persist multi-seed e2e (A2-T68)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False,
# betweenness soft_capacity, seeds 0..2, lean keep fracs {0.05,0.12,0.15}.
# Lean tau* e2e (max_grid_points=12, scale_seed=42+seed): extends T65
# seed0 persist frac grid across seeds — soft×persist /
# soft×conj×persist collapse nested+tori to ≤1 for every seed×frac;
# seeds 1–2 youden also ≤1; only seed0 youden nested K=2
# chance-ARI≈0.01 remains (same T65/T66/T67 singleton). Circle/swiss
# stay 1. Soft≠sample-ARI recovery.

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_SEEDS: tuple[int, ...] = (
    0, 1, 2,
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_FRACS: tuple[float, ...] = (
    0.05, 0.12, 0.15,
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_NESTED_N: int = (
    DENSER_PROPOSED_H0_NESTED_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_TORI_N: int = (
    DENSER_PROPOSED_H0_TORI_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_MAX_NODES: int = (
    DENSER_PROPOSED_H0_MAX_NODES
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_H0: float = (
    PROPOSED_H0_YOUDEN
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_METHOD: str = "betweenness"
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_MAX_GRID_POINTS: int = 12
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_SCALE_SEED_BASE: int = 42
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_KEEP_MAX_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_COLLAPSE_MIN_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC
)

# seed → mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_E2E_TABLE: dict[
    int, dict[str, tuple[int, float | None, int, float | None]]
] = {
    0: {
        "youden": (2, 0.01, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist_0.05": (1, None, 1, None),
        "soft_x_conj_x_persist_0.05": (1, None, 1, None),
        "soft_x_persist_0.12": (1, None, 1, None),
        "soft_x_conj_x_persist_0.12": (1, None, 1, None),
        "soft_x_persist_0.15": (1, None, 1, None),
        "soft_x_conj_x_persist_0.15": (1, None, 1, None),
    },
    1: {
        "youden": (1, None, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist_0.05": (1, None, 1, None),
        "soft_x_conj_x_persist_0.05": (1, None, 1, None),
        "soft_x_persist_0.12": (1, None, 1, None),
        "soft_x_conj_x_persist_0.12": (1, None, 1, None),
        "soft_x_persist_0.15": (1, None, 1, None),
        "soft_x_conj_x_persist_0.15": (1, None, 1, None),
    },
    2: {
        "youden": (1, None, 1, None),
        "persist": (1, None, 1, None),
        "soft_x_persist_0.05": (1, None, 1, None),
        "soft_x_conj_x_persist_0.05": (1, None, 1, None),
        "soft_x_persist_0.12": (1, None, 1, None),
        "soft_x_conj_x_persist_0.12": (1, None, 1, None),
        "soft_x_persist_0.15": (1, None, 1, None),
        "soft_x_conj_x_persist_0.15": (1, None, 1, None),
    },
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_UNIFORMS: dict[
    str, dict[str, int]
] = {
    "circle": {
        "youden": 1,
        "persist": 1,
        "soft_x_persist_0.12": 1,
        "soft_x_conj_x_persist_0.12": 1,
    },
    "swiss": {
        "youden": 1,
        "persist": 1,
        "soft_x_persist_0.12": 1,
        "soft_x_conj_x_persist_0.12": 1,
    },
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_SI_NOTE: str = (
    "A2-T68 denser soft keep-band × require_gabriel_and_h × "
    "persist_agree multi-seed e2e (n=160/240, max_nodes=128, Youden "
    "h0≈0.73, mid=0.5 gabriel=False, betweenness, seeds 0..2, lean "
    "keep fracs {0.05,0.12,0.15}, lean tau* max_grid_points=12 "
    "scale_seed=42+seed): extends T65 seed0 persist frac grid — "
    "soft×persist / soft×conj×persist collapse nested+tori to ≤1 for "
    "every seed×frac; seeds 1–2 youden also ≤1; only seed0 youden "
    "nested K=2 chance-ARI≈0.01 remains (T65/T66/T67 singleton). "
    "Circle/swiss stay 1. Chance-ARI ≠ sample-ARI recovery; defaults "
    "off; no awaiting flip."
)


def format_denser_soft_keep_band_x_gabriel_x_persist_multiseed_table() -> str:
    """TSV export of denser soft keep×gabriel×persist multi-seed e2e (A2-T68)."""

    lines = [
        "# denser soft keep-band × gabriel × persist multi-seed e2e",
        f"# nested_n="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_NESTED_N} "
        f"tori_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_TORI_N} "
        f"max_nodes="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_MAX_NODES} "
        f"h0={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_H0:g} "
        f"method={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_METHOD} "
        f"max_grid_points="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_MAX_GRID_POINTS} "
        f"scale_seed_base="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_SCALE_SEED_BASE} "
        f"seeds="
        f"{list(DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_SEEDS)} "
        f"fracs="
        f"{list(DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_FRACS)} "
        f"keep_max="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_KEEP_MAX_FRAC:g} "
        f"collapse_min="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_COLLAPSE_MIN_FRAC:g}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tleaves\tsample_ari",
    ]
    for seed in DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_SEEDS:
        for mode, (nl, na, tl, ta) in (
            DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_E2E_TABLE[
                seed
            ].items()
        ):
            na_s = "" if na is None else f"{na:.2f}"
            ta_s = "" if ta is None else f"{ta:.2f}"
            lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
            lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_UNIFORMS.items()
    ):
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(
        f"# {DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MULTISEED_SI_NOTE}"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Denser soft keep×gabriel seed0-only keep × soft×persist majors pin (A2-T69)
# ---------------------------------------------------------------------------
# denser n=160/240, max_nodes=128, Youden h0≈0.73, mid=0.5 gabriel=False,
# betweenness soft_capacity, seed0 only, lean keep fracs {0.05,0.12,0.15}.
# Fixed-tau majors (0.27/0.5): T55/T64/T66 seed0-only denser keep-band
# soft_frac≤0.12 → tori K=2 chance-ARI≈0.16–0.18; gabriel conj kills
# keep; soft≥0.15 collapses. Lean tau* e2e (max_grid_points=12,
# scale_seed=42): soft×persist / soft×conj×persist collapse nested+tori
# to ≤1 across keep/collapse fracs — denser seed0 keep is majors-only
# under persist compose (T61 denser+gabriel pin). Youden alone leaves
# nested K=2 chance-ARI≈0.01. Circle/swiss stay 1. Soft≠recovery.

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_SEED: int = 0
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_FRACS: tuple[
    float, ...
] = (
    0.05, 0.12, 0.15,
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_NESTED_N: int = (
    DENSER_PROPOSED_H0_NESTED_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_TORI_N: int = (
    DENSER_PROPOSED_H0_TORI_N
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_MAX_NODES: int = (
    DENSER_PROPOSED_H0_MAX_NODES
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_NESTED_TAU: float = 0.27
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_TORI_TAU: float = 0.5
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_H0: float = (
    PROPOSED_H0_YOUDEN
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_METHOD: str = (
    "betweenness"
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_MAX_GRID_POINTS: int = 12
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_SCALE_SEED_BASE: int = 42
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_KEEP_MAX_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_KEEP_MAX_FRAC
)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_COLLAPSE_MIN_FRAC: float = (
    DENSER_SOFT_SEED0_TORI_ARI_WINDOW_COLLAPSE_MIN_FRAC
)

# Fixed-tau majors: mode → (nested_majors, nested_ari, tori_majors, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (1, None, 2, 0.14),
    "conj": (1, None, 1, None),
    "soft_0.05": (1, None, 2, 0.16),
    "soft_x_conj_0.05": (1, None, 1, None),
    "soft_0.12": (1, None, 2, 0.18),
    "soft_x_conj_0.12": (1, None, 1, None),
    "soft_0.15": (1, None, 1, None),
    "soft_x_conj_0.15": (1, None, 1, None),
}

# E2E soft×persist pin (seed0): soft×persist / soft×conj×persist ≤1;
# youden alone keeps nested K=2 chance-ARI≈0.01.
# mode → (nested_leaves, nested_ari, tori_leaves, tori_ari)
DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_E2E_TABLE: dict[
    str, tuple[int, float | None, int, float | None]
] = {
    "youden": (2, 0.01, 1, None),
    "persist": (1, None, 1, None),
    "soft_x_persist_0.05": (1, None, 1, None),
    "soft_x_conj_x_persist_0.05": (1, None, 1, None),
    "soft_x_persist_0.12": (1, None, 1, None),
    "soft_x_conj_x_persist_0.12": (1, None, 1, None),
    "soft_x_persist_0.15": (1, None, 1, None),
    "soft_x_conj_x_persist_0.15": (1, None, 1, None),
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_UNIFORMS: dict[
    str, dict[str, int]
] = {
    "circle": {
        "youden": 1,
        "persist": 1,
        "soft_x_persist_0.12": 1,
        "soft_x_conj_x_persist_0.12": 1,
    },
    "swiss": {
        "youden": 1,
        "persist": 1,
        "soft_x_persist_0.12": 1,
        "soft_x_conj_x_persist_0.12": 1,
    },
}

DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_SI_NOTE: str = (
    "A2-T69 denser soft keep-band × require_gabriel_and_h seed0-only "
    "keep × soft×persist majors pin (n=160/240, max_nodes=128, Youden "
    "h0≈0.73, mid=0.5 gabriel=False, betweenness, seed0, lean keep "
    "fracs {0.05,0.12,0.15}, fixed-tau majors 0.27/0.5 + lean tau* "
    "max_grid_points=12 scale_seed=42): T55/T64/T66 seed0 denser majors "
    "keep soft_frac≤0.12 → tori K=2 chance-ARI≈0.16–0.18; gabriel conj "
    "kills keep; soft≥0.15 collapses. Soft×persist / soft×conj×persist "
    "e2e collapse nested+tori to ≤1 across keep/collapse fracs — denser "
    "seed0 keep is majors-only under persist compose (T61 denser+gabriel "
    "pin). Youden alone leaves nested K=2 chance-ARI≈0.01. Circle/swiss "
    "stay 1. Chance-ARI ≠ sample-ARI recovery; defaults off; no "
    "awaiting flip."
)


def format_denser_soft_keep_band_x_gabriel_x_persist_majors_pin_table() -> str:
    """TSV export of denser soft keep×gabriel×persist majors pin (A2-T69)."""

    lines = [
        "# denser soft keep-band × gabriel seed0-only keep × "
        "soft×persist majors pin",
        f"# nested_n="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_NESTED_N} "
        f"tori_n={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_TORI_N} "
        f"max_nodes="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_MAX_NODES} "
        f"h0={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_H0:g} "
        f"method={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_METHOD} "
        f"max_grid_points="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_MAX_GRID_POINTS} "
        f"scale_seed_base="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_SCALE_SEED_BASE} "
        f"seed={DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_SEED} "
        f"fracs="
        f"{list(DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_FRACS)} "
        f"keep_max="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_KEEP_MAX_FRAC:g} "
        f"collapse_min="
        f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_COLLAPSE_MIN_FRAC:g}",
        "surface\tseed\tmode\tdataset\ttau_or_e2e\tmajors_or_leaves\tsample_ari",
    ]
    seed = DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_SEED
    for mode, (nm, na, tm, ta) in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(
            f"majors\t{seed}\t{mode}\tnested\t"
            f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_NESTED_TAU:g}\t"
            f"{nm}\t{na_s}"
        )
        lines.append(
            f"majors\t{seed}\t{mode}\ttori\t"
            f"{DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_TORI_TAU:g}\t"
            f"{tm}\t{ta_s}"
        )
    for mode, (nl, na, tl, ta) in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_E2E_TABLE.items()
    ):
        na_s = "" if na is None else f"{na:.2f}"
        ta_s = "" if ta is None else f"{ta:.2f}"
        lines.append(f"e2e\t{seed}\t{mode}\tnested\ttau*\t{nl}\t{na_s}")
        lines.append(f"e2e\t{seed}\t{mode}\ttori\ttau*\t{tl}\t{ta_s}")
    lines.append("dataset\tmode\tleaves")
    for dataset, mode_table in (
        DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_UNIFORMS.items()
    ):
        for mode, leaves in mode_table.items():
            lines.append(f"{dataset}\t{mode}\t{leaves}")
    lines.append(
        f"# {DENSER_SOFT_KEEP_BAND_X_GABRIEL_X_PERSIST_MAJORS_PIN_SI_NOTE}"
    )
    return "\n".join(lines)


def a4_roc_primary_config(**overrides: object) -> HollowEdgeConfig:
    """A4 sheet/bridge ROC primary preset (OPEN_ISSUES #44 / A2-T33).

    Primary preference from ``recommend_hollow_edge_configs``: mid=0.5,
    h0=0.7, gabriel off, min_end=0.5.  Operational / proposal-path until
    sample-ARI recovery is demonstrated; never the RecursionConfig default.
    """

    base = dict(
        mid_radius_frac=A4_PRIMARY_MID_RADIUS_FRAC,
        h0=A4_PRIMARY_H0,
        min_end_count=A4_PRIMARY_MIN_END_COUNT,
        gabriel_fallback=A4_PRIMARY_GABRIEL_FALLBACK,
        require_gabriel_and_h=False,
        mst_critical_only=False,
        bridge_critical_only=False,
        soft_capacity_only=False,
        soft_capacity_frac=0.25,
        soft_capacity_method="betweenness",
    )
    base.update(overrides)
    return HollowEdgeConfig(**base)  # type: ignore[arg-type]


def edge_ball_occupancy(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    *,
    mid_radius_frac: float = 0.35,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(n_mid, n_end, lengths)`` per edge for hollow diagnostics.

    ``n_end`` is the mean of the endpoint-ball counts (same balls as
    :func:`hollowness_scores`).  Used to detect the empty-ball regime
    where ``H`` collapses without discriminating bridges.
    """

    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    if pts.ndim != 2:
        raise ValueError("data must be 2-D")
    frac = float(mid_radius_frac)
    if frac <= 0.0:
        raise ValueError("mid_radius_frac must be positive")
    n_mid = np.empty(len(edges), dtype=float)
    n_end = np.empty(len(edges), dtype=float)
    lengths = np.empty(len(edges), dtype=float)
    for k, (i, j) in enumerate(edges):
        xi = pos[int(i)]
        xj = pos[int(j)]
        length = float(np.linalg.norm(xi - xj))
        lengths[k] = length
        if length <= 0.0:
            n_mid[k] = 0.0
            n_end[k] = 0.0
            continue
        radius = frac * length
        mid = 0.5 * (xi + xj)
        n_mid[k] = float(np.sum(np.linalg.norm(pts - mid, axis=1) <= radius))
        n_i = float(np.sum(np.linalg.norm(pts - xi, axis=1) <= radius))
        n_j = float(np.sum(np.linalg.norm(pts - xj, axis=1) <= radius))
        n_end[k] = 0.5 * (n_i + n_j)
    return n_mid, n_end, lengths


def hollowness_scores(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    *,
    mid_radius_frac: float = 0.35,
    eps: float = _EPS,
) -> np.ndarray:
    """Return ``H(i,j) = n_mid / (n_end + eps)`` for each edge.

    Parameters
    ----------
    positions:
        ``(n_nodes, d)`` scaffold node positions.
    edges:
        Lifted undirected edges as ``(i, j)`` index pairs.
    data:
        ``(n_samples, d)`` raw sample positions (data-side evidence).
    mid_radius_frac:
        Mid / endpoint ball radius as a fraction of edge length ``L``.
    """

    n_mid, n_end, lengths = edge_ball_occupancy(
        positions, edges, data, mid_radius_frac=mid_radius_frac,
    )
    scores = np.empty(len(edges), dtype=float)
    for k in range(len(edges)):
        if lengths[k] <= 0.0:
            scores[k] = 1.0
        else:
            scores[k] = float(n_mid[k]) / (float(n_end[k]) + float(eps))
    return scores


def gabriel_diameter_empty(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
) -> np.ndarray:
    """True iff the open diameter ball of edge ``(i,j)`` contains no data.

    Used as the low-``n_end`` fallback: empty diameter ⇒ treat as hollow bridge
    (cut).  This is the geometric emptiness test, not construction of the
    Gabriel graph (which *keeps* empty-diameter edges).
    """

    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    out = np.zeros(len(edges), dtype=bool)
    for k, (i, j) in enumerate(edges):
        xi = pos[int(i)]
        xj = pos[int(j)]
        length = float(np.linalg.norm(xi - xj))
        if length <= 0.0:
            out[k] = False
            continue
        mid = 0.5 * (xi + xj)
        radius = 0.5 * length
        out[k] = not bool(np.any(np.linalg.norm(pts - mid, axis=1) < radius - 1e-12))
    return out


def mst_edge_mask(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
) -> np.ndarray:
    """Boolean mask ``True`` iff edge is in a Euclidean MST (Kruskal).

    Used by ``mst_critical_only`` hollow pruning (A2-T34): only MST edges
    are capacity-critical bridges in a tree sense; cutting non-MST hollow
    edges often leaves redundant Hebbian paths (non-cut-set failure mode).
    """

    pos = np.asarray(positions, dtype=float)
    n = int(pos.shape[0])
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    ranked: list[tuple[float, int, int, int]] = []
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj:
            continue
        length = float(np.linalg.norm(pos[ii] - pos[jj]))
        ranked.append((length, k, ii, jj))
    ranked.sort(key=lambda t: t[0])
    in_mst = np.zeros(len(edges), dtype=bool)
    used = 0
    for _, k, ii, jj in ranked:
        if union(ii, jj):
            in_mst[k] = True
            used += 1
            if used >= max(0, n - 1):
                break
    return in_mst


def bridge_edge_mask(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Boolean mask ``True`` iff edge is a bridge of the undirected graph.

    Tarjan-style DFS discovery: an edge ``(u,v)`` is a bridge when it is a
    tree edge and ``low[v] > disc[u]`` (no back-edge from ``v``'s subtree
    reaches ``u`` or an ancestor).  Used by ``bridge_critical_only`` hollow
    pruning (capacity/flow beyond MST): only true cut-set edges may be cut.
    """

    if not edges:
        return np.zeros(0, dtype=bool)
    if n_nodes is None:
        n_nodes = 0
        for i, j in edges:
            n_nodes = max(n_nodes, int(i) + 1, int(j) + 1)
    n = int(n_nodes)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj or ii < 0 or jj < 0 or ii >= n or jj >= n:
            continue
        adj[ii].append((jj, k))
        adj[jj].append((ii, k))

    disc = [-1] * n
    low = [-1] * n
    parent = [-1] * n
    time = 0
    is_bridge = np.zeros(len(edges), dtype=bool)

    def dfs(u: int) -> None:
        nonlocal time
        disc[u] = time
        low[u] = time
        time += 1
        for v, ek in adj[u]:
            if disc[v] == -1:
                parent[v] = u
                dfs(v)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    is_bridge[ek] = True
            elif v != parent[u]:
                low[u] = min(low[u], disc[v])

    for s in range(n):
        if disc[s] == -1 and adj[s]:
            dfs(s)
    return is_bridge


def edge_betweenness_scores(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Brandes edge betweenness on the undirected multigraph of ``edges``.

    Soft capacity / flow proxy (A2-T37): high-betweenness edges carry more
    shortest paths and approximate min-cut mass without requiring a hard
    bridge.  Returns one score per input edge (0 for self-loops / OOB).
    """

    from collections import deque

    if not edges:
        return np.zeros(0, dtype=float)
    if n_nodes is None:
        n_nodes = 0
        for i, j in edges:
            n_nodes = max(n_nodes, int(i) + 1, int(j) + 1)
    n = int(n_nodes)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj or ii < 0 or jj < 0 or ii >= n or jj >= n:
            continue
        adj[ii].append((jj, k))
        adj[jj].append((ii, k))

    cb = np.zeros(len(edges), dtype=float)
    for s in range(n):
        if not adj[s]:
            continue
        stack: list[int] = []
        pred: list[list[tuple[int, int]]] = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=float)
        sigma[s] = 1.0
        dist = [-1] * n
        dist[s] = 0
        q: deque[int] = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w, ek in adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append((v, ek))
        delta = np.zeros(n, dtype=float)
        while stack:
            w = stack.pop()
            for v, ek in pred[w]:
                if sigma[w] > 0.0:
                    c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                else:
                    c = 0.0
                cb[ek] += c
                delta[v] += c
    # Undirected convention: each undirected edge counted twice.
    return cb * 0.5


def bridge_mass_scores(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
) -> np.ndarray:
    """Min-cut mass scores: bridge ``min(|comp_u|,|comp_v|)``, else 0.

    Operational soft-capacity alternative to Brandes betweenness
    (A2-T39).  Only true bridges carry positive mass; the mass equals
    the smaller side of the cut after removing that edge (unit-capacity
    global min-cut contribution when the edge is the unique cut edge).
    """

    if not edges:
        return np.zeros(0, dtype=float)
    if n_nodes is None:
        n_nodes = 0
        for i, j in edges:
            n_nodes = max(n_nodes, int(i) + 1, int(j) + 1)
    n = int(n_nodes)
    is_br = bridge_edge_mask(edges, n_nodes=n)
    scores = np.zeros(len(edges), dtype=float)
    if not np.any(is_br):
        return scores

    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (i, j) in enumerate(edges):
        ii, jj = int(i), int(j)
        if ii == jj or ii < 0 or jj < 0 or ii >= n or jj >= n:
            continue
        adj[ii].append((jj, k))
        adj[jj].append((ii, k))

    for k, (i, j) in enumerate(edges):
        if not bool(is_br[k]):
            continue
        ii, jj = int(i), int(j)
        # BFS from ii avoiding edge k; mass = min(|reach|, n-|reach|).
        seen = [False] * n
        stack = [ii]
        seen[ii] = True
        reached = 0
        while stack:
            u = stack.pop()
            reached += 1
            for v, ek in adj[u]:
                if ek == k or seen[v]:
                    continue
                seen[v] = True
                stack.append(v)
        scores[k] = float(min(reached, n - reached))
    return scores


def soft_capacity_edge_mask(
    edges: list[tuple[int, int]],
    *,
    n_nodes: int | None = None,
    frac: float = 0.25,
    method: str = "betweenness",
) -> np.ndarray:
    """Boolean mask ``True`` iff capacity score ≥ ``frac * max``.

    Operational soft-capacity gate (A2-T37 / A2-T39).  ``method`` is
    ``"betweenness"`` (Brandes) or ``"bridge_mass"`` (min-cut mass on
    bridges).  ``frac`` in ``(0, 1]``; values ≤0 keep all edges, values
    >1 keep none with positive max.
    """

    if not edges:
        return np.zeros(0, dtype=bool)
    m = str(method).strip().lower()
    if m in ("bridge_mass", "mincut_mass", "min_cut_mass"):
        scores = bridge_mass_scores(edges, n_nodes=n_nodes)
    else:
        scores = edge_betweenness_scores(edges, n_nodes=n_nodes)
    f = float(frac)
    if f <= 0.0:
        return np.ones(len(edges), dtype=bool)
    peak = float(np.max(scores)) if scores.size else 0.0
    if peak <= 0.0:
        return np.zeros(len(edges), dtype=bool)
    return scores >= (f * peak)


def hollow_edge_mask(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    config: HollowEdgeConfig | None = None,
) -> np.ndarray:
    """Boolean mask ``True`` = cut (hollow) for each edge.

    Default rule (``require_gabriel_and_h=False``):
    - ``n_end >= min_end_count`` → cut iff ``H < h0``;
    - else if ``gabriel_fallback`` → cut iff Gabriel diameter ball is empty;
    - else → keep.

    Conjunction rule (``require_gabriel_and_h=True``, A2-T31 / A4 ROC):
    cut iff ``H < h0`` **and** Gabriel diameter ball is empty.  Suppresses
    empty-ball Gabriel-only spurious cuts; keep proposal-path / default-off.

    When ``mst_critical_only`` is set, intersect the hollow mask with the
    Euclidean MST edge mask (A2-T34).  When ``bridge_critical_only`` is set,
    further (or instead) intersect with graph-theoretic bridges (capacity /
    flow cut-set beyond the MST proxy).      When ``soft_capacity_only`` is set,
    intersect with high soft-capacity scores (A2-T37 betweenness /
    A2-T39 bridge-mass min-cut; see ``soft_capacity_method``).
    """

    cfg = config if config is not None else HollowEdgeConfig()
    pos = np.asarray(positions, dtype=float)
    pts = np.asarray(data, dtype=float)
    H = hollowness_scores(
        pos, edges, pts,
        mid_radius_frac=float(cfg.mid_radius_frac),
        eps=float(cfg.eps),
    )
    _, end_mass, _ = edge_ball_occupancy(
        pos, edges, pts, mid_radius_frac=float(cfg.mid_radius_frac),
    )

    need_gab = bool(cfg.gabriel_fallback) or bool(cfg.require_gabriel_and_h)
    gab = (
        gabriel_diameter_empty(pos, edges, pts)
        if need_gab
        else np.zeros(len(edges), dtype=bool)
    )
    min_end = float(cfg.min_end_count)
    h0 = float(cfg.h0)
    cut = np.zeros(len(edges), dtype=bool)
    if cfg.require_gabriel_and_h:
        for k in range(len(edges)):
            cut[k] = bool(H[k] < h0) and bool(gab[k])
    else:
        for k in range(len(edges)):
            if end_mass[k] >= min_end:
                cut[k] = bool(H[k] < h0)
            elif cfg.gabriel_fallback:
                cut[k] = bool(gab[k])
            else:
                cut[k] = False
    if cfg.mst_critical_only and len(edges) > 0:
        cut = np.logical_and(cut, mst_edge_mask(pos, edges))
    if cfg.bridge_critical_only and len(edges) > 0:
        cut = np.logical_and(
            cut, bridge_edge_mask(edges, n_nodes=int(pos.shape[0])),
        )
    if cfg.soft_capacity_only and len(edges) > 0:
        cut = np.logical_and(
            cut,
            soft_capacity_edge_mask(
                edges,
                n_nodes=int(pos.shape[0]),
                frac=float(cfg.soft_capacity_frac),
                method=str(cfg.soft_capacity_method),
            ),
        )
    return cut


def prune_hollow_edges(
    positions: np.ndarray,
    edges: list[tuple[int, int]],
    data: np.ndarray,
    config: HollowEdgeConfig | None = None,
) -> list[tuple[int, int]]:
    """Return lifted edges that survive hollow-edge pruning."""

    if not edges:
        return []
    cut = hollow_edge_mask(positions, edges, data, config=config)
    return [e for e, c in zip(edges, cut) if not bool(c)]
