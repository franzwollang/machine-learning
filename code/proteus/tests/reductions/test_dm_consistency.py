"""DM gate consistency reduction test (SI S3.5)."""
from __future__ import annotations

import numpy as np

from proteus.evidence import evaluate_edit, f_dm
from proteus.types import EditProposal, EditType
from tests.harness.dm_fixtures import split_regions


def _proposal() -> EditProposal:
    return EditProposal(EditType.SPLIT, [0, 1], diagnostic_strength=1.0)


def test_dm_selects_true_topology():
    """As H_R -> inf, the DM marginal must select the correct topology among a
    finite candidate set with probability -> 1 (SI S3.5).

    True generative model = two sub-populations (split). The gate must prefer the
    split when substructure is real, and reject it when it is absent."""

    trials = 25

    # (a) Real substructure: the split is the true topology and must be selected.
    rng_a = np.random.default_rng(11)
    accepted = 0
    for _ in range(trials):
        keep, split = split_regions(4000, sep=0.97, rng=rng_a)
        verdict = evaluate_edit(keep, split, _proposal(), tau_bf=3.0)
        # F_DM of the true (split) topology is strictly the smaller score.
        assert verdict.f_dm_edit < verdict.f_dm_keep
        accepted += int(verdict.accepted)
    assert accepted == trials  # probability -> 1 at large H_R

    # (b) No substructure: the single-node (keep) topology is true; the gate must
    #     not over-split. The Occam factor keeps expected evidence against the
    #     split (mean log-BF < 0), and the fixed log(tau_BF) margin holds the
    #     false-positive rate low. (A finite margin gives a nonzero FP rate by
    #     design -- that is exactly what tau_BF controls -- so we bound the rate
    #     over many trials rather than demanding zero.)
    rng_b = np.random.default_rng(29)
    null_trials = 400
    margin = np.log(3.0)
    bfs_null = np.empty(null_trials)
    for t in range(null_trials):
        keep, split = split_regions(4000, sep=0.5, rng=rng_b)
        verdict = evaluate_edit(keep, split, _proposal(), tau_bf=3.0)
        bfs_null[t] = verdict.log_bayes_factor
    assert bfs_null.mean() < 0.0
    false_positive_rate = float(np.mean(bfs_null > margin))
    assert false_positive_rate < 0.02


def test_dm_margin_dominates_occam():
    """The O(H_R * Delta_R) likelihood margin must dominate the O(log H_R) Occam
    factor and the fixed log(tau_BF) threshold (SI S3.5)."""

    rng = np.random.default_rng(5)
    sizes = np.array([500, 1000, 2000, 4000, 8000], dtype=float)
    margins = []
    for n in sizes:
        keep, split = split_regions(int(n), sep=0.98, rng=rng)
        verdict = evaluate_edit(keep, split, _proposal(), tau_bf=3.0)
        margins.append(verdict.log_bayes_factor)
    margins = np.array(margins)

    # Linear (not logarithmic) growth: margin / H_R approaches a positive constant
    # (Delta_R ~ log 2), so consecutive ratios track the size ratios, and the
    # margin dwarfs both the log-Occam term and the fixed threshold.
    per_event = margins / sizes
    assert np.all(per_event > 0.4)                      # ~ log 2 = 0.693
    # Grows far faster than the Occam/threshold terms it must dominate.
    assert margins[-1] > 50.0 * np.log(sizes[-1])
    assert margins[-1] > 100.0 * np.log(3.0)
    # Monotone increasing in H_R.
    assert np.all(np.diff(margins) > 0.0)
