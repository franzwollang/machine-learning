"""Diagnostic test for Stage 1 scaffold overgrowth.

Run with ``pipenv run pytest tests/stage1/test_growth_diagnostic.py -s``
to see per-epoch tables and per-split detail.  This test does not assert
node-count bounds; it exists to produce diagnostic output.
"""
from __future__ import annotations

from collections import Counter

import pytest

import numpy as np

from proteus.stage1 import Stage1Scaffold
from proteus.stage1.splits import propose_splits, apply_split
from proteus.stage1.pruning import prune_links, prune_nodes
from proteus.stage1.stabilization import (
    StabilizationConfig,
    compute_neighbor_normalized_cv,
)
from tests.datasets.synthetic.circles import make_circle
from tests.metrics.reconstruction import mean_min_distance


@pytest.mark.slow
def test_circle_growth_diagnostic() -> None:
    dataset = make_circle(
        n_samples=1200,
        radius=1.0,
        noise=0.02,
        extrusion_dim=2,
        seed=21,
    )
    data = dataset.points
    gt = dataset.ground_truth
    tau = gt.expected_tau
    assert tau is not None
    expected_n = gt.expected_node_count
    assert expected_n is not None

    safety_cap = 10 * expected_n
    scaffold = Stage1Scaffold(
        dim=gt.ambient_dim,
        tau=tau,
        k=8,
        min_nodes=4,
        max_nodes=safety_cap,
        prune_after=10,
        ann_backend="naive",
        enable_topology_edits=False,
        rng=np.random.default_rng(77),
    )
    scaffold.init_from(data, n_seeds=8)
    scaffold.enable_topology_edits = True

    print()
    print("=" * 90)
    print(
        f"DIAGNOSTIC: circle growth  |  "
        f"expected_tau={tau:.6f}  expected_nodes={expected_n}  "
        f"safety_cap={safety_cap}"
    )
    print(
        f"{'epoch':>5}  {'pre':>4}  {'prop':>4}  {'split':>5}  "
        f"{'prune':>5}  {'post':>4}  {'overcap%':>8}  "
        f"{'mean_v/t':>8}  {'overcap_v/t':>11}  "
        f"{'mean_tau_l':>10}  {'cv':>8}  {'mmd':>8}"
    )
    print("-" * 90)

    split_log: list[dict] = []

    for epoch in range(30):
        scaffold.run_epoch(data)
        if epoch % 2 == 0:
            scaffold.refresh_intrinsic_dim()

        pre_count = len(scaffold.nodes)

        proposals = propose_splits(scaffold)
        accepted = 0
        for proposal in proposals:
            if len(scaffold.nodes) >= safety_cap:
                break
            parent = scaffold.nodes[proposal.node_id]
            parent_var = parent.variance
            parent_tau_l = float(scaffold.tau_local[proposal.node_id])
            parent_hit = parent.hit_count
            active_hits = [
                n.hit_count for n in scaffold.nodes
                if getattr(n, "update_count", 0) >= scaffold.prune_after
            ]
            mean_active = float(np.mean(active_hits)) if active_hits else 0.0

            if apply_split(scaffold, proposal):
                accepted += 1
                child = scaffold.nodes[-1]
                split_log.append({
                    "epoch": epoch,
                    "parent_id": proposal.node_id,
                    "parent_var": parent_var,
                    "parent_tau_l": parent_tau_l,
                    "parent_v_over_t": parent_var / max(parent_tau_l, 1e-12),
                    "parent_hit": parent_hit,
                    "mean_active_hit": mean_active,
                    "hit_ratio": parent_hit / max(mean_active, 1e-12),
                    "child_var": child.variance,
                    "child_tau_l": float(scaffold.tau_local[-1]),
                    "child_v_over_t": child.variance / max(float(scaffold.tau_local[-1]), 1e-12),
                })

        if accepted:
            scaffold.rebuild_ann()

        prune_links(scaffold)
        removed = prune_nodes(scaffold)

        post_count = len(scaffold.nodes)
        variances = np.array([n.variance for n in scaffold.nodes])
        tau_locals = np.asarray(scaffold.tau_local[:post_count], dtype=float)
        v_over_t = variances / np.maximum(tau_locals, 1e-12)
        overcap_mask = variances > tau_locals
        overcap_frac = float(overcap_mask.mean()) if post_count else 0.0
        mean_vt = float(v_over_t.mean()) if post_count else 0.0
        overcap_vt = float(v_over_t[overcap_mask].mean()) if overcap_mask.any() else 0.0
        mean_tau_l = float(tau_locals.mean()) if post_count else 0.0
        cv = compute_neighbor_normalized_cv(scaffold)
        mmd = mean_min_distance(data, scaffold.node_positions())

        d_final_counts = Counter(int(n.d_final) for n in scaffold.nodes)

        print(
            f"{epoch:5d}  {pre_count:4d}  {len(proposals):4d}  "
            f"{accepted:5d}  {len(removed):5d}  {post_count:4d}  "
            f"{overcap_frac:8.3f}  {mean_vt:8.3f}  {overcap_vt:11.3f}  "
            f"{mean_tau_l:10.6f}  {cv:8.4f}  {mmd:8.5f}"
        )

    print()
    print(f"Final node count: {len(scaffold.nodes)}")
    print(f"Expected ideal:   {expected_n}")
    print(f"2x upper bound:   {gt.max_ent_node_upper(2.0)}")
    print(f"Safety cap:       {safety_cap}")
    hit_cap = len(scaffold.nodes) >= safety_cap
    print(f"Hit safety cap:   {hit_cap}")

    still_overcap = sum(
        1 for i, n in enumerate(scaffold.nodes)
        if n.variance > scaffold.tau_local[i]
    )
    print(f"Nodes still over-cap at end: {still_overcap}")

    d_final_counts = Counter(int(n.d_final) for n in scaffold.nodes)
    print(f"d_final distribution: {dict(d_final_counts)}")
    print(f"tau_local unique: {np.unique(np.round(scaffold.tau_local, 8))}")
    print(f"expected_tau: {tau}")

    if split_log:
        print()
        last_epochs = sorted(set(s["epoch"] for s in split_log))[-3:]
        recent = [s for s in split_log if s["epoch"] in last_epochs]
        print(f"Per-split detail (last {len(recent)} splits from epochs {last_epochs}):")
        for s in recent:
            print(
                f"  epoch={s['epoch']} parent={s['parent_id']} "
                f"var/tau={s['parent_v_over_t']:.3f} "
                f"hit_ratio={s['hit_ratio']:.2f} "
                f"child_var/tau={s['child_v_over_t']:.3f}"
            )

    print("=" * 90)
