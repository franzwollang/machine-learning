# Proteus Paper 1 — Open Issues (Implementation-Guide View)

_This document tracks mathematical and architectural issues in the foundational paper (`paper.tex` + `SI.tex`) that could cause **implementation showstoppers or theoretical dead-ends** during development. It is not oriented toward external publication. The goal is to nail down the maths tightly enough that a first implementation has a reasonable chance of converging and being debuggable._

_Scope: specification gaps, under-determined algorithms, identifiability / conditioning risks, and convergence concerns. Out of scope: submission formatting, venue positioning, baselines, empirical benchmarks, and pure parameter-tuning discussions (those live in `bnp_design_implications.md` and `bnp_upgrades_math.tex`)._

---

## Priority 1 — Showstopper risks (must resolve before first implementation)

These are points where the current maths is either (a) not concrete enough to code, or (b) has a plausible failure mode that would invalidate a naive implementation.

### 3. Dynamic convergence: variance-cap NG monotonicity (foundational lemma; Lemma 1 of S15.1)

_Status: only foundational lemma open; auxiliary lemmas 2--4 proved in SI §S15.1. Does not block first implementation._

#### Architectural status

The Lyapunov framework is fully identified and three of four supporting lemmas are now proved in SI §S15.1. The single remaining foundational result is **Lemma 1**: that the dual-rate motion of S2.3 produces monotone decrease in the vector-quantization energy

```text
L_vertex({w_i}) = Σ_i ∫_{V_i} ||x - w_i||² p(x) dx
```

between split events. This is the variance-cap variant of the classical Neural Gas convergence theorem.

#### Precise statement to prove

> **Lemma 1.** Under the dual-rate motion of S2.3 with EWMA half-life `k` (S2.2), within a fixed topology `M` and between split events, `L_vertex(t)` is a stochastic Lyapunov function: there exist `c_t ≥ 0` with `Σ_t c_t = ∞` (driving condition) and step-size sequences satisfying the Robbins–Monro conditions `Σ η_t = ∞` and `Σ η_t² < ∞`, such that
>
> ```text
> E[L_vertex(t+1) | F_t] ≤ L_vertex(t) - c_t.
> ```

Granting this, Lemmas 2–4 (already proved in SI §S15.1) together establish that the joint potential `L = L_vertex + F_DM` is a Lyapunov function for the full variance-cap + evidence-gated dynamics.

#### Background and prior art

**Classical Neural Gas (Martinetz & Schulten 1991).** Original NG uses a fixed annealing schedule for both the neighborhood-decay parameter `λ_t` and the learning rate `η_t`:

```text
λ_t = λ_0 (λ_T / λ_0)^(t/T)
η_t = η_0 (η_T / η_0)^(t/T)
```

Convergence is proved in expectation under standard stochastic-approximation conditions (Robbins–Monro): `Σ η_t = ∞` and `Σ η_t² < ∞`. The Lyapunov function is exactly `L_vertex` — the quantization energy.

**Growing Neural Gas (Fritzke 1995).** Adds discrete topology changes (insertion of new nodes when accumulated error exceeds a threshold). Convergence is asserted by simulation and analogy to NG; a formal Lyapunov argument is not given because the discrete events are heuristically gated.

**Other adaptive variants.** "Online k-means with adaptive learning rates" (e.g., Bottou & Bengio 1995; Sato 1999) use diminishing step sizes tied to per-cluster sample counts. Convergence proofs follow the standard SA template.

**What's different about Proteus.** The variance cap replaces both the annealing schedule and the discrete error threshold:

```text
η_GNG,i(t) = (ln 2 / 2k) (1 - σ_i²(t) / τ)         [S2.3]
η_cent     = κ (1 - r) / k                          [S2.3]
```

Here `η_GNG,i(t) → 0` as `σ_i²(t) → τ_local,i` automatically, without an explicit time-dependent schedule. The challenge is that `σ_i²(t)` is itself a stochastic process (an EWMA of squared residuals), not a deterministic decreasing schedule.

#### Why the standard NG proof doesn't directly apply

The Robbins–Monro analysis of NG uses the schedule `η_t` directly: shrinking `η_t` deterministically guarantees the second-moment condition `Σ η_t² < ∞`. In the variance-cap variant, `η_GNG,i` depends on `σ_i²(t)`, which:

- Decreases on average as variance reduces toward the cap (the dynamics drive `σ_i² → τ`).
- Has stochastic fluctuations that, in principle, could cause `η_GNG,i` to remain bounded away from zero indefinitely if the EWMA never cleanly stabilizes.

Establishing `Σ η²_GNG,i(t)² < ∞` therefore requires showing that `σ_i²(t)` **converges to** `τ_local,i` rather than merely fluctuating around it. This is a self-referential statement: the proof needs `η → 0` to establish convergence, but `η → 0` requires convergence to be established.

#### Proof template (suggested approach)

The natural template is the **two-time-scale stochastic approximation** framework of Borkar (2008, _Stochastic Approximation: A Dynamical Systems Viewpoint_) or Kushner & Yin (2003). Two-time-scale SA handles exactly this self-referential structure: position updates (slow time scale, governed by `η_GNG,i`) and variance estimates (fast time scale, governed by EWMA half-life `k`).

Outline:

1. **Identify the two time scales.** Position `w_i(t)` evolves on a slow time scale set by `η_GNG,i ∝ (1 - σ_i²/τ)/k`. Variance estimate `σ_i²(t)` evolves on a fast time scale with EWMA half-life `k`. The slow → fast separation is by a factor of `(1 - σ_i²/τ)`, which is small near equilibrium (key insight).

2. **Fast-scale ODE limit.** Under the slow-scale freeze, the EWMA `σ_i²(t)` has a deterministic ODE limit `dσ_i²/dt = -(σ_i² - σ_i,*²)/k`, where `σ_i,*²` is the local conditional second moment at fixed `w_i`. This converges geometrically, half-life `k`.

3. **Slow-scale ODE limit.** With `σ_i² ≈ σ_i,*²(w)` (fast variable equilibrated), position updates follow `dw_i/dt = E[a_i(t) | w] · constant`. The right-hand side is the gradient of `L_vertex` with respect to `w_i`, scaled by `(1 - σ_i,*²(w)/τ)/k`.

4. **Lyapunov on the slow ODE.** `L_vertex` is decreasing along the slow ODE except at fixed points (gradient flow). At fixed points, either `σ_i,*²(w) = τ_local,i` (stable equilibrium) or the gradient vanishes for other reasons.

5. **Discrete jumps from splits.** Splits are bounded jumps in topology with bounded effect on `L_vertex`. Standard SA-with-jumps theory (Borkar 2008, Ch. 11) handles these: the Lyapunov decrease is preserved across jumps if each jump satisfies a bounded-perturbation condition (which the S3.4 acceptance gate plus shape-quality safeguards ensures).

6. **Combine via averaging.** Two-time-scale SA gives that the joint trajectory `(w(t), σ²(t))` converges almost surely to the slow-ODE attractor, which is a local minimum of `L_vertex`.

#### Required prerequisites

- Verify Robbins–Monro conditions for the dual-rate prescription: `η_cent = κ(1-r)/k` is constant (so `Σ η = ∞` is automatic but `Σ η² = ∞` would fail in the strict sense; need to argue via the EWMA-driven `η_GNG,i` decrease instead).
- Verify the soft-assignment connection to S14 (Gaussian-weighted local PCA limit): in the limit `α → 0`, large `k`, slow geometry, the dynamics reduce to the deterministic ODE used in step 3 above.
- Verify boundedness of `L_vertex`: trivially, `L_vertex ≥ 0` and `L_vertex ≤ data_diameter² · N_samples`, finite.

#### Estimated effort

This is paper-length work on its own:

- **Two-time-scale SA setup + ODE derivation:** ~5 pages.
- **Lyapunov analysis of the slow ODE:** ~3 pages (mostly bookkeeping once the ODE is in hand).
- **Bounded-jump treatment of splits:** ~2 pages, citing Borkar Ch. 11.
- **Connection to S14 limit:** ~2 pages.
- **Total:** ~12 pages, plus ~5 pages of preliminaries / setup.

This is in scope for either a standalone technical paper or a substantial appendix in a future Proteus paper. It does not block a first implementation, since:

- All four lemmas have plausible heuristic justification.
- The implementation can detect Lyapunov-violation symptoms empirically (e.g., increasing `L` over a long window) and apply hysteresis as an engineering workaround if needed.
- Lemmas 2--4 are already proved (SI §S15.1), so the implementation framework is on solid footing.

#### Workaround for implementation

If Lemma 1 is genuinely violated in practice (e.g., `L` increases for an extended window), an engineering safeguard exists:

- **Hysteresis on accepted edits.** Require that an edit, once accepted, cannot be reversed by a subsequent edit for at least `T_hysteresis` samples. This prevents oscillation regardless of the Lyapunov guarantee.
- **Global edit budget per epoch.** Cap the total number of accepted topology edits per training epoch to `O(N / log N)`. Once exhausted, freeze topology and let `L_vertex` settle.
- **Termination by evidence-rate threshold.** Stop the dynamics when the average per-sample `F_DM` decrease drops below `ε / (training_epoch_length)` for some small `ε`. This is an empirical check that the gate has stopped finding profitable edits.

These are engineering safeguards, not theoretical guarantees, but they suffice for a first implementation. The architectural framework (Lyapunov candidate identified, Lemmas 2--4 proved) provides confidence that the safeguards are addressing edge cases rather than masking a fundamental problem.

---

## Priority 2 — Specification gaps (concrete but under-documented)

These won't cause failure but will cause confusion and rework if not pinned down.

### 8. Dual-flow weights (λ, μ) and their coupling to prior strengths

- Main text §6.2 writes the objective; no default values given.
- §5.3 mentions "dual-flow hyperparameter tying" as a future refinement — at minimum there should be a starting baseline.
- **Needed**:
  - Default `λ, μ` (e.g., `λ = 1`, `μ = 0.1`, or normalized to count-averages).
  - An explicit tying rule `λ ∝ (α₀ + n_S)`, `μ ∝ τ` as an option.

### 9. Transition count bookkeeping across recursion levels

- Stage 1 controller recurses on partitions. What happens to `n_{i → j}` from the parent scale when recursing into a sub-partition?
  - Option A: reset (child starts from zero counts).
  - Option B: inherit (aggregate upward).
  - Option C: inherit with decay.
- Each option changes the meaning of `N_eff, R` in §S3.4 for the child level.
- **Needed**: pick one, document in SI §S9 or §S11.

### 10. Torsion ladder thresholds — initial values

- The text now labels `0.05, 0.30, 0.60` as empirical. For a first implementation:
  - These values need to hold together with the shape-quality threshold `Q_min = 0.25` and the variance-cap `τ_local`.
  - A scale-aware rescaling might be appropriate (they're dimensionless, but still).
- **Needed**: a starting set of values that are explicitly marked as "first-implementation defaults, retune later"; enumerate variants tried so far (if any).

### 11. Boundary handling in dual flow

- SI §S9 mentions Neumann-like boundary conditions; SI §S5 now says boundary-face columns are zeroed in `A_S`.
- For open meshes with manifold-with-boundary geometry, this is necessary but may cause mass leakage.
- **Needed**:
  - An explicit statement of what the solver does at three kinds of boundary:
    1. True manifold boundary (flux = 0).
    2. Computational-only boundary from finite sampling (should allow flux).
    3. Non-orientable regions (local orientation patches — how stitched?).
  - A diagnostic for mass conservation after solve: `Σ_S m_S = 1 ± ε`.

### 12. Mini-NSF patch size and training budget

- SI §S7 gives rough figures (2–3 layers, 64–128 hidden, 8–12 bins, `P_max = 64` simplices) but doesn't justify them.
- Also: what happens when `P_max` is exceeded — split the patch? Use a shallower flow?
- **Needed**: decision rule for patch-size exceedance and default training budget tied to routed sample count.

---

## Priority 3 — Mathematical risks flagged by current theoretical analysis

These are known open questions in SI §S15. They are less urgent than Priority 1 because workarounds exist, but each has implementation consequences.

### 13. Stuck-junction behavior (SI §S15.2)

- At a dimensionality junction (e.g., 1D filament meeting 2D sheet), the paper argues the system exhibits a stable signature (bimodal simplex activity, divergent link significance, etc.).
- Informal argument only; no quantitative guarantee.
- **Implementation risk**: the mesh may oscillate at junctions without a sharp termination criterion for "we've reached a junction, stop refining."
- **Needed**: a detector for stuck-junction regime (the signature statistics in SI §S12.2 are candidates) and a rule to freeze further refinement locally when detected.

### 14. Expressivity budgets (SI §S15.3)

- SI §S12.1 asserts that finite budgets suffice but doesn't give explicit budget functions as a function of `ε`, Lipschitz constants, and dimension.
- **Implementation risk**: no stopping criterion beyond empirical observation; first implementation may not know when "good enough" is.
- **Needed**: even a crude upper-bound estimate tied to observable quantities (e.g., `max R_S` and `CV`) would help decide run-time budgets.

### 15. Mass-field identifiability (SI §S15.5)

- Related to Priority 1 #1 but at the mesh level: given all transition counts over the whole complex, is `m` uniquely determined?
- If not, different solvers could converge to different `m` fields with equal data fit.
- **Needed**: for the canonical `κ` choice, prove or empirically verify that the global Jacobian `∂q/∂m` has full column rank; if not, identify the null-space structure and whether it's benign.

---

## Priority 4 — Writing / framing (lowest priority; deferred)

These do not affect implementation. Keep as notes for any eventual external release.

### 16. Fuzzy title decision

- Title emphasizes "fuzzy manifold memberships." §6.3 now has an operational anchor for this, but not central.
- **Deferred**: revisit only if paper moves toward submission.

### 17. Architectural overview figure

- Six owned objects without a diagram.
- **Deferred**: does not block implementation; useful for documentation once code exists.

### 18. Formal citations

- §1.5 is prose-only. §13 "References Prep" tracks the intended list.
- **Deferred**: does not affect implementation.

---

## Implementation-readiness checklist

Before starting the first implementation, at minimum the following should be resolved (drawn from Priority 1 items):

- [x] **Canonical `κ` defined and identifiability theorem stated** (item #1; resolved in SI §S13.1 + §S13.7).
- [x] **MAP settle protocol resolved** (item #2): `F_DM` (closed-form, no settle) is the sole evidence score (SI §S3.4); the MAP-on-`m` alternative was removed as unused dead code.
- [x] **Lyapunov framework identified; auxiliary lemmas proved** (item #3, downgraded from showstopper): `L = L_vertex + F_DM` decomposes the dynamics into a continuous side governed by NG/GNG and a discrete side governed by S3.4 acceptance. Voronoi-Delaunay duality makes the two terms complementary. Lemmas 2 (joint monotonicity), 3 (cross-scale inheritance), and 4 (EWMA noise control) are now proved by direct calculation in SI §S15.1, conditional on Lemma 1. Lemma 1 (variance-cap NG monotonicity for `L_vertex`) is the only foundational result still open; full prep, prior art, proof template, and engineering workarounds are documented in this item. Does not block first implementation.
- [x] **Greedy Chaining pseudocode resolved** (item #4): node-seeded BFS algorithm with clique-only verification and canonical-form deduplication; cost `O(N · d²)` (SI §S8.1).
- [x] **Dual-flow solver conditioning resolved** (item #5): canonical solver is loopy Gaussian BP; sliver simplexes (`Q_S < Q_min`) get `μ_S = 0` (conservation factor dropped), making the stencil well-conditioned by construction (SI §S5).
- [x] **A concrete per-cluster `Φ_C(τ)` formula** (item #6; resolved in SI §S2.5).
- [x] **Stage 1 cluster identification fully specified** (AP on Hebbian graph; resolved in SI §S2.6).
- [x] **Dirichlet concentration derived** (`α_{0,i}=1/(d_{\mathrm{final},i}+1)`; resolved in SI §S2.7).

Priority 2 items can be pinned to starting defaults and iterated during implementation.

---

## Recently closed

### This session (Lyapunov framework + auxiliary lemmas proved)

- [x] **Priority 1 #3 — Dynamic convergence: 3 of 4 supporting lemmas proved.**
  - SI §S15.1 now contains explicit proofs (by direct calculation, conditional on Lemma 1) for:
    - **Lemma 2 (joint monotonicity):** continuous and discrete updates yield `E[ΔL | F_t] ≤ -c_t + b(t)` with bounded data-accumulation drift `b(t) ≤ log J`. Robbins–Siegmund convergence theorem applies. Accepted edits give `ΔL ≤ -log τ + ε_topo` with `ε_topo` small relative to `log τ` for recommended thresholds.
    - **Lemma 3 (cross-scale inheritance):** warm-start at scale `s+1` inherits a finite `L` from scale `s`; transition contributes `O(log d · |ΔV|)` for nodes whose intrinsic dimension estimate updates, exact equality otherwise.
    - **Lemma 4 (EWMA noise control):** transient perturbation after a split decays at rate `1/k` (half-life `k`); martingale fluctuation bounded by Azuma–Hoeffding at `O(τ √(α log(2/δ)))` with high probability. Total integrated transient is finite and does not accumulate across splits (split rate is sub-linear by the `log τ` budget).
  - SI §S15.6 summary restructured to a 4-tier classification with explicit "Conditionally rigorous, foundational result open" tier for the joint convergence story.
  - Item #3 in this document expanded with full background on the open foundational result (Lemma 1: variance-cap NG monotonicity), including: precise statement, prior art (Martinetz–Schulten 1991; Fritzke 1995; Bottou–Bengio 1995; Sato 1999), why standard NG proof doesn't directly apply, two-time-scale SA proof template (Borkar 2008; Kushner–Yin 2003), required prerequisites, estimated effort (~12 pages plus preliminaries), and engineering workarounds for first implementation.

### Previous session (Lyapunov candidate identified)

- [x] **Initial Lyapunov candidate identification (preceded the lemma proofs above).**

### Previous session (Greedy Chaining specification)

- [x] **Priority 1 #4 — Greedy Chaining algorithm.**
  - SI §S8.1 now contains a full pseudocode specification for initial $d$-simplex discovery: node-seeded BFS, leveraging the structural fact that a node with degree $\approx d{+}1$ has at most $d{+}1$ incident $d$-simplexes.
  - Verification uses only graph operations: clique check ($\mathcal{O}(d^2)$ adjacency lookups) and canonical-form deduplication ($\mathcal{O}(d\log d)$ hashset insert). No Cayley--Menger / Gram-determinant / shape-quality computation during initialization --- runtime mechanisms (S4 torsion-aligned splits, S5 $\mu_S=0$ rule, S3.3 simplex-arbitrated pruning) handle whatever pathologies survive.
  - Cost: $\mathcal{O}(N \cdot d^2)$ amortized per recursion level (cost envelope in S8.2 updated). Each simplex is paid once across its $d{+}1$ discoverable vertices via canonical-form dedup. BFS ordering compounds the speedup: most simplexes touching a node are already in the set by the time it's visited.
  - Heterogeneous intrinsic dimension handled by per-node enumeration; cross-junction simplexes naturally fail the clique test.
  - Orphans deferred to runtime ``Continuous Simplex Discovery'' rule (`stage_2.md` §3.3.2) for eventual coverage.
  - Built-in junction diagnostic: `incident_count` distribution after chaining matches the addendum's bimodality signature for free.

### Previous session (evidence gate + dual-flow solver simplification)

- [x] **Priority 1 #2 — MAP settle protocol for evidence scoring.**
  - $F_{\mathrm{DM}}$ (the closed-form Dirichlet--multinomial marginal under the BDeu prior of S2.7) is now the sole evidence score in SI §S3.4. No iterative MAP settle on $m$ is required: the Dirichlet posterior over each per-node $q(\cdot\mid i)$ is closed-form, the marginal likelihood is analytic, and the Occam factor is intrinsic to the BDeu prior volume. Topology candidates are directly comparable with no protocol-dependent biases.
  - The MAP-on-$m$ alternative (with Laplace/BIC + projected gradient settle protocol) was removed as unused dead code: per-node $q$ smoothing is already handled by BDeu; the $n_C \ge \max(10d,1000)$ recursion floor (S9) keeps regions away from the genuinely low-evidence regime where logistic-GMRF spatial smoothing on $m$ would matter; no application path consumes posterior uncertainty on $m$.
  - WAIC/PSIS-LOO retained as predictive-accuracy diagnostics; default deployment is $F_{\mathrm{DM}}$ alone.
  - Main paper §5.3, abstract, contributions bullet, and SI §S13 intro updated accordingly.

- [x] **Priority 1 #5 — Dual-flow solver numerical conditioning.**
  - Canonical solver in SI §S5 is now loopy Gaussian belief propagation on the face/simplex factor graph (face variables; per-face data factors weighted by $\lambda$; per-simplex conservation factors weighted by $\mu_S$). Gauss--Seidel was dropped --- it added no architectural value and was the source of the ill-conditioning concern.
  - Sliver simplexes ($Q_S < Q_{\min}=0.25$) now have $\mu_S = 0$, i.e.\ they contribute a data factor only and no conservation factor. This is principled (the discrete divergence theorem is geometrically meaningless on a degenerate simplex) and removes the dominant numerical-conditioning failure mode by construction: ill-conditioned $A_S^{\top}A_S$ blocks no longer enter message precisions.
  - Loopy BP convergence safeguarded by damped Gaussian message updates if non-walk-summable spectra are detected; vanilla loopy BP otherwise.

### Previous session (canonical cluster-scale search)

- [x] **Priority 1 #6 — Concrete per-cluster `Φ_C(τ)` formula.**
  - Replaced the earlier slope-anomaly overcorrection with the canonical reference-doc formula in SI §S2.5: $\Phi_C(\tau)=\sum_{i\in C}R_i(\tau)$ with $R_i(\tau)=(\sqrt{\tau}/c_{d,k})^d h_i k N_C/(V_d r_{k,i}^d)$.
  - Added support trace $V_C(\tau)=\sum_i \widehat V_i$ as a cheap cross-validation signal; robust characteristic scales exhibit both a peak in $\Phi_C$ and a nearby plateau transition in $V_C$.
  - Peak detection now uses the centered second difference $\Delta^2\Phi_C$ with Bayesian refinement inside the bracket.
  - Main text §3.4, §3.5, Algorithm 2, and Appendix notation updated to match the canonical per-cluster response.
- [x] **Cluster identification specified.**
  - SI §S2.6 now specifies Affinity Propagation on the Hebbian graph using smoothed-PMI similarities built from the Dirichlet-smoothed posterior predictive $\widehat q(j\mid i)$.
  - Per-node AP preference is derived, not tuned: $s(i,i)=-2\log(d_{\mathrm{final},i}+1)$.
  - Directional asymmetry $A(i,j)$ is retained as a junction diagnostic and not used for clustering.
  - Cross-scale tracking uses persistent graph identity / node-ID overlap after warm-start.
- [x] **Dirichlet concentration `α₀` closed.**
  - SI §S2.7 derives $\alpha_{0,i}=1/(d_{\mathrm{final},i}+1)$ from the BDeu prior: one equivalent pseudo-observation spread uniformly over the local branching factor.
  - This same $\alpha_{0,i}$ is used for transition smoothing (S2.6), Dirichlet--multinomial evidence (S3.4), and the BNP view (S13), eliminating a separate tuning knob.

### Previous session (router identifiability)

- [x] **Priority 1 #1 — Router identifiability and conditioning.**
  - **Canonical κ committed** in SI §S13.1: uniform facet-share, $\kappa_{iS}=A_{iS}$ (opposite-facet area), $\kappa_{ijS}=A_{iS}/d$ for $j\in S$. Choice is positive, scale-covariant, orientation-invariant, cheap to compute, and yields a clean linear star structure.
  - **Star Matrix Identifiability Theorem** added in SI §S13.7. Under (A1) local star-rank and (A2) connected dual graph, $m$ is uniquely determined (up to the simplex constraint) by the transition probabilities $\{q(\cdot\mid i;m)\}_i$. Proof proceeds via local scalar factorization + connected-dual propagation.
  - **Runtime conditioning check** defined: $\sigma_{\min}(K_i)/\sigma_{\max}(K_i) < \rho_{\min}=10^{-4}$ flags the star as ill-conditioned; MAP settle skips that node's likelihood contribution and the evidence gate falls back to geometric-only acceptance for edits in that region. Strictly conservative — never accepts an edit that would otherwise be rejected.
  - **Fisher-information connection** noted: $K_i$ is (up to normalization) the Jacobian of $q(\cdot\mid i;m)$; (A1) is precisely Fisher nonsingularity on the probability simplex, giving asymptotic MLE consistency as $n_i\to\infty$.
  - **Residual open point**: static identifiability is proved; dynamic preservation (identifiability along trajectories of the variance-cap + evidence loop) left open in SI §S15.5 residual.
  - SI §S15.5 marked RESOLVED (static); SI §S15.6 summary updated to include S13.7 in the "Rigorous" tier.

### Prior session (framing and presentational cleanup)

- [x] **"Free-energy proxy" → "local evidence score"** renamed throughout (paper §5.3, abstract, contributions, §1.5; SI §S3.4 heading + body). "Free-energy proxy" retained as explanatory parenthetical; explicitly distinguished from variational free energy (ELBO) in SI §S3.4.
- [x] **BIC / Laplace framing clarified**: SI §S3.4 now states the evidence score is a leading-order Laplace approximation of the log marginal likelihood, with full Laplace, Dirichlet–multinomial marginals, and WAIC/PSIS-LOO listed as principled substitutes. Paper §5.3 updated correspondingly.
- [x] **Intrinsic vs. ambient dimension** explicit in §4.2.
- [x] **DEC framing for `Ω_S`** clarified in §4.2.
- [x] **Greedy Chaining exposure** mentioned in main text §4.2 (though the algorithm itself is still only named, not specified — now tracked as Priority 1 #4).
- [x] **`A_S` definition** explicit in SI §S5 as discrete-divergence stencil with boundary handling.

### From prior sessions

- [x] Dead `SI~S15` references (retargeted to `SI~S10`).
- [x] Scalar vs. vector `σ_i^2` ambiguity (standardized to trace form).
- [x] Dropped-vertex convention in `E, M` unexplained (now clarified in §4.2).
- [x] `Ω_S` interpretation overclaiming Jacobian antisymmetric part (now clarified as quadratic form).
- [x] `Φ(τ)` deferred entirely to SI without main-text definition (now briefly defined).
- [x] Appendix notation inconsistent with body (aligned).
- [x] `m` vs. `p_f` relationship implicit (now explicit in §6.1).
- [x] "Fuzzy" framing with no operational anchor in body (§6.3 now has the anchor; title decision still open).
- [x] BIC regularity caveat unacknowledged (footnote added in §5.3).
- [x] Node-vs-simplex equilibrium correspondence overclaimed (now labelled heuristic in §4.5 and SI S12.2).
- [x] Torsion Ladder thresholds appearing as universal constants (now labelled empirical in §4.3).
- [x] Scale controller termination condition incomplete (now lists 3 conditions in §3.6).

---

## Notes

- Re-evaluate this list before starting each major implementation phase (Stage 1 alone, Stage 2 alone, end-to-end pipeline).
- Items in Priority 1 that become too costly to resolve formally can be downgraded to "engineering workaround" — but the workaround must be documented in the SI.
- Pure parameter-tuning discussions (e.g., the BNP streaming / node-overgrowth thread) continue to live in `bnp_design_implications.md` and `bnp_upgrades_math.tex`; only items that affect mathematical correctness of the pipeline belong here.
