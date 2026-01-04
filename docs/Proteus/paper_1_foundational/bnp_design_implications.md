## Bayesian-inspired design implications for Proteus (Stage 1/2)

### Purpose

This document brainstorms enhancements suggested by the Bayesian non-parametric (BNP) correspondence added in SI S13. It focuses on practical, low-regret design changes and clear knobs that could improve robustness, interpretability, and statistical coherence without sacrificing speed.

### Context (one scale)

- **Latent field**: simplex masses \(m=\{m_S\}\) on the learned mesh; \(\sum_S m_S=1\).
- **Evidence**: per-node transition counts \(\{n\_{i\to j}\}\) inducing a Dirichlet–multinomial-like factor via a geometric router \(q(j\mid i; m)\).
- **Prior**: compositional (Dirichlet), smooth (logistic-GMRF), or hybrid/hierarchical.
- **Inference**: MaxEnt/MAP as the zero-prior limit; loopy BP/variational as posterior approximations.

---

## High-impact enhancements (prioritized)

- **Evidence-based split/stop tests (replace curvature heuristic):**

  - Use WAIC/LOO or a local Bayes factor (split vs. keep) computed on the transition-likelihood term \(\sum*{j} n*{i\to j}\,\log q(j\mid i; m)\) at candidate cells/regions.
  - Why current approach is insufficient: the curvature/second-difference heuristic is not calibrated to sample size and ignores the actual transition likelihood; it can over-refine noisy regions and under-refine informative ones, and it is sensitive to local branching geometry.
  - Benefits: statistically grounded stopping; adaptive to data complexity; reduces over-refinement.

- **Split gating by Bayes factor:**

  - Trigger a split only if \(\mathrm{BF}_{\mathrm{split}}>\tau_{\mathrm{BF}}\); otherwise defer or raise the Dirichlet concentration \(\alpha_s\) to keep children close to the parent.
  - Why current approach is insufficient: split triggers rely on variance thresholds/torsion and tenure rules, which do not incorporate strength-of-evidence; splits may occur with low effective counts, increasing reversal/regret.
  - Benefits: fewer spurious splits; principled control of model growth.

- **Coarse→fine prior coherence (α schedule):**

  - Tie the Dirichlet refinement concentration \(\alpha_s\) to local geometry: higher torsion/uncertainty → lower \(\alpha_s\) (more freedom); low torsion/high certainty → higher \(\alpha_s\) (conservative).
  - Why current approach is insufficient: the refinement freedom is effectively fixed or heuristic; it does not shrink in flat regions or expand in curved ones, leading to either over-smoothing in complex areas or unnecessary refinement in simple ones.
  - Benefits: aligns refinement freedom with real curvature; stabilizes flat regions.

- **Smoothness prior scheduling (logistic-GMRF on demand):**

  - Enable a small \(\tau>0\) smoothness prior for \(m\) only in regions with weak identifiability (few transitions, poorly conditioned \(\kappa\)). Keep Dirichlet-only elsewhere.
  - Why current approach is insufficient: without local smoothness, low-evidence patches can produce unstable or highly variable mass allocations; with global smoothness, well-identified regions get over-smoothed; scheduling balances both.
  - Benefits: regularizes ill-posed patches without over-smoothing globally.

- **Uncertainty-aware gates (splits/prunes/warps):**

  - Use posterior variance (or Laplace curvature) of \(m\) and \(q\) to block irreversible actions (split/prune/warp) under high uncertainty.
  - Why current approach is insufficient: existing gates use point rules (variance/thresholds, Wilson bounds) without propagating uncertainty in \(m\) and \(q\); this can commit topology changes prematurely.
  - Benefits: reduces regret; defers topology changes until evidence is adequate.

- **Identifiability constraints on the router \(q\):**

  - Constrain \(\kappa\)-weights to be nonnegative, normalized, and minimally separated so that \(q(j\mid i;m)\) is injective in adjacent masses. Add diagnostics for ill-conditioning.
  - Why current approach is insufficient: unconstrained \(\kappa\) can make the mapping from local masses to transitions ill-conditioned or non-injective, hampering inference and leading to ambiguous updates.
  - Benefits: clearer inverse mapping from transitions to local masses; safer inference.

- **Variational/BP refinement passes near convergence:**

  - Periodically run a few iterations of variational/BP updates to refine \(m\), warm-started from MaxEnt, with tighter tolerance only near finalization.
  - Why current approach is insufficient: pure MaxEnt/MAP ignores posterior curvature and can bias estimates where evidence is sparse; short posterior-refinement passes improve calibration at modest cost.
  - Benefits: improved calibrated masses with modest overhead.

- **Posterior predictive reporting:**

  - Report credible intervals for \(m_S\) and for key derived quantities (e.g., simplex mass ratios, selected \(q(j\mid i)\)).
  - Why current approach is insufficient: current reports focus on point metrics (e.g., likelihood, MMD) and do not communicate uncertainty; this limits diagnostics and interpretability.
  - Benefits: communicates confidence; aids diagnostics and ablations.

- **Dual-flow hyperparameter tying:**

  - Tie flow-solver regularization (\(\lambda,\mu\)) to prior precision on \(m\) (Dirichlet strength or GMRF \(\tau\)).
  - Why current approach is insufficient: \(\lambda,\mu\) are set independently of the prior on \(m\), risking mismatched smoothing between geometric mass inference and dual-flow reconstruction.
  - Benefits: harmonizes smoothing across geometry and flow.

- **Streaming-friendly conjugacy:**
  - Maintain online updates for node-local Dirichlet factors; periodically solve for \(m\) with limited-iteration variational passes.
  - Why current approach is insufficient: streaming updates currently maintain EWMAs but not posterior summaries for \(m\); without conjugate updates and periodic refinement, uncertainty can drift or collapse.
  - Benefits: amortizes cost; supports continual learning.

---

## Minimal viable adoption path

1. **Introduce split/stop Bayes factor (local)** while keeping current thresholds as backstops.
2. **Tether \(\alpha_s\) to torsion/uncertainty** (simple linear/clipped mapping).
3. **Enable uncertainty-aware gates**: block split/prune if posterior variance above threshold.
4. **Add short variational/BP passes near convergence** (fixed small iteration budget).
5. **Align dual-flow regularization** with prior strength (static mapping initially).

Each step is orthogonal and can be toggled via flags; roll out incrementally.

---

## Parameter mapping (practical knobs)

- **BFsplit threshold**: \(\tau\_{\mathrm{BF}}\in[1,\,10]\) (log-scale friendly); per-dataset tuning or annealing.
- **α schedule**: \(\alpha*s=\alpha*{\min}+f(\text{torsion},\,n)\cdot(\alpha*{\max}-\alpha*{\min})\) with monotone \(f\in[0,1]\).
- **GMRF precision**: small \(\tau\) (e.g., \(10^{-3}\) to \(10^{-1}\)) in low-evidence regions; 0 otherwise.
- **Uncertainty gates**: block if \(\operatorname{Var}[m_S]>v*{\max}\) or if Laplace Hessian condition number exceeds \(\kappa*{\max}\).
- **Flow coupling**: \(\lambda\propto \alpha_s\), \(\mu\propto \tau\).

---

## Diagnostics & acceptance checks

- Monitor: split BF distribution, gated-action rate, posterior variance histograms, mesh growth vs evidence, flow residuals vs prior strength.
- Success criteria: fewer reversals, smoother mass fields in low-evidence regions, equal-or-better likelihood/MMD with similar or reduced complexity.

---

## Open questions / experiments

- How sensitive is BFsplit to the choice of router \(q\) and \(\kappa\)-normalization? Can we standardize a robust default?
- Does \(\alpha_s\) tied to torsion meaningfully reduce over-refinement in flat regions without underfitting curved patches?
- What is the smallest number of variational/BP iterations that yields reliable uncertainty without material wall-time impact?
- Can posterior predictive checks (held-out transitions) flag bad \(q\)-calibration early?

---

## Notes for implementation

- Start with lightweight Laplace approximations for \(m\) to estimate variance; defer full variational until late.
- Cache per-simplex Jacobians for \(q\) to accelerate local curvature/variance estimates.
- Keep all additions optional (feature flags) to preserve current benchmarks.
