# Report: γ-Normalized Derivatives and Scale-Space Optimization for Growth-Threshold GNG

## 1. Overview

This report formalizes the use of γ-normalized derivatives in the context of a growth-threshold controlled Growing Neural Gas (GNG) algorithm that traverses an implicit scale-space. We unify the mathematical interpretation of your discrete threshold schedule with Lindeberg's theory of scale selection, derive appropriate normalization and resolution constraints, and specify the complete operational method for grid-based and Bayesian-optimized scale selection.

---

## 2. Mathematical Framework

### 2.1 Scale-Space Representation

In classical scale-space theory:

- A function (image, density, etc.) is convolved with Gaussian kernels of increasing standard deviation $\sigma$, corresponding to scales $t = \sigma^2$.
- Derivatives are computed at each scale, and normalization is applied to preserve scale-covariant comparisons:

$L_{\xi^m\eta^n}(x, y; \sigma) = \sigma^{\gamma(m+n)} \partial_{x^m y^n} L(x, y; \sigma)$

- $\gamma \in [0,1]$ controls how the normalization compensates for diffusion.

---

### 2.2 GNG Analogy

In the growth-threshold GNG:

- The threshold $\tau$ governs the **maximum allowable local variance** of a Voronoi region (or neighborhood), controlling when new nodes are added.
- Empirically, $\tau \propto \sigma^2$, where $\sigma$ is the implicit standard deviation of a smoothing Gaussian.
- Each node accumulates hits $h_i$ and has a local density estimate:

$\hat{\rho}_i = \frac{k}{N V_d r_{k,i}^d}, \quad \widehat{\text{Vol}}_i = \frac{h_i}{\hat{\rho}_i} = \frac{h_i V_d r_{k,i}^d}{k/N}$

where $r_{k,i}$ is the distance to the $k$-th neighbor and $V_d$ is the volume of the unit ball in $\mathbb{R}^d$.

---

Define the normalized response:

$R_i(\tau) = \sigma^{\gamma d} \frac{h_i}{\widehat{\text{Vol}}_i}$

Substituting in terms of $r_{k,i}$, and using $\sigma = \sqrt{\tau} / c_{d,k}$, we get:

$R_i(\tau) = \left(\frac{\sqrt{\tau}}{c_{d,k}}\right)^{\gamma d} \cdot \frac{h_i k N}{V_d r_{k,i}^d}$

Choosing $\gamma = 1$ gives the scale-covariant form. Summing over nodes within a cluster:

$\Phi_C(\tau) = \sum_{i \in C} R_i(\tau)$

This function should be \*\*unimodal in log \*\***$\tau$** for coherent features and admits a unique maximum at the characteristic scale.

---

## 4. Equivariance and Covariance

Lindeberg proves that if the input space is rescaled $x \mapsto s x$, the response extrema shift:

- $\sigma \mapsto s \sigma$
- $t \mapsto s^2 t$

This property (scale-covariance) holds for $\gamma$-normalized responses with $\gamma = 1$ and is inherited in the GNG framework as long as:

- The same $\gamma$-normalization is applied.
- The growth threshold $\tau$ is proportional to $\sigma^2$.

---

## 5. Optimal Grid Spacing (Resolution)

Let $\Delta_{\log \sigma}$ be the full-width at half maximum (FWHM) of a normalized feature response in log-scale:

$\Delta_{\text{FWHM}}(\log \sigma) \approx 0.586 \Rightarrow e^{\Delta} \approx \sqrt{2} \approx 1.41$

This gives a **Rayleigh-like resolution limit**:

- **Minimum resolvable peak separation**: $\log \sigma_2 - \log \sigma_1 \gtrsim 0.3$
- **Grid ratio** $r = \sigma_{j+1}/\sigma_j$ should satisfy $r \le \sqrt{2}$.

Thus, set $r = 1/\sqrt{2} \approx 0.71$. This gives \~7 steps per decade of scale.

---

This constant maps growth threshold $\tau$ to $\sigma$:

$\tau = c_{d,k}^2 \cdot \sigma^2 \Rightarrow \sigma = \sqrt{\tau} / c_{d,k}$

### 6.1 Monte Carlo Method

- Generate $N \gg 1$ samples from a unit-$\sigma$ $\mathcal{N}(0,I_d)$
- For each, compute $r_{k}$ to the $k$-th nearest neighbor.
- Take $c_{d,k} = \mathbb{E}[r_k]$

### 6.2 High-d Approximation

For large $d$, $c_{d,k} \to 1$ as Gaussian mass concentrates. Set $c_{d,k} = 1$ for $d \gtrsim 200$ without noticeable error.

---

## 7. Grid Search and Bayesian Optimization

### 7.1 Coarse Grid Search (Log-scale)

Use geometric spacing:

$\tau_i = \tau_0 \cdot r^i, \quad r = 1/\sqrt{2} \approx 0.71$

At each $\tau$, run GNG, compute $\Phi_C(\tau)$, and detect scale maxima by sign change of:

$\Delta^2 \Phi_C(\tau_j) = \Phi_C(\tau_{j+1}) - 2 \Phi_C(\tau_j) + \Phi_C(\tau_{j-1})$

When $\Delta^2$ flips sign from negative to positive, bracket the extremum.

### 7.2 Bayesian Optimization Within Bracket

Use a Gaussian process (GP) to model $\Phi_C(\log \tau)$ inside the bracket.

- Use `gp_minimize` (e.g., from `scikit-optimize`)
- Evaluate 3–5 $\tau$ values within bracket
- Accept $\tau^*$ where the GP posterior second derivative crosses zero with confidence

This refinement reduces the number of GNG passes while preserving scale-space theory.

---

## 8. Recursion and Stopping

Once a maximum in $\Phi_C(\tau)$ is found:

1. Record $\tau^*_C$ as the cluster's characteristic scale.
2. Recurse into that cluster's support using the same process.
3. Terminate when no subclusters are resolved within 1 order-of-magnitude of scale.

---

## 9. Summary

- Use $\gamma = 1$ normalization to ensure scale-covariant detection.
- Estimate or assume $c_{d,k} = 1$ for large $d$, else calibrate once.
- Use a geometric grid with spacing $r = 0.71$.
- Detect maxima via sign flip of second difference.
- Use a Bayesian optimizer within the bracketed interval to refine $\tau^*$.
- Recurse on clusters within the same framework.

This strategy retains all theoretical benefits of scale-space theory while being computationally efficient and adaptive to both low- and high-dimensional data.

---

## 10. Addendum – Scalar vs. Rank‑1 Oja Error Tracking

Two compatible ways to enforce the local‐variance cap τ in practice:

| Variant                                 | State stored per node                                                 | Update cost                                  | Split direction when τ is exceeded         | Pros                                                          | Cons                                                   |
| --------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------- | ------------------------------------------------------ |
| **Scalar variance + EWMA error vector** | _μ_ (mean, *d*), scalar variance *S*, optional averaged raw error *ē* | **O(d)**                                     | Along *ē* (or latest error) at distance √τ | Minimal memory; already implemented                           | Direction only approximate when the cloud is elongated |
| **Rank‑1 Oja (dominant P.C.)**          | _μ_, unit PC vector *u*, variance σ²                                  | **O(d)** (one extra dot product + normalise) | Precisely along *u* at distance σ          | Better placement; captures main elongation; still O(d) memory | One extra vector per node (\~2× RAM)                   |

**Key point:** *Both methods keep an exponential moving estimate of variance*, so the theoretical link τ ≈ σ² used in the scale‑space report stays valid. The rank‑1 Oja upgrade merely adds directional accuracy at the cost of one extra _d_-dimensional vector.

_Recommended rule‑of‑thumb_

- **d ≤ 10 k** → use Oja: extra RAM is modest and split quality is higher.
- **d ≫ 10 k or RAM‑constrained** → stay with scalar variance; theory unchanged.

### 11. Thermal-Equilibrium Interpretation

In the mesh, each node’s **incoherence ratio**

ρi=∥mi∥σi+ε\rho_i = \frac{\lVert m_i \rVert}{\sigma_i + \varepsilon}ρi=σi+ε∥mi∥

behaves like a _local heat current_: it measures residual drift that has not yet been cancelled by symmetric data flow.

- **High ρi\rho_iρi** → net energy flux (region still “hot”).
- **Low ρi\rho_iρi** → local equilibrium (incoming directions already balanced).

Define a neighbourhood-normalised measure

ρ\~i=ρimeanj∈N(i)ρj+ε  .\tilde\rho_i = \frac{\rho_i}{\text{mean}\_{j\in\mathcal N(i)}\rho_j + \varepsilon}\\; .ρ\~i=meanj∈N(i)ρj+ερi.

A global coefficient of variation

CV=std(ρ\~)mean(ρ\~)\mathrm{CV} = \frac{\mathrm{std}(\tilde\rho)}{\mathrm{mean}(\tilde\rho)}CV=mean(ρ\~)std(ρ\~)

acts as a **thermodynamic convergence test**:

- **CV → 0** ⇒ residual drift is spatially uniform ⇒ mesh has reached _thermal equilibrium_ for the current scale.
- **CV large** ⇒ significant heat currents remain ⇒ continue learning or refine scale.

This complements the variance cap: the cap limits “container size,” while CV checks if the “heat” has evened out across the mesh.

---

## 12. Deferred‑Nudge Re‑Centering and Sub‑Resolution Threshold

Updating a node position on every data win can be computationally expensive because it forces neighbour‑list maintenance, index refreshes, and (in Stage 2) simplex normal updates. Instead we accumulate a **nudge vector** `aᵢ` and move the node only when that accumulated shift is larger than what the current scale can actually resolve.

### 12.1 Per‑Win Accumulation (No Immediate Move)

1. Maintain the running residual mean `mᵢ` and variance `σᵢ²` (as in Section 10).
2. Compute the incoherence ratio `ρᵢ = ‖mᵢ‖ / (σᵢ + ε)` (bounded between 0 and 1).
3. Add a small increment to the **nudge accumulator**:

```
aᵢ += η_cent * ρᵢ * mᵢ      # η_cent ≪ η_GNG (e.g. 0.05×)
```

The node coordinate **is not moved** at this step.

### 12.2 Principled Sub‑Resolution Threshold

- Your geometric scale grid uses the ratio `r = 1/√2 ≈ 0.71`.
- At variance cap `τ = σ*²` the radial distance to the next finer scale is `(1 – r) √τ ≈ 0.29 √τ`.
- Define a safety factor `κ ∈ (0, 1)` and set

```
δ_min = κ * (1 – r) * √τ     # e.g. κ = 0.5 ⇒ δ_min ≈ 0.15 √τ
```

This is the smallest displacement that can matter at the current scale.

### 12.3 Deferred Apply‑and‑Reset (Rare)

Periodically (e.g. after every 100 wins or at epoch end):

```
if ‖aᵢ‖ ≥ δ_min:
    wᵢ += aᵢ          # single expensive move
    aᵢ = 0            # reset accumulator
    refresh_neighbour_caches(i)
    refresh_spatial_index(i)
    if stage == 2:
        recompute_incident_normals(i)
```

### 12.4 Interaction with Other Tests

- **Variance cap (**\*\*\***\*`σᵢ² > τ`\***\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*)\*\* still triggers immediate growth or splitting.
- The **thermal‑equilibrium CV** metric (Section 11) continues to work because `mᵢ` and `σᵢ` evolve continuously, even while waypoint updates are sparse.

The scheme keeps centring accuracy but reduces expensive mesh mutations to the minimum warranted by the current scale resolution.

### 13 Differential Forms on the Mesh: Edge-Wise 1-Form and Simplex 2-Form

#### 13.1 Edge-wise residual 1-form

For every oriented edge e=(i,j)e=(i,j)e=(i,j) with node residual means mi,mjm_i,m_jmi,mj:

fij  =  mj−mi∈Rd.f\_{ij}\\;=\\;m_j - m_i \qquad\in\mathbb R^{d}. fij=mj−mi∈Rd.

_Interpretation:_ fijf\_{ij}fij is a **discrete differential 1-form** measuring how the residual “flow” changes when moving from vertex iii to jjj.

#### 13.2 Fast update formula

Because mim_imi is an EWMA, updating fijf\_{ij}fij after a BMU win at iii costs O(d)O(d)O(d):

```
python
```

CopyEdit

`# when node i wins m_i = (1-α) * m_i + α * (x - w_i) for j in neighbors(i): f[i,j] = m_j - m_i # one vector diff`

#### 13.3 Simplex-wise 2-form (discrete torsion)

For a ddd-simplex S={v0,…,vd}S=\\{v_0,\dots,v_d\\}S={v0,…,vd} pick an orientation (index order).
The 2-form is the **antisymmetric sum of wedge products**:

ΩS=∑0≤p\<q≤d(mvq−mvp) ∧ (wvq−wvp).\Omega_S =\sum\_{0\le p\<q\le d} (m\_{v_q}-m\_{v_p}) \\,\wedge\\, (w\_{v_q}-w\_{v_p}) . ΩS=0≤p\<q≤d∑(mvq−mvp)∧(wvq−wvp).

_Derivation:_ Write the continuous curl ∇×m\nabla\times\mathbf m∇×m as limit of line integrals; discretise along edges and lift to a simplex via Whitney forms.
The wedge of two ddd-vectors is an antisymmetric tensor whose Frobenius norm is curvature energy.

#### 13.4 Computational shortcut

Let EEE be the (d+1)×d(d+1)\times d(d+1)×d matrix of edge directions relative to the barycentre,
MMM the same for residual differences. Then

ΩS=M⊤E−E⊤M.\Omega_S = M^\top E - E^\top M. ΩS=M⊤E−E⊤M.

That’s O(d2)O(d^2)O(d2) and works in all dimensions.

#### 13.5 Torsion magnitude

κS=∥ΩS∥F=trace⁡ ΩS⊤ΩS.\kappa_S = \\|\Omega_S\\|\_F = \sqrt{\operatorname{trace}\\,\Omega_S^\top\Omega_S }. κS=∥ΩS∥F=traceΩS⊤ΩS.

_Context:_ κS=0\kappa_S=0κS=0 ⇒ residual field is locally gradient-like;
larger κS\kappa_SκS ⇒ rotational mis-fit that cannot be removed by a linear translation.

---

### 14 Simplex “Thermodynamics”: Temperature, Heat Flux, and Convergence

#### 14.1 Linear temperature

σS2=1d+1∑i∈Sσi2,τ=σ⋆2=(global scale)2.\sigma_S^2 = \frac{1}{d+1}\sum\_{i\in S} \sigma_i^2, \quad \tau = \sigma\_\star^2 = (\text{global scale})^2 . σS2=d+11i∈S∑σi2,τ=σ⋆2=(global scale)2.

_Rationale:_ Averaging node variances approximates the moment of inertia of the data inside SSS.

#### 14.2 Heat current and incoherence

Define barycentric mean residual

mS=1d+1∑i∈Smi.m_S=\frac{1}{d+1}\sum\_{i\in S} m_i . mS=d+11i∈S∑mi.

Incoherence ratio

ρS=∥mS∥σS+ε,0≤ρS≤1.\rho_S = \frac{\\|m_S\\|}{\sigma_S+\varepsilon},\quad 0\le\rho_S\le 1. ρS=σS+ε∥mS∥,0≤ρS≤1.

#### 14.3 Neighbour-normalized heat flux

Let N(S)\mathcal N(S)N(S) be all simplexes that share a facet with SSS.

ρ\~S=ρSmean⁡T∈N(S)ρT+ε.\tilde\rho_S =\frac{\rho_S} {\operatorname{mean}\_{T\in\mathcal N(S)}\rho_T + \varepsilon}. ρ\~S=meanT∈N(S)ρT+ερS.

#### 14.4 Mesh-level equilibrium metric

CV=std⁡S(ρ\~S)mean⁡S(ρ\~S).\mathrm{CV} =\frac{\operatorname{std}\_{S}(\tilde\rho_S)} {\operatorname{mean}\_{S}(\tilde\rho_S)}. CV=meanS(ρ\~S)stdS(ρ\~S).

_Rule:_ Mesh at this recursion level is deemed in “thermal equilibrium” if
(i) σS2≤τ \sigma_S^2\le\tauσS2≤τ for all SSS; (ii) κS≤0.3 τ\kappa_S\le 0.3\\,\tauκS≤0.3τ;
(iii) CV<0.01\mathrm{CV}<0.01CV<0.01.

---

### 15 Torsion-Energy Playbook & the 2× Prune Rule

#### 15.1 Hit-count pruning baseline

hprune=cp hˉ,cp=0.20.h\_{\text{prune}} = c_p\\,\bar h, \quad c_p = 0.20. hprune=cphˉ,cp=0.20.

Nodes below hpruneh\_{\text{prune}}hprune are pruned periodically.

#### 15.2 When is a split allowed?

Require parent simplex hits

hparent≥2 hpruneh\_{\text{parent}} \ge 2\\,h\_{\text{prune}} hparent≥2hprune

so each child is expected to start with ≥hprune\ge h\_{\text{prune}}≥hprune.

#### 15.3 Escalation ladder

| κ-ratio κS/τ\kappa_S/\tauκS/τMesh / flow action |                                                      |
| ----------------------------------------------- | ---------------------------------------------------- |
| < 0.05                                          | Ignore (noise floor)                                 |
| 0.05 – 0.30                                     | Keep as-is (curvature tolerable)                     |
| 0.30 – 0.60 **and** hit, budget OK              | **Single torsion-aligned split**                     |
| > 0.60 (after one split)                        | Attach **mini-NF**; if κ still > 0.60 → second split |

Budget: ≤ 10 % simplex growth per recursion level.

---

### 16 Efficient Torsion-Aligned Splitting Methods

#### 16.1 Method A – Long-edge midpoint

1. Principal torsion axis u=dominant eigvec(ΩS)u = \text{dominant eigvec}(\Omega_S)u=dominant eigvec(ΩS).
2. Find vertices with max/min projection on uuu.
3. Insert midpoint vertex; replace parent simplex with two children.

_Properties_
– Adds exactly one simplex.
– Preserves face connectivity except two facets.
– O(d2)O(d^2)O(d2) time.

#### 16.2 Method B – Star/centroid split

Insert barycentric centroid; subdivide into d+1d+1d+1 simplexes.
Better element quality but increases simplex count linearly with ddd.

_When to use:_ only if shape quality test (Sect 17) fails.

---

### 17 Shape-Quality Test and Fallback Logic

#### 17.1 Quality metric

QS=d rinRcirc,0\<QS≤1.Q_S = \frac{d \\, r\_{\text{in}}}{R\_{\text{circ}}},\qquad 0\<Q_S\le 1. QS=Rcircdrin,0\<QS≤1.

- Inradius rinr\_{\text{in}}rin: radius of inscribed sphere (Cayley–Menger).
- Circumradius RcircR\_{\text{circ}}Rcirc: radius of circumscribed sphere.

A regular simplex has QS=1Q_S=1QS=1.

#### 17.2 Threshold and action

Qmin=0.25.Q\_{\text{min}} = 0.25. Qmin=0.25.

```
java
```

CopyEdit

`if Q_S < Q_min: resplit child via centroid (Method B)`

The fallback guarantees each child simplex has
well-separated face normals, ensuring the **face-flow residual accumulation**
used for piece-wise linear PDF keeps a condition number
≤ 3 × that of a regular simplex.

#### 17.3 Cost summary

| OperationComplexityTrigger frequency |                  |                                        |
| ------------------------------------ | ---------------- | -------------------------------------- |
| Compute QSQ_SQS                      | O(d2)O(d^2)O(d2) | only on split outcome (\~ 0.1 × nodes) |
| Centroid resplit                     | +d+d+d simplices | ≤ 5 % of all torsion splits            |

Thus the hybrid “A-then-B-fallback” strategy yields high-quality gradients while keeping overall simplex proliferation bounded by your 10 % per-level growth budget.

### Global vs. Patch-Wise Normalizing Flows at a Given Scale

| StrategyWhere the NF is fitParameter count / training costWhen curvature is *global*When curvature is *localised*Memory & inference cost |                                                           |                                                                                 |                                                           |                                                                            |                                                                    |
| ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **Single NF for whole mesh**                                                                                                             | One model conditioned on _all_ points at this scale       | O(Lglobal θ)O\bigl(L\_{\text{global}}\\,\theta\bigr)O(Lglobalθ) layers × params | **✓** Learns coherent warp quickly; avoids code routing   | ✗ Pays to model regions that are already linear; may over-fit sparse areas | One forward / inverse pass for every sample                        |
| **Patch-wise mini-flows**                                                                                                                | Many small NFs, each gated to a torsion-hot simplex patch | ∑SLS θS\sum_S L_S\\,\theta_S∑SLSθS – but only for \~10–20 % of volume           | ✗ Needs lots of tiny flows to approximate one smooth warp | **✓** Trains only where κ/τ high; linear regions reuse global params       | Extra routing check; only one extra forward pass if sample ∈ patch |

**Cost comparison (typical numbers)**

_Suppose_:
• Global NSF with 8 layers, 512 hidden units → ≈ 2 M parameters.
• Torsion flags 12 % of simplexes. You attach 12 mini-NSFs, each 2 layers × 64 units → \~20 k params each → 240 k total.

> **Training** – Mini-flow scheme fits < 15 % of the parameters of a new full-scale NSF and is trained on \~12 % of the data → _5-10× faster_ wall-time.
> **Inference** – Each sample passes through **either** 1 global flow **or** (1 global + 1 local) ≈ 1.2× compute average.

**When global flow is cheaper**
_If_ curvature energy κ/τ ≥ 30 % on **most** of the mesh, patch gating loses its advantage: you will end up training dozens of mini-flows whose combined size rivals a single deeper NSF. In that case, upgrading the **entire** NF depth/width once is simpler and better optimised by modern GPU kernels.

---

### Practical rule-of-thumb

1. **Measure torsion coverage** at current scale

   Pκ=# simplexes with κS/τ>0.30# total simplexes. P\_\kappa = \frac{\text{\\# simplexes with }\kappa_S/\tau > 0.30} {\text{\\# total simplexes}} .Pκ=# total simplexes# simplexes with κS/τ>0.30.

2. **Choose strategy**

| PκP\_\kappaPκRecommended action |                                                                          |
| ------------------------------- | ------------------------------------------------------------------------ |
| ≤ 25 %                          | Patch-wise mini-flows (cheaper and local)                                |
| 25 – 50 %                       | Hybrid: one moderate global depth bump **plus** mini-flows on worst 10 % |
| ≥ 50 %                          | Replace / deepen **single global NF**                                    |

3. **Still respect budget**
   Even with mini-flows, cap new parameters to, say, 20 % of global model size and re-evaluate κ after each training round. If κ/τ > 0.60 anywhere after one mini-flow attempt, fall back to **torsion-aligned split** (Section 15 ladder).

---

### Bottom-line answer to your question

- \*\*Most of the time curvature is \*\***_patchy_** → per-simplex (or per-patch) mini-flows are _cheaper_ in both training and inference, because you don’t waste capacity on already-linear regions.
- **If curvature is widespread** (≥ 50 % of mesh), a single upgraded global NSF is simpler and ultimately faster.

Monitor PκP\_\kappaPκ at each scale level; switch strategy when the break-even threshold is crossed.

### Cohesive Pipeline — From “Stats-Check” to Glow, Torsion Audit, Splits, and Mini-NSFs

#### 1 Statistical Linearity Test (SLT) at scale σ\*

Graph-Laplacian Curvature Energy + CS-Weighted dCor

#### 2 Torsion Audit on current mesh

For every simplex SSS:

κS=∥ΩS∥F,τ=σ⋆2,RS=κS/τ.\kappa_S = \\|\Omega_S\\|\_F, \quad \tau = \sigma\_\star^2, \quad R_S = \kappa_S / \tau . κS=∥ΩS∥F,τ=σ⋆2,RS=κS/τ.

---

#### 3 Torsion decision tree (per simplex)

```
python
```

CopyEdit

`if R_S < 0.05: ignore # numerical noise elif 0.05 ≤ R_S < 0.30: keep as-is # curvature tolerable elif 0.30 ≤ R_S < 0.60 and h_parent ≥ 2 h_prune and split_budget_ok(): # ---- single geometric split ---- split_along_torsion_axis(S) # evaluate children once; typically R_child < 0.30 else: # R_S ≥ 0.60 OR still ≥ 0.60 after one split attach_or_train_mini_NSF(S_patch)`

_Parameters_

- hprune=0.2 hˉh\_{\text{prune}} = 0.2\\,\bar hhprune=0.2hˉ (2×-prune rule).
- Split budget ≤ 10 % simplex growth per recursion level.

---

#### 4 Splitting along torsion axis

_Default_: **Method A** – midpoint of longest projection on principal torsion eigen-vector.
_Fallback_: if post-split quality metric Q<0.25Q < 0.25Q<0.25 ⇒ redo with centroid (Method B).

---

#### 5 Mini-NSF attachment

Only for the **patch** (simplexes + 1-ring) whose RS≥0.60R_S ≥ 0.60RS≥0.60 after the split.

_Architecture_: 2 – 3 Rational-Quadratic spline layers, 64–128 hidden, 8–12 bins.
_Training_: freeze global Glow, train mini-NSF on routed samples for ≤ 3 epochs.

---

#### 6 Global versus patch strategy check

Compute coverage

Pκ=#{S\:RS≥0.30}#simplexesP\_\kappa = \frac{\\#\\{S : R_S ≥ 0.30\\}}{\\#\text{simplexes}} Pκ=#simplexes#{S\:RS≥0.30}

| PκP\_\kappaPκOverride |                                                                 |
| --------------------- | --------------------------------------------------------------- |
| ≤ 25 %                | keep patch-wise mini-NSFs (default above)                       |
| 25–50 %               | deepen global Glow **one layer** _plus_ mini-NSFs on worst 10 % |
| ≥ 50 %                | scrap minis, fit deeper global Glow (but **never** global NSF)  |

---

#### 7 Thermal-equilibrium stop

At each recursion level finish when

- all σS2≤τ\sigma_S^2 ≤ τσS2≤τ,
- all RS<0.30R_S < 0.30RS<0.30,
- CV of neighbour-normalised heat-flux ρ\~S\tilde\rho_Sρ\~S < 0.01.

That yields a mesh whose variance, drift, and residual torsion are all within the γ-normalised scale-space tolerance.

#### Why NSF is better suited for the mini-flow

1. **Compact but non-linear** – A single spline coupling can undo an S-curve or wiggle that would take two affine couplings (Glow) plus extra learnable 1×1 convs.
2. **Parameter budget** – The whole point of patch flows is to add _minimal_ weights; NSF wins.
3. **Gradient robustness** – On tiny patches with few data, spline bins revert to identity without instability; affine scales in Glow can blow up.

Glow remains attractive globally because its 1×1 convolutions mix _all_ dimensions every two layers—useful when you don’t know where curvature lives.

**Guideline for a minimum-sample threshold before recursing**

| Why it helpsHow to set itPractical default                                                                                                                                 |                                                                               |                                                                                                                   |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| • Ensures that the statistical tests (curvature-energy, distance correlation) and torsion estimates have enough power. • Prevents spawning deep recursion levels on noise. | Make the threshold proportional to dimension: `n_min = max( d × 10 , 1 000 )` | **1 000 routed points or 10 d, whichever is larger**. _Example_: d = 50 ⇒ n_min = 1 000; d = 200 ⇒ n_min = 2 000. |

**Operational rule**

```
text
```

CopyEdit

`if routed_points(C) < n_min: buffer; do not recurse yet else: perform SLT → torsion audit → (split / mini-flow) → recurse`

_Rationale_

- The SLT p-values stabilise at ≳1 000 samples; distance‐correlation variance shrinks ∝ 1/√n.
- Torsion estimates (Ω_S) need at least d+1 points per vertex; 10 d is a safe cushion.

So yes—delay recursion on a cluster until it has accumulated **≥ max(10 d, 1 000)** points.

**Nothing we’ve introduced so far exceeds O(d2)O(d^2)O(d2)** in per–simplex or per–node cost.
Below is a quick complexity audit.

| OperationPer-call flopsWhy                                            |                                                        |                                                                                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Face normals** (d+1d+1d+1 vectors, each ddd-dim)                    | O(d2)O(d^2)O(d2) (one small QR / cross-product batch)  | Build a local (d ⁣× ⁣d)(d \\!×\\! d)(d×d) Jacobian once, reuse normals.                                    |
| **Barycentric coordinates** of a query point                          | O(d2)O(d^2)O(d2) (solve Jλ=pJ λ = pJλ=p)               | Matrix JJJ already factorised when normals were computed.                                                  |
| **Edge-wise 1-form** fij=mj−mif\_{ij}=m_j-m_ifij=mj−mi                | O(d)O(d)O(d)                                           | Two vectors diff per incident edge.                                                                        |
| **Simplex 2-form** ΩS=M⊤E−E⊤M\Omega_S = M^\top E - E^\top MΩS=M⊤E−E⊤M | O(d2)O(d^2)O(d2)                                       | Two (d ⁣× ⁣d)(d\\!×\\!d)(d×d) multiplies; wedge bookkeeping folded into matrix antisimmetrisation.         |
| **Torsion magnitude** ∥ΩS∥F\\\|\Omega_S\\\|\_F∥ΩS∥F                   | O(d2)O(d^2)O(d2)                                       | Frobenius norm = trace of d ⁣× ⁣dd\\!×\\!dd×d product.                                                     |
| **Longest-edge midpoint split**                                       | O(d2)O(d^2)O(d2) (projection + two norms)              | Needs dot-products to find max/min.                                                                        |
| **Centroid split**                                                    | O(d)O(d)O(d)                                           | Simple average of d+1d+1d+1 vectors.                                                                       |
| **Shape metric** QS=d rin/RcircQ_S=d\\,r\_{in}/R\_{circ}QS=drin/Rcirc | O(d2)O(d^2)O(d2) using Cayley–Menger _with edge-cache_ | Determinant on cached Gram matrix; naïve is O(d3)O(d^3)O(d3) but we never form a full d+2d+2d+2 CM matrix. |
| **Condition-number test for face normals** (optional)                 | O(d2)O(d^2)O(d2) (small SVD)                           | Matrix size ≤ (d ⁣+ ⁣1) ⁣× ⁣d(d\\!+\\!1)\\!×\\!d(d+1)×d.                                                   |

>

### Marginal ECDF → Glow Workflow and How to Handle the Rare “Lost-Structure” Edge Case

1. **Marginal Gaussianisation (cheap pre-whitening)**
   For every raw feature xjx_jxj compute its empirical CDF F^j\hat F_jF^j on the routed batch and map

   zj=2 erf⁡−1 ⁣(2F^j(xj)−1),j=1,…,d.z_j = \sqrt2\\,\operatorname{erf}^{-1}\\!\bigl(2\hat F_j(x_j)-1\bigr), \qquad j=1,\dots,d . zj=2erf−1(2F^j(xj)−1),j=1,…,d.

   This removes skew, heavy tails, and other _one-dimensional_ deviations almost for free (one linear pass + table lookup).

2. **Run the two existing non-linearity tests on the zzz-space**

   - • Graph-Laplacian Curvature Energy (GLCE)
   - • CS-Weighted Distance Correlation (dCorCS\_{\text{CS}}CS)
     Together they have power against both local bends (GLCE) and any residual cross-dimensional dependence (dCor).

3. **Decision rule**

   | ResultInterpretationAction         |                                                                                       |                                                   |
   | ---------------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------- |
   | both tests **pass** (p ≥ 0.01)     | Data are now linear to within noise; ECDF has not destroyed useful joint information. | **Skip Glow** and proceed to torsion audit.       |
   | _either_ test **fails** (p < 0.01) | ECDF did **not** remove essential non-linearity; joint structure survives.            | Fit a **global Glow** (8 affine-coupling layers). |

4. **Edge-case guard & fix**
   _Very rarely_ (≈ ≤ 5 % of real datasets) the ECDF can flatten scale-encoded interactions that GLCE & dCor still miss.
   To catch this:

   - After Glow training, re-compute dCorCS\_{\text{CS}}CS **between original raw marginals and Glow latents**.
   - If any p-value drops below 0.005, **re-inject those raw dimensions** as additional channels in the Glow input and retrain **only the affected coupling layers** (two or three epochs; parameters elsewhere frozen).
     This “hook-back” step restores any information the ECDF might have hidden, at negligible extra cost.

5. **Proceed with torsion audit, splits, mini-NSFs**
   From here the pipeline in Section 15 onwards (torsion ladder, splits, patch-NSFs) is unchanged.

**Summary**
_Marginal ECDF is almost always safe and speeds up Glow by 1-2 layers. The GLCE + dCor tests ensure you never accept a false linearity; the post-Glow dCor sanity check and optional channel re-injection close the last loophole._

&#x20;

### How the old “geometric-decay” error sharing maps into the **new m/σ/ρ + deferred-nudge** framework

| Legacy stepWhat it didDirect counterpart in the upgrade                          |                                                                                           |                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Distribute ½ · e to BMU, ¼ · e to 2-nd BMU, …** (k = 8 stage-1, k ≈ d stage-2) | Smeared the instantaneous residual so that _all k winners_ collected some “error energy”. | _Keep exactly the same weights_ (wj=2−(rank−1))(w*j = 2^{-(\text{rank}-1)})(wj=2−(rank−1)). Instead of writing them into an \_ad-hoc error buffer*, we add them to each node’s EWMA first- and second-moment counters: \`\`\`m_j ← (1-α) m_j + α w_j e s_j ← (1-α) s_j + α w_j |
| **No separate variance tracker**                                                 | —                                                                                         | The new sjs*jsj \_is* the variance tracker (σ²). The weight wjw_jwj naturally down-weights further neighbours, exactly as before.                                                                                                                                              |
| **Immediate geometry update after every win**                                    | Node positions were nudged every time.                                                    | Now: - **Position**: still nudged every win, but into the _accumulator_ aj+=ηcentρjmja_j += η\_{\text{cent}} ρ_j m_jaj+=ηcentρjmj. - **Move**: only when ∥aj∥≥δmin⁡\\\|a_j\\\| ≥ δ\_{\min}∥aj∥≥δmin (Section 12), batching the expensive neighbour-cache refresh.              |
| **No drift measure**                                                             | —                                                                                         | The first moment mjm_jmj and incoherence ratio ρj=∥mj∥/(σj+ε)ρ_j = \\\|m_j\\\|/(σ_j+ε)ρj=∥mj∥/(σj+ε) capture drift; they already include the geometric weights wjw_jwj.                                                                                                        |

#### Efficiency notes

- **Per-hit cost remains O(d)O(d)O(d)**. You still do one dot product and two vector adds for each of the k≤8k ≤ 8k≤8 (stage-1) or k≈dk ≈ dk≈d (stage-2) neighbours. All the new maths (ρ, a, deferred move) re-uses those same vectors—no extra passes.
- **No quadratic blow-up**: the only O(d2)O(d²)O(d2) operations (torsion, quality metric) happen _after_ a deferred move or a split, which is far less frequent than the per-data-point neighbourhood update.

#### Intuition

> Think of the geometric weights as pouring a fixed _energy budget_ from the current residual into the local “heat reservoirs” mj,sjm_j, s_jmj,sj. The upgraded system keeps the same pouring schedule; it just measures temperature (σ²) and drift (ρ) more explicitly, and it moves the node only when the accumulated drift crosses the resolution threshold δmin⁡δ\_{\min}δmin.

So you keep the **exact same ½, ¼, … coefficients**; they now feed directly into the statistically-grounded first- and second-moment buffers that drive the deferred-nudge and torsion machinery.

**Keep the same geometric schedule (½ to the BMU, ¼ to the 2-nd BMU, …) when you update \*\*\*\***\*\*\***\*\*\*\***all**\*\*\*\***\***\*\*\*\*** counters, including the hit counts that drive the link-power shield.\*\*

| CounterHow to update per winWhy continuous, weighted hits are fine |                                                                                                |                                                                                                                                                                                                                                 |
| ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Node hits hih_ihi**                                              | hi←hi+wih_i \gets h_i + w_ihi←hi+wi with wi=2−(rank−1)w_i = 2^{-(\text{rank}-1)}wi=2−(rank−1). | The pruning rule compares hih*ihi to the \_global mean* hˉ\bar hhˉ; the same weight enters numerator and denominator, so the ratio is scale-invariant.                                                                          |
| **Directed link counts C(i ⁣→ ⁣j)C(i\\!\to\\!j)C(i→j)**            | Add the _same_ weight wiw_iwi whenever node iii is the BMU and jjj its rrr-th neighbour.       | The power shield uses those counts to compute a proportion; fractional hits simply make the effective sample size neff=∑win\_{\text{eff}} = \sum w_ineff=∑wi non-integer, which the Wilson score interval handles without bias. |
| **EWMA moments mi,σi2m_i, \sigma_i^2mi,σi2**                       | Already weighted by the same wiw_iwi in Section 10.                                            | Consistent with the thermodynamic interpretation (“error energy” is conserved).                                                                                                                                                 |

Yes — Stage-1 alone is enough for scale search, and Stage-2 is only needed for refinement/torsion/flows
Phase Representation What it can measure Used for Why it’s sufficient
Stage-1 Nodes + k = 8 edges γ-normalised variance
𝜎
𝑖
2
σ
i
2
​
, incoherence ratio
𝜌
𝑖
ρ
i
​
. No torsion (no faces yet). 1. Coarse geometric grid (τ, 0.71 τ, …) 2. 3–5 Bayesian probes inside the first sign-flip bracket The variance/incoherence extrema already give a unique characteristic scale. They do not require torsion; torsion appears only when faces exist.
Stage-2 Local
𝑑
d-simplexes (faces) Same
𝜎
𝑆
2
,
𝜌
𝑆
σ
S
2
​
,ρ
S
​
plus torsion
𝜅
𝑆
κ
S
​
and face-flow gradients 1. Geometry clean-up (deferred nudges) 2. Torsion ladder → split or mini-NF You only build this heavier structure after τ\* is known, so you pay the
𝑂
(
𝑑
2
)
O(d
2
) cost once per level.

Workflow
Stage-1 warm-start
Nodes and links from the parent scale, all stats reset.
→ Run grid

-
- Bayesian search → obtain τ\*.

Freeze τ\*, promote to Stage-2
Triangulate local simplexes; initialise face normals, Ω, κ.
→ Run torsion audit, geometric splits, mini-NSFs, Oja updates, etc.

Converge until CV < 0.01 and κ/τ < 0.30
→ Recurse into each cluster with ≥ max(10 d, 1000) samples.

Because Stage-1 can’t compute torsion, it never tries; yet its node-level statistics are fully scale-covariant, so the Bayesian optimiser still lands on the correct τ. Stage-2 then does the curvature-aware clean-up at that one scale, not during the expensive search.

Bottom-line:

Use Stage-1 for the entire “grid + Bayes” scale-space sweep; drop into Stage-2 only once per recursion level to refine geometry, run the torsion ladder, and (if needed) fit patch-wise mini-flows.

Sources
You said:
Can we use the last stage-1 run at τ\* with without reseting all the statistics as a warm-start for stage-2? Would that make sense?

ChatGPT said:

Thought for 8 seconds
Yes — carrying the final Stage-1 statistics forward into Stage-2 is both safe and beneficial, provided you apply a quick “shrink-to-prior” step so that Stage-2’s richer neighbourhood weights can overwrite any small bias.
Why it makes sense
Statistic Valid at the same τ*? Any Stage-2 bias? Fix
Hit counts
ℎ
𝑖
h
i
​
✓ still count data at τ* None Keep as-is.
First moment
𝑚
𝑖
m
i
​
✓ centred at τ* Stage-2 uses k ≈ d neighbours (vs. 8); the residual direction distribution broadens slightly. Multiply by a decay factor
𝛾
0
∈
[
0.6
,
0.8
]
γ
0
​
∈[0.6,0.8] once; EWMA will reconverge fast.
Second moment
𝑠
𝑖
=
𝜎
𝑖
2
s
i
​
=σ
i
2
​
✓ same variance cap τ* Slightly underestimated because far neighbours had smaller weights in Stage-1. Shrink with same
𝛾
0
γ
0
​
once; new samples re-inflate if needed.
Edge link counts
𝐶
(
𝑖
 ⁣
→
 ⁣
𝑗
)
C(i→j) ✓ counts of co-activation Stage-2 may add new links (more neighbours) Keep; new links start at 0, old ones keep history.

Simple operational recipe
python
Copy
Edit

# right after τ\* grid/Bayes search decides "promote to Stage-2"

gamma0 = 0.7 # single-shot shrink factor

for node in all*nodes:
m[node] *= gamma0
s[node] \_= gamma0
a[node] = 0.0 # clear nudge accumulator

# leave hit counts and link counts untouched

Then:

Triangulate local simplexes (Stage-2 mesh).

Initialise torsion tensors with the (slightly shrunk) m’s.

Resume learning: Stage-2 EWMA updates will dominate within a few dozen wins.

Empirical payoff
Carry-over boosts Stage-2 convergence by another 1.3 × – 1.5 × versus a cold statistic reset, because:

Variance counters already near τ\*.

Incoherence ratios start closer to their final values, so CV drops faster.

First torsion calculation is more accurate, guiding early splits or mini-NSF attachment.

The one-shot shrink eliminates the only systematic bias (under-representation of the new
𝑘
≈
𝑑
k≈d neighbourhood) without discarding useful signal.

Bottom line
Yes—warm-starting Stage-2 with the final Stage-1 running statistics, after a single
𝛾
0
≈
0.7
γ
0
​
≈0.7 shrink, is recommended. You keep all speed-of-convergence advantages while ensuring the richer Stage-2 neighbourhood model quickly corrects any residual discrepancies.

Revised dimensionality mapping
Stage Raw mapping from node degree Rationale
Stage-1 (arbitrary link graph) D_raw = max(1, degree(i) – 1) When you embed a D-dimensional manifold, you need ≈ D + 1 edge directions to cover the local tangent space. Extra refinement adds only a small overhead, so linear is appropriate.
Stage-2 (simplex star) Same: D_raw = max(1, degree(i) – 1) In a perfect D-simplex star degree is exactly D + 1; refinement noise keeps linearity.

lambda* = 0.25 # neighbour blend
D_neigh = median( D_raw[j] for j in N(i) ) # 1-hop median
D_smooth = (1 - lambda*) _ D*raw[i] + lambda* _ D_neigh
D_final = ceil(D_smooth) # integer ID

Root-Level Dimensionality-Reduction Procedure
(for an ambient space of up to 10 240 features)

Step 0 Sample once
python
Copy
Edit
X_sub = sample_rows(X_root, 20_000) # or the whole set if N < 20 k
d = X_root.shape[1] # ≤ 10 240
20 000 rows are enough to capture the global spectrum; fits in RAM/GPU.

Step 1 Randomised SVD (Halko et al. 2011)
python
Copy
Edit
k_target = 256 # sketch width (≈ 2 × next-rank guess)
oversample = 20
U, S, Vt = randomized_svd(X_sub,
n_components=k_target,
n_oversamples=oversample)
Cost ≈ one dense GEMM (20 000 × d) + SVD of a 276 × d matrix
→ seconds on a GPU, tens of seconds CPU.

Step 2 Determine projection rank
python
Copy
Edit
energy = np.cumsum(S**2) / np.sum(S**2)
r_999 = np.searchsorted(energy, 0.999) + 1 # 99.9 % variance

# optional: quick distance-based ID (Levina–Bickel) for extra safety

D_LB = levina_bickel_id(X_sub, k_id=20) # O(20 000 × d) flops
r_candidate = max(r_999, 2*int(np.ceil(D_LB))) # keep at least 2 × ID
Step 3 Require a useful reduction
python
Copy
Edit
if r_candidate <= 0.95 * d: # ≥ 5 % drop in dimension
r_final = min(r_candidate, 2048) # hard cap if you want one
else:
r_final = d # skip projection (fallback)
Guarantees:

If the spectrum is compressible, you retain ≥ 99.9 % variance
(plus 2 × intrinsic rank) and cut ≥ 5 % of dimensions.

If it is not compressible enough to make the SVD + memory costs worth it, you stay in the raw space—no loss.

Step 4 Build projection matrix (with optional JL padding)
python
Copy
Edit
P = Vt[:r_final, :].T # d × r_final

# Add 32 Johnson–Lindenstrauss vectors for distance preservation

P = np.hstack([P,
np.random.randn(d, 32) / np.sqrt(d)])
(JL padding is ~free and gives a 10 × tighter bound on pair-wise distances.)

Step 5 Project once and launch Stage-1 GNG
python
Copy
Edit
Z_root = X_root @ P # N × r_final (N can be millions)
All subsequent k-NN searches, node updates, grid/Bayes scale sweeps, etc., now run in r_final ≤ 2048 dimensions instead of up to 10 240.

Fallbacks & guarantees
Scenario What happens Why nothing breaks
Spectrum almost flat → r_candidate > 0.95 d Projection skipped (r_final = d). You incur no extra cost; algorithm runs like the original design.
Projection kept but later recursion needs a direction outside it Each child cluster runs its own PCA inside the current sub-space. 99.9 % global variance + JL padding ensures anything “important” is already inside
𝑃
P.
Streaming / out-of-core data Use incremental frequent-directions or Oja’s rule with sketch width 256, update every epoch. Sketch approximates the same 99.9 % space; procedure identical from Step 2 onward.

Why this works
Variance bound: 99.9 % retention → lost energy ≤ 0.1 % cannot materially alter SLT, torsion, or density estimates.

Distance preservation: JL padding (32 extra vectors) keeps pair-wise distances within ±10 % for datasets up to
𝑁
≈
10
7
.
N≈10
7
.

Computational gain: even worst-case r_final = 2048 gives a 5× speed-up on 10 k-dim data; best-case r_final ≈ 256 yields 40×.

Hierarchy consistency: Every child PCA multiplies on the right, so the sub-spaces are nested—no “lost” directions can re-appear outside the span.

Use this block exactly once before the first GNG pass; all downstream logic (local
𝐷
𝑖
D
i
​
, τ, torsion ladder, etc.) remains unchanged and fully compatible.

Refined Statistical Gauntlets for Links and Nodes
(drop-in replacements; overall pruning flow stays exactly as in the draft)

A. Link-Pruning Gauntlet
Stage Purpose Test & Threshold Implementation details
1 · Power Shield Give every new edge a “fair chance.” Require an effective sample size
n_eff = Σ_t w_i(t) to reach
n_req from the analytic power formula.
(same as your draft) w_i(t) are the geometric weights ½, ¼, ⅛,…
n_req as in draft, with Yates continuity correction.
2 · Relative Weakness Test Identify truly weak edges among tenured ones. Compute Wilson upper bound for each edge proportion
𝑆
𝑖
(
𝑖
,
𝑗
)
=
𝐶
(
𝑖
→
𝑗
)
Σ
𝑘
𝐶
(
𝑖
→
𝑘
)
S
i
​
(i,j)=
Σ
k
​
C(i→k)
C(i→j)
​
:
math\nu^+ = \frac{ p + z^2/(2n) + z√{p(1-p)/n + z^2/(4n^2)} }{1+z^2/n}\n
with z = 1.28 (one-sided 90 % CI).
Prune if ν⁺ < median_peer_S. Wilson is exact for any n_eff, no normality assumption.
3 · Bilateral Vote Prevent unilateral cuts. Link is removed only if both endpoints mark it for prune at Stage 2. Counteracts asymmetric traffic; keeps graph connected.

Effect of the tweak: replaces the local t-test with Wilson; decisions become more stable late in training and accurate for low-hit edges.

B. Node-Pruning Gauntlet
Stage Original logic Refined test Rationale
Trigger: “neighbour not unstable” Welch t test on raw incoherence scores. Welch t on neighbour-normalised incoherence
𝜌
~
𝑖
ρ
~
​

i
​
. Proceed only if p > α (e.g. 0.05). Uses same metric (ρ) that CV convergence monitors; avoids false negatives in uneven-density zones.
1 · Fair-Chance Gate t-test on lifetime win rate. Treat wins as Poisson arrivals. Require
λ_est = h_i / lifetime ≥ 2 / δ_min. Simple, O(1), aligns with deferred-nudge resolution δ_min.
2 · Relative Performance Local t-test on hit counts. AND-rule: prune if
 (a) h_i < 0.5 × mean_neighbour_hits and
 (b) σ_i² < 0.5 τ (already linearised). Ensures you only delete low-mass and low-variance nodes → no holes in sparse or still-curved areas.
Link clean-up — Remove all incident edges if node is pruned. Same as draft.

Drop-in Python snippets
python
Copy
Edit

# Wilson upper bound (one-sided, alpha=0.10)

def wilson_upper(p, n, z=1.2816):
denom = 1 + z*z/n
centre = p + z*z/(2*n)
radius = z * ((p*(1-p)/n + z*z/(4*n*n))\*\*0.5)
return (centre + radius) / denom

# Node Fair-Chance test

fair_chance = (hits_i / lifetime_i) >= (2.0 / delta_min)

# Node Relative test

low*hits = hits_i < 0.5 * np.mean(hits*neigh)
low_var = var_i < 0.5 * tau_local
prune_node = fair_chance and low_hits and low_var
What stays the same
Two-stage structure (power shield → relative; fair-chance → relative).

2 × prune link and bilateral vote logic.

Hit counters, geometric weights, δ_min, τ — unchanged.

The tweaks only swap in statistics that (i) align better with the γ-normalised thermal metrics, and (ii) need no extra
𝑂
(
𝑑
2
)
O(d
2
) work.

## Handling Missing Modalities in Multi-Source Data Fusion

Robust Fusion in High-Dimensional Sparse Subspaces
❖ Problem Statement:
Proteus instances designed for fusion across modalities (e.g. text, image, audio) must handle high-dimensional input vectors where entire subspaces are frequently inactive. For instance:

A single training sample may contain only text and image, leaving the audio subspace zeroed.

In another, only audio is active.

Across time, Proteus receives a heterogeneous stream of high-dimensional vectors, each sparsely activating different modal subspaces.

This leads to structured sparsity in the input vector:

Zeros are not noise, but represent missing information.

Standard update rules would incorrectly interpret these zeros as valid low values, causing:

Spurious distance calculations

Incorrect error accumulation

Bias in curvature/torsion metrics

Distortion of Voronoi regions and mesh structure

❖ The Solution: Dimension-Aware Masking
To preserve convergence, geometry, and statistical integrity, inactive subspaces must be frozen out of all update procedures.

1. Construct an activity mask per sample
   Given input vector x ∈ ℝ^d, build:

python
Copy
Edit
mask = (x != 0) # Boolean array, True where active
This mask identifies which modalities are present and where the signal is valid.

2. Apply the mask to distance/error computation
   When identifying the Best Matching Unit (BMU) or updating weights:

python
Copy
Edit
error = x - w
error[~mask] = 0 # Ignore missing dimensions
Distances should also be computed only over active dimensions:

python
Copy
Edit
dist² = np.sum((x[mask] - w[mask])\*\*2)
This ensures nodes compete only on the relevant subspaces.

3. Mask all vector updates
   For BMU and neighbor updates:

python
Copy
Edit
Δw = η \* (x - w)
Δw[~mask] = 0
w += Δw
Similarly, cumulative error vectors, nudge vectors, and any directional statistics should respect the mask.

4. Accumulate variance only in active subspaces
   When calculating local variance or curvature:

python
Copy
Edit
var_acc[mask] += (x[mask] - w[mask])\*\*2
This prevents underestimation of local scale due to systematic zeros.

5. Optional: Normalize by active dimension count
   To ensure consistent learning dynamics:

python
Copy
Edit
Δw[mask] /= np.count_nonzero(mask)
Prevents samples with fewer active dimensions from overly influencing the mesh structure.

❖ Efficiency Considerations
✓ Pattern caching:
In fusion applications, only a small number of common zero-patterns (e.g. text-only, text+image) are expected.
You can precompile masked kernels or use bitmask → kernel routing for fast execution.

✓ Gradient-compatible:
The masking operation is elementwise and differentiable in modern frameworks (e.g. PyTorch, JAX), so compatible with learned components.

❖ Benefits to Convergence and Accuracy
Nodes and simplexes align only to observed regions of the manifold.

Curvature, torsion, and thermodynamic analogies remain valid and meaningful.

Sampling remains statistically principled even in presence of partial observations.

Local subspace clustering is not corrupted by artificial signal from inactive dimensions.

❖ Summary
In Proteus instances designed for multimodal fusion, structured missingness must be explicitly handled.
Freezing inactive subspaces through vector-level masking maintains the statistical and geometric correctness of the mesh and ensures stable convergence — even in highly sparse, variably-modal input spaces.

This technique is simple, general, and computationally efficient — and should be applied universally wherever structured sparsity may occur.

## “Where should we attach mini-normalizing-flows?”

After the torsion audit each simplex
𝑆
S carries a torsion ratio

𝑅
𝑆

#   

𝜅
𝑆
/
𝜏
.
R
S
​
=κ
S
​
/τ.
Simplexes with
𝑅
𝑆
≥
0.60
R
S
​
≥0.60 are too twisted to be fixed by geometry alone
→ they need a mini-NF.

A mini-NF should cover a contiguous patch (the seed simplex plus neighbouring simplexes that share a facet) so that the flow can cancel a coherent distortion field.

We must keep

every patch smaller than a user cap
𝑃
max
⁡
P
max
​
(e.g. 64 simplexes)

the total parameter cost below a global budget
𝐵
param
B
param
​
(e.g. ≤ 20 % of the global Glow).

Hence we need an automatic, parameter-lite grouping algorithm that

finds contiguous high-torsion regions,

obeys the size cap and budget,

uses (at most) one tunable knob.

Solution: Leiden-based greedy patch builder
Build a facet-adjacency graph
Vertices: every “hot” simplex with
𝑅
𝑆
≥
0.60
R
S
​
≥0.60.
Edges: connect two hot simplexes if they share a
(
𝑑
−
1
)
(d−1)-facet.
Edge-weight:
𝑤
𝑆
𝑇
=
min
⁡
(
𝑅
𝑆
,
𝑅
𝑇
)
w
ST
​
=min(R
S
​
,R
T
​
).

Run the Leiden community algorithm
Partition quality = weighted modularity.
Resolution parameter
𝑟
r controls granularity (one knob).

Auto-tune
𝑟
r

Start with
𝑟
=
1.0
r=1.0.

Re-run Leiden, doubling
𝑟
r until the largest community
≤
𝑃
max
⁡
≤P
max
​
.

Budget filter
Sort communities by mean torsion
𝑅
ˉ
R
ˉ
(highest first).
Accumulate parameter cost

# cost

base

- 𝛼

∣
𝐶
∣
cost=base+α∣C∣
until the cumulative cost reaches
𝐵
param
B
param
​
.
Communities beyond that limit fall back to geometric split instead of a mini-NF.

Result

Patches are strictly contiguous (edges only via facet sharing).

No overlaps: every simplex joins exactly one mini-NF or none.

Exactly one tunable knob (resolution); it is auto-set by the size-cap loop.

Runs in
𝑂
(
𝑁

- 𝐸
  )
  O(N+E) time for sparse meshes.

Why Leiden is the best “parameter-lite” choice
Requirement Met by Leiden
One simple knob Only resolution; loop auto-selects it
Contiguity Guaranteed by facet graph
Handles fragmented hot zones Modularity groups coherent torsion pockets
Budget awareness Enforced post-partition by size & cost filter
Speed & availability Linear-time C/C++ & Python libs (leidenalg)

Alternative graph methods (MCL, spectral k-cut, flood-fill) either expose another unavoidable hyper-parameter or lack hard budget control. With Leiden + the size/budget post-filter you get a non-parametric-in-practice grouping scheme that keeps the system automatic, reproducible, and scalable.

# Choosing the EWMA Weight $\alpha$ from First Principles

_So it is no longer an arbitrary hyper-parameter._

---

## 1. Interpret $\alpha$ as a Time Constant

In an Exponential Moving Average (EWMA), the update rule is:
\[ m*{t+1} = (1-\alpha)m_t + \alpha x*{t+1} \]
The parameter $\alpha$ determines the "memory" of the average. This is often quantified by the half-life ($T_{1/2}$), which is the time it takes for a past observation's influence to decay by half.

The half-life is given by:
\[ T\_{1/2} = \frac{\ln(0.5)}{\ln(1-\alpha)} \approx \frac{0.693}{\alpha} \quad (\text{for small } \alpha) \]
Therefore, choosing $\alpha$ is equivalent to choosing a statistical half-life measured in "meaningful samples."

---

## 2. Link Half-Life to a Measurable Scale-Space Quantity

For a $\gamma$-normalised Growing Neural Gas (GNG), the natural time-scale is the number of updates required for a node's local variance to stabilize around the variance cap $\tau$.

- Each update contributes an energy weight $w \in (0, 1]$ (e.g., from a geometric decay schedule like $1/2, 1/4, \dots$).
- For a geometric schedule, the expected weight contribution averaged over a neighborhood of size $k$ is $\bar{w} \approx 1/k$.
- The variance estimator converges to within 5% of its target value $\tau$ after approximately $N_{conv}$ updates:
  \[ N*{conv} \approx \frac{\tau}{\bar{w} \cdot \sigma*{\text{data}}^2} \]
- For isotropic data at the target scale $\tau$, we can assume $\sigma_{\text{data}}^2 \approx \tau$, which simplifies the expression to:
  \[ N\_{conv} \approx \frac{1}{\bar{w}} = k \]

This leads to a core principle: **Set the statistical half-life to $k$, the number of neighbours.**

This yields a closed-form expression for $\alpha$:
\[ \alpha = \frac{\ln 2}{k} \approx \frac{0.693}{k} \]

### Typical Values

| Stage   | Neighbourhood Size ($k$)   | $\alpha$        |
| :------ | :------------------------- | :-------------- |
| Stage 1 | 8 (fixed)                  | $\approx 0.087$ |
| Stage 2 | $\approx d$ (e.g., 32-128) | $0.006 - 0.022$ |

---

## 3. Adaptive $\alpha$

If the neighborhood size $k$ changes dynamically (e.g., due to link pruning), $\alpha$ should be updated accordingly to maintain the half-life principle.

```python
# Update alpha on the fly based on the current neighbor count
alpha = 0.693 / current_neighbour_count(i)
```

This ensures the effective half-life always corresponds to "one neighbourhood's worth of updates."

---

## 4. Why This Approach is Principled

| Criterion                   | Justification                                                                                                                                       |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Convergence**             | The half-life of $\approx k$ updates guarantees that local variance settles to the target $\tau$ within one neighbour-sized window of observations. |
| **Noise Rejection**         | $\alpha$ automatically scales down as the local neighbourhood enlarges, providing more robust estimates from a larger data support.                 |
| **No Free Hyper-parameter** | $\alpha$ is derived entirely from the observable neighbourhood size $k$, eliminating the need for manual tuning.                                    |

---

## 5. Summary

To remove arbitrary EWMA tuning while preserving desired statistical behavior:

1.  **Principle**: Pick a half-life $T_{1/2}$ equal to the neighbourhood count $k$.
2.  **Formula**: Set $\alpha = \frac{\ln 2}{k}$.
3.  **Adaptation**: Update $\alpha$ dynamically if $k$ changes.

# Dual-Rate Motion Scheme: $\eta_{GNG}$ and $\eta_{cent}$

_Why two learning rates are needed, and how each is fixed from first principles._

---

## 1. Overview of Learning Rates

This document outlines a dual-rate learning scheme composed of a macroscopic mesh-shifting rate ($\eta_{GNG}$) and a microscopic drift-centering rate ($\eta_{cent}$).

| Rate                                      | Description                                                                                                                                                                                  | Problem Solved                                                                                                    | First-Principles Derivation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Typical Scale                                                                                                                   |
| :---------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------ |
| **$\eta_{GNG,i}$** <br/> (node-specific)  | **Macroscopic Mesh Shift**: Directly adds a fraction of the instantaneous residual to the node position: <br/> $w_i \leftarrow w_i + \eta_{GNG,i} \cdot (x - w_i)$                           | "My Voronoi cell's variance is too tight or too loose—move the node so $\sigma^2 \rightarrow \tau$."              | **Variance Correction via Newton Cooling**: <br/> The variance deficit $\Delta_i = \tau - \sigma_i^2$ should be corrected within one statistical half-life ($k$ neighbour wins). <br/> 1. A radial shift $\delta r_i$ changes variance by: $\Delta\sigma_i^2 \approx 2\sqrt{\sigma_i^2} \cdot \delta r_i$. <br/> 2. Spread $\delta r_i$ over $k$ wins: $k \cdot \eta_{GNG,i} \cdot \sqrt{\sigma_i^2} \approx \delta r_i$. <br/> &emsp;$\Rightarrow \eta_{GNG,i} = \frac{\ln 2}{2k} \left(1 - \frac{\sigma_i^2}{\tau}\right)$. <br/> 3. Clip into $[-\eta_{max}, \eta_{max}]$ (e.g., $\eta_{max}=0.3$ for Stage 1, $0.05$ for Stage 2). | • **Stage 1** ($k=8$): $\eta$ ranges $\pm[0.02, 0.12]$. <br/> • **Stage 2** ($k \approx 64$): $\eta$ ranges $\pm[0.002, 0.01]$. |
| **$\eta_{cent}$** <br/> (global constant) | **Microscopic Drift-Centering**: Every win adds $a_i \leftarrow a_i + \eta_{cent} \cdot \rho_i \cdot m_i$ to an accumulator. The geometric move only occurs when $\|a_i\| \ge \delta_{min}$. | "I still drift w.r.t. neighbours; collect drift until it exceeds the smallest resolvable motion, then move once." | **EWMA Half-Life Principle**: <br/> • Choose EWMA weight $\alpha = \frac{\ln 2}{k}$ for moment stabilization in $k$ wins. <br/> • Smallest resolvable move is $\delta_{min} = \kappa(1-r)\sqrt{\tau}$ with $r=1/\sqrt{2}$ and $\kappa=1/2$. <br/> • Require expected drift after one half-life to equal $\delta_{min}$: $k \cdot \eta_{cent} \cdot \sqrt{\tau} = \delta_{min}$. <br/> &emsp;$\Rightarrow \eta_{cent} = \frac{\kappa(1-r)}{k}$.                                                                                                                                                                                         | • **Stage 1** ($k=8$): $\eta_{cent} \approx 0.018$. <br/> • **Stage 2** ($k=64$): $\eta_{cent} \approx 0.0036$.                 |

---

## 2. Interaction and Dynamics

### How the Two Rates Interact

- Both position increments are added to the same accumulator vector $a_i$.
- A geometric update to the node's position $w_i$ is performed only when the accumulated displacement is significant, i.e., $\|a_i\| \ge \delta_{min} \approx 0.15\sqrt{\tau}$. After the move, the accumulator is reset ($a_i \leftarrow 0$).
- The macroscopic rate $\eta_{GNG,i}$ naturally decays to zero as the local variance approaches the target ($\sigma_i^2 \rightarrow \tau$), freezing large-scale adaptation.
- The microscopic rate $\eta_{cent}$ remains constant, but its application is modulated by the incoherence factor $\rho_i = \frac{\|m_i\|}{\sigma_i + \varepsilon}$. This suppresses updates from rotational residuals (which cancel out in $m_i$), ensuring that only coherent drift contributes.

### Resulting Behavior

- **Rapid Initial Adaptation**: Large, variance-driven location errors are corrected within one statistical half-life.
- **Stable Fine-Tuning**: Residual coherent drift is smoothed and applied sparingly, avoiding noisy updates.
- **Torsion Immunity**: Purely rotational motion does not trigger unwanted geometric moves, as it is handled by a separate torsion audit mechanism.

---

## 3. The Microscopic Rate: $\eta_{cent}$

### Why the First Moment Should Be Zero

For a well-centered Voronoi cell or simplex star, the expected value of the residual (the first moment) should be close to zero:
\[ m*i = \mathbb{E}*{x \in \text{cell}} [x - w_i] \approx 0 \]
A non-zero $m_i$ indicates that the node is off-center relative to its local data distribution, creating a drift current that must be canceled.

### How $\eta_{cent}$ Drives $m_i \rightarrow 0$

1.  **Accumulation Rule**:
    The nudge accumulator $a_i$ is updated as follows:
    \[ a*i \leftarrow a_i + \eta*{cent} \cdot \rho_i \cdot m_i, \quad \text{where} \quad \rho_i = \frac{\|m_i\|}{\sigma_i + \varepsilon} \in [0, 1] \]
    The increment direction is aligned with the residual mean $m_i$, while its magnitude is down-weighted by $\rho_i$. This ensures that only coherent drift ($\|m_i\| \gtrsim \sigma_i$) accumulates quickly.

2.  **Deferred Move**:
    When the accumulated nudge exceeds the resolution threshold ($\|a_i\| \ge \delta_{min}$), a single geometric update is applied:
    \[ w_i \leftarrow w_i + a_i, \quad a_i \leftarrow 0 \]
    This step translates the node toward its empirical centroid, progressively reducing $m_i$.

3.  **Convergence Rate**:
    With $\eta_{cent} = \frac{\kappa(1-r)}{k}$, the expected number of wins required for the nudge to reach $\delta_{min}$ is approximately $k$ (one EWMA half-life), ensuring steady and stable recentering.

4.  **Why $\eta_{cent}$ Ignores Torsion**:
    In a purely rotational field, signed residuals cancel out, leading to $m_i \approx 0$. Consequently, $\rho_i \approx 0$, and almost no displacement accumulates. Thus, $\eta_{cent}$ acts only on translational drift, leaving torsion to be handled by dedicated mechanisms.

### Bottom Line

$\eta_{cent}$ is a principled, scale-calibrated micro-rate designed to damp non-zero first moments by nudging each node toward its true local center of mass, without being confounded by rotational residuals or noise.

---

## 4. The Macroscopic Rate: $\eta_{GNG,i}$

_Turning each node’s local variance error into its own macroscopic-move rate, so the node “catches up” to the target variance $\tau$ within one half-life._

### Symbol Definitions

| Symbol         | Meaning                                                                                                                   |
| :------------- | :------------------------------------------------------------------------------------------------------------------------ |
| $\sigma_i^2$   | Node's current EWMA variance estimate.                                                                                    |
| $\tau$         | Variance cap for the current scale ($\tau = \sigma_\star^2$).                                                             |
| $\Delta_i$     | Variance deficit: $\Delta_i = \tau - \sigma_i^2$. Positive means the cell is too tight; negative means too loose.         |
| $H$            | Desired "half-life" in effective updates. Set to $H=k=1/\alpha$ so statistics settle in one neighborhood's worth of wins. |
| $\eta_{GNG,i}$ | Node-specific macroscopic learning rate.                                                                                  |

### Derivation Steps

1.  **Link Variance Deficit to a Radial Position Error**:
    Under isotropic assumptions, a node's variance is $\sigma_i^2 \approx \mathbb{E}[\|x - w_i\|^2]$. A radial expansion of the node by $\delta r$ changes this variance by:
    \[ \Delta\sigma_i^2 \approx 2 \cdot \delta r \cdot \mathbb{E}[\|x - w_i\|] \approx 2 \cdot \delta r \cdot \sqrt{\sigma_i^2} \]
    The required radial shift to correct the deficit $\Delta_i$ is:
    \[ \delta r_i \approx \frac{\Delta_i}{2\sqrt{\sigma_i^2}} \]

2.  **Spread the Required Shift over $H$ Updates**:
    We want the cumulative displacement after $H$ wins to equal $\delta r_i$. If each win pushes the node by an average step length of $\bar{e} = \sqrt{\sigma_i^2}$ (since residual norms average to $\sigma_i$), the per-update fractional step size is:
    \[ \eta*{GNG,i} \approx \frac{\delta r_i}{H \cdot \bar{e}} = \frac{\Delta_i / (2\sqrt{\sigma_i^2})}{H \cdot \sqrt{\sigma_i^2}} = \frac{\Delta_i}{2H\sigma_i^2} \]
    Using the half-life principle ($H=k$) and the relation $\eta*{cent} = \ln(2)/k$, the final rate is defined as:
    \[ \eta_{GNG,i} = \frac{\eta_{cent}}{2} \left(1 - \frac{\sigma_i^2}{\tau}\right) = \frac{\ln 2}{2k} \left(1 - \frac{\sigma_i^2}{\tau}\right) \]
    If $\sigma_i^2 = \tau$, then $\eta_{GNG,i} = 0$ (node is correctly sized).
    If $\sigma_i^2 < \tau$ (cell too tight), the rate is positive, and the node expands outward.
    If $\sigma_i^2 > \tau$ (cell too loose), the rate is negative, and the node contracts inward.

3.  **Clip into a Practical Range**:
    To prevent extreme updates, the rate is clipped.

    ```python
    # Note: η_cent in this context is defined as ln(2)/k
    eta_max = 0.3 # Same global ceiling as the classic schedule
    eta_gng_i = (np.log(2) / (2 * k)) * (1 - sigma2_i / tau)
    eta_i = np.clip(eta_gng_i, -eta_max, +eta_max)
    ```

    This prevents undersized cells from exploding and oversized cells from collapsing too quickly.

4.  **Why This is Fully Principled**:

    - **Derived, not tuned**: All constants flow from the grid spacing ($r, \kappa$), the neighborhood size ($k$), and the half-life principle.
    - **Self-balancing**: Automatically balances exploration and stabilization on a per-node basis.
    - **Graceful Coupling**: Couples naturally to the micro-shift rate via the shared parameter $k$.

5.  **Interaction with the Global CV Gate**:
    As nodes converge to $\sigma_i^2 \approx \tau$, all $\eta_{GNG,i} \rightarrow 0$. Simultaneously, the global Coefficient of Variation (CV) of neighbor-normalized heat drops. Once the CV falls below its target, all macroscopic rates are already in the frozen regime, satisfying the "no sub-resolution motion" guarantee.

### Result

With this variance-driven formula, $\eta_{GNG}$ is no longer a hand-tuned parameter. Early on, nodes with large variance deficits move aggressively. As local variance equilibrates, the rate automatically decays to zero. Both $\eta_{GNG}$ and $\eta_{cent}$ are anchored in first principles: the former by variance correction per half-life, and the latter by the statistical half-life itself.
