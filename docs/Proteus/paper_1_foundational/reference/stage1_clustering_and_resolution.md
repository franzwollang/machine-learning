# Stage 1 Clustering and Resolution Theory

This is the unified working reference for Stage 1 scale-conditioned clustering,
feature detectability, and inter-mode resolvability. It replaces three formerly
separate notes (`partition_inference_context.md`,
`scale-conditioned_correlation_clustering.md`, `resolution_theory.md`) with a
single deductive narrative that proceeds from foundational axioms through a
precise cluster definition to the operational machinery and its resolution
limits. The formal manuscript treatment is in `paper.tex` and `SI.tex`;
implementation lives in `code/proteus/src/proteus/stage1/clustering.py` and
`recursion.py`.

---

## Part 0: Axioms and Operating Domain

This section establishes the foundational assumptions that define the space in
which Proteus operates. These are not approximations or simplifications — they
are the correct characterization of any real observational setting.

### Axiom 1 (Positive noise)

For any real observation process, measurement noise is strictly positive:

\[
\sigma\_{\mathrm{noise}} > 0.
\]

No physical datum occupies a zero-volume set. Every measurement is the true
signal convolved with some positive-variance observation kernel. This is not a
nuisance to be removed — it is a structural feature of the operating domain.

### Axiom 2 (Density existence)

Since \(\sigma\_{\mathrm{noise}} > 0\), the data-generating process admits a
probability density \(\rho(x)\) with respect to Lebesgue measure on
\(\mathbb{R}^D\):

\[
\rho : \mathbb{R}^D \to \mathbb{R}_{\ge 0},
\qquad
\int_{\mathbb{R}^D} \rho(x)\,dx = 1.
\]

This density is the fundamental object of analysis. That does **not** mean
Proteus is uninterested in lower-dimensional latent structure; rather, it means
that the latent structure of interest appears to us through a noisy,
full-support distribution in ambient space. The relevant object is therefore
not an ideal zero-thickness manifold or a raw discrete point set, but the
continuous latent distribution induced by a fuzzy / noisy manifold-like source.
The data-generating distribution has full-dimensional support in some
neighborhood of any observation.

**Consequence:** scale-space theory applies directly. Gaussian smoothing of
\(\rho\) by variance \(\tau\) produces a well-defined family
\(\rho*\tau = \rho \* \phi*\tau\), and features are defined as structures in this
family that persist across a range of scales.

### Axiom 3 (Tissue universality)

Any density in the operating domain admits a decomposition

\[
\rho(x) = u(x) + \sum\_{i=1}^m f_i(x),
\]

where:

- \(u(x) \ge u\_{\min} > 0\) is a slowly-varying background floor (the
  **tissue**), representing measurement noise, unresolved fine structure,
  contamination, and ambient stochastic flux.
- Each \(f_i(x) \ge 0\) is a localized excess (a **feature** or **mode**)
  concentrated on a region much smaller than the full support.

The tissue floor is bounded below: \(u\_{\min} > 0\) everywhere within the
region of interest. This is the direct consequence of Axiom 1 — positive noise
guarantees a positive density floor throughout the observation domain.

#### Global and local noise floors

The tissue floor \(u(x)\) decomposes further into two conceptually distinct
contributions:

\[
u(x) = u*{\mathrm{global}} + u*{\mathrm{local}}(x),
\qquad u*{\mathrm{global}} > 0,
\quad u*{\mathrm{local}}(x) \ge 0.
\]

- **Global floor \(u\_{\mathrm{global}}\):** The ambient-space-wide minimum
  density, arising from instrument noise, cosmic ray backgrounds, or other general
  irreducible measurement stochasticity. It extends uniformly across the entire
  support of the data space and is independent of any particular manifold or
  feature. In practice, the global floor is subsumed and factored out early in
  the analysis — once the scaffold has identified _any_ local structure, the
  ambient-wide floor ceases to be the relevant contrast reference.

- **Local floor \(u\_{\mathrm{local}}(x)\):** The excess tissue specific to a
  particular manifold region or subregion. This arises from unresolved
  fine-scale structure within a cluster, local contamination, or the transverse
  scatter of nearby features blurring into the inter-feature region. The local
  floor is what the recursion actually encounters at each level: once a coarse
  cluster has been isolated, its _internal_ tissue floor \(u\_{\mathrm{local}}\)
  becomes the relevant detectability reference for discovering sub-structure.

**Operational consequence:** At the root level, the global floor determines
whether any structure is detectable at all. But as recursion narrows to a
subregion (a cluster's own data), the effective tissue floor becomes the local
floor within that subregion. All detectability and separability conditions
(Part III) should be evaluated against the _local_ floor relevant to the
current recursion context, not the global ambient floor. The global floor is
the entry condition; the local floor is the ongoing operating parameter.

### Consequences for the operating domain

These three axioms together establish:

1. **Classical test manifolds are inside the operating domain only in their
   thickened form.** A "Swiss roll" or "circle" in nature is a
   Gaussian-thickened tube embedded in tissue. The idealized zero-thickness,
   zero-background manifold is a degenerate limit _outside_ the operating
   domain. Our faded-density fixtures are the proper test cases — not relaxed
   versions of some purer ideal, but the only physically realizable version.

2. **The density field is the primitive observable object.** All downstream
   analysis (scale search, clustering, recursion) operates on \(\rho\) or its
   smoothed versions \(\rho\_\tau\). This does **not** discard the aim of
   recovering an underlying lower-dimensional fuzzy / noisy manifold. Rather,
   that latent manifold is expressed through the geometry of the density field,
   and point samples are finite-sample approximations to that field.

3. **Tissue is always present.** There is no regime where \(u = 0\). Any
   detection or separation theory must account for the tissue floor as a
   first-class parameter, not an edge case.

4. **Scale-space is well-defined.** With \(\rho\) a proper density on
   \(\mathbb{R}^D\), the Gaussian scale-space family
   \(\{\rho*\tau\}*{\tau > 0}\) inherits all Lindeberg axioms (non-enhancement
   of local extrema, semigroup, etc.) and the resolution theory of Parts
   III--V applies without pathology.

---

## Part I: What Is a Cluster

The cluster concept is the central object of Stage 1. This section therefore
starts from the continuous latent distribution and only then introduces the
scaffold graph as a finite approximation layered on top of it. The goal is to
identify the lower-dimensional fuzzy / noisy structure encoded by the latent
density, not to treat the graph itself as primary.

### Continuous latent object and discrete surrogate

Let \(\rho\) denote the latent continuous distribution from which the observed
sample cloud \(X = \{x_1,\ldots,x_n\}\) is drawn, and let
\(\rho_v = \rho \* \phi_v\) denote its smoothed version at variance scale
\(v > 0\). In the ideal continuous picture, a cluster at scale \(v\) is a
connected region of \(\rho_v\) whose interior is more coherent than its
boundary, whose extent is not sub-resolution at scale \(v\), and whose
neighbors are separated from it by a genuine separating saddle: a local minimum
or low-coupling region relative to the adjacent coherent regions, not
necessarily a drop all the way to the tissue floor.

Stage 1 cannot access \(\rho_v\) directly. Instead, it constructs discrete
surrogates from the samples. The scaffold graph
\(G_v = (N, E_v)\) with edge weights \(W_v\) is one such surrogate: it is
intended to approximate the local geometry, co-activation structure, and
boundary relations of the latent continuous distribution. The formal cluster
definition used by the algorithm is therefore the graph-level discretization of
that continuous notion.

### Operational definition (cluster at scale \(v\))

On the scaffold graph \(G_v\), a **cluster at scale \(v\)** is a non-empty
subset \(S \subseteq N\) satisfying three conditions simultaneously:

**(C1) Coherence.** The mean internal edge evidence exceeds the mean boundary
edge evidence:

\[
Q(S;v) > 0,
\qquad\text{where}\quad
Q(S;v) = \log\frac{\mathrm{LocalIntra}(S;v)+\varepsilon}
{\mathrm{BoundaryInter}(S;v)+\varepsilon}.
\]

**(C2) Extent.** The cluster's empirical variance exceeds the operating scale:

\[
\hat v*S \ge v,
\qquad\text{where}\quad
\hat v_S = \frac{1}{d*{\mathrm{eff}}|S|}
\sum\_{i\in S}\lVert x_i - \mu_S\rVert^2.
\]

**(C3) Maximality.** No adjacent candidate should be absorbed. For every
neighboring cluster \(T\) sharing boundary edges with \(S\):

\[
\mathrm{InterLocal}(S,T;v)
<
\min\!\bigl(\mathrm{LocalIntra}(S;v),\,\mathrm{LocalIntra}(T;v)\bigr).
\]

A **partition at scale \(v\)** is a collection \(\mathcal{P} = \{S_1,\ldots,S_k\}\)
of clusters covering all scaffold nodes, where each \(S_j\) satisfies
(C1)--(C3) and the partition-level quality
\(Q(\mathcal{P};v) = \sum_j \frac{|S_j|}{n} Q(S_j;v) > 0\).

### Derivation from the axioms

Each condition is a necessary consequence of operating in the domain defined by
Axioms 1--3:

**Coherence (C1) from Axiom 2.** Since \(\rho\) is a proper density (A2), the
statement "points in \(S\) are drawn from the same local structure" means the
density restricted to \(S\) is locally elevated relative to the boundary. On the
scaffold graph, this manifests as internal edge weights (which reflect
co-activation under the local density) exceeding boundary edge weights. If
\(Q(S;v) \le 0\), the density within \(S\) at scale \(v\) is no more coherent
than the density at its boundary — the set does not correspond to a density
feature and should not be called a cluster.

**Extent (C2) from Axiom 1.** Positive noise (\(\sigma*{\mathrm{noise}} > 0\))
means any observation has intrinsic spread. At operating scale \(v\), the
minimum resolvable spatial extent is \(\ell(v) = \sqrt{d*{\mathrm{eff}} v}\). A
candidate cluster with \(\hat v_S < v\) has support smaller than the resolution
limit — it is a sub-resolution fluctuation (likely a noise artifact or a
single-node scatter) and is correctly suppressed. This is the scale-space
analog of the Rayleigh criterion: structure below the resolution limit is not
structure at all at that scale.

**Maximality (C3) from Axiom 3.** The tissue decomposition (A3) implies that
between any two genuine features \(f_i\) and \(f_j\), the density must decline
relative to the adjacent coherent regions. In the smoothed density \(\rho_v\),
this appears as a separating saddle or valley (for example, a local minimum
along the inter-center axis). That saddle need not fall all the way to the
tissue floor; it only needs to reduce the between-feature coupling enough that
the two regions remain distinct at scale \(v\). On the graph, this corresponds
to boundary edges being weaker than internal edges on both sides. If
\(\mathrm{InterLocal}(S,T;v)\) is as strong as either cluster's internal
coherence, there is no genuine separating saddle — the density between them is
still part of a single unimodal or single-coherent mass at scale \(v\), and
the split is spurious.

### Taxonomy of cluster types

The definition above is general enough to cover all structures encountered in
practice. The taxonomy distinguishes clusters by the geometric form of their
density excess:

**Bump clusters.** Localized excess modes: \(f_i(x)\) is approximately Gaussian
with finite support. The classic case. Detected by peak contrast above the
tissue floor (Part III, Section 10). Example: individual Gaussians in a mixture
model.

**Manifold clusters.** Extended low-dimensional structures thickened by noise
into tubes, sheets, or shells in ambient space. There is no central density
peak — the density is approximately uniform _along_ the manifold. Detection
relies on _transverse_ contrast: the tube cross-section is a mode in the normal
directions. At the characteristic transverse scale, the tube's cross-sectional
density is elevated above tissue, satisfying (C1). Example: a noisy circle
(tube with annular cross-section) is a single cluster because its internal
transverse coherence is high and there is no internal separating saddle that
would justify splitting it into multiple clusters.

**Hierarchical clusters.** At coarse \(v\), a group of fine-scale bumps that
are close relative to \(\ell(v)\) appears as a single broader mode — the
smoothed density \(\rho_v\) is unimodal over the group. The group satisfies
(C1)--(C3) as one cluster. At a finer \(v'\), each sub-bump independently
satisfies the definition. This is not a contradiction — it is the fundamental
scale-relativity of the cluster concept: "is it a cluster?" is always relative
to a resolution \(v\).

**Tissue (non-cluster).** The background floor \(u(x)\). Within a region of
pure tissue, \(Q \approx 0\) because internal and boundary edge weights are
drawn from the same homogeneous density — there is no excess. Tissue does not
satisfy (C1) at any relevant scale.

### Exhaustiveness

**Claim.** Any connected density structure in the operating domain (A1--A3)
falls into exactly one of: (a) a cluster at some scale \(v\), (b) a proper
sub-cluster of a coarser-scale cluster, or (c) tissue. There is no fourth
category.

**Argument.** Consider any connected subset \(S\) of the scaffold. Either:

1. There exists a scale \(v\) at which \(S\) satisfies (C1)--(C3) → it is a
   cluster at scale \(v\).
2. \(S\) satisfies (C1)--(C2) at some \(v\) but fails (C3) — it should be
   merged with an adjacent set. The merged set is a cluster; \(S\) is a
   sub-cluster at a finer scale.
3. \(S\) fails (C1) at all scales — the density within \(S\) never exceeds its
   boundary density. This is tissue.

Case (2) recurses: the merged set is itself either a cluster or should be
further merged. Since the scaffold is finite, this terminates. The scale
parameter \(v\) is what disambiguates: "is it a cluster?" is not a binary
question but a function of resolution.

Every scale-space feature has a **persistence interval**: the range of scales
over which it satisfies the cluster definition. There are two fundamentally
different cases:

**Intrinsic features** (features of the true unsmoothed distribution — actual
modes, ridges, or manifold components of \(\rho\) itself): these have a
**coarse-end death scale** \(v*{\mathrm{death}}^+\) above which smoothing
destroys their contrast or merges them into a neighbor. By the non-enhancement
property of Gaussian scale-space, they persist monotonically toward finer
resolution — smoothing only destroys structure, never creates it. Their
persistence interval is \((0,\, v*{\mathrm{death}}^+]\).

**Composite features** (scale-space artifacts that exist only at intermediate
scales — for example, a single broad mode at coarse \(v\) that is actually
three well-separated bumps in the true distribution): these have _both_ a
coarse-end death scale \(v*{\mathrm{death}}^+\) and a **fine-end death scale**
\(v*{\mathrm{death}}^-\) below which they resolve into their constituent
sub-features. The composite mode ceases to satisfy (C1) or (C3) at the fine
end because its internal structure becomes resolvable — it splits into children.
Their persistence interval is
\([v_{\mathrm{death}}^-,\, v_{\mathrm{death}}^+]\).

The non-enhancement axiom is not violated: no new extremum is created by
smoothing. Rather, at coarse scales multiple extrema are smoothed _into_ a
single extremum (the composite). As smoothing decreases, that single extremum
resolves back into its constituents. The composite was never a feature of
\(\rho\) — it was a feature of \(\rho_v\) for a specific range of \(v\).

This distinction is central to the hierarchical cluster taxonomy: hierarchical
clusters (Part I) are precisely the composite features. They are valid clusters
at their coarse scale, and their fine-end death is the recursion trigger — the
point at which the system should descend into their sub-structure.

### Characteristic scales of a feature

Each feature \(F_k\) carries not one but several related characteristic scales.
Collecting and relating them clarifies what signals the system can exploit:

| Scale                            | Symbol                      | Definition                                                                                                    | Role                                                    |
| -------------------------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| Intrinsic width                  | \(\sigma_k^2\)              | Variance of the unsmoothed feature                                                                            | Natural size of the mode                                |
| Peak response                    | \(v_k^\*\)                  | Scale maximizing contrast-to-cost: \(\arg\max_v\, C_k(v) / g(v)\) where \(g\) is a cost or resolution penalty | Optimal operating point for discovery                   |
| Coarse-end death                 | \(v\_{\mathrm{death},k}^+\) | Largest \(v\) at which the feature still satisfies (C1)--(C3)                                                 | Hard ceiling — above this, the feature is unrecoverable |
| Fine-end death (composites only) | \(v\_{\mathrm{death},k}^-\) | Smallest \(v\) at which the feature still appears as a single coherent entity before resolving into children  | Recursion trigger                                       |
| Separation scale                 | \(\tau\_{\mathrm{sep},kj}\) | Scale at which bimodality with neighbor \(j\) first appears: \(\Delta\_{kj}^2/4 - \sigma^2\)                  | When a pair becomes distinguishable                     |

**Relations between characteristic scales.** For a well-posed feature,
traversing scale-space from coarse (large \(v\)) to fine (small \(v\)):

\[
v_{\mathrm{death},k}^+ > v_k^* \gtrsim \sigma_k^2 > v_{\mathrm{death},k}^-.
\]

Reading left to right in the direction of descent:

1. \(v_{\mathrm{death},k}^+\) — the feature first appears (barely satisfies
   the definition; minimal contrast).
2. \(v_k^*\) — peak response; optimal contrast-to-cost. Typically near
   \(\sigma_k^2\) because the feature responds most strongly when the smoothing
   scale matches its own width.
3. \(\sigma_k^2\) — intrinsic width of the unsmoothed feature.
4. \(v_{\mathrm{death},k}^-\) (composites only) — the feature dissolves into
   children as internal sub-structure becomes resolvable.

For intrinsic features, \(v_{\mathrm{death},k}^- = 0\) (they persist to
arbitrarily fine resolution), so the ordering simplifies to:

\[
v_{\mathrm{death},k}^+ > v_k^* \gtrsim \sigma_k^2 > 0.
\]

**All of these scales are potentially useful signals.** The coarse-end death is
the *first opportunity* — the earliest point during descent at which the
feature becomes detectable. The peak response is the optimal discovery point.
The fine-end death is the recursion trigger. And the separation scale tells the
system when a pair of features — previously merged — first becomes
distinguishable as two.

#### Parent death and child birth: the handoff relation

For a composite feature \(F_{\mathrm{parent}}\) that resolves into children
\(\{F_{\mathrm{child},1}, \ldots, F_{\mathrm{child},c}\}\), the following
relation holds between their characteristic scales:

\[
v_{\mathrm{death},\mathrm{parent}}^-
\;\approx\;
\max_j\, v_{\mathrm{death},\mathrm{child}_j}^+.
\]

That is: the parent's fine-end death scale coincides (approximately) with the
coarsest child's coarse-end death scale. The parent dissolves at the same scale
at which its most prominent child first becomes independently detectable. This
is not a coincidence — it is the same event viewed from two perspectives:

- From the parent's perspective: internal coherence (C1) degrades because an
  internal separating saddle has emerged. The parent is dying.
- From the child's perspective: contrast above the local floor has crossed
  \(\kappa_{\mathrm{det}}\). The child is being born.

More precisely, for equal-width children with common \(\sigma_c\) separated by
\(\Delta\), the parent's fine-end death occurs when the smoothed density first
becomes bimodal:

\[
v_{\mathrm{death},\mathrm{parent}}^-
=
\tau_{\mathrm{sep}}
=
\frac{\Delta^2}{4} - \sigma_c^2.
\]

And each child's coarse-end death is governed by detectability above the local
floor within the parent's interior:

\[
v_{\mathrm{death},\mathrm{child}}^+
=
\sigma_c^2
\left[
\left(\frac{H_c}{\kappa_{\mathrm{det}}\,u_{\mathrm{local}}}\right)^{2/d}
- 1
\right].
\]

When these two expressions are close (which they are for well-separated,
high-contrast children), the handoff is clean: the parent dies and the children
are immediately discoverable. When there is a gap
(\(v_{\mathrm{death},\mathrm{parent}}^- > v_{\mathrm{death},\mathrm{child}}^+\)),
there is a brief interval of ambiguity where the parent is already incoherent
but the children are not yet individually detectable — this is the regime where
the clustering pipeline must be most careful.

**Operational consequence:** if the system detects the parent beginning to die
(declining Q, emerging internal heterogeneity), it can anticipate that children
should become detectable within approximately one grid step finer. This makes
parent-death monitoring a *predictive* signal for recursion timing.

### The scale-space discovery objective

Given a latent distribution \(\rho\) with scale-space feature set
\(\mathcal{F} = \{F_1, \ldots, F_M\}\) — comprising all intrinsic features
plus all composite features induced by Gaussian convolution — order features
by characteristic scale:

\[
v_1^_ \ge v_2^_ \ge \cdots \ge v_M^\*.
\]

The Stage 1 objective, stated in scale-space terms, is:

> **Discover features from coarsest to finest, operating as close to each
> feature's characteristic scale as possible without overshooting it.**

More precisely:

1. **Coarse-to-fine ordering.** Features should be identified in decreasing
   order of characteristic scale. The coarsest composite features are
   discovered first; their children become discoverable only after the parent
   has been established and the recursion has narrowed to its interior.

2. **Minimal overshoot.** For each feature \(F*k\), the operating scale
   \(v*{\mathrm{op}}\) at which it is identified should satisfy
   \[
   v*k^\* \le v*{\mathrm{op}} \le v*{\mathrm{death},k}^+,
   \]
   with the ideal being \(v*{\mathrm{op}} \approx v_k^_\). Operating too far
   above \(v_k^_\) means the feature is barely detectable (low contrast,
   fragile to noise). Operating below \(v_k^\*\) is not destructive — the
   feature is still present — but is wasteful: it means the system passed
   through the feature's optimal scale without recognizing it and is now
   resolving finer structure prematurely.

3. **Recursion as descent (birth and death as dual signals).** The system
   can detect the need to recurse via two complementary signals:
   - **Birth signal (child emergence):** at scale \(v\), a new feature's
     contrast crosses the detectability threshold — it is "born" into
     visibility. This is the traditional signal: something new appears as we
     refine the scale.

   - **Death signal (parent dissolution):** the composite parent feature
     begins to fail (C1) or (C3) — its internal coherence degrades because
     sub-structure is becoming resolvable. The parent is "dying" at its
     fine-end death scale.

   These are dual views of the same event (composite resolving into children),
   but detecting death may be _easier_ in practice because the system already
   knows the parent's characteristics: its Q-score, its variance, its boundary
   structure. Monitoring the parent for degradation (declining Q, emerging
   internal saddles, rising internal heterogeneity) is a known-signature
   detection problem. By contrast, birth detection requires recognizing
   something new without prior knowledge of what it will look like.

   **Operational implication:** the scale search should monitor _both_ signals.
   A declining parent Q-score or a parent whose internal variance begins to
   significantly exceed \(v\) is a leading indicator that recursion is
   warranted — potentially before any child has cleanly crossed the birth
   threshold on its own.

4. **Completeness.** The process terminates when every feature in
   \(\mathcal{F}\) that lies within the admissible window (detectable above its
   local tissue floor, separable from its neighbors) has been identified at
   some recursion level.

This objective defines what "success" means for Stage 1 independently of any
particular algorithmic strategy. The Q-gated scaffold machinery (Part II), the
resolution theory (Part III), and the scale-response search (Part IV) are all
in service of achieving this objective efficiently and robustly.

### The scaffold graph across scales

The scaffold graph is not a single fixed object — it is a _family_ of graphs
parameterized by scale. Understanding how the graph's structure changes as a
function of the operating scale \(v\) is essential for interpreting the
techniques used to identify clusters.

**The maximally precise limit.** At the finest conceivable resolution, the
scaffold tends toward a **sample graph** whose vertices are the observed data
points themselves. In that limit, the geometric term remains well-defined:
pairwise affinities still reflect raw distances and kernel overlap. But the
usual Proteus notion of **co-activation** changes qualitatively. In a true
scaffold, each node aggregates many nearby observations, so node masses
\(h_i\) and transition counts \(C(i \to j)\) are stable summary statistics of
the latent distribution. As the number of nodes approaches the number of
samples, that aggregation disappears: each node is visited only once or a few
times, \(h_i\) becomes small, and \(C(i \to j)\) becomes sparse and
high-variance.

So the maximally fine limit is **not** simply the same scaffold with more
nodes. It is a regime change:

- In the **compressed scaffold regime**, \(W_v = K_v A_{\mathrm{sym}}\) is the
  natural edge law because both geometry and aggregated co-activation are
  meaningful.
- In the **sample-graph limit**, the geometric affinity \(K_v\) remains
  meaningful, but \(A_{\mathrm{sym}}\) largely degenerates for a static i.i.d.
  point cloud. The object is no longer best understood as a scaffold; it is a
  direct discretization of the observed sample cloud.

There is one important exception: if the data carry genuine sequential or
trajectory structure, then even one-node-per-sample graphs can still support
nontrivial transition statistics from temporal adjacency. But that is a
different source of pairwise evidence than the repeated-region co-activation
used by the compressed scaffold.

This sample-graph limit is still the \(v \to 0\) information-theoretic ceiling
— no structure has been smoothed away. In practice, however, it is
computationally intractable, dominated by noise, and often too uncompressed for
stable co-activation estimation. The practical scaffold lives away from this
limit, in the regime where nodes still summarize local neighborhoods of the
latent distribution.

**Coarse scales: merged islands.** At large \(v\), the kernel \(K_v(i,j)\)
decays slowly with distance, so even well-separated nodes contribute
non-trivial edge weight. The graph becomes densely coupled, all scaffold nodes
appear to belong to a single connected component, and the Q-score for any
multi-cluster partition trends toward zero or negative. This is correct
behavior: at sufficiently coarse resolution, everything looks like one mass.
No structure is resolvable.

At the extreme coarse limit, the scaffold approaches a **single-node graph**:
all data are compressed into one catchment. In this limit, node mass remains
well-defined — the sole node carries essentially the total mass of the dataset
— but pairwise geometry disappears because there are no off-diagonal node pairs
on which to evaluate \(K_v(i,j)\). Co-activation no longer degenerates by
sparsity (as in the sample-graph limit); instead it becomes trivial or
saturated because all transitions collapse into self-mass on the single node.
So the single-node limit is the dual extreme of the sample-graph limit:

- In the **single-node limit**, aggregated mass is maximal but geometry and
  boundary structure vanish.
- In the **sample-graph limit**, geometry remains but aggregated co-activation
  loses statistical meaning.

The practical scaffold must live between these two extremes, where both
geometry and repeated occupancy statistics remain informative.

**Intermediate scales: disconnected islands and emerging modes.** As \(v\)
decreases from the coarse limit, the kernel contracts and weakly-coupled
regions begin to decouple. Characteristic phenomena at intermediate scales
include:

- **Disconnected islands.** Lifted-edge components fragment as the kernel
  radius drops below the inter-feature gap. Two modes that appeared merged at
  the previous grid step may suddenly appear as disjoint components — this is a
  topological signal that a separating saddle has become resolvable.

- **Sub-threshold modes.** A feature with contrast \(C*i(v) < \kappa*{\mathrm{det}}\)
  at the current scale is indistinguishable from the local tissue floor. One
  step finer on the scale grid and \(C*i(v') \ge \kappa*{\mathrm{det}}\) — the
  mode _emerges_ as a detectable structure. This discrete emergence is not a
  defect; it is the finite-grid analog of the continuous birth scale.

- **Scale-grid granularity.** With the Lindeberg-compatible grid ratio
  \(r = 1/\sqrt{2}\), adjacent scales differ by a factor of 2 in variance
  magnitude. This remains true even when tissue is present — it is purely a
  statement about the Gaussian scale grid. However, the resulting change in
  detectability or separability is not uniform, because those depend
  nonlinearly on the local tissue floor \(u_{\mathrm{local}}\) as well as on
  the feature's intrinsic width and amplitude. A mode whose birth scale falls
  between two grid points will therefore often appear to "pop in" at the next
  finer step, and the sharpness of that transition may vary substantially from
  region to region. The grid is coarse enough that these transitions are often
  abrupt rather than gradual.

#### Canonical grid vs adaptive noise-aware search

This motivates an important distinction:

- The **canonical scale grid** should remain Gaussian and globally comparable.
  It is the reference lattice on which scale-space is defined, and preserving
  it retains the semigroup interpretation of smoothing and the ability to
  compare parent and child scales directly.
- The **search policy** on that grid should be adaptive. As the system updates
  its beliefs about the local tissue floor \(u_{\mathrm{local}}\), the likely
  operating window for each feature shifts, and the controller should adapt
  which grid points it probes, skips, refines, or revisits.

In particular, updating \(u_{\mathrm{local}}\) changes the detectability window
\(\tau_{\mathrm{det}}\):

\[
\tau_{\mathrm{det}}
=
\sigma^2
\left[
\left(\frac{H}{\kappa_{\mathrm{det}}\,u_{\mathrm{local}}}\right)^{2/d}
- 1
\right].
\]

If \(u_{\mathrm{local}}\) rises, the admissible window contracts and shifts
toward finer scales. If \(u_{\mathrm{local}}\) falls, coarser scales remain
admissible longer. So the controller should not treat every grid step as
equally informative. Instead, it should maintain a local posterior over the
relevant characteristic scales — \(v^*\), \(v_{\mathrm{death}}^+\),
\(v_{\mathrm{death}}^-\), and the predicted parent-child handoff — and use
that posterior to decide:

1. which scales to evaluate first,
2. where to refine more densely,
3. when to recurse early because parent death is imminent,
4. and when to terminate a local search because no admissible window remains.

So the right theoretical stance is **not** \"replace the Gaussian grid with a
warped noise-dependent grid.\" It is: **keep the canonical Gaussian grid, but
perform a locally adaptive, noise-aware traversal on that grid.** The grid
defines the coordinate system; the updated noise-floor beliefs define the
search strategy.

#### Estimating a heterogeneous local noise floor

The preceding sections treat the local floor \(u_{\mathrm{local}}\) as a scalar
quantity available to the controller. In practice, the floor is neither scalar
nor directly observable. This subsection generalizes it to a heterogeneous,
potentially anisotropic quantity and catalogs concrete approaches for estimating
it.

**Why the floor is not a scalar constant.** Several real-world conditions
violate scalar-isotropic-homogeneous assumptions:

- **Multi-source concatenation.** If observed dimensions come from distinct
  measurement modalities (sensors, instruments, perceptual channels) whose
  vectors are concatenated because they co-occur temporally, each modality
  block may carry a different noise profile. A single scalar floor cannot
  represent this.

- **Region-varying noise.** Even with a single measurement apparatus, some
  regions of a manifold may be intrinsically noisier than others — due to
  physical chaoticity, measurement difficulty, sample sparsity, or biological
  variability. The floor is then a function of position, not a region constant.

- **Unresolved fine structure.** Within a coarse cluster, sub-features that are
  not yet resolved at the current scale contribute a pseudo-floor that varies
  spatially and may be anisotropic (aligned with the sub-feature geometry).

- **PCA ambiguity.** Local PCA yields eigenvalues of the *total* local
  covariance \(\Sigma_{\mathrm{local}} = \Psi + S\), where \(\Psi\) is the
  noise covariance (the generalized floor) and \(S\) is signal covariance. PCA
  alone cannot separate these two additive terms unless additional structure is
  assumed.

The general model at position \(x\) is therefore:

\[
\Sigma_{\mathrm{local}}(x) = \Psi(x) + S(x),
\qquad \Psi(x) \succeq 0,\quad S(x) \succeq 0.
\]

**Hierarchy of floor models.** Ordered from simplest to most general:

| Model | Floor structure | When sufficient |
|-------|----------------|----------------|
| M0: Scalar per-region | \(\Psi = \sigma_{\mathrm{floor}}^2 I\) shared across region | Default case; isotropic, homogeneous noise within a cluster |
| M1: Scalar per-node | \(\Psi_i = \sigma_{\mathrm{floor},i}^2 I\) | Spatial heterogeneity with isotropic local floors |
| M2: Blockwise scalar per-region | \(\Psi = \bigoplus_b \sigma_{\mathrm{floor},b}^2 I_b\) | Multi-modal or block-heterogeneous data when block hints are provided |
| M3: Diagonal per-node | \(\Psi_i = \mathrm{diag}(\psi_{i,1}, \ldots, \psi_{i,d})\) | Theoretical completeness; not implemented unless future data demands it |
| M4: Full covariance per-region | \(\Psi\) is a general PSD matrix | Theoretical completeness; correlated noise with strong prior structure |

The practical rule should be simple:

- **Default:** use M0. This is always available and requires no extra
  configuration.
- **If the user provides modality / block hints:** upgrade to M2 by tracking one
  scalar floor per declared block.
- **Do not infer block boundaries from the data.** That is too fragile and too
  expensive to justify in the current system.
- **M3 and M4 remain theoretical escalation paths** for future work, but should
  not be treated as part of the pragmatic baseline.

This means the only supported escalation path beyond M0 is **block-hint-driven,
not inferred**. If the user knows that the data vector is a concatenation of
distinct measurement modalities, they declare the block structure once, and the
system trivially extends scalar-floor tracking from one global scalar to one
scalar per block. If no hint is given, the system assumes one homogeneous block
and uses a single scalar floor.

**Concrete estimation approaches.** Each method below states what data it uses,
what assumptions it requires, and which floor models it can identify.

**(A) Trailing-spectrum method (PPCA-like).**

- *Data:* local eigenvalue spectrum from PCA.
- *Assumption:* signal is effectively low-rank; floor is isotropic.
- *Identifies:* M0 or M1.
- *Method:* \(\hat\sigma_{\mathrm{floor}}^2 = \mathrm{mean}(\lambda_{d_{\mathrm{int}}+1}, \ldots, \lambda_D)\),
  i.e., the mean of trailing eigenvalues beyond the estimated intrinsic
  dimension.
- *Limitation:* fails when signal occupies all retained dimensions (no spectral
  shelf). In that regime, it gives only an upper bound.

**(B) Cross-region consensus.**

- *Data:* local covariance estimates from multiple neighboring scaffold nodes or
  sibling clusters.
- *Assumption:* the floor is shared (or slowly varying) across neighbors; signal
  structure varies.
- *Identifies:* M0--M2.
- *Method:* robust lower envelope or intersection of local spectra across
  neighbors. The covariance component common to all neighbors — the part that
  does not change when moving from one local region to another — is a candidate
  floor. Operationally: take per-dimension minima or robust lower quantiles of
  local eigenvalues across a neighborhood set.
- *Strength:* does not require the floor to be isotropic or the signal to be
  low-rank. Works whenever neighboring regions share a common noise substrate
  but differ in their signal content.

**(C) Scale-stability criterion.**

- *Data:* local variance estimates at multiple operating scales \(v\).
- *Assumption:* the true floor is already at its own resolution (it does not
  respond to further smoothing); unresolved structure changes with scale.
- *Identifies:* M0--M1.
- *Method:* track local variance as \(v\) changes. The persistent baseline that
  does not respond to smoothing — the component stable across the scale grid —
  is a candidate floor. Components that grow or shrink with \(v\) are likely
  signal or unresolved sub-structure.
- *Limitation:* requires the floor and signal to respond differently to scale
  changes. If both are at similar intrinsic scales, this method has low
  discrimination power.

**(D) Parent-child residual.**

- *Data:* parent cluster covariance vs child cluster covariances after
  recursion.
- *Assumption:* children inherit the parent's floor plus their own signal.
- *Identifies:* M0--M3.
- *Method:* after recursion splits a parent into children \(\{C_j\}\), estimate
  \[
  \hat\Psi \approx \Sigma_{\mathrm{parent}}
  - \sum_j \frac{|C_j|}{n} S_j,
  \]
  where \(S_j\) is the child-specific signal covariance (Steiner-corrected to
  remove the child's own centroid offset from the parent mean).
- *Strength:* naturally available during recursion. Uses the hierarchy itself as
  a signal/floor separator — the parent's total covariance minus the children's
  explained variance yields the unexplained (floor) component.

**(E) Off-subspace residual.**

- *Data:* variance in dimensions discarded by PCA (the "noise subspace").
- *Assumption:* discarded dimensions carry only noise.
- *Identifies:* lower bound on the ambient/global floor.
- *Limitation:* says nothing about the floor *within* the retained subspace. A
  manifold's tangential noise is not constrained by its normal-space noise.
  Useful as a sanity check or initialization, not as a complete floor estimate.

**(F) Multi-source block structure.**

- *Data:* known block structure of dimensions (e.g., dimensions 1--50 are
  modality A, 51--100 are modality B).
- *Assumption:* noise is block-diagonal with known block boundaries.
- *Identifies:* M2 directly; richer within-block models are future work.
- *Method:* the user supplies a dimension-block declaration. The system then
  estimates one scalar floor per block (for example, per-block trailing
  spectra) and combines them into a block-diagonal \(\hat\Psi\).
- *Strength:* directly handles multi-source concatenation. When the user knows
  which dimensions belong to which measurement system, the per-block noise
  profiles can be estimated independently and do not contaminate each other.

**Confidence hierarchy for floor estimates.**

| Confidence | Conditions |
|------------|------------|
| High | Trailing spectrum is flat AND cross-region consensus agrees (A+B confirm each other) |
| Moderate | One method gives a clear estimate; others unavailable or weakly confirmatory |
| Low | Single local PCA with full-rank mixed signal; no cross-region or multi-scale data |
| Unidentifiable | Anisotropic floor with no block structure, no neighbor comparison, and no recursion history |

**Theoretical stance.** The local noise floor is a latent quantity. It cannot in
general be read off from a single PCA decomposition. It must be *inferred* from
multiple complementary signals: spectral structure, cross-region consistency,
scale-space stability, and parent-child residuals. The system should maintain a
belief over the floor model (M0--M4) and update it as evidence accumulates
during the scaffold's lifetime.

This generalizes the scalar \(u_{\mathrm{local}}\) introduced in Axiom 3
("Global and local noise floors") to a matrix-valued latent
\(\Psi(x)\). The detectability and separability conditions (Part III) remain
structurally the same but should be interpreted with the *effective local floor*
— whether scalar or directional — relevant to the current recursion context.
The adaptive search policy ("Canonical grid vs adaptive noise-aware search")
uses this evolving floor belief as its primary input. And the parent-child
handoff relation ("Parent death and child birth") provides a natural estimation
opportunity: each recursion event is simultaneously a clustering decision and a
floor-refinement event.

#### Settled catchments as local probes of the noise tensor

The same local asymptotic regime that justifies treating a mature Voronoi
catchment as approximately affine also suggests a corresponding limiting model
for noise. As a catchment settles and its diameter shrinks, both the geometry
of the signal and the variation of the floor should become locally simpler.

Write a local observation near \(x_0\) as

\[
X = x_0 + U z + \tfrac12 \mathrm{II}(z,z) + \varepsilon,
\]

where \(Uz\) is the local affine / tangent contribution, \(\mathrm{II}(z,z)\)
is the curvature correction, and \(\varepsilon\) is a local noise term with
covariance \(\Psi(x_0)\). If a mature catchment has effective radius \(h\), then
its local covariance takes the form

\[
\Sigma_{\mathrm{catchment}}
\approx
\Psi(x_0)
+
U \Sigma_z(h) U^\top
+
\Sigma_{\mathrm{curv}}(h)
+
O(h \lVert \nabla \Psi \rVert).
\]

The key scaling picture is:

- tangential signal variance scales like \(O(h^2)\),
- curvature-induced normal variance scales like \(O(h^4)\),
- spatial variation of the noise field scales like \(O(h)\),
- and the true local floor \(\Psi(x_0)\) remains as the intercept.

So in the small-catchment limit,

\[
\Sigma_{\mathrm{catchment}} \to \Psi(x_0),
\]

provided the manifold is locally smooth enough, the floor varies slowly enough
across the catchment, and there are enough samples to estimate covariance
stably. This is the noise analogue of the affine-subspace limit.

This observation yields a principled separation:

- **Geometry** is the part of local covariance that shrinks predictably as the
  catchment contracts.
- **True floor** is the part that stabilizes as an intercept under continued
  settlement.
- **Unresolved sub-structure** is the part that temporarily behaves like floor
  at one scale but continues to change, peel away, or resolve into children
  under further refinement.

In other words, a mature node is not just a local probe of the signal geometry;
it is also a local probe of the noise tensor.

**Operationally on existing primitives:**

- **Node:** the atomic noise sensor. Each settled node should track its local
  residual covariance, trailing-spectrum floor estimate, anisotropy ratio, and
  temporal / scale stability as the catchment contracts.

- **1-hop ball:** the basic certify-or-refine patch. Neighboring mature nodes
  are pooled to decide whether their limiting residuals support a shared floor
  model (scalar, diagonal, blockwise) or whether the patch should be split or
  modeled more richly.

- **Simplicial cell:** the transport and interpolation primitive. Once Stage 2
  constructs local cells, node- and ball-level floor estimates can be
  transported into a common chart, interpolated across a cell, and checked for
  smoothness or discontinuity. This turns the collection of local floor
  estimates into a genuine noise atlas.

- **Region / recursion context:** the summary layer. Ball- and cell-level
  estimates are aggregated to produce the effective floor model that drives
  detectability thresholds, parent-death monitoring, child-birth prediction,
  and adaptive scale search.

This leads to a natural certify-or-refine criterion:

> accept a local floor model only if the residual covariance becomes stable as
> the catchment settles; otherwise continue refining the patch, escalate the
> floor model, or treat the residual as unresolved structure rather than true
> floor.

The practical consequence is important. The noise atlas should not require a
separate spatial decomposition. It should live on top of the same primitives
already used to measure scaffold settlement:

- nodes **estimate** local limiting covariances,
- 1-hop balls **certify** shared local floor models,
- simplices **transport and interpolate** those models,
- and regions **summarize** them for the controller.

This makes the settlement machinery and the noise-atlas machinery two views of
the same local asymptotic regime: as the mesh settles, the signal becomes
locally affine and the floor becomes locally constant (or at least locally
simple in an appropriate chart).

#### State inventory: what is tracked and at what cost

The theory above can sound more complex than the practical state required to
support it. In reality, most of the useful quantities are either already
tracked by the scaffold or can be derived cheaply from existing moments.

**Per-node state (already tracked):**

| Quantity | Role | Marginal update cost | Value |
|----------|------|----------------------|-------|
| `position` | Voronoi centroid / local geometry | \(O(d)\) per routed sample | Essential |
| `residual_mean`, `residual_sq` | First and second residual moments | \(O(d)\) per routed sample | Essential |
| `variance` | Scalar summary \(= \mathrm{tr}(s - m \odot m)\) | Derived from existing moments | Essential |
| `nudge` | Deferred position correction | \(O(d)\) per routed sample | Essential |
| `principal_dir` | Streaming principal direction / split axis | \(O(d)\) per routed sample | Moderate |
| `hit_count`, `update_count` | Mass and maturity bookkeeping | \(O(1)\) | Essential |
| `d_final` | Smoothed intrinsic-dimension diagnostic | Cheap diagnostic update | Moderate |
| `m_pos`, `s_pos`, `h_pos`, `m_neg`, `s_neg`, `h_neg` | Partition-aligned shadow moments for candidate splits | \(O(d)\) on BMU events | Essential for recursion |

**Per-node derived quantities (cheap to compute, not necessarily stored):**

| Quantity | Derived from | Cost | Value |
|----------|--------------|------|-------|
| `psi_i` (local scalar floor estimate) | trailing or minimum component of `residual_sq - residual_mean^2` | \(O(d)\) on demand | High |
| `alpha_i` (anisotropy ratio) | max / min residual variance component | \(O(d)\) on demand | Moderate |
| confidence weight | `hit_count`, maturity | \(O(1)\) on demand | Low |

**Per-link state (already tracked):**

| Quantity | Role | Marginal update cost | Value |
|----------|------|----------------------|-------|
| `count_ij`, `count_ji` | Directed transition mass | \(O(1)\) per co-activation | Essential |
| `protected_until` | Neonatal guard | \(O(1)\) | Essential |
| `lifted` | Structural vs shadow status | \(O(1)\) | Essential |

**Per-region derived quantities (computed when needed):**

| Quantity | Source | Typical cost | Value |
|----------|--------|--------------|-------|
| `u_local` / effective floor | median or robust summary of node-level `psi_i` values | \(O(n_{\mathrm{nodes}})\) per region evaluation | High |
| floor range / heterogeneity | spread of node-level `psi_i` values | \(O(n_{\mathrm{nodes}})\) | Moderate |
| parent-child residual floor refinement | parent covariance minus child explained covariance | \(O(n_{\mathrm{nodes}} d)\) at recursion events | High |

**Value for cost: practical tiers.**

- **Tier 1: implement now.** Use only quantities that are already tracked or
  nearly free to derive:
  - node-level `psi_i`,
  - region-level `u_local` as a robust summary of `psi_i`,
  - parent-child residual refinement at recursion time.

- **Tier 2: compute on demand.** Useful diagnostics that do not need to live in
  core per-sample state:
  - anisotropy ratio `alpha_i`,
  - 1-hop ball floor consensus,
  - floor heterogeneity summaries.

- **Tier 3: defer.** Richer atlas machinery whose theoretical value exceeds its
  current implementation value:
  - scale-stability tracking as persistent state,
  - simplicial-cell interpolation of the floor field,
  - M3/M4 model escalation.

This makes the minimal viable noise atlas intentionally small:

> **three derived quantities from existing state, with zero new per-sample
> computational path:** node-level `psi_i`, region-level `u_local`, and
> parent-child residual refinement.

**Fine scales: full resolution.** At scales near or below the intrinsic width
\(\sigma_i^2\) of each feature, the scaffold resolves all intended structure.
Edge weights decay sharply beyond \(\ell(v)\), so only genuinely proximate
nodes remain coupled. Each mode satisfies (C1)--(C3) independently. This is
the regime where the scaffold most faithfully represents the geometry of the
latent continuous distribution.

**Implications for cluster identification.** The techniques in Parts II--IV
must be understood against this backdrop:

- The Q-score and merge/split rules (Part II) evaluate a _single snapshot_ of
  the graph at one fixed \(v\). Their output is only meaningful relative to that
  scale.
- The scale-response search (Part IV, Section 14) _traverses_ this family of
  graphs, looking for the scale at which structure first becomes identifiable.
  It is precisely the search for the transition between "merged islands" and
  "emerging modes."
- Recursion re-enters at a finer scale within each coarse cluster, where the
  local scaffold may again appear as merged islands needing further resolution.

The scaffold is therefore not a static data structure but a lens whose focal
length is \(v\). Clustering at scale \(v\) is the act of reading the graph at
that focal length.

---

## Part II: The Scale-Conditioned Clustering Objective

This part defines the operational machinery that evaluates the cluster
definition (Part I) on a finite scaffold graph at a fixed scale \(v\). It
constructs the edge weights, the quality score, and the merge/split rules that
implement conditions (C1)--(C3) for one particular focal length of the
scaffold lens.

### 2. Characteristic variance scale

Let \(v > 0\) be the characteristic variance scale (units: squared distance).
With effective dimension \(d\_{\mathrm{eff}}\), define the correlation radius

\[
\ell(v) = \sqrt{d\_{\mathrm{eff}}\,v}.
\]

Interpretation:

- \(v\) is the variance resolution at the current level of analysis.
- \(\ell(v)\) is the length scale at which scaffold edges carry meaningful
  correlation evidence.
- Structure smaller than \(\ell(v)\) should not be treated as an independent
  cluster at this level — this is the operational realization of condition (C2).

### 3. Scale-conditioned edge evidence

On scaffold nodes \(i,j\) with positions \(x_i, x_j\), define the Gaussian
kernel

\[
K*v(i,j)
=
\exp\!\left(
-\frac{\lVert x_i-x_j\rVert^2}{2\,d*{\mathrm{eff}}\,v}
\right).
\]

Let \(C(i\to j)\) be the directed Hebbian transition count from \(i\) to \(j\)
and \(h_i\) the total hit mass at node \(i\). Define symmetric co-activation
evidence

\[
A\_{\mathrm{sym}}(i,j)
=
\sqrt{
\frac{C(i\to j)}{h_i+\varepsilon}\,
\frac{C(j\to i)}{h_j+\varepsilon}
}.
\]

The scale-conditioned edge weight is

\[
\boxed{
W*v(i,j) = K_v(i,j)\,A*{\mathrm{sym}}(i,j).
}
\]

This is the primitive evidence for all Stage 1 clustering decisions. The kernel
\(K*v\) supplies geometric scale-conditioning (Axiom 2 guarantees it operates
on a density field), while \(A*{\mathrm{sym}}\) supplies empirical
co-activation evidence from the scaffold's learning dynamics.

### 4. Local intra-cluster and boundary inter-cluster correlation

Let \(E_v\) denote the lifted-edge set at scale \(v\). For a candidate cluster
\(C\):

**Local intra:**

\[
\mathrm{LocalIntra}(C;v)
=
\frac{
\sum*{(i,j)\in E_v,\; i,j\in C} W_v(i,j)
}{
|\mathcal E*{\mathrm{in}}(C)| + \varepsilon
}.
\]

**Boundary inter:**

\[
\mathrm{BoundaryInter}(C;v)
=
\frac{
\sum*{(i,j)\in E_v,\; i\in C,\; j\notin C} W_v(i,j)
}{
|\mathcal E*{\mathrm{bdry}}(C)| + \varepsilon
}.
\]

**Pairwise inter-cluster coupling** (for candidate pair \(A,B\)):

\[
\mathrm{InterLocal}(A,B;v)
=
\frac{
\sum*{(i,j)\in E_v,\; i\in A,\; j\in B} W_v(i,j)
}{
|\mathcal E*{AB}| + \varepsilon
}.
\]

### 5. Q-score and partition quality

The Q-score is the operational test for condition (C1). A cluster is valid at
scale \(v\) when internal coherence exceeds boundary leakage:

\[
\boxed{
Q(C;v)
=
\log
\frac{
\mathrm{LocalIntra}(C;v)+\varepsilon
}{
\mathrm{BoundaryInter}(C;v)+\varepsilon
}.
}
\]

Interpretation:

- \(Q > 0\): internally coherent at scale \(v\) — satisfies (C1).
- \(Q = 0\): no local contrast — indistinguishable from tissue.
- \(Q < 0\): not a meaningful cluster at this scale.

For a full partition \(\mathcal P = \{C_1, \ldots, C_k\}\):

\[
Q(\mathcal P;v)
=
\sum\_{C\in\mathcal P}
\frac{|C|}{n}\,Q(C;v).
\]

Stage 1 uses partition-level \(Q\) to accept, reject, and refine candidate
partitions. Recursion proceeds only when \(Q(\mathcal P;v) > 0\) with more than
one cluster.

### 6. Scale-relative validity condition

The operational test for condition (C2). Let the empirical variance of cluster
\(C\) be

\[
\hat v*C
=
\frac{1}{d*{\mathrm{eff}}|C|}
\sum\_{i\in C}\lVert x_i-\mu_C\rVert^2.
\]

The full validity condition is:

\[
\hat v_C \ge v
\quad\text{and}\quad
Q(C;v) > 0.
\]

The first clause says the cluster's support is large enough to be _resolved_ at
the current scale (not merely an artifact of sub-resolution noise — per Axiom
1, noise > 0 mandates a minimum resolvable extent). The second says its
internal structure is more coherent than its boundary entanglement.

### 7. Merge and split rules

The operational test for condition (C3). Two adjacent clusters \(A,B\) should
**merge** when boundary coupling dominates:

\[
\mathrm{InterLocal}(A,B;v)
\ge
\min\!\bigl(
\mathrm{LocalIntra}(A;v),\,
\mathrm{LocalIntra}(B;v)
\bigr).
\]

They should remain **separate** when:

\[
\mathrm{InterLocal}(A,B;v)
<
\min\!\bigl(
\mathrm{LocalIntra}(A;v),\,
\mathrm{LocalIntra}(B;v)
\bigr).
\]

In practice, a merge is also gated by partition-\(Q\) improvement. This gives
scale meaning to clustering:

- at coarse \(v\), nearby modes may legitimately merge (the smoothed density is
  unimodal over them),
- at fine \(v\), the same modes become distinguishable (bimodality emerges).

---

## Part III: Continuous Resolution Theory

The graph-local Q-score (Part II) is the _algorithmic_ separability criterion.
This part derives the _information-theoretic_ preconditions under which the
cluster definition _can_ be satisfied — that is, under what conditions on the
density \(\rho\) (guaranteed to exist by Axiom 2) the graph will have enough
signal for (C1)--(C3) to succeed.

### 8. Density model

By Axiom 3, the local density in \(d\) effective dimensions decomposes as

\[
\rho(x) = u + \sum\_{i=1}^m H_i
\exp\!\left(-\frac{\lVert x-\mu_i\rVert^2}{2\sigma_i^2}\right),
\]

where:

- \(u > 0\): the tissue floor (strictly positive by Axiom 3 / Axiom 1),
- \(H_i\): peak excess height of mode \(i\) above tissue,
- \(\sigma_i\): intrinsic Gaussian width,
- \(\mu_i\): mode center.

Equivalently, with mass \(a_i\) per mode:

\[
H_i = \frac{a_i}{(2\pi\sigma_i^2)^{d/2}}.
\]

Proteus uses \(\tau\) as its variance scale parameter, so scale-space smoothing
corresponds to Gaussian convolution with variance \(\tau\). By Axiom 2, this
convolution is well-defined and produces the scale-space family
\(\{\rho*\tau\}*{\tau>0}\).

### 9. Scale-space evolution of a single mode

After smoothing by variance \(\tau\), a single bump becomes

\[
\rho\_\tau(x) =
u +
H_i
\left(\frac{\sigma_i^2}{\sigma_i^2+\tau}\right)^{\!d/2}
\exp\!\left(
-\frac{\lVert x-\mu_i\rVert^2}{2(\sigma_i^2+\tau)}
\right).
\]

The smoothed width and smoothed peak excess are:

\[
s_i(\tau) = \sqrt{\sigma_i^2+\tau},
\qquad
h_i(\tau) =
H_i\left(\frac{\sigma_i^2}{\sigma_i^2+\tau}\right)^{\!d/2}.
\]

Key monotonic property: \(h_i(\tau)\) decreases with \(\tau\). Increasing the
operating scale makes a feature broader and less contrastive against the tissue
floor that Axiom 3 guarantees is always present.

### 10. Detectability above tissue

In all formulas below, \(u\) denotes the **local** tissue floor relevant to the
current recursion context (see Axiom 3, "Global and local noise floors"). At
the root level this equals the global floor; within a recursion child it is the
local floor of the parent cluster's interior.

Define peak signal-to-tissue contrast at operating scale \(\tau\):

\[
C_i(\tau) :=
\frac{h_i(\tau)}{u}
=
\frac{H_i}{u}
\left(\frac{\sigma_i^2}{\sigma_i^2+\tau}\right)^{\!d/2}.
\]

**Detectability condition.** A mode satisfies the cluster definition (C1) only
if it retains sufficient contrast above the local tissue floor:

\[
\boxed{
C*i(\tau) \ge \kappa*{\mathrm{det}},
}
\]

where \(\kappa\_{\mathrm{det}} > 1\) is the minimum contrast the scaffold must
retain for the mode to survive against the local tissue floor. This gives the
background-capacity bound

\[
u \le
\frac{H*i}{\kappa*{\mathrm{det}}}
\left(\frac{\sigma_i^2}{\sigma_i^2+\tau}\right)^{\!d/2},
\]

or equivalently the maximum admissible operating scale

\[
\tau \le \tau*{\mathrm{det},i}
:=
\sigma_i^2
\left[
\left(\frac{H_i}{\kappa*{\mathrm{det}}\,u}\right)^{\!2/d}

- 1
  \right].
  \]

**Characteristic-scale detectability** (evaluating at \(\tau = \sigma_i^2\)):

\[
\frac{H*i}{u} \ge \kappa*{\mathrm{det}}\,2^{d/2}.
\]

This is the sharpest single answer to "what is the maximum tissue density a
Gaussian mode of width \(\sigma\) and height \(H\) can tolerate?" — a question
that only arises because Axiom 3 guarantees tissue is always present.

### 11. Clean resolvability of two modes

Two equal-width, equal-height isotropic Gaussians with common \(\sigma\),
separated by \(\Delta\), remain bimodal after smoothing by \(\tau\) if and only
if

\[
\boxed{
\Delta > 2\sqrt{\sigma^2+\tau}.
}
\]

**Proof sketch.** Along the inter-center axis, the smoothed sum is
\(f(t) \propto e^{-(t-\Delta/2)^2/(2s^2)} + e^{-(t+\Delta/2)^2/(2s^2)}\) with
\(s = \sqrt{\sigma^2+\tau}\). Setting \(f''(0)=0\) gives
\(\Delta^2 = 4s^2\). For \(f''(0)<0\) (local minimum at midpoint,
bimodality), we need \(\Delta > 2s\). \(\square\)

The maximum scale at which the pair remains cleanly split is therefore

\[
\tau\_{\mathrm{sep}}
=
\frac{\Delta^2}{4} - \sigma^2.
\]

**Characteristic-scale resolvability** (\(\tau=\sigma^2\)):

\[
\boxed{
\Delta > 2\sqrt{2}\,\sigma \approx 2.83\,\sigma.
}
\]

This is the exact Gaussian scale-space analog of a Rayleigh separation limit
for feature identity at the natural operating scale. It is the
information-theoretic condition under which condition (C3) _can_ be satisfied —
if the smoothed density is unimodal, no graph-level rule can legitimately
separate the pair.

**Unequal widths** (conservative sufficient condition):

\[
\Delta > 2\max\!\bigl(\sqrt{\sigma_a^2+\tau},\,\sqrt{\sigma_b^2+\tau}\bigr).
\]

### 12. Bridge to the graph-local Q criterion

If clusters are well-approximated by Gaussians with empirical scales
\(\hat v*A, \hat v_B\) and centroid separation \(\Delta*{AB}\), the all-pairs
inter-cluster coupling decays as

\[
\mathrm{Inter}(A,B;v)
\approx
\exp\!\left(
-\frac{\Delta*{AB}^2}{
2\,d*{\mathrm{eff}}(v+\hat v_A+\hat v_B)
}
\right).
\]

Clean graph-level separation requires this to be much smaller than the intra
values (\(\approx e^{-1}\) at equilibrium), yielding

\[
\Delta*{AB}^2 \gg d*{\mathrm{eff}}(v+\hat v_A+\hat v_B).
\]

This is the graph-local expression of the bimodality threshold in Section 11.
At equilibrium (\(v \approx \hat v_A \approx \hat v_B \approx \sigma^2\)):

\[
\Delta*{AB}^2 \gg 3\,d*{\mathrm{eff}}\,\sigma^2,
\]

which for \(d\_{\mathrm{eff}} = 1\) recovers a threshold around
\(\Delta > \sqrt{3}\,\sigma \approx 1.73\,\sigma\) on the graph
(weaker than the continuous 2.83 because the graph also has co-activation
evidence). So the continuous-density bimodality test is the conservative outer
envelope, and the Q-gated graph can sometimes resolve slightly tighter pairs.

The gap between these two thresholds — information-theoretic (continuous
bimodality) and algorithmic (graph Q) — is the operating margin. The
continuous condition defines when separation is _possible in principle_; the
graph condition defines when the finite scaffold _achieves_ it.

---

## Part IV: The Multiscale Partition Problem

Parts II--III define what a cluster is and when it can exist. This part
addresses the computational challenge: finding all clusters simultaneously is
hard, and the system requires a structured search strategy.

### 13. Why partition inference is hard

\(W*v\) provides good \_local pairwise* evidence. But a _partition_ is a global
discrete object; its quality is a nonlocal function of the full labeling. The
search space is exponential, and any practical method (AP, spectral, BP,
merge-split) explores only a tiny, biased subset.

On a sparse scaffold graph, absent edges carry no vote, further restricting the
reachable partition family. This is why the system cannot rely on a single-shot
partition proposal; it needs proposals followed by gated refinement.

### 14. Scale response and recursion

- The **scale grid** treats evidence as a function of \(\tau\). Peaks or
  curvature in the response \(\Phi*C(\tau)\) indicate \_when* finer structure
  becomes identifiable — that is, at what scale new features cross the
  detectability threshold (Section 10) and become eligible to satisfy (C1).
- **Recursion** implements a coarse-to-fine strategy: "merged smoothed
  description first, then narrow the subspace and re-enter at a smaller
  effective scale." Submodes can appear only where the coarse description
  explicitly allowed them — this respects the hierarchical taxonomy (Part I):
  a coarse-scale cluster must be established before its sub-clusters can be
  sought.

### 15. Proposal, gating, refinement pipeline

1. **AP on smoothed PMI** proposes exemplar-based fragments on each lifted
   component. Exemplar count supplies a candidate \(K\) without requiring it to
   be pre-specified.
2. **Q-gated pair merges** collapse AP shards only when inter-cluster coupling
   meets or exceeds the weaker intra-cluster coherence _and_ partition \(Q\)
   rises — enforcing (C3).
3. **Damped boundary refinement** reassigns labels by weighted neighbor votes,
   accepting iterations only when partition \(Q\) improves — sharpening (C1).
4. **Residual cleanup** absorbs isolated tiny fragments or satellites that fail
   (C2).

Density or mass proposals constrain where cuts and seeds live when true
multimodality exists. But these are _proposals_, not oracles: rings, tori, and
similar single-component manifolds without central peaks can fool density-only
heuristics. The Q gate prevents spurious splits — a manifold cluster that
satisfies (C1) as a single unit will not be broken apart.

### 16. Previously observed failure mode

One large lifted-edge connected component plus a single median-cut that returns
no Q-improving bipartition yields \(n\_{\mathrm{clusters}}=1\) and halts
recursion prematurely. The current AP-based implementation replaces that
one-shot cut with exemplar proposals plus the Q merge/refine pipeline, which
now recovers six fine leaves on the hierarchical Gaussian fixture.

---

## Part V: Consequences for Synthetic Fixture Design

The axioms (Part 0) and definition (Part I) constrain how test fixtures must be
designed. A fixture is well-posed only if it lives within the operating domain
and the intended structure satisfies the resolution theory.

### 17. The joint operating window

For a synthetic fixture component \(i\) to be recoverable by Stage 1, there
must exist a \(\tau\) satisfying all relevant constraints:

\[
\tau \le \tau*{\mathrm{det},i},
\qquad
\tau \le \tau*{\mathrm{sep},ij}\;\text{for all intended neighbors}\ j,
\qquad
\tau \approx \sigma_i^2.
\]

If the natural scale \(\sigma_i^2\) lies outside the admissible window, the
fixture design itself should be changed rather than weakening the
interpretation of `expected_tau`.

### 18. Role of `transition_radius`

In the exact faded-density fixtures, tissue labels are assigned by

\[
\lambda(d) =
\exp\!\left(-\frac{d^2}{2(r*{\mathrm{fade}}\sigma)^2}\right),
\qquad
d*{1/2} = r\_{\mathrm{fade}}\sigma\sqrt{2\ln 2}.
\]

`transition_radius` is a **label/support parameter**, not a scale-selection
parameter. It influences detectability only through the resulting tissue floor
\(u\) and the signal/tissue mass split. Synthetic metadata should be derived
from the contrast and separation window (Sections 10--11), not from
`transition_radius` directly.

### 19. Three resolution notions in the current system

| Resolution type                     | Formula                                                                        | Source    |
| ----------------------------------- | ------------------------------------------------------------------------------ | --------- |
| Scale-grid spacing (Lindeberg FWHM) | \(r = 1/\sqrt{2}\) adjacent in \(\sigma\)                                      | SI S2.1   |
| Geometric motion threshold          | \(\delta\_{\min} = \kappa(1-r)\sqrt{\tau}\)                                    | SI S2.4   |
| Feature distinguishability          | \(\Delta > 2\sqrt{\sigma^2+\tau}\) and \(C*i(\tau) \ge \kappa*{\mathrm{det}}\) | this note |

The first two are in the paper/SI. The third is the missing anchor that
synthetic fixtures should obey — and it is now derived directly from the
operating domain axioms and cluster definition.

### 20. Practical checklist for fixture design

For each intended signal component \(i\):

1. **Verify operating domain (A1--A3):** Specify \(\sigma_i\), \(H_i\) (or mass
   \(a_i\)), and tissue floor \(u > 0\). The fixture must have positive noise
   and positive tissue — degenerate limits are excluded.

2. **Check characteristic-scale detectability (C1):**
   \[
   \frac{H*i}{u} \ge \kappa*{\mathrm{det}}\,2^{d/2}.
   \]

3. **Check characteristic-scale separation (C3):** For each intended neighbor
   \(j\):
   \[
   \Delta\_{ij} > 2\sqrt{2}\,\max(\sigma_i,\sigma_j).
   \]

4. **Publish `expected_tau` only inside the admissible window.** If the window
   is empty, the fixture is asking Stage 1 to recover structure that the
   smoothed density no longer contains — redesign the fixture.

---

## Appendix: Status and Open Calibration

The continuous theory (Sections 9--12) is exact for isotropic Gaussian bumps
above a uniform floor. The graph-local bridge (Section 12) is an approximation
that becomes tight when cluster populations are large. What remains operational
rather than closed-form is the calibration of \(\kappa\_{\mathrm{det}}\) against
the current Stage 1 load-based \(\tau\) selector and the finite-node scaffold
graph.

That calibration is the next step. But even before fitting
\(\kappa\_{\mathrm{det}}\), the inequalities above provide a principled way to
determine:

- how much tissue a mode can tolerate,
- how far apart two features must be,
- and whether a synthetic fixture is asking Stage 1 to recover structure that
  the underlying smoothed density no longer contains.

The deductive structure of this document — axioms → definition → machinery →
limits → design — ensures that every operational decision traces back to the
foundational properties of the operating domain.

---

## Addendum: Literature-Informed Method Revisions

The focused literature review in
`noisy_manifold_literature_review.md` does not overturn the framework developed
above. But it does sharpen which parts of the current picture should be treated
as primary, which parts should remain conservative, and where the implementation
priority should shift.

The most important lesson is that Stage 1 should not be framed as "guess the
correct scale" at the outset. The noisy local PCA and tangent-perturbation
literature instead suggests that local recovery is only meaningful inside an
**admissible scale window**:

- at scales that are too fine, local statistics are noise-dominated or
  sample-starved;
- at scales that are too coarse, curvature, density drift, or neighboring
  sheets contaminate the local estimate;
- only in an intermediate interval can tangent, extent, and noise-floor
  estimates be trusted.

This suggests the following methodological revisions.

### A. Search for a valid local scale window first

The search policy should first aim to **certify a local operating interval**
\([v_{\min}, v_{\max}]\) rather than immediately commit to a single discovery
scale.

Operationally, a scale \(v\) is admissible only if the local neighborhood or
catchment simultaneously satisfies:

1. **support adequacy:** enough mass / node support to make local moments stable;
2. **geometric coherence:** a dominant local tangent picture or local manifold
   picture exists;
3. **floor stability:** the inferred local floor does not change violently under
   small scale refinement;
4. **boundary separation:** neighboring candidate regions are not already
   strongly mixed at that scale.

Once such an interval exists, the algorithm may still pick a representative
working scale \(v^* \in [v_{\min}, v_{\max}]\), but this choice should be made
_inside_ the certified interval rather than before it. This reframes scale
selection as a constrained selection problem rather than a blind search.

### B. Treat stable finite-scale surrogate features as first-class objects

The ridge and scale-space literature reinforces a point that is already implicit
in Part I: a feature of the smoothed object \(\rho_v\) need not be an intrinsic
feature of \(\rho\) in order to be meaningful.

In particular:

- a **composite feature** that is stable over a nontrivial scale interval is not
  merely an error term;
- it is often the correct coarse summary of unresolved finer structure;
- its fine-end death can therefore be interpreted as a genuine _recursion
  signal_, not merely as a nuisance.

This strengthens the parent-death / child-birth handoff discussed in Section
380. The system should therefore explicitly treat some composite features as
**legitimate coarse-scale discoveries** whose role is to delimit where finer
search is warranted.

### C. Insert a cheap local contraction step before heavier geometry

The local-averaging literature suggests that simple neighborhood contraction can
already suppress a large fraction of ambient noise before more delicate geometry
is estimated.

That motivates the following possible Stage 1 ordering:

1. build the current neighborhood / catchment;
2. perform a small local contraction or averaging step;
3. estimate tangent, local extent, and local floor on the contracted statistics;
4. only then form scale-conditioned scaffold evidence and cluster proposals.

This should not replace the scaffold or Q-based logic. Rather, it suggests a
low-cost **preconditioning layer** that may improve stability of node moments,
settlement, and early cluster proposals in high-noise settings.

### D. Keep the noise atlas conservative

The heteroscedastic PCA literature strongly supports modeling nonuniform noise,
but it also argues against jumping too quickly to rich covariance models unless
they are actually estimable.

The practical implication is:

- default to the minimal viable atlas described in Section 932;
- prefer scalar or blockwise floor corrections when those are the only
  trustworthy quantities;
- rely on off-diagonal structure more than raw diagonal variances when the
  latter are obviously floor-contaminated;
- perform whitening only when the local covariance estimate is sufficiently
  stable, well-conditioned, and confidence-weighted.

So the literature pushes the design further toward **"least ambitious model that
is locally trustworthy"** and away from premature escalation to full
per-node covariance tracking.

### E. Treat tubular noise as a bias problem, not only a variance problem

The manifold-fitting and manifold-MLS literature makes clear that ambient tube
noise is not only an inflation of variance. If the local frame is slightly
misaligned, projection and denoising become systematically biased.

This suggests a more iterative picture of settlement:

1. estimate a provisional tangent / floor / extent;
2. debias or contract locally using that estimate;
3. re-estimate the tangent / floor / extent on the debiased patch;
4. declare the region settled only if these quantities stabilize.

In other words, settlement should not mean merely "one PCA looked clean." It
should mean that the local description is **self-consistent under one step of
refinement**.

### F. Implementation priority: certify, then recurse

Putting the above together, the literature suggests the following priority order
for Stage 1 implementation:

1. **certify admissible local scale windows;**
2. **stabilize local moments and settlement tests;**
3. **treat stable composite features as valid coarse discoveries;**
4. **use parent fine-end death as an explicit recursion trigger;**
5. **only then enrich the noise atlas or local covariance model.**

This ordering is important. It implies that local validity and recursion timing
are likely to yield more immediate returns than richer covariance parameteriza-
tions.

### G. What remains distinctive about Proteus

The reviewed literature covers many individual ingredients:

- local PCA under noise,
- tube-based manifold recovery,
- density-ridge surrogates,
- local denoising by averaging,
- and heteroscedastic covariance correction.

What still appears nonstandard is their combination into a single scaffold-native
recursive system:

- local geometry represented on scaffold primitives;
- discovery framed as coarse-to-fine search through scale-space;
- parent death and child birth used as dual timing signals;
- and a noise atlas tied to settlement and hierarchy rather than only to
  coordinates.

So the literature mostly validates the direction of this document, while also
clarifying a sharper operational stance:

- search for **valid windows**, not just points;
- treat **finite-scale surrogate features** as legitimate objects;
- prefer **cheap contraction before expensive geometry**;
- and keep the **noise model conservative until the local evidence justifies
  more**.

### H. Equilibrium confidence replaces fixed CV tolerance

The current stabilization rule declares equilibrium when the last
`min_equilibrium_epochs` variance-CV values fall below a fixed tolerance
derived from `k`. This is a necessary condition, but not sufficient: a
scaffold can satisfy it while still drifting structurally (topology edits
ongoing, mean-min-distance trending, incoherence rising).

A stronger criterion replaces the fixed threshold with a **confidence-based
equilibrium test** that requires both structural quiet and moment flatness.

#### Layer 1: Topology quiet

Before moment metrics are even consulted, require that structural edits have
ceased over a trailing window of `W` epochs:

- `splits == 0` for the last `W` epochs,
- `nodes_pruned == 0` for the last `W` epochs,
- `node_count` unchanged over the last `W` epochs,
- `lifted_isolated_mature` not increasing.

If topology is still active, the scaffold is not in equilibrium regardless of
what moment statistics say.

#### Layer 2: Moment flatness

Given topology quiet, require that core moment diagnostics are not only low
but also **flat and non-trending** over the trailing window:

| Metric | Condition |
| --- | --- |
| `cv` | slope over last `W` epochs near zero |
| `cv` | rolling std over last `W` epochs small relative to mean |
| `mean_min_distance` | slope near zero (reconstruction error stable) |
| `variance_load` | mean `sigma^2 / tau` flat (not still drifting toward cap) |
| `incoherence_cv` | not increasing (optional, diagnostic) |

The key shift is from "below a fixed threshold" to "flat and quiet." A
variance-CV slope near zero is more informative than an arbitrary absolute
tolerance, because:

- it adapts automatically to different data scales and scaffold sizes;
- it catches slow exponential drift that sits below threshold but has not yet
  converged;
- it does not require hand-tuning a `cv_buffer` constant per problem.

The absolute threshold can still serve as an upper sanity bound (if CV is
enormous, flatness alone is not enough), but the primary stopping signal
should be **trend exhaustion**: the system has stopped changing.

#### Layer 3: Optional frozen verification

As a secondary diagnostic or high-stakes gate (e.g., before committing to a
recursion decision), the system may optionally run 1--2 additional epochs with
topology edits disabled and verify that the moment diagnostics remain within a
small tolerance band. This is not the default stopping mechanism; it is a
post-hoc confidence check.

#### Relationship to scale-search confidence

The above addresses **within-scale equilibrium**: "given this `tau`, is the
scaffold settled?" A separate question is **scale-selection confidence**: "is
`tau_star` itself the right scale?" That is better assessed in the controller
via:

- breadth of the stabilized band around `tau_star` on the grid,
- margin between `tau_star` and the nearest unstabilized grid point,
- local prominence or curvature of the load trace at the selected point.

These two confidence scores (within-scale equilibrium, and scale-selection
robustness) are complementary and should both be available to downstream
decisions like recursion gating.

#### Why this replaces the frozen-settlement addendum (Section E above)

Section E proposed iterative settlement via repeated estimate-debias-reestimate
cycles. On reflection, most of what that aimed to catch — residual EWMA drift,
historical artifacts, premature convergence declarations — is better addressed
by a richer stopping criterion that already uses the metrics being logged. The
frozen verification pass remains available as an optional hardening step, but
the default path is simply to not declare equilibrium until the scaffold is
demonstrably flat, quiet, and structurally stable.
