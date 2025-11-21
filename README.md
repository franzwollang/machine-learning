# Toward a Universal Machine Learning Framework

<small><em>A First-Principles Approach Through Geometric Manifold Learning</em></small>

## The Limits of Monolithic Architectures: An Intuitive Tour

At the heart of most modern AI is a powerful and elegant idea: build a deep, end-to-end differentiable graph, define a global objective (a loss function), and use backpropagation to adjust every parameter to minimize that loss. This approach has proven extraordinarily effective, yet it produces architectures with characteristic and predictable limitations. This section offers an intuitive tour of why these limitations arise and motivates this repository’s aim: a first-principles geometric framework, Proteus, that learns a multi-scale model of the data manifold and uses it to power practical systems.

### 1. The Price of End-to-End Optimization: Balancing a Ball on the Tip of A Pencil

Imagine building a complex machine, like a car engine. A sane design would involve distinct, modular sub-assemblies: a fuel injector, a cooling system, an alternator. Each can be understood, tested, and replaced with minimal side effects on the others. They have clean interfaces and local responsibilities.

A monolithic deep learning model is different. Its end-to-end differentiability means that, in principle, every parameter is coupled to every other, limited only by the structure of the computational graph and data flow. There are no true firewalls or modular boundaries. The gradient of the global loss function flows back through the entire network, creating a dense web of dependencies. The sheer number of interacting degrees of freedom makes the system inherently unstable and difficult to control with precision. It relies on a patchwork of regularizers (like dropout or weight decay) to manage this complexity, but these are statistical guardrails, not hard architectural boundaries.

### 2. The Engine of Entanglement: Fully-Connected and Attention Layers

This tight coupling isn't just theoretical; it's a direct consequence of the workhorse components of modern architectures. In a **fully-connected layer**, every output neuron is a weighted sum of _every_ input neuron. In an **attention module**, every output token is a weighted combination of _every_ input token. These operations are, by design, engines of total information mixing.

This creates a profound inefficiency. Any clean separation of concepts achieved in one layer becomes inherently unstable when passed to the next. The mixing operation, with its high degrees of freedom, forces the model to expend a vast amount of its capacity simply to salvage and maintain that separation. Instead of cleanly building toward a "natural" understanding, it's engaged in a constant, unstable battle to re-isolate the very signals it needs for the next stage of reasoning.

### 3. How "Concepts" Form: The Logic of Incremental Error Correction

Given this constant mixing, how does the system learn to recognize things? It doesn't discover "concepts" in the way a scientist might classify species. Instead, it learns by incrementally correcting its mistakes. The result is that a localized group of neurons is not incentivized to form a clean, "natural" concept like "a wheel." Instead, that group will represent whatever **composite mixture** of signals—a patch of texture, a specific curve, a color gradient from a completely different part of the image—was most effective at nudging the global loss in the right direction. The resulting internal "concepts" are not aligned with the natural, separable structures in the data manifold; they are opportunistic composites shaped by the pressures of optimization.

These composites can be extraordinarily effective because the underlying data manifold is often wrinkled and folded in complex ways within its high-dimensional space. This geometry admits a vast landscape of viable approximations, and end-to-end optimization is a powerful tool for finding one that, while not necessarily "natural", is highly performant for the specific task.

### 4. The Challenge of Cross-Scale Reasoning: A Necessary but Unstable Hierarchy

This leads to a critical, subtle failure in reasoning. Any efficient learning system, when faced with a high-dimensional space, will naturally tend to build a hierarchy of patterns, driven by two constraints. First, to reduce the combinatorial explosion of possibilities, it is vastly more efficient to first identify simpler, statistically common motifs (like edges and textures) and then use those as building blocks for more complex patterns (like objects and scenes). Second, while training adjusts weights globally via backpropagation, inference is causally constrained: the network must process information layer by layer in a forward pass. This causal flow forces a division of labor where early layers, having access only to raw inputs, necessarily specialize in simpler features, and later layers must learn to recognize compositions of what came before. This hierarchical, compositional approach is therefore not just an emergent property; it is a near-necessity for tractable learning.

However, many vital real-world concepts are **cross-scale**. For example, recognizing a "worried expression" requires integrating low-level geometric cues (the precise curve of a lip corner, the angle of an eyebrow) with high-level context (the overall structure of the face).

Or consider a more complex, dynamic example: interpreting a video of a tense negotiation. To infer that one person is bluffing, a system might need to synthesize a vast array of cross-scale signals. It must latch onto a fleeting, low-level visual cue (a person tapping their fingers for a few seconds), correlate it with a subtle, low-level audio cue (a slight tremor in their voice when mentioning a key number), and contextualize both with high-level information from its memory (this person has used this specific combination of behaviors before in similar situations).

In both scenarios, the core task is the same: the model must identify delicate, low-level features from early in the network and transport them, without distortion, to be integrated with high-level context assembled much later. This requirement for high-fidelity, long-range feature transport is fundamental.

But what are the intermediate layers responsible for this transport? They are the very attention and fully-connected modules we just described—engines of total information mixing. This makes the necessary hierarchy an unstable proposition. Trying to preserve a specific, subtle signal through a series of high-dimensional mixing operations is like trying to carry a single drop of blue ink through a series of churning vats. The signal is highly likely to be diluted, distorted, or overwritten. The model often learns statistical shortcuts instead, which is a primary reason why generalization can be so brittle.

#### A note on residual connections

Residual (skip) connections are invaluable for optimization stability and gradient flow, but they do not create hard modular boundaries or guarantee protected signal transport. They additively combine an identity path with a mixing path. Downstream blocks still receive a high-capacity mixture and remain free to remap, overwrite, or entangle the carried signal. In other words, skips tame the symptoms of deep training, like vanishing gradients, without changing the underlying coupling. There is no selective routing, no isolation contract, and no invariant that preserves specific features across depth. As a result, residualized stacks still face the same cross-scale interference problem. They are better behaved, but not modular. Fixing the root issue would require architectural mechanisms that explicitly preserve and compose information—such as protected channels or invertible transports—rather than relying on unconstrained mixing plus additive shortcuts.

<small><em>Footnote:</em> Skips make deep nets trainable by improving gradient/information flow but do not isolate features—downstream mixing remains unconstrained [He et al., 2016](https://arxiv.org/abs/1603.05027). When true preservation is needed, architectures impose invertibility (e.g., i‑RevNets)—useful but restrictive—confirming that plain residuals don’t provide protected channels [Jacobsen et al., 2018](https://arxiv.org/abs/1802.07088).</small>

### 5. The Cascade: Why Entanglement Leads to Systemic Failures

This deep entanglement of coupled parameters and composite concepts has practical, systemic consequences.

- **Catastrophic Forgetting:** When the model learns a new task, the gradient updates will repurpose the shared, composite features that were crucial for previous tasks. Old knowledge can be progressively overwritten in unpredictable ways.
- **Unstable Reinforcement Learning:** The entanglement means that updates can cause widespread, unpredictable shifts in the representation space, making it difficult for credit assignment to find a stable footing.
- **Representation Drift:** For any system that needs a long-term memory, this is a fatal flaw. If the internal meaning of concepts shifts every time the model is updated, old memories become inaccessible to new queries.

In short, while monolithic optimization is excellent at finding a single, high-performing solution for a fixed task, it produces models that are difficult to interpret, extend, or trust over time. They are powerful, but opaque and brittle. The aim of this work is not to discard their power, but to ground it in a more stable and modular foundation.

### 6. Conflating the Map and the Territory: Mixed Objectives

Finally, there is a subtle but critical conflation of goals. Any supervised learning task implicitly involves two distinct problems:

1.  **The Representation Problem:** Learning a faithful model of the data's underlying structure—the shape of its manifold. This is about understanding "what the world is like."
2.  **The Task Problem:** Learning a function on top of that representation to achieve a specific goal (e.g., classification). This is about learning "what to do."

Ideally, these would be separate steps: first build a robust, high-fidelity "map" of the data manifold, and _then_ learn how to navigate it to solve the task. The second problem is vastly simpler if the first is solved well.

End-to-end training, however, fuses these two objectives into one. The optimization process is therefore free to find shortcuts. It will happily sacrifice the accuracy and completeness of its "map" of the world if a distorted, simplified, or biased representation makes it easier to solve the immediate task. This provides yet another powerful incentive for the model to learn non-natural, composite features that are "good enough" for the task but are a poor, brittle foundation for generalization, memory, or any subsequent task. The model doesn't just learn a task; it learns a distorted view of the world that is custom-fit to that task.

### 7. A Note on Modern Objectives: Why the Syndrome Persists

It is true that modern training is more sophisticated than applying a single, simple loss. Techniques like **contrastive learning** or the use of **auxiliary losses** are powerful methods that introduce more "local" objectives, shaping the loss landscape to encourage the formation of better-separated, more useful representations. However, these are improvements to the _objective_, not the _architecture_. They do not build the hard, modular firewalls necessary to prevent entanglement. The gradients from the final, task-specific objective still backpropagate through the entire network, influencing the very same weights shaped by the "local" contrastive loss. The intermediate layers are still high-capacity mixing engines. Consequently, the entire syndrome we have described persists. The system is still engaged in an unstable battle to preserve features across layers; its representations are still plastic and vulnerable to catastrophic forgetting and long-term drift; and it is still incentivized to sacrifice the general fidelity of its world model for task-specific shortcuts. These techniques create a better-behaved monolithic system, but they do not make it a modular one.

### 8. External Empirical Support: Fractured Entangled Representations

Independent work by Kumar et al. provides concrete, neuron-level evidence for many of the architectural concerns outlined above in this section and in §5. Their position paper, **“Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis”** ([`arxiv:2505.11581`](https://arxiv.org/html/2505.11581v1)), compares small CPPNs evolved through open-ended search (Picbreeder/NEAT) to networks trained with conventional SGD to produce _identical_ images. While the outputs match, the internal structure does not: the evolved networks exhibit **Unified Factored Representations (UFR)**—clean, modular units corresponding to coherent parts (e.g., symmetric wings, separately controlled apple stem and body)—whereas the SGD networks exhibit **Fractured Entangled Representations (FER)**, in which each concept is redundantly split across many neurons and entangled with others. Weight sweeps through evolved CPPNs produce smooth, human-meaningful edits, while analogous sweeps in SGD models typically yield high-frequency, non-local distortions.

This FER/UFR contrast is an empirical instantiation of the same syndrome discussed here: scaling and better objectives can improve benchmark performance without producing clean, modular representations. In that sense, the FER results function as external support for Proteus’s starting point. Where Kumar et al. diagnose that standard SGD on fully-coupled architectures tends toward FER, Proteus is explicitly designed to _encode UFR-like structure in the substrate itself_: concepts are realized as contiguous, unimodal clusters on a learned manifold, with fuzzy membership vectors providing a factored coordinate system over these clusters. The aim is to avoid “imposter” performance by ensuring that the internal geometry that makes a task possible is itself stable, interpretable, and reusable across tasks and over time.

## Why Are Architectures Like This? A Pragmatic Response to a Hard Problem

The architectural side-effects detailed in the preceding tour might seem strange, but they are not arbitrary. They are the legacy of a pragmatic pivot away from an even harder problem that dominated machine learning research for decades: directly modeling the structure of data.

Before the rise of modern deep learning, a significant focus of the field was on solving the representation problem head-on. This approach is intellectually pure, but it repeatedly ran into a formidable obstacle: the **curse of dimensionality**. This "curse" is not a single problem, but a syndrome of counter-intuitive geometric and statistical properties that emerge in high-dimensional spaces:

- **Concentration of Measure:** As dimensionality increases, the mass of a distribution concentrates in a thin, equatorial shell. The concept of a dense "center" evaporates, and the volume of any small region becomes negligible, making methods that rely on local density unreliable (see the Gaussian isoperimetric inequality: [Borell, 1975](https://en.wikipedia.org/wiki/Isoperimetric_inequality#Gaussian_isoperimetric_inequality)).-- Fun note: this same concentration helps explain why cosine (shell) distance works well for high-dimensional embedding retrieval in modern RAG systems.
- **Separation and Collapse of Modes:** Nearest and farthest neighbor distances concentrate (distance contrast collapses), making simple distance-based discrimination unreliable (see [Radovanović et al., 2010](https://www.jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf)). For fixed per-dimension signal-to-noise, the separation required between modes to maintain a fixed Bayes error typically grows like √d (e.g., isotropic Gaussians); with finite samples, reliable separation becomes harder unless separation or data scale accordingly.
- **The Multiplicative Effect of Noise:** Small per-dimension perturbations add in quadrature; across d dimensions they yield an aggregate ‖ε‖ ≈ √d·σ that can overwhelm the true signal. In practice, finite machine precision and accumulated round-off from long chains of linear/nonlinear operations act as a continual source of near-isotropic noise; at modern scales (deep pipelines, massive data), this materially degrades SNR and numerical stability.

Directly learning a manifold under these conditions is extraordinarily hard. The breakthrough of convolutional and residual networks was not that they solved this problem, but that they found a brilliant way to **sidestep** it. By creating a deep, end-to-end differentiable system and optimizing for a specific task, they allowed the optimization process itself to find a "good enough" projection of the manifold that was useful for the task, without ever needing to model the manifold explicitly.

This historical context is key. The monolithic architectures we have inherited are the successful, pragmatic legacy of sidestepping these profound geometric challenges. They work, but as the preceding tour detailed, they come with a distinct set of architectural costs.

### Addressing a Skeptical View: Aren't Scaling and SSL Enough?

The critique of monolithic architectures invites two powerful counterarguments from a skeptical perspective: first, that modern self-supervised learning (SSL) has already solved the representation problem, and second, that scaling alone is a sufficient strategy to overcome the remaining architectural weaknesses.

#### 1. On Self-Supervised Learning: A Better Engine on the Same Chassis

Self-supervised methods are a profound leap forward. By creating ingenious pretext tasks (like contrastive learning or masked prediction), they force the model to learn the data's intrinsic structure, providing a far better starting point than random initialization. However, SSL represents a brilliant new objective that still operates within the existing architectural framework. The resulting representations, while powerful, are still diffuse, entangled patterns distributed across millions of parameters. More importantly, they are not protected. The moment you fine-tune an SSL backbone on a downstream task, task-specific gradients begin to flow back through the entire network, subtly repurposing and distorting the general world model to fit the new, specific objective. SSL gives us a more robust monolith, but it is a monolith still, subject to the same long-term drift and lacking the hard modularity needed for truly editable, reliable knowledge.

#### 2. On Scaling Laws: The Modern Epicycle

While scaling has delivered remarkable empirical results, the argument that it will solve all remaining issues resembles Ptolemy's epicycles. By adding more circles, his system could predict planetary motion with arbitrary accuracy, but it was complex, lacked explanatory power, and ultimately missed deeper truths. Similarly, scaling approaches can deliver phenomenal benchmark performance, but may not address fundamental issues of interpretability, efficiency, and trust. A model with trillions of parameters is an opaque oracle. An elegant, geometric model that captures the intrinsic simplicity of the data manifold is more efficient, more powerful, and offers a path toward systems that can reason about their own knowledge, a capability that scaling alone can never provide. The goal is not just prediction, but understanding.

## A complementary, geometric picture

The curse of dimensionality paints a bleak picture of a vast, empty space. If real-world data were uniformly distributed throughout this space, generalization would be impossible; nearly every new data point would be an isolated "outlier," far from any training example.

The fact that machine learning works at all is strong evidence for the **manifold hypothesis**: real-world high-dimensional data is not spread out evenly, but instead concentrates in the vicinity of a much lower-dimensional manifold. This underlying structure is a prerequisite for learning. Without it, there would be no smoothness, no local correlations, and no hope of interpolating between known examples to understand new ones.

This necessary structure—the ridges, sheets, and junctions where the data actually lives—is the foundation of the geometric view. From this perspective, the very idea of a "concept" is inseparable from the manifold's geometry. For a concept to be useful for generalization and prediction, it _must_ correspond to a **cluster**: a tangible geometric object that represents a stable, unimodal peak in the data's probability density function.

Why is this a necessity? Any useful concept must map to a region that is:

1.  **Contiguous and Locally Smooth:** This ensures members of the concept share a notion of proximity, allowing for reliable interpolation. A "concept" mapping to a scattered set of points would have no predictive power.
2.  **A Mode of Higher Density:** The region must be more dense than its surroundings. This is what makes a grouping statistically significant, rather than an accident of uniform sampling.
3.  **Unimodal (at the current scale):** The region should have one primary peak. A multi-modal region is a composite of several sub-concepts that have not yet been resolved.

Therefore, any generalizable concept must correspond to a cluster with these properties. This appears to be the most natural geometric form a generalizable concept can take. This picture has several powerful properties:

- Locality by construction: parts are defined by contiguous regions, not by weight sharing.
- Scale as a Geometric Property: In a DNN, "scale" is an emergent and entangled property of the feature hierarchy. In a direct geometric model, scale is a controllable, physical parameter. The manifold can be analyzed at different levels of resolution by applying a smoothing kernel with a specific bandwidth ($\sigma$), akin to changing the focus on a microscope. Different structures—from fine-grained sub-clusters to broad super-clusters—will become distinct and unimodal at different, characteristic smoothing scales. This allows the hierarchy of concepts to be discovered and modeled explicitly, rather than existing only implicitly in the weights of a network.
- Editability: geometry can be refined (move/split) or minimally warped where curvature demands it, without dissolving unrelated parts.
- Direct control over local dimensionality: In a DNN, a latent concept is represented by the collective activity of a localized group of neurons. The _effective dimensionality_ of this representation—a measure of its richness and complexity—is itself a high-degree-of-freedom parameter that is only indirectly constrained by the global objective. The model must find a delicate balance: if the effective dimensionality is too high, it re-introduces the curse of dimensionality internally; if too low, it creates compression artifacts. A geometry-first approach, by contrast, can measure and explicitly use the data's true local intrinsic dimension to guide the model's structure, ensuring the complexity of the representation is well-matched to the complexity of the data at every point.

## The Proteus Approach: A Principled Geometric Solution

The preceding sections have laid out a fundamental tension: monolithic architectures are pragmatically effective but architecturally flawed, while a direct, purely geometric approach is intellectually appealing but historically intractable due to the curse of dimensionality.

The goal of Proteus is therefore not to replace deep neural networks. DNNs, and attention mechanisms in particular, are extraordinarily powerful function approximators. The issue is not the tool, but the task to which it is applied. Proteus is designed to solve the foundational representation problem: building a stable, interpretable, multi-scale geometric map of the data. With this map in place, DNNs can be deployed to their greatest strength: learning complex, task-specific functions on top of a meaningful and robust coordinate system. Proteus builds the skeleton; DNNs provide the muscle.

Proteus is designed to resolve this tension by creating a practical, scalable framework that takes the geometric picture seriously. It's an attempt to overcome the historical difficulties not by sidestepping them, but by tackling them with a more principled and powerful set of tools.

The core strategy is a **two-stage, coarse-to-fine decomposition** of the problem:

1.  **Stage 1: Fast, Non-Parametric Scaffolding.** First, find the essential structure of the manifold without making strong prior assumptions.
2.  **Stage 2: High-Fidelity Refinement.** Then, using that structure as a scaffold, build a detailed, generative model.

### How Proteus Finds Structure: The Multi-Scale Search

Proteus confronts the curse of dimensionality and the problem of scale by refusing to pick one scale at all. Instead, it performs a principled, **non-parametric search across all possible scales**, letting the data reveal its own inherent structure. This is operationalized through a fast, exploratory engine that analyzes the data under a range of different "smoothing" parameters.

This process allows Proteus to automatically:

- **Discover all characteristic scales:** It identifies the specific resolutions at which the data naturally forms stable, unimodal clusters. This allows it to decompose what might look like a single, complex multi-modal distribution into a hierarchy of simpler, unimodal components.
- **Measure local intrinsic dimensionality:** At each point on the emerging manifold, it estimates the "true" local dimensionality of the data. This allows the model to adapt its complexity, dedicating more resources to information-rich regions (e.g., a 2D sheet) and fewer to simpler ones (e.g., a 1D filament).

The output of this stage is not a final model, but a **scaffold**: a coarse but topologically sound map of the data's primary structures, their characteristic scales, and their local dimensionalities.

### How Proteus Models Structure: The "Flattening" Principle

With this scaffold in place, Stage 2 begins the high-fidelity refinement. The core idea is not to fit one single, fantastically complex function to the entire manifold. Instead, it is to find a set of local coordinate systems that make the manifold look as simple as possible in each local patch.

The goal is to apply targeted, **invertible non-linear corrections (warps)** only where necessary, with the explicit aim of transforming each local cluster region until it is approximately **Gaussian** (up to an affine transformation). This "flattening" principle is the key to tractability. Instead of a monolithic black box, the final model is a collection of simple, well-understood "Gaussian atoms," stitched together by a simplicial complex and composed with learned local transformations that account for the manifold's curvature. The model is only as complex as the data requires it to be, at any given point.

### Why This Geometric Approach Is Viable Now

This effort is not an attempt to revive a failed idea, but an argument that the idea's time has come. The original turn away from direct geometric modeling was a pragmatic choice based on the tools and compute of the day. The landscape is now different. Proteus stands on the shoulders of a recent confluence of mature, powerful innovations that make its ambitions tractable:

- **Performant Algorithmic Building Blocks:** Fast, high-quality community detection algorithms (e.g., Leiden, 2019), scalable approximate nearest-neighbor search (e.g., HNSW, ~2016), and efficient, invertible density models like normalizing flows (e.g., RealNVP/Glow, 2016-2018) provide the core machinery needed to execute the multi-scale search and high-fidelity modeling stages efficiently.
- **Synthesis of Mature Fields:** We can now synthesize decades of progress from parallel fields: non-parametric statistics, computational topology, and geometric deep learning. The ideas that stalled in the 2000s are no longer isolated threads; they can be woven together into a cohesive framework.
- **A Pull from the Field:** The limitations of the dominant paradigm are now defining the frontier of AI research—reliability, lifelong learning, and true interpretability. There is a growing recognition that these are architectural problems that require foundational alternatives, creating a strong demand for the very properties a geometric approach provides.

### The Result: A New Coordinate System for Data

This process yields a final model that is far more than just a picture of the data. By identifying the fundamental, unimodal "atomic" clusters of the manifold, Proteus creates a new, powerful coordinate system. Any data point can be transformed into a **fuzzy membership vector**, where each component describes its degree of belonging to each of these atomic concepts.

This representation is powerful because it builds on the connection between fuzzy logic and statistical metrics. The relationships _between_ the atomic clusters—the "fuzzy predicates"—define a reconstructed manifold metric that is grounded in the statistical reality of the data. This makes the geometry fully actionable, providing a robust foundation for the applications that follow, like Mimir and Ensemble ρ-SFCs, which depend on this stable, meaningful representation.

<small><em>Footnote:</em> For the fuzzy/statistical link, see Bezdek’s fuzzy c‑means formulation linking memberships and distances [Bezdek, 1981](https://link.springer.com/book/10.1007/978-1-477-0450-1) and comparative analyses of fuzzy similarity/distance measures [Bouchon‑Meunier, Dubois & Prade, 2000](https://www.sciencedirect.com/science/article/pii/S0888613X00000777). For approximating general finite metrics from predicate-like components, see L1/cut‑metric decompositions and probabilistic tree embeddings—mixtures of cuts/trees approximate metrics with logarithmic distortion [Deza & Laurent, 2009](https://link.springer.com/book/10.1007/978-3-540-37126-2), [Bourgain, 1985](https://onlinelibrary.wiley.com/doi/abs/10.1002/cpa.3160380405), [Fakcharoenphol, Rao & Talwar, 2004](https://dl.acm.org/doi/10.1145/1007352.1007355).

## Downstream Applications: What a True Manifold Model Unlocks

The promise of Proteus is that it can solve the fundamental, historically intractable problem of learning a high-fidelity, multi-scale model of a data manifold. Conditional on this core representation problem being solved, it acts as a key that unlocks straightforward and natural solutions to other notoriously difficult problems. This repository explores two such direct applications that are enabled by this foundation:

- **Manifold-Aware Indexing (Ensemble ρ-SFCs):** The clusters discovered by Proteus are not just pictures; they are detailed geometric and statistical descriptions of how data is shaped. This information is precisely what is needed to overcome the limitations of standard, geometry-agnostic database indexes. The Ensemble ρ-SFCs project leverages this structure to design custom, adaptive space-filling curves that "respect" the shape of the data. This ensures that points belonging to a single concept are mapped to a single, contiguous range in a 1D index, achieving maximal compression and query performance.

- **A More Human-like Cognitive Architecture (Mimir):** Similarly, the stable, interpretable, and hierarchical concepts produced by Proteus provide the ideal building blocks for a more robust cognitive system. The Mimir project uses these "atomic concepts" as the vocabulary for a learning system—which in turn uses powerful DNN-based components—that can reason, remember, and adapt in ways that are more analogous to the human mind. Because the concepts are grounded in the data's geometry, they are stable across model upgrades (preventing catastrophic forgetting) and are naturally interpretable. The system can even apply Proteus to its own internal representations, creating a capacity for recursive self-understanding that is impossible with entangled, black-box models.

This research program is therefore composed of three tightly coupled efforts:

- **Proteus**: a two-stage, scale-aware framework that learns high-fidelity, generative manifold models via a simplicial core and a dual-flow field, with first-principles choices for scale grids, EWMA half-lives, and growth/motion rules.
- **Ensemble ρ-SFCs (Omega-/rho-SFCs)**: a practical indexing architecture that unifies density-guided, adaptive space-filling curves with functional “recipe trees” and a hierarchical cascade for high-dimensional embeddings.
- **Mimir**: a goal-driven cognitive stack that uses Proteus-derived structure for grounded perception, a perpetual memory system with consolidation and mapper networks, and a predictive-evaluative core for agency.

## Current State of Progress (high level)

### Proteus (Foundational, docs/Proteus/paper_1_foundational)

- Draft paper and SI with algorithms, derivations (S1–S12), and protocols; placeholders for figures/tables.
- Stage 1/Stage 2 specification complete enough to implement a reference model; experiments/visuals to be added.

### Proteus (Augmentation, docs/Proteus/paper_2_augmentation)

- Draft outlining query engine, representation layer (Elastic ANFIS AEs), recursive analysis tower, and streaming (Temporal Pyramids).
- Implementation guidance present; empirical benchmarks and API surfaces TBD.

### Ensemble ρ-SFCs (docs/Ensemble ρ-SFCs)

- ACM-style draft with core sections scaffolded (density-guided curves, global metric, recipe trees, cascade, feasibility study).
- Proofs, algorithms, and experiments to be inlined; references/metadata pending.

### Mimir (docs/Mimir)

- Three conceptual drafts: foundational cognitive core, perceptual engine, memory system.
- Clear architectural flows (Teacher→Proteus→Student bootstrapping; EMS; consolidation with mapper nets); implementation plan and evals pending.

### Code (code/)

- Early skeleton with Pipfile and tests scaffold; no full reference implementation yet.
- Benchmark and visualization stubs exist; prototypes to follow the Proteus Stage 1/2 specs.

## Repository Structure

- `docs/Proteus/` — Foundational and augmentation papers (+ SI), references.
- `docs/Ensemble ρ-SFCs/` — ACM draft for adaptive curves and indexing.
- `docs/Mimir/` — Foundational, perceptual engine, and memory system drafts.
- `code/` — Future reference implementation; `Pipfile` present; tests scaffold.
- `code/tests/` — Visualization and evaluation stubs (figures saved to tests by default).

## Getting Started

### Prerequisites

- Python with pipenv (preferred)
- LaTeX toolchain (for PDFs) — typically handled by a VS Code LaTeX plugin; no Makefile required.

### Setup (pipenv)

```bash
pipenv install --dev
pipenv shell
```

Run a placeholder script (when added) or explore tests:

```bash
python code/skeleton.py
```

### Building Papers

Open the desired `paper.tex` in VS Code and use your LaTeX plugin to build. Outputs are under the corresponding `docs/...` folder (e.g., `paper.pdf`, `SI.pdf`).

## Milestones and Roadmap

### Proteus (reference implementation)\*\*

- Stage 1: ANN loop, rank-weighted EWMAs, variance thresholds, pruning gauntlets, CV stopping.
- Stage 2: simplex-native updates, torsion ladder + shape guards, dual-flow solver, OOS queries.
- Experiments: PH, MMD, log-likelihood; ablations.

### ρ-SFCs (systemization)

- Formalize density-guided inverse map; implement recipe trees + cascade; feasibility benchmarks.
- Keying strategies (global metric vs hierarchical composite) and integration points.

### Mimir (modules)

- EMS k-NN index on sparse memberships; codec autoencoders; mapper networks for consolidation.
- Policy/critic scaffolding and SEE integration (for symbolic actions).

## Origin and Context

I have been working on these problems since 2017. I taught myself computer science and machine learning from scratch, spending untold hours from 2017–2019 reading and re-reading core CS/ML papers and banging my head against them until they made sense. This repository is the culmination of that early effort, followed by years of settling and maturation. What has changed recently is tooling: modern LLMs now help with the laborious parts of turning ideas and understanding into something communicable, testable, and complete. Proteus, and the downstream systems it unlocks, are my attempt to finally make that long arc concrete and find closure.

## Contributing

Draft-first, iterate quickly: prefer conservative, high-signal edits that reduce ambiguity and add empirical hooks.

Save generated visualizations under `code/tests/` by default; avoid interactive displays in commits.

## Citation (placeholder)

Formal BibTeX entries will be added as papers stabilize. For now, cite the Proteus draft via Zenodo once the draft date is finalized.

## License

TBD
