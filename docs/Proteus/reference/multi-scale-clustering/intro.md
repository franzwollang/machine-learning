# Proteus: A Unified Framework for Hierarchical, Multi-Scale Manifold Learning and Analysis

## 1. Abstract

We present **Proteus**, a novel, multi-stage framework for discovering and modeling the structure of complex, high-dimensional data at multiple scales. Traditional methods often force a choice between computational scalability and the fidelity of the learned model. Proteus overcomes this by decomposing the problem into two distinct stages. **Stage 1** employs a fast, statistically-driven, node-centric Growing Neural Gas (GNG) to rapidly perform a scale-space analysis and identify the characteristic scales and approximate topology of the data manifold. The output of this stage—a hierarchical tree of coarse, piecewise-linear models—is then used to initialize and guide **Stage 2**. The second stage uses a more computationally intensive but theoretically rich model based on a simplicial complex and a "dual flow" of probability. The final result is not just a clustering, but a complete, queryable system that produces a high-fidelity, generative model of the manifold suitable for a wide array of machine learning tasks, from search and classification to data synthesis and interpretation.

## 2. Introduction: Beyond Clustering

Real-world data rarely conforms to simple structures. It often exists on a manifold with varying intrinsic dimensionality, significant non-linearity, and meaningful structures at multiple, nested scales. While many algorithms aim to solve parts of this problem, a truly comprehensive understanding requires a system that can:

1.  Efficiently discover the "natural" scales of the data without prior knowledge.
2.  Handle non-linear geometry and variable dimensionality.
3.  Produce a final model that is not just descriptive, but provides a rich set of operational primitives for a wide range of downstream tasks.

Proteus is designed to be such a system. While its core is a powerful hierarchical clustering engine, its ultimate purpose is to produce a final artifact that supports five fundamental modes of data interaction: **k-Nearest Neighbor Search**, **Constrained Search**, **Constrained Generation**, **Classification (Pattern Discovery)**, and **Stateful Feature Engineering**. While other specialized architectures will always have their place for specific, resource-constrained, or domain-specific problems, these five primitives provide a powerful, general-purpose foundation for tackling most machine learning tasks. This document describes the architecture of the Proteus learning engine that makes this possible.

## 3. The Proteus Architecture: A Two-Stage Pipeline

The core of the Proteus framework is a recursive pipeline composed of two primary stages.

### 3.1. Stage 1: Fast Exploration and Scaffolding

The goal of this stage is speed and scalability. It uses a highly-optimized, node-centric GNG architecture (detailed in `stage_1.md`) to rapidly explore the data.

- **Process:** A meta-learning loop couples the GNG with a Bayesian Optimizer to efficiently search the scale-space and identify the scales at which the data's clustered structure changes most significantly.
- **Output:** The result of Stage 1 is a **hierarchical scaffold**. This is a tree of GNG models, where each branch represents a recursive call on a data partition that has been projected onto a linearized subspace. This scaffold provides a complete, piecewise-linear "skeleton" of the data manifold at all its characteristic scales.

### 3.2. Stage 2: High-Fidelity Refinement and Generative Modeling

The goal of this stage is descriptive power and theoretical elegance. It takes the scaffold produced by Stage 1 as its input and "fleshes it out" by building a much richer geometric and probabilistic model. This process is detailed in `stage_2.md`.

- **Process:** For each GNG model in the hierarchical scaffold, Stage 2 materializes a full **simplicial complex**. It then uses a more sophisticated, physically-motivated learning rule and a novel **"dual flow"** mechanism to refine the model.
- **Output:** The final output of the full Proteus pipeline is a high-fidelity, generative model of the data manifold, suitable for a wide range of ML tasks.

## 4. Core Theoretical Concepts: The Simplicial Model and the Dual Flow

The high-fidelity model used in Stage 2 is built on two core theoretical concepts that are too computationally expensive for the initial exploratory stage, but provide immense descriptive power for the final refinement.

### 4.1. The Simplicial Complex

Instead of just nodes and links, the Stage 2 model represents the manifold as a **simplicial complex**—a collection of `d`-dimensional simplexes that form a continuous, volumetric tiling of the data space. This provides a much richer geometric foundation than a simple graph.

### 4.2. Physically-Motivated Learning

The learning rule in Stage 2 is based on a physical analogy of energy partition. The error from a data point is treated as an "impulsive force." The system's response is determined by its **Rigidity**, a measure of local node stability.

- In **plastic** regions (low rigidity), the energy is converted to kinetic energy, resulting in geometric adjustments (moving nodes).
- In **rigid** regions (high rigidity), the energy is stored as potential energy, building up internal structural stress that eventually leads to a topological split.

### 4.3. The Dual Flow and the Divergence Theorem

This is the most significant theoretical contribution of the framework. The Stage 2 model tracks not only the primal flow of nodes moving in space, but also a **dual flow** of probability across the faces of the simplexes.

- **Mechanism:** The error vector from each data point is projected onto the face normals of its containing simplex, creating a measure of "pressure" on each face.
- **Theoretical Grounding:** By tracking the net pressure (flux) on every face, we are, by the **Divergence Theorem**, capturing all the necessary boundary information to reconstruct the full, continuous probability gradient field within each simplex.
- **Capability:** This explicit modeling of the probability gradient is what elevates the final model from a descriptive one to a **generative** one. It allows for high-fidelity data synthesis, provides robust decision boundaries for classification, and enables precise probability density queries.

## 5. From Model to Primitives: The Five Core Capabilities

The final output of the Proteus pipeline is a queryable system and a set of feature representations built to support five key operations, making it a comprehensive tool for data analysis and machine learning.

1.  **k-Nearest Neighbor Search:** By generating a 1D semantic key for each data point (via the integrated Omega SFC framework), the system enables extremely fast and accurate similarity search.
2.  **Constrained Search:** By generating a fuzzy membership vector for each point's position in the hierarchy, the system can answer complex logical queries using a high-performance inverted index.
3.  **Constrained Generation:** The system can synthesize new data points that conform to a set of fuzzy predicates by finding a valid starting point and then performing an informed random walk using the learned generative model.
4.  **Classification (Pattern Discovery):** In a "reverse query," the system can take a set of data points and analyze their common membership vectors to produce a human-readable, fuzzy predicate description of what makes them a coherent group.
5.  **Stateful Feature Engineering:** The outputs themselves (`1D_key` and `membership_vector`) serve as powerful, semantically rich feature vectors. Using these as inputs for external models (e.g., classifiers, regressors) can dramatically improve their performance compared to using the raw data alone.

## 6. Conclusion

The Proteus framework, with its two-stage, recursive architecture, resolves the classic trade-off between speed and fidelity in manifold learning. By using a fast engine to first discover the structural "scaffold," it can then deploy a more powerful simplicial model to build a final, high-fidelity, generative representation. This hybrid approach, combined with the final query engine, provides a complete solution for the analysis of complex, multi-scale, non-linear data.
