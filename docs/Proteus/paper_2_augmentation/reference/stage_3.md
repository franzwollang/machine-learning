# Stage 3: Indexing and Query Engine Construction

## 1. Abstract

This document specifies the final stage of the Proteus-Omega framework, where the fully-refined, high-fidelity model from Stage 2 is used to construct a set of powerful, compact, and queryable artifacts. This stage creates two primary outputs for each data point: a **1D semantic key** for locality-based queries, and a **fuzzy membership vector** for predicate-based queries. It then builds a suite of state-of-the-art indexes—including Omega SFCs, Wavelet Trees, and Inverted Indexes—to create a comprehensive, high-performance query engine capable of a wide range of analytical tasks.

## 2. The Final System Artifacts: Production vs. Archival

The output of the full pipeline is separated into two distinct sets of artifacts: a lean set of "production" components required for efficient querying, and a richer set of "archival" components used for analysis, upgrades, and debugging.

### 2.1. Production Artifacts

This is the minimal, high-performance query engine deployed for serving requests.

1.  **A "Document Table":** A master key-value store mapping each `item_id` to its `(1D_key, terminal_cluster_id, membership_vector)`. This is a mandatory component, as it provides the source data required to build the primary kNN and fuzzy indexes.
2.  **A Hierarchy of LSH Routers:** A lightweight, functional representation of the GNG hierarchy used for fast routing of new data points.
3.  **A Collection of Normalizing Flow Models (`F_i`):** A heterogeneous collection of trained flow models. Each model `F_i` is associated with a flag indicating if it was trained on raw or ECDF-transformed coordinates.
4.  **A Collection of ECDF Transformation Maps:** For each cluster that used a "warm start" in Stage 1, this stores the learned ECDF transformation for each dimension. This artifact is required for processing new data points.
5.  **A Collection of Omega SFC Recipe Trees (`SFC_i`):** Built from the final dual flow densities.
6.  **A Primary Wavelet Tree:** Built on the `item_id`s, ordered by their 1D keys. This provides the core semantic locality index.
7.  **A Tag Index (Compressed Bitmaps):** (Optional Speed-up) A highly compressed bitmap index (e.g., using Roaring Bitmaps) that stores the sparse matrix of `(item_id, tag_id)` assignments. This allows for hyper-fast set operations on the tags but is not strictly required for the system to function.
8.  **An Inverted Index:** Built on the membership vectors for fast fuzzy predicate search (e.g., range queries). Without this index, such queries would require a full scan of the Document Table.
9.  **A Tag Definition Store:** A mapping from a `tag_id` to its defining set of fuzzy predicates, which have been **converted to crisp rules via a threshold** (e.g., `Tag_A := P_1(x) > 0.8`).
10. **A Hierarchy Logic Store:** A representation of the cluster hierarchy (e.g., as a DAG or parent/ancestor tables) used by the Query Planner to validate queries.

### 2.2. Archival Artifacts

This is the complete, stateful model of the learned manifold, not required for standard queries but essential for deeper analysis or future re-training.

1.  **The Complete GNG/Simplicial Hierarchy:** The full tree of `Cluster_Node` objects.
2.  **The Full Node and Simplex State:** All state variables from the Stage 2 model.

## 3. The Build Process

This describes the step-by-step construction of the final query engine from the refined Stage 2 model.

1.  **Build LSH Routers:** A **Learned LSH Router** (e.g., Leech Lattice-based) is created for each non-terminal GNG map in the hierarchy.
2.  **Build Omega SFCs:** An Omega SFC recipe tree is built for each terminal cluster by interpolating from the final `face_pressures`.
3.  **Generate Outputs for All Data:** Every data point is processed through the pipeline to generate its `(1D_key, membership_vector)` pair.
4.  **Generate Atomic Tags:** For every cluster `C_i` in the learned hierarchy, a corresponding **Atomic Tag** is automatically created. Its definition is the simple, crisp predicate `membership_in_C_i > 0`. This provides a complete, baseline set of queryable tags for every structural component the system discovered.
5.  **Define and Pre-compute Composed Tags:** Higher-level, human-readable **Composed Tags** are then defined, typically by applying a threshold to a fuzzy Predicate Function discovered in Stage 4 (e.g., `Tag_ConceptA := P_A(x) > 0.8`).
6.  **Build Final Payload Indexes:**
    - A **Primary Wavelet Tree** is built on the `item_id`s, ordered by their 1D keys.
    - A **Tag Index** is created. For each tag (both Atomic and Composed), a compressed bitmap is generated, with a bit set for each `item_id` that possesses that tag.
    - An **Inverted Index** is built on the sparse membership vectors.

## 4. The Query Engine: A Four-Part System

The final system provides four primary query capabilities, enabled by a **Query Planner** that routes requests to three distinct index structures and one functional engine.

**Core Components:**

- **The Query Planner:** The entry point for all queries. It validates user-provided predicates against the **Hierarchy Logic Store** to detect contradictions (e.g., asking for membership in two mutually exclusive clusters) and simplify redundant constraints before passing the query to the execution engines.
- **The kNN Index (Wavelet Tree):** For 1D semantic locality searches.
- **The Tag Index (Roaring Bitmaps):** For crisp, logical predicate resolution.
- **The Fuzzy Index (Inverted Index):** For fuzzy, degree-of-membership queries.
- **The Manifold Engine (LSH Routers, NSFs, SFCs):** This is the functional core of the system. It is responsible for a) transforming new, unseen data points into their latent representations (`1D_key`, `membership_vector`), and b) synthesizing new data points from the learned manifold.
  - **Forward Transformation (Inference):** To transform a new point, it is first routed to a terminal cluster and projected into its `d`-dimensional PCA subspace. The engine then checks if that cluster's model requires an ECDF pre-processing step, applies it if necessary, and finally passes the result through the appropriate Normalizing Flow model (`F_i`).
  - **Inverse Transformation (Synthesis):** To synthesize a point from a latent coordinate, the engine applies the exact inverse of the forward pipeline: 1) apply the inverse Normalizing Flow, 2) apply the inverse ECDF transformation (a fast quantile lookup) if the model requires it, and 3) project the point from the PCA subspace back into the original data space.

The use of multiple, specialized index structures is a deliberate architectural choice. While the **kNN Index (Wavelet Tree)**, sorted by the primary 1D semantic key, is unparalleled for locality-based queries, it is inefficient for filtering on arbitrary attributes. The **Tag Index** and **Fuzzy Index** are therefore not redundant, but are essential for efficiently answering feature-based predicate queries. It is a common misconception that such inverted indexes are space-inefficient. A production-grade implementation leverages modern compression techniques to minimize their footprint. For the crisp **Tag Index**, **Roaring Bitmaps** provide exceptional compression for the sparse `(item_id, tag_id)` matrix. For the **Fuzzy Index**, floating-point membership scores are first **quantized** (e.g., to 8-bit integers), and the sorted lists of item IDs are then compressed by storing d-gaps using methods like **Variable-Byte Encoding**. This multi-index design ensures state-of-the-art performance across all primary query patterns (locality, crisp predicates, and fuzzy predicates) with high space efficiency.

### 4.1. k-Nearest Neighbor Search (kNN)

- **Uses:** kNN Index, Tag Index.
- **Procedure:** A query point `x_orig` is processed by the Manifold Engine to generate its `1D_key`. A 1D range query is performed on the **kNN Index (Wavelet Tree)** around this key to find the IDs of the nearest neighbors. The **Tag Index** can optionally be used to pre-filter the set of valid `item_id`s, ensuring the kNN search only considers items that satisfy a given crisp logical predicate.

### 4.2. Constrained Search (Fuzzy & Crisp Predicates)

- **Uses:** Query Planner, Tag Index, Fuzzy Index.
- **Procedure:** Answers complex queries that combine crisp logical predicates on **Tags** (e.g., `(Tag_A AND Tag_B)`) and fuzzy predicates on raw cluster membership (e.g., `(membership_in_C5 > 0.4)`). The query planner resolves the crisp part on the Tag Index, the fuzzy part on the Fuzzy Index, and intersects the results.

### 4.3. Fast Constrained Synthesis (via Random Walk)

- **Uses:** Tag Index, Fuzzy Index, Manifold Engine.
- **Goal:** To quickly generate a set of novel "prototype" data points that satisfy a complex set of user-defined constraints, even if no such item exists. This mode prioritizes speed and validity over statistical purity.
- **Procedure:** A multi-walk process that can generate points from a pure description. The core synthesis of each point is handled by the Manifold Engine's standard inverse transformation procedure.
  1.  **Identify Target Space:** The user provides a set of fuzzy and crisp constraints. The system identifies all clusters referenced in the fuzzy constraints (e.g., `(membership_in_C5 > 0.4) AND (membership_in_C6 > 0.4)` references `C5` and `C6`).
  2.  **Find All Viable Starting Points:** The system must find the best possible starting points by considering the perspective of each constrained cluster. It performs a **parallel optimization search**. For each referenced cluster `C_i`, it finds the point `x_i` that maximizes the probability density _within that cluster's local model_, while also satisfying the full set of user constraints. This results in a set of candidate starting points, `{x_1, x_2, ...}`.
  3.  **Assign to Best Maps and Initiate Walks:** For each candidate point `x_i`, the system determines its single "dominant" cluster by finding which cluster has the highest fuzzy membership value at that location. The random walk for `x_i` is then initiated on the local map of its determined dominant cluster, ensuring the walk always proceeds on the highest-fidelity model available. Even if multiple starting points are assigned to the same map, a separate walk is launched for each to ensure all potential entry points to the target region are explored. The final result of the query is the collection of all points from all walks.
      - **Rationale for Independent Walks:** This "embarrassingly parallel" approach is a deliberate design choice over more complex, synchronized methods (e.g., communicating walkers or PSO-like systems). While those systems can be powerful, they introduce significant implementation complexity, synchronization overhead, and a high risk of introducing new, hard-to-control sampling biases (e.g., a "repulsion force" between walkers would alter the underlying probability distribution being sampled). The independent multi-walk strategy provides a robust, scalable, and computationally efficient engineering solution that effectively solves the core problem of exploring all "arms" of a complex region without these added risks.
      - **Boundary Handling:** The random walk is intentionally constrained to the local manifold of its starting cluster to ensure the generated point continues to satisfy the query's constraints (e.g., `membership_in_C5 > 0.8`). The boundary with a neighboring cluster is not treated as a hard wall, but as a "probabilistic cliff." Since the walk is guided by the local probability gradient, and the density naturally falls off at the cluster's edge, the walk is intrinsically biased to turn back towards the higher-density regions of its own cluster. This prevents the walk from "falling off" the manifold of its source cluster. While this introduces a slight statistical bias against generating points at the extreme edges of a cluster, it is the correct trade-off to ensure the generated samples are always valid and the process is computationally efficient. Unbiased global sampling is handled by the hierarchical top-down sampler, not this constrained walk.

### 4.4. Unbiased Constrained Sampling (via Hierarchical Rejection)

- **Uses:** Tag Index, Fuzzy Index, Manifold Engine.
- **Goal:** To generate a single, statistically perfect sample from the distribution defined by a set of user constraints. This mode prioritizes statistical purity over speed.
- **Procedure:** This method uses the full hierarchical model to perform a constrained, top-down sampling pass. The final step of generating a point in a terminal cluster uses the standard inverse transformation pipeline defined by the Manifold Engine.
  1.  **Constrained Hierarchical Descent:** The system traverses the hierarchy from the root. At each level, it re-weights the probability of descending into a child cluster based on the total probability mass of the region satisfying the user's constraints within that child. To make this step computationally feasible, this mass is **approximated** by checking only the nodes of the child cluster's GNG map. The system sums the activity of all nodes whose positions satisfy the constraints, and uses this sum as a proxy for the constrained mass. This is a robust approximation that makes the process tractable.
  2.  **Guided Rejection Sampling:** When the descent reaches a terminal cluster, the system performs an optimized **guided rejection sampling**.
      a. **SFC Pruning:** Analytically identifying the precise geometric sub-volume defined by the user's constraints is often computationally intractable. Instead, the system uses a robust approximation to prune the Omega SFC tree. For each hyperrectangle in the SFC tree, it checks if any of the GNG nodes residing within that hyperrectangle satisfy the user's constraints. If a hyperrectangle contains at least one such "valid" GNG node, it is kept; otherwise, it is pruned. This pre-computation has a one-time cost proportional to the number of SFC leaf nodes multiplied by the number of GNG nodes, but it makes the subsequent sampling step vastly more efficient.
      b. **Targeted Sampling & Final Rejection:** The system then draws a sample from this much smaller, pruned SFC sub-tree. This serves as a highly efficient proposal, as the point is guaranteed to be in the correct vicinity. A final rejection check is still performed to handle cases where a sampled hyperrectangle only partially overlaps the true constraint region. This two-step process dramatically reduces the rejection rate, making the unbiased sampler efficient even for highly specific constraints, while the final check guarantees statistical purity.
- **Trade-off:** This process is significantly slower than the Fast Constrained Synthesis method but is guaranteed to produce a sample that is free of the boundary biases inherent in the random walk approach. It provides a "gold standard" generative capability when statistical rigor is paramount.

### 4.5. Classification (Pattern Discovery)

- **Uses:** Fuzzy Index, Document Table.
- **Procedure:** A "reverse query" where a user provides a set of items. The system retrieves their `membership_vector`s from the **Document Table** and uses the **Fuzzy Index** to find the common clusters and the distribution of membership scores. It then generates a statistical summary, such as: "This set has a mean membership of 0.9 in `Cluster C5` and 0.2 in `C5.A`."
