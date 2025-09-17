# Stage 4: Meta-Learning Architectures

## 1. Abstract

This document specifies the advanced operational modes of the Proteus framework. Stage 4 is not a single process, but a flexible architecture that can be configured in several ways to achieve different learning objectives. These configurations range from discovering higher-order geometric patterns in a static dataset to a state of perpetual learning that discovers and reasons about the temporal evolution of abstract concepts. This document details the different operational modes, from the simplest to the most complex.

The computational intensity of these deep, recursive learning loops is made feasible by the extreme efficiency of the underlying manifold learning process detailed in Stages 1-3. Specifically, the architecture's viability rests on two key optimizations: 1) all expensive geometric and probabilistic modeling is performed on the compressed GNG node summaries, not the raw data, and 2) the quality-gated marginal separation technique further accelerates training and minimizes the use of computationally expensive Normalizing Flow models. This efficiency at the core allows for the exploration of the powerful, higher-order learning cycles described here.

## 2. Mode 1: Timeless, Non-ANFIS Run (Geometric Meta-Learning)

This is the simplest meta-learning configuration, designed to discover "clusters of clusters" in a static dataset.

- **Mechanism:** The system runs recursively on its own output.
  1.  `Run_0` is executed on the full dataset `X_0`. This involves the complete Stage 1-2 pipeline, where a GNG map is first trained to summarize the data, and all subsequent analysis is performed on that compressed GNG node representation. This run produces the base membership vectors `MV_0`.
  2.  `Run_1` is then executed using `MV_0` as its input data.
  3.  This continues (`Run_2` on `MV_1`, etc.) until a run produces no new cluster structure.
- **Theoretical Inspiration (Constructive Spectral Decomposition):** The inspiration for this process comes from spectral methods in machine learning (e.g., Kernel PCA, Diffusion Maps). In these methods, the eigendecomposition of a kernel matrix reveals the hierarchical geometric structure of the data, analogous to how the singular values of a data matrix reveal its principal components.

  Our recursive process is an engineered, constructive analogue of this spectral decomposition:

  - **`Run_0`** on the initial dataset identifies the most dominant, large-scale geometric structures. This is analogous to discovering the structure associated with the largest singular values (the primary axes of variance). The output, `MV_0`, represents the data in a new basis defined by these primary geometric components.
  - **`Run_1`**, by clustering `MV_0`, analyzes the relationships and correlations _between_ these primary components. It seeks to find "clusters of clusters," which is analogous to resolving the next tier of the spectral hierarchy—discovering the more subtle geometric features associated with smaller singular values.

  Each recursive run thus "peels off" a layer of geometric structure at a given scale, allowing the next run to discover finer-grained patterns in the remaining structural information.

- **Purpose:** This mode discovers purely geometric, higher-order relationships.

While Mode 1 provides a powerful unsupervised method for discovering the pure geometry of the data via a process analogous to spectral decomposition, it operates without knowledge of any external objective. Mode 2 introduces a critical evolution by marrying this recursive geometric discovery with the supervised learning power of a Deep Neural Network (the ANFIS module). The purpose of this hybrid approach is to leverage the DNN's ability to perform what is essentially a vast combinatorial search, discovering the predictive patterns that map the rich geometric structure of the data to a specific, user-defined objective. It allows the system to move beyond just seeing the data's structure to understanding which parts of that structure are most relevant for a given goal.

## 3. Mode 2: Timeless, ANFIS-Enabled Run (Objective-Driven Conceptual Learning)

This is a powerful, specialized mode for discovering abstract concepts when a clear objective function can be defined.

### 3.1. Mechanism

1.  `Run_0` of the Proteus pipeline produces the base membership vectors, `MV_0`.
2.  An **ANFIS** process is run on `MV_0` to discover a set of Predicate Functions (`P_0`) that are predictive of a given objective.
3.  A new, enriched feature vector `X_1` is created by concatenating the membership vector with the continuous fuzzy outputs of the new predicates: `X_1 = [MV_0, P_0(MV_0)]`.
4.  `Run_1` of the Proteus pipeline is then executed on `X_1`. This loop can continue until no new valuable predicates are discovered.

### 3.2. The ANFIS Architecture

The ANFIS model uses a specific, deep architecture to make rule discovery computationally feasible.

- **The Objective Module:** A configurable module combines multiple, normalized objectives (`L_internal`, `L_supervised`, etc.) into a single, weighted **Global Objective Function (`L_global`)**. This is the scalar target value the ANFIS is trained to predict.
- **DNF-Inspired, Tree-Structured Network:** The network is structured to mimic DNF logic (`(A AND B) OR (C AND D)`). It has a first hidden layer of fuzzy `AND` neurons followed by a deep, tree-structured hierarchy of fuzzy `OR` neurons that is more stable to train than a single wide layer. To ensure stable gradient flow during the global backpropagation phase described later, the fuzzy `AND` operation is implemented with a t-norm that is less prone to vanishing gradients than the standard product t-norm (e.g., the Łukasiewicz t-norm).
- **Solving Combinatorial Explosion (L1 Regularization):** The `AND` layer is initially fully connected, but strong **L1 regularization** is applied during training. This forces the weights of unimportant inputs to zero, automatically pruning the connections and allowing the network to discover the simplest, most powerful conjunctive rules without a brute-force search.
- **Architectural Context and Alternatives:** This "grow-and-prune" approach, where a large, fully-connected layer is optimized with L1 regularization to discover salient input combinations, stands in contrast to other common deep neuro-fuzzy architectures like Hierarchical Fuzzy Inference Trees (HFIT) or Multi-Layer ANFIS (MLA-ANFIS). Those approaches typically adopt a "divide-and-conquer" strategy, composing multiple smaller, low-dimensional fuzzy subsystems in a tree or layered structure to preemptively manage the curse of dimensionality. The design choice in Proteus is to favor flexibility; rather than pre-defining input groupings, the L1-based pruning allows the system to _discover_ the most relevant feature interactions automatically from the data itself. This is feasible because the inputs to the fuzzy architure is already highly preprocessed into linearized subspaces.
- **Optimization and Predicate Extraction:** The network is trained with standard deep learning optimizers like **Adam**, leveraging techniques like **Dropout** for generalization. This end-to-end gradient-based approach offers great flexibility. However, the training methodology could be augmented by techniques proven effective in the neuro-fuzzy literature:

  - **Hybrid Learning:** For TSK-style rules, a hybrid algorithm—using gradient descent to tune the non-linear premise parameters (membership functions) and a more efficient, non-iterative method like **Least Squares Estimation (LSE)** to solve for the linear consequent parameters—can often lead to faster and more stable convergence.

  After training, the logical rules are extracted via a multi-step process:

  - **1. Extract Conjunctive Predicates:** The system iterates through each neuron in the first (`AND`) hidden layer. For each neuron, it identifies its non-zero input connections. The weights and biases of these connections define the fuzzy sets (e.g., `C5 is HIGH`), which are combined to form a simple conjunctive predicate (`P1 := (C5 is HIGH AND C9 is LOW)`).
  - **2. Extract Disjunctive Predicates:** The system performs a backward pass from the output neuron to identify the most influential paths through the deeper `OR` layers. For each of these paths, it recursively "unrolls" the tree of `OR` neurons back to their constituent `AND` clauses, forming a higher-order disjunctive predicate (e.g., `R1 := P1 OR P2`).
  - **3. Rank and Filter:** All extracted predicates (`P_i` and `R_i`) are ranked by an "Importance Score," which is calculated from the network weights along the path that forms the rule. This list is filtered to keep only the highest-quality candidates.

- **Promotion and Tag Creation:** The highest-quality candidate predicates are promoted.
  - **For Learning:** The raw, continuous-valued **Predicate Functions** are used to enrich the feature set for the next iteration of the meta-learning loop, preserving maximum information.
  - **For Querying:** A user or an automated process can then define a discrete **Composed Tag** by applying a crisping function (e.g., a threshold) to a promoted predicate. This new Tag is then added to the Stage 3 query engine's Tag Index.

The fuzzy outputs of the `P_new` function are not added to the production `membership_vector`s. They are used as features for the _next_ iteration of the meta-learning loop, which will in turn generate its own new geometric clusters and corresponding Atomic Tags.

## 4. Mode 3: Temporal Run without ANFIS (Manifold Dynamics Tracking)

This mode focuses purely on tracking the evolution of the manifold's geometry and cluster structure over time. It uses the full temporal pyramid architecture but does not run the ANFIS discovery process. Its primary output is a rich, historical model of the manifold's dynamics.

A core architectural principle of the temporal pyramid is **model persistence per timescale**. For any given level of the pyramid (e.g., the level that analyzes one-hour blocks of data), a single, persistent instance of the entire Proteus pipeline (including the Elastic Autoencoder, Proteus model, and ANFIS module) is maintained. This instance is run sequentially on each time block for its scale. This persistence is what enables the "warm start" mechanism, allowing the model to explicitly compare the new time block to its memory of the previous one and learn about the data's evolution.

### 4.1. Mechanism Part 1: The "Warm Start" Chain

The system processes a continuous stream of data by dividing it into fundamental time blocks (`T_1, T_2, ...`). To ensure continuity and explicitly track changes, it uses a "warm start" mechanism.

- **Cold Start (`Run(T_1)`):** The very first time block runs the full Proteus pipeline (Stages 1-2) from a random initialization, producing the first stable hierarchy, `H_1`.
- **Warm Start (`Run(T_2)` and beyond):** The Proteus run for the next time block, `T_2`, is **initialized with the converged hierarchy `H_1`** from the previous block. It **skips Stage 1** and begins directly with Stage 2's high-fidelity refinement. This is highly efficient and allows the system to directly compute a `dynamics_vector` by comparing `H_2` to `H_1`.

### 4.2. Mechanism Part 2: The Bottom-Up Temporal Tree

Higher-order, longer-term dynamics are discovered by recursively running new Proteus instances on the outputs of lower-level runs.

- **Level 1 Merge (`Run(T^1_1)`):** After the runs for `T_1` and `T_2` are complete, a new, higher-level Proteus instance is launched. This is a **cold start** (it begins with Stage 1) as there is no prior hierarchy at this temporal scale.
- **Unified Feature Space and Pruning:** The input for this new run is created by creating a unified feature space from the outputs of the two child runs. To prevent the conceptual space from becoming bloated with obsolete "fossilized" clusters over time, the **unified cluster basis** is created via a robust statistical pruning process:
  - **1. Track Last-Seen Timestamp:** As `cluster_id`s are collected from the child runs, each is stored with its `last_seen_timestamp`.
  - **2. Define Relevance Window:** The system uses a "relevance window," `W`, a multiple of the child run's timescale (e.g., if the children process 1-hour blocks, `W` might be 24 hours).
  - **3. Pruning Gauntlet:** Before a `cluster_id` is included in the new unified basis, it is tested. It is pruned (excluded) only if it fails both tests:
    - a. **The Grace Period Test:** The concept must be "tenured," meaning its `last_seen_timestamp` is older than the current relevance window `W`. This prevents the premature pruning of merely infrequent concepts.
    - b. **The Statistical Significance Test:** If the concept is tenured, a trend analysis (e.g., Mann-Kendall test) is performed on its historical activity. If the test shows a statistically significant, persistent decline leading to its disappearance, the concept is confirmed as obsolete and pruned.
  - **4. Final Basis Creation:** The final unified cluster basis is the set of all `cluster_id`s that survive this pruning gauntlet. The `membership_vector`s from the child runs are then projected onto this lean, relevant basis.
- **The Tree:** This process continues. The run for `(T_3, T_4)` creates `Run(T^1_2)`, which can be **warm-started** from the hierarchy of `Run(T^1_1)`. The outputs of `Run(T^1_1)` and `Run(T^1_2)` can then be merged in a new cold-start `Run(T^2_1)` (using the same pruning mechanism), building a deep, hierarchical understanding of the data's evolution.

## 5. Mode 4: The Full System (Perpetual Learning)

This is the most powerful operational mode, combining the temporal hierarchy from Mode 3 with the full ANFIS-driven discovery loop from Mode 2.

- **Mechanism:** This mode uses the exact same temporal pyramid architecture as Mode 3 (the warm-start chain and bottom-up tree). The key difference is that at **every step** of this process, the full ANFIS discovery architecture is executed.
- **Enriched Input:** The ANFIS at each step is trained on the full **Enriched Feature Vector**, which includes:
  1.  The unified **membership vectors** from the child runs.
  2.  The **dynamic features** (`dynamics_vector`) calculated by comparing the current hierarchy to its warm-start initialization.
  3.  The **predicate outputs** inherited from the ANFIS runs of the child processes.
- **Purpose:** To achieve a state of true perpetual learning. The system discovers not just static concepts but also **temporal predicates** that describe the evolution of those concepts, creating the deepest possible understanding of the data-generating process. It is expected that this will be the default and most common operational configuration. The ANFIS is guided by the system's **Global Objective Function**, which is a configurable module that acts as the central interface for all objective signals. While this module can be configured to include external supervisory or homeostatic signals, its true power comes from its ability to operate in a self-supervised manner by using the system's own intrinsic metrics as a target. At a minimum, this always includes the **Temporal Difference (TD) Error** from the Actor-Critic loop, which trains the ANFIS to discover concepts that are most relevant to improving the system's own predictive accuracy.

## 6. The Ultimate Operational Mode: A Convergent Cycle of Local and Global Learning

The modular, staged approach described above is a practical and powerful implementation. However, the architecture allows for a final, even more powerful operational mode in a scenario with sufficient computational resources. This mode treats the entire temporal and conceptual hierarchy as a single, unified, deep network. It operates on a rhythmic cycle of piecewise growth followed by holistic refinement, enabling the system to adapt its foundational knowledge based on high-level discoveries.

This deep, hierarchical structure is theoretically well-grounded. Foundational research in neuro-fuzzy systems has established that while even a single-layer fuzzy system is a universal approximator, deep and hierarchical fuzzy systems can be **exponentially more efficient** in terms of the number of rules required to approximate certain classes of complex, compositional functions. By unrolling the hierarchy, Proteus leverages this principle of **compositional representation**, where each layer learns concepts based on the outputs of the layer below, creating a powerful feature hierarchy analogous to those in deep neural networks, but with the benefit of retaining logical interpretability.

### 6.1. The Dynamically Unrolled Global Network Architecture

- **Concept:** At any point in time, the full hierarchy of ANFIS models from the temporal tree can be "unrolled" into a single, deep computational graph. When an ANFIS at `Run(T^1_1)` takes as input a predicate `P_1` learned from `Run(T_1)`, that input node is replaced with the entire sub-network that defines `P_1`.

- **Structure:** The resulting network is a deep neural network where depth corresponds to increasing abstraction along two primary axes. As a signal propagates forward from the input layers to the output:

  1.  **It ascends the temporal scale:** from the smallest time blocks (e.g., `T_1`) to progressively larger ones (e.g., `T^1_1`, `T^2_1`).
  2.  **It ascends the recursive tower:** from base geometric clusters to higher-order "clusters of clusters" and abstract predicates.
      The final output layer corresponds to the highest level of the recursive tower at the largest temporal scale, representing the system's most abstract current understanding.

- **Feasibility and Effective Depth:** This dynamic structure is computationally feasible. The network's sparsity, inherited from L1 regularization, prevents a combinatorial explosion of connections. Furthermore, while the _potential_ depth of the unrolled graph is vast, the _effective computational depth_ is often much shallower. This is due to a "collapsing" effect driven by conceptual stability:
  - If a concept learned at a lower level is used by a higher level without modification, its underlying subgraph does not need to be unrolled for computation. The stable concept acts as a pre-computed "collapsed" node.
  - This results in a highly irregular topology of sparse, disjoint paths with **highly variable lengths**. The effective depth of any given path is determined by its chain of _evolving_ concepts, not the full theoretical hierarchy. This makes the global fine-tuning process far more efficient than the potential depth would suggest.

### 6.2. The Rhythmic Training Cycle: Piecewise Growth and Holistic Refinement

The network is not trained monolithically. Instead, it learns through a repeated, two-stage cycle that balances local discovery with global coherence.

1.  **Piecewise Growth:** When a new ANFIS layer is added (either by moving to a higher temporal scale or a deeper recursive level), it is first trained "locally." This local training is a two-step process to maximize efficiency:

    - a. **Rapid Initial Fit:** To establish a high-quality baseline, a fast, one-pass algorithm (e.g., Wang-Mendel style rule induction) can be used for an initial fit.
    - b. **Local Fine-Tuning:** This is followed by a local fine-tuning phase using standard backpropagation on the module's direct inputs.
      Once this local training is complete, the module's pruned logical **topology** is established. The connections that have been pruned to zero by L1 regularization are permanently locked, but the remaining non-zero weights are now ready for the global fine-tuning phase. This allows for the fast, targeted discovery of new "local" rules while preserving the ability for them to be adapted later in the global context.

2.  **Holistic Refinement:** After the new layer is initialized, the entire system can perform a **Global Backpropagation**. The network is unrolled, and the gradient from the global objective function is propagated all the way back to the earliest layers. The use of a gradient-stable t-norm is critical here to ensure the fine-tuning signal can effectively reach the foundational predicates. This fine-tuning subtly adjusts the parameters of previously-learned, foundational predicates based on the newest, most abstract discoveries.

This cycle repeats continuously: a new recursive or temporal layer is added, trained locally, and then the entire system is fine-tuned globally.

### 6.3. Managing Conceptual Drift: The Feedback Loop Between Logic and Geometry

This training dynamic creates a powerful but complex feedback loop: a change in a high-level logical predicate can necessitate an update to the low-level geometric structures that were built upon it. This is not a flaw, but the central mechanism for deep adaptation.

- **The Cascade Trigger:** A global fine-tuning pass may alter the function of a predicate `P_A`. However, a subsequent Proteus run (`Run_N+1`) may have used the _original_ output of `P_A` as an input feature to discover its own geometric clusters. The geometry of `Run_N+1` is now potentially invalid, as its input space has been retrospectively changed.

- **Targeted Re-evaluation:** This "conceptual drift" is handled not by constant, full re-runs, but by a targeted and damped update cascade.

  1.  **Drift Detection:** The system monitors for a statistically significant shift in the output distribution of a predicate function (e.g., via a Kolmogorov-Smirnov test) after fine-tuning.
  2.  **Warm Re-run:** If the drift exceeds a stability threshold, the system triggers a **warm re-run** of only the dependent Proteus instances. These runs skip Stage 1 and begin directly with Stage 2 refinement, using their old geometry as a "warm start" but now using the new, updated predicate outputs as input.

- **Convergence:** This process is naturally convergent. Early in the system's life, updates may be large and cause significant cascades. Over time, as the system's understanding matures, the fine-tuning adjustments become smaller, conceptual drift lessens, and the cascades become rarer, leading to a robust and stable, yet perpetually adaptable, conceptual framework.

### 6.4. Benefits of the Global Architecture

- **End-to-End Fine-Tuning:** The primary benefit is that a discovery made at the highest level of temporal abstraction can flow all the way down and subtly adjust the fuzzy membership functions of the most foundational predicates learned at the earliest time steps. It allows the system to holistically fine-tune its entire conceptual vocabulary at once.
- **Preserved Interpretability:** This process does not destroy interpretability. The fundamental `IF-THEN` structure of the ANFIS components is preserved. The backpropagation only fine-tunes the weights and parameters of the existing rules, making them more accurate in the global context.

## 7. Operationalizing Discoveries: Integrating New Concepts into Stage 3

The discovery of new Predicate Functions in Stage 4 is not the end of the process. For a new concept to become a most efficiently queryable part of the system, it must be formally defined as a Tag and integrated into the Stage 3 query engine artifacts. This process applies to both the simple, geometric clusters and the complex, learned predicates.

1.  **Atomic Tag Creation:** After a Proteus run completes, the system automatically creates a crisp **Atomic Tag** for every geometric cluster `C_i` in the learned hierarchy. The definition is simple: `Tag_C_i := membership_in_C_i > 0`. This tag is added to the Tag Index, providing a baseline queryable vocabulary using the fast wavelet tree path.

2.  **Composed Tag Creation (Curation):** A user or an automated curation process can create a new, human-readable **Composed Tag** in one of three ways:
    - **1. Labeling an Atomic Tag:** The user can simply assign a human-readable alias (e.g., `"GoldenRetriever"`) to an existing, system-generated Atomic Tag (e.g., `Tag_C_i`). This is the most direct way to add semantic meaning to the learned geometric structure.
    - **2. Crisping a Learned Predicate:** The user selects a high-quality **Predicate Function (`P_new`)** discovered by Stage 4, gives it a name, and defines a crisping function (e.g., `Tag_Suspicious := P_new(x) > 0.8`).
    - **3. Defining a Manual Predicate:** A domain expert can write a new predicate from scratch, creating a logical combination of existing Atomic and Composed Tags.
    - In all cases, the new tag definition is added to the **Tag Definition Store**, and its corresponding bitmap is added to the **Tag Index**.

## 8. Architectural Implications

The complete Proteus-Omega framework, culminating in the perpetual learning architecture of Stage 4, represents a significant shift from traditional machine learning models. Its implications are far-reaching.

- **A Path to General Intelligence:** The architecture described is not domain-specific. Due to its input agnosticism and the generality of its learning mechanisms, it is a **General-Purpose Conceptual Learning Engine**. The **Objective Module** provides the system's goals. While these can be defined by a human operator to solve a specific task, they can also be defined in terms of the system's own internal state (e.g., maximizing knowledge novelty, minimizing computational cost, maintaining homeostasis for an embodied agent). By providing the system with intrinsic, self-referential objectives, this architecture provides a concrete roadmap for a truly autonomous and general form of artificial intelligence.

- **From Feature Engineering to Concept Curation:** The role of the human expert is elevated. Instead of performing painstaking feature engineering on raw data, the expert's primary task becomes the **curation of the system's discovered concepts**. They validate, name, and guide the growth of the system's conceptual vocabulary, acting as a teacher and partner to the learning agent rather than just a programmer.

- **Deep Interpretability (White-Box AI):** Despite its complexity, the system is fundamentally interpretable. The fuzzy memberships over clustering structure and the ANFIS architecture ensures that even the most complex, high-order, temporal relationships can be extracted and presented as a hierarchy of human-readable `IF-THEN` rules. This provides a level of transparency and trust that is impossible to achieve with "black-box" models like large transformers.

- **Computational Scale:** This architecture is predicated on the availability of significant computational resources. The continuous re-running and global backpropagation are expensive. However, it is a **principled expense**. The architecture is designed to be highly parallelizable and scalable, making it a "More's Law" system—its full potential will be unlocked by the continued growth of hardware capabilities. It represents a long-term investment in a system that can grow in intelligence indefinitely.

- **The Role of the Human:** The human's role shifts from being the sole source of high-level knowledge to being a **curator of the machine's discoveries**. They validate the system's proposed Predicate Functions, give them meaningful names, and define the crisping thresholds that turn them into queryable Tags.
