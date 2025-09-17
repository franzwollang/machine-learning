# Stage 1: Fast Scale-Space Search

## 1. Abstract and Design Philosophy

This document provides a complete, stand-alone specification for the "Stage 1" component of the Proteus algorithm. The primary design goal of this architecture is **maximum speed and scalability**. Its sole purpose is to rapidly explore a high-dimensional dataset's scale-space to discover its natural characteristic scales.

This architecture is a pure search engine. It makes a deliberate and strategic trade-off, sacrificing all geometric and topological refinement capabilities for a massive gain in computational efficiency. Its output is simple: one or more optimal control parameters, `s_control*`, that correspond to characteristic scales of the data. This information is then passed to the high-fidelity Stage 2, which performs the actual manifold construction and refinement at the specific physical scales (`τ*`) derived from `s_control*`.

## 2. High-Level Algorithm Overview

The algorithm is a recursive, two-phase meta-learning loop that searches for `s_control*`.

1.  **Phase 1: Manifold Probing at a Given Scale**. A highly streamlined, node-based Growing Neural Gas (GNG) is exposed to the data. Its operations are optimized for speed to quickly build a statistical summary of the manifold at a single, fixed scale, governed by the control parameter `s_control`. The GNG's growth is adaptive, using a _local_ growth threshold `τ_local_i` for each node.
2.  **Phase 2: Scale Adaptation**. The GNG runs until it reaches thermal equilibrium. The resulting network state is analyzed to produce a single value, `Φ(τ_global)`, quantifying the structural information content at that scale against a _global_ threshold. A Bayesian optimizer uses this value to guide its search for the `s_control` that maximizes `Φ`.

## 3. Data Structures and State

The state is minimized to only what is necessary for a fast, statistically grounded search.

```
// --- Algorithm Parameters ---
k_neighbors: int         // Default: 8. Number of nearest neighbors for updates.
p_value_threshold: float    // Default: 0.05. Confidence level for statistical tests.
s_control: float            // Optimizer's normalized control parameter [0,1).

// --- Data Structures ---
Node {
    w: vector[d_embedding]    // Position vector
    m: vector[d_embedding]    // EWMA of residual error (mean)
    s: vector[d_embedding]    // EWMA of squared residual error (second moment)
    a: vector[d_embedding]    // Deferred nudge accumulator
    u: vector[d_embedding]    // Dominant eigenvector of local error covariance

    hit_count: float          // Weighted BMU win count
    update_count: int         // Number of statistical updates
    links: list[Link*]
    N_creation: int           // The value of N_total at the moment of creation
}

Link {
    node1, node2: Node*
    C12: float, C21: float    // Directed, weighted hit counters
    activity_snapshot1: float, activity_snapshot2: float
}

GlobalState {
    Nodes: list[Node]
    Links: list[Link]
    ANN_Index: HNSW_Index // For fast k-NN search
    s_control: float      // The current control parameter under evaluation
    D_subspace: int       // The dimensionality of the current sub-problem
}
```

**Justification:** The data structures are designed to support a physically-grounded statistical learning rule. The `m` and `s` vectors track the first two moments of the local error distribution, from which variance (`σ² = s - m²`) and incoherence (`ρ = ||m|| / σ`) can be derived. This is superior to heuristic error vectors as it provides a direct, physical analogy: `m` represents a local "heat current" or drift, while `σ²` represents "temperature." The deferred nudge accumulator `a` ensures the network's geometry only changes in response to signals coherent with the current scale `τ`, preventing high-frequency noise from destabilizing the mesh. The `u` vector, tracking the principal error direction via Oja's rule, provides a principled, data-driven direction for topological growth instead of relying on a single, noisy error vector.

## 4. The GNG Learning Component

### 4.1. The Four-Tier Scale Representation

The system uses a four-part representation of scale to separate optimizer concerns from the adaptive, local physics of the GNG.

1.  **Control Parameter `s_control`**: A normalized parameter `∈ [0,1)` that the optimizer proposes.
2.  **Subspace Dimensionality `D_subspace`**: The dimensionality of the current manifold or sub-problem.
3.  **Global Growth Threshold `τ_global`**: A single threshold, `τ_global = -D_subspace * log(1 - s_control)`, used _only_ to compute the scale-space response function `Φ(τ_global)` for the optimizer's evaluation.
4.  **Local Growth Threshold `τ_local_i`**: A per-node/simplex threshold, `τ_local_i = -d_final_i * log(1 - s_control)`, used _inside_ the GNG learning loop for the splitting condition (`σ_i² > τ_local_i`). `d_final_i` is the smoothed local intrinsic dimension at node `i`.

### 4.2. The Unified Statistical Update

For each data point `\mathbf{x}`:

1.  **Find Neighborhood:** The `k_neighbors` nearest nodes are found.
2.  **Distribute and Apply Updates:** For each node `j` in the neighborhood, weighted by `w_j = 2^-(j-1)`:
    - Calculate error: `\mathbf{e} = \mathbf{x} - \mathbf{w}_j`.
    - Update moments: `\mathbf{m}_j \leftarrow (1-αw_j)\mathbf{m}_j + αw_j\mathbf{e}` and `\mathbf{s}_j \leftarrow (1-αw_j)\mathbf{s}_j + αw_j\mathbf{e}^2`.
    - Update principal error direction `u_j` with an Oja's rule step.
    - Update nudge accumulator `a_j` based on the derived incoherence `ρ_j`.
    - Increment `hit_count_j` by `w_j`.

### 4.3. Principled Action Thresholds

1.  **Geometric Adjustment (Deferred Nudge):** A node `i` moves (`\mathbf{w}_i += \mathbf{a}_i`) only when `||\mathbf{a}_i|| > δ_{min}`, a sub-resolution threshold derived from the grid spacing of `τ_global`.
    - **Derivation:** The geometric scale grid uses a ratio `r ≈ 0.71`. The smallest meaningful displacement at a given scale `τ` is the distance to the next finer scale, which is proportional to `(1-r)√τ`. The threshold `δ_{min} = κ * (1 - r) * √τ` (with `κ=0.5`) sets a safe, principled limit, ensuring that mesh updates are only performed when the accumulated change is large enough to be resolved by the current scale analysis.
2.  **Topological Split:** A node `i` splits when its local variance exceeds its local cap: `σ_i² > τ_local_i`. The new node is positioned at `\mathbf{w}_k = \mathbf{w}_i + \mathbf{m}_i`, and new links are created based on the principal error direction `u_i`.

## 5. Topological Management

- **Node Creation:** As described above, triggered by `σ_i² > τ_local_i`.
- **Link Creation:** A link is created between `BMU_1` and `BMU_2` if one doesn't exist.

### 5.1. Link Pruning: The Power-Aware Statistical Gauntlet

The process for pruning links is a full, unabridged mechanism for robustly removing connections.

1.  **Link Creation & State:** A link `(i, j)` is created between `BMU_1` and `BMU_2` if one doesn't exist. The directed counter `C(i → j)` is incremented, and snapshots of node activities are stored.
2.  **Power Analysis (The "Fair Chance" Test):** A link is shielded from pruning until its effective sample size (`n_eff`, the sum of its weighted hits) is large enough for a fair statistical judgment. This is determined via a power analysis for proportions.
3.  **Relative Performance Test:** A node `i` evaluates its "tenured" links (those that passed the power analysis). Instead of a t-test, it uses the more robust **Wilson score interval**. It computes the one-sided 90% confidence upper bound (`ν⁺`) for each link's **Asymmetric Relative Significance**. A link is voted for pruning if its `ν⁺` is less than the median significance of its peer links.
    - **Justification:** The Wilson score is superior to a t-test for this task as it is an "exact" test for binomial proportions that does not assume normality and is accurate even for very small sample sizes (i.e., low-hit links), making it more reliable early in the learning process.
4.  **Bilateral Agreement:** A link is only removed if it is "tenured" and receives a prune vote from **both** of its endpoints.

### 5.2. Node Pruning via a Two-Stage Statistical Gauntlet

A "zombie" node `i` is pruned only when its local neighborhood has stabilized and after it has passed two refined statistical tests.

- **Trigger:** The test is activated only if the local neighborhood is stable. This is determined via a **two-sample, one-tailed Welch's t-test** on the _neighbor-normalized incoherence scores (`ρ̃`)_.
  - **Justification:** Using the same `ρ̃` metric that the thermal convergence test uses makes the trigger more consistent and robust, especially in regions of non-uniform data density.
- **Stage 1 (Fair Chance):** A Poisson arrival rate test. A node has had a fair chance if its estimated win rate `λ_est = hit_count / lifetime` is greater than a threshold derived from the system's deferred-nudge resolution (`≥ 2 / δ_min`).
- **Stage 2 (Relative Performance):** If the node has had a fair chance, it is pruned only if it satisfies a strict **AND-rule**: its `hit_count` is less than half the mean of its neighbors' hits, AND its local variance `σ_i²` is less than half the local growth threshold `τ_local_i`.
  - **Justification:** This composite rule is critical. It ensures the system only prunes nodes that are both low-mass (unimportant) and already in well-linearized regions (low variance). Pruning a node that is low-mass but still has high variance would risk creating a hole in a part of the manifold that is still actively being learned and flattened.

## 6. Scale-Space Analysis and Adaptation

The GNG component serves as the engine for the higher-level meta-learning loop that finds `s_control*`.

1.  **Convergence Trigger (Thermal Equilibrium):** For a given `s_control`, the GNG runs until the network reaches thermal equilibrium (`CV(ρ̃) < 0.01`).
2.  **Scale-Space Response Function `Φ(τ_global)`:** Once converged, a single value, `Φ(τ_global)`, is computed to quantify the structural information against the global scale threshold.
3.  **Bayesian and Grid-Based Optimization:** The system uses a hybrid strategy to find the maximum of `Φ`.
    - **Coarse Grid Search:** The optimizer evaluates `Φ` on a coarse, geometric grid of `s_control` values to find a bracket around a local maximum.
    - **Bayesian Refinement:** A Gaussian Process-based Bayesian optimizer performs a fine-grained search in `log(τ_global)` space _only within this bracket_ to find the optimal `τ_global*` (and thus `s_control*`).

## 7. Conclusion

This node-centric architecture is a fast, scalable, and robust engine for Stage 1. It does not produce a refined manifold model itself. Instead, it efficiently generates the single most important input required by Stage 2: a set of one or more characteristic control parameters, `s_control*`, from which the physical scales (`τ*`) for high-fidelity refinement are derived.
