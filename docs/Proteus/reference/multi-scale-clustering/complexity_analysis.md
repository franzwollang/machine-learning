# Proteus Computational Complexity Analysis

This document provides an approximation of the computational complexity (in time and space) for each stage of the Proteus algorithm.

**Variable Definitions:**

- `N`: Number of data points in the input.
- `d`: Ambient dimensionality of the data.
- `d_intrinsic`: True intrinsic dimensionality of the manifold in a specific region of space.
- `N_nodes`: Number of nodes in a GNG model. Typically `N_nodes << N` due to implicit entropy maximization objective--likely some sort of log or polylog function of the entropy of the distribution.
- `k_neighbors`: Number of nearest neighbors for GNG updates.
- `d_avg`: Average degree of a node in the GNG graph.
- `N_c`: Number of GNG nodes within a single cluster.
- `d_c`: Intrinsic dimensionality of a single cluster's manifold.
- `d_cand`: Average size of the candidate neighbor set for clique finding, after geometric and topological pruning.
- `N_data_c`: Number of raw data points belonging to a cluster.
- `N_simplices`: Number of simplices in a Stage 2 model.
- `N_clusters`: Number of clusters found at a given level (becomes the dimension of the membership vector).
- `N_predicates`: Number of predicates discovered by ANFIS.
- `R`: Number of recursive meta-learning runs (Modes 1 & 2).
- `N_T`: Number of time blocks in a temporal run (Modes 3 & 4).

---

## Stage 1: Fast Manifold Search Engine

Stage 1 is designed for speed, operating primarily on a lean GNG graph structure.

### Time Complexity

- **GNG Learning (per epoch):** `O(N * d * k_neighbors * log(N_nodes))`
  - The dominant cost is the k-NN search for each of the `N` data points against the `N_nodes` GNG nodes. This assumes an efficient ANN index like HNSW (`O(log(N_nodes))`). Vector operations add a `* d` factor.
- **Topological Management (periodic):** `O(N_nodes * d_avg^2)`
  - Link and node pruning involves statistical tests (t-tests) on a node's local neighborhood. This cost is incurred periodically, not per data point, so its amortized cost is low.
- **Scale-Space Analysis (per convergence):** `O(N_nodes * d_avg * log(N_nodes))`
  - The Chi-Squared test is fast (`O(N_nodes)`). The graph clustering is the main cost. While LPA is near-linear, the more robust Leiden algorithm is the bottleneck for this phase.
- **Recursive Subspace Definition (per cluster):** `O(d * N_c^2 + N_data_c * log(N_data_c))`
  - **PCA:** Performed on the `N_c` nodes of a cluster. The complexity depends on the relationship between the number of nodes `N_c` and the current dimensionality `d`:
    - **High-to-Low (`d > N_c`):** This is common at the root of a learning stage (e.g., the first level of Stage 1 or the start of a Stage 4 meta-run). Here, an efficient SVD-based approach is used with complexity `O(d * N_c^2)`.
    - **Low-d Refinement (`d < N_c`):** In deeper recursive steps, the dimensionality is the smaller intrinsic dimension `d_c` found previously. Here, `d_c` is typically smaller than `N_c`, and the complexity of the standard method is `O(N_c * d_c^2)`.
      The `d > N_c` case represents the primary bottleneck, so we analyze its impact in Stage 4.
  - **Distance Correlation:** Performed on the raw data points within the cluster to check for non-linearity. This is a fast `O(N_data_c * log(N_data_c))` operation.
  - **Normalizing Flow Training:** Training the `Glow` model on `N_c` nodes is efficient. The cost is `O(N_c * d_c * C_Glow)` where `C_Glow` is a constant related to the network's architecture (depth, layers).

### Space Complexity

- **Overall:** `O(N_nodes * d + N_nodes * d_avg)`
  - Dominated by storing the `N_nodes`. Each node stores its position `w`, nudge `n`, and stress `s` vectors (`O(N_nodes * d)`). The links add `O(N_nodes * d_avg)`. The ANN index also requires space, roughly `O(N_nodes * d * log(N_nodes))`.
  - Critically, space complexity is **independent of `N`**, the number of raw data points.

---

## Stage 2: High-Fidelity Refinement

Stage 2 upgrades the model to a full simplicial complex, increasing geometric fidelity but also computational cost. It is a "warm start" and skips the initial Stage 1 GNG creation.

### Time Complexity

- **Initial Seeding (Warm Start):** `O(N_c * d_cand^2 * d_c)`
  - The goal of the warm start is to massively accelerate Stage 2 convergence. A fast, quadratic-cost heuristic (the "Greedy Chaining" method) is used to discover the vast majority (hopefully ~99%) of the simplices.
  - This avoids any exponential complexity while being sufficient for the self-healing learning loop.
- **Simplex-Native Learning Loop (per epoch):** `O(N * d * log(N_nodes) + C_topology)`
  - The primary cost is the per-epoch learning loop. The key is that this loop is **self-healing**.
  - **Dynamic Topology Management (`C_topology`):** Data points that fall into "holes" (where simplices are missing) will trigger GNG-style link creation between the boundary nodes. This, in turn, can dynamically materialize the missing simplex. This amortizes the cost of finding the final few, difficult simplices over the entire learning process.
- **Geometric Audit (Torsion Check):** `O(N_simplices * (d_c+1)^2 * d)`
  - This cost grows as the simplicial complex is dynamically discovered. It is performed periodically.
- **Belief Propagation (final convergence):** `O(N_simplices * d_c^2)`
  - Runs on the dual graph of simplex faces. The size of this graph is `O(N_simplices * d_c)`. The cost is polynomial in the size of this dual graph.

### Space Complexity

- **Overall:** `O(N_nodes * d + N_simplices * d_c)`
  - The node storage is the same as Stage 1. The primary new cost is storing the `N_simplices` objects. Each simplex stores pointers to its vertices and other state, with a size proportional to `d_c`.

---

## Stage 3: Indexing and Query Engine Construction

This is a one-time build process that processes the entire dataset to create the final, queryable artifacts.

### Time Complexity

- **Primary Cost - Output Generation:** `O(N * (d*d_c + d_c*C_NF))`
  - This is the most significant cost in Stage 3. Every one of the `N` data points must be routed through the hierarchy, projected via PCA into a `d_c`-dimensional subspace (`O(d*d_c)`), and then passed through a Normalizing Flow (`O(d_c * C_NF)`).
- **Index Construction:**
  - Wavelet Tree: `O(N log(key_range))`, effectively `O(N)`.
  - Tag Index (Bitmaps): `O(N * N_tags)`.
  - Inverted Index: `O(N * N_clusters)`.
- **Query Time:**
  - **kNN:** `O(log N)` for the wavelet tree, plus the cost of processing the query point itself. Extremely fast.
  - **Constrained Search:** `O(result_size)`. Dominated by the final intersection of results from the highly optimized bitmap and inverted indexes.
  - **Fast Constrained Synthesis:** `O(N_walks * N_steps * (d_c*C_NF_inv + d_c*d))`. A multi-walk random walk, where each step involves an inverse NF transform and projection back to the ambient space.
  - **Unbiased Constrained Sampling:** Slower than the random walk. Dominated by the SFC pruning step: `O(N_sfc_leaves * N_nodes_in_terminal)`.

### Space Complexity

- **Overall:** `O(N * (N_clusters + N_tags) + Index_Overhead)`
  - The dominant space cost for the final artifacts is the "Document Table" that stores the `1D_key` and `membership_vector` for all `N` items (`O(N * N_clusters)`).
  - The Tag Index also requires `O(N * N_tags)` space, though this is highly compressed with Roaring Bitmaps.
  - The Wavelet Tree and Inverted Index have their own overhead, typically proportional to the data size `N`.

---

## Stage 4: Meta-Learning Architectures

Stage 4 runs the entire Proteus pipeline recursively. The complexity here is a function of the complexity of the previous stages.

### Mode 1 & 2: Geometric & Conceptual Learning

- **Time Complexity:** The key bottleneck is the growth of dimensionality in the recursive runs.
  - The input for `Run_i` is the `N_{nodes}` from `Run_{i-1}`. `N` for the next stage becomes small very quickly.
  - However, the dimension `d` for `Run_i` becomes `N_clusters` from `Run_{i-1}` (plus `N_predicates` in Mode 2).
  - The **PCA step** at the root of this new run becomes the bottleneck. This is a high-`d` / low-`N_c` scenario, where `d` can be 1k-10k or higher, while `N_c` is the number of nodes in a root cluster.
    - **Solution:** This step is made tractable by using an efficient **Randomized SVD**-based PCA (`e.g., sklearn.decomposition.PCA(svd_solver='randomized')`). This avoids the `O(d^2)` memory cost of a full covariance matrix and is significantly faster than standard SVD.
  - ANFIS training in Mode 2 adds a standard deep learning training cost at each recursive step.

### Mode 3 & 4: Temporal Learning

- **Time Complexity:** The same dimensionality bottleneck from Modes 1 & 2 appears, but now at the higher levels of the temporal tree.
  - The `Warm Start` chain is efficient, roughly `(Cost_{Stage2-3}) * N_T`.
  - The `Bottom-Up Temporal Tree` merge steps are where the cost is high. A `Level 1` run on `T_1` and `T_2` uses `N = N_{nodes,1} + N_{nodes,2}` and `d = |clusters_1 U clusters_2|`.
  - Again, the **root PCA step** of this new temporal run is the dominant cost. It is handled with the same Randomized SVD approach to make it feasible.
- **Global Backpropagation (Mode 4):**
  - The cost of one global fine-tuning pass is `O(Total_Params)` of the unrolled network. While the document argues the "effective depth" is much smaller than the theoretical depth, this is still a massive computation, likely reserved for offline, periodic refinement cycles.
  - **Conceptual Drift Cascade:** The cost of a warm re-run is `Cost_{Stage2-3}`. In a worst-case scenario, a change at the root could cascade through the entire tree, but this is expected to be rare as the system stabilizes. The amortized cost is lower.

### Space Complexity (All Modes)

- The space complexity is dominated by holding all the models (GNGs, NFs, ANFIS nets) for every node in the recursive and/or temporal hierarchy. The memory footprint can become very large, requiring a distributed storage solution.

---

## Production Artifacts Space Complexity

This section summarizes the space complexity of only the lean, queryable "production" artifacts detailed in Stage 3, distinct from the much larger "archival" model state. The total footprint is a sum of data-dependent and model-dependent components.

- **Data-Dependent Storage:** `O(N * (N_clusters + N_tags))`

  - This is typically the dominant component and scales linearly with the number of data items `N`.
  - **Document Table:** `O(N * N_clusters)` to store the final membership vector for each of the `N` items. This is a mandatory component as it's the source data for building the other primary indexes.
  - **Tag Index (Bitmaps):** `O(N * N_tags)` (Optional Speed-up) in the worst case for the bitmap index that maps tags to items, though this is heavily compressed in practice (e.g., via Roaring Bitmaps).
  - **Other Indexes:** The primary Wavelet Tree and the Inverted Index also scale roughly with `O(N)`.

- **Model-Dependent Storage:** `O(Model_Complexity)`
  - This component's size depends on the geometric complexity of the learned manifold, not the number of raw data points `N`.
  - **Normalizing Flow Models:** The storage for the parameters of all learned `Glow`/`NSF` models.
  - **Omega SFC Recipe Trees:** Each terminal cluster has its own recipe tree. The size of a single tree is proportional to the number of GNG nodes in that cluster's map (`O(N_c)`). The total storage is the sum across all terminal clusters, which scales with the total number of GNG nodes at that level, `O(N_nodes)`.
  - **Routers & Metadata:** Space for the LSH routers, ECDF maps, tag definitions, and hierarchy logic. This is typically small compared to the other components.
