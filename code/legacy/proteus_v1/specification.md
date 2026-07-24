# Proteus System Specification

## 1. Abstract and Design Philosophy

The Proteus algorithm is a hierarchical, three-level system for discovering the geometric and topological structure of high-dimensional data through γ-normalized derivatives and scale-space optimization. The system is built on a rigorous mathematical foundation combining Lindeberg's scale-space theory with Growing Neural Gas (GNG) dynamics.

### 1.1 Mathematical Foundation

The core mathematical framework is based on γ-normalized derivatives from scale-space theory:

```
L_ξ^m η^n(x, y; σ) = σ^(γ(m+n)) ∂_{x^m y^n} L(x, y; σ)
```

Where γ ∈ [0,1] controls scale-covariant normalization. For γ = 1, the response is fully scale-covariant.

In the GNG context, each node accumulates hits h_i with local density estimate:

```
ρ̂_i = k/(N V_d r_{k,i}^d)
Vol_i = h_i/ρ̂_i = h_i V_d r_{k,i}^d / (k/N)
```

The normalized response becomes:

```
R_i(τ) = σ^(γd) * h_i/Vol_i
```

With τ = σ²/c\_{d,k}² and γ = 1, this gives scale-covariant detection.

### 1.2 System Architecture

The system operates at three hierarchical levels:

1. **Map Level**: Single-map operations on nodes, links, and simplexes
2. **Proteus Level**: Cross-map coordination and algorithm orchestration
3. **ProteusAlgorithm Level**: Public API and component management

### 1.3 Per-Map vs System-Level Operations

**Important**: Most operations, including nonlinearity handling, parameter budgets, and patch-NSF management, are handled **per-map** rather than system-wide. This ensures:

- Each map can have its own Glow model and parameter budget
- Patch-NSFs are allocated relative to the current map's Glow model size
- The 50% torsion coverage threshold triggers map-level NSF upgrade
- Maps can be at different stages of the upgrade sequence simultaneously

**Upgrade Sequence per Map**:

1. Map-level Glow (initial nonlinearity handling)
2. Attempt splitting and patch-NSFs (if < 50% torsion coverage)
3. Map-level NSF (if ≥ 50% coverage or parameter budget exhausted)

## 2. Core Mathematical Concepts

### 2.1 Four-Tier Scale Representation

The system uses a four-part scale representation:

1. **s_control ∈ [0,1)**: Normalized optimizer control parameter
2. **D_subspace**: Dimensionality of current manifold/sub-problem
3. **τ_global = -D_subspace \* log(1 - s_control)**: Global evaluation threshold
4. **τ_local_i = -d_final_i \* log(1 - s_control)**: Per-node adaptive threshold

### 2.2 EWMA Statistics with Principled Alpha

The EWMA weight α is derived from first principles:

```
α = ln(2) / k_neighbors ≈ 0.693 / k_neighbors
```

This sets the half-life equal to the neighborhood size, ensuring variance reaches τ within one neighborhood's worth of updates.

### 2.3 Thermal Equilibrium Interpretation

Each node's incoherence ratio:

```
ρ_i = ||m_i|| / (σ_i + ε)
```

Acts as a local heat current. The neighbor-normalized measure:

```
ρ̃_i = ρ_i / (mean_{j∈N(i)} ρ_j + ε)
```

Gives a global coefficient of variation:

```
CV = std(ρ̃) / mean(ρ̃)
```

The system reaches thermal equilibrium when CV < 0.01.

### 2.4 Deferred Nudge System

Updates accumulate in a nudge vector a_i:

```
a_i += η_cent * ρ_i * m_i
```

Actual movement occurs only when:

```
||a_i|| ≥ δ_min = κ * (1 - r) * √τ
```

Where r = 1/√2 ≈ 0.71 is the geometric grid ratio and κ = 0.5 is a safety factor.

## 3. Data Structures

### 3.1 Core Node Structure

```
Node {
    w: vector[d]           // Position vector
    m: vector[d]           // EWMA residual mean (first moment)
    s: vector[d]           // EWMA squared residual (second moment)
    a: vector[d]           // Deferred nudge accumulator
    u: vector[d]           // Oja's rule principal direction
    hit_count: float       // Weighted win count
    update_count: int      // Number of statistical updates
    links: list[Link]      // Adjacent links
    N_creation: int        // Creation timestamp
}
```

Derived quantities:

- Variance: σ_i² = s_i - m_i²
- Incoherence: ρ_i = ||m_i|| / (σ_i + ε)

### 3.2 Link Structure

```
Link {
    node1, node2: Node*
    C12, C21: float        // Directed weighted hit counters
    activity_snapshot1, activity_snapshot2: float
}
```

### 3.3 Simplex Structure (Stage 2)

```
Simplex {
    vertices: list[Node*]
    d_simplex: int
    face_pressures: array[d_simplex+1] of float
    patch_id: int          // SimplexPatch membership

    // Torsion audit fields
    torsion_sum: float
    torsion_sum_sq: float
    simplex_update_count: int
}
```

### 3.4 SimplexPatch Structure

```
SimplexPatch {
    simplexes: list[Simplex*]
    centroid: vector[d]
    mean_torsion: float
    mini_nsf: MiniNSF*     // Attached mini-normalizing flow
}
```

### 3.5 Map Structure

```
Map {
    nodes: list[Node*]
    links: list[Link*]
    simplexes: list[Simplex*]    // Stage 2 only
    patches: list[SimplexPatch*] // Stage 2 only

    // Nonlinearity handling models (per-map)
    glow_model: GlowModel*       // Map-level Glow model
    nsf_model: NSFModel*         // Map-level NSF (if upgraded)

    // State
    stage: int                   // 1 or 2
    mode: string                 // "FAST" or "DETAIL"
    data: array[N, d]           // Current transformed data
    raw_data: array[N, d]       // Original data
}
```

## 4. Map Level Operations

### 4.1 Batch Node Operations

**compute_node_variances()**

```
for each node i:
    σ_i² = s_i - m_i²    // Element-wise
    σ_i = sqrt(mean(σ_i²))  // Scalar variance
```

**compute_node_incoherence_ratios()**

```
for each node i:
    ρ_i = ||m_i|| / (σ_i + ε)
    neighbors = get_neighbors(i)
    ρ̃_i = ρ_i / (mean(ρ_j for j in neighbors) + ε)
```

### 4.2 Thermal Equilibrium Computation

**compute_thermal_equilibrium()**

```
ρ̃_values = compute_node_incoherence_ratios()
CV = std(ρ̃_values) / mean(ρ̃_values)
return CV < 0.01
```

### 4.3 Node Update Process

**update_node_statistics(x, neighborhood)**

```
for each node j in neighborhood with weight w_j:
    e = x - w_j
    α = ln(2) / k_neighbors

    // Update EWMA moments
    m_j = (1 - α*w_j) * m_j + α*w_j * e
    s_j = (1 - α*w_j) * s_j + α*w_j * e²

    // Update Oja's rule
    u_j = u_j + η_oja * (e * dot(e, u_j) - ||e||² * u_j)
    u_j = u_j / ||u_j||

    // Update deferred nudge
    ρ_j = ||m_j|| / (σ_j + ε)
    a_j += η_cent * ρ_j * m_j

    // Update hit count
    hit_count_j += w_j
```

### 4.4 Deferred Nudge Application

**apply_deferred_nudges()**

```
for each node i:
    if ||a_i|| ≥ δ_min:
        w_i += a_i
        a_i = 0
        refresh_neighbor_caches(i)
        if stage == 2:
            recompute_incident_normals(i)
```

### 4.5 Torsion Computation (Stage 2)

**compute_torsion_ratios()**

```
for each simplex S:
    // Build edge and residual matrices
    E = [edge_vectors relative to centroid]  // (d+1)×d
    M = [residual_vectors relative to centroid]  // (d+1)×d

    // Compute 2-form
    Ω_S = M^T * E - E^T * M

    // Torsion magnitude
    κ_S = ||Ω_S||_F = sqrt(trace(Ω_S^T * Ω_S))

    // Torsion ratio
    R_S = κ_S / τ_global
```

### 4.6 Patch Building with Leiden Clustering

**build_simplex_patches()**

```
// Step 1: Build facet-adjacency graph
hot_simplexes = [S for S in simplexes if R_S ≥ 0.60]
graph = FacetAdjacencyGraph()
for S in hot_simplexes:
    for T in hot_simplexes:
        if S and T share a (d-1)-facet:
            weight = min(R_S, R_T)
            graph.add_edge(S, T, weight)

// Step 2: Leiden clustering with auto-tuned resolution
resolution = 1.0
while True:
    communities = leiden_algorithm(graph, resolution)
    largest_community_size = max(len(c) for c in communities)
    if largest_community_size <= P_max:
        break
    resolution *= 2.0

// Step 3: Budget filtering (per-map budget)
communities.sort(key=lambda c: mean(R_S for S in c), reverse=True)
map_budget = 0.20 * map.glow_model.parameter_count()
used_budget = sum(patch.mini_nsf.parameter_count() for patch in map.patches)
available_budget = map_budget - used_budget

selected_patches = []
for community in communities:
    cost = base_cost + α * len(community)
    if cost <= available_budget:
        selected_patches.append(SimplexPatch(community))
        available_budget -= cost
```

## 5. Proteus Level Operations

### 5.1 Per-Map Parameter Budget Management

**check_map_parameter_budget(map)**

```
total_mini_nsf_params = sum(patch.mini_nsf.parameter_count()
                           for patch in map.patches)
map_glow_params = map.glow_model.parameter_count()
return total_mini_nsf_params / map_glow_params <= 0.20
```

**allocate_parameters(map, requested_patches)**

```
available_budget = 0.20 * map.glow_model.parameter_count()
used_budget = sum(patch.mini_nsf.parameter_count() for patch in map.patches)
remaining_budget = available_budget - used_budget

requested_patches.sort(key=lambda p: p.mean_torsion, reverse=True)
allocated = []
for patch in requested_patches:
    cost = estimate_mini_nsf_cost(patch)
    if remaining_budget >= cost:
        allocated.append(patch)
        remaining_budget -= cost
return allocated
```

### 5.2 Per-Map Statistics

**get_map_torsion_coverage(map)**

```
total_simplexes = len(map.simplexes)
high_torsion_count = sum(1 for S in map.simplexes if R_S > 0.30)
return high_torsion_count / total_simplexes if total_simplexes > 0 else 0.0
```

### 5.3 Algorithm Orchestration

**run_complete_algorithm()**

```
// Phase 1: Scale-space search (per-map)
for each map in atlas:
    s_control_star = run_scale_space_search(map)
    map.set_optimal_scale(s_control_star)

// Phase 2: High-fidelity refinement (per-map)
for each map in atlas:
    if map.should_refine():
        run_stage_2_refinement(map)

// Phase 3: Per-map nonlinearity handling
for each map in atlas:
    run_map_nonlinearity_pipeline(map)

    // Each map handles its own torsion strategy
    if map.stage == 2:
        handle_map_torsion_strategy(map)
```

**run_scale_space_search(map)**

```
// Coarse grid search
tau_values = geometric_grid(tau_min, tau_max, ratio=0.71)
phi_values = []
for tau in tau_values:
    s_control = 1 - exp(-tau / D_subspace)
    phi = evaluate_scale_response(map, s_control)
    phi_values.append(phi)

// Find bracket around maximum
bracket = find_maximum_bracket(tau_values, phi_values)

// Bayesian refinement
gp_optimizer = GaussianProcessOptimizer()
tau_star = gp_optimizer.optimize(bracket, evaluate_scale_response)
return 1 - exp(-tau_star / D_subspace)
```

### 5.4 Hierarchy Management

**create_hierarchical_node(parent_clusters)**

```
node = HierarchicalNode()
node.level = parent_level + 1
node.parent_clusters = parent_clusters
node.map = create_child_map(parent_clusters)
return node
```

### 5.5 Per-Map Nonlinearity Pipeline

**run_map_nonlinearity_pipeline(map)**

```
// Step 1: Marginal ECDF transformation
z = apply_marginal_ecdf(map.data)

// Step 2: Test for remaining nonlinearity
glce_p = graph_laplacian_curvature_energy_test(z)
dcor_p = cs_weighted_distance_correlation_test(z)

// Step 3: Decide on initial Glow training
if glce_p > 0.01 or dcor_p > 0.01:
    map.glow_model = train_map_glow(z)
    z_transformed = map.glow_model.transform(z)

    // Step 4: Edge case guard
    raw_latent_dcor = distance_correlation(map.raw_data, z_transformed)
    if any(p < 0.005 for p in raw_latent_dcor):
        # Re-inject raw dimensions
        map.glow_model.add_raw_channels(problematic_dimensions)
        map.glow_model.retrain_affected_layers()

    map.data = z_transformed
else:
    map.data = z  // Use ECDF-transformed data

// Step 5: After Stage 2 refinement, handle torsion
if map.stage == 2:
    handle_map_torsion_strategy(map)

return map.data
```

**handle_map_torsion_strategy(map)**

```
P_κ = get_map_torsion_coverage(map)

if P_κ ≥ 0.50:
    // High torsion coverage - switch to map-level NSF
    switch_to_map_level_nsf(map)
else:
    // Low to moderate coverage - use patch-NSF strategy
    run_torsion_ladder_with_patches(map)
```

## 6. Statistical Tests and Pruning

### 6.1 Link Pruning Gauntlet

**Stage 1: Power Shield**

```
n_eff = sum(weights for all hits on link)
n_req = power_analysis_for_proportions(effect_size, α=0.05, β=0.20)
if n_eff < n_req:
    return PROTECTED  // Fair chance not yet given
```

**Stage 2: Relative Weakness Test**

```
for each link (i,j):
    proportion = C(i→j) / sum(C(i→k) for all k)
    wilson_upper = wilson_score_upper_bound(proportion, n_eff, z=1.28)

median_peer_proportion = median(peer_proportions)
if wilson_upper < median_peer_proportion:
    vote_for_pruning = True
```

**Stage 3: Bilateral Agreement**

```
if both_endpoints_vote_for_pruning:
    remove_link()
```

### 6.2 Node Pruning Gauntlet

**Trigger: Neighborhood Stability**

```
neighbor_rho_tilde = [ρ̃_j for j in neighbors(i)]
stability_p = welch_t_test(neighbor_rho_tilde, one_tailed=True)
if stability_p <= 0.05:
    return  // Neighborhood not stable
```

**Stage 1: Fair Chance Test**

```
λ_est = hit_count_i / lifetime_i
if λ_est < 2.0 / δ_min:
    return  // Fair chance not given
```

**Stage 2: Relative Performance**

```
neighbor_hits = [hit_count_j for j in neighbors(i)]
mean_neighbor_hits = mean(neighbor_hits)

low_hits = hit_count_i < 0.5 * mean_neighbor_hits
low_variance = σ_i² < 0.5 * τ_local_i

if low_hits and low_variance:
    prune_node(i)
```

### 6.3 Torsion Ladder Decision Tree

**Torsion Audit Process**

```
// First, check if we need to switch to map-level NSF
P_κ = get_map_torsion_coverage(map)
if P_κ ≥ 0.50:
    // Switch to map-level NSF instead of patch-NSFs
    switch_to_map_level_nsf(map)
    return

// Otherwise, process simplexes individually
for each simplex S:
    R_S = κ_S / τ_global

    if R_S < 0.05:
        // Ignore (numerical noise)
        continue
    elif 0.05 ≤ R_S < 0.30:
        // Keep as-is (tolerable curvature)
        continue
    elif 0.30 ≤ R_S < 0.60:
        if h_parent ≥ 2 * h_prune and split_budget_ok():
            // Single torsion-aligned split
            split_along_torsion_axis(S)
        else:
            // Fall back to mini-NSF (if under budget)
            if check_map_parameter_budget(map):
                attach_mini_nsf(S)
            else:
                // Budget exhausted, fall back to splitting
                split_along_torsion_axis(S)
    else:  // R_S ≥ 0.60
        // Attach mini-NSF (if under budget)
        if check_map_parameter_budget(map):
            attach_mini_nsf(S)
        else:
            // Budget exhausted, switch to map-level NSF
            switch_to_map_level_nsf(map)
            return
```

### 6.4 Torsion-Aligned Splitting

**Method A: Long-Edge Midpoint (Default)**

```
// Find principal torsion axis
u = dominant_eigenvector(Ω_S)

// Find vertices with max/min projection
projections = [dot(vertex_i - centroid, u) for vertex_i in vertices]
max_vertex = vertices[argmax(projections)]
min_vertex = vertices[argmin(projections)]

// Insert midpoint
midpoint = (max_vertex + min_vertex) / 2
new_vertex = create_node(midpoint)

// Replace simplex with two children
child1 = create_simplex([new_vertex] + other_vertices_1)
child2 = create_simplex([new_vertex] + other_vertices_2)
```

**Method B: Centroid Split (Fallback)**

```
// Quality check
Q_S = d * r_in / R_circ
if Q_S < 0.25:
    // Poor quality - use centroid split
    centroid = mean(vertices)
    new_vertex = create_node(centroid)

    // Create d+1 child simplexes
    for i in range(d+1):
        face_vertices = vertices[:i] + vertices[i+1:]
        child = create_simplex([new_vertex] + face_vertices)
```

## 7. Mini-NSF Architecture and Training

### 7.1 Mini-NSF Structure

Each mini-NSF is a compact Rational-Quadratic spline flow:

```
Architecture:
- 2-3 coupling layers
- 64-128 hidden units per layer
- 8-12 spline bins
- RQ-spline transformations
```

### 7.2 Training Process

**attach_mini_nsf(map, patch)**

```
// Freeze map's Glow parameters
map.glow_model.freeze()

// Initialize mini-NSF
mini_nsf = RationalQuadraticSplineFlow(
    layers=3,
    hidden_units=64,
    spline_bins=8
)

// Route training data from map's data
training_data = [x for x in map.data if x falls in patch]

// Train for limited epochs
for epoch in range(3):
    loss = mini_nsf.train_step(training_data)
    if converged(loss):
        break

patch.mini_nsf = mini_nsf
```

### 7.3 Per-Map Strategy Selection

**Nonlinearity Upgrade Sequence**

The upgrade sequence for handling nonlinearity is:

1. **Map-level Glow** (initial)
2. **Attempt splitting and patch-NSFs** (if < 50% coverage)
3. **Map-level NSF** (if ≥ 50% coverage or budget exhausted)

```
P_κ = get_map_torsion_coverage(map)

if P_κ ≤ 0.25:
    // Low torsion coverage - use patch-wise mini-NSFs
    use_patch_wise_mini_nsfs(map)
elif 0.25 < P_κ < 0.50:
    // Moderate coverage - try patches first, then upgrade if needed
    attempt_patch_wise_mini_nsfs(map)
    if not successful or budget_exhausted:
        switch_to_map_level_nsf(map)
else:  // P_κ ≥ 0.50
    // High coverage - directly upgrade to map-level NSF
    switch_to_map_level_nsf(map)
```

**switch_to_map_level_nsf(map)**

```
// Remove existing patch mini-NSFs
for patch in map.patches:
    if patch.mini_nsf is not None:
        patch.mini_nsf = None

// Replace map's Glow with NSF
map.glow_model = None
map.nsf_model = create_map_level_nsf(map)
map.nsf_model.train(map.data)
```

## 8. Dimension-Aware Masking for Multi-Modal Data

### 8.1 Activity Mask Construction

```
construct_activity_mask(x):
    mask = (x != 0)  // Boolean array
    return mask
```

### 8.2 Masked Distance Computation

```
compute_masked_distance(x, w, mask):
    error = x - w
    error[~mask] = 0
    distance_squared = sum(error[mask]²)
    return sqrt(distance_squared)
```

### 8.3 Masked Updates

```
apply_masked_update(x, w, mask):
    Δw = η * (x - w)
    Δw[~mask] = 0

    // Optional: normalize by active dimensions
    active_count = count(mask)
    if active_count > 0:
        Δw[mask] /= active_count

    w += Δw
```

## 9. Root-Level Dimensionality Reduction

### 9.1 Randomized SVD Procedure

```
dimensionality_reduction_pipeline(X_root):
    // Step 1: Sample subset
    X_sub = sample_rows(X_root, 20_000)
    d = X_root.shape[1]

    // Step 2: Randomized SVD
    k_target = 256
    U, S, Vt = randomized_svd(X_sub, k_target, n_oversamples=20)

    // Step 3: Determine projection rank
    energy = cumsum(S²) / sum(S²)
    r_999 = searchsorted(energy, 0.999) + 1

    // Optional: Levina-Bickel intrinsic dimension
    D_LB = levina_bickel_id(X_sub, k_id=20)
    r_candidate = max(r_999, 2 * ceil(D_LB))

    // Step 4: Check for useful reduction
    if r_candidate <= 0.95 * d:
        r_final = min(r_candidate, 2048)
    else:
        r_final = d  // Skip projection

    // Step 5: Build projection with JL padding
    P = Vt[:r_final, :].T
    P = hstack([P, randn(d, 32) / sqrt(d)])  // JL padding

    // Step 6: Project data
    Z_root = X_root @ P
    return Z_root, P
```

## 10. Convergence and Stopping Criteria

### 10.1 Thermal Equilibrium Conditions

A map reaches thermal equilibrium when:

1. All σ_S² ≤ τ for all simplexes S
2. All R_S < 0.30 for all simplexes S
3. CV(ρ̃) < 0.01 for neighbor-normalized incoherence

### 10.2 Recursion Termination

**Minimum Sample Rule**

```
min_samples_for_recursion = max(10 * d, 1000)
if cluster.sample_count < min_samples_for_recursion:
    delay_recursion()
```

**Terminal Conditions**

- Only one cluster found at current scale _if_ previous scale was also only one cluster.
- No well-separated features remain (optimizer proves this)
- Insufficient samples for reliable statistics

## 11. Integration and Workflow

### 11.1 Complete Algorithm Flow

```
ProteusAlgorithm.fit(X):
    // Step 1: Root-level preprocessing
    if X.shape[1] > 256:
        X_proj, projection_matrix = dimensionality_reduction(X)
    else:
        X_proj = X

    // Step 2: Create root proteus instance
    root_proteus = Proteus(X_proj)

    // Step 3: Run complete algorithm
    hierarchy = root_proteus.run_complete_algorithm()

    // Step 4: Build final model
    return ProteusModel(hierarchy, projection_matrix)
```

### 11.2 Stage 1 to Stage 2 Transition

```
transition_to_stage_2(stage1_result):
    // Warm-start statistics
    gamma0 = 0.7
    for node in stage1_result.nodes:
        node.m *= gamma0
        node.s *= gamma0
        node.a = zeros_like(node.a)
        // Keep hit_count and links unchanged

    // Switch to Stage 2 mode
    map.mode = DETAIL_MODE
    map.k_neighbors = approximate_dimension

    // Initialize simplicial complex
    initial_complex = greedy_chaining_heuristic(stage1_result.nodes)

    // Begin high-fidelity refinement
    run_stage_2_refinement()
```

This specification provides a comprehensive guide to implementing the Proteus algorithm with all the mathematical foundations, detailed algorithms, and implementation specifics needed for a complete system.
