# Proteus-Omega: Complete Pseudocode

This document provides a high-level pseudocode representation of the complete Proteus-Omega pipeline, broken down by its three main stages.

---

## 1. Data Structures

These are the primary data structures used throughout the framework.

```
// --- Core Structures ---
Node {
    id: int
    w: vector // Position
    n: vector // Kinetic impulse ("nudge")
    s: vector // Accumulated structural stress
    H: float  // Incoherence accumulator
    activity: int
}

Simplex {
    vertices: list[Node*]
    face_pressures: array of float
    T_C_smoothed: float
    Cancellation_smoothed: float
}

// --- Hierarchy & Mappings ---
GNG_Map {
    nodes: list[Node]
    links: list[Link]
    simplices: list[Simplex] // Populated in Stage 2
}

Cluster_Node {
    id: string
    gng_map: GNG_Map
    parent: Cluster_Node*
    children: list[Cluster_Node*]
    pca_map: PCA_Model
    nsf_map: NSF_Model
}

// --- Indexing Artifacts (Stage 1 & 3) ---
HNSW_Index {}
Omega_SFC_Recipe_Tree {}
LSH_Router {
    hyperplanes: list[(vector, float)]
}
```

---

## 2. Stage 1: Fast Manifold Scaffolding

This stage recursively discovers the hierarchical structure of the data.

```
function Run_Stage_1(data, parent_map=null):
    // 1. Initialize GNG and Local Index
    gng = Initialize_GNG(parent_map)
    hnsw = Build_Local_HNSW(gng.nodes) // Build HNSW for this subspace

    // 2. Converge GNG
    for each data_point in data:
        Update_GNG_Stage1(gng, hnsw, data_point)

    // 3. Analyze and Recurse
    clusters = Community_Detection(gng.graph)
    if Is_Multi_Modal(clusters):
        hierarchy_node = Create_Hierarchy_Node(gng)
        for each cluster in clusters:
            sub_data = Get_Data_For_Cluster(data, cluster)
            sub_map = Get_Sub_Map(gng, cluster)

            // a. Non-linearity Check
            if Is_Non_Linear(sub_map.nodes):
                pca, nsf = Learn_Linearization_Maps(sub_data)
                sub_map.nodes = Apply_Maps(sub_map.nodes, pca, nsf)
                sub_data = Apply_Maps(sub_data, pca, nsf)
                hierarchy_node.children[cluster.id].pca_map = pca
                hierarchy_node.children[cluster.id].nsf_map = nsf

            // b. Build specialized index for the subspace
            density_approx = Gaussian_Approximation(sub_map.nodes)
            omega_sfc = Build_Omega_SFC(density_approx)
            // Store omega_sfc for use in Stage 2 acceleration

            // c. Recurse
            Run_Stage_1(sub_data, parent_map=sub_map)

    return Final_GNG_Hierarchy

function Update_GNG_Stage1(gng, hnsw, x):
    k_nearest = hnsw.Find_Nearest(x, k)
    error = x - k_nearest[0].w

    // Distribute 50% of error to nudge vectors (n)
    Distribute_Error_Geometric(k_nearest, error * 0.5, "n")
    // Distribute 50% of error to stress vectors (s)
    Distribute_Error_Geometric(k_nearest, error * 0.5, "s")

    // Check for geometric and topological updates
    for node in k_nearest:
        if ||node.n|| > NUDGE_THRESHOLD:
            node.w += node.n
            node.n = 0
            hnsw.Update_Node(node) // Update node position in index

        if ||node.s|| > SPLIT_THRESHOLD:
            new_node = Create_New_Node(node)
            gng.Add_Node(new_node)
            hnsw.Add_Node(new_node)
            node.s = 0

    // Manage links and pruning periodically
    ...
```

---

## 3. Stage 2: High-Fidelity Refinement and Audit

This stage re-runs the full hierarchy to refine the geometry and audit the Stage 1 findings.

```
function Run_Stage_2(gng_hierarchy_s1):
    Run_Stage_2_Recursive(gng_hierarchy_s1.root)

function Run_Stage_2_Recursive(cluster_node_s1):
    // 1. Initialize from Stage 1
    gng_s2 = Initialize_From_Stage1_Map(cluster_node_s1.gng_map)
    Materialize_Simplices(gng_s2)
    omega_sfc = Get_Omega_SFC_For_Subspace(cluster_node_s1.id)

    // 2. Refinement Loop with Geometric Audit
    for each data_point in GetDataForNode(cluster_node_s1):
        Update_GNG_Stage2(gng_s2, omega_sfc, data_point)

        // Audit Torsion
        torsion_signal = Check_Torsion(gng_s2.simplices)
        if torsion_signal > TORSION_THRESHOLD:
            // Upgrade map and restart refinement for this subspace
            nsf_upgraded = Retrain_NSF(cluster_node_s1.nsf_map)
            cluster_node_s1.nsf_map = nsf_upgraded
            // Transform gng_s2 nodes into new latent space
            // Re-materialize simplices
            // Re-build Omega SFC for new latent space
            // Restart refinement loop for this subspace
            ...

    // 3. Recurse if children exist
    for child in cluster_node_s1.children:
        Run_Stage_2_Recursive(child)

function Update_GNG_Stage2(gng, omega_sfc, x):
    // Find winning simplex using the accelerated Omega SFC index
    winning_simplex = Find_Simplex_For_Point(omega_sfc, x)

    error = x - Find_BMU_In_Simplex(winning_simplex, x).w

    // Distribute error to n and s vectors of simplex vertices
    Distribute_Error_Ranked(winning_simplex.vertices, error)

    // Perform topological corrections managed at the simplex level
    Manage_Simplex_Splits(winning_simplex)
    Manage_Node_Pruning(winning_simplex)

    // Update dual flow face pressures
    Update_Face_Pressures(winning_simplex, error)
```

---

## 4. Stage 3: Indexing and Query Engine Construction

This stage takes the final, refined model from Stage 2 and builds the queryable artifacts.

```
function Build_Final_Engine(gng_hierarchy_s2):
    // 1. Build Routers and Final SFCs
    lsh_routers = {}
    omega_sfcs_final = {}
    for each cluster_node in gng_hierarchy_s2:
        lsh_routers[cluster_node.id] = Train_LSH_Router(cluster_node)
        if Is_Terminal_Node(cluster_node):
            density_final = Reconstruct_Density_From_Dual_Flow(cluster_node)
            omega_sfcs_final[cluster_node.id] = Build_Omega_SFC(density_final)

    // 2. Generate Outputs for all original data points
    document_table = {}
    all_membership_vectors = {}
    for each original_point in ALL_DATA:
        key, vector = Generate_Outputs(original_point, gng_hierarchy_s2, lsh_routers, omega_sfcs_final)
        doc_id = original_point.id
        document_table[doc_id] = { "1D_key": key, "terminal_cluster_id": ... }
        all_membership_vectors[doc_id] = vector

    // 3. Build Final Payload Indexes
    wavelet_tree = Build_Wavelet_Tree(document_table)
    inverted_index = Build_Inverted_Index(all_membership_vectors)

    production_engine = {
        "lsh_routers": lsh_routers,
        "nsf_maps": ...,
        "omega_sfcs": omega_sfcs_final,
        "wavelet_tree": wavelet_tree,
        "inverted_index": inverted_index,
        "document_table": document_table
    }

    archival_model = {
        "gng_hierarchy": gng_hierarchy_s2,
        // (Includes full node/simplex states)
    }

    return production_engine, archival_model
```
