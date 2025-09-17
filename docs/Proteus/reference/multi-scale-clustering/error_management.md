# Proteus: Error Management and Learning Dynamics

## 1. Abstract and Motivation

This document provides a complete, exhaustive specification for the error management and learning dynamics at the core of the Proteus framework. The central challenge in any adaptive system is correctly interpreting an "error" signal. An error can indicate a need for **geometric adjustment** (moving existing components) or **topological growth** (creating new components). A naive system conflates these, leading to instability and inefficiency.

The Proteus learning rule is designed to intelligently distinguish between these two cases. It does this by establishing a "state" for each local region of the learned manifold, which determines how error is partitioned and handled. This state is governed by a single, principled, and self-normalizing metric that allows the system to prioritize computationally cheap topological growth when error is high and coherent, and only engage in expensive geometric adjustment when the local manifold has stabilized. This document details the precise mechanisms that make this sophisticated, adaptive behavior possible.

## 2. The Core Components

### 2.1. State Variables (per Node)

To cleanly separate concerns, each node `i` in the network maintains two distinct error vectors:

- `w_i`: The node's position vector.
- `n_i`: The **Kinetic Nudge Vector**. It accumulates the geometric portion of the error (`e_geo`), is used for positional adjustments.
- `s_i`: The **Potential Stress Vector**. It accumulates the topological portion of the error (`e_topo`), is used for splitting.
- `H_i`: An **Incoherence Accumulator**. It measures the "heat" generated _only_ by updates to the kinetic `n_i` vector.
- `activity_i`: A simple counter of how many times the node has been the Best Matching Unit (BMU).
- `update_count_i`: A counter for the number of updates the node has received, used for normalizing `H_i`.

### 2.2. Global State Variables

The system maintains two running statistics for the entire GNG map at the current level of the hierarchy:

- `Global_Avg_H_norm`: The mean of the normalized incoherence across all nodes.
- `Global_Avg_CV`: The mean of the coefficient of variation of incoherence across all simplexes.

## 3. The Unified Learning Rule: A State-Dependent Dynamical System

The system has no discrete "phases." Instead, its behavior emerges naturally from a single, continuous learning rule that smoothly adapts based on the local region's distance from a stable equilibrium.

### 3.1. The Update Rule

For each data point `x`:

1.  **Find Neighborhood:** The local neighborhood is determined (k-NN for Stage 1, the containing simplex for Stage 2).
2.  **Calculate Error:** The error vector `e` is calculated (`e = x - w_bmu`).
3.  **Calculate State-Dependent Behavior:** The system determines how to interpret this error based on the local stability. This is governed by a **Behavioral Bias** parameter, `B_C`, for the local simplex `C`. `B_C` is calculated as described in section 3.2.
4.  **Partition Error:** The error `e` is partitioned based on the behavioral bias:
    - **Topological Impulse (`e_topo`):** `e_topo = B_C * e`.
    - **Geometric Impulse (`e_geo`):** `e_geo = (1 - B_C) * e`.
5.  **Apply Updates:** The impulses are distributed among the nodes in the neighborhood. For each node `i` receiving an update:
    - The **Incoherence `H_i`** is updated based on the geometric impulse, normalized by `CS`: `H_i_new = H_i_old + abs(||n_i_old + e_geo|| - (||n_i_old|| + ||e_geo||)) / CS`.
    - The **Nudge Vector `n_i`** is updated: `n_i_new = n_i_old + e_geo`.
    - The **Stress Vector `s_i`** is updated: `s_i_new = s_i_old + e_topo`.

### 3.2. Calculating the Behavioral Bias (`B_C`)

The bias `B_C` determines whether the system prioritizes growth or adjustment. It is a direct, continuous function of how far the local region is from equilibrium.

1.  **Node "Temperature":** For each node `i`, calculate its age-invariant information quality: `H_norm_i = H_i / update_count_i`.
2.  **Local Disequilibrium:** For simplex `C`, calculate the Coefficient of Variation of its nodes' temperatures: `CV_local = std_dev(H_norm_i) / mean(H_norm_i)`.
3.  **Distance from Attractor:** Normalize the local value against the global average: `Dist_From_Eq = CV_local / Global_Avg_CV`.
4.  **Final Bias Calculation:** The bias `B_C` is calculated with a sigmoid function. A high `Dist_From_Eq` means the region is unstable and should prioritize topological growth.
    - `B_C = sigmoid(Dist_From_Eq - x_0)`
    - This means when a region is unstable, `B_C` is high (e.g., 0.9), and ~90% of error becomes `e_topo`. When stable, `B_C` is low (e.g., 0.1), and ~90% of error becomes `e_geo`.

### 3.3. State-Dependent Action Triggers

The two error vectors now drive two distinct actions with different thresholds.

1.  **Geometric Adjustment:** Triggered when a node's kinetic energy is high: `if ||n_i|| > CS_nudge`. The node's position is adjusted (`w_i = w_i + n_i`), and `n_i` is bled off.
2.  **Topological Split:** Triggered when a node's potential energy is high: `if ||s_i|| > CS_split`. A new node is created to relieve the stress, and `s_i` is reset.
3.  **Torsional Stress Audit:** In Stage 2, when the system is in the stable regime (`B_C` is low), the collection of `s` vectors across a simplex provides the high-fidelity stress field required for the Torsional Stress audit.

## 4. Conclusion

This complete error management system is a self-normalizing dynamical system. By using two separate error vectors and a sophisticated, self-normalizing metric to partition energy between them, the system correctly prioritizes cheap topological splits when error is high and coherent, and only performs expensive geometric adjustments when the manifold has stabilized. This provides the foundation for a robust, efficient, and deeply principled learning process.
