# Addendum: On the Relationship Between Simplex and Voronoi Equilibria

This addendum addresses a critical theoretical point regarding the Proteus algorithm: the relationship between the **Simplex Equilibrium** explicitly sought by the algorithm and the **Node Equilibrium** (defined by a Voronoi tessellation) that is the goal of traditional vector quantization methods like k-means or standard GNG.

We will show that while the two equilibria are not identical, optimizing for Simplex Equilibrium creates a powerful and direct force that pushes the system toward Node Equilibrium, and that the minor divergences between them are themselves informative.

## 1. Defining the Two States of Equilibrium

Let `P(R)` denote the probability mass (or, in practice, the integrated data point activity) contained within a region `R` of the data space.

- **Node Equilibrium (The Voronoi Goal):** The system is in Node Equilibrium if the probability mass contained within the Voronoi cell `V_k` of each node `k` is approximately equal. That is, for a network with `N` nodes, the goal is:
  `P(V_k) ≈ 1/N` for all nodes `k = 1, ..., N`.

- **Simplex Equilibrium (The Proteus Goal):** The system is in Simplex Equilibrium if the probability mass contained within each simplex `C_i` is approximately equal. For a network with `M` simplexes, the goal is:
  `P(C_i) ≈ 1/M` for all simplexes `i = 1, ..., M`.

The Proteus algorithm's learning rules, specifically the **Topological Split** triggered when `||\mathbf{E}_C|| > CS`, directly optimize for Simplex Equilibrium. The question is how this affects Node Equilibrium.

## 2. Mathematical Sketch: How Simplex Equilibrium Drives Node Equilibrium

We can demonstrate the coupling by considering a system that is out of Node Equilibrium and observing how the algorithm's response pushes it back toward that state.

1.  **Initial State (Disequilibrium):** Assume a node `i` is "over-active," meaning its Voronoi cell `V_i` contains significantly more than its fair share of probability mass (`P(V_i) >> 1/N`). This implies that data points are landing disproportionately in the region of space closest to the vertex `\mathbf{w}_i`.

2.  **Impact on Adjacent Simplices:** Node `i` is a vertex for a set of adjacent simplexes `S_i = {C_1, C_2, ...}`. Because `P(V_i)` is high, these simplexes will experience a highly biased stream of data points. For any data point `\mathbf{x}` landing in a simplex `C_k ∈ S_i`, it is, on average, much closer to `\mathbf{w}_i` than to the other vertices of `C_k`.

3.  **Accumulation of Structural Stress:** For a given simplex `C_k`, the instantaneous error vector is `\mathbf{e} = \mathbf{x} - \text{barycenter}(C_k)`. Due to the data bias, the average direction of `\mathbf{e}` will contain a significant vector component pointing _radially away_ from `\mathbf{w}_i`. Over many data points, this causes the accumulated structural stress vector `\mathbf{E}_{C_k}` to grow in this outward direction. The simplex is being "inflated" by the high probability pressure near node `i`.

4.  **Topological Correction (Node Insertion):** Eventually, one of these over-stressed simplexes, `C_{split}`, will breach the topological split threshold (`||\mathbf{E}_{C_{split}}|| > CS`). The Quantized Growth procedure is triggered:

    - The insertion vector `\mathbf{E}_{quantized}` is calculated, which points in the primary direction of the error (away from `\mathbf{w}_i`).
    - A new node `j` is created at `\mathbf{w}_j = \text{barycenter}(C_{split}) + \mathbf{E}_{quantized}`.

5.  **Restoration of Equilibrium:** This new node `j` now defines its own Voronoi cell, `V_j`. By being placed away from `\mathbf{w}_i` in the direction of high data density, `V_j` is formed almost entirely from territory that previously belonged to `V_i`. The immediate effect is a reduction in the volume and, therefore, the contained probability mass `P(V_i)`. The mass is transferred from `V_i` to the new cell `V_j`.

This demonstrates a direct, causal link: a deviation from **Node Equilibrium** creates the precise conditions (biased structural stress) that trigger a topological correction, and this correction acts to restore **Node Equilibrium**. The system is a self-regulating negative feedback loop.

## 3. The Informative Mismatch

The two equilibria are not, however, perfectly identical. The primary reason for this is that the geometry of the simplicial tiling does not perfectly align with the geometry of the Voronoi tessellation. The regions where the two equilibria cannot be simultaneously satisfied are highly informative.

A prime example is a **dimensionality junction** (e.g., a 1D filament meeting a 2D sheet). A node at this junction will be a vertex for both high-activity 2D simplexes and low-activity 1D simplexes (line segments). This neighborhood will never be in perfect Simplex Equilibrium. This persistent, stable imbalance is not a failure of the algorithm; it is an **emergent property that correctly signals the complex local topology of the data manifold.** The algorithm's attempt to achieve Simplex Equilibrium, and its "failure" to do so in this specific, patterned way, provides a robust signature for identifying these important structural features.

### The Concrete Signature of a Dimensionality Junction

The robust signature of a node `i` being at such a junction is the simultaneous and persistent observation of several statistical patterns:

1.  **Bimodal Simplex Activity:** The distribution of `internal_activity` values for the simplexes adjacent to node `i` will be sharply bimodal, with one group (the high-dimensional part) having high activity and another group (the low-dimensional part) having low activity.

2.  **Divergent Link Significance:** The node's links will be cleanly partitioned into two groups: high-significance "backbone" links and low-significance "bridge" links. Crucially, the bridge links will resist pruning because their endpoints in the low-dimensional structure view them as essential.

3.  **High Asymmetry, Low Pressure on Bridge Links:** The "bridge" links will exhibit a unique combination of signals:
    - **High, Stable Hit-Counter Asymmetry:** The directional counters `C(i → bridge_node)` will be persistently much larger than `C(bridge_node → i)`.
    - **Low, Stable Net Face Pressure:** Despite the hit-counter imbalance, the net pressure on the simplex faces adjacent to the bridge link will be low. This distinguishes the stable junction from an active, growing frontier, where both asymmetry and pressure would be high.

This combination of signals is unique and allows the system to move beyond simply modeling the manifold to making claims about its intrinsic local topology.
