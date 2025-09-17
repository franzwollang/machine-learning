# Stage 2: High-Fidelity Refinement and Audit at a Fixed Scale

## 1. Abstract and Design Philosophy

This document provides a complete specification for the "Stage 2" component of the Proteus algorithm. The primary design goal of this architecture is **maximum descriptive power and theoretical rigor**. It is not a search algorithm, but a high-fidelity refinement engine that operates at a single, fixed characteristic scale `τ*`, which is derived from the `s_control*` parameter provided by Stage 1. For Stage 2, this `τ*` is the global scale threshold (`τ_global*`) identified by the optimizer.

Stage 2 takes the converged node positions from the Stage 1 search at `s_control*` as a "warm start" and then performs a complete, high-fidelity re-evaluation of the manifold structure at the corresponding fixed scale `τ*`. It employs more sophisticated, computationally intensive diagnostics based on a simplicial complex model to audit the initial findings, upgrade mappings, and refine the geometry. The output of Stage 2 is the final, audited, and generative model of the manifold at the specified scale.

## 2. The Stage 2 Pipeline

The Stage 2 process is a single, intensive refinement pass at a fixed `τ*`.

1.  **Initialize and Warm-Start from Stage 1:** The process begins with the final GNG node positions and statistics from the Stage 1 run that identified `s_control*`. To leverage this information while accommodating Stage 2's richer neighborhood model (using `k ≈ d` neighbors instead of `k=8`), the statistics are warm-started:
    - The EWMA moments `m` and `s` are shrunk by a single decay factor `γ₀ ≈ 0.7`.
    - Hit counts and link counts are carried over unchanged.
    - The nudge accumulator `a` is reset to zero.
2.  **Initial Simplex Seeding:** A fast, quadratic-cost "Greedy Chaining" heuristic is run on the GNG graph to discover the vast majority of `d`-simplices, forming an initial simplicial complex.
3.  **High-Fidelity Dynamic Refinement:** The learning algorithm is run on the dynamic simplicial complex until it reaches thermal equilibrium (`CV(ρ̃) < 0.01`). This phase uses simplex-native updates and continuously "heals" the mesh by discovering new simplices.
4.  **The Torsion Ladder (Geometric Audit):** Throughout the refinement, the system continuously audits the geometric validity of the manifold using the **Torsion Ladder**. For each simplex, it measures the ratio of torsional stress to the scale threshold (`R_S = κ_S/τ*`). Based on this ratio, it decides to:
    - **Ignore** (noise, `R_S < 0.05`).
    - **Keep As-Is** (tolerable curvature, `0.05 ≤ R_S < 0.30`).
    - **Perform a Torsion-Aligned Split** (high, coherent stress, `0.30 ≤ R_S < 0.60`).
    - **Attach a Mini-NSF** (very high or incoherent stress, `R_S ≥ 0.60`).
5.  **Dual Flow Calculation and Convergence:** The `face_pressures` on the simplices are computed and refined via Belief Propagation, resulting in a final, converged probability gradient field across the manifold.

## 3. The High-Fidelity Model

### 3.1. Data Structures

The data structures are expanded from Stage 1 to include `Simplex` objects and track the statistics necessary for the torsion audit.

```
// Node structure is identical to Stage 1.
Node {
    w: vector[d_embedding], m: vector[d_embedding], s: vector[d_embedding], a: vector[d_embedding], u: vector[d_embedding],
    hit_count: float, update_count: int, links: list[Link*], N_creation: int
}

Simplex {
    vertices: list[Node*]
    d_simplex: int
    face_pressures: array[d_simplex+1] of float

    // Fields for Torsion Audit
    torsion_sum: float         // Cumulative sum of torsion measurements (κ_S)
    torsion_sum_sq: float      // Cumulative sum of squared torsion measurements
    simplex_update_count: int  // Number of updates recorded for the audit
}
```

### 3.2. Theoretical Basis: Torsion as Discrete Curvature

The core of Stage 2's geometric audit is the concept of **torsion**. In the continuous world, the "curl" of a vector field (`∇ × m`) measures its local rotational tendency. A field with zero curl is "irrotational" or "conservative," meaning it can be expressed as the gradient of some scalar potential. In our system, the residual mean field `m` should ideally be conservative; any non-zero curl indicates a rotational mis-fit that cannot be fixed by simple node translation.

The **simplex 2-form `Ω_S`** is the direct discrete analogue of curl. It is derived by discretizing the line integral of the `m` field around the edges of a simplex. Its Frobenius norm, `κ_S = ||Ω_S||_F`, gives a single scalar value for the "torsional stress" or "curvature energy" within that simplex. A high `κ_S` indicates that the GNG's linear approximation of the manifold is failing within that simplex, signaling a need for intervention.

### 3.3. Simplex-Aware Learning and Analysis

Stage 2 uses a dynamic, geometrically-grounded learning rule that continuously grows and refines the simplicial complex.

1.  **Simplex-Native Update:** For each data point `x`, the single winning simplex `C` that contains `x` is identified. The vertices of this simplex become the exclusive neighborhood for the update. The error `e` is calculated relative to each vertex, and the `m` and `s` statistics for each vertex are updated, weighted by their rank-ordered distance to `x`.
2.  **Dynamic Topology Management:** Stage 2 uses a sophisticated hybrid of GNG and simplex-native rules to evolve its topology.
    - **Link Creation (GNG-style):** A GNG-style link is created between the two best-matching nodes (`BMU_1`, `BMU_2`), providing a mechanism to explore new topology.
    - **Continuous Simplex Discovery:** Whenever a new link is created, a local search checks if it completes a new `d+1` clique. If so, a new simplex is materialized, healing any holes in the initial complex.
    - **The Torsion Ladder (Geometric Audit & Model Upgrade):** This is the core of Stage 2's refinement. For each simplex `S`:
      - **Measure Torsion:** Compute the discrete 2-form `Ω_S` and its magnitude, `κ_S`.
      - **Assess Ratio & Decide:** The critical ratio `R_S = κ_S / τ*` determines the action.
        - **Justification for Triage:** The thresholds for `R_S` act as a principled triage system. Ratios near zero (`<0.05`) are treated as numerical noise. Small, non-zero ratios (`<0.3`) indicate curvature that is tolerable and within the expressive power of the current linear simplex. Ratios from `0.3-0.6` indicate significant, coherent stress that can be resolved by a geometric split. Very high ratios (`>0.6`) signal that the local curvature is too strong for a simple geometric fix, requiring a more powerful, non-linear model (a mini-NSF).
      - **Global vs. Patch-wise Strategy:** The system monitors the total percentage of the mesh flagged with high torsion (`P_κ`). If `P_κ` exceeds a threshold (e.g., 25-50%), it may be more efficient to upgrade the entire level's `Glow` map to a deeper `Glow` or a global `NSF`, rather than attaching many small, independent mini-flows.
    - **Torsion-Aligned Splitting:** When the ladder prescribes a split:
      - **Method A (Default):** Find the principal axis of the torsion tensor `Ω_S`. The new node is inserted at the midpoint of the simplex's longest edge projected onto this axis.
        - **Justification:** This method directly resolves the primary rotational stress captured by the torsion tensor, making it the most targeted and efficient geometric fix.
      - **Method B (Fallback):** If the resulting child simplices have poor shape quality (`Q_S = d*r_in/R_circ < 0.25`), the split is redone using a simple centroid insertion.
        - **Justification:** A poor shape quality (e.g., a "sliver" simplex) can lead to numerical instability in subsequent gradient and flow calculations. The centroid split fallback is a crucial guarantee of mesh quality, ensuring that the face normals of child simplices remain well-separated.
    - **Pruning (Simplex-Arbitrated):** Node deletion is a formal vote arbitrated by the simplexes containing the node, with a geometric veto (reconstruction error test) to protect structurally critical "tent pole" nodes.

### 3.4. Dual Flow and Generative Capability

The dual flow provides a high-fidelity map of the probability density across the manifold.

- **Mechanism (Proportional Representation):** For each data point, the hit is distributed _fractionally_ among the faces of the winning simplex, proportional to the projection of the error vector onto each face normal.
- **Convergence and "Sub-Pixel" Resolution:** A **Belief Propagation** algorithm is run on the dual graph of faces to enforce global flow conservation. The resulting converged `face_pressures` define a piecewise-linear probability density function across the manifold, allowing the model to compute a probability for any point `x` and to be used for generative sampling.

## 4. Conclusion

Stage 2 transforms the raw scale information from Stage 1 into a complete, audited, and generative model of the manifold. By operating at a fixed `τ*` and employing advanced techniques like torsion analysis and simplex-native updates, it can achieve a level of geometric and probabilistic accuracy that would be computationally prohibitive during an open-ended search. It delivers the final, high-fidelity representation of the data's structure at the discovered characteristic scale.
