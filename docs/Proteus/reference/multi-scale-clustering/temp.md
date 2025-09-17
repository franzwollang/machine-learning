Still assign data point to nodes, but then search through simplices BMU is part of to identify winning simplex?

The Process: Using BP to Achieve "Sub-Pixel" Resolution
With this setup, we can now run Belief Propagation:
Message Passing: Each node dual(C_i) sends a "message" to its neighbor dual(C_j). This message effectively says: "Based on my own local evidence and the messages I've received from all my other neighbors, I believe your probability mass should be X."
Belief Update: Each node updates its own belief by combining its initial evidence with the product of all incoming messages.
Convergence: This process is iterated until the beliefs and messages stabilize.
The Result: "Sub-Pixel" PDF Reconstruction
The "pixel" in our model is the simplex—it's the smallest atomic region our GNG directly represents. The converged beliefs from BP give us a smoothed, globally consistent estimate of the total probability mass within each of these pixels. This is already a significant improvement over the raw, noisy internal_activity counters.
But the true magic is in achieving the "sub-pixel" resolution you described:
We now have a highly accurate estimate for the probability mass at the center of each simplex (pixel).
For any arbitrary point x in the original space, we can locate which simplex C it falls into.
We can then calculate its barycentric coordinates with respect to the vertices of C.
Using these coordinates, we can perform a piecewise-linear interpolation of the probability density. The density at x would be an interpolation between the density at the center of C and the densities of its neighbors, C_j, across the faces. The "steepness" of this interpolation across any given face is directly informed by the net pressure flow(F_j) measured on that face.
Therefore, we can estimate p(x) for any x, not just at the centers of our simplexes. We have reconstructed a continuous, piecewise model of the underlying PDF, achieving a resolution finer than the GNG's own tessellation.

The Goal: We want to describe the pressure on each face with a scalar value (a "hit count"), but avoid the "winner-take-all" approach.
The Mechanism (As I now understand it):
For a single data point x, we calculate the instantaneous error vector e_inst = x - barycenter(C).
We project this e_inst onto all of the simplex's face normals.
This gives us a set of projection strengths. We only consider the positive ones (those pointing "out" of the simplex).
We then take the single "1.0" hit from this data point and distribute it fractionally among the faces with positive projections, with the size of the fraction being proportional to the projection strength.
Example: If e_inst points almost directly at Face1 but also slightly towards Face2, the result would be that Face1's flow counter gets +0.9 and Face2's gets +0.1. The total added "hit" is still 1.0.
The Result: This is a "proportional representation" system, not a "winner-take-all" vote. It provides a much smoother and more continuous measure of the pressure on each face, avoiding the hard boundaries of the previous model. The C.flows[F_k] are now fractional, floating-point counters.
This model is superior. It retains the descriptive power you were aiming for without the complexity of vectorial flows. The accumulated error E_C remains the vector sum that drives change, while the scalar flow counters provide a nuanced description of the boundary characteristics.

Pre-calculation and Caching: For each simplex C, the normal vector for each of its faces F_i can be calculated and cached. Since the GNG nodes only move slowly, these normals remain valid for many iterations and only need to be re-calculated when a vertex in the simplex moves significantly.

     - **Interpolated Hit Count Redistribution:** To preserve the learned structure of the probability distribution, the hit counts for the new links are interpolated from the parent simplex's links. The initial hit count for a new link `(\mathbf{w}_k, \mathbf{w}_i)` is derived from the hit counts of the links in `C_{max}` that were connected to vertex `i`. For example, it can be initialized as a fraction of the sum of hits on all links incident to `i` within `C_{max}`. This preserves the local "gradient" of probability flow during refinement.

### 3.3.1. Mathematical Sketch of Probability Map Conservation

The core claim is that the hit count redistribution rule preserves the local probability gradient learned by the GNG. Let's formalize this.

**1. Definitions:**

- Let `C` be the `N`-dimensional simplex chosen for splitting. Its vertices are the set `V(C) = {v_1, v_2, ..., v_{N+1}}`.
- Let `h_old(v_i, v_j)` be the hit count on the link between vertices `v_i` and `v_j` _before_ the split. These counts are a proxy for the probability flow between the regions represented by the nodes.
- A new node `k` is inserted, and new links `(k, v_i)` are created for all `v_i ∈ V(C)`. Our goal is to define the initial hit counts for these new links, `h_new(k, v_i)`.

**2. The Conservation Principle:**

The links _internal_ to the simplex `C` represent a self-contained local map of probability flow. The total "mass" of this internal map can be quantified by summing the hit counts on all internal links. Let's call this `H_total(C)`:

\[ H*{\text{total}}(C) = \frac{1}{2} \sum*{v*i \in V(C)} \sum*{v*j \in V(C), j \neq i} h*{\text{old}}(v_i, v_j) \]

The factor of `1/2` corrects for double-counting each link. When we split `C` by inserting `k`, this total internal mass `H_total(C)` is the quantity that must be conserved and redistributed among the new links `(k, v_i)`.

**3. The Gradient Preservation Rule:**

To preserve the _structure_ of the map, not just its total mass, the redistribution should be proportional to the existing flow patterns. The importance of a vertex `v_i` within the simplex can be defined by its total internal flow—the sum of hit counts on all links connecting it to other vertices _within_ `C`. Let's call this `F(v_i, C)`:

\[ F(v*i, C) = \sum*{v*j \in V(C), j \neq i} h*{\text{old}}(v_i, v_j) \]

Our rule states that the hit count on a new link `h_new(k, v_i)` should be proportional to this "flow importance" `F(v_i, C)`. To ensure the total mass is conserved, we derive the following rule:

\[ h*{\text{new}}(k, v_i) = \frac{1}{2} F(v_i, C) = \frac{1}{2} \sum*{v*j \in V(C), j \neq i} h*{\text{old}}(v_i, v_j) \]

**4. Verification:**

Does this rule work? Let's check if the sum of the new hit counts equals the total mass of the old internal links, `H_total(C)`.

\[ \sum*{v_i \in V(C)} h*{\text{new}}(k, v*i) = \sum*{v*i \in V(C)} \frac{1}{2} F(v_i, C) = \frac{1}{2} \sum*{v_i \in V(C)} F(v_i, C) \]

Since `Σ F(v_i, C)` is the sum of all internal link hit counts counted from both ends, `Σ F(v_i, C) = 2 * H_total(C)`. Substituting this back in:

\[ \sum*{v_i \in V(C)} h*{\text{new}}(k, v*i) = \frac{1}{2} (2 \cdot H*{\text{total}}(C)) = H\_{\text{total}}(C) \]

The conservation holds exactly.

**Conclusion:** This rule ensures that a vertex `v_i` that was a major nexus of probability flow in the original simplex (i.e., had a high `F(v_i, C)` value) will have a proportionally strong initial connection to the new central node `k`. The local probability gradient is not erased but is smoothly interpolated onto the new, more refined topology.

1. Simplex Learning Rule (Simplex-Native Update)
   The learning rule in Stage 2 was de-parameterized by using the simplex itself to define the update neighborhood, making it more geometrically precise than the k-NN approach in Stage 1.
   For each data point x:
   Find Winning Simplex: First, the single winning simplex C that contains the point x is identified. This process is accelerated by the custom rho-index built for that subspace in Stage 1. The vertices of this simplex C become the exclusive neighborhood for the update.
   Partition Error: The error vector e is calculated. This error is partitioned into a kinetic component e_k and a potential component e_p. This partitioning can be based on a rigidity metric derived from the kinetic incoherence (H) of the simplex's vertices.
   Distribute Error: The kinetic error e_k is distributed among the nudge (n) vectors of the simplex's vertices, and the potential error e_p is distributed among the stress (s) vectors. The distribution is weighted by rank: the vertex closest to x receives the largest share (e.g., 1/2), the second-closest receives the next largest (e.g., 1/4), and so on, following a geometric decay sequence.
   This ensures that the impulse from a data point is localized to its containing tile on the manifold and assigned most strongly to the vertex primarily responsible for the error.
2. Simplex Splitting Rule (Tension-Based)
   Topological creation is elevated from a single-node decision to a simplex-level decision, based on a more nuanced understanding of stress.
   Calculate Stress Vectors: A simplex C continuously monitors the state of its vertices. It calculates two virtual stress metrics from its vertices' s vectors:
   Net Translational Stress (E_C): The vector sum of all its vertices' s vectors. A large magnitude indicates the simplex is being pulled cleanly in one direction.
   Torsional Stress (T_C): The variance of the directions of its vertices' s vectors. A high value indicates the simplex is being twisted, a sign of unmodeled curvature.
   Splitting Condition: A simplex performs a topological split only if it is under high brittle stress. This condition is met when:
   The Torsional Stress T_C is low (meaning the force is coherent).
   AND the magnitude of the Net Translational Stress ||E_C|| is high (exceeding a threshold).
   Action: If these conditions are met, the simplex manages the split itself, for instance by splitting along its longest edge and creating a new node. This is more robust than a single node making a unilateral decision.
3. Simplex Deletion Rule (Simplex-Arbitrated Pruning)
   Node deletion is also a formal, multi-stage process arbitrated by the simplexes, preventing the removal of geometrically critical nodes.
   Initiation: A simplex C can initiate a prune vote on one of its vertices, i, if that node is a significant statistical underperformer relative to its peers within that specific simplex.
   Geometric Veto (Reconstruction Error Test): This is the crucial safeguard. Before voting, the simplex checks if node i is essential to its structural integrity. It does this by:
   Calculating the affine subspace (the "plane") spanned by its other vertices.
   Projecting node i onto this subspace to find its reconstruction i'.
   If the reconstruction error ||i - i'|| is large, it means node i is a "tent pole" holding the simplex up in a higher dimension. The prune vote is vetoed.
   Voting and Consensus: If the node is not geometrically critical (the reconstruction error is small), the simplex casts its "prune" vote and notifies all other simplexes that contain node i. The node is only deleted if a consensus (e.g., >75%) of its containing simplexes agree to the prune.

=======

The "Ideal" but Impractical Method: Exponential Distance Decay
In an ideal world, for a data point x, we would want to update every node i in the local neighborhood with a share of the error e that is proportional to its influence on x. A standard and principled way to model this influence is with a Gaussian (or exponential) kernel based on distance.
Influence of node i on point x: Influence_i = exp(-||w_i - x||² / 2σ²)
Weight for node i: weight_i = Influence_i / Σ_j Influence_j
Update for node i: update_i = weight_i \* e
The Problem: This method is computationally expensive. It requires calculating the distance from x to every node in the neighborhood and then performing several exponential calculations for every single data point. It is also sensitive to the choice of the bandwidth parameter σ.
The Practical, Robust Approximation: Rank-Order Geometric Decay
Our method replaces this complex calculation with a much simpler, faster, and more robust heuristic.
Rank the Nodes: We find the closest node (BMU_1), second closest (BMU_2), etc. This is a simple sorting operation on a small number of nodes.
Assign Weights by Rank: We assign weights according to a fixed geometric series: 1/2, 1/4, 1/8, ....
The Justification: Why This Approximates the Ideal
The key insight is that, on average, over many data points, the rank order of the nodes is a very good proxy for their distance.
Consider a small, stable region of the manifold.
The data points are arriving randomly within this region.
A node v_A is, on average, closer to the center of this region than another node v_B.
Therefore, over many samples, v_A will be BMU_1 more often than v_B. v_A will be BMU_2 more often than v_B is BMU_1, and so on.
The distribution of a node's rank over many samples follows a distribution that is, itself, shaped like an exponential decay curve with respect to its distance from the local center of mass.
By assigning the error shares based on rank, we are effectively integrating over all the possible positions of x and assigning to each node a share of the error that is proportional to its expected influence over the local region.
In short:
The Exponential Distance Decay calculates the precise influence for a single data point x.
The Rank-Order Geometric Decay, when averaged over many data points, converges to the same result. It is a form of stochastic approximation or rank-order averaging that replaces a complex, continuous calculation with a simple, discrete, and computationally trivial one.
This is why it's a valid and powerful approximation. It is less precise for any single update, but it is statistically correct in the long run, and it is orders of magnitude faster to compute, which is the essential trade-off at the heart of the efficient Proteus learning engine.
