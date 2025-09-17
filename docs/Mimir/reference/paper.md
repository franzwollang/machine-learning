# A System for Grounding Language in Learned Visual Geometry (Hierarchical Context Architecture)

## 1. Abstract

This document specifies a complete, self-improving system architecture designed to learn a grounded model of language and other modalities. The architecture leverages a symmetric **Teacher-Student bootstrapping** paradigm for all sensory streams (vision, text, audio, etc.). Initially, powerful pre-trained transformer models act as "Teachers" to train dedicated Proteus instances on the geometry of primitive perceptual embeddings. These Proteus models then provide the "ground truth" for training streamlined, Hebbian-based "Student" models. Each student uses a **modality-appropriate, local context-forming mechanism**—an E(2)-Equivariant Graph Network (EGNN) for vision and a multi-rate Leaky Integrator for sequential data—to produce a context-aware data stream. These individual streams are then projected into a **Shared Multimodal Latent Space**, where their fusion provides a unified context that can handle any combination of missing or present modalities. This unified context is the input for a final, high-level Proteus Temporal Engine that discovers abstract, cross-modal relationships.

## 2. Design Philosophy: A Hierarchy of Context

This architecture is built on four core principles that mirror a plausible model of cognition:

1.  **Symmetric Bootstrapping:** The system kickstarts its learning by independently distilling the knowledge from large pre-trained Teacher models for each modality. Proteus is used as the common "distillation engine."
2.  **Modality-Appropriate Local Context:** The system uses different, specialized mechanisms for initial context formation. It respects the spatial, order-invariant, and geometric nature of images (`EGNN`) and the temporal, sequential nature of text, audio, and video streams (a unified family of recurrent cells).
3.  **Unified Multimodal Fusion:** Rather than learning combinatorial pairwise interactions, all modalities are projected into a common latent space. A single, unified context vector is formed via a simple "star-shaped" mixing procedure, providing a scalable and extensible foundation for high-level reasoning.
4.  **Recursive Self-Improvement:** The system can recursively use its own generated outputs as a new "ground truth" to train more refined versions of itself, allowing for continuous improvement beyond its initial teacher.

## 3. The Multi-Stage Training Pipeline

The system is a deep, hierarchical pipeline that progressively builds richer context through a sequence of distinct training stages.

### Stage 1: Primitive Concept Distillation (The Teacher Phase)

This phase runs in parallel for each modality to create foundational concept dictionaries.

1.  **Visual Primitives (`Proteus-V`):** A Vision Transformer (ViT) Teacher generates contextual embeddings for millions of 3x3 image patches. A `Proteus-V` instance is trained on this stream to learn a dictionary of **Base Visual Concept Vectors**.
2.  **Textual Primitives (`Proteus-T`):** An LLM Teacher generates contextual embeddings for millions of trigrams. A `Proteus-T` instance is trained on this stream to learn a dictionary of **Base Textual Concept Vectors**.
3.  **Auditory Primitives (`Proteus-A`):** An Audio Spectrogram Transformer (AST) Teacher generates embeddings for millions of short audio snippets (e.g., 25ms). A `Proteus-A` instance is trained on this stream to learn a dictionary of **Base Auditory Concept Vectors**.

### Stage 2: Unimodal Student Training

This phase trains independent, efficient "Student" models that can reproduce the concepts from Stage 1 using only local context. The core mechanism is a Hebbian-style delta rule where a gating matrix is trained to satisfy the objective: `v_out ≈ v_target`.

1.  **The Visual Student (`Student-V`): An E(2)-Equivariant Graph Network**

    - **Task:** To learn a generative model that produces the correct `BaseVisualConceptVector` for any given 3x3 patch, using its spatial context.
    - **Procedure:**
      - a. **Component Vectors:** For each patch, we define its `v_raw` (a simple fixed representation, e.g., via a 32-D Prismatic SFC map), its `v_context` (derived from its neighbors in the EGNN), and its `v_target` (the correct Base Visual Concept from `Proteus-V`).
      - b. **EGNN for Context:** A k-NN graph is built on patch coordinates. Message passing refines each patch's hidden state (`h_i`) based on its neighbors. The final `v_context` for a patch is derived from this hidden state.
      - c. **Learning & Output:** A gating mechanism learns to transform `v_context` into a filter that shapes `v_raw` to approximate `v_target`. The final output for the whole image is a **global mean pool** over all final patch states, producing the geometry-aware **`v_frame_signature`**.

2.  **The Sequential Students (`Student-T`, `Student-A`): Gated Recurrence**
    - **Task:** To learn generative models for text and audio that produce the correct Base Concept Vector for a given primitive (trigram or audio snippet), using sequential context.
    - **Procedure:**
      - a. **Component Vectors:** For each primitive, we define its `v_raw`, its `v_context` (from the recurrent state), and its `v_target` (from `Proteus-T` or `Proteus-A`).
      - b. **Context State (`v_context`):** The model maintains a short-term memory of its recent outputs using a robust recurrent cell.
      - c. **Learning & Output:** A gating mechanism learns to transform `v_context` into a filter that shapes `v_raw` to approximate `v_target`. The final output is a sequence of context-aware vectors (e.g., `v_T_final`).

### Stage 2.5: Mechanism Refinements and Upgrades

The baseline mechanisms for the student models are simple but can be upgraded to enhance performance while remaining Hebbian-friendly.

**1. Upgrading the Sequential Context (Recurrent Cell):**
The simple leaky integrator is a solid baseline. A low-cost, high-impact upgrade is to use a **multi-rate EMA bank**, or a "mixture of decays."

- **Mechanism:** Instead of a single context vector, the student maintains several (e.g., 3), each with a different decay rate `λ`, and concatenates the results.
- **Benefit:** This allows the model to capture context over multiple time scales simultaneously. For even better performance, this can be upgraded to an **RWKV-style WKV recurrence**.

**2. Upgrading the Gating Function:**
The simple bilinear fusion `v_out = v_raw ◦ (v_context ⋅ W_gate)` is effective but limited. Two low-cost upgrades are possible:

- **Add a Learned Bias Gate:** `v_out = v_raw ◦ g_gate + b_gate`. This allows the context to additively influence the output.
- **Two-Head Gating:** `v_out = (A * v_raw) ◦ (B * v_context)`. This allows for cross-dimensional coupling before the final gating.

### Stage 3: Shared Context Bootstrapping (Learning the SMLS Projections)

This crucial stage teaches the mature unimodal experts to communicate by learning to map their outputs to a **Shared Multimodal Latent Space (SMLS)** of a fixed dimension (e.g., `D_smls = 512`).

1.  **The Projection Matrices:** We introduce new learnable components, `W_proj_V` and `W_proj_T`.
2.  **Contrastive Learning:** These matrices are trained using the `(image, text)` pair dataset and a Hebbian-friendly contrastive objective. For each pair:
    - a. **Get Unimodal Signatures:** The image is processed by `Student-V` to get its `v_frame_signature`. The text is processed by `Student-T` to get an aggregated `v_text_signature`.
    - b. **Project to SMLS:** `z_V = v_frame_signature ⋅ W_proj_V` and `z_T = v_text_signature ⋅ W_proj_T`.
    - c. **Attractive Update (Positive Pair):** The projections `z_V` and `z_T` are pulled closer together via a delta rule.
    - d. **Repulsive Update (Negative Pair):** `z_V` is pushed away from the projection of a text signature from a random negative pair.
3.  **Result:** Trained projection matrices that map modality-specific concepts to a shared "Rosetta Stone" space.

### Stage 4: High-Order Learning and Inference

This is the final operational phase.

1.  **Star-Shaped Mixing:** For any given data, the system gets the outputs from all **currently active** Student models, projects them into the SMLS, and computes the **Unified Multimodal Context Vector (`v_Context_Unified`)** by averaging the results. This gracefully handles any combination of missing or present modalities.
2.  **Proteus Temporal Pyramid:** This unified context stream is fed into the high-level Proteus engine (Mode 3/4) to learn the deepest abstract and temporal patterns.

### Stage 5: Recursive Self-Improvement

The entire system can be made self-improving. The outputs of the trained Student models can be used to train new, more refined Proteus instances (e.g., `Proteus-V2`), which in turn are used to train better Student models with upgraded raw vector generators (e.g., via Sparse PCA projection), creating a virtuous cycle of improvement.

## 4. Extension to Exotic Perceptual Systems

The unified perspective of "Iterative Refinement on a Neighborhood Graph" (which underpins the EGNN and recurrent cells) allows this architecture to be conceptually extended to more exotic sensory modalities. The core Proteus engine remains unchanged; only the low-level "Student" model for local context formation needs to be adapted to the data's native structure.

- **Chemosensation (Smell/Taste):** The input is an unordered "bag of molecules." The neighborhood graph is not spatial but **biochemical**. A Relaxation Gating mechanism would be an appropriate frontend.
- **Somatosensation (Touch):** The input comes from receptors on a deformable 2D surface. The neighborhood graph is a **dynamic, non-Euclidean mesh**. Message passing would need to be modulated by the current 3D state of the mesh.
- **Graph-Native Data (e.g., Social Networks):** For data that is already a graph, the neighborhood is explicitly defined. The context for a node can be formed by iteratively passing messages from its direct connections, a process that is the foundation of modern **Graph Neural Networks (GNNs)**.

## 5. Conclusion

This staged, bootstrapping architecture provides a complete, robust, and extensible framework for a perceptually grounded AI. By separating the system into clear stages—unimodal expertise, shared context learning, and high-order temporal analysis—it resolves complex dependencies in a principled way. The use of a contrastive learning objective to train the projections into a shared multimodal space creates a truly unified and scalable foundation for the final Proteus engine to discover deep, abstract concepts from any combination of sensory inputs.
