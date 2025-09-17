# Plan for Figures, Tables, and Algorithms

A strong academic paper uses figures to build intuition, tables to summarize key information, and formal algorithm blocks for precision. This document proposes a list of visual and formal elements to be created for the paper.

## 1. Figures

- **Figure 1: Framework Overview.**

  - **Content:** A high-level diagram illustrating the end-to-end pipeline.
  - **Flow:** `High-Dimensional Data` -> `[Proteus Clustering Engine]` -> `Set of N Clusters` -> `[Ensemble ρ-SFC Builder]` -> `N distinct Recipe Trees` -> `[Query-time Synthesizer]` -> `Global Metric κ_F`.
  - **Purpose:** To give readers a mental map of the entire system and how the different components interact.

- **Figure 2: Hierarchical Cascade.**

  - **Content:** A funnel diagram showing the multi-stage kNN query process.
  - **Flow:**
    - Start with `Query Point` and `Full Dataset (N items)`.
    - `Tier 1 (8-D ρ-index)` filters this to `k_1` candidates.
    - `Tier 2 (16-D ρ-index)` refines this to `k_2` candidates.
    - `Tier 3 (32-D ρ-index)` refines this to the final `k` candidates.
    - `Final Re-ranking (768-D)` produces the final sorted list.
  - **Purpose:** To visually explain the coarse-to-fine search strategy that makes high-dimensional queries feasible.

- **Figure 3: Hierarchical Composite Key.**
  - **Content:** A diagram illustrating the structure of the composite key.
  - **Visual:** Show a key as a concatenation of blocks: `[Block_L0 | Block_L1 | Block_L2]`. Each block contains the `(ChildIndex, FuzzySFCValue)` tuple. Add annotations to explain how the `FuzzySFCValue` is interpolated.
  - **Purpose:** To demystify the structure of the advanced keying strategy and visually explain how it enables hierarchical queries.

## 2. Algorithms

The paper currently includes pseudo-code in a numbered-list format. This should be converted into formal LaTeX algorithm blocks using a package like `algorithm2e`.

- **Algorithm 1: Recipe Tree Query (`SFC_inverse`).**

  - **Content:** Formalize the pseudo-code from Section 5.3. Use standard algorithm keywords like `\KwIn`, `\KwOut`, `\For`, `\If`.
  - **Purpose:** To provide a precise, unambiguous definition of the core query algorithm.

- **Algorithm 2: Hierarchical kNN Query (`Hierarchical_kNN`).**
  - **Content:** Formalize the pseudo-code from Section 3.7.
  - **Purpose:** To provide a clear, step-by-step definition of the high-dimensional query process.

## 3. Tables

- **Table 1: Key Naming Conventions.**
  - **Content:** A simple two-column table clarifying the terminology used throughout the paper.
  - **Rows:**
    - `Ensemble ρ-SFC framework`: The complete indexing architecture.
    - `ρ-Curve`: A single, density-guided curve within the ensemble.
    - `ρ-index`: An index built using one or more ρ-Curves (e.g., a tier in the cascade).
  - **Purpose:** To provide a quick reference that eliminates potential ambiguity for the reader.
