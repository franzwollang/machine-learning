# Paper 1: Foundational Architecture - Content Plan

## Title

"Mimir: A Reflexive Cognitive Architecture for Temporal Goal-Directed Learning"

## Guiding Principles & Caveats

- **Treat Subsystems as Abstract Contracts:** This paper must treat the memory (EMS) and perceptual (TDE) modules as abstract components or "black boxes." The focus should be on defining the _properties and behaviors_ these components must guarantee for the cognitive architecture to function, rather than detailing their internal implementations.
- **Explain Shared Mechanisms:** This principle is detailed in the "Shared Technical Foundations" section below.

## Shared Technical Foundations (To be detailed in each paper)

This section outlines the core technical methodology underpinning all Mimir components. This paper must introduce these concepts comprehensively as the foundation of the Mimir architecture.

### 1. ANFIS-based Modular Toolkit

Mimir is constructed from a common toolkit of adaptive neuro-fuzzy inference system (ANFIS) based modules. This neuro-fuzzy approach provides a powerful and verifiable blend of neural network learning capacity and the semantic interpretability of fuzzy logic. The key modules are:

- **Elastic ANFIS Autoencoder:** A foundational component used for adaptive, non-linear dimensionality reduction and feature extraction. Its elasticity allows it to adjust its internal complexity (number of fuzzy rules) in response to the data, creating efficient, salient representations.
- **DNF-DNN (Disjunctive Normal Form DNN):** A specialized neural network trained to learn transparent boolean logic rules (specifically, disjunctions of conjunctions) from data. This is achieved via heavy L1-regularization to enforce sparsity and a novel importance mapping technique. This technique isolates multiple high-importance "or" nodes as distinct, interpretable output features, preventing them from being collapsed into a single opaque output.
- **Multi-Headed Mapper Networks:** These are complex transformation modules, typically constructed from DNF-DNNs. They are designed to learn intricate mappings between different high-dimensional vector spaces (e.g., from an old representation `C` to a new one `C'`). Each "head" of the network is a distinct output layer specialized in mapping a specific, disjoint partition of the input vector. This allows the network to learn more robust and nuanced transformations for structured data, where different parts of the data may follow different mapping rules.

### 2. Unified Learning Framework: Local & Global Objectives

Mimir's components are trained using a unified, two-stage strategy that ensures both modular competence and holistic, goal-directed optimization.

- **Local Objectives (Modular Pre-training):** Every module is first trained independently on a local objective function relevant to its specific task. For autoencoders, this is reconstruction error. For DNF-DNNs used as predictors, it is prediction error. For Mapper Networks, it is the mapping error between the transformed source and the target domain. This stage ensures each component is individually functional and efficient.
- **Proteus Integration (Unsupervised Scaffolding):** The ANFIS modules and Proteus instances are deeply interleaved. Proteus instances perform unsupervised discovery of topological structure in the data, creating a meaningful, clustered landscape. The differentiable ANFIS modules are then applied to learn specific functions (e.g., predictions, mappings, classifications) over this discovered structure, using the Proteus-generated topology as an inductive bias.
- **Global Unrolling & Fine-Tuning (End-to-End Optimization):** The entire network of interconnected modules and Proteus instances forms a single, large, end-to-end differentiable computational graph. This graph, often featuring branching and disjoint paths, can be conceptually "unrolled." A global error signal, typically derived from the primary task objective (i.e., the reward signal from the Actor-Critic loop), is then backpropagated through this entire graph. This allows the high-level system goals to fine-tune every underlying component, ensuring their locally-learned functions are cohesively optimized for the agent's overall temporal objectives.

## Core Contribution

This paper introduces the complete Mimir architecture at a high level, establishing the relationships between its core components. The main technical focus will be on the **cognitive core**: the Actor-Critic loop, the `Proteus-Reflexive` model for self-monitoring, and the Symbolic Execution Environment (SEE) that enables goal-directed action.

## Target Audience

General AI/ML conferences and journals (e.g., NeurIPS, ICML, ICLR, JMLR).

## High-Level Structure

### 1. Abstract

- High-level pitch: Introduce Mimir as a novel, integrated architecture.
- Briefly mention the key components: reflexive self-monitoring, robust memory, and a unified learning objective.
- State the core contribution: a demonstration of emergent goal-directed agency from these components.

### 2. Introduction

- **Problem:** Motivate the need for integrated architectures beyond specialized models. Discuss the limitations of systems that lack robust memory or a coherent self-evaluation mechanism.
- **Proposed Solution:** Introduce Mimir as a layered architecture built atop the Proteus engine. Present the high-level diagram showing the separation of concerns.
- **Contributions Summary:**
  - The overall Mimir cognitive architecture.
  - A detailed description of the Actor-Critic based cognitive core.
  - A demonstration of how the system integrates perception, memory, and reflection.
- **Companion Papers:** Explicitly state that the `Akashic Engine` and the `Temporal Dynamics Engine` will be detailed in companion papers, citing them as (in preparation).

### 3. The Mimir Architecture: An Overview

- Present the full architectural diagram.
- **Proteus Engine:** Briefly describe its role as the foundational learning/compression engine.
- **The Akashic Engine:** Explain its _function_ (maintaining persistent entity representations from transient observations) without going into deep implementation details.
- **Temporal Dynamics Engine (TDE) & Proteus-Temporal:** Explain their _function_ (processing sequential data and ensuring short-term coherence).
- **Cognitive Core:** Introduce the key components that will be the focus of this paper: `Proteus-Reflexive`, the Actor-Critic loop, and the `SEE`.

### 4. The Cognitive Core: Mechanism for Agency (Deep Dive)

- **Source Material:** `cognitive_core.md`
- **Proteus-Reflexive:** Detail how it learns "states of mind" from the system's own operational vectors.
- **Actor-Critic Loop:** Provide the full technical details of the Actor, the Critic, the reward signal, and the Temporal Difference error calculation. Explain how this drives the system's behavior.
- **Symbolic Execution Environment (SEE):** Describe its role as the bridge between the cognitive core's decisions and actionable outputs.

### 5. Experiments & Results

- **Goal:** Design experiments to showcase the emergent capabilities that arise from the _integration_ of components.
- **Experiment 1: Goal-Directed Navigation.** A simulated task where the agent must navigate a simple environment to reach a goal, demonstrating the full loop from perception to action.
- **Experiment 2: Self-Correction.** Show how the `Proteus-Reflexive` model identifies a suboptimal internal state (e.g., high prediction error) and how the Actor-Critic loop adjusts cognitive parameters to correct it.

### 6. Discussion

- Analyze the results and their implications.
- Discuss the advantages of this unified architecture (e.g., robustness, interpretability of the reflexive layer).
- Mention limitations and future work.

### 7. Conclusion

- Summarize the key contributions and reiterate the importance of integrated cognitive architectures.
