# System Components Requiring Special Oversight

This document outlines the components within the Mimir architecture that require special consideration during operation. These modules fall into three primary categories:

1.  **Black Box Components:** Modules whose internal operations are not designed to be human-readable. They are functionally contained and judged by the quality of their outputs, not by inspecting their internal states.
2.  **Glass-Box Components:** Modules whose internal operations may be complex, but which produce semantically structured and inspectable outputs (like attention maps or fuzzy rules) that allow for a degree of interpretability.
3.  **Strategically Monitored Components:** Modules whose core capacity is a significant architectural feature. Upgrading these components is a strategic investment orchestrated by the cognitive core, rather than a routine, automated adjustment.

A component can belong to multiple categories.

---

### 1. The `Student-V` EGNN (E(2)-Equivariant Graph Network)

- **Category:** Black Box & Strategically Monitored.
- **Why it's a Black Box:** The dense hidden state vectors passed between graph nodes are complex, high-dimensional representations of local geometric context and are not directly interpretable.
- **Why it's Monitored:** The core **hidden state dimensionality** of the EGNN determines its expressive power for visual processing. If the system persistently fails at complex visual tasks, the cognitive core can issue a command to upgrade this capacity.

### 2. The Proposer Networks (Transformer-based)

- **Category:** Black Box & Strategically Monitored.
- **Why it's a Black Box:** The internal state vectors within the Transformer layers are not designed for direct interpretation. Their function is judged by their ability to generate coherent, multi-step proposals.
- **Why it's Monitored:** The core capacity is the **maximum context window (`k_max`)**, which limits the maximum size of a generative chunk proposal. The cognitive core monitors the average effective `k` used during generation. If `k` consistently hits `k_max`, it's a signal that the proposer's capacity is a system bottleneck, prompting a strategic upgrade to a version with a larger context window.

### 3. The "Heavyweight" Student/Verifier Context Engines

- **Category:** Strategically Monitored.
- **Why it's Monitored:** The core capacity of the sequential students (`-T`, `-A`) is the **number of windows in their multi-rate EMA bank**. An insufficient number of windows can lead to failures on tasks requiring long-range temporal understanding. The cognitive core monitors global objectives and can issue a command (e.g., `initiate_upgrade(module='student_t', new_capacity='5_windows')`) to orchestrate an offline upgrade if it detects a persistent capability gap.

### 4. The Attention-Based Aggregators

- **Category:** Glass-Box & Strategically Monitored.
- **Why it's a Glass-Box:** This component is highly interpretable for two key reasons. First, its intermediate **attention weights** are fully inspectable, providing an audit trail of which input primitives the module "paid attention to." Second, the aggregator is trained as the encoder of an autoencoder. Its training objective is to produce a signature from which a corresponding **decoder** can reconstruct the original input. This persisted decoder is a powerful diagnostic tool. Its two primary uses are:
  1.  **Forensic Analysis:** For any past event, a signature can be decoded to see a reconstruction of what the aggregator captured.
  2.  **Predictive Introspection:** For generative or predictive tasks, a _future predicted signature_ can be decoded to produce a "mental image" of what the system is imagining. This is critical for debugging and human oversight.
- **Interfacing with Peripherals:** The system uses a hybrid approach to interface with peripherals.
  - **For simple, well-defined concepts** (e.g., "enemy plane"), the system can directly query the semantic patch vectors within the `Student-V`'s generative process to retrieve coordinates.
  - **For ambiguous, high-level concepts** (e.g., "danger" in a crowd), this direct query is insufficient. In this case, the system relies on the aggregator's decoder. It passes the predicted abstract signature (containing "danger") to the decoder, which generates both a reconstructed "mental image" and, crucially, an **attention map** that highlights which patches in that image are most responsible for the "danger" concept. The coordinates of these highly-attended patches are then passed to the peripheral system.
- **Why it's Monitored:** The core capacity is the **number of learnable latent queries**. This parameter determines how much information can be distilled from a variable-sized input into the fixed-size signature. If the system determines it's losing too much fine-grained detail in perception, it can command an upgrade to an aggregator with a higher number of latent queries.

### 5. The Normalizing Flow Models (`Glow` / `NSF`)

- **Category:** Black Box.
- **Why it's a Black Box:** These models learn a complex, high-dimensional, non-linear function to "flatten" a manifold's geometry. The specific parameters of the learned flow are not directly interpretable.
- **Containment:** This component is considered "safe" because it is fully auditable. Its function is validated by statistical tests (like Distance Correlation) that prove it successfully linearized the space, and it is perfectly invertible, meaning no information is lost during its transformation.

### 6. The Elastic ANFIS-Autoencoders (Input Compression)

- **Category:** Strategically Monitored.
- **Why it's Monitored:** While the growth of these autoencoders is automatic and error-driven, their behavior requires high-level oversight. The cognitive core can monitor their growth rate and reconstruction loss across the system. An autoencoder that grows uncontrollably or fails to converge could signal a problematic, highly non-stationary input stream, which might require strategic intervention (e.g., commanding a re-initialization or altering its growth-factor strategy).
