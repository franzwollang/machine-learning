# Paper 1 (Proteus Foundational) — Restructuring Plan

This document is a detailed plan to restructure **Paper 1** into digestible, independently readable sections that build on each other via **progressive disclosure**. The goal is to reduce perceived density by:

- splitting the architecture into coherent “owned objects” (one main concept/data structure per section),
- introducing definitions only when they are contextually needed,
- using conceptual placeholders early and deferring formalism until the reader has the mental model to understand it,
- placing pseudo-code only after the narrative loop is predictable.

The plan below is written as an implementation-ready outline: each planned section includes its purpose, the reader-questions it answers, what can be introduced, what must be deferred, and where equations/pseudo-code belong.

---

## Design Principles (Progressive Disclosure Rules)

### Rule PD-1: Name → Role → Payoff (before math)
When a new construct appears, introduce it in this order:

1. **Name**: what we call it (scaffold, torsion ladder, dual flow, …).
2. **Role**: what problem it solves (proposal signal, representation upgrade, generative field, …).
3. **Payoff**: what capability it unlocks (scale discovery, evidence gating, queries/sampling, …).
4. **Mechanics**: how it works (only now introduce symbols/equations).

### Rule PD-2: One primary “owned object” per section
Each section should have one primary object that it “owns.” Other objects may be mentioned only as:

- **promises** (“introduced later in §X”), or
- **conceptual placeholders** (“an evidence test decides acceptance; formalized in §X”).

### Rule PD-3: Symbols appear only after the object exists
Do not define symbols for an object before the reader has been told:

- what the object is, and
- why the paper needs it *now*.

Example: don’t define torsion \(\Omega_S\) in Stage 2 overview; define it in the torsion audit section where curvature failure is introduced.

### Rule PD-4: Pseudo-code only after the narrative loop is clear
Pseudo-code should appear only when the reader can already predict its structure from the prose.

### Rule PD-5: SI separation rule (“architecture vs. reproduction”)
Main text should include enough detail to understand the architecture and its logic. SI holds:

- derivations,
- default threshold values (unless essential),
- detailed solver/procedure variants,
- implementation details required to reproduce results.

---

## High-Level Restructured Outline (Digestible Pieces)

The current draft has “The Proteus Framework” as one dense section. The plan is to expand this into multiple independent sections that align with how a reader builds understanding.

### Proposed top-level structure

1. **Introduction**
2. **Framework Overview (Proteus in one page)**
3. **Stage 1: The Distributional Scaffold (single-scale)**
4. **Stage 1 Controller: Multiscale Discovery (scale-space + recursion)**
5. **Stage 2: Simplicial Lifting and the Core Objective**
6. **Stage 2 Diagnostics: Torsion Audit (when PL fails)**
7. **Evidence-Based Structure Selection (the accept/reject test)**
8. **Generative Modeling + Inference Interface (the payoff)**
9. **Evaluation**
10. **Limitations and Scaling**
11. **Conclusion**

---

## Definition Inventory (Hard Guardrail)

This table specifies where each object/symbol is first introduced **formally**.

| Object / definition | First formal introduction | Mention allowed earlier? |
|---|---:|---|
| Scaffold (node–edge graph), node vs edge semantics | §3 | Yes: conceptual (“coarse scaffold”) in §2 |
| Node state: \(\mathbf m_i, \mathbf s_i, \sigma_i^2, \rho_i, \mathbf u_i\) | §3 | No symbols before §3; can say “local moments / directions” in §2 |
| Neighborhood size \(k\), rank weights, fractional hits | §3 | Mention “nearest neighbors” conceptually in §2 |
| Scale knob \(\tau\) meaning (variance cap / coarseness) | §4 | In §2: “candidate scales” only |
| Control mapping: \(s_{control}\), \(\tau_{global}\), \(\tau_{local,i}\), \(D_{subspace}\), \(d_{final,i}\) | §4 | Not before §4 |
| Scale-space response \(\Phi(\tau)\) | §4 | Not before §4 |
| Scale recursion algorithm (meta-controller) | §4 | Summarize as “search & recurse” in §2 |
| Simplicial complex as representation, simplex adjacency | §5 | Mention “lift to simplices” in §2 |
| Simplex equilibrium objective (formal) | §5 | Mention “refinement objective” conceptually in §2 |
| Node vs simplex equilibrium relationship | §5 | No earlier |
| Torsion: \(\Omega_S\), \(\kappa_S\), torsion ratio \(R_S\) | §6 | Mention “curvature audit” conceptually in §2 |
| Shape metric \(Q_S\) | §6 | No earlier |
| Free-energy proxy equation \(F(\mathcal R;M)\), region \(\mathcal R\) | §7 | Mention “local evidence test” conceptually in §2–§3 |
| Dual flow quantities: face pressures \(p_f\), dual graph \(\mathcal G_{dual}\), objective | §8 | Mention “probability field reconstruction” conceptually in §2 |
| Inference interface: density query, sampling, conditional synthesis, constraint extraction | §8 | Mention as “capabilities” (no mechanics) in §1–§2 |
| Warp strategy statistic \(P_\kappa\) | §8 (or SI) | Mention “optional warps” in §2 |

---

## Source Corpus (Current + Reference) and Status

This plan is grounded in the following existing materials. The intent is **not** to copy large chunks verbatim, but to re-use *the right pieces in the right places* under the progressive-disclosure rules.

### Canonical “current” sources (preferred)
- **Main draft (current narrative)**: `docs/Proteus/paper_1_foundational/paper.tex`
  - Status: **canonical for what is currently claimed in the paper**.
  - Use for: existing prose that will be redistributed into the new section structure; existing equations already in main text.
- **Supplementary Information (current formalism + defaults)**: `docs/Proteus/paper_1_foundational/SI.tex`
  - Status: **canonical for derivations, default thresholds, and “gauntlet” details**.
  - Use for: moving dense details out of the main narrative while keeping them citable and consistent.

### Reference / older-spec sources (background + selective re-use)
- `docs/Proteus/paper_1_foundational/reference/content_plan.md`
  - Status: **outdated as a structure plan** (superseded by this plan), but useful as a sanity-check that Paper 1 stays focused on the “minimal core.”
  - Use for: scope discipline and section purpose framing only.
- `docs/Proteus/paper_1_foundational/reference/stage_1.md`
  - Status: older Stage 1 spec; much of it is now reflected in SI (e.g., EWMA \(\alpha\), dual-rate motion, pruning gauntlets).
  - Use for: crisp “design philosophy” language and the *data-structure inventory*; treat numeric thresholds and long gauntlet procedures as **SI-only** material.
  - Contains also some **out-of-scope add-ons** (see “Outdated/out-of-scope” section below).
- `docs/Proteus/paper_1_foundational/reference/stage_2.md`
  - Status: older Stage 2 spec; largely consistent with the current main draft and SI; useful as a “systems spec” for how Stage 2 hangs together.
  - Use for: pipeline ordering, data-structure expansions (simplex fields), and compact explanations of torsion ladder actions and dual flow.
- `docs/Proteus/paper_1_foundational/reference/addendum.md`
  - Status: focused theoretical note; overlaps with current main draft’s simplex-vs-node equilibrium argument.
  - Use for: improving the explanation in the future §5 (“Simplicial lifting + objective”) and as a source for SI proof sketches / signatures.
- `docs/Proteus/paper_1_foundational/reference/redraft.md`
  - Status: a long scale-space optimization report; portions are already integrated into `paper.tex` and SI S2.1.
  - Use for: derivation backing and alternative phrasing; do **not** copy raw content (formatting is inconsistent and includes tangents).

### Optional / future-work source (not core Paper 1)
- `docs/Proteus/paper_1_foundational/bnp_design_implications.md`
  - Status: future enhancements inspired by BNP correspondence in SI S13.
  - Use for: **future work** and/or a discussion appendix; do not let it expand the core architecture sectioning.

---

## Detailed Section-by-Section Plan

Each section below includes:

- **Primary object owned**
- **Reader questions**
- **Content goals**
- **Allowed introductions**
- **Must defer**
- **Where pseudo-code/equations go**
- **Exit criteria** (what the reader should understand when leaving)

---

## 1. Introduction (keep lean; no symbol dump)

### Owns
Problem framing, SoTA positioning, contributions, roadmap.

### Reader questions
- Why do we need this at all?
- What is missing in the current toolchain?
- What does Proteus aim to deliver (without details)?

### Content goals
- Establish “fragmented toolchain” framing.
- State high-level promise: explicit multiscale geometry + evidence-gated structure + generative inference.
- Contributions list and roadmap.

### Allowed introductions
- Names: “Stage 1 scaffold”, “Stage 2 simplicial complex”, “evidence-gated edits”, “probability field”, “optional local warps”.
- Only conceptual “propose–test–settle” phrasing (no formula).

### Must defer
- Any symbols (\(\tau\), \(\Phi\), \(\Omega\), \(F\), …).
- Any pseudo-code.
- Any threshold/default values.

### Exit criteria
Reader can explain Proteus in one sentence and can list 2–4 contributions.

---

## 2. Framework Overview — “Proteus in one page”

### Owns
The **pipeline contract**: what Proteus learns, what artifacts exist, and how the two stages relate—*without* dumping internal mechanics.

### Reader questions
- What are the stages, in one sentence each?
- What objects exist after training (what is the “model”)?
- What is the single unifying control principle?
- What can I do with the trained model (high-level only)?

### Content goals
- Provide the reader a stable mental map they can carry through the rest of the paper.
- Establish a single throughline: **propose → test → settle**, but keep it conceptual.
- Make the output artifacts concrete enough that later sections can refer back to them.
- Introduce the three inference-time “modes” only as **capabilities** (no mechanics).

### Allowed introductions (conceptual placeholders only)
- **Stage 1**: “build a scaffold at candidate scales; stabilize; summarize evidence.”
- **Stage 2**: “lift scaffold to simplicial complex; refine; reconstruct probability field.”
- **Artifacts**: scaffold hierarchy, chosen characteristic scales, simplicial complex, probability field, optional warps.
- **Control loop**: propose/test/settle as a narrative device (no formula).
- **Capabilities**: semantic querying, conditional synthesis, constraint extraction (no algorithms yet).

### Must defer
- All concrete Stage 1 state definitions: \(\mathbf m_i,\sigma_i^2,\rho_i,\mathbf u_i\), rank weights, etc. (defer to §3).
- All scale-space definitions: \(\tau, \Phi(\tau)\), control mapping (defer to §4).
- All Stage 2 formalism: simplex equilibrium math, torsion, dual flow (defer to §5–§8).
- The free-energy proxy equation \(F(\mathcal R;M)\) (defer to §7).

### Where pseudo-code / equations go
- **No equations** other than perhaps a tiny “artifact list” notation (e.g., “learned model = (complex, field, optional warps)”).
- **No pseudo-code**.

### Suggested internal structure (subsections)
- **2.1 What problem we solve (one paragraph)**: explicit geometry + evidence-gated structure + generative inference.
- **2.2 Two-stage pipeline (one paragraph)**: Stage 1 “where/at what scale” vs Stage 2 “spend compute locally for fidelity.”
- **2.3 Learned artifacts (bullets)**: scaffold hierarchy; selected scale(s); simplicial complex; probability field; optional warps.
- **2.4 Unifying principle (one paragraph)**: propose/test/settle, with examples of “proposal signals” but no details.
- **2.5 Capabilities (3 bullets)**: semantic querying, conditional synthesis, constraint extraction (only as promises; mechanics later in §8).

### Exit criteria
Reader can answer:
- “What do I get after training?”
- “Why are there two stages?”
- “What is the single rule that governs discrete structure changes (conceptually)?”

---

## 3. Stage 1: The Distributional Scaffold (single-scale view)

### Owns
The **scaffold representation** and its **single-scale learning loop**.

### Reader questions
- What is a “scaffold” concretely (what are nodes/edges)?
- What state is stored per node and why?
- What happens when one sample arrives (how evidence is routed)?
- How do we propose local structure changes (split/prune) at this stage?
- When do we stop at a fixed scale?

### Content goals
- Make Stage 1 feel like a coherent architecture, not a list of tricks.
- Give the reader a “just-so” story: **state → routing → proposals → evidence gate (conceptual) → stabilize**.
- Establish “what counts as evidence” at scaffold level (hits, transitions), but defer the acceptance math.

### Allowed introductions (first formal definitions happen here)
Formalize:
- **Data structure**: node–edge graph; what a node represents (Voronoi region proxy); what an edge represents (adjacency + transition evidence).
- **Per-node tracked state**: \(\mathbf m_i\), \(\mathbf s_i\), \(\sigma_i^2\), \(\rho_i\), \(\mathbf u_i\); EWMA idea (not full derivation).
- **Neighborhood update**: \(k\); rank-ordered weights (describe; exact sequence optional).
- **Proposal triggers**: variance exceedance proposes split; weak support proposes prune (no free-energy equation yet).
- **Stabilization**: qualitative “equilibration of local statistics,” with numeric default in SI or a “defaults” table.

### Must defer
- Scale-space: \(\tau\) as global knob, \(\tau_{local,i}\) mapping, \(\Phi(\tau)\) response (defer to §4).
- The formal acceptance test (free-energy proxy) and region definition \(\mathcal R\) (defer to §7).
- Any Stage 2 concepts (simplices, torsion, dual flow).

### Where pseudo-code / equations go
Recommended:
- **Mini pseudo-code box** (optional): `UpdateScaffoldWithSample(x)` at the end of §3. Keep it local (no recursion).
- **Equations**: define \(\sigma_i^2 = \mathbf s_i - \mathbf m_i^2\) (or the correct vector/scalar form used); define \(\rho_i\) (incoherence). Keep them minimal.
Defer all other formulas to later sections/SI.

### Suggested internal structure (subsections)
- **3.1 What Stage 1 is for**: “fast front end” and what it must produce for Stage 2 and for scale selection.
- **3.2 Scaffold as a data structure**:
  - What nodes represent, what edges represent.
  - What statistics are attached to nodes/edges (high-level list).
- **3.3 Node state: what we track and why** (motivations):
  - Mean residual \(\mathbf m_i\) → coherent drift signal.
  - Variance \(\sigma_i^2\) → unresolved mass / heterogeneity signal.
  - Incoherence \(\rho_i\) → normalize drift by uncertainty (stability signal).
  - Principal direction \(\mathbf u_i\) → where to place new capacity.
- **3.4 Per-sample routing and updates**:
  - k-NN selection.
  - Rank weights (interpretation: robust, localized credit assignment).
  - Accumulation of hit/transition evidence.
- **3.5 Proposed discrete edits (conceptual)**:
  - Split proposal and placement heuristic.
  - Prune proposal and safety checks (conceptual).
  - State explicitly: “accepted only if a local evidence test approves; formalized in §7.”
- **3.6 Stabilization at a fixed scale**:
  - What “equilibrated” means in words.
  - Only one criterion in main text; defaults in SI.
- **3.7 Outputs handed to the controller**:
  - What the scale controller needs from this stage (scaffold + routed evidence + stability status).

### Exit criteria
Reader can sketch Stage 1 as a loop and can name what state it tracks and why.

---

## 4. Stage 1 Controller: Multiscale Discovery (scale-space + recursion)

### Owns
The **scale-space controller**: how Stage 1 is orchestrated over \(\tau\), how characteristic scales are found, and how recursion is triggered and bounded.

### Reader questions
- What is the “scale” parameter in Proteus and what does changing it do?
- How do we decide which scales are meaningful (what is the response signal)?
- How does the system avoid overfitting small/noisy partitions?
- What does the recursion actually do, and when does it stop?

### Content goals
- Provide a clean conceptual mapping: \(\tau\) controls coarse vs fine; the controller searches \(\tau\) and recurses only when supported by evidence.
- Make scale-space feel like a controller around Stage 1, not a separate topic.
- Ensure the reader sees that scale selection is **evidence-limited**, not infinite recursion.

### Allowed introductions (first formal definitions happen here)
Formalize:
- **Scale parameterization**: introduce \(\tau\) as the growth/variance cap; relate to smoothing scale conceptually.
- **Control mapping**: \(s_{control}\), \(\tau_{global}\), \(\tau_{local,i}\), and their relation to \(D_{subspace}\) and \(d_{final,i}\).
- **Response**: define \(\Phi(\tau)\) as a scalar signal measuring structural salience from routed evidence (keep derivation in SI).
- **Grid and refinement**: geometric grid in \(\tau\); local maxima bracketing; Bayesian refinement.
- **Recursion criteria**: minimum evidence/sample thresholds; how partitions are defined (by scaffold routing).

### Must defer
- Stage 2 representations (simplices), torsion, dual flow.
- Free-energy proxy \(F(\mathcal R;M)\) (still deferred to §7).

### Where pseudo-code / equations go
- **Algorithm box**: `FindScales(X)` belongs here (this is the first moment where recursion is contextual).
- **Equations**:
  - \(\tau_{global} = -D_{subspace}\log(1-s_{control})\) and \(\tau_{local,i} = -d_{final,i}\log(1-s_{control})\).
  - Define \(\Phi(\tau)\) (one-line definition), with “details in SI.”

### Suggested internal structure (subsections)
- **4.1 Why scale search is needed** (reader motivation).
- **4.2 The scale knob \(\tau\)**: what increasing/decreasing \(\tau\) does to the scaffold.
- **4.3 Control-to-threshold mapping**: the 4-tier scale representation and why it exists (decouple optimizer from local adaptation).
- **4.4 Scale response \(\Phi(\tau)\)**: what it measures, why peaks correspond to characteristic scales.
- **4.5 Grid + optimizer**: geometric grid, bracketing maxima, GP refinement.
- **4.6 Recursive decomposition**:
  - How the scaffold induces partitions.
  - Evidence gates for recursion (minimum samples / routed evidence).
  - Stopping criteria (no maxima / insufficient evidence / depth limits if any).

### Exit criteria
Reader can articulate:
- what \(\tau\) controls,
- how \(\tau^*\) is selected,
- why recursion doesn’t explode,
- what gets passed into Stage 2.

---

## 5. Stage 2: Simplicial Lifting and the Core Objective

### Owns
The **representation upgrade** (graph → simplicial complex) and the **core refinement objective** (simplex equilibrium), at a fixed scale \(\tau^*\).

### Reader questions
- Why isn’t the scaffold enough? What do simplices buy us?
- What exactly changes when we lift to a simplicial complex?
- What is the refinement objective at this stage (conceptually and formally)?
- How does simplex equilibrium relate to node-level balance (intuition)?
- What is the handoff from Stage 1 to Stage 2 (warm-start conceptually)?

### Content goals
- Justify the simplicial lift as an architectural necessity for a generative model.
- Introduce “simplex equilibrium” as the central Stage 2 goal before adding audits or flows.
- Keep initialization mechanics minimal in main text; preserve interpretability of why the representation is right.

### Allowed introductions (first formal definitions happen here)
Formalize:
- **Simplicial complex** as the Stage 2 model domain (only what is needed: vertices/nodes, simplices, adjacency).
- **Simplex equilibrium**: define what “equal mass per simplex” means and why it is a natural objective.
- **Node equilibrium vs simplex equilibrium** relationship (your current argument can live here).
Mention (conceptually):
- Warm-start: “carry forward nodes/edges/stats; initialize complex from graph” (mechanics in SI).

### Must defer
- Torsion formalism \(\Omega_S\), torsion ladder regimes (defer to §6).
- Dual flow quantities \(p_f\), \(\mathcal G_{dual}\), reconstruction objective (defer to §8).
- Formal free-energy acceptance proxy \(F(\mathcal R;M)\) (defer to §7).
- Warp strategy details (defer to §8 or SI).

### Where pseudo-code / equations go
- **Equations allowed**:
  - Formal statement of node equilibrium and simplex equilibrium (e.g., \(P(V_i)\approx 1/N\) and \(P(C_j)\approx 1/M\)).
  - If you want to keep “free-energy objective” as a phrase, do not write the full proxy here; reserve it for §7.
- **Pseudo-code**: avoid in main text; optionally add a 5–8 line “Stage2WarmStart” sketch only if readers truly need it to understand later sections.

### Suggested internal structure (subsections)
- **5.1 From scaffold to complex: why lift?**
  - Scaffold gives adjacency; simplices give volumes/patches needed for generative fields.
- **5.2 What is a simplicial complex in Proteus?**
  - Minimal definitions: vertices are scaffold nodes; simplices are local cliques/patches.
- **5.3 The refinement target: simplex equilibrium**
  - Define it and interpret it.
- **5.4 Why simplex equilibrium is compatible with node-level balance**
  - Keep your “negative feedback loop” story here; cite SI for proof sketch.
- **5.5 Stage 2 workflow at a glance (no details)**
  - “Initialize → refine → audit (later) → accept edits (later) → reconstruct field (later).”

### Exit criteria
Reader understands:
- why simplices are introduced,
- what simplex equilibrium is,
- why Stage 2 exists at all, separate from audits/flows.

---

## 6. Stage 2 Diagnostics: Torsion Audit (when PL fails)

### Owns
Torsion as a **proposal signal**: detecting when the PL approximation fails and triggering localized corrective interventions.

### Reader questions
- What failure mode is torsion detecting (and why is it inevitable)?
- What is torsion in this discrete setting?
- How does the ladder decide between “do nothing / refine geometry / add non-linear capacity”?
- What safeguards prevent mesh degeneration?

### Content goals
- Establish a clear separation: torsion is a **diagnostic / proposer**, not the accept/reject mechanism.
- Provide a concrete, local signal that a reader can understand without the full generative machinery.

### Allowed introductions (first formal definitions happen here)
Formalize:
- Discrete torsion 2-form \(\Omega_S\), torsional stress \(\kappa_S\), torsion ratio \(R_S\).
- The ladder as regimes (keep numeric defaults in SI unless essential).
- Shape-quality guard \(Q_S\) (introduced only when explaining split rejection).
Mention (conceptual):
- Warp attachments as an escalation option (details deferred to §8/SI).

### Must defer
- Free-energy proxy acceptance equation (defer to §7).
- Dual flow reconstruction details (defer to §8).

### Where pseudo-code / equations go
- **Equations allowed**:
  - \(\Omega_S = M^\top E - E^\top M\)
  - \(\kappa_S = \|\Omega_S\|_F\), \(R_S = \kappa_S/\tau^*\)
  - \(Q_S\) definition (inradius/circumradius) if kept in main text.
- **Pseudo-code**: optional tiny “AuditPatch(S)” conceptual box (not required).

### Suggested internal structure (subsections)
- **6.1 Why PL fails**: curvature + residual field non-conservativity.
- **6.2 Torsion definition**: the discrete 2-form and interpretation.
- **6.3 The torsion ladder (triage)**: regimes and their actions.
- **6.4 Shape-quality guard**: why it matters and how it constrains edits.

### Exit criteria
Reader can answer: “What is torsion measuring, and what actions does it propose?”

---

## 7. Evidence-Based Structure Selection (the accept/reject test)

### Owns
The **disposer**: one local evidence criterion that arbitrates all discrete edits across the system.

### Reader questions
- What exactly is the “evidence test” hinted at earlier?
- What is the affected region \(\mathcal R\) and why local scoring is valid?
- How is model complexity penalized?
- How do we avoid double-counting evidence with dual flow?

### Content goals
- Provide the first formal statement of the acceptance test.
- Tie back to earlier proposals:
  - Stage 1 split/prune proposals,
  - Stage 2 torsion-driven interventions,
  - optional warp attachments.
- Make it explicit that this is a local model-selection mechanism under finite data.

### Allowed introductions (first formal definitions happen here)
Formalize:
- Affected region \(\mathcal R\).
- Transition likelihood restricted to \(\mathcal R\).
- Prior on mass field (by example).
- Complexity term and effective sample size \(N_{\mathrm{eff},\mathcal R}\).
- Free-energy proxy \(F(\mathcal R;M)\) equation.

### Must defer
- Full derivation of the proxy and router mechanics (SI).
- Dual flow solver details (SI; only the high-level objective in §8).

### Where pseudo-code / equations go
- **Equation (required)**: the free-energy proxy \(F(\mathcal R;M)\).
- **Pseudo-code (optional)**: a 6–10 line “EvaluateEdit(edit)” to show propose/test/settle pattern.

### Suggested internal structure (subsections)
- **7.1 Why a single disposer is needed**: unify diverse proposals.
- **7.2 Defining the local region \(\mathcal R\)**: scope of change.
- **7.3 The free-energy proxy**: equation + interpret terms.
- **7.4 Acceptance protocol**: warm-start, settle locally, compare \(F\), accept/reject.
- **7.5 No double counting**: how dual flow is conditioned on accepted topology.

### Exit criteria
Reader understands the accept/reject mechanism independently of torsion and dual flow.

---

## 8. Generative Modeling + Inference Interface (the payoff)

### Owns
The **probability field** and the **inference-time interface**: querying, sampling, conditional synthesis, and constraint extraction.

### Reader questions
- What field is reconstructed, and on what domain?
- How does dual flow turn training evidence into a usable probability field?
- How do I compute density / likelihood for a query?
- How do I sample, including under constraints?
- Where do warps enter, and when are they needed?

### Content goals
- Make the payoff explicit: one trained representation supports multiple inference tasks.
- Present dual flow as a reconstruction step that produces a divergence-consistent field.
- Separate “what is the object” from “how to compute it” (solver details in SI).

### Allowed introductions (first formal definitions happen here)
Formalize:
- Face pressures \(p_f\), empirical tallies \(\hat p_f\), dual graph \(\mathcal G_{dual}\).
- The quadratic objective enforcing conservation.
Define inference-time operations at a high level:
- locate simplex, barycentric coordinates, evaluate density.
- sampling: sample simplex by mass; sample barycentric coordinates; map to data space.
Introduce:
- Constraint extraction as an aggregation over routed mass/transition structure (mechanics can remain high level).
Warp strategy:
- define torsion coverage statistic \(P_\kappa\) (if kept in main text) and describe global vs patch-wise choices; training details in SI.

### Must defer
- Solver implementation details (belief propagation vs Gauss–Seidel) to SI.
- Patch identification heuristics and training budgets to SI (only conceptual in main text).

### Where pseudo-code / equations go
- **Equation (allowed)**: dual flow objective (quadratic) and the face pressure update.
- **Pseudo-code (recommended, high-level)**:
  - `QueryDensity(x)`
  - `Sample()`
  - `ConditionalSample(constraints)` (even if it just says “use rejection / constrained simplex selection” at a conceptual level)
  Keep these short and interface-level.

### Suggested internal structure (subsections)
- **8.1 What is reconstructed**: conservative flux/pressure field → probability field.
- **8.2 Dual flow mechanics (minimal)**: face updates + conservation objective.
- **8.3 Querying**: density/likelihood, anomaly, membership.
- **8.4 Sampling and conditional synthesis**: sampling pipeline and constraint conditioning.
- **8.5 Constraint extraction**: what is extracted and from what evidence.
- **8.6 Optional warps**: global vs patch-wise strategy; how it composes with the PL core.

### Exit criteria
Reader can describe how Proteus supports density queries and generation from the learned complex.

---

## 9. Evaluation (model quality, not task-specific performance)

### Owns
Experimental validation of the learned representation: topology, geometry, and generative fidelity.

### Reader questions
- Does the learned complex match ground-truth topology?
- Does it produce good density estimates / samples?
- Which components are necessary (ablation story)?

### Content goals
- Ensure each metric maps back to a specific architecture section:
  - PH/topology ↔ §5–§6
  - reconstruction ↔ §5–§6
  - log-likelihood ↔ §8 (+ §7 for selection)
  - MMD ↔ §8
- Use ablations that correspond to owned objects:
  - remove scale controller (§4),
  - remove torsion audit (§6),
  - remove dual flow (§8),
  - remove evidence disposer (§7) if feasible.

### Allowed introductions
- Datasets, metrics, protocols.

### Must defer
- Hyperparameter tables and implementation specifics to SI.

### Where pseudo-code / equations go
- None expected in main text.

### Exit criteria
Reader believes the architecture pieces matter and are validated by appropriate metrics.

---

## 10. Limitations and Scaling

### Owns
Practical constraints and computational envelopes.

### Content goals
- Be explicit about failure regimes (ambient dimension, sparse data, anisotropy).
- Keep complexity analysis tied to objects:
  - Stage 1 per-sample updates ↔ §3
  - scale controller cost ↔ §4
  - torsion audit cost ↔ §6
  - dual flow cost ↔ §8

### Exit criteria
Reader can assess feasibility and knows where the bottlenecks are.

---

## 11. Conclusion

### Owns
Re-state contribution as a coherent integrated toolchain, without overclaiming.

### Content goals
- Return to “fragmented toolchain” framing.
- Summarize the owned objects as a coherent stack.
- Identify immediate future work that aligns with the architecture (e.g., streaming, better warps, more formal guarantees).

---

## Pseudo-code Placement Map (by section)

This is where algorithm boxes should live to avoid premature formalism.

- **§3 (optional)**: `UpdateScaffoldWithSample(x)` — local update only; no recursion.
- **§4 (required)**: `FindScales(X)` — the meta-controller (already exists in current draft).
- **§5 (optional)**: `InitializeComplexFromScaffold(scaffold)` — only if needed; otherwise SI.
- **§7 (optional)**: `EvaluateEdit(edit)` — show propose/test/settle loop explicitly.
- **§8 (recommended)**: interface-level pseudo-code for `QueryDensity`, `Sample`, `ConditionalSample`.

Guideline: if an algorithm box introduces a symbol or data structure not yet owned by that section, it is placed too early.

---

## Figure Plan (reduce perceived density)

Figures should be used to establish stable mental models early.

1. **Figure 1 (after §2)**: pipeline overview with artifacts:
   - Stage 1: scaffolds at multiple \(\tau\) → select \(\tau^*\)
   - Stage 2: lift → refine → audit/propose edits → accept/reject → reconstruct probability field
   - show propose/test/settle loop as a small inset.
2. **Figure 2 (in §3–§4)**: Stage 1 loop + scale controller around it (controller-as-wrapper diagram).
3. **Figure 3 (in §5–§6)**: graph → simplicial lift; torsion ladder regimes (high-level).
4. **Figure 4 (in §8)**: dual graph of faces, face pressures, reconstruction to probability field.

Rule: a figure should appear before or at the same time as the first formal definition it visually explains.

---

## Content Migration Checklist (from current draft)

Use this as a guide when moving text from the current `paper.tex` into the new structure.

### Move / split
- Current “Stage 1” text:
  - Keep single-scale scaffold loop in §3.
  - Move scale representation + \(\Phi(\tau)\) + recursion algorithm to §4.
- Current “Stage 2” text:
  - Keep simplicial lift + simplex equilibrium (+ node-vs-simplex argument) in §5.
  - Move torsion formalism + ladder + \(Q_S\) to §6.
  - Move free-energy proxy equation and region definition to §7 (and ensure earlier mentions reference it).
  - Move dual flow + queries/sampling + warp strategy to §8.

### Replace early details with placeholders
- Anywhere Stage 1 references “accept only if free energy improves” should be phrased as:
  - “accepted only if a local evidence test improves (formalized in §7)”
  until §7 appears.
- Any numeric thresholds (CV cutoffs, torsion regimes) should be:
  - described qualitatively in main text,
  - listed as defaults in SI (or a compact “defaults” table).

### Notation discipline
- Do not add symbols to §2 just because they exist.
- Ensure each section introduces a small “local notation block” only for symbols first defined there.

---

## Source Crosswalk (Planned Sections ↔ Current Draft ↔ Reference Docs)

This crosswalk makes the plan actionable by indicating (a) what text/math already exists in the **current draft**, (b) what material exists in the **reference docs**, and (c) what should be treated as **SI-only** or **outdated**.

### §2 Framework Overview (“Proteus in one page”)
- **Use from current draft**:
  - `paper.tex` “Introduction” overview paragraphs (architecture + principle): L63–L75.
  - `paper.tex` “At a Glance”: L95–L107.
- **Mine from references**:
  - `reference/content_plan.md` §4 (framework overview intent): keep as scope guardrails only.
  - `reference/stage_1.md` §1–§2 and `reference/stage_2.md` §1–§2: “design philosophy” language (speed vs fidelity), but keep high-level.
- **Avoid / defer**:
  - Do not introduce \(\tau\), \(\Phi(\tau)\), torsion, dual flow, or free-energy equations here.

### §3 Stage 1: Distributional Scaffold (single-scale)
- **Use from current draft**:
  - Stage 1 conceptual decomposition: `paper.tex` L109–L124.
  - Keep any *minimal* notation already in main: \(\mathbf m_i,\mathbf s_i,\sigma_i^2,\rho_i,\mathbf u_i\), \(k\).
- **Mine from references (selective)**:
  - `reference/stage_1.md` §3 “Data Structures and State”: use as a checklist of node/link fields.
  - `reference/stage_1.md` §4.2 “Unified Statistical Update”: use to motivate the update loop; keep derivations out of main.
  - `SI.tex` S2.2–S2.3: use as the canonical derivation source for EWMA \(\alpha\) and dual-rate motion; keep in SI.
  - `SI.tex` S3.1–S3.2: pruning gauntlets (link/node pruning) belong in SI; in main, summarize as “statistical gauntlets.”
- **Earmark SI-only (do not expand main)**:
  - Power shield / Wilson pruning mechanics (Stage 1 spec §5.1) → SI S3.1.
  - Node pruning gauntlet (§5.2) → SI S3.2.
  - Deferred-nudge derivations / \(\delta_{\min}\) formulas → SI S2.3.

### §4 Stage 1 Controller: Multiscale Discovery
- **Use from current draft**:
  - “Principled Scale Representation” + scale controller narrative: `paper.tex` L127–L165.
  - Algorithm box “Recursive Scale-Space Search”: `paper.tex` L143–L163.
- **Mine from references (selective)**:
  - `reference/redraft.md`:
    - scale-covariance / \(\gamma\)-normalization motivation,
    - grid spacing (FWHM → \(r\approx 0.71\)),
    - bracketing and GP refinement narrative.
  - Prefer `SI.tex` S2.1 for the canonical derivation of \(r\) and scale-space resolution.
  - `reference/stage_1.md` §6 is useful as a concise “controller loop” story (but do not reintroduce symbols early).
- **Earmark outdated / don’t import verbatim**:
  - Any “operational method” details in `reference/redraft.md` that contradict current `paper.tex`/SI naming (keep only what matches).

### §5 Stage 2: Simplicial Lifting and Core Objective
- **Use from current draft**:
  - Stage 2 motivation: `paper.tex` L167–L172.
  - “Simplex equilibrium” + node-vs-simplex argument: `paper.tex` L174–L177.
- **Mine from references (selective)**:
  - `reference/addendum.md`:
    - stronger, more stepwise argument for “simplex equilibrium drives node equilibrium,”
    - “informative mismatch” at junctions (treat as optional, possibly SI or a short paragraph if it supports interpretability claims).
  - `reference/stage_2.md` §2 and §3.3: warm-start + simplex-native updates; keep mechanics light in main; initialization details in SI (currently referenced as SI~S9).
- **Avoid / defer**:
  - Do not introduce torsion \(\Omega_S\) until §6.
  - Do not introduce dual-flow variables until §8.

### §6 Stage 2 Diagnostics: Torsion Audit
- **Use from current draft**:
  - Torsion definition and ladder (already in main): `paper.tex` L179–L193.
  - Shape metric \(Q_S\) definition: `paper.tex` L190–L190.
- **Mine from references (selective)**:
  - `reference/stage_2.md` §3.2–§3.3: additional intuition for torsion-as-curl and for ladder regimes.
  - Threshold defaults: keep in SI (`SI.tex` contains many defaults; if not, cite SI~S7/S3 as planned).
- **Earmark SI-only**:
  - Detailed splitting mechanics (“Method A/B”) and shape-metric computation details should live in SI unless the main text needs a single sentence describing the fallback.

### §7 Evidence-Based Structure Selection (free-energy proxy)
- **Use from current draft**:
  - Free-energy proxy equation + affected-region story: `paper.tex` L215–L220.
- **Mine from current SI (preferred)**:
  - `SI.tex` S3.4 provides the most complete and consistent definition of affected region and the settle protocol; treat it as canonical.
- **Mine from references (background)**:
  - `bnp_design_implications.md` suggests stronger evidence tests (WAIC/LOO/Bayes factors), but treat as future work; do not alter the “minimal core” without rewriting experiments.

### §8 Generative Modeling + Inference Interface (dual flow, queries, sampling, warps)
- **Use from current draft**:
  - Dual flow narrative + equations: `paper.tex` L202–L214.
  - Out-of-sample query/sampling interface paragraph: `paper.tex` L222–L227.
  - Warp strategy (conceptual): `paper.tex` L230–L231.
- **Mine from references (selective)**:
  - `reference/stage_2.md` §3.4: dual flow framing and solver-as-consensus story; keep solver specifics in SI.
  - `reference/stage_2.md` ladder thresholds for warp vs split: keep numeric cutoffs in SI S7.
- **Earmark outdated / out-of-scope**:
  - Any recommendations like “never global NSF” or specific architecture prescriptions should be treated as *implementation notes* (SI) unless the main paper commits to them experimentally.

### Evaluation / limitations / conclusion
- **Use from current draft**:
  - `paper.tex` “Model Quality Analysis” onward (L236+).
  - “Limitations and Complexity/Scaling” (L302+) and “Conclusion” (L319+).
- **Mine from references**:
  - `reference/content_plan.md` is consistent with evaluation scope; no need to import.

---

## Outdated / Out-of-Scope Material in Reference Docs (Explicit “Do Not Import” List)

These items appear in the older reference materials but should **not** be pulled into Paper 1’s core narrative unless you explicitly decide to expand scope and re-run ablations accordingly.

- **Multimodal missingness / masking in fusion settings** (appears in `reference/redraft.md` toward the end):
  - Keep as a separate future application note; it distracts from the foundational manifold-learning architecture.
- **Marginal Gaussianisation + linearity tests (GLCE, distance correlation) as a gate for Glow**:
  - This is already present in `SI.tex` S1.1 as *optional preprocessing*. It should remain SI-only unless the main paper includes experiments demonstrating the gate’s effect.
- **Very detailed pruning gauntlets in the main paper text**:
  - Keep these in SI (already present as SI S3.1–S3.2). The main paper should only summarize the idea and cite SI.
- **Hard-coded engineering budgets and parameter caps** (e.g., “≤10% simplex growth per level”, explicit flow layer/width prescriptions):
  - Keep in SI/implementation notes; the main paper can state that budgets exist but should not read like an implementation manual.
- **BNP-inspired enhancements as requirements** (from `bnp_design_implications.md`):
  - Treat as future work; adding them changes the core algorithmic claim and would require updated evaluation.

## SI Routing Guidelines (what stays out of the main narrative)

Main text retains:
- definitions essential to understand the architecture,
- one canonical equation per “owned object” section (when needed),
- interface-level algorithms for inference.

SI retains:
- derivations of PD/scale-space connections,
- proofs (e.g., simplex → node equilibrium correspondence),
- solver details and convergence diagnostics,
- default thresholds/tuning, unless essential for meaning,
- implementation details and pseudocode variants.

