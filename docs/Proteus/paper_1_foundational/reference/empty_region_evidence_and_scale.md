# Empty-Region Edge Evidence and the Two Notions of Scale

**Status:** theory note / directive (human + orchestrator, 2026-08-09). Supporting
draft feeding SI S2.5/S2.6; not canonical until promoted. Written to redirect the
OPEN_ISSUES **#44** work after four geometry-specific prepasses (radial gap, radial
band, non-centroid, signal-density, PCA-axis) failed to generalize (linked_tori
unrecovered; swiss shatters at `steps=8`).

## 1. Diagnosis: a category error hiding in the word "scale"

Two different quantities are both being called "scale" in the current pipeline:

- **Resolution scale** — the tau at which a *given* feature's support is resolved by
  the scaffold. This is what the `L = 1` variance-load up-crossing finds (S2.5.1),
  and it finds it well. Nothing here needs fixing.
- **Separation scale** — the tau at which two parts become distinguishable *to the
  scaffold graph*. The #44 descent machinery treats this as a "finer-scale feature"
  to be found by walking tau down.

The second notion is a mirage. **Disconnection of supports is a scale-free,
topological property of the data.** tau caps the *code's* variance; it never smooths
the data. A two-component support is two-component at every tau. It only *looks*
connected to the pipeline because (a) the lifted Hebbian graph radius-bridges gaps
when node spacing is comparable to the gap (A2's own finding), and (b) `K_v`
conditions edge evidence on *length*, which cannot distinguish "long edge inside
sparse support" from "short edge across a void."

Decisive evidence already in hand (probe data, OPEN_ISSUES #44): the nested-spheres
scaffold at `tau = 0.27` and at `tau = 0.004` has the **same 64 nodes**; only the
link/kernel structure differs. Descent did not buy resolution — the disconnection
information was present at the coarse scale and the statistic could not see it.

Contrast the hierarchical-Gaussian case: coarse blobs genuinely merge in the
*density*. That is real scale-dependent structure, and persistence + DM already
handle it correctly (6 leaves, heuristic-free). The #28 "category mismatch" finding
(persistence tau* ~9x the fine-leaf `expected_tau`) is the same two-scales confusion
surfacing from the other side.

**Consequence:** stop trying to recover topological separation by tau descent plus
per-geometry coordinate prepasses. Radial/PCA/tube/linking-number cues all impose a
global coordinate; the evidence is local to each bridging edge. This family will
keep failing on the next scene shape.

## 2. The deeper principle: a good adaptive code hides geometry from itself

At equilibrium the scaffold equalizes hit mass and pins variances to the cap, so
**every scaffold-side statistic self-normalizes away the geometry it was meant to
measure**. This single principle explains three independent refutations already in
the log: the flat Lindeberg response at equilibrium, the failed compensated
node-count knee (#28), and the under-determined single-scale Q (#27). `L = 1` works
precisely because it is a *disequilibrium* signal (where the cap starts to bind).

Structural tests must therefore be **data-side**, computed from sample positions —
the one thing adaptation cannot normalize away. The routing counts in `A_sym` do not
record *where* samples were relative to an edge; that is exactly the missing channel.

## 3. The missing primitive: hollow-edge (empty-region) evidence

For a lifted edge `(i, j)`, ask: **does data actually occupy the region the edge
spans?**

### 3.1 Batch form (implement first — cheapest, decisive)

For edge `(i, j)` with endpoints `x_i, x_j`, length `L`:

- `n_mid` = number of data samples within radius `r = L/4` of the midpoint
  `(x_i + x_j)/2` (lens/Gabriel-ball approximation);
- `n_end` = mean of the same counts around `x_i` and `x_j`;
- hollowness ratio `H(i,j) = n_mid / (n_end + eps)`.

A within-support edge has `H = O(1)`; a bridge over a void has `H ≈ tissue-rate /
signal-rate ≈ 0`. Cut edges with `H < h_0` **before** connected-components /
AP clustering. Using the *ratio* to endpoint density makes the test robust to the
uniform tissue background (tissue contributes to numerator and denominator alike)
and free of absolute-density constants.

The proper null is Poisson: under a locally uniform manifold of density `rho`, the
mid-ball count is `Poisson(rho * V_ball)` with `rho` estimated from the endpoint
counts — so "hollow" is a calibrated likelihood-ratio (or e-value) decision, in line
with the S14.3 constant discipline (acceptance-path quantities need a declared
protocol, not a tuned knob like `keep_frac = 0.55`).

Classical anchors: **Gabriel graph** (Gabriel & Sokal 1969) prunes `(i,j)` if the
ball with diameter `[x_i, x_j]` contains any other point; **relative neighborhood
graph** (Toussaint 1980) is the looser lens variant; Isomap-era **shortcut-edge
removal** is the same operation on neighborhood graphs. We are not inventing a new
test — we are importing a 50-year-old one and calibrating its null.

### 3.2 Online form (Stage-1-native, for the runtime loop later)

When a sample co-activates edge `(i, j)`, record its projection parameter
`t ∈ [0, 1]` onto the segment. Maintain a tiny EWMA histogram (3 bins suffices:
ends / middle). Connected support ⇒ mid-bin mass `O(1/3)`; bridge edge ⇒ bimodal
end-concentration, mid-bin ~ 0. This makes hollowness a first-class edge statistic
alongside `A_sym`, updated at `O(1)` per sample, and is the natural SI extension of
the "scale-conditioned edge evidence" paragraph (S2.6.1): `W_v = K_v * A_sym *
M_transit`, with `M_transit` the measured mid-support factor replacing a
geometric prior (length kernel) by evidence.

### 3.3 Predicted verdicts on the current failure suite

| Scene | Hollow-edge verdict | Ground truth |
|---|---|---|
| nested_spheres | shell-to-shell edges cross annular void -> cut -> cc = 2 | 2 (correct) |
| linked_tori | every A-B edge crosses empty space (tori never touch); **no global coordinate needed** | 2 (correct) |
| manifold_zoo | junctions carry data -> nothing cut -> cc = 1 | 1 (correct) |
| swiss_roll | inter-wrap shortcut edges hollow -> cut; sheet stays connected -> cc = 1, graph quality improves | 1 (correct) |
| circle / hierarchy | no voids -> no-op; persistence handles density structure as now | correct |

One coordinate-free statistic covers every case the five bespoke prepasses were
built for — including linked_tori, which defeated all of them — with **no tau
descent**, so the swiss-shattering failure mode never arises.

### 3.4 Division of labor after this lands

- **Support topology (disconnection):** hollow-edge test, at the region's own
  tau*. Scale-free. Runs as a prepass; `allow_finer_research` descent becomes a
  fallback, not the mechanism.
- **Density structure (blob hierarchies):** persistence (S2.6.2) + DM margin
  (S2.6.3), unchanged — already validated.
- **Resolution:** `L = 1` up-crossing, unchanged.

## 4. The right formal object: the cluster tree

What Stage 1's structural layer is estimating is **Hartigan's density cluster
tree**. Use the literature instead of re-deriving it piecemeal:

- **Chaudhuri & Dasgupta (2010), robust single linkage:** provably consistent
  cluster-tree estimator; its two parameters play exactly the role of our
  connectivity radius + density floor. Their consistency proof is the theory
  backstop for "cut hollow edges, then take connected components."
- **ToMATo (Chazal–Guibas–Oudot–Skraba 2013):** density-mode merging gated by
  **persistence** with correctness guarantees — this is the S2.6.2 persistence gate,
  formalized. Citing/aligning with it strengthens the SI and may sharpen the
  `P_persist`/`theta_ovl` choices.
- **Stuetzle (2003) runt pruning:** principled version of what the S2.6.1 cleanup
  stand-ins (tiny-cluster absorption) do by hand — relevant to the eventual #27
  stand-in deletion.
- Wishart (1969) / DBSCAN density-reachability: the same empty-region idea in
  clustering form; useful for the null-model write-up.

Framing the tau sweep as cluster-tree estimation turns disconnected supports into
ordinary high-level tree splits and gives access to known consistency results.

## 5. Longer-horizon: joint (position, scale) adaptation

Classical scale-space detects features as maxima in *joint* (x, t) space; the
current architecture collapses to one global tau per region and re-enters via
recursion — #44 is the price of that collapse. The machinery for the joint version
already exists in-repo: `tau_local,i` is per-node (currently forced uniform, S2.4),
the load `L_i = sigma_i^2 / tau` is a per-node field, splits are local edits, and
the S3.4 DM gate is specified as a *node/star edit* arbiter. Letting locally binding
load drive local refinement, gated by the evidence gate once it is wired into the
runtime loop (M4 board item), is the principled replacement for region-level tau
descent. This is the cheap isotropic cousin of the SI's deferred anisotropic
scale-space, and it is not blocked on anything external.

## 6. Experimental protocol (do these in order)

1. **Frozen-scaffold probe (decisive, ~hours):** rebuild the nested-spheres scaffold
   at coarse tau (~0.27, 64 nodes, where clustering sees K=1). Compute `H(i,j)` for
   every lifted edge against the raw dataset. Expected: cross-shell edges hollow,
   intra-shell edges not; pruning + connected components -> cc = 2 **at the coarse
   scale, zero descent**. Repeat for linked_tori (tau~0.5) and manifold_zoo
   (expect: junction edges NOT hollow, cc stays 1). If this fails, the theory note
   is wrong — say so loudly in COORDINATION and stop here.
2. **Swiss-roll guard:** verify hollow pruning removes inter-wrap shortcuts without
   disconnecting the sheet (cc stays 1; optionally check graph geodesics improve).
3. **Calibrate `h_0`** as a likelihood-ratio / e-value threshold from the Poisson
   lens null with endpoint-estimated `rho`, on adversarial nulls: (a) connected
   sheets with strong density gradients and curvature (must NOT cut), (b) two
   components with tissue in the gap at increasing rates (must cut until tissue ~
   signal). Report a ROC, not a pass/fail on four scenes. No fixture-seed tuning.
4. **Wire as flag-gated prepass** (`prefer_hollow_edge_prepass`, default off) in
   the clustering path; rerun the fuller suite under `persist (+dm)` with the
   prepass on and **no** finer-scale descent. Score with mechanism attribution:
   every accepted split logs *which* signal fired (hollow / density mode /
   persistence).
5. Only then revisit: retiring the radial/PCA prepass family, the descent flags'
   default pairing, and (with A1) the awaiting-flip decisions.

## 7. Cautions

- Keep everything flag-gated and default-off until step 4 passes; never weaken
  tests; awaiting flips still require A1 sign-off with green evidence (unchanged).
- The mid-ball radius `L/4` and 3-bin histogram are proposal-path defaults; the
  *cut decision* threshold is acceptance-path and must carry the Poisson-null
  calibration (S14.3 tiering).
- Sample counts: mid-ball tests need enough data per edge neighborhood; at very low
  `n` fall back to the Gabriel criterion (any point in the diameter ball) which
  needs no density estimate.
- Tissue robustness comes from the endpoint-ratio form; do not switch to absolute
  counts.
