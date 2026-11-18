# HBLLM Core Knowledge & Innate Priors Declaration

**Document Version:** 1.0.0  
**Authority:** AGENTS.md Invariant 7 (Task-Agnostic Core Directive) & Gap Analysis Property 16  
**Status:** Canonical & Binding  

---

## 1. Purpose & Epistemic Boundary

Intelligence cannot operate in a vacuum. Under Wolpert's No-Free-Lunch theorem and Chollet's *On the Measure of Intelligence*, general intelligence is defined as **skill-acquisition efficiency over a distribution of tasks relative to a set of prior knowledge**.

To prevent the creeping accretion of domain-specific heuristics and game-specific monkey-patches into `hbllm/`, this document formally declares the **exact, exhaustive list of innate priors** that the HBLLM cognitive core is permitted to assume.

> [!IMPORTANT]
> **The Prior Invariant:**  
> Any structure, expectation, predicate, affordance, or relation that is **not** explicitly listed in this catalog is considered *empirical world knowledge* and **must be learned in-environment** through observation, active inference, Pearl's causal calculus ($do(a)$), or linguistic interaction. Hardcoding rules outside this catalog in `hbllm/` violates Invariant 7.

---

## 2. Declared Core Innate Priors

The innate core priors are grounded in developmental cognitive psychology (Spelke & Kinzler 2007 "Core Knowledge") and spatial topology.

### 2.1 Spatiotemporal Objectness (Spelke Prior 1)
1. **Cohesion & Boundary:** The sensory field contains bounded, connected contiguous clusters of points/cells ("objects" / `SpatialEntity`). Objects move as cohesive wholes.
2. **Spatiotemporal Continuity:** Objects move along continuous, spatiotemporally connected paths through coordinate space. An object cannot teleport across disconnected coordinates without cause.
3. **Solidarity & Occlusion:** Two objects cannot occupy the exact same coordinate volume simultaneously. When an object vanishes behind another object, it is assumed occluded, not destroyed (`object_permanence`).
4. **Coordinate Invariance:** Spatial entity identity and concept clustering are invariant under translation in coordinate space. Concepts cannot depend on absolute $(x, y)$ positions.

### 2.2 Elementary Spatial Physics & Topology (Spelke Prior 2)
1. **Manifold & Metric:** Space is metricized with a defined distance function (Euclidean $\ell_2$ or Manhattan $\ell_1$).
2. **Topological Relations:** Elementary topological invariants are innate:
   - `ADJACENT` (sharing an edge/boundary)
   - `CONTAINED_IN` / `INSIDE` (boundary closure)
   - `SEPARATED` (non-zero distance, cut-set separation)
3. **Inertia & Kinematic Projection:** Moving bodies tend to maintain velocity unless acted upon by a collision, barrier, or self-actuation (`dorsal_kinetic_stream`).

### 2.3 Agency & Intention (Spelke Prior 3)
1. **Self-Agency Attribution:** Motor commands emitted by the ego agent correlate with immediate, proximal state changes (`metacognitive_agency_cortex`).
2. **Goal-Directed Agency Detection:** Entities whose spatiotemporal trajectories deviate systematically from passive inertial physics toward specific state attractors are modeled as intentional agents with latent subgoals (`Bayesian Theory of Mind`).
3. **Collision Avoidance:** Moving agents tend to avoid lethal hazard cones and irreversible terminal traps (`hazard_tracker`, `collision_cones`).

### 2.4 Discrete Numerosity & Counting (Spelke Prior 4)
1. **Cardinality & Comparison:** The core can count discrete entities, compute sets, and evaluate equality, subset, and set difference operations ($=, \neq, <, >$).
2. **One-to-One Correspondence:** Entity tracking matches candidates across observation frames via bijective or cost-minimal assignment (e.g. Hungarian matching).

### 2.5 Causal Structure & Active Inference
1. **Temporal Precedence:** Causes strictly precede effects ($\Delta t > 0$).
2. **Intervention Distinction (Pearl's $do(a)$):** Passive correlation $P(y \mid x)$ is distinct from causal intervention $P(y \mid do(x))$. Motor actions are recognized as active interventions.
3. **Expected Free Energy Minimization:** Cognitive decision-making balances extrinsic utility (satisfying subgoals/drives) with epistemic value (reducing predictive uncertainty/surprise).

---

## 3. Prohibited Domain-Specific Knowledge (Must Be Learned)

The following must **never** be hardcoded in the cognitive core:
- Specific grid color meanings (e.g., "color 0 is background, color 2 is goal, color 8 is key").
- Game-specific mechanics (e.g., keys open doors, pushing blocks onto switches toggles gates, stepping on lava causes death).
- Fixed level dimensions or puzzle-specific shapes.
- Bespoke action indices (action 1 = UP, action 2 = DOWN must be discovered via motor assimilation).
- Specific linguistic aliases (e.g., "door", "avatar" must be bound dynamically via linguistic grounding).

---

## 4. Architectural Verification Gate

Every pull request or modification to `hbllm/hcir/` and `hbllm/brain/` is audited against this document. Introducing heuristic assumptions about particular environments constitutes an architectural failure.

