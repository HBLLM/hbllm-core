---
title: "Symbol Grounding Architecture (SGE)"
description: "Authoritative specification of hbllm.hcir.grounding: handcrafted/snn/learned FeatureExtractors, ThreeTierEntityTracker, ConceptLearner, HCIRCommitter, and CorticalGroundingBridge."
---

# Symbol Grounding Architecture (SGE)

> **Document Status:** Authoritative Architectural Specification  
> **Subsystem:** `hbllm.hcir.grounding` & `hbllm.hcir.world.grounding`  
> **Invariants:** 1 (State Ownership), 3 (Zero LLM Dependency in Core), 4 (Constitutional Governance), 7 (Domain-Agnostic Core)

---

## 1. Executive Summary

The **Symbol Grounding Engine (SGE)** bridges raw multimodal sensory observations (2D lattices, continuous kinematics, acoustic streams, spatial frames) to formal symbolic representations within the **HCIR** substrate.

Rather than relying on ungrounded token statistics or external LLM APIs, HBLLM grounds symbols empirically via physical invariants, bipartite spatiotemporal tracking, concept clustering, and transactional state commitment.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                   Raw Multimodal Percept                     │
  │     (Grid Observation, Kinematic Joint, Audio Envelope)     │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. PERCEPTUAL FEATURE EXTRACTION (Protocol: FeatureExtractor) │
  │    - SpatialGridFeatureExtractor ("handcrafted")             │
  │    - SNNPerceptualFeatureExtractor ("snn", LIF spiking)      │
  │    - LearnedPerceptualFeatureExtractor ("learned")           │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 2. SPATIOTEMPORAL TRACKING & IDENTITY PERSISTENCE            │
  │    - ThreeTierEntityTracker (TrackletState & CandidateState) │
  │    - CandidateGenerator & IdentityOperations                 │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 3. CONCEPT INDUCTION & REGISTRATION                          │
  │    - ConceptLearner (WorkingConceptStore, split/merge/bind)  │
  │    - GroundedConceptRegistry & ConceptLifecycleManager       │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 4. CONSTITUTIONAL COMMITMENT & WORKSPACE ATTACHMENT          │
  │    - HCIRCommitter (Constructs TransactionOperations)        │
  │    - ConstitutionalVerifier (Validates invariant constraints)│
  │    - TransactionManager.commit() ──► TieredWorkspace         │
  └──────────────────────────────────────────────────────────────┘
```

---

## 2. Core Invariants

1. **Transactional Commitment (Invariants 1 & 4):**
   The grounding subsystem never directly mutates `CognitiveGraph` or `TieredWorkspace`. Every newly recognized entity, position update, or induced concept is proposed as a `TransactionOperation` and committed through `TransactionManager` under `ConstitutionalVerifier`.
2. **Zero LLM Dependency in Sensory-Motor Loop (Invariant 3):**
   Perceptual feature extraction, tracking, and concept induction run 100% locally and deterministically without cloud APIs. Permitted encoders are zero-parameter topological extractors or deterministic offline-trained feed-forward / spiking neural encoders conforming to `FeatureExtractor`.
3. **Domain-Agnostic Symbol Formation (Invariant 7):**
   Grounding faculties operate on universal topological, geometric, and dynamic primitives (connectivity, bounding contours, displacement vectors, velocity, morphology) rather than task-specific names.

---

## 3. Subsystem Components

### 3.1 Perceptual Feature Extraction (`hbllm.hcir.grounding.features`)

Extractors conform to the `FeatureExtractor` runtime protocol (`feature_extractor.py`):
- **`SpatialGridFeatureExtractor`:** Extracts topological connected components, contours, and normalized spatial properties using zero-parameter algorithms (`perception_mode="handcrafted"`).
- **`SNNPerceptualFeatureExtractor`:** Simulates leaky integrate-and-fire (LIF) spiking populations (`perception_mode="snn"`) for morphology-invariant spatial contrast and temporal dynamics.
- **`LearnedPerceptualFeatureExtractor`:** Deterministic offline representation encoder (`perception_mode="learned"`).
- **`KinematicFeatureExtractor` & `SemanticTextFeatureExtractor`:** Continuous control and lexical feature extractors coordinated via `FeatureExtractorRegistry`.

### 3.2 Spatiotemporal Tracking (`hbllm.hcir.grounding.tracking`)

- **`ThreeTierEntityTracker` (`tracker.py`):** Maintains object permanence and identity across frames via `TrackletState` and `CandidateEntityState`, associating entities via `auto_associate()` and spatial distance gating.
- **`CandidateGenerator` (`candidate_generator.py`) & `IdentityOperation` (`operations.py`):** Emit formal association, merge, split, re-identification, and alias operations.

### 3.3 Concept & Adaptive Relation Induction (`hbllm.hcir.grounding.concepts`, `relations`)

- **`ConceptLearner` (`concept_learner.py`):** Clusters perceptual feature vectors inside `WorkingConceptStore` (`store.py`), performing dynamic prototype assignment, splitting, merging, and role binding (`bind_role_to_exemplar`, `bind_linguistic_alias`). Under ADR 006 (`enable_fuzzy_grounding=True`), `learn_fuzzy()` computes overlapping graded `MultiConceptMembership` scores ($\mu_C(x) \in [0, 1]$) from online Welford `(mean, variance)` distributions via dimensionless `mahalanobis_rbf` kernels, and `infer_role_priors_fuzzy()` consumes both working-store prototypes and persisted `GroundedConceptNode`s in `CognitiveGraph`.
- **`RelationLearner` (`relation_learner.py`):** Induces crisp relations alongside continuous, context- and scale-normalized memberships (`evaluate_fuzzy_relations`: `IN`, `ABOVE`, `TOUCHING`, `NEAR`, `APPROACHING`) parameterized by immutable versioned `AdaptiveMembershipModel`s (`AdaptiveMembershipNode`) over `entity_extent`, `navigable_diameter`, and `closing_ttc`.
- **`GroundedConceptRegistry` (`registry.py`) & `ConceptLifecycleManager` (`lifecycle.py`):** Manage `GroundedConcept` prototypes (`model.py`) and promotion proposals (`GroundedConceptNode` + `AdaptiveMembershipNode`) across workspace tiers.
- **`RelationalGraphMatcher` (`relational_graph_matcher.py`):** Computes structural correspondences across `ObjectGraph`s.

### 3.4 Dual-Channel Transactional Commitment (`hbllm.hcir.grounding.committer`)

- **`HCIRCommitter` (`committer.py`):** Translates crisp grounding decisions (`commit_decision()`) and graded assertions (`commit_fuzzy_assertions()`) into atomic node-before-edge `TransactionOperation` batches via `TransactionManager`. `FuzzyAssertion` payloads (`degree`, `model_id`, `model_version`, `profile_id`) and Bayesian `EpistemicBelief` posteriors (`posterior_belief`, `BipolarEvidenceState`) remain strictly separate.

---

## 4. Cortical Epistemic Integration (`CorticalGroundingBridge`)

`CorticalGroundingBridge` (`hbllm.hcir.world.grounding.bridge`) binds the SGE pipeline to `AutonomousEpistemicEngine` and `KernelServices` inside `build_cognitive_system()`:

```python
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

system = build_cognitive_system(
    CognitiveSystemConfig(
        world_id="sge_demo",
        enable_grounding=True,
        enable_fuzzy_grounding=True,  # Overlapping concepts & scale-normalized relations
        enable_fuzzy_reasoning=True,  # Soft VM unification & RelationCompositionPolicy
        perception_mode="snn",  # "handcrafted" | "snn" | "learned"
        memory_dir=None,
    )
)
# system.bridge is attached to system.engine sharing the same KernelServices
```

During each step, `CorticalGroundingBridge` processes perception (`process_perception` with unmoved-entity caching and salience-gated batched commits), synchronizes learned roles (`sync_theory_roles`), integrates assimilation and motor outcomes (`process_assimilation`, `process_motor`, `verify_post_action_outcome`), and handles boundary retention (`on_boundary`).
