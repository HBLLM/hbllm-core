# Symbol Grounding Architecture (SGE)

> **Document Status:** Authoritative Architectural Specification  
> **Subsystem:** `hbllm.hcir.grounding`  
> **Invariants:** 1 (State Ownership), 3 (Zero LLM Dependency in Cognitive Core), 4 (Constitutional Governance), 7 (Domain-Agnostic Core)

---

## 1. Executive Summary

The **Symbol Grounding Engine (SGE)** bridges raw multimodal sensory observations (2D lattices, continuous kinematics, acoustic streams, spatial point clouds) to formal symbolic representations within the **HCIR** (Human-Cognitive Intermediate Representation) substrate.

Rather than relying on ungrounded token statistics or hallucination-prone external models, HBLLM grounds symbols empirically via physical invariants, active inference, spatiotemporal tracking, and transactional state commitment.

```
  ┌──────────────────────────────────────────────────────────────┐
  │                   Raw Multimodal Percept                     │
  │     (Grid Observation, Kinematic Joint, Audio Envelope)     │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. PERCEPTUAL FEATURE EXTRACTION (Protocol: FeatureExtractor) │
  │    - Handcrafted Feature Extractor (Zero-param topology)     │
  │    - SNN Perceptual Feature Extractor (Spiking LIF dynamics) │
  │    - Learned Feature Extractor (Deterministic offline encoder)│
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 2. SPATIOTEMPORAL TRACKING & IDENTITY PERSISTENCE            │
  │    - SpatiotemporalTracker (Bipartite matching, velocity)   │
  │    - TemporalIdentityManager (Centroid distance, IoU gating) │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 3. CONCEPT INDUCTION & REGISTRATION                          │
  │    - ConceptLearner (Affordance clustering, DBSCAN/K-Means)  │
  │    - GroundedConceptNode (Symbolic token ↔ sensory anchor)   │
  └──────────────────────────────┬───────────────────────────────┘
                                 │
                                 ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ 4. CONSTITUTIONAL COMMITMENT & WORKSPACE ATTACHMENT          │
  │    - GroundingCommitter (Constructs TransactionOperations)   │
  │    - ConstitutionalVerifier (Validates invariant constraints)│
  │    - TransactionManager.commit() ──► TieredWorkspace         │
  └──────────────────────────────────────────────────────────────┘
```

---

## 2. Core Invariants

1. **Transactional Commitment (Invariant 1 & 4):**
   The grounding subsystem never directly mutates `CognitiveGraph` or `TieredWorkspace`. Every newly recognized entity, object position update, or induced concept is proposed as a `TransactionOperation` (`ADD_NODE`, `UPSERT_NODE`, `ADD_EDGE`, `REMOVE_NODE`) and committed through `TransactionManager` under `ConstitutionalVerifier`.

2. **Zero LLM Dependency in Sensory-Motor Loop (Invariant 3):**
   Perceptual feature extraction, coordinate lifting, tracking, and concept clustering execute 100% locally and deterministically with sub-millisecond to low-millisecond latencies ($p95 < 20\text{ms}$). External cloud API calls are strictly forbidden. Permitted encoders are deterministic, offline-trained feed-forward or spiking neural networks conforming to `FeatureExtractor`.

3. **Domain-Agnostic Symbol Formation (Invariant 7):**
   Grounding faculties operate on universal topological, geometric, and dynamic primitives (connectivity, bounding boxes, displacement vectors, velocity, collision manifolds) rather than task-specific or game-specific token names.

---

## 3. Subsystem Components

### 3.1 Perceptual Feature Extraction (`hbllm.hcir.grounding.features`)

Extractors conform to the `FeatureExtractor` runtime protocol:

- **`HandcraftedFeatureExtractor`:** Evaluates topological connected components, color/property partitions, and bounding contours using zero-parameter algorithms.
- **`SNNPerceptualFeatureExtractor`:** Simulates leaky integrate-and-fire (LIF) spiking neuronal populations for spatial contrast, edge detection, and temporal event dynamics with sub-millisecond CPU execution.
- **`LearnedPerceptualFeatureExtractor`:** Offline-trained deterministic representation network producing fixed-dimensional dense embeddings for invariant cluster discovery.

### 3.2 Spatiotemporal Tracking (`hbllm.hcir.grounding.tracking`)

- **`SpatiotemporalTracker`:** Maintains object permanence and identity across frames despite partial occlusion, rapid displacement, or sensory noise.
- **`TemporalIdentityManager`:** Employs optimal transport (bipartite Hungarian matching over centroid distance, size delta, and appearance features) to assign stable entity IDs across consecutive epistemic cycles.

### 3.3 Concept Induction (`hbllm.hcir.grounding.concepts`)

- **`ConceptLearner`:** Clusters grounded perceptual observations into discrete semantic categories (`GroundedConceptNode`) based on shared visual, dynamic, and affordance features.
- **`ConceptRegistry`:** Persistent registry storing learned prototypes and spatial relations.

### 3.4 Transactional Commitment (`hbllm.hcir.grounding.committer`)

- **`GroundingCommitter`:** Synthesizes `TransactionOperation` lists for atomic application to `TieredWorkspace`. It guarantees:
  - Idempotent upserts for existing entities.
  - Safe retraction of disappeared or annihilated entities.
  - Edge assertions linking entities to spatial regions and parent frames.

---

## 4. Cortical Epistemic Integration (`CorticalGroundingBridge`)

The `CorticalGroundingBridge` binds the sensory-motor grounding pipeline with the 5-stage cortical epistemic cycle (`AutonomousEpistemicEngine`):

```python
bridge = CorticalGroundingBridge(
    services=kernel_services,
    feature_extractor=extractor,
    tracker=tracker,
)
await bridge.process_observation(raw_observation, step=step_counter)
```

The bridge extracts entities, updates tracking histories, proposes state mutations via `kernel_services.transaction_manager`, and injects stabilized spatial entities into the perception stage of the cortical loop.

