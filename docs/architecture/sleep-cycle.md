# Lifelong Continual Learning & Sleep Consolidation Substrate

The **Lifelong Continual Learning Substrate** enables HBLLM to accumulate experience, concepts, vocabulary, and relational schemas across long operating lifetimes without catastrophic forgetting, representational drift, or unchecked graph entropy.

Unlike stateless language models that forget every interaction after a session ends, HBLLM mirrors human cognitive consolidation by executing multi-stage offline sleep cycles orchestrated by the `SleepCycleNode`.

```text
                                REAL-WORLD INTERACTION
                                          │
                                          ▼
                               Fast Episodic Buffer (Fast)
                                          │
                                          ▼
                         Metacognitive Salience Filter
                                          │
                                          ▼
                             SLEEP CONSOLIDATION ENGINE
                                          │
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
             Error Replay           Success Replay        Contrastive Replay
         (Discover boundaries)   (Consolidate schemas)   (Isolate critical deltas)
                  │                       │                       │
                  └───────────────────────┼───────────────────────┘
                                          ▼
                             Adaptive Semantic Folding
                       (Provenance-Preserving Compaction)
                                          │
                                          ▼
                               Candidate Knowledge Update
                                          │
                                    STABILITY GATE
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
             Update Accepted                             Update Rejected
                    │                                           │
                    ▼                                           ▼
         Slow Consolidated Store                     Preserve Mature State
    (Concepts, Schemas, Lexicon, Profiles)
                    │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                        IMMUTABLE EVIDENCE & PROVENANCE
```

---

## 1. The Three-Layer Memory Model

The architecture enforces a strict three-tier memory separation:

| Memory Layer | Storage Characteristics | Cognitive Role |
| :--- | :--- | :--- |
| **`FAST_EPISODIC`** | In-memory transient buffer, cleared post-consolidation | Captures high-resolution raw interaction traces, sensory observations, and immediate execution steps. |
| **`SLOW_CONSOLIDATED`** | Durable, compacted knowledge store with semantic indexing | Houses generalized conceptual prototypes, relational schemas, grounded vocabulary, and competence profiles. |
| **`IMMUTABLE_PROVENANCE`** | Append-only, cryptographically hashed event log | The ultimate authoritative source of truth; justifies consolidated knowledge and enables exact historical state reconstruction. |

### The Epistemic Invariant: Compaction $\ne$ Forgetting
Slow consolidated knowledge is an efficient cognitive summary; the immutable evidence log remains the epistemic justification:
> *"A compact abstraction does not have to contain every original detail to remain explainable, because it retains explicit provenance pointers (`source_event_ids: [E1, E2, ...]`) back to immutable evidence."*

---

## 2. Tri-Modal Offline Sleep Replay (`SleepReplayEngine`)

During offline sleep cycles, the replay engine executes non-destructive mental simulations across three complementary modalities:

1. **Error Replay**:
   - Replays episodes with high prediction error ($PE \ge 0.50$) to diagnose failure boundaries and identify missing physical constraints.
2. **Success Replay**:
   - Replays novel successful trajectories to synthesize generalized, reusable action schemas.
3. **Contrastive Replay**:
   - Pairs near-identical action sequences that produced divergent physical outcomes (e.g. stacking on flat support vs stacking on curved support) to isolate distinguishing structural invariants:
     $$\text{Isolated Delta} = \Delta(\text{Success Context}, \text{Failure Context})$$

---

## 3. Provenance-Preserving Adaptive Compaction (`ProvenancePreservingCompactor`)

Raw episodic graphs are folded into compact schemas under a strict four-part verification contract:

1. **Behavioral Invariants Preserved**: Consolidated schemas faithfully reproduce successful episodic action plans.
2. **Predictive Invariants Preserved**: Precondition constraints capture all necessary physical support conditions.
3. **Causal Invariants Preserved**: Pre-state to post-state causal links remain intact.
4. **Provenance Preserved**: Every source event ID is reachable in the authoritative immutable log.

Compression ratios adapt dynamically to preserve all necessary invariants rather than enforcing a destructive fixed quota.

---

## 4. Dependency-Aware Stability Gate (`PlasticityStabilityEngine`)

Before any candidate schema or concept update is committed to the slow consolidated store, the Stability Gate performs targeted dependency analysis:

```text
                  Candidate Knowledge Update
                              │
                              ▼
                     Dependency Analysis
                              │
                     Targeted Regression
                              │
               ┌──────────────┼──────────────┐
               ▼              ▼              ▼
           ACCEPTED     SPECIALIZATION    REVISION
         (Zero loss)  (Boundary narrow) (Error fix)
               │              │              │
               └──────────────┼──────────────┘
                              ▼
                     Commit to Slow Store
```

### Gate Decisions
- **`ACCEPTED`**: Clean improvement preserving historical knowledge integrity.
- **`EXPECTED_SPECIALIZATION`**: Legitimate boundary narrowing to exclude incompatible physical regimes.
- **`BENEFICIAL_REVISION`**: Correction of previously flawed knowledge triggered by contradictory evidence.
- **`REJECTED_REGRESSION`**: Collateral accuracy degradation on unrelated mature domains $\to$ Update rejected.

---

## 5. Multi-Stage Operational Sleep Cycle Sequence

The `SleepCycleNode` executes an automated 5-stage consolidation sequence when the system enters idle mode:

### Stage 1: Memory Replay & Knowledge Consolidation
- **Selective Replay & Compaction:** Folds active `EpisodeNode` entries into durable `ConceptNode` and `BeliefNode` representations.
- **Temporal Normalization:** Scans relative temporal references ("yesterday", "last month") and annotates them with absolute ISO timestamps based on event creation time.
- **Contradiction Resolution:** Scans exclusive relation types (`prefers`, `is_a`, `has`) and resolves conflicting targets by preserving the latest verified evidence.
- **Knowledge Staleness Audit:** Flags expired web or external entries past their TTL for re-verification.
- **Task Knowledge Promotion:** Promotes frequently accessed task-level research into permanent long-term memory.

### Stage 2: Artificial Neuroplasticity & Skill Optimization
- **Continuous Preference Learning:** Evaluates critic feedback and refines local adapter weights without modifying the base read-only model.
- **Skill Optimization:** Re-evaluates custom procedural routines and repairs low-success actions in the Skill Registry.

### Stage 3: Curiosity-Driven Exploration
- Replays unanswered curiosity goals and explores knowledge gaps across the Knowledge Graph.

### Stage 4: Dream Journaling & Observability
- Emits a structured `system.sleep.report` summarizing:
  - Memories compacted and temporal references normalized
  - Contradictions resolved and schemas specialized
  - Skills refined and curiosity gaps investigated
- Accessible programmatically or via UI dashboards.

### Stage 5: Proactive Memory Warming
- Pre-warms the fast-path memory cache with recent topic summaries to ensure zero-latency response upon wakeup.

---

## 6. Execution & Triggers

Consolidation cycles can be triggered via three mechanisms:

1. **Auto-Trigger (Idle Timeout):** Automatically activates when no user query is detected for a configurable duration (default: 6 hours). Wakes up immediately if a live query arrives.
2. **Bus Trigger (`/dream`):** Any node or CLI can publish to `system.sleep.force` for an immediate cycle:
   ```python
   await bus.publish(
       "system.sleep.force",
       Message(type=MessageType.QUERY, source_node_id="cli", topic="system.sleep.force", payload={}),
   )
   ```
3. **REST API Trigger:**
   ```bash
   curl -X POST "https://api.hbllm.ai/v1/system/sleep" \
        -H "Authorization: Bearer <TOKEN>" \
        -d '{"tenant_id": "tenant-001", "mode": "deep"}'
   ```

---

## 7. Versioned Knowledge Revisions

Mature knowledge is never overwritten destructively. Revisions are tracked monotonically with full audit logs:

```python
@dataclass
class VersionedKnowledgeRecord:
    knowledge_id: str
    knowledge_type: str
    revision: int = 1
    supersedes_revision: int | None = None
    revision_reason: str = "initial_induction"
    content: dict[str, Any] = field(default_factory=dict)
    source_event_ids: list[str] = field(default_factory=list)
    confidence: float = 0.75
    created_at: float = field(default_factory=time.time)
```

---

## 8. Lifelong Multi-Task Retention Evaluation

Continual learning evaluates performance across sequential task curricula ($T_1 \to \dots \to T_N$) using the full task-performance matrix $R_{i,j}$:

$$R_{i,j} = \text{Performance on task } j \text{ after learning task } i$$

- **Backward Transfer ($\text{BWT}$)**: Quantifies retention of earlier tasks:
  $$\text{BWT} = \frac{1}{T - 1} \sum_{j=1}^{T-1} (R_{T, j} - R_{j, j}) \ge 0.0$$
- **Forward Transfer ($\text{FWT}$)**: Quantifies accelerated learning on novel tasks:
  $$\text{FWT} = \frac{1}{T - 1} \sum_{j=2}^{T} (R_{j-1, j} - b_j) > 0.0$$
