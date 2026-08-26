# Lifelong Continual Learning & Sleep Consolidation Substrate

The **Lifelong Continual Learning Substrate** enables HBLLM to accumulate experience, concepts, vocabulary, and relational schemas across long operating lifetimes without catastrophic forgetting, representational drift, or unchecked graph entropy.

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

## 5. Versioned Knowledge Revisions

Mature knowledge is never overwritten in place. Revisions are tracked monotonically with full rollback capabilities:

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

## 6. Lifelong Multi-Task Retention Evaluation

Lifelong continual learning evaluates performance across sequential task curricula ($T_1 \to \dots \to T_N$) using the full task-performance matrix $R_{i,j}$:

$$R_{i,j} = \text{Performance on task } j \text{ after learning task } i$$

- **Backward Transfer ($\text{BWT}$)**: Quantifies retention of earlier tasks:
  $$\text{BWT} = \frac{1}{T - 1} \sum_{j=1}^{T-1} (R_{T, j} - R_{j, j}) \ge 0.0$$
- **Forward Transfer ($\text{FWT}$)**: Quantifies accelerated learning on novel tasks:
  $$\text{FWT} = \frac{1}{T - 1} \sum_{j=2}^{T} (R_{j-1, j} - b_j) > 0.0$$
