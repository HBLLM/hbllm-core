---
title: "Epistemic Runtime — Domain-Neutral Discovery Engine"
description: "Architecture deep-dive into HBLLM's epistemic runtime: continuous perceptual integration, multidimensional temporal evidence modeling, odds-space Bayesian belief revision, deterministic replay, decoupled provider reputation, calibration, and counterfactual reasoning — all domain-neutral."
---

# Epistemic Runtime — Domain-Neutral Discovery Engine

> **Core thesis**: The epistemic runtime should never know what medicine, robotics, chemistry, or finance are. It should only know about observations, evidence, uncertainty, hypotheses, predictions, experiments, and belief revision.
>
> **Architectural Invariant**: Perception produces observations and candidate signals. Epistemics owns interpretation, evidence evaluation, and belief revision. HCIR owns the resulting state and immutable history. Replay reconstructs the same state from the same causal inputs.

The Epistemic Runtime is HBLLM's autonomous discovery and belief lifecycle layer. It gives the system the ability to **evaluate continuous perceptual streams**, **model temporal dependence and novelty**, **form hypotheses**, **design experiments**, **evaluate evidence**, **revise beliefs in odds-space**, **calibrate its own reasoning**, **reconstruct deterministic states via event sourcing**, and **isolate provider reputation from circular internal feedback** — without domain-specific knowledge.

**Module:** `hbllm.brain.epistemics`

---

## Design Philosophy

### Epistemic vs. Domain Knowledge

```
Reality belongs to the World Model.
Perception extracts structured observations.
Epistemics reasons ABOUT reality and forms beliefs.
```

The runtime operates on abstract epistemic primitives:

| Primitive | Description | Not |
|---|---|---|
| Observation | Raw perceptual input with provider provenance | "A person is speaking" |
| Evidence | Evaluated observation with temporal novelty & reliability | "High-quality acoustic evidence (novelty=0.85)" |
| Hypothesis | Testable claim | "Speaker is Alice" |
| Prediction | Falsifiable expectation with deadline | "Alice will appear on camera within 2s" |
| Experiment | Test design ranked by information gain | "Cross-modal sensor probe" |
| Belief | Confidence-weighted claim with transition audit chain | "Alice is in the room (p=0.88, rev=4)" |
| BeliefTransition | Event-sourced, immutable record of belief confidence delta | "prior=0.50 → post=0.82 via LR_eff=4.5" |

### Epistemic Hierarchy & Continuous Flow

```
Perception (Audio / Vision / Sensor Streams)
   │
   │ observations + provenance
   ▼
HCIR Graph (AudioObservationNode / VisualObservationNode)
   │
   │ evidence extraction + correlation
   ▼
Evidence Quality Evaluation (PerceptualEvidenceEvaluator)
   │
   ├── sensory clarity & signal quality
   ├── model confidence
   └── temporal stability
   │
   ▼
Multidimensional Novelty Modeling (TemporalEvidenceModel)
   │
   ├── temporal half-life decay: n_t = 1 - 2^(-Δt / T_1/2)
   ├── semantic distance (Jaccard token divergence)
   ├── state-change override (LabelStateChangeDetector)
   └── temporal pattern classification (persistent, transition, transient, periodic)
   │
   ▼
Proposition Likelihood Evaluation (EpistemicLikelihoodEvaluator)
   │
   ├── P(E | H) and P(E | ¬H) discrimination
   ├── raw Likelihood Ratio: LR = P(E|H) / P(E|¬H)
   └── effective Likelihood Ratio: LR_effective = LR^novelty
   │
   ▼
Bayesian Belief Revision (DiscoveryBeliefManager)
   │
   ├── (evidence_id, proposition_id) idempotency guard
   ├── odds-space update: O(H|E) = O(H) * LR_effective
   ├── posterior confidence: P(H|E) = O(H|E) / (1 + O(H|E))
   └── BeliefTransitionNode emission + EvidenceNode incorporation mark
   │
   ▼
HCIR Persistent State & Epistemic Event Journal (replay.py)
```

---

## Four Decoupled Epistemic Dimensions

To avoid compounding errors and score collapse in continuous streaming perception, HBLLM explicitly decouples evidence evaluation into four orthogonal dimensions:

```
Evidence
 ├── 1. Identity   → (evidence_id, proposition_id) idempotency key
 ├── 2. Novelty    → TemporalEvidenceModel (temporal, semantic, state-change)
 ├── 3. Validity   → SourceReputation (decoupled: signal, concordance, empirical)
 └── 4. Likelihood → PropositionLikelihood (LR_raw, LR_effective = LR^novelty)
```

### 1. Identity & Idempotency Key
An evidence item can cause **at most one authoritative belief transition per proposition**:
- Key: `(evidence_id, proposition_id)`.
- Re-submitting the same evidence for the same belief produces an immediate no-op transition ($\Delta = 0.0, \text{LR}_{\text{effective}} = 1.0$) without modifying belief revision history.
- The same evidence can legitimately update multiple independent propositions (e.g., "person detected" updates both "room occupied" and "sensor operational").

### 2. Multidimensional Novelty & Temporal Dependence
Continuous perception (e.g., 30 fps video or continuous audio) generates highly correlated observations. Rather than treating each frame as independent:
- **Temporal Novelty**: Modeled via half-life $T_{1/2}$:
  $$n_t = 1 - 2^{-\Delta t / T_{1/2}}$$
- **Semantic Novelty**: Jaccard distance over candidate labels and tags:
  $$n_s = 1 - J(\text{tags}_{\text{current}}, \text{tags}_{\text{prior}})$$
- **State Change Override**: When a physical state transition occurs (e.g., "sitting" $\to$ "standing"), the state detector overrides temporal decay, granting high novelty even at small $\Delta t$.
- **Pattern Classification**: Labels streams as `PERSISTENT`, `TRANSITION`, `TRANSIENT`, `PERIODIC`, or `UNKNOWN`.

### 3. Decoupled Provider Reputation & Anti-Circularity Guard
Provider reputation remains strictly **outside the internal belief feedback loop**:
```
                 PROVIDER / SENSOR
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
   signal_quality  concordance  empirical_accuracy
                                       ▲
                                       │
                               external ground truth
```
- **`signal_quality`**: SNR, sensor clarity, raw signal quality.
- **`cross_modal_concordance`**: Agreement with other modalities (e.g., audio $\leftrightarrow$ vision agreement). Does **not** affect empirical accuracy.
- **`empirical_accuracy`**: Updated **only** by external ground truth (`OutcomeType.EXPERIMENT`, `USER_CONFIRMATION`, `TOOL_EXECUTION`, `EXTERNAL_VERIFICATION`). Internal belief convergence cannot self-reinforce provider reputation.

### 4. Proposition Likelihood & Dependence Correction
- The evaluator computes hypothesis discrimination $P(E|H)$ and $P(E|\neg H)$.
- Raw likelihood ratio: $\text{LR} = \frac{P(E|H)}{P(E|\neg H)}$.
- Dependence-corrected effective likelihood ratio:
  $$\text{LR}_{\text{effective}} = \text{LR}^{\text{novelty}}$$
- When novelty approaches 0 (redundant continuous frame), $\text{LR}_{\text{effective}} \to 1.0$, preventing confidence runaway.

---

## Architecture & Subsystems

```mermaid
graph TB
    subgraph "Epistemic Runtime (brain/epistemics/)"
        subgraph "Perceptual Epistemic Integration"
            PEE["PerceptualEvidenceEvaluator\n(signal quality & reliability)"]
            TEM["TemporalEvidenceModel\n(novelty & pattern classification)"]
            ELE["EpistemicLikelihoodEvaluator\n(odds-space LR_effective)"]
        end

        subgraph "Belief & Hypothesis Subsystems"
            BM["DiscoveryBeliefManager\n(Bayesian odds revision)"]
            REP["SourceReputationTracker\n(decoupled 3-score trust)"]
            WS["DiscoveryWorkspace\n(research programs)"]
            HB["HypothesisBuilder\n(validate + promote)"]
            PT["PredictionTracker\n(competing predictions)"]
        end

        subgraph "Discovery & Anomaly Engines"
            CUR["CuriosityEngine\n(investigation ranking)"]
            IG["IdeaGenerator\n(creative hypotheses)"]
            EP["ExperimentPlanner\n(info-gain ranking)"]
            CE["ContradictionEngine\n(cross-modal anomaly scan)"]
            EX["ExplanationEngine\n(provenance chains)"]
            RS["StrategyManager\n(pluggable strategies)"]
            LOOP["EpistemicLoop\n(orchestrator)"]
        end

        subgraph "Meta-Cognition & Deterministic Replay"
            MEM["EpistemicMemory\n(SQLite history)"]
            CAL["CalibrationEngine\n(ECE & meta-epistemics)"]
            CF["CounterfactualReasoner\n(what-if sensitivity)"]
            JRN["EpistemicEventJournal\n(causal event stream)"]
            RPL["JournalReplayHarness\n(deterministic verification)"]
        end
    end

    PEE --> TEM
    TEM --> ELE
    ELE --> BM
    BM --> LOOP

    CUR --> LOOP
    LOOP --> IG
    IG --> HB
    HB --> PT
    PT --> EP
    EP --> PEE
    CE --> CUR

    LOOP --> JRN
    JRN --> RPL
    MEM --> CAL
    CAL --> RS
    CF --> EP
```

---

## Module Inventory

### Perceptual Epistemics & Scalability (Waves A9–A11)

| Module | Lines | Purpose |
|---|---|---|
| `temporal_evidence_model.py` | 460 | Multidimensional novelty assessment ($T_{1/2}$ half-life, Jaccard distance, state-change override, pattern classification). |
| `likelihood_evaluator.py` | 320 | Proposition-specific likelihood evaluation with dependence correction ($\text{LR}_{\text{eff}} = \text{LR}^n$). |
| `perceptual_evaluator.py` | 180 | General sensory evidence quality and reliability assessment. |
| `replay.py` | 560 | Event-sourced `EpistemicEventJournal` and `JournalReplayHarness` for deterministic state reconstruction. |
| `belief_manager.py` | 530 | Odds-space Bayesian revision, $(E, H)$ idempotency enforcement, `BeliefTransitionNode` creation. |
| `reputation.py` | 370 | Three-dimensional source reputation (`signal_quality`, `concordance`, `empirical_accuracy`) with anti-circularity guard. |

### Closed Discovery Loop (Waves 1–3)

| Module | Lines | Purpose |
|---|---|---|
| `curiosity_engine.py` | 300 | Prioritizes investigation targets based on unknowns and cross-modal contradictions. |
| `idea_generator.py` | 350 | Generates hypotheses from unknowns, contradictions, and anomalies. Memory-filtered. |
| `hypothesis_builder.py` | 280 | Validates, deduplicates, and promotes raw ideas to HCIR `HypothesisNode`s. |
| `prediction_tracker.py` | 320 | Registers competing predictions with deadlines and evaluates outcomes. |
| `experiment_planner.py` | 380 | Designs discriminative experiments ranked by information gain. |
| `contradiction_engine.py` | 340 | Cross-modal contradiction scanning across belief, evidence, and perceptual nodes. |
| `explanation.py` | 334 | Graph-traversal provenance chains: belief $\to$ transition $\to$ evidence $\to$ observation. |
| `research_strategy.py` | 270 | Pluggable research strategies with dynamic weighting. |
| `epistemic_loop.py` | 510 | Orchestrates autonomous cycles on cognitive ticks. |
| `workspace.py` | 867 | Research program lifecycle: objectives, questions, hypotheses, unknowns, experiments. |
| `epistemic_memory.py` | 600 | SQLite-backed reasoning history (hypotheses, predictions, snapshots, biases). |
| `calibration.py` | 400 | Meta-epistemic self-calibration: ECE curves, bias detection, strategy adjustment. |
| `counterfactual.py` | 450 | "What if..." analysis: hypothesis falsification, evidence removal, sensitivity analysis. |

---

## Event Sourcing & Deterministic Replay

Replay correctness invariant:
$$\text{REPLAY}(\text{EVENT\_JOURNAL}) \equiv \text{LIVE\_STATE} \quad (\epsilon \le 10^{-5})$$

1. **`SESSION_CONFIG`**: First entry in the journal. Captures full `EpistemicRuntimeConfig`, `config_hash`, and `algorithm_version`. Replay aborts on hash mismatch.
2. **Monotonic Sequence Numbers**: Event ordering is governed authoritatively by integer `sequence_number`, never floating-point timestamps. Gaps or duplicates trigger `SequenceIntegrityError`.
3. **Causal Event Capture**: Only causal decisions (evidence committed, belief revised, contradictions detected) are journaled. Sliding windows and mutable caches are deterministically reconstructed during replay.

```python
from hbllm.brain.epistemics.replay import EpistemicEventJournal, JournalReplayHarness

# Recording during live session
journal = EpistemicEventJournal(config=runtime_config)
journal.record(EpistemicEventType.PERCEPTION_RECEIVED, {"observation_id": "obs_1"})
journal.record(EpistemicEventType.EVIDENCE_COMMITTED, {"evidence_id": "ev_1"})
journal.record(EpistemicEventType.BELIEF_REVISED, {"belief_id": "b1", "posterior_confidence": 0.82})

# Deterministic replay and verification
harness = JournalReplayHarness()
replayed_graph = harness.replay(journal.to_list(), expected_config=runtime_config)
harness.assert_graphs_equivalent(live_graph, replayed_graph, epsilon=1e-5)
```

---

## Test Verification

| Subsystem / Suite | Test Count | Status |
|---|---|---|
| `tests/unit/epistemics/` | 163 | All passing |
| `tests/integration/epistemics/` | 12 | All passing |
| `tests/unit/perception/` | 163 | All passing |
| `tests/unit/hcir/` | 479 | All passing |
| **Combined Epistemics & HCIR Suite** | **1,099** | **100% passing** |
| **Full Core Repository (`make test`)** | **5,854** | **100% passing** |
