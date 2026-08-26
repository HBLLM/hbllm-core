# Explicit Metacognitive Self-Model Architecture

The **Metacognitive Self-Model** is the internal self-regulatory substrate of HBLLM. While the persistent world model tracks external physical reality and the relational transfer engine maps structural knowledge across domains, the Metacognitive Self-Model **models the reliability, competence boundaries, calibration, computational budgets, and failure modes of the system's own cognitive machinery**.

```text
                 ┌──────────────────────────────────────────┐
                 │       EXTERNAL COGNITIVE ACTIVITY        │
                 └────────────────────┬─────────────────────┘
                                      │
                                 predictions
                                 actions
                                 outcomes
                                      │
                                      ▼
                        Metacognitive Monitor Engine
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
             Competence          Calibration         Structural
              Evidence            Evidence          Model Errors
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      ▼
                      EXPLICIT SELF-MODEL REPERTOIRE
                                      │
                  ┌───────────────────┼───────────────────┐
                  ▼                   ▼                   ▼
              Epistemic           Cognitive            Strategy
             Calibration           Budgets             Switching
                  │                   │                   │
                  └───────────────────┼───────────────────┘
                                      ▼
                            Decision & Action Policy
```

---

## 1. Core Architectural Invariants

1. **Performance-Grounded Self-Evidence Invariant**:
   - The self-model updates exclusively from recorded empirical attempts, outcomes, and prediction errors (`SelfModelEvidence`), never from self-assigned or ungrounded assertions.
   - Symmetrical with external epistemics: *"Claims about the world require perceptual evidence; claims about the self require empirical performance evidence."*
2. **Competence vs Calibration Dissociation**:
   - **Competence**: Historical task success rate $\frac{\text{successes}}{\text{attempts}}$ within a specific domain context.
   - **Calibration**: The mathematical alignment between predicted probability and empirical outcomes, quantified via quadratic Brier scores and Expected Calibration Error ($ECE$).
   - A system can be low-competence but exceptionally well-calibrated (accurately predicting low probability), or high-competence but poorly calibrated (delusional overconfidence).
3. **Three-Way Uncertainty Decomposition**:
   - **Epistemic Uncertainty**: Uncertainty arising from sparse data or novel contexts ($N < 3$), which can be reduced by gathering information.
   - **Aleatoric Noise**: Inherent physical variance or environmental stochasticity ($p \approx 0.5$).
   - **Structural / Model Uncertainty**: Systematic prediction failures under high confidence signaling a fundamental flaw or missing variable in the world model.
4. **Epistemic Honesty Under Resource Constraints**:
   - When elevated cognitive load forces simulation depth or branch throttling, the budget manager applies an explicit `uncertainty_penalty` rather than pretending that shallow simulation maintains full certainty.
5. **Event-Driven Strategy Switching State Machine**:
   - Repeated prediction failures or structural model mismatches halt blind retries and transition into root-cause diagnosis to select targeted remediation (epistemic probing, schema specialization, model learning, or budget adjustment).

---

## 2. Contextual Competence Profiles (`CompetenceProfile`)

Rather than maintaining a single scalar score, the system tracks contextual competence boundaries:

```python
@dataclass
class CompetenceProfile:
    domain: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    competence: float = 0.50
    epistemic_maturity: EpistemicMaturity = EpistemicMaturity.NOVICE  # NOVICE -> CALIBRATING -> MATURE
    known_competent_conditions: list[dict[str, Any]] = field(default_factory=list)
    uncalibrated_conditions: list[dict[str, Any]] = field(default_factory=list)
    evidences: list[SelfModelEvidence] = field(default_factory=list)
```

### Provenance-Bearing Evidence (`SelfModelEvidence`)
Every attempt is recorded with full provenance:
- `evidence_id`: Cryptographic identifier.
- `domain` & `context`: Domain category and relevant physical properties.
- `predicted_confidence`: Confidence score predicted before execution.
- `actual_outcome`: True binary outcome observed in physical reality.
- `timestamp`: Monotonic execution timestamp.

---

## 3. Epistemic Calibration Engine (`EpistemicCalibrator`)

The calibrator evaluates the reliability of the system's probabilistic forecasts:

### Brier Calibration Score
Calculates quadratic error between predicted confidence $p_i$ and true binary outcome $o_i \in \{0, 1\}$:
$$BS = \frac{1}{N} \sum_{i=1}^N (p_i - o_i)^2$$

### Expected Calibration Error ($ECE$)
Groups predictions into $M$ confidence bins $[b_{m-1}, b_m)$ and computes the weighted absolute difference between average accuracy and average confidence:
$$ECE = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### Overconfidence Penalty
Detects high-confidence failures ($p \ge 0.80 \land \text{outcome} = \text{False}$) and applies an explicit calibration penalty, signaling potential structural model deficiencies.

---

## 4. Cognitive Resource Budgeting (`CognitiveBudgetManager`)

The budget manager allocates computational depth based on current cognitive load:

```text
Cognitive Load < 0.70  ──►  Full Simulation Depth & Branching (No penalty)
Cognitive Load >= 0.70 ──►  Throttled Depth (Incurs explicit uncertainty_penalty >= 0.20)
```

This prevents the system from mistaking reduced computational search for high epistemic certainty.

---

## 5. Metacognitive Monitoring & Strategy Switching (`MetacognitiveMonitor`)

The monitor oversees execution and drives a formal self-correction state machine:

```text
                  ┌─────────────────────────┐
                  │         NORMAL          │
                  └────────────┬────────────┘
                               │ Prediction Error
                               ▼
                  ┌─────────────────────────┐
                  │    PREDICTION_ERROR     │
                  └────────────┬────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    (Random Noise / 1st Fail)       (Repeated Fail / Structural)
  ┌───────────────────────────┐   ┌───────────────────────────┐
  │       RETRY_ALLOWED       │   │         DIAGNOSE          │
  └───────────────────────────┘   └────────────┬──────────────┘
                                               │ Root Cause
                 ┌──────────────┬──────────────┼──────────────┐
                 ▼              ▼              ▼              ▼
           ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
           │ Epistemic │  │Relational │  │Predictive │  │  Adjust   │
           │   Probe   │  │Specialize │  │ Learning  │  │  Budget   │
           └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                 │              │              │              │
                 └──────────────┴──────┬───────┴──────────────┘
                                       ▼
                          ┌───────────────────────────┐
                          │        RE_EVALUATE        │
                          └────────────┬──────────────┘
                                       │ Verification
                                       ▼
                          ┌───────────────────────────┐
                          │          NORMAL           │
                          └───────────────────────────┘
```

### Diagnostic Failure Categorization (`FailureCause`)
- `KNOWLEDGE_GAP`: Missing property $\to$ Triggers active epistemic probing.
- `SCHEMA_MISMATCH`: Relational mapping failure $\to$ Triggers analogical schema specialization.
- `MODEL_INADEQUACY`: Structural predictive error $\to$ Triggers causal rule revision.
- `RESOURCE_STARVATION`: Truncated search $\to$ Requests budget escalation.
- `SEARCH_CYCLE_DETECTED`: Circular action oscillation ($A \to B \to A \to B$) $\to$ Forces exploratory branching.

---

## 6. Integration with Decision Policy

The Metacognitive Self-Model modulates the decision engine by scaling risk and value of information:

$$R_{\text{effective}} = \min(1.0, R_{\text{sim}} + \lambda_m \cdot U_{\text{model}})$$
$$VoI_{\text{effective}} = VoI \cdot (1.0 + \lambda_u \cdot U_{\text{epistemic}})$$

In uncalibrated or novel domains, elevated epistemic uncertainty dynamically prioritizes discriminative information gathering over premature goal commitment.
