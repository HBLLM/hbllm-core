# Developmental Cognitive Learning Architecture (Milestone A23 & A23.5)

## 1. Overview

The **Developmental Cognitive Learning Subsystem** (`plugins/developmental_adapter/`) enables embodied cognitive agents to bootstrap knowledge from a minimal cognitive state (**Blank Brain Substrate**) through active physical interaction, Piagetian sensorimotor exploration, and Socratic pedagogical instruction.

Milestone **A23.5** specifically introduces **Active Interventional Causal Discovery under Confounding**, proving that an agent can resolve Simpson's paradox and observational illusions purely through hypothesis-guided contrastive interventions ($do(X)$ calculus) without pre-trained language model hallucinations.

---

## 2. Architectural Diagram

```mermaid
graph TB
    subgraph SENSORIMOTOR["🌍 BabyWorld Physical Simulator"]
        ENV["BabyWorldEnvironment<br/>2D Continuous Physics & Objects"]
        CONF["Confounded World Generator<br/>Spurious Correlations (Color/Shape)"]
        PROBE["Physical Actuation<br/>PUSH, PULL, GRASP, LIFT"]
    end

    subgraph ADAPTER["🔌 Developmental Learning Plugin"]
        PERC["DevelopmentalPerceptionAdapter<br/>Pre-semantic raw visual/tactile tokens"]
        CAUSAL["InterventionalCausalDiscoveryEngine<br/>Active Inference & Shannon Entropy Probing"]
        SCHOOL["CognitiveSchool & PedagogicalTeacher<br/>Vygotskian Scaffolding & Exams"]
        CONSOL["ContinualDevelopmentEngine<br/>Dual-Store Memory & Sleep Cycles"]
        TELEM["DevelopmentalTelemetryEmitter<br/>Real-Time OTel & Prometheus Bridge"]
    end

    subgraph SUBSTRATE["🧠 Minimal Blank Brain Substrate"]
        RULES["Causal Rules Registry<br/>Induced Physical Laws"]
        AFFORD["Affordances Registry<br/>Action Predicates"]
        LEX["Grounded Lexicon<br/>Concept Mappings"]
    end

    ENV --> CONF
    CONF --> PERC
    PERC --> CAUSAL
    CAUSAL -->|Optimal Probe do(X)| PROBE
    PROBE --> ENV
    CAUSAL -->|Induced Rules| RULES
    SCHOOL --> CAUSAL
    SCHOOL --> CONSOL
    CONSOL --> SUBSTRATE
    CAUSAL --> TELEM
    SCHOOL --> TELEM
```

---

## 3. Core Components

### 3.1 Blank Brain Substrate (`blank_brain.py`)
To rigorously evaluate developmental learning, the agent starts tabula rasa:
- **Zero Pre-compiled Schemas**: No innate assumptions regarding physical mass, gravity, or friction.
- **Semantic Leak Prevention**: Ensures raw perceptual inputs (`mass_sensation`, `spatial_coordinates`) cannot access ground truth simulation metadata or labels.
- **Dual-Store Memory**: Integrates episodic event buffers with consolidated long-term semantic storage.

### 3.2 Active Interventional Causal Discovery (`causal_discovery.py`)
Resolves confounded observational correlations where spurious visual cues (e.g. all yellow objects move, purple objects do not) mask true physical laws (e.g., $effective\_resistance = mass \times friction < 5.0$):
1. **Hypothesis Formulation**: Generates candidate rules correlating discrete features to outcomes.
2. **Entropy-Driven Contrastive Probing**: Selects physical probes that maximize pairwise hypothesis disagreement (expected information gain / entropy reduction):
   $$\Delta H = H(\mathcal{H}_t) - \mathbb{E}[H(\mathcal{H}_{t+1} \mid do(a))]$$
3. **Strict Falsification**: Irreversibly falsifies invalidated hypotheses upon encountering counterexamples ($confidence \to 0.0$).
4. **Substrate Rule Induction**: Promotes hypotheses with posterior confidence $\ge 0.95$ into permanent HCIR causal rules.

### 3.3 Five-Cohort Benchmark Framework (`cohorts.py`)
Provides standardized, reproducible evaluation of causal learning across 5 distinct cognitive paradigms:
- **Cohort A (Scripted Baseline)**: Hand-coded oracle rules.
- **Cohort B (Neural Learner)**: Gradient-based policy without explicit causal reasoning.
- **Cohort C (Mature HCIR)**: Pre-compiled adult cognitive substrate.
- **Cohort D (Active Developmental HCIR)**: Blank brain + active contrastive intervention.
- **Cohort E (Passive Developmental HCIR)**: Blank brain + random observational exploration.

Key finding: Cohort D achieves $\ge 90\%$ causal discovery with minimal interventions ($N_\tau \le 8$), whereas passive and pure neural learners fail or require orders of magnitude more samples due to observational confounding.

### 3.4 Cognitive School & Pedagogical Curriculum (`school.py`, `teacher.py`, `trainer.py`)
A 3-semester Piagetian learning journey:
- **Semester 1 (Freshman)**: Perceptual lexicon grounding, fast-mapping, and sensory exposure.
- **Semester 2 (Sophomore)**: Causal mechanics, Socratic counterexamples, tool synthesis, and syntax.
- **Semester 3 (Senior)**: Relational analogy transfer and metacognitive deception defense.
- **Lifelong Learning Guarantees**: Backward transfer ($BWT \ge 0$) verified via dual-store sleep consolidation cycles.

---

## 4. Observability and Distributed Tracing

The developmental learning plugin integrates directly with the HBLLM enterprise telemetry plane:
- **Distributed Tracing**: OpenTelemetry spans (`trace_span`) track probe executions, hypothesis updates, and semester curriculum phases with graceful zero-dependency fallback.
- **Prometheus & In-Memory Metrics**:
  - `hbllm_developmental_hypotheses_total`: Counter tracking generated, falsified, and confirmed hypotheses.
  - `hbllm_developmental_concepts_total`: Counter tracking domain schema acquisitions.
  - `hbllm_developmental_interventions_total`: Counter tracking probe executions by motor action and result.
  - `hbllm_developmental_entropy`: Gauge tracking real-time Shannon belief entropy.
  - `hbllm_developmental_learning_duration_seconds`: Histogram tracking phase latencies.
- **DevelopmentalTelemetryEmitter**: Thread-safe facade providing real-time telemetry extraction (`get_telemetry_snapshot()`) for monitoring dashboards and operational audits.
