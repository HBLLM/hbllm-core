# Experimental Protocol: A23.5-E1 Active Interventional Causal Discovery

## 1. Research Question
Can a minimally initialized HBLLM (Blank-Brain cognitive profile) discover true causal relationships through active intervention under deliberate observational confounding, and transfer the resulting causal rule to previously unseen entities and environments without task-specific neural training?

## 2. Innate Substrate vs. Learned Knowledge
* **Innate Cognitive Machinery (Frozen)**:
  HCIR graph representation, event-sourcing log, epistemic bookkeeping, mental sandbox simulation, active inference curiosity formulation.
* **Developmentally Learned (Empty at Initialization)**:
  Semantic concepts, causal rules, affordances, spatial schemas, object categories, procedural skills, lexical mappings.

## 3. Strict Perception Invariance
Perception outputs only raw sensory features (neutral IDs `entity_001`, `entity_002`, spatial coordinates, shape descriptors, tactile contact, mass estimate). Perception is forbidden from outputting semantic labels (`ball`, `container`) or pre-baked affordances (`rollable`, `pushable`).

## 4. Confounding Design
* **Observational Phase**:
  - Red Ball (mass=1.2) -> Moves
  - Red Block (mass=2.5) -> Moves
  - Blue Ball (mass=12.0) -> Does not move
  - Blue Block (mass=15.0) -> Does not move
  - Spurious Correlation: Color == 'red' correlates 100% with movement.
* **True Physical Causal Law**:
  - PUSH(x) ∧ MASS(x) < 5.0 => MOVE(x).
* **Interventional Probe Candidates**:
  - Blue Light Ball (mass=1.1, color='blue') -> Disproves Color Hypothesis.
  - Red Heavy Block (mass=14.5, color='red') -> Disproves Color Hypothesis.

## 5. Five-Cohort Comparison
1. **Cohort A: Scripted Baseline** (Hand-coded oracle).
2. **Cohort B: Neural Learner** (Parameterized 2-Layer MLP Baseline):
   - **Architecture**: 2-layer Multi-Layer Perceptron (Input: 7 -> Hidden: 16 -> Output: 1).
   - **Parameter Count**: 145 parameters ((7 * 16 + 16) + (16 * 1 + 1)).
   - **Input Representation**: $x = [\text{is\_red}, \text{is\_blue}, \text{other\_color}, \text{is\_ball}, \text{is\_block}, \text{other\_shape}, \text{normalized\_mass}]$.
   - **Activation**: Hidden ReLU, Output Sigmoid.
   - **Optimizer**: Gradient Descent with Momentum ($\text{learning\_rate} = 0.05, \text{momentum} = 0.90$).
   - **Exploration Policy**: $\epsilon$-greedy exploration ($\epsilon = 0.20$).
   - **Observation & Training Parity**: Receives identical perceptual observations and interaction history as HCIR cohorts.
3. **Cohort C: Mature HCIR** (HCIR with pre-compiled causal rules; serves as control condition).
4. **Cohort D: Active Developmental HCIR** (Blank Brain + active contrastive probe selection driven by epistemic surprise).
5. **Cohort E: Passive Developmental HCIR** (Blank Brain + random probe selection; curiosity ablation).

## 6. Dependent Variables & Metrics
* **Primary Metric**: $N_\tau$ (interventions to confirm true causal rule, reported as full discrete empirical distribution):
  - Discrete Histogram ($N_\tau = 1, 2, 3, \dots, \text{failure}$).
  - Median $N_\tau$ and Interquartile Range (IQR = $[Q_1, Q_3]$).
  - 95% Non-parametric Bootstrap Confidence Interval (1,000 resamples).
* **Efficiency & Epistemic Metrics**:
  - Wasted interventions (probes that do not resolve hypothesis disagreement).
  - False hypotheses generated and falsified.
  - Belief transition trajectory (`created` -> `tested` -> `falsified` -> `confirmed` -> `confidence_changed` -> `generalized`).
  - Brier score calibration.
* **Three-Tier Generalization**:
  - Level 1: Training world accuracy.
  - Level 2: Held-out unseen entities (green cylinder, yellow cone, purple torus).
  - Level 3: Held-out unseen environment (ramp, container, orange block, heavy box).

## 7. Experiment A23.5-E2: Causal Variable Invariance
* **Objective**: Test whether Developmental HCIR continues to identify the invariant physical cause (`mass < 5.0`) when surface correlations change across worlds:
  - World A: Red -> Light, Blue -> Heavy.
  - World B: Red -> Heavy, Blue -> Light (Color inversion).
  - World C: Square -> Light, Circle -> Heavy (Shape confounder).
  - World D: Large -> Light, Small -> Heavy (Size confounder).
* **Invariant Law**: $\text{PUSH}(x) \land \text{mass}(x) < \theta \implies \text{MOVES}(x)$.
* **Evaluation**: Developmental HCIR must consistently isolate `mass_sensation` and reject spurious surface correlations in all variations.

