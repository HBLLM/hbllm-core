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
2. **Cohort B: Neural Learner** (Gradient/tabular Q baseline over feature vector).
3. **Cohort C: Mature HCIR** (HCIR with pre-compiled causal rules).
4. **Cohort D: Active Developmental HCIR** (Blank Brain + active contrastive probe selection).
5. **Cohort E: Passive Developmental HCIR** (Blank Brain + random probe selection).

## 6. Dependent Variables & Metrics
* **Primary**: $N_\tau$ (interventions to confirm true causal rule, reported as median, mean, and 95% CI).
* **Secondary**: Wasted interventions, false hypotheses generated, Brier score.
* **Generalization**:
  - Level 1: Training world accuracy.
  - Level 2: Held-out unseen entities (green cylinder, yellow cone, purple torus).
  - Level 3: Held-out unseen environment (ramp, container, orange block, heavy box).
