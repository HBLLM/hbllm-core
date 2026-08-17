---
title: "API Reference — Epistemic Discovery & Calibration Runtime"
description: "API reference for the 15 domain-neutral epistemic engines: BeliefManager, CuriosityEngine, HypothesisBuilder, CalibrationEngine, and EpistemicMemory."
---

# Epistemic Discovery & Calibration API

The **Epistemic Runtime** gives HBLLM the capability to autonomously generate hypotheses, design experiments, evaluate evidence, revise beliefs via Bayesian updates, and calibrate its own uncertainty.

**Package:** `hbllm.brain.epistemics`

---

## Subsystem Index

| Class | Module | Role |
|---|---|---|
| `BeliefManager` | `belief_manager.py` | Bayesian belief confidence updates from evidence |
| `CuriosityEngine` | `curiosity_engine.py` | Prioritizes self-directed investigations from unknowns |
| `IdeaGenerator` | `idea_generator.py` | Generates candidate ideas filtered by past failure memory |
| `HypothesisBuilder` | `hypothesis_builder.py` | Validates, deduplicates, and promotes ideas to graph hypotheses |
| `PredictionTracker` | `prediction_tracker.py` | Registers competing predictions with verification deadlines |
| `ExperimentPlanner` | `experiment_planner.py` | Designs discriminative experiments ranked by information gain |
| `EvidenceEvaluator` | `evidence_evaluator.py` | Scores evidence quality and credibility |
| `ContradictionEngine`| `contradiction_engine.py` | Scans for graph anomalies and conflicting evidence edges |
| `ExplanationEngine` | `explanation.py` | Traces belief provenance (Belief → Evidence → Observation) |
| `ResearchStrategyManager` | `research_strategy.py`| Pluggable research strategies (Exploration, Verification, etc.) |
| `EpistemicLoop` | `epistemic_loop.py` | Orchestrates the full discovery cycle on each cognitive tick |
| `EpistemicMemory` | `epistemic_memory.py` | SQLite-backed historical database of hypothesis outcomes |
| `EpistemicCalibrationEngine` | `calibration.py` | Computes Expected Calibration Error (ECE) and detects bias |
| `CounterfactualReasoner` | `counterfactual.py` | "What-if" sensitivity analysis on belief networks |

---

## One-Line Integration: `wire_epistemics()`

**Module:** `hbllm.brain.epistemics.integration.wire_epistemics`

Wired directly to `AutonomyCore` or `BrainContainer`:

```python
from hbllm.brain.epistemics import wire_epistemics

epistemic_loop = wire_epistemics(
    autonomy_core=autonomy_core,
    graph=reality_graph,
    data_dir="./data/epistemics",
    calibration_interval=10,  # Calibrate every 10 ticks
)
```

---

## Core Engine Interfaces

### `BeliefManager`

```python
from hbllm.brain.epistemics.belief_manager import BeliefManager

bm = BeliefManager(graph=graph)

# Revise confidence based on new evidence
updated_belief = await bm.revise_belief(
    belief_id="belief-42",
    evidence_id="ev-101",
    evidence_quality=0.88,
    is_supporting=True,
)
print(f"New confidence: {updated_belief.confidence:.3f}")
```

### `CuriosityEngine`

```python
from hbllm.brain.epistemics.curiosity_engine import CuriosityEngine

curiosity = CuriosityEngine(workspace=workspace)
signals = await curiosity.prioritize_investigations(limit=5)

for sig in signals:
    print(
        f"Investigation Target: {sig.target_id}, Information Value: {sig.priority_score:.2f}"
    )
```

### `EpistemicCalibrationEngine`

```python
from hbllm.brain.epistemics.calibration import EpistemicCalibrationEngine
from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory

memory = EpistemicMemory(db_path="./data/epistemic_history.db")
calibrator = EpistemicCalibrationEngine(memory=memory)

# Generate calibration report
report = await calibrator.calibrate()
print(f"Expected Calibration Error (ECE): {report.ece:.4f}")
print(f"Prediction Accuracy: {report.accuracy:.1%}")
print(f"Detected Biases: {report.biases}")
```

### `CounterfactualReasoner`

```python
from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner

cf = CounterfactualReasoner(graph=graph)

# Perform sensitivity analysis
impact = await cf.what_if_hypothesis_wrong(hypothesis_id="hypo-99")
print(f"Impacted Dependent Beliefs: {impact.affected_beliefs}")
```
