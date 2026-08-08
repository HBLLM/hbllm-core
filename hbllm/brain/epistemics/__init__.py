"""Epistemics — the epistemic runtime layer of HBLLM.

A domain-neutral epistemic operating system layer that sits between
Memory and Planning in the cognitive architecture::

    Perception → World Model → Memory → **EPISTEMICS** → Planning

The epistemic runtime never knows what medicine, robotics, chemistry,
or finance are.  It only knows observations, evidence, uncertainty,
hypotheses, predictions, experiments, and belief revision.  Domain
knowledge arrives through plugins and ontologies.

Wave 1 — Epistemic Foundations::

    DiscoveryBeliefManager  — Bayesian belief revision
    DiscoveryWorkspace      — Research program lifecycle
    SourceReputationTracker — Epistemic trust tracking

Wave 2 — Closed Discovery Loop::

    EvidenceEvaluator       — Evidence quality/weight/bias → BeliefConfidence
    IdeaGenerator           — Raw creative generation from LLM
    HypothesisBuilder       — Filter funnel: ideas → validated hypotheses
    PredictionTracker       — Competing prediction tracking
    ExperimentPlanner       — Info-gain + economics experiments
    ContradictionEngine     — Proactive contradiction & anomaly hunting
    CuriosityEngine         — Self-directed investigation + economics
    ExplanationEngine       — "Why do you believe this?" via graph traversal
    ResearchStrategyManager — Pluggable research strategies
    EpistemicLoop           — Orchestrator (AutonomyCore-compatible)

Wave 3 — Epistemic Meta-Cognition::

    EpistemicMemory              — Universal reasoning history
    EpistemicCalibrationEngine   — "How good am I at knowing things?"
    CounterfactualReasoner       — "What if...?" via graph analysis

Design principles:
    1. Evolve existing cognition, don't duplicate it.
    2. Epistemics is a cognitive mode, not a separate subsystem.
    3. One brain, not two.
    4. Domain-neutral: never encode domain-specific knowledge.
    5. Everything can create new unknowns. Discovery is recursive.
    6. The system reasons about how it reasons (meta-cognition).
"""

# ── Wave 1: Epistemic Foundations ──────────────────────────────────────
from hbllm.brain.epistemics.belief_manager import DiscoveryBeliefManager
from hbllm.brain.epistemics.reputation import SourceReputation, SourceReputationTracker
from hbllm.brain.epistemics.workspace import DiscoveryWorkspace, ResearchProgram

# ── Wave 2: Epistemic Engines ─────────────────────────────────────────
from hbllm.brain.epistemics.contradiction_engine import ContradictionEngine
from hbllm.brain.epistemics.curiosity_engine import CuriosityEngine
from hbllm.brain.epistemics.epistemic_loop import EpistemicLoop
from hbllm.brain.epistemics.evidence_evaluator import EvidenceEvaluator
from hbllm.brain.epistemics.experiment_planner import ExperimentPlanner
from hbllm.brain.epistemics.explanation import ExplanationEngine
from hbllm.brain.epistemics.hypothesis_builder import HypothesisBuilder
from hbllm.brain.epistemics.idea_generator import IdeaGenerator
from hbllm.brain.epistemics.prediction_tracker import PredictionTracker
from hbllm.brain.epistemics.research_strategy import (
    ResearchStrategyManager,
    StrategyConfig,
)

# ── Wave 3: Meta-Cognition ────────────────────────────────────────────
from hbllm.brain.epistemics.calibration import EpistemicCalibrationEngine
from hbllm.brain.epistemics.counterfactual import CounterfactualReasoner
from hbllm.brain.epistemics.epistemic_memory import EpistemicMemory

# ── Wave 4: Integration ──────────────────────────────────────────
from hbllm.brain.epistemics.integration import wire_epistemics

# ── Protocols ──────────────────────────────────────────────────────────
from hbllm.brain.epistemics.interfaces import (
    # Wave 2 Protocols
    IBeliefReviser,
    IContradictionSeeker,
    ICuriosityEngine,
    IEvidenceEvaluator,
    IExperimentDesigner,
    IExplanationEngine,
    IHypothesisBuilder,
    IIdeaGenerator,
    IPredictionTracker,
    ISourceReputationTracker,
    # Wave 3 Protocols
    ICounterfactualReasoner,
    IEpistemicCalibrator,
    IEpistemicMemory,
    # Wave 2 Data Types
    BeliefRevision,
    ContradictionReport,
    CuriositySignal,
    EpistemicTask,
    EvidenceAssessment,
    ExperimentDesign,
    ExplanationChain,
    ExplanationStep,
    HypothesisCandidate,
    InvestigationBudget,
    PredictionOutcome,
    RawIdea,
    # Wave 3 Data Types
    CalibrationReport,
    ConfidenceSnapshot,
    CounterfactualResult,
)

__all__ = [
    # ── Wave 1: Foundations ────────────────────────────────────────
    "DiscoveryBeliefManager",
    "DiscoveryWorkspace",
    "ResearchProgram",
    "SourceReputation",
    "SourceReputationTracker",
    # ── Wave 2: Engines ────────────────────────────────────────────
    "ContradictionEngine",
    "CuriosityEngine",
    "EpistemicLoop",
    "EvidenceEvaluator",
    "ExperimentPlanner",
    "ExplanationEngine",
    "HypothesisBuilder",
    "IdeaGenerator",
    "PredictionTracker",
    "ResearchStrategyManager",
    "StrategyConfig",
    # ── Wave 3: Meta-Cognition ─────────────────────────────────────
    "EpistemicCalibrationEngine",
    "CounterfactualReasoner",
    "EpistemicMemory",
    # ── Wave 4: Integration ──────────────────────────────────────
    "wire_epistemics",
    # ── Protocols ──────────────────────────────────────────────────
    "IBeliefReviser",
    "IContradictionSeeker",
    "ICuriosityEngine",
    "IEvidenceEvaluator",
    "IExperimentDesigner",
    "IExplanationEngine",
    "IHypothesisBuilder",
    "IIdeaGenerator",
    "IPredictionTracker",
    "ISourceReputationTracker",
    "ICounterfactualReasoner",
    "IEpistemicCalibrator",
    "IEpistemicMemory",
    # ── Data Types ─────────────────────────────────────────────────
    "BeliefRevision",
    "ContradictionReport",
    "CuriositySignal",
    "EpistemicTask",
    "EvidenceAssessment",
    "ExperimentDesign",
    "ExplanationChain",
    "ExplanationStep",
    "HypothesisCandidate",
    "InvestigationBudget",
    "PredictionOutcome",
    "RawIdea",
    "CalibrationReport",
    "ConfidenceSnapshot",
    "CounterfactualResult",
]
