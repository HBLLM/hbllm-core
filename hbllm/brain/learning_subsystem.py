"""Learning Subsystem — shared learning infrastructure container.

Eliminates late binding and None-check proliferation by providing a
single container that all learning consumers receive at construction time.

Consumers:
    LearningEventHandler  — experience-time lightweight routing
    AutonomousLearner     — research-time heavy operations
    SleepNode             — consolidation (decay, promotion, strategy)
    SkillEngine           — skill-mechanism linking

Future evolution:
    This is the seed of what eventually becomes CognitiveSubsystem or
    UnifiedCognitiveGraph — a single shared substrate for all cognitive
    operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hbllm.brain.belief_store import BeliefStore
    from hbllm.brain.causality.causal_model_builder import CausalModelBuilder
    from hbllm.brain.concept_formation import ConceptFormationEngine
    from hbllm.brain.contradiction_detector import (
        BeliefRevisionEngine,
        ContradictionDetector,
    )
    from hbllm.brain.failure_analyzer import FailureAnalyzer
    from hbllm.brain.mechanism_store import MechanismStore
    from hbllm.brain.meta_learner import MetaLearner


@dataclass
class LearningSubsystem:
    """Shared learning infrastructure.

    Both AutonomousLearner (research-time) and LearningEventHandler
    (experience-time) receive the same instance.  This ensures:

    - No late binding (everything injected at construction)
    - No dependency drift (single source of truth)
    - Each evidence source updates beliefs exactly once

    The subsystem is a plain container — no behavior, no bus subscriptions.
    Behavior lives in the consumers.
    """

    # Always available (created with SkillEngine)
    mechanism_store: MechanismStore | None = None
    failure_analyzer: FailureAnalyzer | None = None

    # Available when autonomous learning is injected
    belief_engine: BeliefRevisionEngine | None = None
    contradiction_detector: ContradictionDetector | None = None
    meta_learner: MetaLearner | None = None
    causal_model_builder: CausalModelBuilder | None = None

    # Phase 3: Persistent beliefs (storage layer)
    belief_store: BeliefStore | None = None

    # Phase 2: Concept formation
    concept_engine: ConceptFormationEngine | None = None

    @property
    def has_belief_infrastructure(self) -> bool:
        """True if belief revision pipeline is available."""
        return self.belief_engine is not None and self.contradiction_detector is not None

    @property
    def has_research_infrastructure(self) -> bool:
        """True if full research pipeline is available."""
        return (
            self.causal_model_builder is not None
            and self.meta_learner is not None
            and self.has_belief_infrastructure
        )

    def summary(self) -> dict[str, Any]:
        """Return a summary of available components."""
        return {
            "mechanism_store": self.mechanism_store is not None,
            "failure_analyzer": self.failure_analyzer is not None,
            "belief_engine": self.belief_engine is not None,
            "belief_store": self.belief_store is not None,
            "contradiction_detector": self.contradiction_detector is not None,
            "meta_learner": self.meta_learner is not None,
            "causal_model_builder": self.causal_model_builder is not None,
            "concept_engine": self.concept_engine is not None,
            "has_belief_infrastructure": self.has_belief_infrastructure,
            "has_research_infrastructure": self.has_research_infrastructure,
        }
