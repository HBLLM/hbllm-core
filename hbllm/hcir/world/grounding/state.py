"""Cortical Grounding State & Retention Management."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.grounding.concepts.concept_learner import ConceptAssignment
from hbllm.hcir.grounding.contracts.actions import ActionExecutionRecord, SymbolicActionIntent
from hbllm.hcir.grounding.contracts.dialogue import GoalProposal
from hbllm.hcir.grounding.features.feature_extractor import PerceptualFeatureVector
from hbllm.hcir.grounding.predictive.assimilator import GroundingSurpriseAlert
from hbllm.hcir.grounding.predictive.predictor import PredictedEntityState
from hbllm.hcir.world.percept import RetentionPolicy


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class CorticalGroundingState:
    """Working state maintained by the CorticalGroundingBridge across cognitive cycles."""

    FIELD_RETENTION = {
        "active_predictions": RetentionPolicy.EPISODE_LOCAL,
        "last_surprise_alerts": RetentionPolicy.EPISODE_LOCAL,
        "pending_epistemic_goals": RetentionPolicy.EPISODE_LOCAL,
    }

    step_counter: int = 0
    active_predictions: dict[str, PredictedEntityState] = field(default_factory=dict)
    last_dispatched_intent: SymbolicActionIntent | None = None
    last_execution_record: ActionExecutionRecord | None = None
    last_surprise_alerts: list[GroundingSurpriseAlert] = field(default_factory=list)
    pending_epistemic_goals: list[GoalProposal] = field(default_factory=list)
    last_feature_vectors: list[PerceptualFeatureVector] = field(default_factory=list)
    last_concept_assignments: list[ConceptAssignment] = field(default_factory=list)
    last_feature_concept_ids: dict[int, str] = field(default_factory=dict)
    last_role_priors: dict[int, dict[str, float]] = field(default_factory=dict)
    dialogue_role_priors: dict[int, dict[str, float]] = field(default_factory=dict)
    last_scene_sig: tuple[Any, ...] | None = None
    episode_feature_memory: dict[int, PerceptualFeatureVector] = field(default_factory=dict)
    cached_entity_features: dict[str, PerceptualFeatureVector] = field(default_factory=dict)
    cached_entity_concepts: dict[str, ConceptAssignment] = field(default_factory=dict)
    cached_entity_priors: dict[str, dict[str, float]] = field(default_factory=dict)
    adapted_role_pairs: set[tuple[int, str]] = field(default_factory=set)
    verified_outcomes_count: int = 0
    rejected_actions_count: int = 0
