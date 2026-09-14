"""Phase 2: Blank Brain Cognitive Initialization Profile.

Enforces strict separation between innate cognitive architecture (frozen)
and acquired developmental knowledge (initialized strictly empty).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.continual.store import DualStoreMemory
from hbllm.brain.reasoning.operators.active_inference import ActiveInferenceOperator
from hbllm.brain.reasoning.operators.registry import create_default_operator_registry
from hbllm.brain.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.brain.simulation.counterfactual_engine import MentalSandbox
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState

from .types import DevelopmentalProfile

logger = logging.getLogger(__name__)


@dataclass
class BlankBrainSubstrate:
    """Pre-existing, innate cognitive architecture with empty learned knowledge."""

    # --- Layer 1: Innate Cognitive Machinery ---
    workspace: HCIRWorkspaceState = field(default_factory=HCIRWorkspaceState)
    transaction_mgr: TransactionManager | None = None
    memory: DualStoreMemory = field(default_factory=DualStoreMemory)
    mental_sandbox: MentalSandbox = field(default_factory=MentalSandbox)
    runtime: UnifiedReasoningRuntime | None = None
    active_inference: ActiveInferenceOperator = field(default_factory=ActiveInferenceOperator)

    # --- Layer 2: Developmentally Learned Stores (Initialized Empty) ---
    semantic_concepts: dict[str, Any] = field(default_factory=dict)
    object_categories: dict[str, Any] = field(default_factory=dict)
    causal_rules: list[dict[str, Any]] = field(default_factory=list)
    affordances: dict[str, list[str]] = field(default_factory=dict)
    spatial_schemas: list[dict[str, Any]] = field(default_factory=list)
    procedural_skills: dict[str, Any] = field(default_factory=dict)
    lexical_mapping: dict[str, str] = field(default_factory=dict)

    profile: DevelopmentalProfile = field(default_factory=DevelopmentalProfile)

    def __post_init__(self) -> None:
        if self.transaction_mgr is None:
            self.transaction_mgr = TransactionManager(workspace=self.workspace)
        if self.runtime is None:
            registry = create_default_operator_registry()
            self.runtime = UnifiedReasoningRuntime(registry)

    def verify_isolation(self) -> bool:
        """Scientifically assert that all learned knowledge layers are strictly empty."""
        assert len(self.semantic_concepts) == 0, "Learned concepts must be empty initially!"
        assert len(self.object_categories) == 0, (
            "Learned object categories must be empty initially!"
        )
        assert len(self.causal_rules) == 0, "Learned causal rules must be empty initially!"
        assert len(self.affordances) == 0, "Learned affordances must be empty initially!"
        assert len(self.spatial_schemas) == 0, "Learned spatial schemas must be empty initially!"
        assert len(self.procedural_skills) == 0, (
            "Learned procedural skills must be empty initially!"
        )
        assert len(self.lexical_mapping) == 0, "Learned lexicon must be empty initially!"
        return self.profile.verify_blank_brain()


def create_blank_brain_substrate() -> BlankBrainSubstrate:
    """Instantiate a pristine developmental Blank Brain substrate."""
    profile = DevelopmentalProfile()
    substrate = BlankBrainSubstrate(profile=profile)
    substrate.verify_isolation()
    return substrate
