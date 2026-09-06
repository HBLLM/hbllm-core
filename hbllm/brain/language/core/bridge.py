"""Language-to-Capability Bridge for HBLLM.

Translates language-neutral SemanticFrames (from English, Sinhala, Tamil, etc.)
into typed HCIR capability execution intents, enforcing GovernanceEngine
fail-closed security policies before dispatch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.brain.language.core.semantic_frame import (
    FrameType,
    GroundedSemanticFrame,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.governance.epistemic_gate import EpistemicSafetyGate
from hbllm.hcir.kernel.governance.governance_engine import (
    GovernanceDecision,
    GovernanceEngine,
    StructuredIntent,
)

if TYPE_CHECKING:
    from hbllm.hcir.graph import CognitiveGraph

logger = logging.getLogger(__name__)


@dataclass
class CapabilityDispatchIntent:
    """A typed capability dispatch request derived from natural language."""

    capability_name: str
    action: str
    target: str
    parameters: dict[str, Any] = field(default_factory=dict)
    structured_intent: StructuredIntent = field(default_factory=StructuredIntent)


@dataclass
class BridgeExecutionResult:
    """Result of attempting to dispatch a linguistic command to a capability."""

    is_allowed: bool
    governance_decision: GovernanceDecision
    dispatch_intent: CapabilityDispatchIntent | None = None
    execution_result: dict[str, Any] | None = None
    error_message: str = ""


class LanguageCapabilityBridge:
    """Connects language comprehension to governed capability dispatch."""

    def __init__(
        self,
        governance_engine: GovernanceEngine | None = None,
        capability_resolver: CapabilityResolver | None = None,
        graph: CognitiveGraph | None = None,
        epistemic_gate: EpistemicSafetyGate | None = None,
    ) -> None:
        self._gov = governance_engine or GovernanceEngine(
            graph=graph, epistemic_gate=epistemic_gate
        )
        if epistemic_gate is not None and self._gov.epistemic_gate is None:
            self._gov.attach_epistemic_gate(epistemic_gate)
        elif graph is not None and self._gov.epistemic_gate is None:
            self._gov.attach_epistemic_gate(EpistemicSafetyGate(graph=graph))
        self._resolver = capability_resolver

    def compile_intent(
        self,
        frame: SemanticFrame | GroundedSemanticFrame,
    ) -> CapabilityDispatchIntent:
        """Compile a SemanticFrame into a typed CapabilityDispatchIntent."""
        base_frame = frame.frame if isinstance(frame, GroundedSemanticFrame) else frame

        predicate = (base_frame.predicate or "").lower().strip()
        patient_ref = (
            base_frame.get_role(ThematicRole.PATIENT)
            or base_frame.get_role(ThematicRole.THEME)
            or base_frame.get_role(ThematicRole.DESTINATION)
        )
        target_name = ((patient_ref.concept_name or "") if patient_ref else "").lower().strip()

        # Map semantic frame to capability domain
        structured_intent = StructuredIntent(
            action=predicate,
            target=target_name,
        )

        if predicate in ("open", "unlock", "lock", "close") or any(
            p in target_name for p in ("door", "gate", "entrance", "lock", "safe", "vault")
        ):
            cap_name = "perimeter_control"
            action = predicate or "open"
        elif predicate in ("rotate", "swing", "move", "push", "drive", "lift") or any(
            a in target_name for a in ("arm", "robot", "gripper", "actuator", "manipulator")
        ):
            cap_name = "actuator_control"
            action = predicate or "actuate"
        elif base_frame.frame_type == FrameType.QUERY:
            cap_name = "environment_query"
            action = "query"
        else:
            cap_name = f"task_{predicate or 'execute'}"
            action = predicate or "execute"

        grounded_entity_ids: list[str] = []
        if isinstance(frame, GroundedSemanticFrame):
            grounded_entity_ids = list(frame.grounded_entities.values())

        params: dict[str, Any] = {"action": action, "target": target_name}
        if grounded_entity_ids:
            params["grounded_entity_ids"] = grounded_entity_ids

        return CapabilityDispatchIntent(
            capability_name=cap_name,
            action=action,
            target=target_name,
            parameters=params,
            structured_intent=structured_intent,
        )

    async def execute_language_command(
        self,
        frame: SemanticFrame | GroundedSemanticFrame,
        context: dict[str, Any] | None = None,
    ) -> BridgeExecutionResult:
        """Evaluate linguistic intent under GovernanceEngine and execute if safe."""
        eval_ctx = dict(context or {})
        dispatch_intent = self.compile_intent(frame)

        if "grounded_entity_ids" in dispatch_intent.parameters:
            eval_ctx.setdefault(
                "grounded_entity_ids", dispatch_intent.parameters["grounded_entity_ids"]
            )

        # Merge intent params with runtime context
        call_args = {**dispatch_intent.parameters, **eval_ctx}
        call_args["intent"] = dispatch_intent.structured_intent

        decision = self._gov.evaluate_execution(
            capability_name=dispatch_intent.capability_name,
            arguments=call_args,
            context=eval_ctx,
        )

        if not decision.allowed:
            logger.warning(
                "Governance blocked linguistic command '%s %s': %s",
                dispatch_intent.action,
                dispatch_intent.target,
                decision.violations,
            )
            return BridgeExecutionResult(
                is_allowed=False,
                governance_decision=decision,
                dispatch_intent=dispatch_intent,
                error_message="; ".join(decision.violations),
            )

        exec_res: dict[str, Any] | None = None
        if self._resolver is not None:
            exec_res = await self._resolver.resolve_and_execute(
                capability_name=dispatch_intent.capability_name,
                params=call_args,
            )

        return BridgeExecutionResult(
            is_allowed=True,
            governance_decision=decision,
            dispatch_intent=dispatch_intent,
            execution_result=exec_res,
        )
