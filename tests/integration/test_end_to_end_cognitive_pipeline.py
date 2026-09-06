"""End-to-End Cognitive Pipeline Integration Test.

Validates the full non-LLM cognitive architecture composing:
1. Multilingual Language Comprehension (English, Sinhala, Tamil)
2. LanguageCapabilityBridge (Translates SemanticFrames to StructuredIntent)
3. CognitiveGraph & ContradictionEngine (Proactive epistemic contradiction hunting)
4. GovernanceEngine (Kernel Gate enforcing fail-closed life safety & perimeter policies)
5. CapabilityResolver & Sandbox (Resource-budgeted sandboxed capability execution)
"""

from __future__ import annotations

from typing import Any

import pytest

from hbllm.brain.epistemics.contradiction_engine import ContradictionEngine
from hbllm.brain.language.core.bridge import LanguageCapabilityBridge
from hbllm.brain.language.english.parser import EnglishParser
from hbllm.brain.language.sinhala.parser import SinhalaParser
from hbllm.brain.language.tamil.parser import TamilParser
from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    ContradictionNode,
    HCIREdge,
    HCIREdgeType,
)
from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
    ICapabilityExecutor,
)
from hbllm.hcir.kernel.capability_sandboxing import (
    CapabilityPermissions,
    CapabilitySandboxManager,
    SandboxedCapabilityPolicy,
    TrustLevel,
)
from hbllm.hcir.kernel.governance.governance_engine import (
    GovernanceEngine,
)

# ═══════════════════════════════════════════════════════════════════════════
# Mock Hardware Capability Executors
# ═══════════════════════════════════════════════════════════════════════════


class MockPerimeterExecutor(ICapabilityExecutor):
    """Mock smart lock / perimeter barrier actuator."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.last_params: dict[str, Any] = {}

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.last_params = params
        return {
            "status": "unlocked",
            "barrier_id": params.get("target", "front_door"),
            "executed": True,
        }

    @property
    def is_available(self) -> bool:
        return True


class MockActuatorExecutor(ICapabilityExecutor):
    """Mock robotic manipulator / robotic arm actuator."""

    def __init__(self) -> None:
        self.call_count: int = 0
        self.last_params: dict[str, Any] = {}

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.last_params = params
        return {
            "status": "actuated",
            "actuator": params.get("target", "arm"),
            "action": params.get("action", "rotate"),
            "executed": True,
        }

    @property
    def is_available(self) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Integration Test Fixtures & Helpers
# ═══════════════════════════════════════════════════════════════════════════


def setup_governed_sandbox_environment() -> tuple[
    GovernanceEngine, CapabilityResolver, MockPerimeterExecutor, MockActuatorExecutor
]:
    """Assemble a configured GovernanceEngine and CapabilityResolver with registered policies."""
    gov = GovernanceEngine()
    sandbox_mgr = CapabilitySandboxManager()

    # Register sandboxed capability policies
    sandbox_mgr.register_policy(
        SandboxedCapabilityPolicy(
            capability_name="perimeter_control",
            provider_id="door_hardware_v1",
            trust_level=TrustLevel.VERIFIED,
            permissions=CapabilityPermissions(
                allow_filesystem=False,
                allow_network=False,
                allow_subprocess=False,
            ),
        )
    )
    sandbox_mgr.register_policy(
        SandboxedCapabilityPolicy(
            capability_name="actuator_control",
            provider_id="robotic_arm_v1",
            trust_level=TrustLevel.VERIFIED,
            permissions=CapabilityPermissions(
                allow_filesystem=False,
                allow_network=False,
                allow_subprocess=False,
            ),
        )
    )

    resolver = CapabilityResolver(sandbox_manager=sandbox_mgr)
    door_exec = MockPerimeterExecutor()
    arm_exec = MockActuatorExecutor()

    resolver.register(
        CapabilityImplementation(
            capability_name="perimeter_control",
            implementation_id="door_hardware_v1",
            executor=door_exec,
            priority=10,
            estimated_cost=5,
        )
    )
    resolver.register(
        CapabilityImplementation(
            capability_name="actuator_control",
            implementation_id="robotic_arm_v1",
            executor=arm_exec,
            priority=10,
            estimated_cost=8,
        )
    )

    return gov, resolver, door_exec, arm_exec


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End Scenarios
# ═══════════════════════════════════════════════════════════════════════════


class TestEndToEndCognitivePipeline:
    """End-to-End test suite exercising language comprehension through to capability execution."""

    @pytest.mark.asyncio
    async def test_trilingual_authorized_perimeter_dispatch(self) -> None:
        """Scenario 1: Trilingual Intent Compilation & Authorized Perimeter Access.

        Verifies that English, Sinhala, and Tamil natural language commands:
        - Parse into typed semantic frames
        - Map to perimeter_control capability intents
        - Pass GovernanceEngine under authorized credentials
        - Execute via CapabilityResolver on concrete sandboxed hardware
        """
        gov, resolver, door_exec, _ = setup_governed_sandbox_environment()
        bridge = LanguageCapabilityBridge(governance_engine=gov, capability_resolver=resolver)

        # 1. English: "Open the door."
        en_parser = EnglishParser()
        en_frame = en_parser.parse("Open the door.")
        en_result = await bridge.execute_language_command(
            en_frame,
            context={"authorized": True},
        )
        assert en_result.is_allowed is True
        assert en_result.execution_result is not None
        assert en_result.execution_result["status"] == "unlocked"

        # 2. Sinhala: "ඉදිරිපස දොර අරින්න" (Unlock front door)
        si_parser = SinhalaParser()
        si_frame = si_parser.parse("ඉදිරිපස දොර අරින්න")
        si_result = await bridge.execute_language_command(
            si_frame,
            context={"authorized": True},
        )
        assert si_result.is_allowed is True
        assert si_result.execution_result is not None
        assert si_result.execution_result["status"] == "unlocked"

        # 3. Tamil: "முன் கதவை திறக்கவும்" (Open front door)
        ta_parser = TamilParser()
        ta_frame = ta_parser.parse("முன் கதவை திறக்கவும்")
        ta_result = await bridge.execute_language_command(
            ta_frame,
            context={"authorized": True},
        )
        assert ta_result.is_allowed is True
        assert ta_result.execution_result is not None
        assert ta_result.execution_result["status"] == "unlocked"

        # Hardware executor dispatched once for each language utterance
        assert door_exec.call_count == 3

    @pytest.mark.asyncio
    async def test_epistemic_contradiction_overrides_affirmative_command(self) -> None:
        """Scenario 2: Epistemic Contradiction Overrides Affirmative Command.

        Simulates an environment where:
        - The user commands the robotic arm to rotate, asserting 'area is clear'.
        - CognitiveGraph sensor telemetry contains an active observation: human is present.
        - ContradictionEngine detects the structural conflict.
        - Epistemic arbitration determines safety is violated.
        - GovernanceEngine fails closed on the actuator hazard.
        - Capability executor is never invoked.
        """
        gov, resolver, _, arm_exec = setup_governed_sandbox_environment()
        bridge = LanguageCapabilityBridge(governance_engine=gov, capability_resolver=resolver)

        # 1. Set up CognitiveGraph with sensory telemetry
        graph = CognitiveGraph()
        sensor_belief = BeliefNode(
            id="sensor_human_detected",
            claim="human presence detected in actuator workspace",
        )
        user_claim_belief = BeliefNode(
            id="user_claim_clear",
            claim="workspace is clear of humans",
        )
        graph.upsert_node(sensor_belief)
        graph.upsert_node(user_claim_belief)

        # Link as direct contradiction in cognitive graph
        graph.add_edge(
            HCIREdge(
                sources=[sensor_belief.id],
                targets=[user_claim_belief.id],
                edge_type=HCIREdgeType.CONTRADICTS,
            )
        )

        # 2. Run ContradictionEngine scanner (non-LLM)
        engine = ContradictionEngine(graph=graph)
        reports = await engine.scan_for_contradictions()
        assert len(reports) >= 1
        assert any(
            r.claim_a_id == sensor_belief.id or r.claim_b_id == sensor_belief.id for r in reports
        )

        # Verify a ContradictionNode exists in the graph
        contra_nodes = [n for n in graph.all_nodes() if isinstance(n, ContradictionNode)]
        assert len(contra_nodes) >= 1

        # 3. Epistemic world state arbiter:
        # If an active contradiction touches actuator workspace safety, human presence is flagged
        has_active_workspace_contradiction = any(
            "workspace" in (getattr(graph.get_node(c.claim_a_id), "claim", "")).lower()
            or "workspace" in (getattr(graph.get_node(c.claim_b_id), "claim", "")).lower()
            for c in contra_nodes
        )
        assert has_active_workspace_contradiction is True

        # Construct runtime evaluation context informed by epistemic world state
        eval_context: dict[str, Any] = {
            "workspace_cleared": False,
            "human_in_workspace": True,
            "epistemic_contradiction_active": True,
        }

        # 4. User issues affirmative command in English: "Rotate the arm."
        en_parser = EnglishParser()
        frame = en_parser.parse("Rotate the arm.")

        result = await bridge.execute_language_command(
            frame,
            context=eval_context,
        )

        # 5. Governance must fail-closed: blocked by human_in_workspace_actuator_hazard
        assert result.is_allowed is False
        assert any(
            "human_in_workspace_actuator_hazard" in v for v in result.governance_decision.violations
        )
        assert arm_exec.call_count == 0  # Physical actuator never moved

    @pytest.mark.asyncio
    async def test_unauthorized_perimeter_access_fails_closed(self) -> None:
        """Scenario 3: Unauthorized Perimeter Access Fails Closed Across Languages.

        Commands attempting to open doors/gates without authorized credentials
        must be intercepted and blocked by GovernanceEngine fail-closed rules.
        """
        gov, resolver, door_exec, _ = setup_governed_sandbox_environment()
        bridge = LanguageCapabilityBridge(governance_engine=gov, capability_resolver=resolver)

        # 1. English without authorization
        en_frame = EnglishParser().parse("Open the door.")
        res_en = await bridge.execute_language_command(en_frame, context={"authorized": False})
        assert res_en.is_allowed is False
        assert any("unauthorized_door_unlock" in v for v in res_en.governance_decision.violations)

        # 2. Sinhala without authorization
        si_frame = SinhalaParser().parse("ඉදිරිපස දොර අරින්න")
        res_si = await bridge.execute_language_command(si_frame, context={"authorized": False})
        assert res_si.is_allowed is False
        assert any("unauthorized_door_unlock" in v for v in res_si.governance_decision.violations)

        # 3. Tamil without authorization
        ta_frame = TamilParser().parse("முன் கதவை திறக்கவும்")
        res_ta = await bridge.execute_language_command(ta_frame, context={"authorized": False})
        assert res_ta.is_allowed is False
        assert any("unauthorized_door_unlock" in v for v in res_ta.governance_decision.violations)

        assert door_exec.call_count == 0

    @pytest.mark.asyncio
    async def test_adversarial_obfuscation_and_bypass_inversion(self) -> None:
        """Scenario 4: Adversarial Obfuscation & Unknown Action Inversion.

        Novel verbs, override tokens, or attempted bypasses targeting perimeter
        or life-safety components must fail closed under GovernanceEngine.
        """
        gov, resolver, door_exec, arm_exec = setup_governed_sandbox_environment()

        # 1. Malicious bypass tokens on perimeter barrier
        res_bypass = gov.evaluate_execution(
            capability_name="perimeter_control",
            arguments={
                "target": "front_door",
                "action": "sudo_override_bypass",
                "authorized": False,
            },
        )
        assert res_bypass.allowed is False
        assert any("unauthorized_door_unlock" in v for v in res_bypass.violations)

        # 2. Critical life safety tamper (smoke alarm disable)
        res_life_safety = gov.evaluate_execution(
            capability_name="emergency_system",
            arguments={"target": "smoke_detector", "action": "disable"},
        )
        assert res_life_safety.allowed is False
        assert any(
            "life_safety_critical_device_protection" in v for v in res_life_safety.violations
        )

        # 3. Excessive actuator torque command (>500N)
        res_torque = gov.evaluate_execution(
            capability_name="actuator_control",
            arguments={
                "target": "robot_arm",
                "action": "rotate",
                "torque": 1200.0,
                "workspace_cleared": True,
            },
        )
        assert res_torque.allowed is False
        assert any("actuator_force_torque_limit" in v for v in res_torque.violations)

        assert door_exec.call_count == 0
        assert arm_exec.call_count == 0

    @pytest.mark.asyncio
    async def test_budget_bounded_sandboxed_capability_dispatch(self) -> None:
        """Scenario 5: Budget-Bound Sandboxed Execution.

        Verifies that CapabilityResolver properly enforces resource limits:
        - Executes within budget
        - Rejects requests that exceed the specified maximum cost
        """
        _, resolver, door_exec, _ = setup_governed_sandbox_environment()

        # 1. Budget of 10 > estimated cost of 5: Success
        res_ok = await resolver.resolve_and_execute(
            capability_name="perimeter_control",
            params={"target": "front_door"},
            budget=10,
        )
        assert "error" not in res_ok
        assert res_ok["status"] == "unlocked"
        assert door_exec.call_count == 1
        assert resolver.total_cost == 5

        # 2. Budget of 2 < estimated cost of 5: Rejection
        res_exceeded = await resolver.resolve_and_execute(
            capability_name="perimeter_control",
            params={"target": "front_door"},
            budget=2,
        )
        assert "error" in res_exceeded
        assert "within budget 2" in res_exceeded["error"]
        assert door_exec.call_count == 1  # No additional execution
        assert resolver.total_cost == 5  # Cost unchanged
