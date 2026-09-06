"""Unit tests for Multilingual Cognition, Tamil Engine, and LanguageCapabilityBridge."""

from __future__ import annotations

import pytest

from hbllm.brain.language.core.bridge import LanguageCapabilityBridge
from hbllm.brain.language.core.epistemic_policy import (
    CognitiveEpistemicState,
    EpistemicRealizationPolicy,
)
from hbllm.brain.language.core.semantic_frame import FrameType, ThematicRole
from hbllm.brain.language.runtime import MultilingualLanguageRuntime
from hbllm.brain.language.sinhala.parser import SinhalaParser
from hbllm.brain.language.tamil.lexicon import TamilLexicon, TamilPOS
from hbllm.brain.language.tamil.parser import TamilParser
from hbllm.brain.language.tamil.realizer import TamilRealizer
from hbllm.hcir.graph import CognitiveGraph
from hbllm.hcir.kernel.governance.governance_engine import GovernanceEngine


class TestTamilLanguageEngine:
    """Test Tamil lexicon, parsing, and surface realization."""

    def test_tamil_lexicon_lookup(self) -> None:
        lex = TamilLexicon()
        door_entries = lex.lookup("கதவு")
        assert len(door_entries) > 0
        assert door_entries[0].pos == TamilPOS.NOUN
        assert door_entries[0].semantic_predicate == "door"

        open_entries = lex.lookup("திறக்க")
        assert len(open_entries) > 0
        assert open_entries[0].semantic_predicate == "open"

    def test_tamil_parser_wh_question(self) -> None:
        parser = TamilParser()
        frame = parser.parse("பந்து எங்கே?")
        assert frame.frame_type == FrameType.QUERY
        assert frame.query_target == "location"
        assert frame.predicate == "located_on"
        theme = frame.get_role(ThematicRole.THEME)
        assert theme is not None
        assert theme.concept_name == "ball"

    def test_tamil_parser_verification_query(self) -> None:
        parser = TamilParser()
        frame = parser.parse("பந்து மேசை மீது இருக்கிறதா?")
        assert frame.frame_type == FrameType.QUERY
        assert frame.query_target == "verification"
        assert frame.predicate == "located_on"
        theme = frame.get_role(ThematicRole.THEME)
        loc = frame.get_role(ThematicRole.LOCATION)
        assert theme is not None and theme.concept_name == "ball"
        assert loc is not None and loc.concept_name == "table"

    def test_tamil_parser_imperative_command(self) -> None:
        parser = TamilParser()
        frame = parser.parse("முன் கதவை திறக்கவும்")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "open"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "front_door"

    def test_tamil_parser_actuator_command(self) -> None:
        parser = TamilParser()
        frame = parser.parse("ரோபோ கையை சுழற்றவும்")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "rotate"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "robot_arm"

    def test_tamil_parser_assertion(self) -> None:
        parser = TamilParser()
        frame = parser.parse("சிவப்பு பந்து மேசை மீது இருக்கிறது")
        assert frame.frame_type == FrameType.ASSERTION
        assert frame.predicate == "located_on"
        theme = frame.get_role(ThematicRole.THEME)
        loc = frame.get_role(ThematicRole.LOCATION)
        assert theme is not None and theme.concept_name == "ball"
        assert theme.properties.get("color") == "red"
        assert loc is not None and loc.concept_name == "table"

    def test_tamil_realizer(self) -> None:
        realizer = TamilRealizer(EpistemicRealizationPolicy())
        state_certain = CognitiveEpistemicState(
            target_predicate="located_on",
            target_subject="ball",
            target_object="table",
            confidence=0.98,
            support_count=5,
            is_known=True,
        )
        text_certain = realizer.realize(state_certain)
        assert "பந்து" in text_certain
        assert "மேசை" in text_certain
        assert "இருக்கிறது" in text_certain

        state_contradicted = CognitiveEpistemicState(
            target_predicate="located_on",
            target_subject="ball",
            target_object="table",
            confidence=0.50,
            contradiction_count=2,
            is_known=True,
        )
        text_contra = realizer.realize(state_contradicted)
        assert "முரண்பட்ட" in text_contra or "சான்றுகள்" in text_contra

    def test_multilingual_runtime_registers_tamil(self) -> None:
        from hbllm.hcir.graph import PhysicalEntityNode

        graph = CognitiveGraph()
        graph.add_node(PhysicalEntityNode(entity_type="ball", observed_properties={"color": "red"}))
        runtime = MultilingualLanguageRuntime(graph)
        assert "ta" in runtime._parsers
        assert "ta" in runtime._realizers

        res = runtime.process_utterance("பந்து எங்கே?", language="ta")
        assert res.is_success
        assert res.semantic_frame.frame_type == FrameType.QUERY


class TestSinhalaPhysicalSecurityCommands:
    """Test extended Sinhala physical security commands."""

    def test_sinhala_front_door_open(self) -> None:
        parser = SinhalaParser()
        frame = parser.parse("ඉදිරිපස දොර අරින්න")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "open"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "front_door"

    def test_sinhala_robot_arm_rotate(self) -> None:
        parser = SinhalaParser()
        frame = parser.parse("රොබෝ අත කරකවන්න")
        assert frame.frame_type == FrameType.COMMAND
        assert frame.predicate == "rotate"
        patient = frame.get_role(ThematicRole.PATIENT)
        assert patient is not None
        assert patient.concept_name == "robot_arm"


class TestLanguageCapabilityBridge:
    """Test end-to-end translation from natural language to governed capability dispatch."""

    @pytest.mark.asyncio
    async def test_bridge_blocks_unauthorized_sinhala_command(self) -> None:
        bridge = LanguageCapabilityBridge(GovernanceEngine())
        parser = SinhalaParser()
        frame = parser.parse("ඉදිරිපස දොර අරින්න")

        result = await bridge.execute_language_command(
            frame,
            context={"authorized": False},
        )
        assert not result.is_allowed
        assert any(
            "unauthorized_door_unlock_and_perimeter_security" in v
            for v in result.governance_decision.violations
        )

    @pytest.mark.asyncio
    async def test_bridge_allows_authorized_sinhala_command(self) -> None:
        bridge = LanguageCapabilityBridge(GovernanceEngine())
        parser = SinhalaParser()
        frame = parser.parse("ඉදිරිපස දොර අරින්න")

        result = await bridge.execute_language_command(
            frame,
            context={"authorized": True},
        )
        assert result.is_allowed

    @pytest.mark.asyncio
    async def test_bridge_blocks_unauthorized_tamil_command(self) -> None:
        bridge = LanguageCapabilityBridge(GovernanceEngine())
        parser = TamilParser()
        frame = parser.parse("முன் கதவை திறக்கவும்")

        result = await bridge.execute_language_command(
            frame,
            context={"authorized": False},
        )
        assert not result.is_allowed
        assert any(
            "unauthorized_door_unlock_and_perimeter_security" in v
            for v in result.governance_decision.violations
        )

    @pytest.mark.asyncio
    async def test_bridge_actuator_clearance_governance(self) -> None:
        bridge = LanguageCapabilityBridge(GovernanceEngine())
        parser = TamilParser()
        frame = parser.parse("ரோபோ கையை சுழற்றவும்")

        # 1. Unknown clearance -> Blocked
        result_uncleared = await bridge.execute_language_command(frame, context={})
        assert not result_uncleared.is_allowed
        assert any(
            "human_in_workspace_actuator_hazard" in v
            for v in result_uncleared.governance_decision.violations
        )

        # 2. Confirmed cleared -> Allowed
        result_cleared = await bridge.execute_language_command(
            frame,
            context={"workspace_cleared": True},
        )
        assert result_cleared.is_allowed

    @pytest.mark.asyncio
    async def test_bridge_executes_with_capability_resolver(self) -> None:
        from unittest.mock import AsyncMock

        from hbllm.hcir.kernel.capability_resolver import CapabilityResolver

        resolver = AsyncMock(spec=CapabilityResolver)
        resolver.resolve_and_execute.return_value = {"status": "unlocked"}

        bridge = LanguageCapabilityBridge(GovernanceEngine(), capability_resolver=resolver)
        parser = SinhalaParser()
        frame = parser.parse("ඉදිරිපස දොර අරින්න")

        result = await bridge.execute_language_command(
            frame,
            context={"authorized": True},
        )
        assert result.is_allowed
        assert result.execution_result == {"status": "unlocked"}
        resolver.resolve_and_execute.assert_awaited_once()
        _, kwargs = resolver.resolve_and_execute.call_args
        assert kwargs["capability_name"] == "perimeter_control"
        assert kwargs["params"]["authorized"] is True
