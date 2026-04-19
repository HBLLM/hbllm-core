"""Tests for Skill Lifecycle System and Skill Intelligence Layer."""

from typing import Any

import pytest

from hbllm.brain.failure_analyzer_node import FailureAnalyzerNode
from hbllm.brain.skill_intelligence_node import SkillIntelligenceNode
from hbllm.brain.skill_registry import SkillRegistry
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


class MiniMockLLM:
    """Mock LLM for testing failure repair without importing entire MockLLM."""
    async def generate_json(self, prompt: str) -> dict[str, Any]:
        return {"new_steps": ["step1", "repaired_step2"]}


@pytest.fixture
def skill_registry(tmp_path):
    registry = SkillRegistry(data_dir=str(tmp_path))
    return registry


@pytest.mark.asyncio
async def test_skill_registry_versioning(skill_registry):
    # Extract
    skill = skill_registry.extract_and_store(
        "Open a file and read contents",
        [{"action": "open_file"}, {"action": "read_data"}],
        ["file_tool"],
        True,
        "general"
    )
    assert skill.version == 1

    # Version
    new_skill = skill_registry.version_skill(
        skill.skill_id,
        ["open_file", "secure_read_data"]
    )
    assert new_skill is not None
    assert new_skill.version == 2
    assert new_skill.parent_skill_id == skill.skill_id
    assert new_skill.steps == ["open_file", "secure_read_data"]


@pytest.mark.asyncio
async def test_failure_analyzer_node():
    bus = InProcessBus()
    await bus.start()

    mock_llm = MiniMockLLM()

    node = FailureAnalyzerNode("failure", llm=mock_llm)  # type: ignore
    await node.start(bus)

    req = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.analyze_failure",
        payload={
            "skill_name": "BrokenSkill",
            "steps": ["step1", "broken_step2"],
            "execution_trace": [],
            "error_message": "Timeout on step2"
        }
    )

    resp = await bus.request("action.analyze_failure", req)

    assert resp is not None
    assert resp.type == MessageType.RESPONSE
    assert resp.payload["failure_type"] == "Timeout"
    assert resp.payload["repaired"] is True
    assert resp.payload["new_steps"] == ["step1", "repaired_step2"]

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_skill_intelligence_node(skill_registry):
    bus = InProcessBus()
    await bus.start()

    # Store a highly confident skill
    skill = skill_registry.extract_and_store(
        "Do complex task",
        [{"action": "a"}, {"action": "b"}],
        [],
        True,
        "general"
    )

    # Needs a mock execute handler for action.execute_code to succeed
    async def mock_execute(msg: Message) -> Message:
        return msg.create_response({"status": "SUCCESS", "output": "ok"})
    await bus.subscribe("action.execute_code", mock_execute)

    node = SkillIntelligenceNode("sil", skill_registry=skill_registry)
    await node.start(bus)

    req = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.sil_execute",
        payload={"task": "Do complex task"}
    )

    resp = await bus.request("action.sil_execute", req)

    assert resp is not None
    assert resp.type == MessageType.RESPONSE
    assert resp.payload["status"] == "SUCCESS"
    assert resp.payload["skill"] == skill.name

    await node.stop()
    await bus.stop()
