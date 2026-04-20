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
        "general",
    )
    assert skill.version == 1

    # Version
    new_skill = skill_registry.version_skill(skill.skill_id, ["open_file", "secure_read_data"])
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
            "error_message": "Timeout on step2",
        },
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
        "Do complex task", [{"action": "a"}, {"action": "b"}], [], True, "general"
    )

    # Needs a mock execute handler for action.execute_code to succeed
    async def mock_execute(msg: Message) -> Message:
        return msg.create_response({"status": "SUCCESS", "output": "ok"})

    await bus.subscribe("action.execute_code", mock_execute)

    # Needs a mock simulation handler for marginal confidence skills
    async def mock_simulate(msg: Message) -> Message:
        return msg.create_response({"status": "SUCCESS", "prediction": "SUCCESS"})

    await bus.subscribe("workspace.simulate", mock_simulate)

    node = SkillIntelligenceNode("sil", skill_registry=skill_registry)
    await node.start(bus)

    req = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.sil_execute",
        payload={"task": "Do complex task"},
    )

    resp = await bus.request("action.sil_execute", req)

    assert resp is not None
    assert resp.type == MessageType.RESPONSE
    assert resp.payload["status"] == "SUCCESS"
    assert resp.payload["skill"] == skill.name

    await node.stop()
    await bus.stop()


# ─── Hierarchical Skill Execution ─────────────────────────────────


@pytest.mark.asyncio
async def test_hierarchical_skill_execution(skill_registry):
    """Test that SIL handles recursive sil_execute steps inside a skill."""
    bus = InProcessBus()
    await bus.start()

    # Create a child skill that the parent will call
    child_skill = skill_registry.extract_and_store(
        "child task helper",
        [{"action": "child_step_1"}],
        [],
        True,
        "general",
    )
    assert child_skill is not None
    # Boost confidence so it's viable
    for _ in range(5):
        skill_registry.record_execution(child_skill.skill_id, True, 50.0)

    # Create a parent skill whose step is a JSON sil_execute directive
    import json as _json

    sil_step = _json.dumps({"action": "sil_execute", "task": "child task helper"})
    parent_skill = skill_registry.extract_and_store(
        "parent orchestrator",
        [{"action": sil_step}],
        [],
        True,
        "general",
    )
    assert parent_skill is not None
    for _ in range(5):
        skill_registry.record_execution(parent_skill.skill_id, True, 100.0)

    # Mock execution handler for actual code steps
    async def mock_execute(msg: Message) -> Message:
        return msg.create_response({"status": "SUCCESS", "output": "ok"})

    await bus.subscribe("action.execute_code", mock_execute)

    # Mock simulation handler
    async def mock_simulate(msg: Message) -> Message:
        return msg.create_response({"status": "SUCCESS", "prediction": "SUCCESS"})

    await bus.subscribe("workspace.simulate", mock_simulate)

    node = SkillIntelligenceNode("sil", skill_registry=skill_registry)
    await node.start(bus)

    req = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.sil_execute",
        payload={"task": "parent orchestrator"},
    )

    resp = await bus.request("action.sil_execute", req)
    assert resp is not None
    assert resp.type == MessageType.RESPONSE
    assert resp.payload["status"] == "SUCCESS"

    await node.stop()
    await bus.stop()


# ─── Dry-Run Simulation Gate ──────────────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_simulation_blocks_on_failure(skill_registry):
    """Test that marginal-confidence skills are blocked when simulation fails."""
    bus = InProcessBus()
    await bus.start()

    # Create a skill with marginal confidence (0.8 — in the 0.7-0.85 range)
    skill = skill_registry.extract_and_store(
        "risky operation",
        [{"action": "dangerous_step"}],
        [],
        True,
        "general",
    )
    assert skill is not None
    # Default confidence is 0.8, which is in the marginal range

    # Mock simulate to return FAILURE
    async def mock_simulate_fail(msg: Message) -> Message:
        return msg.create_response(
            {
                "status": "FAILURE",
                "prediction": "FAILURE",
                "content": "Detected unsafe import in step",
            }
        )

    await bus.subscribe("workspace.simulate", mock_simulate_fail)

    # Mock execute (should NOT be called)
    call_count = {"execute": 0}

    async def mock_execute(msg: Message) -> Message:
        call_count["execute"] += 1
        return msg.create_response({"status": "SUCCESS", "output": "ok"})

    await bus.subscribe("action.execute_code", mock_execute)

    node = SkillIntelligenceNode("sil", skill_registry=skill_registry)
    await node.start(bus)

    req = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.sil_execute",
        payload={"task": "risky operation"},
    )

    resp = await bus.request("action.sil_execute", req)
    assert resp is not None
    assert resp.type == MessageType.ERROR
    assert "Dry-run simulation failed" in str(resp.payload.get("error", ""))
    # Execution handler should never have been called
    assert call_count["execute"] == 0

    await node.stop()
    await bus.stop()


# ─── DERIVED Optimization ────────────────────────────────────────


@pytest.mark.asyncio
async def test_derived_skill_optimization(skill_registry):
    """Test offline skill optimization via SkillCompilerNode."""
    from hbllm.brain.skill_compiler_node import SkillCompilerNode

    bus = InProcessBus()
    await bus.start()

    # Create a "clunky" skill: high confidence but high cost/latency
    skill = skill_registry.extract_and_store(
        "slow bloated task",
        [{"action": "redundant_step_1"}, {"action": "redundant_step_2"}, {"action": "actual_work"}],
        ["tool_a"],
        True,
        "general",
    )
    assert skill is not None
    # Boost confidence above 0.8 threshold and make it clunky
    for _ in range(10):
        skill_registry.record_execution(skill.skill_id, True, 3000.0, tokens=6000)

    # Verify it's now clunky
    updated = skill_registry.get_skill(skill.skill_id)
    assert updated is not None
    assert updated.confidence_score >= 0.8

    class MockOptimizeLLM:
        """Mock LLM that returns optimized steps."""

        async def generate(self, prompt: str, **kwargs: Any) -> str:
            return '["optimized_step_1", "actual_work"]'

        async def generate_json(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
            return {"steps": ["optimized_step_1", "actual_work"]}

    compiler = SkillCompilerNode(
        node_id="compiler",
        skill_registry=skill_registry,
        llm=MockOptimizeLLM(),
    )
    await compiler.start(bus)

    # Trigger skill optimization
    opt_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="system.sleep.skill_optimize",
        payload={},
    )
    await bus.publish("system.sleep.skill_optimize", opt_msg)

    # Give async handler time to process
    import asyncio

    await asyncio.sleep(0.2)

    # Check that a new version was created
    all_skills = skill_registry.list_skills(limit=50)
    versioned = [s for s in all_skills if s.version > 1 and s.name == updated.name]
    assert len(versioned) >= 1, "Expected a new versioned skill from optimization"
    assert versioned[0].steps == ["optimized_step_1", "actual_work"]

    await compiler.stop()
    await bus.stop()


# ─── P2P Sync Skills ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_collective_sync_skills(skill_registry):
    """Test that CollectiveNode promotes high-confidence skills to global."""
    from hbllm.brain.collective_node import CollectiveNode

    bus = InProcessBus()
    await bus.start()

    # Create a skill for tenant "org_abc" with very high confidence
    skill = skill_registry.extract_and_store(
        "tenant specific excellence",
        [{"action": "excellent_step"}],
        ["tool_x"],
        True,
        "general",
        tenant_id="org_abc",
    )
    assert skill is not None
    assert skill.tenant_id == "org_abc"
    # Boost confidence above 0.95
    for _ in range(10):
        skill_registry.record_execution(skill.skill_id, True, 50.0)

    # Verify confidence is high
    updated = skill_registry.get_skill(skill.skill_id)
    assert updated is not None
    assert updated.confidence_score > 0.95

    # Create collective node with skill registry
    collective = CollectiveNode(
        node_id="collective",
        instance_id="test_instance",
        skill_registry=skill_registry,
    )
    await collective.start(bus)

    # Track broadcasts
    broadcasts: list[Message] = []

    async def track_broadcast(msg: Message) -> None:
        broadcasts.append(msg)

    await bus.subscribe("collective.broadcast", track_broadcast)

    # Trigger sync
    sync_msg = Message(
        type=MessageType.EVENT,
        source_node_id="test",
        topic="collective.sync_skills",
        payload={"tenant_id": "org_abc"},
    )
    await bus.publish("collective.sync_skills", sync_msg)

    # Wait for the async handler to process
    import asyncio

    await asyncio.sleep(0.1)

    # Verify the skill was promoted to global
    promoted = skill_registry.get_skill(skill.skill_id)
    assert promoted is not None
    assert promoted.tenant_id == "global"

    await collective.stop()
    await bus.stop()


# ─── Tenant Isolation in SIL ─────────────────────────────────────


@pytest.mark.asyncio
async def test_tenant_isolation_in_sil(skill_registry):
    """Test that SIL only selects skills from the matching tenant or global."""
    bus = InProcessBus()
    await bus.start()

    # Create a skill for tenant "org_a"
    skill_a = skill_registry.extract_and_store(
        "tenant a task",
        [{"action": "step_a"}],
        [],
        True,
        "general",
        tenant_id="org_a",
    )
    assert skill_a is not None
    for _ in range(5):
        skill_registry.record_execution(skill_a.skill_id, True, 50.0)

    # Create a skill for tenant "org_b"
    skill_b = skill_registry.extract_and_store(
        "tenant b task",
        [{"action": "step_b"}],
        [],
        True,
        "general",
        tenant_id="org_b",
    )
    assert skill_b is not None
    for _ in range(5):
        skill_registry.record_execution(skill_b.skill_id, True, 50.0)

    # Mock execution
    async def mock_execute(msg: Message) -> Message:
        return msg.create_response({"status": "SUCCESS", "output": "ok"})

    await bus.subscribe("action.execute_code", mock_execute)

    async def mock_simulate(msg: Message) -> Message:
        return msg.create_response({"prediction": "SUCCESS"})

    await bus.subscribe("workspace.simulate", mock_simulate)

    node = SkillIntelligenceNode("sil", skill_registry=skill_registry)
    await node.start(bus)

    # Request as tenant "org_a" — should only match org_a (or global) skills
    req = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.sil_execute",
        payload={"task": "tenant a task"},
        tenant_id="org_a",
    )

    resp = await bus.request("action.sil_execute", req)
    assert resp is not None
    if resp.type == MessageType.RESPONSE and resp.payload.get("status") == "SUCCESS":
        # If a skill matched, it should be from org_a, not org_b
        assert resp.payload["skill"] == skill_a.name

    await node.stop()
    await bus.stop()


# ─── Promote Skill ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_promote_skill(skill_registry):
    """Test that promote_skill changes tenant_id to global."""
    skill = skill_registry.extract_and_store(
        "local only skill",
        [{"action": "local_step"}],
        [],
        True,
        "general",
        tenant_id="org_private",
    )
    assert skill is not None
    assert skill.tenant_id == "org_private"

    # Promote
    skill_registry.promote_skill(skill.skill_id)

    # Verify
    promoted = skill_registry.get_skill(skill.skill_id)
    assert promoted is not None
    assert promoted.tenant_id == "global"
