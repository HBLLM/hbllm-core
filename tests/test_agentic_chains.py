"""
Tests for Multi-Step Agentic Tool Chains (Step 7).

Validates that:
1. ThoughtNode correctly inherits trajectory_history from parent nodes.
2. ThoughtGraph.branch propagates trajectory and observation metadata.
3. PlannerNode._score_thought intercepts <tool_call> XML tags and routes
   them via ToolRouterNode, expanding the MCTS tree with Observation nodes.
4. ToolRouterNode correctly multiplexes tool calls to the right bus topic.
"""

import json

import pytest

from hbllm.actions.tool_router import ToolRouterNode
from hbllm.brain.planner_node import PlannerNode, ThoughtGraph, ThoughtNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ─── Unit Tests: ThoughtNode trajectory_history ──────────────────────────────

def test_thought_node_has_trajectory_history():
    """ThoughtNode should initialize with an empty trajectory_history."""
    node = ThoughtNode()
    assert node.trajectory_history == []


def test_branch_inherits_trajectory():
    """Children should inherit the parent's trajectory plus the parent's content."""
    g = ThoughtGraph()
    root = g.add_root("Query: What is 2+2?")
    child = g.branch(root.id, "Let me think about this...")

    assert child.trajectory_history == ["Query: What is 2+2?"]


def test_branch_multi_depth_trajectory():
    """Trajectory should accumulate across multiple depths."""
    g = ThoughtGraph()
    root = g.add_root("Root thought")
    d1 = g.branch(root.id, "Depth 1 thought")
    d2 = g.branch(d1.id, "Depth 2 thought")
    d3 = g.branch(d2.id, "Depth 3 thought")

    assert d3.trajectory_history == [
        "Root thought",
        "Depth 1 thought",
        "Depth 2 thought",
    ]


def test_branch_observation_flag():
    """Branching with is_observation=True should set metadata flag."""
    g = ThoughtGraph()
    root = g.add_root("Root")
    obs = g.branch(root.id, "Observation: result=42", is_observation=True)

    assert obs.metadata.get("is_observation") is True


def test_observation_node_does_not_propagate_flag():
    """Children of observation nodes should not automatically be observations."""
    g = ThoughtGraph()
    root = g.add_root("Root")
    obs = g.branch(root.id, "Observation: x=1", is_observation=True)
    child = g.branch(obs.id, "Next reasoning step")

    assert child.metadata.get("is_observation") is None


# ─── Unit Tests: ToolRouterNode ──────────────────────────────────────────────

@pytest.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_tool_router_routes_execute_python(bus):
    """ToolRouterNode should route execute_python to action.execute_code."""
    router = ToolRouterNode(node_id="router_1")
    await router.start(bus)

    execution_received = []

    async def mock_execution_handler(msg: Message) -> Message | None:
        execution_received.append(msg.payload)
        return msg.create_response({"status": "SUCCESS", "output": "42"})

    await bus.subscribe("action.execute_code", mock_execution_handler)

    try:
        req = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="action.tool_call",
            payload={
                "tool_name": "execute_python",
                "arguments": json.dumps({"code": "print(42)"}),
            },
        )

        resp = await bus.request("action.tool_call", req, timeout=5.0)

        assert len(execution_received) == 1
        assert execution_received[0]["code"] == "print(42)"
        assert resp.payload.get("status") == "SUCCESS"
    finally:
        await router.stop()


@pytest.mark.asyncio
async def test_tool_router_routes_generic_tool(bus):
    """ToolRouterNode should route unknown tools to action.tool.<name>."""
    router = ToolRouterNode(node_id="router_2")
    await router.start(bus)

    generic_received = []

    async def mock_generic_handler(msg: Message) -> Message | None:
        generic_received.append(msg.payload)
        return msg.create_response({"result": "secret_code_42"})

    await bus.subscribe("action.tool.find_secret", mock_generic_handler)

    try:
        req = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="action.tool_call",
            payload={
                "tool_name": "find_secret",
                "arguments": json.dumps({"query": "fibonacci"}),
            },
        )

        resp = await bus.request("action.tool_call", req, timeout=5.0)

        assert len(generic_received) == 1
        assert generic_received[0]["tool_name"] == "find_secret"
        assert resp.payload.get("result") == "secret_code_42"
    finally:
        await router.stop()


@pytest.mark.asyncio
async def test_tool_router_handles_invalid_json(bus):
    """ToolRouterNode should gracefully handle invalid JSON arguments."""
    router = ToolRouterNode(node_id="router_3")
    await router.start(bus)

    try:
        req = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="action.tool_call",
            payload={
                "tool_name": "some_tool",
                "arguments": "not valid json {{{",
            },
        )

        resp = await bus.request("action.tool_call", req, timeout=5.0)
        assert "error" in resp.payload or resp.payload.get("status") == "ERROR"
    finally:
        await router.stop()


# ─── Integration Tests: PlannerNode tool call interception ───────────────────

@pytest.mark.asyncio
async def test_score_thought_intercepts_tool_call(bus):
    """_score_thought should detect <tool_call> XML, route via bus, and create observation child."""
    router = ToolRouterNode(node_id="router_int")
    planner = PlannerNode(node_id="planner_int", branch_factor=2, max_depth=4)

    await router.start(bus)
    await planner.start(bus)

    # Mock the tool endpoint
    async def mock_tool_handler(msg: Message) -> Message | None:
        return msg.create_response({"result": "The answer is 42"})

    await bus.subscribe("action.tool.lookup", mock_tool_handler)

    try:
        graph = ThoughtGraph()
        root = graph.add_root("Query: what is the answer?")
        tool_node = graph.branch(
            root.id,
            'I need to look this up. <tool_call name="lookup">{"query": "answer"}</tool_call>'
        )

        # Before scoring, tool_node should be a leaf
        assert tool_node.is_leaf

        await planner._score_thought(graph, tool_node)

        # After scoring, tool_node should have an observation child
        assert not tool_node.is_leaf, "Tool call node should have spawned an observation child"
        assert tool_node.score == 1.0, "Emitting a valid tool call should be rewarded"

        obs_child = graph.nodes[tool_node.children_ids[0]]
        assert obs_child.metadata.get("is_observation") is True
        assert "Observation" in obs_child.content
    finally:
        await planner.stop()
        await router.stop()


@pytest.mark.asyncio
async def test_score_thought_observation_nodes_score_1(bus):
    """Observation nodes should always be scored 1.0 without hitting the bus."""
    planner = PlannerNode(node_id="planner_obs", branch_factor=2, max_depth=4)
    await planner.start(bus)

    try:
        graph = ThoughtGraph()
        root = graph.add_root("Query")
        obs = graph.branch(root.id, "Observation: result=42", is_observation=True)

        await planner._score_thought(graph, obs)
        assert obs.score == 1.0
    finally:
        await planner.stop()


@pytest.mark.asyncio
async def test_trajectory_in_refine_prompt(bus):
    """_refine_thought should include trajectory history in the prompt sent to the LLM."""
    planner = PlannerNode(node_id="planner_traj", branch_factor=2, max_depth=4)
    await planner.start(bus)

    captured_prompts = []

    async def mock_llm_handler(msg: Message) -> Message | None:
        captured_prompts.append(msg.payload.get("text", ""))
        return msg.create_response({"text": "Final answer: 42"})

    await bus.subscribe("domain.general.query", mock_llm_handler)

    try:
        graph = ThoughtGraph()
        root = graph.add_root("What is 6*7?")
        step1 = graph.branch(root.id, "Let me compute 6*7")
        obs = graph.branch(step1.id, "Observation: calculator says 42", is_observation=True)

        await planner._refine_thought(graph, obs, "What is 6*7?", 0)

        assert len(captured_prompts) == 1
        prompt = captured_prompts[0]
        # The trajectory should contain the root, step1, and observation
        assert "What is 6*7?" in prompt
        assert "Let me compute 6*7" in prompt
        assert "Observation: calculator says 42" in prompt
        assert "tool_call" in prompt  # Instructions to emit tool calls
    finally:
        await planner.stop()
