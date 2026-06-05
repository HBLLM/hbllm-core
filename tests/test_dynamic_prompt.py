import asyncio
from pathlib import Path

import pytest

from hbllm.actions.browser_node import BrowserNode
from hbllm.actions.execution_node import ExecutionNode
from hbllm.actions.logic_node import LogicNode
from hbllm.brain.identity_node import IdentityNode, IdentityProfile
from hbllm.brain.prompt_helper import get_dynamic_system_prompt
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import HealthStatus, NodeHealth
from hbllm.network.registry import ServiceRegistry


@pytest.mark.asyncio
async def test_dynamic_prompt_synthesis(tmp_path):
    bus = InProcessBus()
    await bus.start()

    # 1. Start ServiceRegistry
    registry = ServiceRegistry()
    await registry.start(bus)

    # Helper function to register nodes manually for discovery since in-process registry relies on lifecycle listener
    async def register_node(node):
        await node.start(bus)
        await registry.register(node.get_info())
        await registry.update_health(NodeHealth(node_id=node.node_id, status=HealthStatus.HEALTHY))

    # 2. Start IdentityNode
    identity_db = tmp_path / "identity_test.db"
    id_node = IdentityNode(node_id="identity_test", db_path=identity_db)
    await register_node(id_node)

    # 3. Create a tenant identity profile
    profile = IdentityProfile(
        tenant_id="tenant_x",
        persona_name="CustomSentra",
        system_prompt="You are CustomSentra, a highly specialized agent.",
        goals=["Help users write clean code", "Be extremely concise"],
        constraints=["Do not output markdown code blocks"],
    )

    # Store it via update message
    update_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test_runner",
        tenant_id="tenant_x",
        topic="identity.update",
        payload=profile.to_dict(),
    )
    await bus.request("identity.update", update_msg, timeout=2.0)

    # 4. Generate system prompt with no capability nodes active
    prompt_initial = await get_dynamic_system_prompt(bus, "tenant_x", "test_runner")
    assert "CustomSentra" in prompt_initial
    assert "Help users write clean code" in prompt_initial
    assert "Do not output markdown code blocks" in prompt_initial
    assert "BrowserNode" not in prompt_initial
    assert "ExecutionNode" not in prompt_initial

    # 5. Start capability nodes
    browser = BrowserNode(node_id="browser_test")
    await register_node(browser)

    exec_node = ExecutionNode(node_id="exec_test")
    await register_node(exec_node)

    # 6. Generate prompt with active capabilities
    prompt_final = await get_dynamic_system_prompt(bus, "tenant_x", "test_runner")
    assert "BrowserNode" in prompt_final
    assert "ExecutionNode" in prompt_final
    assert "LogicNode" not in prompt_final

    # Clean stop
    await browser.stop()
    await exec_node.stop()
    await id_node.stop()
    await registry.stop()
    await bus.stop()
