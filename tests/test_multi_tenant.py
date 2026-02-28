import pytest
import asyncio
import uuid

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.workspace_node import WorkspaceNode
from hbllm.brain.router_node import RouterNode
from tests.mock_llm import MockLLM

# Mock for a Domain Engine that returns quick answers
class MockDomainNode:
    def __init__(self, bus):
        self.bus = bus
        self.evals = []
        
    async def start(self):
        await self.bus.subscribe("module.evaluate", self.handle)
        
    async def handle(self, msg: Message):
        self.evals.append(msg)
        
        thought = Message(
            type=MessageType.EVENT,
            source_node_id="domain_mock",
            tenant_id=msg.tenant_id,
            session_id=msg.session_id,
            topic="workspace.thought",
            payload={
                "type": "intuition_general",
                "confidence": 0.9,
                "content": f"Answer for tenant {msg.tenant_id} session {msg.session_id}"
            },
            correlation_id=msg.correlation_id
        )
        await self.bus.publish("workspace.thought", thought)


@pytest.fixture
async def mt_env():
    bus = InProcessBus()
    await bus.start()
    
    workspace = WorkspaceNode("workspace_01")
    await workspace.start(bus)
    
    mock_llm = MockLLM()
    router = RouterNode("router_01", llm=mock_llm)
    await router.start(bus)
    
    domain = MockDomainNode(bus)
    await domain.start()
    
    yield bus, workspace, router, domain
    
    await workspace.stop()
    await router.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_concurrent_multi_tenant_workspaces(mt_env):
    bus, workspace, router, domain = mt_env
    
    req1 = Message(
        type=MessageType.QUERY,
        source_node_id="api_gateway",
        tenant_id="tenant_A",
        session_id="session_A_1",
        topic="router.query",
        payload={"text": "Hello A"}
    )
    
    req2 = Message(
        type=MessageType.QUERY,
        source_node_id="api_gateway",
        tenant_id="tenant_B",
        session_id="session_B_1",
        topic="router.query",
        payload={"text": "Hello B"}
    )
    
    await bus.publish("router.query", req1)
    await bus.publish("router.query", req2)
    
    await asyncio.sleep(0.5)
    
    assert len(domain.evals) == 2
    
    tenants_seen = set()
    for eval_msg in domain.evals:
        tenants_seen.add(eval_msg.tenant_id)
        
    assert "tenant_A" in tenants_seen
    assert "tenant_B" in tenants_seen
