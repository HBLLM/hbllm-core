import asyncio

import pytest

from hbllm.brain.evaluation.critic_node import CriticNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


class DummyLLM:
    async def generate_json(self, prompt: str) -> dict:
        return {"violations": [], "rationale": "All clear"}


@pytest.mark.asyncio
async def test_critic_web_search_bypass():
    bus = InProcessBus()
    await bus.start()

    critic = CriticNode("critic_test", llm=DummyLLM())
    await critic.start(bus)

    # 1. Mock the cached query payload (with intent='web_search')
    evaluate_msg = Message(
        type=MessageType.EVENT,
        source_node_id="workspace",
        topic="module.evaluate",
        payload={"text": "what is the capital of France?", "intent": "web_search"},
        correlation_id="corr_search_123",
    )
    await bus.publish("module.evaluate", evaluate_msg)
    await asyncio.sleep(0.1)  # Allow message to be processed and cached

    # Verify query cached successfully
    assert "corr_search_123" in critic._query_cache
    assert critic._query_cache["corr_search_123"]["intent"] == "web_search"

    # 2. Subscribe to workspace.thought to intercept the CriticNode's PASS critique
    critiques = []

    async def on_thought(msg: Message):
        if msg.payload.get("type") == "critique":
            critiques.append(msg)

    await bus.subscribe("workspace.thought", on_thought)

    # 3. Propose a thought to trigger evaluation
    proposal_msg = Message(
        type=MessageType.EVENT,
        source_node_id="domain_general",
        topic="workspace.thought",
        payload={
            "type": "intuition_general",
            "confidence": 0.8,
            "content": "I am searching the web using BrowserNode for the capital of France...",
        },
        correlation_id="corr_search_123",
    )
    await bus.publish("workspace.thought", proposal_msg)
    await asyncio.sleep(0.1)

    # 4. Assert that the critique was bypassed and PASSED
    assert len(critiques) == 1
    critique = critiques[0]
    assert critique.payload["status"] == "PASS"
    assert critique.payload["confidence"] == 0.95
    assert "web search intent bypass" in critique.payload["reason"]

    await critic.stop()
    await bus.stop()
