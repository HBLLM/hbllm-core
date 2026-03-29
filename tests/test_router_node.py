import asyncio
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock

# Create a deterministic mock encoder before importing RouterNode
class MockSentenceTransformer:
    def __init__(self, model_name: str):
        pass

    def encode(self, text: str | list[str]) -> np.ndarray:
        if isinstance(text, list):
            return np.array([self.encode(t) for t in text])
            
        # Create a simple deterministic 5-dim embedding
        # coding -> [1, 0, 0, 0, 0]
        # math   -> [0, 1, 0, 0, 0]
        # general-> [0, 0, 1, 0, 0]
        emb = np.zeros(5, dtype=np.float32)
        text_lower = text.lower()
        if "code" in text_lower or "script" in text_lower or "coding" in text_lower:
            emb[0] += 1.0
        if "math" in text_lower or "calculate" in text_lower or "integral" in text_lower:
            emb[1] += 1.0
        if "general" in text_lower or "hello" in text_lower:
            emb[2] += 1.0
        if "planner" in text_lower or "design" in text_lower:
            emb[3] += 1.0
        
        # fallback if all are zero
        if np.sum(emb) == 0:
            emb[4] += 1.0
            
        emb[0] += len(text) * 0.001
            
        # Add a tiny bit of noise so we don't get divide by zero
        emb += 1e-5
        
        return emb / np.linalg.norm(emb)

from hbllm.brain.router_node import RouterNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from unittest.mock import patch

@pytest.mark.asyncio
async def test_self_learning_vector_routing():
    """
    Test Phase 11/12 Expansion: Self-Learning Vector Router
    Simulates sending queries, extracting the routed domain, 
    submitting negative feedback, and verifying the centroid shifts.
    """
    bus = InProcessBus()
    await bus.start()
    
    # Initialize the router node 
    router = RouterNode(node_id="test_router")
    
    # Inject the mock encoder explicitly
    router.encoder = MockSentenceTransformer("mock")
    router.use_vectors = True
    router._bootstrap_centroids()
    
    await router.start(bus)
    
    assert router.use_vectors is True, "Router failed to initialize vector routing."
    
    # Listen for workspace.update to intercept decisions
    decisions = []
    async def _capture_decision(msg: Message):
        decisions.append(msg)
    await bus.subscribe("workspace.update", _capture_decision)
    
    try:
        # 1. Send an ambiguous query that leans slightly "math" based on our mock
        query_msg = Message(
            type=MessageType.QUERY,
            source_node_id="user_api",
            topic="router.query",
            payload={"text": "Hello, please calculate something random for me."},
            correlation_id="routing_test_1"
        )
        await bus.publish("router.query", query_msg)
        
        # Wait for router to process
        await asyncio.sleep(0.1)
        
        assert len(decisions) == 1, f"Expected 1 decision, got {len(decisions)}"
        assert decisions[0].payload["domain_hint"] == "math"
        
        # 2. Issue a Thumbs Down (-1) feedback to push "math" centroid away!
        feedback_msg = Message(
            type=MessageType.EVENT,
            source_node_id="user_api",
            topic="system.feedback",
            payload={
                "message_id": "routing_test_1",
                "rating": -1,
                "domain": "math",
                "prompt": "Hello, please calculate something random for me."
            }
        )
        await bus.publish("system.feedback", feedback_msg)
        await asyncio.sleep(0.1)
        
        query_msg2 = Message(
            type=MessageType.QUERY,
            source_node_id="user_api",
            topic="router.query",
            payload={"text": "Hello, please calculate something random for me."},
            correlation_id="routing_test_2"
        )
        await bus.publish("router.query", query_msg2)
        await asyncio.sleep(0.1)
        
        # 3. Check where it routed.
        assert len(decisions) == 2, f"Expected 2 decisions, got {len(decisions)}"
        assert decisions[1].payload["domain_hint"] == "general"
        
    finally:
        await router.stop()
        await bus.stop()
