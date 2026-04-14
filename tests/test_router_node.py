import asyncio

import numpy as np
import pytest


# Create a deterministic mock encode method
def mock_encode_text(self, text: str) -> np.ndarray:
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
    router = RouterNode("test_router")

    # Inject the mock encoder explicitly
    router._encode_text = mock_encode_text.__get__(router, RouterNode)
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
            correlation_id="routing_test_1",
        )
        await bus.publish("router.query", query_msg)

        # Wait for router to process
        await asyncio.sleep(0.1)

        assert len(decisions) == 1, f"Expected 1 decision, got {len(decisions)}"

        # Test 1 baseline extraction
        first_payload = decisions[0].payload
        first_domain = first_payload["domain_hint"]

        # With MoE enabled, close domains might blend. Check if math is the top pick or the chosen string.
        if isinstance(first_domain, dict):
            top_domain = max(first_domain.items(), key=lambda x: x[1])[0]
            assert top_domain == "math", f"Expected primary MoE math, got {first_domain}"
        else:
            assert first_domain == "math", f"Expected 'math', got {first_domain}"

        first_confidence = first_payload["confidence"]

        # 2. Issue a Thumbs Down (-1) feedback to push "math" centroid away!
        feedback_msg = Message(
            type=MessageType.EVENT,
            source_node_id="user_api",
            topic="system.feedback",
            payload={
                "message_id": "routing_test_1",
                "rating": -1,
                "domain": "math",
                "prompt": "Hello, please calculate something random for me.",
            },
        )
        await bus.publish("system.feedback", feedback_msg)
        await asyncio.sleep(0.1)

        query_msg2 = Message(
            type=MessageType.QUERY,
            source_node_id="user_api",
            topic="router.query",
            payload={"text": "Hello, please calculate something random for me."},
            correlation_id="routing_test_2",
        )
        await bus.publish("router.query", query_msg2)
        await asyncio.sleep(0.1)

        # 3. Check where it routed and how confidence changed
        assert len(decisions) == 2, f"Expected 2 decisions, got {len(decisions)}"
        second_payload = decisions[1].payload
        second_confidence = second_payload["confidence"]

        # The key logic: Did negative feedback properly push the centroid away?
        # With MoE blending, the top domain may change (math → general) after
        # pushing math away, so compare math-specific centroid distance instead.
        first_domain_hint = first_payload["domain_hint"]
        second_domain_hint = second_payload["domain_hint"]

        # If domain hint is a dict (MoE blend), check math weight dropped
        if isinstance(first_domain_hint, dict) and isinstance(second_domain_hint, dict):
            first_math_weight = first_domain_hint.get("math", 0.0)
            second_math_weight = second_domain_hint.get("math", 0.0)
            assert second_math_weight < first_math_weight or second_math_weight == 0.0, (
                f"Math weight did not drop! {second_math_weight} >= {first_math_weight}"
            )
        else:
            # Either confidence dropped OR the domain switched away from math
            domain_switched = (
                isinstance(second_domain_hint, str) and second_domain_hint != "math"
            ) or (isinstance(second_domain_hint, dict) and "math" not in second_domain_hint)
            assert second_confidence < first_confidence or domain_switched, (
                f"Confidence did not drop and domain did not switch! "
                f"{second_confidence} >= {first_confidence}, domain={second_domain_hint}"
            )

    finally:
        await router.stop()
        await bus.stop()
