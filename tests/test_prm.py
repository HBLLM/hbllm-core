"""
Process Reward Model Tests.
"""

import pytest
import torch

from hbllm.brain.process_reward_node import ProcessRewardNode
from hbllm.model.config import get_config
from hbllm.model.transformer import HBLLMForProcessReward
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


def test_prm_forward_pass():
    """Verify HBLLMForProcessReward outputs correctly scaled continuous scores."""
    config = get_config("125m")
    # Make config tiny for speed
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_layers = 1
    config.num_attention_heads = 4
    config.num_key_value_heads = 4

    model = HBLLMForProcessReward(config)

    # Dummy input
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([1.0])

    output = model(input_ids, labels=labels)

    assert "logits" in output
    assert "scores" in output
    assert "loss" in output

    # Score must be bounded [0.0, 1.0] due to sigmoid
    score = output["scores"].item()
    assert 0.0 <= score <= 1.0

    # Loss must be a real number
    assert torch.is_tensor(output["loss"])
    assert output["loss"].item() > 0


@pytest.mark.asyncio
async def test_process_reward_node_fallback():
    """Verify ProcessRewardNode gracefully handles untaught state."""
    bus = InProcessBus()
    await bus.start()

    node = ProcessRewardNode(node_id="prm", model_name="125m")
    await node.start(bus)

    # Query it via bus
    req = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.score_thought",
        payload={"content": "If I add 2 and 2, I get 4."},
    )

    resp = await bus.request("action.score_thought", req, timeout=2.0)

    assert resp is not None
    assert "score" in resp.payload
    score = resp.payload["score"]
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    await node.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_process_reward_node_neural():
    """Verify ProcessRewardNode uses neural head when trained."""
    node = ProcessRewardNode(node_id="prm", model_name="125m")

    # Replace the heavy model with a mock tiny model to simulate "trained" state
    config = get_config("125m")
    config.vocab_size = 500  # Small
    config.hidden_size = 32
    config.num_layers = 1
    config.num_attention_heads = 2

    node.prm_model = HBLLMForProcessReward(config)
    node.prm_is_trained = True  # Force to True for test

    # Simulate a thought
    score = await node.score_thought("This is a logical deduction.")

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
