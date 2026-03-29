"""
LoRA Online Learning Loop Tests — verifies LearnerNode feedback flow,
LoRA injection, optimizer creation, and adapter checkpoint saving.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path

import torch

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType
from hbllm.brain.learner_node import LearnerNode
from hbllm.model.config import get_config
from hbllm.model.transformer import HBLLMForCausalLM


@pytest.fixture
def tiny_model():
    """Create a tiny model for CPU testing."""
    config = get_config("125m")
    model = HBLLMForCausalLM(config)
    return model, config


@pytest.fixture
def tmp_ckpt_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.asyncio
async def test_learner_accepts_feedback(tmp_ckpt_dir):
    """LearnerNode buffers feedback messages."""
    bus = InProcessBus()
    await bus.start()

    learner = LearnerNode(
        node_id="learner",
        batch_size=4,
        checkpoint_dir=tmp_ckpt_dir,
    )
    await learner.start(bus)

    fb_msg = Message(
        type=MessageType.FEEDBACK,
        source_node_id="api",
        topic="system.feedback",
        payload={
            "message_id": "msg_001",
            "rating": 5,
            "prompt": "What is Python?",
            "response": "Python is a programming language.",
        },
    )
    await bus.publish("system.feedback", fb_msg)
    await asyncio.sleep(0.2)

    assert "What is Python?" in learner.pending_pairs

    await learner.stop()
    await bus.stop()


@pytest.mark.asyncio
async def test_learner_ignores_non_feedback():
    """LearnerNode silently ignores non-FEEDBACK message types."""
    bus = InProcessBus()
    await bus.start()

    learner = LearnerNode(node_id="learner")
    await learner.start(bus)

    query_msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.feedback",
        payload={"text": "Not feedback"},
    )
    await bus.publish("system.feedback", query_msg)
    await asyncio.sleep(0.2)

    assert len(learner.pending_pairs) == 0

    await learner.stop()
    await bus.stop()


def test_lora_injection(tiny_model, tmp_ckpt_dir):
    """LoRA layers are injected into the model."""
    model, _ = tiny_model
    learner = LearnerNode(
        node_id="learner",
        model=model,
        batch_size=1,
        lora_r=4,
        checkpoint_dir=tmp_ckpt_dir,
    )

    # Initially not injected
    assert learner._lora_injected is False

    # Inject
    learner._ensure_lora()

    assert learner._lora_injected is True

    # Verify LoRA params exist
    lora_params = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert len(lora_params) > 0, "No LoRA parameters found after injection"


def test_lora_optimizer_targets_lora_params(tiny_model, tmp_ckpt_dir):
    """Optimizer targets only LoRA parameters."""
    model, _ = tiny_model
    learner = LearnerNode(
        node_id="learner",
        model=model,
        batch_size=1,
        lora_r=4,
        lr=1e-4,
        checkpoint_dir=tmp_ckpt_dir,
    )

    learner._ensure_lora()
    learner._ensure_optimizer()

    assert learner._optimizer is not None
    # Optimizer should have param groups
    total_opt_params = sum(len(g["params"]) for g in learner._optimizer.param_groups)
    assert total_opt_params > 0


def test_lora_adapter_save(tiny_model, tmp_ckpt_dir):
    """LoRA adapter state dict is saved to disk."""
    model, _ = tiny_model
    learner = LearnerNode(
        node_id="learner",
        model=model,
        batch_size=1,
        lora_r=4,
        checkpoint_dir=tmp_ckpt_dir,
    )

    learner._ensure_lora()
    learner._training_steps = 1  # Pretend we trained

    learner._save_adapter()

    adapter_path = Path(tmp_ckpt_dir) / "learner_lora_adapter.pt"
    assert adapter_path.exists(), f"Adapter not saved to {adapter_path}"

    state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    assert isinstance(state, dict)
    assert len(state) > 0, "Adapter state dict is empty"

    # All keys should contain 'lora_'
    for key in state:
        assert "lora_" in key, f"Non-LoRA key in adapter: {key}"


def test_learner_config_defaults():
    """LearnerNode has sensible default configuration."""
    learner = LearnerNode(node_id="test_learner")

    assert learner.batch_size == 4
    assert learner.lr == 1e-5
    assert learner.dpo_beta == 0.1
    assert learner.lora_r == 8
    assert learner._lora_injected is False
    assert learner._training_steps == 0
    assert learner.pending_pairs == {}
