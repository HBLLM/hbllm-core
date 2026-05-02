"""Tests for EmotionEngine cognitive plugin."""

import asyncio
import sys
from pathlib import Path

# Add plugin directory to path for direct import
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
from emotion_engine import (
    _EMOTION_LEXICON,
    EmotionEngine,
    EmotionState,
)

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def engine(bus):
    e = EmotionEngine(node_id="test_emotion")
    await e.start(bus)
    yield e
    await e.stop()


class TestEmotionState:
    def test_default_neutral(self):
        s = EmotionState()
        assert s.valence == 0.0
        assert s.primary_emotion == "neutral"

    def test_to_dict(self):
        s = EmotionState(valence=0.5, arousal=0.3)
        d = s.to_dict()
        assert d["valence"] == 0.5
        assert "primary_emotion" in d


class TestEmotionLexicon:
    def test_has_positive_words(self):
        assert "happy" in _EMOTION_LEXICON
        assert _EMOTION_LEXICON["happy"][0] > 0

    def test_has_negative_words(self):
        assert "frustrated" in _EMOTION_LEXICON
        assert _EMOTION_LEXICON["frustrated"][0] < 0


class TestEmotionEngine:
    async def test_starts_neutral(self, engine):
        assert engine.state.primary_emotion == "neutral"
        assert engine.state.valence == 0.0

    async def test_positive_text_shifts_valence(self, engine, bus):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="system.experience",
            payload={"text": "This is amazing and wonderful! I love it!"},
        )
        await bus.publish("system.experience", msg)
        await asyncio.sleep(0.05)
        assert engine.state.valence > 0.0

    async def test_negative_text_shifts_valence(self, engine, bus):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="system.experience",
            payload={"text": "I am frustrated and angry. This is terrible."},
        )
        await bus.publish("system.experience", msg)
        await asyncio.sleep(0.05)
        assert engine.state.valence < 0.0

    async def test_neutral_text_decays(self, engine):
        engine.state.valence = 0.5
        engine._update_from_text("how do I configure the server settings")
        assert engine.state.valence < 0.5

    async def test_low_eval_score_reduces_valence(self, engine, bus):
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="eval",
            topic="system.evaluation",
            payload={"overall_score": 0.2, "flags": []},
        )
        await bus.publish("system.evaluation", msg)
        await asyncio.sleep(0.05)
        assert engine.state.valence < 0.0

    async def test_adaptation_hints_empathetic(self, engine):
        engine.state.valence = -0.5
        engine.state.primary_emotion = "frustration"
        hints = engine.get_adaptation_hints()
        assert hints["tone"] == "patient"
        assert hints["empathy_level"] == "high"
        assert hints["encouragement"] is True

    async def test_adaptation_hints_enthusiastic(self, engine):
        engine.state.valence = 0.8
        hints = engine.get_adaptation_hints()
        assert hints["tone"] == "enthusiastic"

    async def test_stats_structure(self, engine):
        stats = engine.stats()
        assert "current_state" in stats
        assert "adaptation_hints" in stats
        assert "history_size" in stats
        assert "trend" in stats

    async def test_history_tracking(self, engine, bus):
        for word in ["happy", "sad", "excited"]:
            msg = Message(
                type=MessageType.EVENT,
                source_node_id="test",
                topic="system.experience",
                payload={"text": f"I feel {word}"},
            )
            await bus.publish("system.experience", msg)
            await asyncio.sleep(0.02)
        assert engine.stats()["history_size"] >= 3
