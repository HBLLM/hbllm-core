"""Unit tests for ConversationTurnManager — voice conversation state machine."""

import pytest

from hbllm.perception.conversation_turn import ConversationTurnManager, TurnState


class TestTurnState:
    def test_enum_values(self):
        assert TurnState.IDLE == "idle"
        assert TurnState.LISTENING == "listening"
        assert TurnState.PROCESSING == "processing"
        assert TurnState.SPEAKING == "speaking"
        assert TurnState.INTERRUPTED == "interrupted"


class TestConversationTurnManager:
    def test_init(self):
        mgr = ConversationTurnManager(node_id="turn_test")
        assert mgr.node_id == "turn_test"
        assert mgr.state == TurnState.IDLE

    def test_initial_snapshot(self):
        mgr = ConversationTurnManager()
        snap = mgr.snapshot()
        assert snap["state"] == "idle"
        assert snap["turn_count"] == 0
        assert snap["session_id"] is None
        assert snap["continuous_listen"] is True

    def test_state_property(self):
        mgr = ConversationTurnManager()
        assert mgr.state == TurnState.IDLE
        mgr._state = TurnState.LISTENING
        assert mgr.state == TurnState.LISTENING

    def test_capabilities(self):
        mgr = ConversationTurnManager()
        assert "turn_management" in mgr.capabilities
        assert "barge_in" in mgr.capabilities
        assert "conversation_state" in mgr.capabilities

    @pytest.mark.asyncio
    async def test_on_stop(self):
        mgr = ConversationTurnManager()
        # Should not raise even without start
        await mgr.on_stop()

    @pytest.mark.asyncio
    async def test_handle_message_returns_none(self):
        from hbllm.network.messages import Message, MessageType

        mgr = ConversationTurnManager()
        msg = Message(
            type=MessageType.QUERY,
            source_node_id="test",
            topic="test",
            payload={},
        )
        result = await mgr.handle_message(msg)
        assert result is None
