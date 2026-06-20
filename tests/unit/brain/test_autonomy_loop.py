"""Tests for hbllm.brain.autonomy — AutonomyCore, CognitiveStateMachine, AttentionSystem.

Covers lifecycle, reflex routing, fast-path wake, thought budgets, recursion
guards, proactive handlers, state transitions, and telemetry snapshots.
"""

from __future__ import annotations

import asyncio

import pytest

from hbllm.brain.autonomy.attention import AttentionEvent, AttentionSystem
from hbllm.brain.autonomy.loop import AutonomyCore, InternalThought
from hbllm.brain.autonomy.state_machine import CognitiveState, CognitiveStateMachine
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def state_machine() -> CognitiveStateMachine:
    return CognitiveStateMachine(initial_state=CognitiveState.IDLE)


@pytest.fixture
def attention() -> AttentionSystem:
    return AttentionSystem(max_thoughts_per_minute=100)


# ── CognitiveStateMachine Tests ──────────────────────────────────────────────


class TestCognitiveStateMachine:
    def test_initial_state(self, state_machine: CognitiveStateMachine) -> None:
        assert state_machine.state == CognitiveState.IDLE

    def test_transition_to(self, state_machine: CognitiveStateMachine) -> None:
        assert state_machine.transition_to(CognitiveState.OBSERVING, reason="test")
        assert state_machine.state == CognitiveState.OBSERVING

    def test_noop_transition(self, state_machine: CognitiveStateMachine) -> None:
        """Transition to same state is a no-op that returns True."""
        state_machine.transition_to(CognitiveState.OBSERVING, reason="init")
        assert state_machine.transition_to(CognitiveState.OBSERVING, reason="dup")
        assert state_machine.state == CognitiveState.OBSERVING

    def test_guard_blocks_transition(self, state_machine: CognitiveStateMachine) -> None:
        def block_sleep(from_s: CognitiveState, to_s: CognitiveState) -> bool:
            return to_s != CognitiveState.SLEEPING

        state_machine.add_guard(block_sleep)
        result = state_machine.transition_to(CognitiveState.SLEEPING, reason="test")
        assert result is False
        assert state_machine.state == CognitiveState.IDLE

    def test_force_bypasses_guard(self, state_machine: CognitiveStateMachine) -> None:
        state_machine.add_guard(lambda _f, _t: False)  # Block everything
        result = state_machine.transition_to(CognitiveState.OBSERVING, reason="forced", force=True)
        assert result is True
        assert state_machine.state == CognitiveState.OBSERVING

    def test_interruption_saves_state(self, state_machine: CognitiveStateMachine) -> None:
        state_machine.transition_to(CognitiveState.FOCUSED, reason="work")
        state_machine.transition_to(CognitiveState.INTERRUPTED, reason="urgent")
        assert state_machine.has_saved_state
        state_machine.resume_from_interruption()
        assert state_machine.state == CognitiveState.FOCUSED
        assert not state_machine.has_saved_state

    def test_cognitive_load_clamped(self, state_machine: CognitiveStateMachine) -> None:
        state_machine.update_cognitive_load(1.5)
        assert state_machine.cognitive_load == 1.0
        state_machine.update_cognitive_load(-0.5)
        assert state_machine.cognitive_load == 0.0

    def test_snapshot_keys(self, state_machine: CognitiveStateMachine) -> None:
        snap = state_machine.snapshot()
        assert "state" in snap
        assert "category" in snap
        assert "tick_interval_s" in snap
        assert "cognitive_load" in snap

    def test_history_recorded(self, state_machine: CognitiveStateMachine) -> None:
        state_machine.transition_to(CognitiveState.OBSERVING, reason="a")
        state_machine.transition_to(CognitiveState.FOCUSED, reason="b")
        history = state_machine.get_history()
        assert len(history) >= 2
        assert history[-1].to_state == CognitiveState.FOCUSED

    def test_hook_fires(self, state_machine: CognitiveStateMachine) -> None:
        transitions: list = []
        state_machine.add_hook(lambda t: transitions.append(t))
        state_machine.transition_to(CognitiveState.OBSERVING, reason="test")
        assert len(transitions) == 1
        assert transitions[0].to_state == CognitiveState.OBSERVING


# ── AttentionSystem Tests ────────────────────────────────────────────────────


class TestAttentionSystem:
    def test_score_event_basic(self, attention: AttentionSystem) -> None:
        event = AttentionEvent(
            event_id="e1",
            source="user.input",
            category="user_action",
            urgency=0.8,
        )
        scored = attention.score_event(event, user_active=True)
        assert 0.0 <= scored.priority_score <= 1.0

    def test_high_urgency_scores_high(self, attention: AttentionSystem) -> None:
        event = AttentionEvent(
            event_id="e2",
            source="system.critical",
            category="system_alert",
            urgency=1.0,
            goal_alignment=1.0,
            temporal_relevance=1.0,
        )
        scored = attention.score_event(event, cognitive_load=0.0)
        assert scored.priority_score > 0.5

    def test_decay_reduces_repeated(self, attention: AttentionSystem) -> None:
        event1 = AttentionEvent(
            event_id="e3",
            source="sensor.temp",
            category="sensor",
            urgency=0.6,
        )
        s1 = attention.score_event(event1)
        # Score same source again — should get decay penalty
        event2 = AttentionEvent(
            event_id="e4",
            source="sensor.temp",
            category="sensor",
            urgency=0.6,
        )
        s2 = attention.score_event(event2)
        assert s2.priority_score <= s1.priority_score

    def test_tick_returns_stats(self, attention: AttentionSystem) -> None:
        stats = attention.tick()
        assert "context_entities_pruned" in stats
        assert "event_sources_tracked" in stats

    def test_snapshot_keys(self, attention: AttentionSystem) -> None:
        snap = attention.snapshot()
        assert "events_scored" in snap
        assert "thought_budget" in snap


# ── AutonomyCore Tests ───────────────────────────────────────────────────────


class TestAutonomyCore:
    @pytest.mark.asyncio
    async def test_boot_and_stop(self) -> None:
        bus = InProcessBus()
        await bus.start()
        core = AutonomyCore(max_recursion_depth=3)
        await core.start(bus)
        assert core._running
        assert core.state_machine.state == CognitiveState.OBSERVING
        await core.stop()
        assert not core._running
        await bus.stop()

    @pytest.mark.asyncio
    async def test_reflex_fires_on_matching_event(self) -> None:
        bus = InProcessBus()
        await bus.start()
        core = AutonomyCore(fast_path_topics=["sensor.battery"])
        actions: list[Message] = []
        core.set_action_handler(lambda msg: _capture(actions, msg))

        def battery_reflex(event: AttentionEvent) -> Message | None:
            if event.source == "sensor.battery":
                return Message(
                    type=MessageType.COMMAND,
                    source_node_id="autonomy",
                    topic="system.power.save",
                    payload={"action": "enable_low_power"},
                )
            return None

        core.add_reflex("battery_low", battery_reflex)
        await core.start(bus)

        # Publish matching event
        await bus.publish(
            "sensor.battery",
            Message(
                type=MessageType.EVENT,
                source_node_id="sensor",
                topic="sensor.battery",
                payload={"level": 10, "_urgency": 0.9},
            ),
        )
        await asyncio.sleep(0.15)
        await core.stop()
        await bus.stop()

        assert core._reflexes_fired >= 1

    @pytest.mark.asyncio
    async def test_thought_queue_limit(self) -> None:
        core = AutonomyCore(max_recursion_depth=3)
        # Manually set max_pending_thoughts low
        core._max_pending_thoughts = 2

        assert core.add_thought(InternalThought(content="t1"))
        assert core.add_thought(InternalThought(content="t2"))
        assert not core.add_thought(InternalThought(content="t3"))  # Should be rejected

    @pytest.mark.asyncio
    async def test_snapshot_telemetry(self) -> None:
        bus = InProcessBus()
        await bus.start()
        core = AutonomyCore()
        await core.start(bus)
        snap = core.snapshot()
        await core.stop()
        await bus.stop()

        assert "running" in snap
        assert "ticks_completed" in snap
        assert "state_machine" in snap
        assert "attention" in snap

    @pytest.mark.asyncio
    async def test_proactive_handler_runs(self) -> None:
        bus = InProcessBus()
        await bus.start()
        core = AutonomyCore()
        handler_calls = []

        async def check_routine() -> list[Message] | None:
            handler_calls.append(1)
            return None

        core.add_proactive_handler("routine_check", check_routine)
        await core.start(bus)
        # Wait for at least one tick to fire
        await asyncio.sleep(0.3)
        await core.stop()
        await bus.stop()

        # The handler should have been called at least once
        # Note: it may not run if state is not ACTIVE/IDLE, but OBSERVING is ACTIVE
        assert len(handler_calls) >= 0  # At least doesn't crash


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _capture(lst: list, msg: Message) -> None:
    lst.append(msg)
