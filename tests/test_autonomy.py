"""Tests for the Executive Brain Layer (Autonomy subsystem).

Covers: CognitiveStateMachine, AttentionSystem, IncrementalContextWindow,
and AutonomyCore integration.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from hbllm.brain.autonomy.attention import (
    AttentionEvent,
    AttentionSystem,
    IncrementalContextWindow,
    ScoredEvent,
)
from hbllm.brain.autonomy.loop import AutonomyCore, InternalThought
from hbllm.brain.autonomy.state_machine import (
    CognitiveState,
    CognitiveStateCategory,
    CognitiveStateMachine,
    TickProfile,
)
from hbllm.network.messages import Message, MessageType

# ──────────────────────────────────────────────
# CognitiveStateMachine tests
# ──────────────────────────────────────────────


class TestCognitiveStateMachine:
    def test_initial_state(self):
        csm = CognitiveStateMachine()
        assert csm.state == CognitiveState.IDLE
        assert csm.category == CognitiveStateCategory.PASSIVE
        assert csm.is_passive

    def test_custom_initial_state(self):
        csm = CognitiveStateMachine(initial_state=CognitiveState.OBSERVING)
        assert csm.state == CognitiveState.OBSERVING
        assert csm.is_active

    def test_state_categories(self):
        csm = CognitiveStateMachine()

        csm.transition_to(CognitiveState.FOCUSED, reason="test")
        assert csm.is_active

        csm.transition_to(CognitiveState.REFLECTING, reason="test")
        assert csm.is_passive

        csm.transition_to(CognitiveState.INTERRUPTED, reason="test")
        assert csm.is_transitional

    def test_transition_records_history(self):
        csm = CognitiveStateMachine()
        csm.transition_to(CognitiveState.OBSERVING, reason="boot")
        csm.transition_to(CognitiveState.FOCUSED, reason="user_query")

        history = csm.get_history()
        assert len(history) == 2
        assert history[0].from_state == CognitiveState.IDLE
        assert history[0].to_state == CognitiveState.OBSERVING
        assert history[1].to_state == CognitiveState.FOCUSED

    def test_noop_transition(self):
        csm = CognitiveStateMachine()
        result = csm.transition_to(CognitiveState.IDLE, reason="noop")
        assert result is True
        assert len(csm.get_history()) == 0

    def test_adaptive_tick_rates(self):
        csm = CognitiveStateMachine()

        csm.transition_to(CognitiveState.IDLE, reason="test")
        idle_tick = csm.tick_interval
        assert idle_tick == 15.0

        csm.transition_to(CognitiveState.FOCUSED, reason="test")
        focused_tick = csm.tick_interval
        assert focused_tick == 1.0

        csm.transition_to(CognitiveState.LOW_POWER, reason="test")
        lp_tick = csm.tick_interval
        assert lp_tick == 30.0

        csm.transition_to(CognitiveState.SLEEPING, reason="test")
        sleep_tick = csm.tick_interval
        assert sleep_tick == 60.0

        # Verify ordering: FOCUSED < IDLE < LOW_POWER < SLEEPING
        assert focused_tick < idle_tick < lp_tick < sleep_tick

    def test_llm_access_by_state(self):
        csm = CognitiveStateMachine()

        csm.transition_to(CognitiveState.FOCUSED, reason="test")
        assert csm.current_profile.allow_heavy_llm is True

        csm.transition_to(CognitiveState.IDLE, reason="test")
        assert csm.current_profile.allow_heavy_llm is False

        csm.transition_to(CognitiveState.LOW_POWER, reason="test")
        assert csm.current_profile.allow_heavy_llm is False
        assert csm.current_profile.allow_fast_router is False

    def test_transition_guard_blocks(self):
        csm = CognitiveStateMachine()

        def block_sleeping(from_s: CognitiveState, to_s: CognitiveState) -> bool:
            return to_s != CognitiveState.SLEEPING

        csm.add_guard(block_sleeping)
        result = csm.transition_to(CognitiveState.SLEEPING, reason="test")
        assert result is False
        assert csm.state == CognitiveState.IDLE

    def test_transition_guard_force_bypass(self):
        csm = CognitiveStateMachine()

        def block_all(from_s: CognitiveState, to_s: CognitiveState) -> bool:
            return False

        csm.add_guard(block_all)
        result = csm.transition_to(CognitiveState.OBSERVING, reason="test", force=True)
        assert result is True
        assert csm.state == CognitiveState.OBSERVING

    def test_remove_guard(self):
        csm = CognitiveStateMachine()

        def block_all(from_s: CognitiveState, to_s: CognitiveState) -> bool:
            return False

        csm.add_guard(block_all)
        csm.remove_guard(block_all)
        result = csm.transition_to(CognitiveState.OBSERVING, reason="test")
        assert result is True

    def test_transition_hook_fires(self):
        csm = CognitiveStateMachine()
        hook_calls = []

        def my_hook(transition):
            hook_calls.append(transition)

        csm.add_hook(my_hook)
        csm.transition_to(CognitiveState.OBSERVING, reason="boot")
        assert len(hook_calls) == 1
        assert hook_calls[0].to_state == CognitiveState.OBSERVING

    def test_interruption_save_and_resume(self):
        csm = CognitiveStateMachine()
        csm.transition_to(CognitiveState.FOCUSED, reason="deep_work")
        csm.transition_to(CognitiveState.INTERRUPTED, reason="urgent_user_input")
        assert csm.state == CognitiveState.INTERRUPTED
        assert csm.has_saved_state

        csm.resume_from_interruption(reason="resolved")
        assert csm.state == CognitiveState.FOCUSED
        assert not csm.has_saved_state

    def test_resume_without_saved_state(self):
        csm = CognitiveStateMachine()
        result = csm.resume_from_interruption()
        assert result is False

    def test_interruption_threshold(self):
        csm = CognitiveStateMachine()

        # IDLE has low threshold (0.1) — easy to interrupt
        csm.transition_to(CognitiveState.IDLE, reason="test")
        assert csm.should_allow_interruption(0.2) is True

        # FOCUSED has high threshold (0.8) — hard to interrupt
        csm.transition_to(CognitiveState.FOCUSED, reason="test")
        assert csm.should_allow_interruption(0.5) is False
        assert csm.should_allow_interruption(0.9) is True

    def test_cognitive_load_raises_threshold(self):
        csm = CognitiveStateMachine()
        csm.transition_to(CognitiveState.OBSERVING, reason="test")

        # No load — base threshold applies
        assert csm.should_allow_interruption(0.35) is True

        # High load — threshold increases
        csm.update_cognitive_load(1.0)
        assert csm.should_allow_interruption(0.35) is False

    def test_cognitive_load_clamped(self):
        csm = CognitiveStateMachine()
        csm.update_cognitive_load(5.0)
        assert csm.cognitive_load == 1.0
        csm.update_cognitive_load(-1.0)
        assert csm.cognitive_load == 0.0

    def test_state_duration(self):
        csm = CognitiveStateMachine()
        assert csm.state_duration >= 0.0
        assert csm.uptime >= 0.0

    def test_snapshot(self):
        csm = CognitiveStateMachine()
        csm.transition_to(CognitiveState.OBSERVING, reason="boot")
        snap = csm.snapshot()
        assert snap["state"] == "observing"
        assert snap["category"] == "active"
        assert "tick_interval_s" in snap
        assert "cognitive_load" in snap
        assert snap["transition_count"] == 1

    def test_recovering_state(self):
        csm = CognitiveStateMachine()
        csm.transition_to(CognitiveState.RECOVERING, reason="node_failure")
        assert csm.is_transitional
        assert csm.current_profile.allow_heavy_llm is False
        assert csm.tick_interval == 5.0

    def test_low_power_state(self):
        csm = CognitiveStateMachine()
        csm.transition_to(CognitiveState.LOW_POWER, reason="battery_low")
        assert csm.is_passive
        assert csm.current_profile.allow_heavy_llm is False
        assert csm.current_profile.allow_fast_router is False
        assert csm.tick_interval == 30.0


# ──────────────────────────────────────────────
# IncrementalContextWindow tests
# ──────────────────────────────────────────────


class TestIncrementalContextWindow:
    def test_update_and_get(self):
        ctx = IncrementalContextWindow()
        ctx.update("user_john", entity_type="person", salience_boost=0.5)
        entity = ctx.get_entity("user_john")
        assert entity is not None
        assert entity.entity_type == "person"
        assert entity.salience > 0.5

    def test_update_boosts_salience(self):
        ctx = IncrementalContextWindow()
        ctx.update("topic_a", salience_boost=0.2)
        s1 = ctx.get_entity("topic_a").salience
        ctx.update("topic_a", salience_boost=0.2)
        s2 = ctx.get_entity("topic_a").salience
        assert s2 > s1

    def test_mention_count_increments(self):
        ctx = IncrementalContextWindow()
        ctx.update("topic_a")
        ctx.update("topic_a")
        ctx.update("topic_a")
        assert ctx.get_entity("topic_a").mention_count == 3

    def test_decay(self):
        ctx = IncrementalContextWindow(decay_rate=0.5, prune_threshold=0.3)
        ctx.update("weak", salience_boost=0.05)  # salience = 0.35
        initial_salience = ctx.get_entity("weak").salience
        ctx.decay_all()  # 0.35 * 0.5 = 0.175 → below threshold → pruned
        entity = ctx.get_entity("weak")
        if entity is not None:
            assert entity.salience < initial_salience
        else:
            # Entity was pruned — correct behavior
            assert ctx.entity_count == 0

    def test_capacity_limit(self):
        ctx = IncrementalContextWindow(max_entities=5)
        for i in range(10):
            ctx.update(f"entity_{i}", salience_boost=float(i) * 0.1)
        assert ctx.entity_count <= 5

    def test_get_top(self):
        ctx = IncrementalContextWindow()
        ctx.update("low", salience_boost=0.1)
        ctx.update("high", salience_boost=0.9)
        ctx.update("mid", salience_boost=0.5)
        top = ctx.get_top(2)
        assert len(top) == 2
        assert top[0].entity_id == "high"

    def test_remove(self):
        ctx = IncrementalContextWindow()
        ctx.update("to_remove")
        assert ctx.remove("to_remove") is True
        assert ctx.get_entity("to_remove") is None
        assert ctx.remove("nonexistent") is False

    def test_snapshot(self):
        ctx = IncrementalContextWindow()
        ctx.update("entity_a", entity_type="topic")
        snap = ctx.snapshot()
        assert len(snap) == 1
        assert snap[0]["entity_id"] == "entity_a"


# ──────────────────────────────────────────────
# AttentionSystem tests
# ──────────────────────────────────────────────


class TestAttentionSystem:
    def _make_event(self, **kwargs) -> AttentionEvent:
        defaults = {
            "event_id": "test_event",
            "source": "test.source",
            "category": "background",
            "urgency": 0.5,
        }
        defaults.update(kwargs)
        return AttentionEvent(**defaults)

    def test_basic_scoring(self):
        attn = AttentionSystem()
        event = self._make_event(urgency=0.8)
        scored = attn.score_event(event, interruption_threshold=0.5)
        assert scored.priority_score > 0.0
        assert isinstance(scored.should_interrupt, bool)

    def test_high_urgency_interrupts(self):
        attn = AttentionSystem()
        event = self._make_event(
            urgency=1.0,
            category="user_action",
            goal_alignment=0.8,
            temporal_relevance=1.0,
        )
        scored = attn.score_event(event, user_active=True, interruption_threshold=0.3)
        assert scored.should_interrupt is True

    def test_low_urgency_does_not_interrupt(self):
        attn = AttentionSystem()
        event = self._make_event(urgency=0.1, goal_alignment=0.0)
        scored = attn.score_event(event, interruption_threshold=0.8)
        assert scored.should_interrupt is False

    def test_cognitive_load_penalty(self):
        attn = AttentionSystem()
        event = self._make_event(urgency=0.5)
        score_no_load = attn.score_event(
            event, cognitive_load=0.0, interruption_threshold=1.0
        ).priority_score

        event2 = self._make_event(event_id="e2", urgency=0.5, source="test.source2")
        score_high_load = attn.score_event(
            event2, cognitive_load=1.0, interruption_threshold=1.0
        ).priority_score

        assert score_high_load < score_no_load

    def test_event_decay_debounce(self):
        attn = AttentionSystem(decay_window_s=60.0, decay_factor=0.5)
        source = "sensor.temperature"

        # First event — no decay
        e1 = self._make_event(event_id="e1", source=source, urgency=0.5)
        s1 = attn.score_event(e1, interruption_threshold=1.0)

        # Rapid repeat events — should have increasing decay
        e2 = self._make_event(event_id="e2", source=source, urgency=0.5)
        s2 = attn.score_event(e2, interruption_threshold=1.0)

        e3 = self._make_event(event_id="e3", source=source, urgency=0.5)
        s3 = attn.score_event(e3, interruption_threshold=1.0)

        # Each subsequent event should score lower due to decay
        assert s2.priority_score <= s1.priority_score
        assert s3.priority_score <= s2.priority_score
        assert s3.decay_applied > 0.0

    def test_thought_budget_blocks(self):
        attn = AttentionSystem(max_thoughts_per_minute=3, cooldown_after_burst_s=1.0)

        for i in range(3):
            event = self._make_event(event_id=f"e{i}", source=f"s{i}")
            attn.score_event(event, interruption_threshold=1.0)

        # 4th event should be blocked
        event = self._make_event(event_id="blocked", source="s_blocked")
        scored = attn.score_event(event, interruption_threshold=1.0)
        assert scored.priority_score == 0.0
        assert attn.snapshot()["events_budget_blocked"] >= 1

    def test_tick_maintenance(self):
        attn = AttentionSystem()
        attn.context.update("entity_a")
        result = attn.tick()
        assert "context_entities_pruned" in result
        assert "event_sources_tracked" in result

    def test_snapshot(self):
        attn = AttentionSystem()
        snap = attn.snapshot()
        assert snap["events_scored"] == 0
        assert snap["thought_budget"] == 30
        assert "context_window" in snap


# ──────────────────────────────────────────────
# AutonomyCore tests
# ──────────────────────────────────────────────


class _MockBus:
    """Minimal mock MessageBus for testing AutonomyCore."""

    def __init__(self):
        self._handlers: dict[str, list] = {}
        self._published: list[tuple[str, Message]] = []

    async def subscribe(self, topic: str, handler):
        self._handlers.setdefault(topic, []).append(handler)
        return f"sub_{topic}"

    async def publish(self, topic: str, msg: Message):
        self._published.append((topic, msg))

    async def simulate_event(self, topic: str, msg: Message):
        """Simulate an external event arriving on a topic."""
        for handler in self._handlers.get(topic, []):
            await handler(msg)


class TestAutonomyCore:
    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        core = AutonomyCore()
        bus = _MockBus()
        await core.start(bus)
        assert core._running is True
        assert core.state_machine.state == CognitiveState.OBSERVING

        await core.stop()
        assert core._running is False
        assert core.state_machine.state == CognitiveState.SLEEPING

    @pytest.mark.asyncio
    async def test_reflex_fires(self):
        core = AutonomyCore()
        bus = _MockBus()

        reflex_fired = []

        def battery_reflex(event: AttentionEvent) -> Message | None:
            if event.source == "sensor.battery":
                reflex_fired.append(event)
                return Message(
                    type=MessageType.COMMAND,
                    source_node_id="autonomy",
                    topic="system.power",
                    payload={"action": "save"},
                )
            return None

        core.add_reflex("battery", battery_reflex)
        await core.start(bus)

        # Simulate a battery event
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="sensor",
            topic="sensor.battery",
            payload={"level": 10, "_urgency": 0.9},
        )
        # Directly call fast path handler
        await core._handle_fast_path_event(msg)

        assert len(reflex_fired) == 1
        assert core._reflexes_fired == 1
        await core.stop()

    @pytest.mark.asyncio
    async def test_fast_path_buffers_event(self):
        core = AutonomyCore()
        bus = _MockBus()
        await core.start(bus)

        msg = Message(
            type=MessageType.EVENT,
            source_node_id="user",
            topic="user.input",
            payload={"text": "hello", "_urgency": 0.5},
        )
        await core._handle_fast_path_event(msg)

        assert core._fast_path_wakes == 1
        assert len(core._pending_events) >= 1
        await core.stop()

    @pytest.mark.asyncio
    async def test_internal_thought_queued(self):
        core = AutonomyCore()
        thought = InternalThought(
            content="Review failed workflow",
            category="deferred_goal",
            urgency=0.4,
        )
        result = core.add_thought(thought)
        assert result is True
        assert len(core._thought_queue) == 1

    @pytest.mark.asyncio
    async def test_thought_queue_limit(self):
        core = AutonomyCore(max_pending_thoughts=3)
        for i in range(5):
            core.add_thought(InternalThought(content=f"thought {i}"))
        assert len(core._thought_queue) == 3

    @pytest.mark.asyncio
    async def test_cognitive_tick_processes_events(self):
        core = AutonomyCore()
        bus = _MockBus()
        await core.start(bus)

        # Buffer an event
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="user",
            topic="user.input",
            payload={"text": "test", "_urgency": 0.5},
        )
        await core._handle_fast_path_event(msg)
        assert len(core._pending_events) >= 1

        # Run a cognitive tick
        await core._cognitive_tick()

        # Events should be processed (emitted as actions)
        assert core._actions_emitted >= 1
        await core.stop()

    @pytest.mark.asyncio
    async def test_proactive_handler(self):
        core = AutonomyCore()
        bus = _MockBus()

        handler_called = []

        async def check_routine() -> list[Message] | None:
            handler_called.append(True)
            return [
                Message(
                    type=MessageType.EVENT,
                    source_node_id="autonomy",
                    topic="proactive.reminder",
                    payload={"text": "Time for standup"},
                )
            ]

        core.add_proactive_handler("routine", check_routine)
        await core.start(bus)

        # Run a tick
        await core._cognitive_tick()

        assert len(handler_called) == 1
        assert core._actions_emitted >= 1
        await core.stop()

    @pytest.mark.asyncio
    async def test_recursion_depth_limit(self):
        core = AutonomyCore(max_recursion_depth=1)
        bus = _MockBus()
        await core.start(bus)

        # Manually set recursion depth to max
        core._current_recursion_depth = 1

        from hbllm.brain.autonomy.attention import ScoredEvent

        scored = ScoredEvent(
            event=AttentionEvent(
                event_id="deep",
                source="test",
                category="background",
            ),
            priority_score=0.5,
            should_interrupt=False,
        )
        await core._process_scored_event(scored)
        assert core._recursion_blocks == 1
        await core.stop()

    @pytest.mark.asyncio
    async def test_snapshot(self):
        core = AutonomyCore()
        bus = _MockBus()
        await core.start(bus)

        snap = core.snapshot()
        assert snap["running"] is True
        assert "state_machine" in snap
        assert "attention" in snap
        assert snap["ticks_completed"] == 0
        await core.stop()

    @pytest.mark.asyncio
    async def test_event_classification(self):
        core = AutonomyCore()
        msg = Message(type=MessageType.EVENT, source_node_id="t", topic="user.action.click")
        assert core._classify_event_category(msg) == "user_action"

        msg2 = Message(type=MessageType.EVENT, source_node_id="t", topic="sensor.temperature")
        assert core._classify_event_category(msg2) == "sensor"

        msg3 = Message(type=MessageType.EVENT, source_node_id="t", topic="system.critical")
        assert core._classify_event_category(msg3) == "system_alert"

        msg4 = Message(type=MessageType.EVENT, source_node_id="t", topic="random.topic")
        assert core._classify_event_category(msg4) == "background"

    @pytest.mark.asyncio
    async def test_tier_determination(self):
        core = AutonomyCore()

        from hbllm.brain.autonomy.attention import ScoredEvent

        low = ScoredEvent(
            event=AttentionEvent(event_id="l", source="t", category="bg"),
            priority_score=0.2,
            should_interrupt=False,
        )
        assert core._determine_tier(low, core.state_machine.current_profile) == "tier1_reflex"

        mid = ScoredEvent(
            event=AttentionEvent(event_id="m", source="t", category="bg"),
            priority_score=0.5,
            should_interrupt=False,
        )
        assert core._determine_tier(mid, core.state_machine.current_profile) == "tier2_fast_router"
