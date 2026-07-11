"""Tests for Phase 6 — Advanced Capabilities.

Covers: OnlineLearner, ExplorationSandbox, PushBackends, MultiAgent,
ParallelWorkspace, ReflexLearner, SocialTiming, CognitiveLoad.
"""

import time

import pytest
import pytest_asyncio

# ── Online Learner ────────────────────────────────────────────────────────
from hbllm.training.online_learner import OnlineLearner


class TestOnlineLearner:
    @pytest_asyncio.fixture
    async def learner(self, tmp_path):
        l = OnlineLearner(db_path=tmp_path / "learning.db")
        await l.init_db()
        return l

    @pytest.mark.asyncio
    async def test_record_correction(self, learner):
        learner.record_correction("t1", "Use Celsius not Fahrenheit")
        ctx = learner.get_learnings_context("t1")
        assert "Celsius" in ctx

    @pytest.mark.asyncio
    async def test_correction_reinforcement(self, learner):
        learner.record_correction("t1", "Always use dark mode")
        learner.record_correction("t1", "Always use dark mode")
        ctx = learner.get_learnings_context("t1")
        assert "×2" in ctx

    @pytest.mark.asyncio
    async def test_record_preference(self, learner):
        learner.record_preference("t1", "response_style", "concise")
        ctx = learner.get_learnings_context("t1")
        assert "concise" in ctx

    @pytest.mark.asyncio
    async def test_tool_outcome_tracking(self, learner):
        learner.record_tool_outcome("t1", "web_search", success=True, latency_ms=300)
        learner.record_tool_outcome("t1", "web_search", success=True, latency_ms=200)
        learner.record_tool_outcome("t1", "web_search", success=False)
        rate = learner.get_tool_success_rate("t1", "web_search")
        assert rate is not None
        assert abs(rate - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_empty_context(self, learner):
        ctx = learner.get_learnings_context("nonexistent")
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_stats(self, learner):
        learner.record_correction("t1", "test")
        s = learner.stats("t1")
        assert "learnings_by_category" in s


# ── Exploration Sandbox ───────────────────────────────────────────────────

from hbllm.training.exploration_sandbox import ExplorationSandbox


class TestExplorationSandbox:
    @pytest.fixture
    def sandbox(self):
        return ExplorationSandbox()

    def test_simulate_light_on(self, sandbox):
        result = sandbox.simulate_action("light.on", {"device_id": "kitchen"})
        assert result["success"]
        assert "on" in result["status"].lower()

    def test_simulate_lock_unlock(self, sandbox):
        result = sandbox.simulate_action("lock.unlock", {"device_id": "front"})
        assert result["success"]

    def test_validate_plan(self, sandbox):
        plan = [
            {"action": "thermostat.set_temp", "params": {"temp": 22}},
            {"action": "light.on", "params": {"device_id": "bedroom"}},
        ]
        result = sandbox.validate_plan(plan)
        assert result["plan_valid"]
        assert result["total_steps"] == 2

    def test_what_if_side_effects(self, sandbox):
        result = sandbox.what_if("lock.unlock", {"device_id": "front"})
        assert "side_effects" in result
        assert len(result["side_effects"]) > 0

    def test_history_tracking(self, sandbox):
        sandbox.simulate_action("light.on", {"device_id": "a"})
        sandbox.simulate_action("light.off", {"device_id": "a"})
        history = sandbox.get_history()
        assert len(history) == 2

    def test_reset(self, sandbox):
        sandbox.simulate_action("light.on", {"device_id": "a"})
        sandbox.reset()
        assert len(sandbox.get_history()) == 0


# ── Push Backends ─────────────────────────────────────────────────────────

from hbllm.serving.push_backends import InMemoryBackend, MultiBackend, PushNotification


class TestPushBackends:
    @pytest.mark.asyncio
    async def test_in_memory_send(self):
        backend = InMemoryBackend()
        result = await backend.send(
            PushNotification(title="Test", body="Hello", device_token="abc")
        )
        assert result.success
        assert len(backend.sent) == 1

    @pytest.mark.asyncio
    async def test_in_memory_batch(self):
        backend = InMemoryBackend()
        notifs = [
            PushNotification(title=f"N{i}", body="body", device_token=f"t{i}") for i in range(3)
        ]
        results = await backend.send_batch(notifs)
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_multi_backend_routing(self):
        fcm = InMemoryBackend()
        apns = InMemoryBackend()
        multi = MultiBackend(fcm=fcm, apns=apns)

        await multi.send(PushNotification(title="A", body="a", device_type="fcm", device_token="t"))
        await multi.send(
            PushNotification(title="B", body="b", device_type="apns", device_token="t")
        )

        assert len(fcm.sent) == 1
        assert len(apns.sent) == 1

    @pytest.mark.asyncio
    async def test_notification_to_dict(self):
        n = PushNotification(title="Test", body="Body", priority="high", category="iot")
        d = n.to_dict()
        assert d["title"] == "Test"
        assert d["priority"] == "high"


# ── Multi-Agent ───────────────────────────────────────────────────────────

from hbllm.network.multi_agent import (
    AgentIdentity,
    AgentMessage,
    AgentMessageType,
    MultiAgentCoordinator,
)


class TestMultiAgent:
    def test_register_peer(self):
        coord = MultiAgentCoordinator(identity=AgentIdentity(name="Main"))
        peer = AgentIdentity(name="IoT", capabilities=["iot_control"])
        coord.register_peer(peer)
        assert len(coord.get_peer_list()) == 1

    def test_find_capable_peers(self):
        coord = MultiAgentCoordinator(identity=AgentIdentity(name="Main"))
        p1 = AgentIdentity(name="IoT", capabilities=["iot_control", "vision"])
        p2 = AgentIdentity(name="Code", capabilities=["code_execution"])
        coord.register_peer(p1)
        coord.register_peer(p2)
        found = coord.find_capable_peers(["iot_control"])
        assert len(found) == 1
        assert found[0].name == "IoT"

    def test_find_no_capable_peers(self):
        coord = MultiAgentCoordinator()
        found = coord.find_capable_peers(["quantum_computing"])
        assert len(found) == 0

    def test_agent_message_to_dict(self):
        msg = AgentMessage(
            message_type=AgentMessageType.DELEGATE,
            sender_id="a1",
            recipient_id="a2",
            payload={"task": "test"},
        )
        d = msg.to_dict()
        assert d["message_type"] == "delegate"
        assert d["sender_id"] == "a1"

    def test_agent_message_from_dict(self):
        d = {"message_type": "heartbeat", "sender_id": "a1", "payload": {"load": 0.5}}
        msg = AgentMessage.from_dict(d)
        assert msg.message_type == AgentMessageType.HEARTBEAT

    def test_message_reply(self):
        msg = AgentMessage(sender_id="a1", recipient_id="a2")
        reply = msg.create_reply(AgentMessageType.RESULT, {"status": "ok"})
        assert reply.sender_id == "a2"
        assert reply.recipient_id == "a1"

    def test_message_expiry(self):
        msg = AgentMessage(timestamp=time.time() - 600, ttl_s=300)
        assert msg.is_expired

    def test_stats(self):
        coord = MultiAgentCoordinator()
        s = coord.stats()
        assert s["total_peers"] == 0
        assert s["tasks_delegated"] == 0


# ── Parallel Workspace ────────────────────────────────────────────────────

from hbllm.brain.planning.parallel_workspace import ParallelWorkspaceManager, WorkspaceStatus


class TestParallelWorkspace:
    @pytest.fixture
    def manager(self):
        return ParallelWorkspaceManager(max_concurrent=3)

    def test_create_workspace(self, manager):
        ws_id = manager.create_workspace("Research", "Find X")
        assert ws_id is not None
        ws = manager.get_workspace(ws_id)
        assert ws.name == "Research"
        assert ws.status == WorkspaceStatus.IDLE

    @pytest.mark.asyncio
    async def test_run_workspace_no_provider(self, manager):
        ws_id = manager.create_workspace("Test", "Do something")
        result = await manager.run_workspace(ws_id)
        assert result["status"] == "no_provider"

    def test_get_all_statuses(self, manager):
        manager.create_workspace("A", "task a")
        manager.create_workspace("B", "task b")
        statuses = manager.get_all_statuses()
        assert len(statuses) == 2

    def test_stats(self, manager):
        s = manager.stats()
        assert s["max_concurrent"] == 3


# ── Reflex Learner ────────────────────────────────────────────────────────

from hbllm.brain.autonomy.reflex_learner import ReflexLearner, ReflexStore


class TestReflexLearner:
    @pytest_asyncio.fixture
    async def store(self, tmp_path):
        s = ReflexStore(db_path=tmp_path / "reflexes.db")
        s.init_db()
        return s

    @pytest_asyncio.fixture
    async def learner(self, store):
        return ReflexLearner(store=store)

    @pytest.mark.asyncio
    async def test_learn_from_instruction(self, learner):
        reflex = await learner.learn_from_instruction(
            "t1", "When humidity above 70% then turn on dehumidifier"
        )
        assert reflex.name
        assert not reflex.approved  # Requires approval

    @pytest.mark.asyncio
    async def test_learned_reflex_stored(self, learner, store):
        await learner.learn_from_instruction("t1", "When temp below 18 then turn on heater")
        pending = store.get_pending_approval("t1")
        assert len(pending) >= 1

    @pytest.mark.asyncio
    async def test_approve_reflex(self, learner, store):
        reflex = await learner.learn_from_instruction("t1", "Test instruction")
        store.approve(reflex.reflex_id)
        active = store.get_active("t1")
        assert len(active) >= 1

    def test_learn_from_pattern(self, learner):
        reflex = learner.learn_from_pattern("t1", {"trigger": "bedtime", "action": "lights.off"})
        assert reflex.source == "behavior_pattern"
        assert reflex.confidence < 0.5  # Low confidence for patterns


# ── Social Timing ─────────────────────────────────────────────────────────

from hbllm.brain.social.social_timing import SocialContext, SocialTimingEngine


class TestSocialTiming:
    @pytest.fixture
    def engine(self):
        return SocialTimingEngine()

    def test_critical_always_delivers(self, engine):
        d = engine.evaluate(priority="critical")
        assert d.deliver_now

    def test_suggestion_held_in_meeting(self, engine):
        engine.update_context(SocialContext(in_meeting=True))
        d = engine.evaluate(priority="suggestion", category="weather")
        assert not d.deliver_now
        assert "meeting" in d.reason.lower()

    def test_sleeping_holds_everything(self, engine):
        engine.update_context(SocialContext(is_sleeping=True))
        d = engine.evaluate(priority="normal")
        assert not d.deliver_now

    def test_sleeping_allows_critical(self, engine):
        engine.update_context(SocialContext(is_sleeping=True))
        d = engine.evaluate(priority="critical")
        assert d.deliver_now

    def test_driving_uses_voice(self, engine):
        engine.update_context(SocialContext(is_driving=True))
        d = engine.evaluate(priority="high")
        assert d.channel == "voice"

    def test_stats(self, engine):
        s = engine.stats()
        assert "delivered_now" in s
        assert "current_context" in s


# ── Cognitive Load ────────────────────────────────────────────────────────

from hbllm.brain.self_model.cognitive_load_estimator import CognitiveLoadEstimator


class TestCognitiveLoad:
    @pytest.fixture
    def estimator(self):
        return CognitiveLoadEstimator()

    def test_initial_estimate(self, estimator):
        est = estimator.estimate()
        assert 0.0 <= est.load_level <= 1.0
        assert est.load_label in ("relaxed", "normal", "focused", "high", "overloaded")

    def test_messages_affect_load(self, estimator):
        for _ in range(20):
            estimator.record_message("fix this bug now")
        est = estimator.estimate()
        assert est.load_level > 0

    def test_engagement_update(self, estimator):
        estimator.update_engagement("idle")
        est = estimator.estimate()
        assert est.signals["engagement"] < 0.5

    def test_system_prompt_modifier(self, estimator):
        modifier = estimator.get_system_prompt_modifier()
        assert isinstance(modifier, str)

    def test_estimate_to_dict(self, estimator):
        est = estimator.estimate()
        d = est.to_dict()
        assert "load_level" in d
        assert "recommended_verbosity" in d

    def test_suppress_proactive_when_overloaded(self):
        est = CognitiveLoadEstimator(session_start=time.time() - 86400)
        for _ in range(30):
            est.record_message("*typo* actually nvm sorry")
        result = est.estimate()
        # With long session + typo corrections, load should be elevated
        assert result.load_level > 0.3

    def test_stats(self, estimator):
        s = estimator.stats()
        assert "current_load" in s
        assert "session_duration_min" in s
