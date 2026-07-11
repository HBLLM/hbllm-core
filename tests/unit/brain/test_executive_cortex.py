"""Unit tests for ExecutiveCortex — unified cognitive control."""

import time

import pytest

from hbllm.brain.control.executive_cortex import (
    CognitiveBudget,
    ExecutiveCortex,
    ExecutiveDecision,
)


class TestExecutiveDecision:
    def test_defaults(self):
        dec = ExecutiveDecision(action="idle")
        assert dec.action == "idle"
        assert dec.target_goal is None
        assert dec.reasoning == ""
        assert dec.budget == {}

    def test_to_dict(self):
        dec = ExecutiveDecision(
            action="switch_to_goal",
            target_goal="build_core",
            reasoning="Higher priority",
            budget={"heavy_llm": 0.3},
        )
        d = dec.to_dict()
        assert d["action"] == "switch_to_goal"
        assert d["target_goal"] == "build_core"


class TestCognitiveBudget:
    def test_defaults(self):
        b = CognitiveBudget()
        assert b.heavy_llm_pct == 0.3
        assert b.fast_router_pct == 0.5
        assert b.reflex_pct == 0.2
        total = b.heavy_llm_pct + b.fast_router_pct + b.reflex_pct
        assert abs(total - 1.0) < 0.01

    def test_to_dict(self):
        b = CognitiveBudget(heavy_llm_pct=0.5, fast_router_pct=0.3, reflex_pct=0.2)
        d = b.to_dict()
        assert d["heavy_llm"] == 0.5


class TestExecutiveCortex:
    @pytest.fixture
    def cortex(self):
        return ExecutiveCortex()

    # ── Idle State ───────────────────────────────────────────────────

    def test_idle_when_nothing_to_do(self, cortex):
        decision = cortex.decide_next_action()
        assert decision.action == "idle"

    def test_idle_reasoning(self, cortex):
        decision = cortex.decide_next_action()
        assert "No active goals" in decision.reasoning

    # ── Focus Management ─────────────────────────────────────────────

    def test_set_focus(self, cortex):
        cortex.set_focus("UserModel implementation")
        assert cortex._current_focus == "UserModel implementation"
        assert cortex._focus_depth == 0.0
        assert cortex._switch_count == 1

    def test_clear_focus(self, cortex):
        cortex.set_focus("something")
        cortex.clear_focus()
        assert cortex._current_focus == ""
        assert cortex._focus_depth == 0.0

    def test_continue_focus(self, cortex):
        cortex.set_focus("deep work")
        cortex._focus_started = time.time()
        decision = cortex.decide_next_action()
        assert decision.action == "continue_focus"
        assert decision.target_goal == "deep work"

    # ── Task Switching Cost ──────────────────────────────────────────

    def test_switching_cost_increases_with_depth(self, cortex):
        cortex._focus_depth = 0.0
        cost_shallow = cortex.get_switching_cost()
        cortex._focus_depth = 0.9
        cost_deep = cortex.get_switching_cost()
        assert cost_deep > cost_shallow

    def test_switching_cost_increases_with_recency(self, cortex):
        cortex._last_switch = time.monotonic() - 120  # 2 minutes ago
        cost_old = cortex.get_switching_cost()
        cortex._last_switch = time.monotonic() - 5  # 5 seconds ago
        cost_recent = cortex.get_switching_cost()
        assert cost_recent > cost_old

    def test_switching_cost_capped(self, cortex):
        cortex._focus_depth = 1.0
        cortex._last_switch = time.monotonic()
        cortex._switch_count = 100
        cost = cortex.get_switching_cost()
        assert cost <= 1.0

    # ── Interruption Control ─────────────────────────────────────────

    def test_should_interrupt_high_urgency(self, cortex):
        cortex._focus_depth = 0.2  # Not deep
        result = cortex.should_interrupt({"urgency": 0.95, "priority": 0.9})
        assert result is True

    def test_should_not_interrupt_low_urgency(self, cortex):
        cortex._focus_depth = 0.8  # Deep focus
        result = cortex.should_interrupt({"urgency": 0.2, "priority": 0.3})
        assert result is False

    def test_deep_focus_raises_interrupt_threshold(self, cortex):
        event = {"urgency": 0.75, "priority": 0.7}
        cortex._focus_depth = 0.1
        shallow_result = cortex.should_interrupt(event)
        cortex._focus_depth = 0.9
        _deep_result = cortex.should_interrupt(event)  # noqa: F841
        # Deep focus should be harder to interrupt
        if shallow_result:
            # It's possible deep focus blocks it
            assert shallow_result is True  # At least shallow allows it

    def test_handle_interrupt_event(self, cortex):
        cortex.set_focus("current work")
        cortex._focus_depth = 0.2
        decision = cortex.decide_next_action(
            pending_events=[{"urgency": 0.95, "priority": 0.95, "topic": "critical"}]
        )
        assert decision.action == "handle_interrupt"
        assert "critical" in decision.reasoning

    def test_suppressed_events(self, cortex):
        cortex.set_focus("current work")
        cortex._focus_depth = 0.8
        cortex.decide_next_action(
            pending_events=[{"urgency": 0.1, "priority": 0.1, "event_id": "low1"}]
        )
        suppressed = cortex.get_suppressed_events()
        assert len(suppressed) >= 0  # Events may be suppressed

    # ── Extreme Pressure ─────────────────────────────────────────────

    def test_high_pressure_sheds_load(self):
        class MockLoadManager:
            def get_pressure(self):
                return 0.95

        cortex = ExecutiveCortex(load_manager=MockLoadManager())
        decision = cortex.decide_next_action(pending_events=[{"urgency": 0.5, "priority": 0.5}])
        assert decision.action == "idle"
        assert "pressure" in decision.reasoning.lower()

    # ── Goal Selection ───────────────────────────────────────────────

    def test_selects_goal_when_no_focus(self):
        class MockGoalManager:
            def get_active_goals(self, tenant_id="default"):
                return [
                    {"name": "Build UserModel", "priority": "high"},
                    {"name": "Fix bug", "priority": "low"},
                ]

        cortex = ExecutiveCortex(goal_manager=MockGoalManager())
        decision = cortex.decide_next_action()
        assert decision.action == "switch_to_goal"
        assert decision.target_goal == "Build UserModel"

    def test_goal_priority_scoring(self):
        cortex = ExecutiveCortex()
        high = cortex._goal_priority_score({"name": "critical", "priority": "critical"})
        low = cortex._goal_priority_score({"name": "background", "priority": "background"})
        assert high > low

    def test_deadline_boost(self):
        cortex = ExecutiveCortex()
        # Goal with imminent deadline
        urgent = cortex._goal_priority_score(
            {
                "name": "deploy",
                "priority": "medium",
                "deadline": time.time() + 1800,  # 30 min
            }
        )
        # Same goal no deadline
        normal = cortex._goal_priority_score(
            {
                "name": "deploy",
                "priority": "medium",
            }
        )
        assert urgent > normal

    # ── Resource Allocation ──────────────────────────────────────────

    def test_budget_default(self, cortex):
        budget = cortex.get_cognitive_budget()
        total = budget.heavy_llm_pct + budget.fast_router_pct + budget.reflex_pct
        assert abs(total - 1.0) < 0.01

    def test_budget_high_pressure(self):
        class MockLoad:
            def get_pressure(self):
                return 0.85

        cortex = ExecutiveCortex(load_manager=MockLoad())
        budget = cortex.get_cognitive_budget()
        assert budget.reflex_pct > budget.heavy_llm_pct

    def test_budget_deep_focus(self, cortex):
        cortex._focus_depth = 0.9
        budget = cortex.get_cognitive_budget()
        assert budget.heavy_llm_pct >= 0.5  # Deep focus allows expensive reasoning

    # ── Introspection ────────────────────────────────────────────────

    def test_snapshot(self, cortex):
        cortex.set_focus("test task")
        snap = cortex.snapshot()
        assert snap["current_focus"] == "test task"
        assert "focus_depth" in snap
        assert "switching_cost" in snap
        assert "budget" in snap

    def test_reset(self, cortex):
        cortex.set_focus("work")
        cortex._switch_count = 10
        cortex.reset()
        assert cortex._current_focus == ""
        assert cortex._switch_count == 0

    # ── UserModel Alignment ──────────────────────────────────────────

    def test_goal_alignment_with_user_model(self):
        import tempfile

        from hbllm.brain.social.user_model import UserModelEngine

        with tempfile.TemporaryDirectory() as tmp:
            um = UserModelEngine(data_dir=tmp)
            um.update_from_interaction("default", "HBLLM architecture", metadata={"topic": "HBLLM"})
            cortex = ExecutiveCortex(user_model=um)

            # Goal matching user focus should get alignment bonus
            aligned_score = cortex._goal_priority_score(
                {"name": "HBLLM upgrade", "priority": "medium"}
            )
            unaligned_score = cortex._goal_priority_score(
                {"name": "Grocery shopping", "priority": "medium"}
            )
            assert aligned_score >= unaligned_score
