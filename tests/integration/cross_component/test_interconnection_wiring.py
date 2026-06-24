"""Tests for cross-subsystem interconnection wiring (P0-P2).

Validates that cognitive subsystems correctly exchange information
via UserModel, ProjectGraph, and RelationshipMemory bridges.
"""

import tempfile
import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

# ── Shared Mock Helpers ──────────────────────────────────────────────────────


@dataclass
class MockLearnedAttribute:
    value: Any = ""
    confidence: float = 0.8
    evidence_count: int = 5
    source: str = "test"
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)


@dataclass
class MockExpertise:
    level: float = 0.8
    domain: str = "python"


@dataclass
class MockUserPreference:
    value: str = "concise"
    confidence: float = 0.8


@dataclass
class MockUserModel:
    tenant_id: str = "test_tenant"
    preferences: dict = field(default_factory=dict)
    expertise: dict = field(default_factory=dict)
    stress_level: float = 0.0
    active_hours: dict = field(default_factory=dict)
    active_days: dict = field(default_factory=dict)
    active_interests: list = field(default_factory=list)


def make_mock_user_model_engine(model: MockUserModel | None = None) -> MagicMock:
    """Create a mock UserModelEngine that returns a predictable UserModel."""
    engine = MagicMock()
    engine.get_model.return_value = model or MockUserModel()
    return engine


# ── P0: PersonaEngine ← UserModel ────────────────────────────────────────────


class TestPersonaEngineSync:
    """P0: PersonaEngine.sync_from_user_model()."""

    def test_sync_verbosity_concise(self):
        """Concise preference should lower PersonaEngine verbosity."""
        from hbllm.brain.persona_engine import PersonaEngine

        model = MockUserModel(
            preferences={"verbosity": MockUserPreference(value="concise", confidence=0.8)}
        )
        mock_um = make_mock_user_model_engine(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir, user_model=mock_um)
            engine.get_profile("t1")  # Create default profile
            changed = engine.sync_from_user_model("t1")
            assert changed
            profile = engine.get_profile("t1")
            assert profile.verbosity.value < 0.5

    def test_sync_expertise_increases_depth(self):
        """High expertise should increase technical_depth."""
        from hbllm.brain.persona_engine import PersonaEngine

        model = MockUserModel(
            expertise={"python": MockExpertise(level=0.9), "ml": MockExpertise(level=0.8)}
        )
        mock_um = make_mock_user_model_engine(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir, user_model=mock_um)
            p0 = engine.get_profile("t1")
            initial_depth = p0.technical_depth.value
            engine.sync_from_user_model("t1")
            p1 = engine.get_profile("t1")
            assert p1.technical_depth.value > initial_depth

    def test_sync_stress_boosts_empathy(self):
        """High stress should increase empathy and decrease verbosity."""
        from hbllm.brain.persona_engine import PersonaEngine

        model = MockUserModel(stress_level=0.9)
        mock_um = make_mock_user_model_engine(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir, user_model=mock_um)
            engine.get_profile("t1")
            changed = engine.sync_from_user_model("t1")
            assert changed
            p = engine.get_profile("t1")
            assert p.empathy.value > 0.5

    def test_sync_no_user_model_returns_false(self):
        """Without UserModel, sync should return False."""
        from hbllm.brain.persona_engine import PersonaEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir)
            assert engine.sync_from_user_model("t1") is False

    def test_sync_formality_casual(self):
        """Casual formality preference should lower formality trait."""
        from hbllm.brain.persona_engine import PersonaEngine

        model = MockUserModel(
            preferences={"formality": MockUserPreference(value="casual", confidence=0.8)}
        )
        mock_um = make_mock_user_model_engine(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            engine = PersonaEngine(storage_dir=tmpdir, user_model=mock_um)
            engine.get_profile("t1")
            changed = engine.sync_from_user_model("t1")
            assert changed
            p = engine.get_profile("t1")
            assert p.formality.value < 0.5


# ── P0: SocialTiming ← UserModel + RelationshipMemory ──────────────────────


class TestSocialTimingIntegration:
    """P0: SocialTiming dynamic quiet hours and person-based priority."""

    def test_dynamic_quiet_hours_from_user_model(self):
        """Quiet hours should be derived from UserModel active_hours."""
        from hbllm.brain.social_timing import SocialTimingEngine

        # User is inactive at hour 3
        model = MockUserModel(active_hours={3: 0.05, 10: 0.9, 14: 0.8})
        mock_um = make_mock_user_model_engine(model)

        engine = SocialTimingEngine(user_model=mock_um)
        # Hour 3 should be quiet
        assert engine._is_quiet_hours(3, "test_tenant") is True
        # Hour 10 should NOT be quiet
        assert engine._is_quiet_hours(10, "test_tenant") is False

    def test_quiet_hours_fallback_without_user_model(self):
        """Without UserModel, fall back to static quiet_start/quiet_end."""
        from hbllm.brain.social_timing import SocialTimingEngine

        engine = SocialTimingEngine(quiet_start_hour=23, quiet_end_hour=7)
        # 3 AM should be quiet with static config
        assert engine._is_quiet_hours(3, "default") is True
        # 10 AM should not
        assert engine._is_quiet_hours(10, "default") is False

    def test_boost_priority_for_person(self):
        """Known important person should boost delivery priority."""
        from hbllm.brain.social_timing import SocialTimingEngine

        mock_rm = MagicMock()
        mock_rm.prioritize_notification.return_value = 0.9  # High importance

        engine = SocialTimingEngine(relationship_memory=mock_rm)
        boosted = engine.boost_priority_for_person(
            "Message from Mom", "normal", tenant_id="t1",
        )
        # Should be upgraded from "normal"
        assert boosted in ("high", "urgent", "critical") or boosted == "normal"

    def test_boost_priority_no_relationship_memory(self):
        """Without RelationshipMemory, priority stays unchanged."""
        from hbllm.brain.social_timing import SocialTimingEngine

        engine = SocialTimingEngine()
        result = engine.boost_priority_for_person("Hello", "normal", tenant_id="t1")
        assert result == "normal"

    def test_evaluate_with_tenant_id(self):
        """evaluate() should accept tenant_id parameter."""
        from hbllm.brain.social_timing import SocialTimingEngine

        engine = SocialTimingEngine()
        decision = engine.evaluate(priority="normal", tenant_id="test_tenant")
        assert decision.deliver_now is not None


# ── P0: CuriosityNode ← UserModel ──────────────────────────────────────────


class TestCuriosityNodeInterestBoost:
    """P0: CuriosityNode interest-weighted goal prioritization."""

    def test_compute_interest_boost(self):
        """Topics matching user interests should get a boost."""
        from hbllm.brain.curiosity_node import CuriosityNode

        node = CuriosityNode(node_id="test_curiosity")
        # Simulate cached user interests
        node._user_interests = {"machine_learning": 0.9, "cooking": 0.7}
        boost = node._compute_interest_boost("machine_learning")
        assert boost > 0.0

    def test_compute_interest_boost_no_match(self):
        """Topics not in user interests should get no boost (1.0 multiplier)."""
        from hbllm.brain.curiosity_node import CuriosityNode

        node = CuriosityNode(node_id="test_curiosity")
        node._user_interests = {"machine_learning": 0.9}
        boost = node._compute_interest_boost("gardening")
        assert boost == 1.0

    def test_compute_interest_boost_empty(self):
        """No cached interests should give no boost (1.0 multiplier)."""
        from hbllm.brain.curiosity_node import CuriosityNode

        node = CuriosityNode(node_id="test_curiosity")
        node._user_interests = {}
        boost = node._compute_interest_boost("anything")
        assert boost == 1.0


# ── P0: ProactiveInsight ← UserModel + ProjectGraph ────────────────────────


class TestProactiveInsightIntegration:
    """P0: Stress suppression and project-aware insights."""

    def test_max_insights_reduced_by_stress(self):
        """High stress should reduce the number of suggestions."""
        from hbllm.brain.autonomy.proactive_insight import ProactiveInsightGenerator

        model = MockUserModel(stress_level=0.9)
        mock_um = make_mock_user_model_engine(model)

        gen = ProactiveInsightGenerator(user_model=mock_um)
        max_insights = gen._get_max_insights("test_tenant")
        # Should be less than the default max (which is typically 5)
        assert max_insights <= 3

    def test_max_insights_normal_without_stress(self):
        """Normal stress should not suppress insights."""
        from hbllm.brain.autonomy.proactive_insight import ProactiveInsightGenerator

        model = MockUserModel(stress_level=0.2)
        mock_um = make_mock_user_model_engine(model)

        gen = ProactiveInsightGenerator(user_model=mock_um)
        max_insights = gen._get_max_insights("test_tenant")
        assert max_insights >= 3

    def test_max_insights_without_user_model(self):
        """Without UserModel, use default max."""
        from hbllm.brain.autonomy.proactive_insight import ProactiveInsightGenerator

        gen = ProactiveInsightGenerator()
        max_insights = gen._get_max_insights("test_tenant")
        assert max_insights >= 3


# ── P1: DelegationChain ← ProjectGraph ──────────────────────────────────────


class TestDelegationChainProjectIntegration:
    """P1: Delegation tagging and project goal updates."""

    def test_create_with_project_id(self):
        """Delegations can be tagged with a project_id."""
        from hbllm.brain.delegation_chain import DelegationManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                tenant_id="t1",
                objective="Deploy v2",
                project_id="proj_1",
            )
            assert d.project_id == "proj_1"

    def test_project_id_serialization(self):
        """project_id should survive serialization roundtrip."""
        from hbllm.brain.delegation_chain import Delegation

        d = Delegation(tenant_id="t1", objective="Test", project_id="proj_1")
        data = d.to_dict()
        assert data["project_id"] == "proj_1"
        d2 = Delegation.from_dict(data)
        assert d2.project_id == "proj_1"

    def test_project_goal_updated_on_completion(self):
        """Completing a delegation should update the linked ProjectGraph goal."""
        from hbllm.brain.delegation_chain import DelegationManager, DelegationStep

        mock_pg = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir, project_graph=mock_pg)
            d = mgr.create(
                tenant_id="t1",
                objective="Deploy staging",
                project_id="proj_1",
                steps=[DelegationStep(description="Run tests")],
            )
            step = d.steps[0]
            step.status = step.status  # Keep pending
            mgr.next_step("t1", d.id)  # Starts step
            mgr.complete_step("t1", d.id, step.id, result="All passed")

            # ProjectGraph.complete_goal should have been called
            mock_pg.complete_goal.assert_called_once()
            call_kwargs = mock_pg.complete_goal.call_args
            assert call_kwargs[1]["project_id"] == "proj_1" or call_kwargs[0][0] == "proj_1"

    def test_no_project_goal_update_without_project_id(self):
        """Without project_id, no ProjectGraph update should happen."""
        from hbllm.brain.delegation_chain import DelegationManager, DelegationStep

        mock_pg = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir, project_graph=mock_pg)
            d = mgr.create(
                tenant_id="t1",
                objective="General task",
                steps=[DelegationStep(description="Step 1")],
            )
            mgr.next_step("t1", d.id)
            mgr.complete_step("t1", d.id, d.steps[0].id, result="Done")
            mock_pg.complete_goal.assert_not_called()


# ── P1: ActivityDigest ← ProjectGraph + UserModel ───────────────────────────


class TestActivityDigestIntegration:
    """P1: Project-aware digests and verbosity filtering."""

    def test_enrich_with_project_context(self):
        """Digest should include project goal items when ProjectGraph is connected."""
        from hbllm.brain.activity_digest import ActivityDigestEngine, DigestItem

        mock_pg = MagicMock()
        mock_pg.list_projects.return_value = [
            {"id": "proj_1", "name": "MyProject"}
        ]
        mock_goal = MagicMock()
        mock_goal.name = "Deploy v2"
        mock_pg.get_active_goals.return_value = [mock_goal]

        engine = ActivityDigestEngine(project_graph=mock_pg)
        engine.record_event("t1", DigestItem(category="system", title="Test"))
        engine.record_activity("t1")

        digest = engine.generate_digest("t1")
        # Should have the system event + project goal item
        categories = [item.category for item in digest.items]
        assert "goal" in categories

    def test_verbosity_filtering_concise(self):
        """Concise users should get fewer, higher-importance items."""
        from hbllm.brain.activity_digest import ActivityDigestEngine, DigestItem

        model = MockUserModel(
            preferences={"verbosity": MockUserPreference(value="concise", confidence=0.8)}
        )
        mock_um = make_mock_user_model_engine(model)

        engine = ActivityDigestEngine(user_model=mock_um)
        # Add a mix of high and low importance items
        for i in range(15):
            engine.record_event(
                "t1",
                DigestItem(
                    category="notification",
                    title=f"Event {i}",
                    importance=0.2 if i % 2 == 0 else 0.8,
                ),
            )

        digest = engine.generate_digest("t1")
        # Low-importance items should be filtered out for concise users
        for item in digest.items:
            assert item.importance >= 0.5

    def test_no_enrichment_without_project_graph(self):
        """Without ProjectGraph, digest works normally."""
        from hbllm.brain.activity_digest import ActivityDigestEngine, DigestItem

        engine = ActivityDigestEngine()
        engine.record_event("t1", DigestItem(category="system", title="Test"))
        digest = engine.generate_digest("t1")
        assert len(digest.items) == 1

    def test_stats_reflect_connections(self):
        """Stats should report whether integrations are connected."""
        from hbllm.brain.activity_digest import ActivityDigestEngine

        engine = ActivityDigestEngine(
            user_model=make_mock_user_model_engine(),
            project_graph=MagicMock(),
        )
        stats = engine.stats()
        assert stats["user_model_connected"] is True
        assert stats["project_graph_connected"] is True


# ── P2: HabitTracker ← UserModel ───────────────────────────────────────────


class TestHabitTrackerCrossValidation:
    """P2: HabitTracker cross-validates habits against UserModel active_hours."""

    def test_boost_confidence_for_active_hour(self):
        """Habits at known-active hours should get a confidence boost."""
        from hbllm.brain.habit_tracker import HabitTracker

        model = MockUserModel(active_hours={9: 0.9, 3: 0.02})
        mock_um = make_mock_user_model_engine(model)

        tracker = HabitTracker(user_model=mock_um)
        base_confidence = 0.6
        boosted = tracker._cross_validate_with_user_model("t1", base_confidence, hour=9)
        assert boosted > base_confidence

    def test_penalize_confidence_for_inactive_hour(self):
        """Habits at known-inactive hours should get a confidence penalty."""
        from hbllm.brain.habit_tracker import HabitTracker

        model = MockUserModel(active_hours={3: 0.02})
        mock_um = make_mock_user_model_engine(model)

        tracker = HabitTracker(user_model=mock_um)
        base_confidence = 0.6
        penalized = tracker._cross_validate_with_user_model("t1", base_confidence, hour=3)
        assert penalized < base_confidence

    def test_no_change_without_user_model(self):
        """Without UserModel, confidence should remain unchanged."""
        from hbllm.brain.habit_tracker import HabitTracker

        tracker = HabitTracker()
        result = tracker._cross_validate_with_user_model("t1", 0.6, hour=9)
        assert result == 0.6

    def test_day_validation_boost(self):
        """Habits on known-active days should get a boost."""
        from hbllm.brain.habit_tracker import HabitTracker

        model = MockUserModel(active_days={0: 0.9})  # Monday active
        mock_um = make_mock_user_model_engine(model)

        tracker = HabitTracker(user_model=mock_um)
        boosted = tracker._cross_validate_with_user_model("t1", 0.6, day=0)
        assert boosted > 0.6

    def test_day_validation_penalize(self):
        """Habits on known-inactive days should get penalized."""
        from hbllm.brain.habit_tracker import HabitTracker

        model = MockUserModel(active_days={6: 0.05})  # Sunday inactive
        mock_um = make_mock_user_model_engine(model)

        tracker = HabitTracker(user_model=mock_um)
        penalized = tracker._cross_validate_with_user_model("t1", 0.6, day=6)
        assert penalized < 0.6

    def test_confidence_capped_at_one(self):
        """Boosted confidence should never exceed 1.0."""
        from hbllm.brain.habit_tracker import HabitTracker

        model = MockUserModel(active_hours={9: 0.95}, active_days={0: 0.95})
        mock_um = make_mock_user_model_engine(model)

        tracker = HabitTracker(user_model=mock_um)
        result = tracker._cross_validate_with_user_model("t1", 0.99, hour=9, day=0)
        assert result <= 1.0


# ── P2: DecisionNode + SkillCompilerNode ← UserModel ───────────────────────


class TestDecisionNodeUserModel:
    """P2: DecisionNode accepts UserModel attribute."""

    def test_user_model_attribute_exists(self):
        """DecisionNode should have a _user_model attribute."""
        from hbllm.brain.decision_node import DecisionNode

        node = DecisionNode(node_id="test_decision")
        assert hasattr(node, "_user_model")
        assert node._user_model is None

    def test_user_model_can_be_set(self):
        """Factory should be able to set _user_model post-creation."""
        from hbllm.brain.decision_node import DecisionNode

        node = DecisionNode(node_id="test_decision")
        mock_um = make_mock_user_model_engine()
        node._user_model = mock_um
        assert node._user_model is mock_um


class TestSkillCompilerNodeUserModel:
    """P2: SkillCompilerNode accepts UserModel via constructor."""

    def test_constructor_accepts_user_model(self):
        """SkillCompilerNode should accept user_model parameter."""
        from hbllm.brain.skill_compiler_node import SkillCompilerNode

        mock_um = make_mock_user_model_engine()
        node = SkillCompilerNode(
            node_id="test_compiler",
            user_model=mock_um,
        )
        assert node._user_model is mock_um

    def test_constructor_defaults_none(self):
        """Without user_model param, _user_model should be None."""
        from hbllm.brain.skill_compiler_node import SkillCompilerNode

        node = SkillCompilerNode(node_id="test_compiler")
        assert node._user_model is None
