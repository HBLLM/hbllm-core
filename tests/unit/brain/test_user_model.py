"""Unit tests for UserModel — predictive user simulator."""

import os
import time
import tempfile
import pytest

from hbllm.brain.user_model import (
    LearnedAttribute,
    UserBelief,
    UserExpertise,
    UserModel,
    UserModelEngine,
    UserPreference,
    TrustDimension,
)


class TestLearnedAttribute:
    def test_defaults(self):
        attr = LearnedAttribute(value="test")
        assert attr.value == "test"
        assert attr.confidence == 0.3
        assert attr.evidence_count == 0
        assert attr.source == "inferred"

    def test_update_inferred(self):
        attr = LearnedAttribute(value=0.5)
        attr.update(0.8)
        assert attr.evidence_count == 1
        # 1 - exp(-1/5) ≈ 0.18 — confidence starts low with little evidence
        assert attr.confidence > 0.1
        assert attr.source == "inferred"
        # EWMA: 0.2 * 0.8 + 0.8 * 0.5 = 0.56
        assert abs(attr.value - 0.56) < 0.01

    def test_update_explicit(self):
        attr = LearnedAttribute(value="old")
        attr.update("new", source="explicit")
        assert attr.value == "new"
        assert attr.confidence == 0.95  # Explicit overrides formula
        assert attr.source == "corrected"

    def test_confidence_grows_with_evidence(self):
        attr = LearnedAttribute(value=0.5)
        for _ in range(10):
            attr.update(0.6)
        assert attr.confidence > 0.8

    def test_confidence_caps_at_1(self):
        attr = LearnedAttribute(value=0.5)
        for _ in range(100):
            attr.update(0.6)
        assert attr.confidence <= 1.0

    def test_decay(self):
        attr = LearnedAttribute(value="test", confidence=0.9)
        # Simulate 30 days old
        attr.last_observed = time.time() - (30 * 86400)
        attr.decay(half_life_days=30.0)
        assert attr.confidence < 0.9
        assert attr.confidence > 0.0

    def test_decay_no_effect_when_fresh(self):
        attr = LearnedAttribute(value="test", confidence=0.9)
        attr.last_observed = time.time()  # Just now
        old_conf = attr.confidence
        attr.decay()
        assert attr.confidence == old_conf

    def test_to_dict(self):
        attr = LearnedAttribute(value="test", confidence=0.75, evidence_count=5)
        d = attr.to_dict()
        assert d["value"] == "test"
        assert d["confidence"] == 0.75
        assert d["evidence_count"] == 5

    def test_from_dict(self):
        d = {
            "value": "python",
            "confidence": 0.8,
            "evidence_count": 10,
            "first_observed": 1000.0,
            "last_observed": 2000.0,
            "source": "inferred",
        }
        attr = LearnedAttribute.from_dict(d)
        assert attr.value == "python"
        assert attr.confidence == 0.8
        assert attr.evidence_count == 10

    def test_update_string_value(self):
        attr = LearnedAttribute(value="initial")
        attr.update("replaced")
        assert attr.value == "replaced"
        assert attr.evidence_count == 1


class TestUserBelief:
    def test_defaults(self):
        belief = UserBelief(topic="AGI", stance="important")
        assert belief.confidence == 0.5
        assert belief.evidence_count == 0

    def test_reinforce(self):
        belief = UserBelief(topic="AGI", stance="brain architectures matter")
        belief.reinforce()
        assert belief.evidence_count == 1
        # After reinforce: 1 - exp(-1/3) ≈ 0.28
        # With initial confidence of 0.5, reinforcing doesn't necessarily increase
        # since the formula recalculates from evidence_count alone
        assert belief.confidence > 0.0

    def test_reinforce_multiple(self):
        belief = UserBelief(topic="testing", stance="essential")
        for _ in range(10):
            belief.reinforce()
        assert belief.confidence > 0.9
        assert belief.evidence_count == 10

    def test_to_dict(self):
        belief = UserBelief(topic="AI", stance="positive")
        d = belief.to_dict()
        assert d["topic"] == "AI"
        assert d["stance"] == "positive"


class TestTrustDimension:
    def test_delegation_increases_trust(self):
        trust = TrustDimension(
            domain="coding",
            trust_level=LearnedAttribute(value=0.5),
        )
        trust.record_delegation()
        assert float(trust.trust_level.value) > 0.5
        assert trust.delegations_count == 1

    def test_override_decreases_trust(self):
        trust = TrustDimension(
            domain="coding",
            trust_level=LearnedAttribute(value=0.5),
        )
        trust.record_override()
        assert float(trust.trust_level.value) < 0.5
        assert trust.overrides_count == 1
        # Override adds 0.1 to confidence after evidence formula
        # Evidence formula gives ~0.18, then +0.1 = ~0.28
        assert trust.trust_level.confidence > 0.2

    def test_trust_bounded(self):
        trust = TrustDimension(
            domain="coding",
            trust_level=LearnedAttribute(value=0.95),
        )
        for _ in range(10):
            trust.record_delegation()
        assert float(trust.trust_level.value) <= 1.0


class TestUserModel:
    def test_defaults(self):
        model = UserModel(tenant_id="test")
        assert model.tenant_id == "test"
        assert len(model.expertise) == 0
        assert len(model.beliefs) == 0
        assert model.stress_level == 0.0

    def test_to_dict(self):
        model = UserModel(tenant_id="test")
        model.expertise["python"] = UserExpertise(
            domain="python",
            level=LearnedAttribute(value=0.8),
        )
        d = model.to_dict()
        assert d["tenant_id"] == "test"
        assert "python" in d["expertise"]


class TestUserModelEngine:
    @pytest.fixture
    def engine(self, tmp_path):
        return UserModelEngine(data_dir=str(tmp_path))

    def test_get_model_creates_new(self, engine):
        model = engine.get_model("tenant1")
        assert model.tenant_id == "tenant1"
        assert len(model.expertise) == 0

    def test_get_model_returns_cached(self, engine):
        m1 = engine.get_model("tenant1")
        m2 = engine.get_model("tenant1")
        assert m1 is m2

    def test_update_from_interaction_python(self, engine):
        changed = engine.update_from_interaction(
            tenant_id="test",
            query="I'm using asyncio with pydantic dataclass in my fastapi project",
        )
        model = engine.get_model("test")
        assert "python" in model.expertise
        assert float(model.expertise["python"].level.value) > 0

    def test_update_from_interaction_flutter(self, engine):
        engine.update_from_interaction(
            tenant_id="test",
            query="the widget uses stateful bloc provider with riverpod",
        )
        model = engine.get_model("test")
        assert "flutter" in model.expertise

    def test_update_from_interaction_laravel(self, engine):
        engine.update_from_interaction(
            tenant_id="test",
            query="use eloquent with artisan migration and nova dashboard",
        )
        model = engine.get_model("test")
        assert "laravel" in model.expertise

    def test_update_focus(self, engine):
        engine.update_from_interaction(
            tenant_id="test",
            query="working on HBLLM architecture",
            metadata={"topic": "HBLLM"},
        )
        model = engine.get_model("test")
        assert model.current_focus.value == "HBLLM"

    def test_learn_preference_explicit(self, engine):
        engine.learn_preference("test", "verbosity", "concise", source="explicit")
        model = engine.get_model("test")
        assert "verbosity" in model.preferences
        assert model.preferences["verbosity"].learned.value == "concise"
        assert model.preferences["verbosity"].learned.confidence == 0.95

    def test_learn_preference_inferred(self, engine):
        engine.learn_preference("test", "style", "technical", source="inferred")
        model = engine.get_model("test")
        assert model.preferences["style"].learned.confidence == 0.3

    def test_record_belief(self, engine):
        engine.record_belief("test", "AGI", "brain-inspired architectures matter")
        model = engine.get_model("test")
        assert len(model.beliefs) == 1
        assert model.beliefs[0].topic == "AGI"
        assert model.beliefs[0].stance == "brain-inspired architectures matter"

    def test_record_belief_reinforcement(self, engine):
        engine.record_belief("test", "testing", "essential")
        engine.record_belief("test", "testing", "essential")
        model = engine.get_model("test")
        assert len(model.beliefs) == 1
        assert model.beliefs[0].evidence_count == 1

    def test_update_trust_delegation(self, engine):
        engine.update_trust("test", "coding", delegated=True)
        model = engine.get_model("test")
        assert "coding" in model.trust
        assert float(model.trust["coding"].trust_level.value) > 0.5

    def test_update_trust_override(self, engine):
        engine.update_trust("test", "infrastructure", overridden=True)
        model = engine.get_model("test")
        assert float(model.trust["infrastructure"].trust_level.value) < 0.5

    def test_update_stress(self, engine):
        engine.update_stress("test", 0.8)
        model = engine.get_model("test")
        assert model.stress_level == 0.8

    def test_update_engagement(self, engine):
        engine.update_engagement("test", 0.3)
        model = engine.get_model("test")
        assert model.engagement_level == 0.3

    def test_stress_clamped(self, engine):
        engine.update_stress("test", 1.5)
        model = engine.get_model("test")
        assert model.stress_level == 1.0

    @pytest.mark.asyncio
    async def test_get_context_empty(self, engine):
        ctx = await engine.get_context("hello", "empty_tenant", 100)
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_get_context_with_data(self, engine):
        engine.update_from_interaction(
            "test",
            "using asyncio pydantic fastapi in my python project",
            metadata={"topic": "python"},
        )
        engine.learn_preference("test", "style", "technical", source="explicit")
        engine.update_stress("test", 0.8)
        ctx = await engine.get_context("python question", "test", 500)
        # Should have at least preferences and stress warning
        assert len(ctx) > 0
        assert "technical" in ctx.lower() or "stress" in ctx.lower()

    def test_predict_next_actions(self, engine):
        predictions = engine.predict_next_actions("test")
        assert isinstance(predictions, list)

    def test_snapshot(self, engine):
        engine.update_from_interaction(
            "test", "asyncio pydantic test"
        )
        snap = engine.snapshot("test")
        assert snap["tenant_id"] == "test"
        assert "expertise" in snap

    def test_stats(self, engine):
        engine.update_from_interaction("test", "asyncio test")
        stats = engine.stats("test")
        assert "expertise_domains" in stats

    def test_persistence_round_trip(self, tmp_path):
        # Create and save
        engine1 = UserModelEngine(data_dir=str(tmp_path))
        engine1.update_from_interaction("test", "asyncio pydantic fastapi mypy")
        engine1.learn_preference("test", "color", "blue", source="explicit")
        engine1.record_belief("test", "AI", "transformative")

        # Load in new engine
        engine2 = UserModelEngine(data_dir=str(tmp_path))
        model = engine2.get_model("test")
        assert "python" in model.expertise
        assert "color" in model.preferences
        assert len(model.beliefs) == 1
        assert model.beliefs[0].topic == "AI"

    def test_temporal_patterns(self, engine):
        engine.update_from_interaction("test", "some query")
        model = engine.get_model("test")
        assert len(model.active_hours) > 0
        assert len(model.active_days) > 0

    def test_beliefs_cap(self, engine):
        for i in range(60):
            engine.record_belief("test", f"topic_{i}", f"stance_{i}")
        model = engine.get_model("test")
        assert len(model.beliefs) <= 50

    def test_interests_cap(self, engine):
        for i in range(25):
            engine.update_from_interaction(
                "test", f"topic_{i}", metadata={"topic": f"unique_topic_{i}"}
            )
        model = engine.get_model("test")
        assert len(model.active_interests) <= 20
