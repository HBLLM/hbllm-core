"""Unit tests for brain modules — value_system, source_verifier, activity_digest,
confidence_estimator, persona_engine, habit_tracker."""

import time

from hbllm.brain.value_system import (
    DynamicValueArbitrator,
    InterruptionPenaltyPolicy,
    ResourceConservationPolicy,
    UrgencyOverridePolicy,
)


class TestValueSystem:
    """Test utility policy and value arbitration."""

    def test_resource_conservation_high_usage(self):
        policy = ResourceConservationPolicy()
        result = policy.apply_modifiers(1.0, {"resource_usage": 0.95})
        assert isinstance(result, float)

    def test_resource_conservation_low_usage(self):
        policy = ResourceConservationPolicy()
        result = policy.apply_modifiers(1.0, {"resource_usage": 0.1})
        assert isinstance(result, float)

    def test_urgency_override(self):
        policy = UrgencyOverridePolicy()
        result = policy.apply_modifiers(0.5, {"urgency": "high"})
        assert isinstance(result, float)

    def test_interruption_penalty_true(self):
        policy = InterruptionPenaltyPolicy()
        result = policy.apply_modifiers(1.0, {"is_interruption": True})
        assert isinstance(result, float)

    def test_interruption_penalty_false(self):
        policy = InterruptionPenaltyPolicy()
        result = policy.apply_modifiers(1.0, {"is_interruption": False})
        assert isinstance(result, float)

    def test_arbitrator_no_policies(self):
        arb = DynamicValueArbitrator(policies=[])
        result = arb.compute_utility(1.0, {})
        assert result == 1.0

    def test_arbitrator_chains(self):
        arb = DynamicValueArbitrator(
            policies=[ResourceConservationPolicy(), UrgencyOverridePolicy()]
        )
        result = arb.compute_utility(1.0, {"resource_usage": 0.5})
        assert isinstance(result, float)

    def test_arbitrator_default(self):
        arb = DynamicValueArbitrator()
        assert len(arb.policies) > 0


# ── Source Verifier ──────────────────────────────────────────────────────────

from hbllm.brain.source_verifier import SourceCredibility, SourceVerifier


class TestSourceVerifier:
    def test_init(self):
        sv = SourceVerifier(min_trust_score=0.5)
        assert sv.min_trust_score == 0.5

    def test_get_domain_reputation_known(self):
        sv = SourceVerifier()
        score, tier = sv.get_domain_reputation("https://arxiv.org/paper/123")
        assert isinstance(score, float)
        assert isinstance(tier, str)

    def test_get_domain_reputation_unknown(self):
        sv = SourceVerifier()
        score, tier = sv.get_domain_reputation("https://random-blog-xyz.com/post")
        assert isinstance(score, float)

    def test_compute_corroboration(self):
        sv = SourceVerifier()
        # claim is a string, all_results is a list of dicts
        score = sv.compute_corroboration("test content about Python", [])
        assert isinstance(score, float)

    def test_compute_recency(self):
        sv = SourceVerifier()
        score = sv.compute_recency("Updated January 2024")
        assert isinstance(score, float)

    def test_verify_source(self):
        sv = SourceVerifier()
        result = {"url": "https://arxiv.org/paper/1", "snippet": "AI research"}
        cred = sv.verify_source(result, all_results=[result])
        assert isinstance(cred, SourceCredibility)
        assert isinstance(cred.trust_score, float)

    def test_verify_sources(self):
        sv = SourceVerifier()
        results = [
            {"url": "https://arxiv.org/paper/1", "snippet": "AI research"},
            {"url": "https://example.com/blog", "snippet": "Random blog"},
        ]
        creds = sv.verify_sources(results)
        assert len(creds) == 2

    def test_credibility_to_dict(self):
        cred = SourceCredibility(
            url="https://example.com",
            domain="example.com",
            trust_score=0.7,
            domain_reputation=0.6,
            corroboration_score=0.5,
            recency_score=0.8,
            is_trusted=True,
        )
        d = cred.to_dict()
        assert d["url"] == "https://example.com"

    def test_extract_domain(self):
        sv = SourceVerifier()
        domain = sv._extract_domain("https://www.example.com/path")
        assert "example" in domain


# ── Activity Digest ──────────────────────────────────────────────────────────

from hbllm.brain.activity_digest import ActivityDigestEngine, Digest, DigestItem


class TestActivityDigest:
    def test_digest_item_creation(self):
        item = DigestItem(category="query", title="User asked about Python")
        assert item.category == "query"

    def test_digest_item_to_dict(self):
        item = DigestItem(category="action", title="Executed search")
        d = item.to_dict()
        assert "category" in d

    def test_digest_is_empty(self):
        digest = Digest(tenant_id="t1", items=[])
        assert digest.is_empty is True

    def test_digest_not_empty(self):
        digest = Digest(
            tenant_id="t1",
            items=[DigestItem("query", "test")],
        )
        assert digest.is_empty is False

    def test_digest_duration_hours(self):
        now = time.time()
        digest = Digest(
            tenant_id="t1", items=[], period_start=now - 7200, period_end=now
        )
        assert abs(digest.duration_hours - 2.0) < 0.1

    def test_digest_to_natural_language(self):
        digest = Digest(
            tenant_id="t1",
            items=[DigestItem("query", "Asked about Python")],
        )
        text = digest.to_natural_language()
        assert isinstance(text, str)

    def test_digest_to_dict(self):
        digest = Digest(tenant_id="t1", items=[])
        d = digest.to_dict()
        assert "tenant_id" in d

    def test_engine_record_event(self):
        engine = ActivityDigestEngine()
        item = DigestItem("query", "Test query")
        engine.record_event("tenant1", item)

    def test_engine_record_activity(self):
        engine = ActivityDigestEngine()
        engine.record_activity("tenant1")

    def test_engine_absence_duration(self):
        engine = ActivityDigestEngine()
        duration = engine.get_absence_duration("tenant1")
        assert isinstance(duration, float)


# ── Confidence Estimator ─────────────────────────────────────────────────────

from hbllm.brain.confidence_estimator import ConfidenceEstimator, ConfidenceReport


class TestConfidenceEstimator:
    def test_init(self):
        est = ConfidenceEstimator()
        assert est is not None

    def test_estimate_returns_report(self):
        est = ConfidenceEstimator()
        report = est.estimate(
            query="What is Python?",
            response="Python is a high-level programming language.",
        )
        assert isinstance(report, ConfidenceReport)
        assert isinstance(report.overall, float)

    def test_score_returns_float(self):
        est = ConfidenceEstimator()
        score = est.score(
            query="What is AI?",
            response="Artificial Intelligence is the simulation of human intelligence.",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_relevance(self):
        est = ConfidenceEstimator()
        score = est._score_relevance("Python", "Python is a language.")
        assert isinstance(score, float)

    def test_score_coherence(self):
        est = ConfidenceEstimator()
        score = est._score_coherence("This is a well-written response.")
        assert isinstance(score, float)

    def test_score_factuality_risk(self):
        est = ConfidenceEstimator()
        score = est._score_factuality_risk("I think this might be correct.")
        assert isinstance(score, float)

    def test_empty_response(self):
        est = ConfidenceEstimator()
        report = est.estimate(query="test", response="")
        assert isinstance(report, ConfidenceReport)


# ── Habit Tracker ────────────────────────────────────────────────────────────

from hbllm.brain.habit_tracker import HabitPattern, HabitTracker, InteractionEvent


class TestHabitTracker:
    def test_interaction_event(self):
        event = InteractionEvent(
            tenant_id="t1", timestamp=time.time(), action_type="query", topic="python"
        )
        assert event.tenant_id == "t1"
        assert event.action_type == "query"

    def test_event_hour(self):
        event = InteractionEvent(
            tenant_id="t1", timestamp=time.time(), action_type="query"
        )
        assert 0 <= event.hour <= 23

    def test_event_weekday(self):
        event = InteractionEvent(
            tenant_id="t1", timestamp=time.time(), action_type="query"
        )
        assert 0 <= event.weekday <= 6

    def test_habit_pattern(self):
        pattern = HabitPattern(
            id="p1",
            tenant_id="t1",
            description="Active between 9-17",
            occurrence_count=10,
            first_seen=time.time() - 86400,
            last_seen=time.time(),
        )
        assert pattern.occurrence_count == 10

    def test_habit_pattern_frequency(self):
        now = time.time()
        pattern = HabitPattern(
            id="p1",
            tenant_id="t1",
            description="test",
            occurrence_count=10,
            first_seen=now - 86400,
            last_seen=now,
        )
        freq = pattern.frequency
        assert isinstance(freq, float)

    def test_habit_pattern_to_dict(self):
        pattern = HabitPattern(
            id="p1", tenant_id="t1", description="test", occurrence_count=5
        )
        d = pattern.to_dict()
        assert "tenant_id" in d

    def test_tracker_init(self):
        tracker = HabitTracker()
        assert tracker is not None

    def test_tracker_record_event(self):
        tracker = HabitTracker()
        event = InteractionEvent(
            tenant_id="t1", timestamp=time.time(), action_type="query", topic="python"
        )
        tracker.record_event(event)

    def test_tracker_multiple_events(self):
        tracker = HabitTracker()
        for i in range(10):
            event = InteractionEvent(
                tenant_id="t1",
                timestamp=time.time() - i * 3600,
                action_type="query",
                topic="python",
            )
            tracker.record_event(event)


# ── Persona Engine ───────────────────────────────────────────────────────────

from hbllm.brain.persona_engine import PersonaEngine, PersonaProfile, PersonaTrait


class TestPersonaEngine:
    def test_trait_creation(self):
        trait = PersonaTrait(name="formality", value=0.7)
        assert trait.name == "formality"

    def test_trait_nudge(self):
        trait = PersonaTrait(name="formality", value=0.5)
        trait.nudge(0.1)
        assert trait.value != 0.5

    def test_trait_nudge_clamps(self):
        trait = PersonaTrait(name="formality", value=0.95)
        trait.nudge(1.0, learning_rate=1.0)
        assert trait.value <= 1.0

    def test_profile_creation(self):
        profile = PersonaProfile(tenant_id="t1")
        assert profile.tenant_id == "t1"

    def test_profile_traits(self):
        profile = PersonaProfile(tenant_id="t1")
        traits = profile.traits
        assert isinstance(traits, dict)

    def test_profile_to_dict(self):
        profile = PersonaProfile(tenant_id="t1")
        d = profile.to_dict()
        assert "tenant_id" in d

    def test_profile_from_dict(self):
        profile = PersonaProfile(tenant_id="t1")
        d = profile.to_dict()
        restored = PersonaProfile.from_dict(d)
        assert restored.tenant_id == "t1"

    def test_profile_system_prompt(self):
        profile = PersonaProfile(tenant_id="t1")
        fragment = profile.to_system_prompt_fragment()
        assert isinstance(fragment, str)

    def test_engine_init(self, tmp_path):
        engine = PersonaEngine(storage_dir=tmp_path / "personas")
        assert engine is not None

    def test_engine_get_profile(self, tmp_path):
        engine = PersonaEngine(storage_dir=tmp_path / "personas")
        profile = engine.get_profile("t1")
        assert isinstance(profile, PersonaProfile)
