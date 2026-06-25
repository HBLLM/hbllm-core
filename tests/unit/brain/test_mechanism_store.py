"""Tests for the Mechanism Memory System."""

from __future__ import annotations

import pytest

from hbllm.brain.mechanism_store import Mechanism, MechanismStore


@pytest.fixture
def store(tmp_path):
    return MechanismStore(data_dir=str(tmp_path))


class TestMechanismCRUD:
    """Basic create, read, update, delete operations."""

    def test_create_and_get(self, store):
        mech = store.create(
            description="Package Manager Dependency Resolution",
            preconditions=["package manager installed", "internet access"],
            process_steps=["resolve dependencies", "download packages", "install"],
            expected_outcomes=["packages installed", "dependencies satisfied"],
            domain="devops",
        )
        assert mech.id.startswith("mech_")
        assert mech.confidence == 0.8
        assert mech.domain == "devops"

        retrieved = store.get(mech.id)
        assert retrieved is not None
        assert retrieved.description == "Package Manager Dependency Resolution"
        assert retrieved.preconditions == ["package manager installed", "internet access"]
        assert retrieved.expected_outcomes == ["packages installed", "dependencies satisfied"]

    def test_get_nonexistent(self, store):
        assert store.get("nonexistent") is None

    def test_store_and_update(self, store):
        mech = store.create(
            description="Test Mechanism",
            preconditions=["a"],
            process_steps=["b"],
            expected_outcomes=["c"],
        )
        mech.confidence = 0.95
        mech.usage_count = 5
        store.store(mech)

        retrieved = store.get(mech.id)
        assert retrieved.confidence == 0.95
        assert retrieved.usage_count == 5


class TestMechanismSearch:
    """Search by domain and preconditions."""

    def test_find_by_domain(self, store):
        store.create(
            description="Auth check",
            preconditions=["credentials"],
            process_steps=["validate"],
            expected_outcomes=["access granted"],
            domain="security",
        )
        store.create(
            description="Deploy app",
            preconditions=["container"],
            process_steps=["push"],
            expected_outcomes=["app running"],
            domain="devops",
        )

        security = store.find_by_domain("security")
        assert len(security) == 1
        assert security[0].description == "Auth check"

        devops = store.find_by_domain("devops")
        assert len(devops) == 1
        assert devops[0].description == "Deploy app"

    def test_find_by_preconditions(self, store):
        store.create(
            description="SQL Injection Prevention",
            preconditions=["user input", "database query", "web application"],
            process_steps=["sanitize input", "parameterize query"],
            expected_outcomes=["safe query execution"],
            domain="security",
        )
        store.create(
            description="Caching Strategy",
            preconditions=["frequent reads", "redis installed"],
            process_steps=["check cache", "fallback to db"],
            expected_outcomes=["faster response"],
            domain="performance",
        )

        # Should match SQL injection by "user input" keywords
        results = store.find_by_preconditions(["user", "input", "form"])
        assert len(results) >= 1
        assert results[0].description == "SQL Injection Prevention"

        # Should match caching by "redis" keyword
        results = store.find_by_preconditions(["redis", "cache"])
        assert len(results) >= 1
        assert results[0].description == "Caching Strategy"

    def test_find_by_preconditions_empty(self, store):
        results = store.find_by_preconditions(["completely", "unrelated", "words"])
        assert results == []


class TestMechanismConfidence:
    """Confidence management: decay, reinforce, record usage."""

    def test_record_success(self, store):
        mech = store.create(
            description="Test", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.8,
        )
        store.record_usage(mech.id, success=True)

        updated = store.get(mech.id)
        assert updated.usage_count == 1
        assert updated.success_count == 1
        assert updated.confidence > 0.8  # Reinforced

    def test_record_failure(self, store):
        mech = store.create(
            description="Test", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.8,
        )
        store.record_usage(mech.id, success=False)

        updated = store.get(mech.id)
        assert updated.usage_count == 1
        assert updated.failure_count == 1
        assert updated.confidence < 0.8  # Penalized

    def test_reinforce(self, store):
        mech = store.create(
            description="Test", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.5,
        )
        store.reinforce(mech.id, confidence_boost=0.2)
        assert store.get(mech.id).confidence == pytest.approx(0.7)

    def test_decay_confidence(self, store):
        store.create(
            description="Regular", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.5,
        )
        core = store.create(
            description="Core", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.9,
        )
        store.promote_to_core(core.id)

        decayed = store.decay_confidence(rate=0.1)
        assert decayed >= 1  # Regular decayed

        regular = store.find_by_domain("general")
        non_core = [m for m in regular if not m.is_core]
        assert all(m.confidence <= 0.5 for m in non_core)

        # Core should NOT be decayed
        core_refreshed = store.get(core.id)
        assert core_refreshed.confidence == 0.9


class TestMechanismPromotion:
    """Core mechanism promotion and pruning."""

    def test_promote_to_core(self, store):
        mech = store.create(
            description="HTTP Request Pattern",
            preconditions=["api endpoint"],
            process_steps=["build request", "send", "parse response"],
            expected_outcomes=["response received"],
        )
        assert not store.get(mech.id).is_core

        store.promote_to_core(mech.id)
        assert store.get(mech.id).is_core

    def test_get_core_mechanisms(self, store):
        m1 = store.create(
            description="Core 1", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
        )
        store.create(
            description="Non-core", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
        )
        store.promote_to_core(m1.id)

        cores = store.get_core_mechanisms()
        assert len(cores) == 1
        assert cores[0].id == m1.id

    def test_find_promotable(self, store):
        mech = store.create(
            description="Well-used", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.9,
        )
        # Simulate 15 successful uses
        for _ in range(15):
            store.record_usage(mech.id, success=True)

        promotable = store.find_promotable(min_usage=10, min_success_rate=0.9)
        assert len(promotable) == 1
        assert promotable[0].id == mech.id

    def test_prune_weak(self, store):
        weak = store.create(
            description="Weak", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.05,
        )
        strong = store.create(
            description="Strong", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.9,
        )

        pruned = store.prune_weak(max_confidence=0.1)
        assert pruned == 1
        assert store.get(weak.id) is None
        assert store.get(strong.id) is not None

    def test_get_weak(self, store):
        store.create(
            description="Weak", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.2,
        )
        store.create(
            description="Strong", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            confidence=0.9,
        )

        weak = store.get_weak(threshold=0.3)
        assert len(weak) == 1
        assert weak[0].description == "Weak"


class TestMechanismStats:
    """Stats and serialization."""

    def test_stats(self, store):
        store.create(
            description="A", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            domain="security",
        )
        store.create(
            description="B", preconditions=["a"],
            process_steps=["b"], expected_outcomes=["c"],
            domain="devops",
        )

        stats = store.stats()
        assert stats["total_mechanisms"] == 2
        assert "security" in stats["by_domain"]
        assert "devops" in stats["by_domain"]

    def test_to_dict(self, store):
        mech = store.create(
            description="Test",
            preconditions=["a", "b"],
            process_steps=["c"],
            expected_outcomes=["d"],
            domain="test",
        )
        d = mech.to_dict()
        assert d["id"] == mech.id
        assert d["description"] == "Test"
        assert d["preconditions"] == ["a", "b"]
        assert d["is_core"] is False

    def test_success_rate_property(self):
        mech = Mechanism(
            id="test",
            description="test",
            preconditions=[],
            process_steps=[],
            expected_outcomes=[],
            success_count=9,
            failure_count=1,
        )
        assert mech.success_rate == pytest.approx(0.9)

    def test_success_rate_zero_uses(self):
        mech = Mechanism(
            id="test",
            description="test",
            preconditions=[],
            process_steps=[],
            expected_outcomes=[],
        )
        assert mech.success_rate == 1.0
