"""Tests for ActivityDigest — missed activity summarization."""

import time

from hbllm.brain.activity_digest import ActivityDigestEngine, Digest, DigestItem


class TestDigestItem:
    def test_defaults(self):
        item = DigestItem(category="goal", title="Test")
        assert item.importance == 0.5
        assert item.timestamp > 0

    def test_to_dict(self):
        item = DigestItem(category="system", title="Alert", importance=0.9)
        d = item.to_dict()
        assert d["category"] == "system"
        assert d["importance"] == 0.9


class TestDigest:
    def test_empty_digest(self):
        d = Digest(tenant_id="t1")
        assert d.is_empty
        assert "Nothing significant" in d.to_natural_language()

    def test_digest_with_items(self):
        d = Digest(
            tenant_id="t1",
            items=[
                DigestItem(category="goal", title="Goal completed: Deploy v2"),
                DigestItem(category="insight", title="Server load anomaly detected"),
            ],
            period_start=time.time() - 3600,
        )
        assert not d.is_empty
        text = d.to_natural_language()
        assert len(text) > 0

    def test_digest_to_dict(self):
        d = Digest(tenant_id="t1")
        result = d.to_dict()
        assert result["tenant_id"] == "t1"
        assert "summary" in result
        assert "item_count" in result

    def test_duration_hours(self):
        d = Digest(
            tenant_id="t1",
            period_start=time.time() - 7200,
            period_end=time.time(),
        )
        assert abs(d.duration_hours - 2.0) < 0.1

    def test_critical_items_first(self):
        d = Digest(
            tenant_id="t1",
            items=[
                DigestItem(category="system", title="Low priority", importance=0.3),
                DigestItem(category="system", title="Critical alert", importance=0.9),
            ],
            period_start=time.time() - 3600,
        )
        text = d.to_natural_language()
        assert "Critical alert" in text


class TestActivityDigestEngine:
    def test_record_event(self):
        engine = ActivityDigestEngine()
        engine.record_event("t1", DigestItem(category="goal", title="Test"))
        assert engine.pending_count("t1") == 1

    def test_generate_empty_digest(self):
        engine = ActivityDigestEngine()
        d = engine.generate_digest("t1")
        assert d.is_empty

    def test_generate_digest_with_events(self):
        engine = ActivityDigestEngine()
        engine.record_event("t1", DigestItem(category="goal", title="Goal completed: Test"))
        engine.record_event("t1", DigestItem(category="insight", title="Found pattern"))
        d = engine.generate_digest("t1")
        assert not d.is_empty
        assert len(d.items) == 2

    def test_digest_clears_buffer(self):
        engine = ActivityDigestEngine()
        engine.record_event("t1", DigestItem(category="system", title="Test"))
        engine.generate_digest("t1")
        assert engine.pending_count("t1") == 0

    def test_has_pending_events(self):
        engine = ActivityDigestEngine()
        assert not engine.has_pending_events("t1")
        engine.record_event("t1", DigestItem(category="system", title="Test"))
        assert engine.has_pending_events("t1")

    def test_record_activity(self):
        engine = ActivityDigestEngine()
        engine.record_activity("t1")
        assert engine.get_absence_duration("t1") < 1.0

    def test_stats(self):
        engine = ActivityDigestEngine()
        engine.record_event("t1", DigestItem(category="system", title="Test"))
        stats = engine.stats()
        assert stats["total_pending"] == 1
        assert stats["tenants_with_events"] == 1

    def test_max_items_limit(self):
        engine = ActivityDigestEngine()
        for i in range(30):
            engine.record_event("t1", DigestItem(category="system", title=f"Event {i}"))
        d = engine.generate_digest("t1", max_items=5)
        assert len(d.items) == 5

    def test_importance_ordering(self):
        engine = ActivityDigestEngine()
        engine.record_event("t1", DigestItem(category="system", title="Low", importance=0.1))
        engine.record_event("t1", DigestItem(category="system", title="High", importance=0.9))
        d = engine.generate_digest("t1")
        assert d.items[0].title == "High"
