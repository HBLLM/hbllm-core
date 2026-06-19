"""Tests for HabitTracker — temporal pattern mining."""

import time

import pytest

from hbllm.brain.habit_tracker import HabitTracker, InteractionEvent


class TestInteractionEvent:
    def test_hour_and_weekday(self):
        ts = time.time()
        ev = InteractionEvent(tenant_id="t1", timestamp=ts, action_type="query")
        assert 0 <= ev.hour <= 23
        assert 0 <= ev.weekday <= 6


class TestHabitTracker:
    def _make_events(self, tracker, tenant_id, action, n, hour=9):
        """Generate n events at a specific hour to force a pattern."""
        import calendar
        import datetime

        # Use a fixed date to make events predictable
        base = datetime.datetime(2026, 6, 16, hour, 0, 0)  # Monday
        for i in range(n):
            ts = calendar.timegm(base.timetuple()) + i * 60
            tracker.record_event(
                InteractionEvent(
                    tenant_id=tenant_id,
                    timestamp=ts,
                    action_type=action,
                    topic="test_topic",
                )
            )

    def test_record_event(self):
        tracker = HabitTracker(min_observations=3)
        tracker.record_event(
            InteractionEvent(tenant_id="t1", timestamp=time.time(), action_type="query")
        )
        stats = tracker.stats("t1")
        assert stats["total_events"] == 1

    def test_no_habits_with_few_events(self):
        tracker = HabitTracker(min_observations=10)
        for i in range(3):
            tracker.record_event(
                InteractionEvent(tenant_id="t1", timestamp=time.time(), action_type="query")
            )
        habits = tracker.get_habits("t1")
        assert len(habits) == 0

    def test_topic_pattern_detection(self):
        tracker = HabitTracker(min_observations=3, confidence_threshold=0.1)
        # Generate concentrated events with a topic
        for i in range(20):
            tracker.record_event(
                InteractionEvent(
                    tenant_id="t1",
                    timestamp=time.time() + i,
                    action_type="query",
                    topic="email",
                )
            )
        habits = tracker.get_habits("t1")
        topic_habits = [h for h in habits if h.topic == "email"]
        assert len(topic_habits) >= 1

    def test_suppress_habit(self):
        tracker = HabitTracker(min_observations=3, confidence_threshold=0.1)
        for i in range(20):
            tracker.record_event(
                InteractionEvent(
                    tenant_id="t1",
                    timestamp=time.time() + i,
                    action_type="query",
                    topic="email",
                )
            )
        habits = tracker.get_habits("t1")
        if habits:
            tracker.suppress_habit("t1", habits[0].id)
            active = tracker.get_habits("t1")
            assert len(active) < len(habits)

    def test_suppressed_not_in_suggestions(self):
        tracker = HabitTracker(min_observations=3, confidence_threshold=0.1)
        for i in range(20):
            tracker.record_event(
                InteractionEvent(
                    tenant_id="t1",
                    timestamp=time.time() + i,
                    action_type="query",
                    topic="email",
                )
            )
        habits = tracker.get_habits("t1")
        for h in habits:
            tracker.suppress_habit("t1", h.id)
        assert tracker.get_habits("t1", include_suppressed=False) == []

    def test_stats(self):
        tracker = HabitTracker()
        tracker.record_event(
            InteractionEvent(tenant_id="t1", timestamp=time.time(), action_type="query")
        )
        stats = tracker.stats("t1")
        assert "total_events" in stats
        assert "total_habits" in stats
        assert "active_habits" in stats

    def test_event_eviction(self):
        tracker = HabitTracker(max_events_per_tenant=5)
        for i in range(10):
            tracker.record_event(
                InteractionEvent(tenant_id="t1", timestamp=time.time() + i, action_type="query")
            )
        assert tracker.stats("t1")["total_events"] <= 5

    def test_habit_to_dict(self):
        tracker = HabitTracker(min_observations=3, confidence_threshold=0.1)
        for i in range(20):
            tracker.record_event(
                InteractionEvent(
                    tenant_id="t1",
                    timestamp=time.time() + i,
                    action_type="query",
                    topic="coding",
                )
            )
        habits = tracker.get_habits("t1")
        if habits:
            d = habits[0].to_dict()
            assert "id" in d
            assert "confidence" in d
            assert "frequency" in d
