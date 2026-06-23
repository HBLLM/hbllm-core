"""Unit tests for RelationshipMemory — KG-integrated social graph."""

import time
import pytest

from hbllm.brain.relationship_memory import (
    Person,
    RelationshipEvent,
    RelationshipHistory,
    RelationshipMemory,
    extract_person_mentions,
)


class TestExtractPersonMentions:
    def test_simple_names(self):
        names = extract_person_mentions("I had a meeting with Alice Chen today")
        assert "Alice Chen" in names

    def test_multiple_names(self):
        names = extract_person_mentions("John Smith and Jane Doe are collaborating")
        assert "John Smith" in names
        assert "Jane Doe" in names

    def test_filters_common_phrases(self):
        names = extract_person_mentions("Thank You for the help")
        assert "Thank You" not in names

    def test_filters_locations(self):
        names = extract_person_mentions("Meeting in New York")
        assert "New York" not in names

    def test_no_names(self):
        names = extract_person_mentions("just a regular sentence without names")
        assert len(names) == 0

    def test_single_word_ignored(self):
        # Single-word names (like "Alice") shouldn't be detected
        names = extract_person_mentions("Alice said hello")
        assert len(names) == 0  # Requires multi-word


class TestPerson:
    def test_defaults(self):
        person = Person(person_id="p1", name="Alice Chen")
        assert person.trust_level == 0.5
        assert person.importance == 0.5
        assert person.sentiment == 0.0
        assert person.mention_count == 0

    def test_to_dict(self):
        person = Person(
            person_id="p1",
            name="Alice Chen",
            role="manager",
            organization="Google",
        )
        d = person.to_dict()
        assert d["name"] == "Alice Chen"
        assert d["role"] == "manager"
        assert d["organization"] == "Google"


class TestRelationshipEvent:
    def test_defaults(self):
        event = RelationshipEvent(
            timestamp=time.time(),
            event_type="meeting",
            context="Weekly sync",
        )
        assert event.sentiment_delta == 0.0

    def test_to_dict(self):
        event = RelationshipEvent(
            timestamp=1000.0,
            event_type="positive_interaction",
            context="Good collaboration",
            sentiment_delta=0.3,
        )
        d = event.to_dict()
        assert d["event_type"] == "positive_interaction"
        assert d["sentiment_delta"] == 0.3


class TestRelationshipHistory:
    def test_compute_trend_stable(self):
        history = RelationshipHistory(person_id="p1", person_name="Alice")
        assert history.compute_trend() == "stable"

    def test_compute_trend_not_enough_data(self):
        events = [
            RelationshipEvent(timestamp=time.time(), event_type="mention", context="", sentiment_delta=0.5),
        ]
        history = RelationshipHistory(person_id="p1", person_name="Alice", events=events)
        assert history.compute_trend() == "stable"

    def test_compute_trend_improving(self):
        now = time.time()
        events = [
            # Old events (negative)
            RelationshipEvent(timestamp=now - 50 * 86400, event_type="conflict", context="", sentiment_delta=-0.5),
            RelationshipEvent(timestamp=now - 45 * 86400, event_type="conflict", context="", sentiment_delta=-0.3),
            # Recent events (positive)
            RelationshipEvent(timestamp=now - 5 * 86400, event_type="collaboration", context="", sentiment_delta=0.5),
            RelationshipEvent(timestamp=now - 2 * 86400, event_type="positive", context="", sentiment_delta=0.4),
        ]
        history = RelationshipHistory(person_id="p1", person_name="Alice", events=events)
        trend = history.compute_trend()
        assert trend == "improving"


class TestRelationshipMemory:
    @pytest.fixture
    def memory(self, tmp_path):
        return RelationshipMemory(knowledge_graph=None, data_dir=str(tmp_path))

    def test_record_mention_creates_person(self, memory):
        person = memory.record_mention("Alice Chen", context="in meeting")
        assert person.name == "Alice Chen"
        assert person.mention_count == 1

    def test_record_mention_increments_count(self, memory):
        memory.record_mention("Bob Smith")
        person = memory.record_mention("Bob Smith")
        assert person.mention_count == 2

    def test_record_mention_with_sentiment(self, memory):
        memory.record_mention("Alice Chen", sentiment=0.5)
        person = memory.get_person("Alice Chen")
        assert person.sentiment > 0.0

    def test_record_mention_with_topic(self, memory):
        memory.record_mention("Alice Chen", topic="architecture")
        person = memory.get_person("Alice Chen")
        assert "architecture" in person.topics_discussed

    def test_importance_grows_with_mentions(self, memory):
        for _ in range(10):
            memory.record_mention("Alice Chen")
        person = memory.get_person("Alice Chen")
        assert person.importance > 0.5

    def test_record_event(self, memory):
        memory.record_mention("Bob Smith")
        memory.record_event("Bob Smith", "collaboration", context="Pair programming", sentiment_delta=0.3)
        person = memory.get_person("Bob Smith")
        assert person.sentiment > 0.0

    def test_record_negative_event(self, memory):
        memory.record_mention("Bob Smith")
        memory.record_event("Bob Smith", "conflict", context="Disagreement", sentiment_delta=-0.5)
        person = memory.get_person("Bob Smith")
        assert person.sentiment < 0.0

    def test_learn_relationship(self, memory):
        memory.record_mention("Alice Chen")
        memory.record_mention("Bob Smith")
        memory.learn_relationship("Alice Chen", "Bob Smith", "colleague", context="same team")
        # Check roles updated
        alice = memory.get_person("Alice Chen")
        bob = memory.get_person("Bob Smith")
        assert alice.role == "colleague"
        assert bob.role == "colleague"

    def test_learn_manager_relationship(self, memory):
        memory.record_mention("Alice Chen")
        memory.record_mention("Bob Smith")
        memory.learn_relationship("Alice Chen", "Bob Smith", "reports_to")
        bob = memory.get_person("Bob Smith")
        assert bob.role == "manager"

    def test_get_person_exact(self, memory):
        memory.record_mention("Alice Chen")
        person = memory.get_person("Alice Chen")
        assert person is not None
        assert person.name == "Alice Chen"

    def test_get_person_fuzzy(self, memory):
        memory.record_mention("Alice Chen")
        person = memory.get_person("Alice")
        assert person is not None
        assert person.name == "Alice Chen"

    def test_get_person_not_found(self, memory):
        assert memory.get_person("Nobody") is None

    def test_get_network(self, memory):
        memory.record_mention("Alice Chen")
        memory.record_mention("Bob Smith")
        network = memory.get_network()
        assert len(network) == 2

    def test_get_relevant_people(self, memory):
        memory.record_mention("Alice Chen", topic="python")
        memory.record_mention("Bob Smith", topic="rust")
        relevant = memory.get_relevant_people("python")
        assert len(relevant) == 1
        assert relevant[0].name == "Alice Chen"

    def test_get_history(self, memory):
        memory.record_mention("Alice Chen")
        memory.record_event("Alice Chen", "meeting", context="Weekly sync")
        history = memory.get_history("Alice Chen")
        assert len(history.events) >= 1
        assert history.person_name == "Alice Chen"

    def test_get_history_nonexistent(self, memory):
        history = memory.get_history("Nobody")
        assert len(history.events) == 0

    def test_compute_trend(self, memory):
        memory.record_mention("Alice Chen")
        trend = memory.compute_trend("Alice Chen")
        assert trend in ("improving", "stable", "declining")

    def test_prioritize_notification_known(self, memory):
        for _ in range(5):
            memory.record_mention("Alice Chen")
        priority = memory.prioritize_notification("Alice Chen")
        assert 0.0 <= priority <= 1.0
        assert priority > 0.3

    def test_prioritize_notification_unknown(self, memory):
        priority = memory.prioritize_notification("Unknown Person")
        assert priority == 0.3

    @pytest.mark.asyncio
    async def test_get_context_empty(self, memory):
        ctx = await memory.get_context("hello world", "default", 100)
        # May return empty or key people
        assert isinstance(ctx, str)

    @pytest.mark.asyncio
    async def test_get_context_with_mention(self, memory):
        memory.record_mention("Alice Chen", topic="coding")
        ctx = await memory.get_context("Ask Alice Chen about the design", "default", 500)
        assert "Alice Chen" in ctx

    def test_stats(self, memory):
        memory.record_mention("Alice Chen")
        s = memory.stats()
        assert s["total_persons"] >= 1
        assert "total_events" in s

    def test_persistence_round_trip(self, tmp_path):
        # Create and populate
        mem1 = RelationshipMemory(data_dir=str(tmp_path))
        mem1.record_mention("Alice Chen", topic="python")
        mem1.record_event("Alice Chen", "collaboration", sentiment_delta=0.3)

        # Load in new instance
        mem2 = RelationshipMemory(data_dir=str(tmp_path))
        person = mem2.get_person("Alice Chen")
        assert person is not None
        assert person.name == "Alice Chen"
        assert "python" in person.topics_discussed
