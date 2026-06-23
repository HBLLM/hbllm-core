"""
RelationshipMemory — KG-integrated social graph with temporal history.

Models the user's relationships with other people: trust, importance,
frequency, sentiment, and how these change over time.

Architecture:
    - People are stored as KG entities (entity_type="person")
    - Relationships are KG relations (e.g., "reports_to", "friend")
    - Temporal history is stored in SQLite (events over time)
    - RelationshipMemory is a typed wrapper, not a separate store

Integration:
    - Reads from: KnowledgeGraph (entity/relation storage)
    - Writes to: KnowledgeGraph (person entities, relationship edges)
    - SQLite: relationship_events table (temporal history only)
    - Bus: subscribes system.experience, system.evaluation, calendar.event
    - Bus: publishes relationship.updated, relationship.new

Bus Topics:
    relationship.updated  → When a relationship quality changes
    relationship.new      → When a new person is detected
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────────


@dataclass
class Person:
    """A person in the user's social graph."""

    person_id: str  # KG entity ID
    name: str
    role: str = ""  # "manager", "colleague", "friend", "family"
    organization: str = ""

    # Learned qualities
    trust_level: float = 0.5  # 0.0-1.0
    trust_confidence: float = 0.3
    importance: float = 0.5  # 0.0-1.0
    importance_confidence: float = 0.3
    interaction_frequency: float = 0.0  # Interactions per week (EWMA)
    sentiment: float = 0.0  # -1.0 to 1.0
    sentiment_confidence: float = 0.3

    # History
    mention_count: int = 0
    first_mentioned: float = field(default_factory=time.time)
    last_mentioned: float = field(default_factory=time.time)
    topics_discussed: list[str] = field(default_factory=list)

    # Communication
    preferred_channel: str = ""  # "email", "slack", "call"

    def to_dict(self) -> dict[str, Any]:
        return {
            "person_id": self.person_id,
            "name": self.name,
            "role": self.role,
            "organization": self.organization,
            "trust_level": round(self.trust_level, 3),
            "trust_confidence": round(self.trust_confidence, 3),
            "importance": round(self.importance, 3),
            "importance_confidence": round(self.importance_confidence, 3),
            "interaction_frequency": round(self.interaction_frequency, 3),
            "sentiment": round(self.sentiment, 3),
            "sentiment_confidence": round(self.sentiment_confidence, 3),
            "mention_count": self.mention_count,
            "first_mentioned": self.first_mentioned,
            "last_mentioned": self.last_mentioned,
            "topics_discussed": self.topics_discussed[:10],
            "preferred_channel": self.preferred_channel,
        }


@dataclass
class RelationshipEvent:
    """A point in relationship history."""

    timestamp: float
    event_type: str  # "positive_interaction" | "conflict" | "collaboration" | "mention" | "meeting"
    context: str
    sentiment_delta: float = 0.0  # How this event changed sentiment

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "context": self.context,
            "sentiment_delta": round(self.sentiment_delta, 3),
        }


@dataclass
class RelationshipHistory:
    """Temporal history of a relationship."""

    person_id: str
    person_name: str
    events: list[RelationshipEvent] = field(default_factory=list)
    trend: str = "stable"  # "improving" | "stable" | "declining"

    def compute_trend(self, window_days: int = 30) -> str:
        """Compute sentiment trend from recent vs older events."""
        if len(self.events) < 3:
            self.trend = "stable"
            return self.trend

        now = time.time()
        cutoff = now - (window_days * 86400)

        recent = [e for e in self.events if e.timestamp > cutoff]
        older = [e for e in self.events if e.timestamp <= cutoff]

        if not recent or not older:
            self.trend = "stable"
            return self.trend

        recent_avg = sum(e.sentiment_delta for e in recent) / len(recent)
        older_avg = sum(e.sentiment_delta for e in older) / len(older)

        if recent_avg > older_avg + 0.1:
            self.trend = "improving"
        elif recent_avg < older_avg - 0.1:
            self.trend = "declining"
        else:
            self.trend = "stable"

        return self.trend


# ── Name Extraction ──────────────────────────────────────────────────────────

# Simple pattern-based name extraction (NER-lite)
_NAME_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b"  # Capitalized multi-word names
)

# Common false positives to filter out
_NAME_BLACKLIST = {
    "I", "The", "This", "That", "What", "How", "When", "Where", "Why", "Who",
    "Let Me", "Can You", "Do You", "Thank You", "Good Morning", "Good Night",
    "New York", "San Francisco", "Los Angeles", "United States",
    "Google Cloud", "Amazon Web", "Microsoft Azure",
}


def extract_person_mentions(text: str) -> list[str]:
    """Extract potential person names from text.

    Uses capitalization patterns. Not perfect — but lightweight and
    runs on every interaction without LLM cost.
    """
    matches = _NAME_PATTERN.findall(text)
    names = []
    for match in matches:
        if match not in _NAME_BLACKLIST and len(match.split()) <= 4:
            names.append(match)
    return list(set(names))


# ── RelationshipMemory Engine ────────────────────────────────────────────────


class RelationshipMemory:
    """KG-integrated social graph with temporal history.

    People and relationships live in the KnowledgeGraph.
    Temporal event history lives in SQLite.
    This class is a typed wrapper that provides social graph semantics.
    """

    def __init__(
        self,
        knowledge_graph: Any | None = None,
        data_dir: str = "data",
    ) -> None:
        self._kg = knowledge_graph
        self._db_path = Path(data_dir) / "relationship_events.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory person cache (loaded from KG + SQLite)
        self._persons: dict[str, Person] = {}

    def _init_db(self) -> None:
        """Initialize SQLite for temporal event history."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationship_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    event_type TEXT NOT NULL,
                    context TEXT DEFAULT '',
                    sentiment_delta REAL DEFAULT 0.0,
                    tenant_id TEXT DEFAULT 'default'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role TEXT DEFAULT '',
                    organization TEXT DEFAULT '',
                    trust_level REAL DEFAULT 0.5,
                    trust_confidence REAL DEFAULT 0.3,
                    importance REAL DEFAULT 0.5,
                    importance_confidence REAL DEFAULT 0.3,
                    interaction_frequency REAL DEFAULT 0.0,
                    sentiment REAL DEFAULT 0.0,
                    sentiment_confidence REAL DEFAULT 0.3,
                    mention_count INTEGER DEFAULT 0,
                    first_mentioned REAL NOT NULL,
                    last_mentioned REAL NOT NULL,
                    topics_json TEXT DEFAULT '[]',
                    preferred_channel TEXT DEFAULT '',
                    tenant_id TEXT DEFAULT 'default'
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_re_person ON relationship_events(person_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_re_tenant ON relationship_events(tenant_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_p_tenant ON persons(tenant_id)"
            )

    # ── Core Operations ──────────────────────────────────────────────

    def record_mention(
        self,
        person_name: str,
        context: str = "",
        sentiment: float = 0.0,
        tenant_id: str = "default",
        topic: str = "",
    ) -> Person:
        """Record that a person was mentioned in conversation.

        Creates the person if not seen before.
        Updates mention count, sentiment, topics, and frequency.
        """
        person = self._get_or_create(person_name, tenant_id)
        now = time.time()

        # Update mention stats
        person.mention_count += 1
        person.last_mentioned = now

        # EWMA for interaction frequency (approximate weekly rate)
        days_since_last = (now - person.last_mentioned) / 86400.0 if person.mention_count > 1 else 7.0
        weekly_rate = 7.0 / max(0.01, days_since_last)
        person.interaction_frequency = 0.7 * person.interaction_frequency + 0.3 * weekly_rate

        # Update sentiment
        if abs(sentiment) > 0.01:
            alpha = 0.2
            person.sentiment = person.sentiment * (1 - alpha) + sentiment * alpha
            person.sentiment_confidence = min(1.0, person.sentiment_confidence + 0.05)

        # Update topics
        if topic and topic not in person.topics_discussed:
            person.topics_discussed.append(topic)
            person.topics_discussed = person.topics_discussed[-20:]  # Keep last 20

        # Update importance based on mention frequency
        person.importance = min(1.0, 0.3 + person.mention_count * 0.05)
        person.importance_confidence = min(1.0, person.mention_count * 0.1)

        # Record event
        self._record_event(person.person_id, "mention", context, sentiment, tenant_id)

        # Persist
        self._save_person(person)

        # Sync to KG if available
        self._sync_to_kg(person)

        return person

    def record_event(
        self,
        person_name: str,
        event_type: str,
        context: str = "",
        sentiment_delta: float = 0.0,
        tenant_id: str = "default",
    ) -> None:
        """Record a relationship event (meeting, collaboration, conflict)."""
        person = self._get_or_create(person_name, tenant_id)

        # Update sentiment
        if abs(sentiment_delta) > 0.01:
            person.sentiment = max(-1.0, min(1.0, person.sentiment + sentiment_delta * 0.3))
            person.sentiment_confidence = min(1.0, person.sentiment_confidence + 0.05)

        self._record_event(person.person_id, event_type, context, sentiment_delta, tenant_id)
        self._save_person(person)

    def learn_relationship(
        self,
        person_a: str,
        person_b: str,
        relation_type: str,
        context: str = "",
        tenant_id: str = "default",
    ) -> None:
        """Learn a relationship between two people.

        Also syncs to KnowledgeGraph as a relation edge.
        """
        pa = self._get_or_create(person_a, tenant_id)
        pb = self._get_or_create(person_b, tenant_id)

        # Update roles based on relationship
        role_map = {
            "reports_to": ("", "manager"),
            "manages": ("manager", ""),
            "friend": ("friend", "friend"),
            "family": ("family", "family"),
            "colleague": ("colleague", "colleague"),
            "mentors": ("mentor", "mentee"),
        }
        roles = role_map.get(relation_type, ("", ""))
        if roles[0] and not pa.role:
            pa.role = roles[0]
            self._save_person(pa)
        if roles[1] and not pb.role:
            pb.role = roles[1]
            self._save_person(pb)

        # Sync to KG
        if self._kg:
            try:
                self._kg.add_relation(
                    pa.person_id, pb.person_id, relation_type,
                    metadata={"context": context, "learned_at": time.time()},
                )
            except Exception as e:
                logger.debug("Failed to sync relationship to KG: %s", e)

        logger.info(
            "Learned relationship: %s --%s--> %s",
            person_a, relation_type, person_b,
        )

    # ── Queries ──────────────────────────────────────────────────────

    def get_person(self, name: str, tenant_id: str = "default") -> Person | None:
        """Lookup by name with fuzzy matching."""
        name_lower = name.lower()

        # Exact match first
        for person in self._persons.values():
            if person.name.lower() == name_lower:
                return person

        # Fuzzy: check if name is contained
        for person in self._persons.values():
            if name_lower in person.name.lower() or person.name.lower() in name_lower:
                return person

        # Try loading from DB
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM persons WHERE LOWER(name) = LOWER(?) AND tenant_id = ?",
                (name, tenant_id),
            ).fetchone()
            if row:
                person = self._row_to_person(row)
                self._persons[person.person_id] = person
                return person

        return None

    def get_network(self, tenant_id: str = "default") -> list[Person]:
        """Get the full relationship graph for a user."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM persons WHERE tenant_id = ? ORDER BY importance DESC",
                (tenant_id,),
            ).fetchall()
        persons = [self._row_to_person(r) for r in rows]
        # Update cache
        for p in persons:
            self._persons[p.person_id] = p
        return persons

    def get_relevant_people(self, topic: str, tenant_id: str = "default") -> list[Person]:
        """Find people relevant to a given topic."""
        network = self.get_network(tenant_id)
        relevant = []
        topic_lower = topic.lower()
        for person in network:
            for discussed in person.topics_discussed:
                if topic_lower in discussed.lower() or discussed.lower() in topic_lower:
                    relevant.append(person)
                    break
        return sorted(relevant, key=lambda p: p.importance, reverse=True)

    def get_history(
        self, person_name: str, tenant_id: str = "default", limit: int = 50
    ) -> RelationshipHistory:
        """Get temporal event history for a relationship."""
        person = self.get_person(person_name, tenant_id)
        if not person:
            return RelationshipHistory(person_id="", person_name=person_name)

        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM relationship_events WHERE person_id = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (person.person_id, limit),
            ).fetchall()

        events = [
            RelationshipEvent(
                timestamp=r["timestamp"],
                event_type=r["event_type"],
                context=r["context"],
                sentiment_delta=r["sentiment_delta"],
            )
            for r in rows
        ]

        history = RelationshipHistory(
            person_id=person.person_id,
            person_name=person.name,
            events=events,
        )
        history.compute_trend()
        return history

    def compute_trend(self, person_name: str, tenant_id: str = "default") -> str:
        """Get relationship trend for a person."""
        history = self.get_history(person_name, tenant_id)
        return history.trend

    def prioritize_notification(
        self, person_name: str, tenant_id: str = "default"
    ) -> float:
        """Compute notification priority for messages about this person.

        Higher = more important to notify about.
        """
        person = self.get_person(person_name, tenant_id)
        if not person:
            return 0.3  # Unknown person — moderate priority

        # Importance × recency × sentiment
        recency_days = (time.time() - person.last_mentioned) / 86400.0
        recency_factor = max(0.1, 1.0 - recency_days / 30.0)

        score = (
            person.importance * 0.4
            + recency_factor * 0.3
            + max(0.0, person.sentiment + 1.0) / 2.0 * 0.3
        )
        return min(1.0, score)

    # ── Context Generation ───────────────────────────────────────────

    async def get_context(
        self, query: str, tenant_id: str, budget: int
    ) -> str:
        """ContextFusion-compatible provider.

        Returns relevant people for the current query.
        """
        # Extract person mentions from query
        mentions = extract_person_mentions(query)
        parts: list[str] = []

        for name in mentions:
            person = self.get_person(name, tenant_id)
            if person:
                role_str = f" ({person.role})" if person.role else ""
                org_str = f" at {person.organization}" if person.organization else ""
                sentiment_label = (
                    "positive" if person.sentiment > 0.3
                    else "negative" if person.sentiment < -0.3
                    else "neutral"
                )
                parts.append(
                    f"{person.name}{role_str}{org_str}: "
                    f"importance={person.importance:.0%}, "
                    f"sentiment={sentiment_label}"
                )

        if not parts:
            # Show top important people if no specific mention
            network = self.get_network(tenant_id)
            important = [p for p in network if p.importance > 0.5][:3]
            if important:
                names = [f"{p.name} ({p.role})" if p.role else p.name for p in important]
                parts.append(f"Key people: {', '.join(names)}")

        return "\n".join(parts) if parts else ""

    # ── Internal Helpers ─────────────────────────────────────────────

    def _get_or_create(self, name: str, tenant_id: str = "default") -> Person:
        """Get existing person or create new one."""
        person = self.get_person(name, tenant_id)
        if person:
            return person

        now = time.time()
        person_id = f"person_{int(now)}_{hash(name) % 10000}"
        person = Person(
            person_id=person_id,
            name=name,
            first_mentioned=now,
            last_mentioned=now,
        )
        self._persons[person_id] = person
        self._save_person(person)
        logger.info("New person detected: %s (%s)", name, person_id)
        return person

    def _record_event(
        self,
        person_id: str,
        event_type: str,
        context: str,
        sentiment_delta: float,
        tenant_id: str,
    ) -> None:
        """Record a temporal event."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT INTO relationship_events "
                "(person_id, timestamp, event_type, context, sentiment_delta, tenant_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, time.time(), event_type, context, sentiment_delta, tenant_id),
            )

    def _save_person(self, person: Person) -> None:
        """Persist person to SQLite."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO persons "
                "(person_id, name, role, organization, trust_level, trust_confidence, "
                "importance, importance_confidence, interaction_frequency, sentiment, "
                "sentiment_confidence, mention_count, first_mentioned, last_mentioned, "
                "topics_json, preferred_channel, tenant_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    person.person_id, person.name, person.role, person.organization,
                    person.trust_level, person.trust_confidence,
                    person.importance, person.importance_confidence,
                    person.interaction_frequency, person.sentiment,
                    person.sentiment_confidence, person.mention_count,
                    person.first_mentioned, person.last_mentioned,
                    json.dumps(person.topics_discussed), person.preferred_channel,
                    "default",
                ),
            )

    def _row_to_person(self, row: sqlite3.Row) -> Person:
        return Person(
            person_id=row["person_id"],
            name=row["name"],
            role=row["role"],
            organization=row["organization"],
            trust_level=row["trust_level"],
            trust_confidence=row["trust_confidence"],
            importance=row["importance"],
            importance_confidence=row["importance_confidence"],
            interaction_frequency=row["interaction_frequency"],
            sentiment=row["sentiment"],
            sentiment_confidence=row["sentiment_confidence"],
            mention_count=row["mention_count"],
            first_mentioned=row["first_mentioned"],
            last_mentioned=row["last_mentioned"],
            topics_discussed=json.loads(row["topics_json"]),
            preferred_channel=row["preferred_channel"],
        )

    def _sync_to_kg(self, person: Person) -> None:
        """Sync person data to KnowledgeGraph."""
        if not self._kg:
            return
        try:
            self._kg.add_entity(
                person.name,
                entity_type="person",
                attributes={
                    "role": person.role,
                    "organization": person.organization,
                    "trust": person.trust_level,
                    "importance": person.importance,
                    "sentiment": person.sentiment,
                },
            )
        except Exception as e:
            logger.debug("Failed to sync person to KG: %s", e)

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        with sqlite3.connect(str(self._db_path)) as conn:
            total_persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
            total_events = conn.execute("SELECT COUNT(*) FROM relationship_events").fetchone()[0]
        return {
            "total_persons": total_persons,
            "total_events": total_events,
            "cached_persons": len(self._persons),
        }
