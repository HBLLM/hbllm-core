"""
UserModel — predictive model of the human operator.

Continuously learns who the user is from every interaction:
    - Expertise levels per domain (with confidence)
    - Preferences for response style, tools, languages
    - Beliefs and worldview on topics
    - Trust relationship per domain (delegation vs override)
    - Current focus, stress, engagement
    - Predicted next actions

Every learned attribute carries confidence, evidence count, and provenance.
Confidence grows with observations and decays with staleness.

Integration:
    - Reads from: EmotionEngine, HabitTracker, CognitiveLoadEstimator
    - Writes to: ContextFusionEngine (registered as 'user_model' source)
    - Bus: subscribes system.experience, system.feedback, system.evaluation
    - Bus: publishes user.model.updated, user.model.focus_changed

Bus Topics:
    user.model.updated      → Emitted when model changes significantly
    user.model.focus_changed → Emitted when user's focus topic shifts
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Core Data Types ──────────────────────────────────────────────────────────


@dataclass
class LearnedAttribute:
    """Base for every learned fact. Everything carries confidence.

    Confidence grows with evidence and decays with staleness.
    Explicit corrections override inferred values.
    """

    value: Any
    confidence: float = 0.3  # 0.0-1.0
    evidence_count: int = 0
    first_observed: float = field(default_factory=time.time)
    last_observed: float = field(default_factory=time.time)
    source: str = "inferred"  # "explicit" | "inferred" | "corrected"

    def update(self, new_value: Any, source: str = "inferred") -> None:
        """Update with new evidence."""
        now = time.time()
        self.evidence_count += 1
        self.last_observed = now

        if source == "explicit" or source == "corrected":
            self.value = new_value
            self.source = "corrected"
        else:
            # For numeric values, use EWMA
            if isinstance(self.value, (int, float)) and isinstance(new_value, (int, float)):
                alpha = 0.2
                self.value = alpha * new_value + (1 - alpha) * self.value
            else:
                self.value = new_value
            self.source = "inferred"

        # Confidence approaches 1.0 with more evidence
        self.confidence = min(1.0, 1.0 - math.exp(-self.evidence_count / 5.0))

        # Explicit corrections always override to high confidence
        if source == "explicit" or source == "corrected":
            self.confidence = 0.95

    def decay(self, now: float | None = None, half_life_days: float = 30.0) -> None:
        """Time-based confidence decay for stale attributes."""
        if now is None:
            now = time.time()
        age_days = (now - self.last_observed) / 86400.0
        if age_days > 1.0:
            decay_factor = math.exp(-0.693 * age_days / half_life_days)
            self.confidence *= decay_factor

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedAttribute:
        return cls(
            value=data["value"],
            confidence=data.get("confidence", 0.3),
            evidence_count=data.get("evidence_count", 0),
            first_observed=data.get("first_observed", time.time()),
            last_observed=data.get("last_observed", time.time()),
            source=data.get("source", "inferred"),
        )


@dataclass
class UserExpertise:
    """User's expertise in a domain, with confidence."""

    domain: str
    level: LearnedAttribute  # value: 0.0 (novice) to 1.0 (expert)

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "level": self.level.to_dict()}


@dataclass
class UserPreference:
    """A learned user preference."""

    key: str
    learned: LearnedAttribute  # value: the preference value (str or numeric)

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "learned": self.learned.to_dict()}


@dataclass
class UserBelief:
    """User's belief or stance on a topic."""

    topic: str
    stance: str  # "brain-inspired architectures matter"
    confidence: float = 0.5
    evidence_count: int = 0
    last_expressed: float = field(default_factory=time.time)

    def reinforce(self) -> None:
        """Strengthen belief with repeated expression."""
        self.evidence_count += 1
        self.confidence = min(1.0, 1.0 - math.exp(-self.evidence_count / 3.0))
        self.last_expressed = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "stance": self.stance,
            "confidence": round(self.confidence, 3),
            "evidence_count": self.evidence_count,
            "last_expressed": self.last_expressed,
        }


@dataclass
class TrustDimension:
    """Trust level in a specific domain."""

    domain: str
    trust_level: LearnedAttribute  # value: 0.0 (no trust) to 1.0 (full autonomy)
    delegations_count: int = 0
    overrides_count: int = 0

    def record_delegation(self) -> None:
        """User accepted our suggestion — trust increases."""
        self.delegations_count += 1
        current = float(self.trust_level.value)
        self.trust_level.update(min(1.0, current + 0.05))

    def record_override(self) -> None:
        """User rejected our suggestion — trust decreases, confidence increases."""
        self.overrides_count += 1
        current = float(self.trust_level.value)
        self.trust_level.update(max(0.0, current - 0.1))
        # Overrides make us MORE confident about the trust level (we learned something)
        self.trust_level.confidence = min(1.0, self.trust_level.confidence + 0.1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "trust_level": self.trust_level.to_dict(),
            "delegations_count": self.delegations_count,
            "overrides_count": self.overrides_count,
        }


@dataclass
class UserModel:
    """Complete predictive model of a user."""

    tenant_id: str
    display_name: str = ""

    # Learned dimensions
    expertise: dict[str, UserExpertise] = field(default_factory=dict)
    preferences: dict[str, UserPreference] = field(default_factory=dict)
    beliefs: list[UserBelief] = field(default_factory=list)
    trust: dict[str, TrustDimension] = field(default_factory=dict)

    # Live state
    current_focus: LearnedAttribute = field(
        default_factory=lambda: LearnedAttribute(value="", confidence=0.0)
    )
    stress_level: float = 0.0
    engagement_level: float = 0.5

    # Predictive
    likely_next_actions: list[str] = field(default_factory=list)
    active_interests: list[LearnedAttribute] = field(default_factory=list)

    # Temporal
    active_hours: dict[int, float] = field(default_factory=dict)
    active_days: dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "display_name": self.display_name,
            "expertise": {k: v.to_dict() for k, v in self.expertise.items()},
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "beliefs": [b.to_dict() for b in self.beliefs],
            "trust": {k: v.to_dict() for k, v in self.trust.items()},
            "current_focus": self.current_focus.to_dict(),
            "stress_level": round(self.stress_level, 3),
            "engagement_level": round(self.engagement_level, 3),
            "likely_next_actions": self.likely_next_actions,
            "active_interests": [i.to_dict() for i in self.active_interests],
            "active_hours": self.active_hours,
            "active_days": self.active_days,
        }


# ── Expertise Inference Signals ──────────────────────────────────────────────

# Words that indicate domain expertise when used naturally
_EXPERTISE_SIGNALS: dict[str, list[str]] = {
    "python": [
        "asyncio",
        "dataclass",
        "pydantic",
        "fastapi",
        "uvicorn",
        "mypy",
        "pytest",
        "typing",
        "decorator",
        "metaclass",
        "generator",
        "coroutine",
        "walrus",
        "comprehension",
        "dunder",
        "virtualenv",
        "pyproject",
    ],
    "rust": [
        "borrow",
        "ownership",
        "lifetime",
        "cargo",
        "crate",
        "trait",
        "impl",
        "macro",
        "unsafe",
        "tokio",
        "async-std",
        "serde",
    ],
    "docker": [
        "dockerfile",
        "compose",
        "swarm",
        "kubernetes",
        "k8s",
        "container",
        "volume",
        "network",
        "registry",
        "multi-stage",
        "buildkit",
    ],
    "machine_learning": [
        "transformer",
        "attention",
        "backprop",
        "gradient",
        "epoch",
        "batch_size",
        "learning_rate",
        "fine-tune",
        "lora",
        "qlora",
        "embedding",
        "tokenizer",
        "snn",
        "spiking",
        "neural",
    ],
    "devops": [
        "ci/cd",
        "pipeline",
        "terraform",
        "ansible",
        "nginx",
        "caddy",
        "systemd",
        "cron",
        "monitoring",
        "grafana",
        "prometheus",
    ],
    "database": [
        "postgresql",
        "mysql",
        "sqlite",
        "migration",
        "index",
        "join",
        "transaction",
        "orm",
        "query",
        "normalization",
        "sharding",
    ],
    "frontend": [
        "react",
        "vue",
        "svelte",
        "tailwind",
        "css-grid",
        "flexbox",
        "webpack",
        "vite",
        "typescript",
        "jsx",
        "component",
    ],
    "flutter": [
        "widget",
        "stateful",
        "stateless",
        "bloc",
        "riverpod",
        "provider",
        "pubspec",
        "dart",
        "material",
        "cupertino",
        "navigator",
    ],
    "laravel": [
        "eloquent",
        "artisan",
        "blade",
        "migration",
        "seeder",
        "middleware",
        "nova",
        "filament",
        "livewire",
        "sanctum",
        "passport",
    ],
}


# ── UserModelEngine ──────────────────────────────────────────────────────────


class UserModelEngine:
    """Manages the lifecycle of UserModel instances.

    Handles persistence (SQLite), inference from interactions,
    and context generation for LLM injection.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self._db_path = Path(data_dir) / "user_model.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._models: dict[str, UserModel] = {}

    def _init_db(self) -> None:
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_attributes (
                    tenant_id TEXT NOT NULL,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    confidence REAL DEFAULT 0.3,
                    evidence_count INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'inferred',
                    first_observed REAL NOT NULL,
                    last_observed REAL NOT NULL,
                    PRIMARY KEY (tenant_id, category, key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_beliefs (
                    tenant_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    stance TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    evidence_count INTEGER DEFAULT 0,
                    last_expressed REAL NOT NULL,
                    PRIMARY KEY (tenant_id, topic)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ua_tenant ON user_attributes(tenant_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ub_tenant ON user_beliefs(tenant_id)")

    # ── Model Access ─────────────────────────────────────────────────

    def get_model(self, tenant_id: str) -> UserModel:
        """Get or create a UserModel for a tenant."""
        if tenant_id not in self._models:
            model = self._load_from_db(tenant_id)
            if model is None:
                model = UserModel(tenant_id=tenant_id)
            self._models[tenant_id] = model
        return self._models[tenant_id]

    # ── Learning from Interactions ───────────────────────────────────

    def update_from_interaction(
        self,
        tenant_id: str,
        query: str,
        response: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Main entry point — extract signals from every interaction.

        Returns True if the model changed significantly.
        """
        model = self.get_model(tenant_id)
        meta = metadata or {}
        changed = False

        # 1. Infer expertise from vocabulary
        expertise_changes = self._infer_expertise(model, query)
        if expertise_changes:
            changed = True

        # 2. Update current focus
        focus_changed = self._update_focus(model, query, meta)
        if focus_changed:
            changed = True

        # 3. Track temporal patterns
        self._update_temporal_patterns(model)

        # 4. Infer interests from recurring topics
        self._update_interests(model, query, meta)

        # 5. Decay stale attributes
        self._decay_all(model)

        # Persist if changed
        if changed:
            self._save_to_db(model)

        return changed

    def _infer_expertise(self, model: UserModel, text: str) -> list[str]:
        """Detect domain expertise from vocabulary usage."""
        text_lower = text.lower()
        words = set(text_lower.split())
        changes = []

        for domain, signals in _EXPERTISE_SIGNALS.items():
            hits = sum(1 for s in signals if s in words or s in text_lower)
            if hits == 0:
                continue

            # Signal strength based on how many domain-specific terms used
            signal = min(1.0, hits / 3.0)

            if domain not in model.expertise:
                model.expertise[domain] = UserExpertise(
                    domain=domain,
                    level=LearnedAttribute(value=signal * 0.5),
                )
                changes.append(domain)
            else:
                model.expertise[domain].level.update(signal)
                changes.append(domain)

        return changes

    def _update_focus(self, model: UserModel, query: str, metadata: dict[str, Any]) -> bool:
        """Track what the user is currently focused on."""
        # Extract topic from metadata or query
        topic = metadata.get("topic", metadata.get("intent", ""))
        if not topic and len(query) > 10:
            # Use first significant phrase as topic proxy
            words = query.split()[:6]
            topic = " ".join(words)

        if not topic:
            return False

        old_focus = model.current_focus.value
        model.current_focus.update(topic)

        return old_focus != topic

    def _update_temporal_patterns(self, model: UserModel) -> None:
        """Record when the user is active."""
        now = time.localtime()
        hour = now.tm_hour
        day = now.tm_wday

        # EWMA for hourly activity
        model.active_hours[hour] = model.active_hours.get(hour, 0.0) * 0.9 + 0.1
        # Decay other hours slightly
        for h in list(model.active_hours.keys()):
            if h != hour:
                model.active_hours[h] *= 0.995

        # EWMA for daily activity
        model.active_days[day] = model.active_days.get(day, 0.0) * 0.9 + 0.1
        for d in list(model.active_days.keys()):
            if d != day:
                model.active_days[d] *= 0.995

    def _update_interests(self, model: UserModel, query: str, metadata: dict[str, Any]) -> None:
        """Track recurring interests."""
        topic = metadata.get("topic", metadata.get("intent", ""))
        if not topic:
            return

        # Check if this interest already exists
        for interest in model.active_interests:
            if interest.value == topic:
                interest.update(topic)
                return

        # New interest
        if len(model.active_interests) >= 20:
            # Evict lowest-confidence interest
            model.active_interests.sort(key=lambda i: i.confidence)
            model.active_interests.pop(0)

        model.active_interests.append(LearnedAttribute(value=topic))

    def _decay_all(self, model: UserModel) -> None:
        """Apply time-based confidence decay across all attributes."""
        now = time.time()

        for exp in model.expertise.values():
            exp.level.decay(now, half_life_days=60.0)  # Expertise decays slowly

        for pref in model.preferences.values():
            pref.learned.decay(now, half_life_days=90.0)  # Preferences very stable

        for interest in model.active_interests:
            interest.decay(now, half_life_days=14.0)  # Interests shift faster

        model.current_focus.decay(now, half_life_days=1.0)  # Focus is ephemeral

    # ── Explicit Learning ────────────────────────────────────────────

    def learn_preference(
        self,
        tenant_id: str,
        key: str,
        value: Any,
        source: str = "inferred",
    ) -> None:
        """Record a user preference."""
        model = self.get_model(tenant_id)
        if key in model.preferences:
            model.preferences[key].learned.update(value, source=source)
        else:
            confidence = 0.95 if source == "explicit" else 0.3
            model.preferences[key] = UserPreference(
                key=key,
                learned=LearnedAttribute(value=value, confidence=confidence, source=source),
            )
        self._save_to_db(model)

    def record_belief(self, tenant_id: str, topic: str, stance: str) -> None:
        """Track a user's belief or worldview on a topic."""
        model = self.get_model(tenant_id)
        for belief in model.beliefs:
            if belief.topic == topic:
                belief.stance = stance
                belief.reinforce()
                self._save_to_db(model)
                return

        model.beliefs.append(UserBelief(topic=topic, stance=stance))
        # Cap beliefs list
        if len(model.beliefs) > 50:
            model.beliefs.sort(key=lambda b: b.confidence, reverse=True)
            model.beliefs = model.beliefs[:50]
        self._save_to_db(model)

    def update_trust(
        self,
        tenant_id: str,
        domain: str,
        delegated: bool = False,
        overridden: bool = False,
    ) -> None:
        """Adjust trust based on delegation/override signals."""
        model = self.get_model(tenant_id)
        if domain not in model.trust:
            model.trust[domain] = TrustDimension(
                domain=domain,
                trust_level=LearnedAttribute(value=0.5),
            )

        dim = model.trust[domain]
        if delegated:
            dim.record_delegation()
        if overridden:
            dim.record_override()

        self._save_to_db(model)

    def update_stress(self, tenant_id: str, stress: float) -> None:
        """Update stress level from external signals."""
        model = self.get_model(tenant_id)
        model.stress_level = max(0.0, min(1.0, stress))

    def update_engagement(self, tenant_id: str, engagement: float) -> None:
        """Update engagement level from external signals."""
        model = self.get_model(tenant_id)
        model.engagement_level = max(0.0, min(1.0, engagement))

    # ── Context Generation ───────────────────────────────────────────

    async def get_context(self, query: str, tenant_id: str, budget: int) -> str:
        """Generate NL context summary for ContextFusionEngine.

        Produces a concise description of the user suitable for
        system prompt injection.
        """
        model = self.get_model(tenant_id)
        parts: list[str] = []

        # Current focus
        if model.current_focus.value and model.current_focus.confidence > 0.3:
            parts.append(f"Currently focused on: {model.current_focus.value}")

        # Top expertise areas
        strong = sorted(
            model.expertise.values(),
            key=lambda e: float(e.level.value) * e.level.confidence,
            reverse=True,
        )[:5]
        if strong:
            areas = []
            for exp in strong:
                if exp.level.confidence > 0.3:
                    level_label = (
                        "expert"
                        if float(exp.level.value) > 0.7
                        else "proficient"
                        if float(exp.level.value) > 0.4
                        else "familiar"
                    )
                    areas.append(f"{exp.domain} ({level_label})")
            if areas:
                parts.append(f"Expertise: {', '.join(areas)}")

        # Key preferences
        confident_prefs = [p for p in model.preferences.values() if p.learned.confidence > 0.5]
        if confident_prefs:
            pref_strs = [f"{p.key}={p.learned.value}" for p in confident_prefs[:5]]
            parts.append(f"Preferences: {', '.join(pref_strs)}")

        # Active interests
        active = [i for i in model.active_interests if i.confidence > 0.3]
        if active:
            interests = [
                str(i.value) for i in sorted(active, key=lambda x: x.confidence, reverse=True)[:5]
            ]
            parts.append(f"Current interests: {', '.join(interests)}")

        # Stress/engagement
        if model.stress_level > 0.6:
            parts.append("User appears stressed — keep responses concise.")
        if model.engagement_level < 0.3:
            parts.append("Low engagement — consider being more proactive.")

        # Trust areas
        high_trust = [
            t
            for t in model.trust.values()
            if float(t.trust_level.value) > 0.7 and t.trust_level.confidence > 0.5
        ]
        if high_trust:
            domains = [t.domain for t in high_trust]
            parts.append(f"High trust in: {', '.join(domains)}")

        # Key beliefs
        strong_beliefs = [b for b in model.beliefs if b.confidence > 0.6][:3]
        if strong_beliefs:
            belief_strs = [f"{b.topic}: {b.stance}" for b in strong_beliefs]
            parts.append(f"Known stances: {'; '.join(belief_strs)}")

        return "\n".join(parts) if parts else ""

    # ── Prediction ───────────────────────────────────────────────────

    def predict_next_actions(
        self, tenant_id: str, context: dict[str, Any] | None = None
    ) -> list[str]:
        """Predict what the user is likely to do next.

        Uses temporal patterns and action co-occurrence.
        """
        model = self.get_model(tenant_id)
        predictions: list[str] = []

        # Time-based predictions from habits
        now = time.localtime()
        hour = now.tm_hour
        if model.active_hours.get(hour, 0) > 0.3:
            # User is typically active at this hour
            predictions.append(f"likely_active_at_{hour}:00")

        model.likely_next_actions = predictions[:5]
        return predictions

    # ── Persistence ──────────────────────────────────────────────────

    def _save_to_db(self, model: UserModel) -> None:
        """Persist a UserModel to SQLite."""
        now = time.time()
        with sqlite3.connect(str(self._db_path)) as conn:
            # Save expertise
            for domain, exp in model.expertise.items():
                conn.execute(
                    "INSERT OR REPLACE INTO user_attributes "
                    "(tenant_id, category, key, value_json, confidence, evidence_count, "
                    "source, first_observed, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        model.tenant_id,
                        "expertise",
                        domain,
                        json.dumps(exp.level.value),
                        exp.level.confidence,
                        exp.level.evidence_count,
                        exp.level.source,
                        exp.level.first_observed,
                        exp.level.last_observed,
                    ),
                )

            # Save preferences
            for key, pref in model.preferences.items():
                conn.execute(
                    "INSERT OR REPLACE INTO user_attributes "
                    "(tenant_id, category, key, value_json, confidence, evidence_count, "
                    "source, first_observed, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        model.tenant_id,
                        "preference",
                        key,
                        json.dumps(pref.learned.value),
                        pref.learned.confidence,
                        pref.learned.evidence_count,
                        pref.learned.source,
                        pref.learned.first_observed,
                        pref.learned.last_observed,
                    ),
                )

            # Save trust
            for domain, trust in model.trust.items():
                conn.execute(
                    "INSERT OR REPLACE INTO user_attributes "
                    "(tenant_id, category, key, value_json, confidence, evidence_count, "
                    "source, first_observed, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        model.tenant_id,
                        "trust",
                        domain,
                        json.dumps(
                            {
                                "trust_level": trust.trust_level.value,
                                "delegations": trust.delegations_count,
                                "overrides": trust.overrides_count,
                            }
                        ),
                        trust.trust_level.confidence,
                        trust.trust_level.evidence_count,
                        trust.trust_level.source,
                        trust.trust_level.first_observed,
                        trust.trust_level.last_observed,
                    ),
                )

            # Save interests
            for i, interest in enumerate(model.active_interests):
                conn.execute(
                    "INSERT OR REPLACE INTO user_attributes "
                    "(tenant_id, category, key, value_json, confidence, evidence_count, "
                    "source, first_observed, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        model.tenant_id,
                        "interest",
                        f"interest_{i}",
                        json.dumps(interest.value),
                        interest.confidence,
                        interest.evidence_count,
                        interest.source,
                        interest.first_observed,
                        interest.last_observed,
                    ),
                )

            # Save focus
            conn.execute(
                "INSERT OR REPLACE INTO user_attributes "
                "(tenant_id, category, key, value_json, confidence, evidence_count, "
                "source, first_observed, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model.tenant_id,
                    "state",
                    "current_focus",
                    json.dumps(model.current_focus.value),
                    model.current_focus.confidence,
                    model.current_focus.evidence_count,
                    model.current_focus.source,
                    model.current_focus.first_observed,
                    model.current_focus.last_observed,
                ),
            )

            # Save temporal patterns
            conn.execute(
                "INSERT OR REPLACE INTO user_attributes "
                "(tenant_id, category, key, value_json, confidence, evidence_count, "
                "source, first_observed, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model.tenant_id,
                    "temporal",
                    "active_hours",
                    json.dumps(model.active_hours),
                    0.8,
                    0,
                    "inferred",
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO user_attributes "
                "(tenant_id, category, key, value_json, confidence, evidence_count, "
                "source, first_observed, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    model.tenant_id,
                    "temporal",
                    "active_days",
                    json.dumps(model.active_days),
                    0.8,
                    0,
                    "inferred",
                    now,
                    now,
                ),
            )

            # Save beliefs
            for belief in model.beliefs:
                conn.execute(
                    "INSERT OR REPLACE INTO user_beliefs "
                    "(tenant_id, topic, stance, confidence, evidence_count, last_expressed) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        model.tenant_id,
                        belief.topic,
                        belief.stance,
                        belief.confidence,
                        belief.evidence_count,
                        belief.last_expressed,
                    ),
                )

    def _load_from_db(self, tenant_id: str) -> UserModel | None:
        """Load a UserModel from SQLite."""
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM user_attributes WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()

            if not rows:
                return None

            model = UserModel(tenant_id=tenant_id)

            for row in rows:
                category = row["category"]
                key = row["key"]
                value = json.loads(row["value_json"])
                attr = LearnedAttribute(
                    value=value,
                    confidence=row["confidence"],
                    evidence_count=row["evidence_count"],
                    first_observed=row["first_observed"],
                    last_observed=row["last_observed"],
                    source=row["source"],
                )

                if category == "expertise":
                    model.expertise[key] = UserExpertise(domain=key, level=attr)
                elif category == "preference":
                    model.preferences[key] = UserPreference(key=key, learned=attr)
                elif category == "trust":
                    trust_data = value if isinstance(value, dict) else {}
                    model.trust[key] = TrustDimension(
                        domain=key,
                        trust_level=LearnedAttribute(
                            value=trust_data.get("trust_level", 0.5),
                            confidence=attr.confidence,
                            evidence_count=attr.evidence_count,
                            source=attr.source,
                            first_observed=attr.first_observed,
                            last_observed=attr.last_observed,
                        ),
                        delegations_count=int(trust_data.get("delegations", 0)),
                        overrides_count=int(trust_data.get("overrides", 0)),
                    )
                elif category == "interest" and key.startswith("interest_"):
                    model.active_interests.append(attr)
                elif category == "state" and key == "current_focus":
                    model.current_focus = attr
                elif category == "temporal" and key == "active_hours":
                    if isinstance(value, dict):
                        model.active_hours = {int(k): v for k, v in value.items()}
                elif category == "temporal" and key == "active_days":
                    if isinstance(value, dict):
                        model.active_days = {int(k): v for k, v in value.items()}

            # Load beliefs
            belief_rows = conn.execute(
                "SELECT * FROM user_beliefs WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchall()
            for row in belief_rows:
                model.beliefs.append(
                    UserBelief(
                        topic=row["topic"],
                        stance=row["stance"],
                        confidence=row["confidence"],
                        evidence_count=row["evidence_count"],
                        last_expressed=row["last_expressed"],
                    )
                )

            return model

    # ── Introspection ────────────────────────────────────────────────

    def snapshot(self, tenant_id: str) -> dict[str, Any]:
        """Full serializable state for debugging."""
        model = self.get_model(tenant_id)
        return model.to_dict()

    def stats(self, tenant_id: str) -> dict[str, Any]:
        """Summary statistics."""
        model = self.get_model(tenant_id)
        return {
            "expertise_domains": len(model.expertise),
            "preferences_count": len(model.preferences),
            "beliefs_count": len(model.beliefs),
            "trust_domains": len(model.trust),
            "active_interests": len(model.active_interests),
            "current_focus": model.current_focus.value,
            "focus_confidence": round(model.current_focus.confidence, 3),
            "stress_level": round(model.stress_level, 3),
            "engagement_level": round(model.engagement_level, 3),
        }
