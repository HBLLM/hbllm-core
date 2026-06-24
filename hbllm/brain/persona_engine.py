"""
PersonaEngine — Persistent personality profiles for intelligent assistant interactions.

Unlike IdentityNode (which handles ethics/constraints), PersonaEngine manages
the *style* of interaction: formality level, humor, verbosity, emoji usage,
empathy, and how these adapt over time based on user feedback and detected
emotional state.

Each tenant/user gets their own persona profile that evolves with interactions.
The engine exposes a `modulate(text, context)` method that downstream nodes
(ExpressionStream, Decision) use to style their output.

Integrations:
    UserModelEngine  → Syncs verbosity, technical_depth, stress from learned preferences

Bus Topics:
    persona.get        → Returns current persona profile
    persona.feedback   → User feedback adjusts persona traits
    emotion.state      → Emotion events modulate response style
    user.model.updated → Sync persona traits from learned user preferences
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Persona Profile ──────────────────────────────────────────────────────────


@dataclass
class PersonaTrait:
    """A single personality dimension with a value in [0.0, 1.0]."""

    name: str
    value: float = 0.5  # 0.0 = minimum, 1.0 = maximum
    momentum: float = 0.0  # Running average of feedback direction
    update_count: int = 0

    def nudge(self, direction: float, learning_rate: float = 0.05) -> None:
        """Adjust trait value based on feedback (-1.0 to +1.0)."""
        self.momentum = 0.9 * self.momentum + 0.1 * direction
        self.value = max(0.0, min(1.0, self.value + learning_rate * direction))
        self.update_count += 1


@dataclass
class PersonaProfile:
    """Complete personality profile for a tenant/user."""

    tenant_id: str
    name: str = "Sentra"
    # Core traits — each is 0.0 to 1.0
    formality: PersonaTrait = field(default_factory=lambda: PersonaTrait("formality", 0.4))
    humor: PersonaTrait = field(default_factory=lambda: PersonaTrait("humor", 0.3))
    verbosity: PersonaTrait = field(default_factory=lambda: PersonaTrait("verbosity", 0.5))
    emoji_usage: PersonaTrait = field(default_factory=lambda: PersonaTrait("emoji_usage", 0.2))
    empathy: PersonaTrait = field(default_factory=lambda: PersonaTrait("empathy", 0.6))
    enthusiasm: PersonaTrait = field(default_factory=lambda: PersonaTrait("enthusiasm", 0.5))
    technical_depth: PersonaTrait = field(
        default_factory=lambda: PersonaTrait("technical_depth", 0.6)
    )
    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    interaction_count: int = 0

    @property
    def traits(self) -> dict[str, PersonaTrait]:
        """All trait objects as a dict."""
        return {
            "formality": self.formality,
            "humor": self.humor,
            "verbosity": self.verbosity,
            "emoji_usage": self.emoji_usage,
            "empathy": self.empathy,
            "enthusiasm": self.enthusiasm,
            "technical_depth": self.technical_depth,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage."""
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "traits": {k: asdict(v) for k, v in self.traits.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PersonaProfile:
        """Deserialize from storage."""
        profile = cls(tenant_id=data["tenant_id"], name=data.get("name", "Sentra"))
        profile.created_at = data.get("created_at", time.time())
        profile.updated_at = data.get("updated_at", time.time())
        profile.interaction_count = data.get("interaction_count", 0)
        for trait_name, trait_data in data.get("traits", {}).items():
            if hasattr(profile, trait_name):
                trait = PersonaTrait(**trait_data)
                setattr(profile, trait_name, trait)
        return profile

    def to_system_prompt_fragment(self) -> str:
        """Generate a system prompt fragment that captures the persona style."""
        parts = []

        # Formality
        if self.formality.value < 0.3:
            parts.append("Use a casual, friendly tone. Contractions are fine.")
        elif self.formality.value > 0.7:
            parts.append("Use a professional, formal tone.")

        # Verbosity
        if self.verbosity.value < 0.3:
            parts.append("Be extremely concise — bullet points over paragraphs.")
        elif self.verbosity.value > 0.7:
            parts.append("Provide detailed, thorough explanations with examples.")

        # Humor
        if self.humor.value > 0.6:
            parts.append("Feel free to use light humor and wit when appropriate.")

        # Empathy
        if self.empathy.value > 0.6:
            parts.append("Be warm and empathetic. Acknowledge the user's feelings.")

        # Technical depth
        if self.technical_depth.value > 0.7:
            parts.append("Assume technical expertise — use precise terminology.")
        elif self.technical_depth.value < 0.3:
            parts.append("Explain things simply — avoid jargon.")

        # Emoji
        if self.emoji_usage.value > 0.5:
            parts.append("Use relevant emojis to make responses more engaging.")

        # Enthusiasm
        if self.enthusiasm.value > 0.7:
            parts.append("Be enthusiastic and encouraging!")

        return " ".join(parts) if parts else ""


# ── Emotion-Aware Modulation ─────────────────────────────────────────────────

# Maps detected emotion states to temporary trait overrides
EMOTION_MODULATIONS: dict[str, dict[str, float]] = {
    "stressed": {"verbosity": -0.3, "empathy": +0.2, "humor": -0.2},
    "frustrated": {"verbosity": -0.2, "empathy": +0.3, "formality": -0.1},
    "curious": {"verbosity": +0.2, "technical_depth": +0.2, "enthusiasm": +0.1},
    "happy": {"humor": +0.1, "enthusiasm": +0.2, "emoji_usage": +0.1},
    "confused": {"verbosity": +0.2, "technical_depth": -0.2, "empathy": +0.1},
    "bored": {"humor": +0.2, "enthusiasm": +0.3, "verbosity": -0.1},
    "neutral": {},
}


# ── PersonaEngine ────────────────────────────────────────────────────────────


class PersonaEngine:
    """
    Manages persistent personality profiles and emotion-aware style modulation.

    Usage:
        engine = PersonaEngine(storage_dir="data/personas")

        # Get persona for a tenant
        persona = engine.get_profile("tenant_123")

        # Generate system prompt fragment
        style = persona.to_system_prompt_fragment()

        # Apply user feedback
        engine.apply_feedback("tenant_123", "too_verbose")

        # Modulate based on emotion
        style = engine.get_modulated_prompt("tenant_123", emotion="stressed")
    """

    def __init__(self, storage_dir: str | Path = "data/personas", user_model: Any | None = None) -> None:
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, PersonaProfile] = {}
        self._user_model = user_model  # Optional UserModelEngine for preference sync

        # Load existing profiles
        self._load_all()
        logger.info(
            "PersonaEngine initialized with %d profiles from %s (user_model=%s)",
            len(self._profiles),
            self._storage_dir,
            "connected" if user_model else "none",
        )

    def _load_all(self) -> None:
        """Load all persona profiles from disk."""
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                profile = PersonaProfile.from_dict(data)
                self._profiles[profile.tenant_id] = profile
            except Exception as e:
                logger.warning("Failed to load persona from %s: %s", path, e)

    def _save(self, profile: PersonaProfile) -> None:
        """Persist a profile to disk."""
        path = self._storage_dir / f"{profile.tenant_id}.json"
        try:
            path.write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to save persona for %s: %s", profile.tenant_id, e)

    def get_profile(self, tenant_id: str) -> PersonaProfile:
        """Get or create a persona profile for a tenant."""
        if tenant_id not in self._profiles:
            self._profiles[tenant_id] = PersonaProfile(tenant_id=tenant_id)
            self._save(self._profiles[tenant_id])
            logger.info("Created new persona profile for tenant '%s'", tenant_id)
        return self._profiles[tenant_id]

    def apply_feedback(self, tenant_id: str, feedback_type: str) -> PersonaProfile:
        """
        Apply user feedback to adjust persona traits.

        Supported feedback types:
            too_verbose, too_concise, too_formal, too_casual,
            more_humor, less_humor, more_technical, less_technical,
            more_empathy, more_enthusiasm, positive, negative
        """
        profile = self.get_profile(tenant_id)

        adjustments: dict[str, float] = {
            "too_verbose": {"verbosity": -0.15},
            "too_concise": {"verbosity": +0.15},
            "too_formal": {"formality": -0.15},
            "too_casual": {"formality": +0.15},
            "more_humor": {"humor": +0.1},
            "less_humor": {"humor": -0.1},
            "more_technical": {"technical_depth": +0.1},
            "less_technical": {"technical_depth": -0.1},
            "more_empathy": {"empathy": +0.1},
            "more_enthusiasm": {"enthusiasm": +0.1},
            "positive": {"humor": +0.02, "enthusiasm": +0.02},
            "negative": {"empathy": +0.03, "verbosity": -0.02},
        }.get(feedback_type, {})

        for trait_name, direction in adjustments.items():
            trait = profile.traits.get(trait_name)
            if trait:
                trait.nudge(direction)

        profile.updated_at = time.time()
        profile.interaction_count += 1
        self._save(profile)

        logger.debug(
            "Applied feedback '%s' to tenant '%s' (interaction #%d)",
            feedback_type,
            tenant_id,
            profile.interaction_count,
        )
        return profile

    def get_modulated_prompt(
        self,
        tenant_id: str,
        emotion: str | None = None,
    ) -> str:
        """
        Generate a style-modulated system prompt fragment.

        Combines the persistent persona profile with temporary
        emotion-based modulations for the current interaction.
        """
        profile = self.get_profile(tenant_id)
        base_prompt = profile.to_system_prompt_fragment()

        if not emotion or emotion not in EMOTION_MODULATIONS:
            return base_prompt

        # Apply temporary emotion modulations
        modulations = EMOTION_MODULATIONS[emotion]
        emotion_parts = []

        if modulations.get("verbosity", 0) < 0:
            emotion_parts.append("The user seems pressed for time — keep it brief.")
        elif modulations.get("verbosity", 0) > 0:
            emotion_parts.append("The user wants to understand deeply — elaborate freely.")

        if modulations.get("empathy", 0) > 0:
            emotion_parts.append("The user may need extra support — be gentle and reassuring.")

        if emotion_parts:
            return f"{base_prompt} {' '.join(emotion_parts)}".strip()

        return base_prompt

    def sync_from_user_model(self, tenant_id: str) -> bool:
        """Sync persona traits from UserModel learned preferences.

        Bridges biometric/behavioral learning (UserModel) into style
        modulation (PersonaEngine). Called when user.model.updated fires.

        Returns True if any traits were updated.
        """
        if not self._user_model:
            return False

        try:
            model = self._user_model.get_model(tenant_id)
        except Exception as e:
            logger.debug("Failed to read UserModel for persona sync: %s", e)
            return False

        profile = self.get_profile(tenant_id)
        changed = False

        # Sync verbosity preference
        verbosity_pref = model.preferences.get("verbosity")
        if verbosity_pref and verbosity_pref.confidence > 0.5:
            target = 0.3 if verbosity_pref.value == "concise" else 0.7
            if abs(profile.verbosity.value - target) > 0.1:
                profile.verbosity.value = 0.7 * profile.verbosity.value + 0.3 * target
                changed = True

        # Sync technical depth from expertise
        if model.expertise:
            avg_expertise = sum(e.level for e in model.expertise.values()) / len(model.expertise)
            target_depth = min(1.0, 0.3 + avg_expertise * 0.7)
            if abs(profile.technical_depth.value - target_depth) > 0.1:
                profile.technical_depth.value = (
                    0.7 * profile.technical_depth.value + 0.3 * target_depth
                )
                changed = True

        # Sync stress → empathy/brevity
        if model.stress_level > 0.7:
            if profile.empathy.value < 0.7:
                profile.empathy.value = min(1.0, profile.empathy.value + 0.05)
                changed = True
            if profile.verbosity.value > 0.4:
                profile.verbosity.value = max(0.2, profile.verbosity.value - 0.05)
                changed = True

        # Sync formality preference
        formality_pref = model.preferences.get("formality")
        if formality_pref and formality_pref.confidence > 0.5:
            target = 0.3 if formality_pref.value == "casual" else 0.7
            if abs(profile.formality.value - target) > 0.1:
                profile.formality.value = 0.7 * profile.formality.value + 0.3 * target
                changed = True

        if changed:
            profile.updated_at = time.time()
            self._save(profile)
            logger.info(
                "Synced persona from UserModel for tenant '%s': "
                "verbosity=%.2f, technical_depth=%.2f, empathy=%.2f",
                tenant_id,
                profile.verbosity.value,
                profile.technical_depth.value,
                profile.empathy.value,
            )

        return changed

    def record_interaction(self, tenant_id: str) -> None:
        """Record that an interaction occurred (for tracking engagement)."""
        profile = self.get_profile(tenant_id)
        profile.interaction_count += 1
        profile.updated_at = time.time()
        # Auto-save periodically (every 10 interactions)
        if profile.interaction_count % 10 == 0:
            self._save(profile)

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all persona profiles (for admin/debug)."""
        return [
            {
                "tenant_id": p.tenant_id,
                "name": p.name,
                "interaction_count": p.interaction_count,
                "traits": {k: round(v.value, 2) for k, v in p.traits.items()},
            }
            for p in self._profiles.values()
        ]

    def reset_profile(self, tenant_id: str) -> PersonaProfile:
        """Reset a tenant's persona to defaults."""
        self._profiles[tenant_id] = PersonaProfile(tenant_id=tenant_id)
        self._save(self._profiles[tenant_id])
        logger.info("Reset persona profile for tenant '%s'", tenant_id)
        return self._profiles[tenant_id]
