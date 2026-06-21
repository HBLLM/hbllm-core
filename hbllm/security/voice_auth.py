"""Voice Authenticator — speaker embedding to tenant identity bridge.

Maps speaker voice embeddings (from ``SpeakerIdNode``) to HBLLM
tenant/user identities. Provides enrollment, verification, continuous
re-authentication, and anti-spoofing detection.

Architecture:
    Microphone → SpeakerIdNode → VoiceAuthenticator → RBAC/TenantGuard
                 (embedding)       (identity)           (permissions)

Bus Topics:
    security.voice.authenticated  — Successful voice authentication
    security.voice.failed         — Failed voice authentication
    security.voice.enrolled       — New voice profile enrolled
    security.voice.spoof_detected — Possible spoofing attempt

Usage::

    auth = VoiceAuthenticator()
    auth.enroll(embedding, tenant_id="t1", user_id="admin")

    result = auth.authenticate(new_embedding)
    if result.authenticated:
        print(f"Welcome, {result.user_id}!")
    elif result.fallback_required:
        print("Voice not recognised — please enter PIN.")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class VoiceProfile:
    """An enrolled voice identity."""

    speaker_id: str
    tenant_id: str
    user_id: str
    embedding: np.ndarray
    enrolled_at: float = field(default_factory=time.time)
    last_verified: float = 0.0
    verification_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker_id": self.speaker_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "enrolled_at": self.enrolled_at,
            "last_verified": self.last_verified,
            "verification_count": self.verification_count,
        }


@dataclass
class AuthResult:
    """Outcome of a voice authentication attempt."""

    authenticated: bool
    speaker_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    similarity: float = 0.0
    fallback_required: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "authenticated": self.authenticated,
            "speaker_id": self.speaker_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "similarity": round(self.similarity, 4),
            "fallback_required": self.fallback_required,
            "reason": self.reason,
        }


@dataclass
class VoiceAuthConfig:
    """Tunable voice authentication parameters."""

    # Cosine similarity threshold for positive identification
    similarity_threshold: float = 0.85

    # Continuous verification
    continuous_reauth_interval_s: float = 300.0  # 5 minutes
    continuous_similarity_threshold: float = 0.80  # Slightly relaxed

    # Anti-spoofing
    spoof_drift_threshold: float = 0.3  # Flag if embedding suddenly shifts
    max_failed_attempts: int = 5  # Lock after N failures


# ── Authenticator ────────────────────────────────────────────────────────────


class VoiceAuthenticator:
    """Maps voice embeddings to tenant/user identities.

    Args:
        config: Authentication parameters.
        bus: Optional MessageBus for event publishing.
    """

    def __init__(
        self,
        config: VoiceAuthConfig | None = None,
        bus: Any | None = None,
    ) -> None:
        self.config = config or VoiceAuthConfig()
        self.bus = bus

        self._profiles: dict[str, VoiceProfile] = {}  # speaker_id → profile
        self._active_sessions: dict[str, str] = {}  # session_id → speaker_id
        self._failed_attempts: dict[str, int] = {}  # session_id → count

        # Telemetry
        self._total_auth_attempts = 0
        self._successful_auths = 0
        self._failed_auths = 0
        self._spoof_detections = 0

    # ── Enrollment ───────────────────────────────────────────────────

    def enroll(
        self,
        embedding: np.ndarray,
        tenant_id: str,
        user_id: str,
        speaker_id: str | None = None,
    ) -> VoiceProfile:
        """Enroll a new voice profile.

        Args:
            embedding: Speaker embedding vector (typically 256-dim).
            tenant_id: Tenant to associate the voice with.
            user_id: User within the tenant.
            speaker_id: Optional explicit speaker ID; auto-generated if omitted.

        Returns:
            The created VoiceProfile.
        """
        if speaker_id is None:
            speaker_id = f"spk_{tenant_id}_{user_id}_{int(time.time())}"

        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        profile = VoiceProfile(
            speaker_id=speaker_id,
            tenant_id=tenant_id,
            user_id=user_id,
            embedding=embedding,
        )
        self._profiles[speaker_id] = profile
        logger.info(
            "Enrolled voice profile: speaker=%s tenant=%s user=%s",
            speaker_id,
            tenant_id,
            user_id,
        )
        return profile

    def unenroll(self, speaker_id: str) -> bool:
        """Remove a voice profile."""
        if speaker_id in self._profiles:
            del self._profiles[speaker_id]
            logger.info("Unenrolled speaker: %s", speaker_id)
            return True
        return False

    # ── Authentication ───────────────────────────────────────────────

    def authenticate(
        self,
        embedding: np.ndarray,
        session_id: str = "",
    ) -> AuthResult:
        """Authenticate a speaker against enrolled profiles.

        Args:
            embedding: Speaker embedding from current audio.
            session_id: Optional session context for lockout tracking.

        Returns:
            AuthResult with identity if matched, or fallback_required if not.
        """
        self._total_auth_attempts += 1

        # Check lockout
        if (
            session_id
            and self._failed_attempts.get(session_id, 0) >= self.config.max_failed_attempts
        ):
            self._failed_auths += 1
            return AuthResult(
                authenticated=False,
                fallback_required=True,
                reason=f"Locked out after {self.config.max_failed_attempts} failed attempts",
            )

        if not self._profiles:
            return AuthResult(
                authenticated=False,
                fallback_required=True,
                reason="No enrolled voice profiles",
            )

        # Normalize query embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Find best match
        best_similarity = -1.0
        best_profile: VoiceProfile | None = None

        for profile in self._profiles.values():
            similarity = float(np.dot(embedding, profile.embedding))
            if similarity > best_similarity:
                best_similarity = similarity
                best_profile = profile

        # Check threshold
        if best_profile and best_similarity >= self.config.similarity_threshold:
            best_profile.last_verified = time.time()
            best_profile.verification_count += 1
            self._successful_auths += 1

            # Track session
            if session_id:
                self._active_sessions[session_id] = best_profile.speaker_id
                self._failed_attempts.pop(session_id, None)

            logger.info(
                "Voice authenticated: %s (similarity=%.3f)",
                best_profile.speaker_id,
                best_similarity,
            )
            return AuthResult(
                authenticated=True,
                speaker_id=best_profile.speaker_id,
                tenant_id=best_profile.tenant_id,
                user_id=best_profile.user_id,
                similarity=best_similarity,
                reason="Voice match",
            )

        # Failed
        self._failed_auths += 1
        if session_id:
            self._failed_attempts[session_id] = self._failed_attempts.get(session_id, 0) + 1

        return AuthResult(
            authenticated=False,
            similarity=best_similarity,
            fallback_required=True,
            reason=f"Best similarity {best_similarity:.3f} below threshold "
            f"{self.config.similarity_threshold}",
        )

    # ── Continuous Verification ──────────────────────────────────────

    def verify_continuous(
        self,
        embedding: np.ndarray,
        session_id: str,
    ) -> bool:
        """Re-verify the speaker during an active session.

        Uses a slightly relaxed threshold for continuous verification
        (the speaker has already been identified once).

        Args:
            embedding: Current speaker embedding.
            session_id: Active session to verify against.

        Returns:
            True if the speaker is still the authenticated user.
        """
        speaker_id = self._active_sessions.get(session_id)
        if not speaker_id:
            return False

        profile = self._profiles.get(speaker_id)
        if not profile:
            return False

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        similarity = float(np.dot(embedding, profile.embedding))

        # Anti-spoofing: detect sudden embedding drift
        if similarity < (1.0 - self.config.spoof_drift_threshold):
            self._spoof_detections += 1
            logger.warning(
                "Possible voice spoofing detected: session=%s similarity=%.3f",
                session_id,
                similarity,
            )
            # Invalidate session
            self._active_sessions.pop(session_id, None)
            return False

        if similarity >= self.config.continuous_similarity_threshold:
            profile.last_verified = time.time()
            return True

        return False

    # ── Session Management ───────────────────────────────────────────

    def end_session(self, session_id: str) -> None:
        """End an authenticated session."""
        self._active_sessions.pop(session_id, None)
        self._failed_attempts.pop(session_id, None)

    def get_session_identity(self, session_id: str) -> VoiceProfile | None:
        """Get the authenticated profile for an active session."""
        speaker_id = self._active_sessions.get(session_id)
        if speaker_id:
            return self._profiles.get(speaker_id)
        return None

    # ── Profile Management ───────────────────────────────────────────

    def list_profiles(self, tenant_id: str | None = None) -> list[VoiceProfile]:
        """List enrolled voice profiles, optionally filtered by tenant."""
        profiles = list(self._profiles.values())
        if tenant_id:
            profiles = [p for p in profiles if p.tenant_id == tenant_id]
        return profiles

    def get_profile(self, speaker_id: str) -> VoiceProfile | None:
        """Get a specific voice profile."""
        return self._profiles.get(speaker_id)

    # ── Telemetry ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Authenticator statistics."""
        return {
            "enrolled_profiles": len(self._profiles),
            "active_sessions": len(self._active_sessions),
            "total_auth_attempts": self._total_auth_attempts,
            "successful_auths": self._successful_auths,
            "failed_auths": self._failed_auths,
            "spoof_detections": self._spoof_detections,
        }
