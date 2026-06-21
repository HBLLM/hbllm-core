"""Voice Authentication — speaker embedding-based identity verification.

Uses speaker embeddings from SpeakerIdNode to authenticate users:
    1. Enrollment: store N voice samples → compute average embedding
    2. Verification: compare incoming embedding against enrolled profiles
    3. Continuous re-verification during conversation
    4. Fallback to password/PIN if voice confidence is low

Links speaker_id → tenant_id → RBAC permissions for seamless auth.

Architecture:
    - Speaker embeddings are d-dimensional vectors (typically 192 or 256)
    - Cosine similarity for comparison
    - SQLite-backed profile storage
    - Anti-spoofing checks (optional, via liveness detection)
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Types ───────────────────────────────────────────────────────────


@dataclass
class VoiceProfile:
    """An enrolled voice profile for a user."""

    profile_id: str
    tenant_id: str
    display_name: str = ""
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sample_count: int = 0
    enrolled_at: float = field(default_factory=time.time)
    last_verified: float = 0.0
    verification_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "tenant_id": self.tenant_id,
            "display_name": self.display_name,
            "sample_count": self.sample_count,
            "enrolled_at": self.enrolled_at,
            "last_verified": self.last_verified,
            "verification_count": self.verification_count,
            "embedding_dim": len(self.embedding),
        }


@dataclass
class VerificationResult:
    """Result of a voice verification attempt."""

    verified: bool
    profile_id: str | None = None
    tenant_id: str | None = None
    similarity: float = 0.0
    confidence: str = "none"  # "high", "medium", "low", "none"
    method: str = "voice"  # "voice", "pin", "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "profile_id": self.profile_id,
            "tenant_id": self.tenant_id,
            "similarity": round(self.similarity, 4),
            "confidence": self.confidence,
            "method": self.method,
        }


# ── Voice Authenticator ─────────────────────────────────────────────────


class VoiceAuthenticator:
    """Speaker embedding-based authentication system.

    Usage::

        auth = VoiceAuthenticator(db_path="data/voice_profiles.db")
        await auth.init_db()

        # Enroll a user (accumulate samples)
        auth.enroll(
            profile_id="user1",
            tenant_id="household_01",
            embedding=speaker_embedding,
            display_name="Alice",
        )

        # Verify a speaker
        result = auth.verify(incoming_embedding)
        if result.verified:
            grant_access(result.tenant_id)
    """

    def __init__(
        self,
        db_path: str | Path = "data/voice_profiles.db",
        similarity_threshold: float = 0.85,
        high_confidence_threshold: float = 0.95,
        medium_confidence_threshold: float = 0.90,
        min_samples_for_verification: int = 3,
        re_verify_interval_s: float = 300.0,  # Re-verify every 5 minutes
    ) -> None:
        self.db_path = Path(db_path)
        self.similarity_threshold = similarity_threshold
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.min_samples = min_samples_for_verification
        self.re_verify_interval_s = re_verify_interval_s

        # In-memory profile cache
        self._profiles: dict[str, VoiceProfile] = {}

        # PIN fallback storage: profile_id → hashed PIN
        self._pin_hashes: dict[str, str] = {}

        # Session tracking: profile_id → last verified timestamp
        self._active_sessions: dict[str, float] = {}

        # Telemetry
        self._total_verifications = 0
        self._successful_verifications = 0
        self._failed_verifications = 0
        self._pin_fallbacks = 0

    async def init_db(self) -> None:
        """Initialize the voice profile database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS voice_profiles (
                    profile_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    display_name TEXT,
                    embedding BLOB NOT NULL,
                    sample_count INTEGER DEFAULT 1,
                    enrolled_at REAL NOT NULL,
                    last_verified REAL DEFAULT 0,
                    verification_count INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pin_auth (
                    profile_id TEXT PRIMARY KEY,
                    pin_hash TEXT NOT NULL,
                    FOREIGN KEY (profile_id) REFERENCES voice_profiles(profile_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_voice_tenant
                ON voice_profiles(tenant_id)
            """)
            conn.commit()

            # Load profiles into memory
            cursor = conn.execute(
                "SELECT profile_id, tenant_id, display_name, embedding, "
                "sample_count, enrolled_at, last_verified, verification_count "
                "FROM voice_profiles"
            )
            for row in cursor.fetchall():
                embedding = np.frombuffer(row[3], dtype=np.float32)
                self._profiles[row[0]] = VoiceProfile(
                    profile_id=row[0],
                    tenant_id=row[1],
                    display_name=row[2] or "",
                    embedding=embedding,
                    sample_count=row[4],
                    enrolled_at=row[5],
                    last_verified=row[6],
                    verification_count=row[7],
                )

            # Load PIN hashes
            cursor = conn.execute("SELECT profile_id, pin_hash FROM pin_auth")
            for row in cursor.fetchall():
                self._pin_hashes[row[0]] = row[1]

        finally:
            conn.close()

        logger.info(
            "VoiceAuthenticator initialized: %d profiles loaded",
            len(self._profiles),
        )

    def enroll(
        self,
        profile_id: str,
        tenant_id: str,
        embedding: np.ndarray,
        display_name: str = "",
    ) -> VoiceProfile:
        """Enroll or update a voice profile with a new sample.

        Each call adds a sample and recomputes the average embedding.
        Call multiple times (min 3) for robust enrollment.
        """
        embedding = self._normalize(embedding)

        if profile_id in self._profiles:
            # Update existing profile with running average
            profile = self._profiles[profile_id]
            n = profile.sample_count
            profile.embedding = (profile.embedding * n + embedding) / (n + 1)
            profile.embedding = self._normalize(profile.embedding)
            profile.sample_count += 1
            if display_name:
                profile.display_name = display_name
        else:
            # New profile
            profile = VoiceProfile(
                profile_id=profile_id,
                tenant_id=tenant_id,
                display_name=display_name,
                embedding=embedding,
                sample_count=1,
            )
            self._profiles[profile_id] = profile

        # Persist to database
        self._save_profile(profile)

        logger.info(
            "Voice enrollment: profile=%s, samples=%d",
            profile_id,
            profile.sample_count,
        )
        return profile

    def verify(self, embedding: np.ndarray) -> VerificationResult:
        """Verify a speaker against all enrolled profiles.

        Returns the best matching profile if similarity >= threshold.
        """
        self._total_verifications += 1

        if not self._profiles:
            self._failed_verifications += 1
            return VerificationResult(verified=False)

        embedding = self._normalize(embedding)
        best_match: VoiceProfile | None = None
        best_similarity = -1.0

        for profile in self._profiles.values():
            if profile.sample_count < self.min_samples:
                continue  # Not enough samples for reliable matching

            sim = self._cosine_similarity(embedding, profile.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_match = profile

        if best_match is None or best_similarity < self.similarity_threshold:
            self._failed_verifications += 1
            return VerificationResult(
                verified=False,
                similarity=best_similarity if best_match else 0.0,
            )

        # Determine confidence level
        if best_similarity >= self.high_confidence_threshold:
            confidence = "high"
        elif best_similarity >= self.medium_confidence_threshold:
            confidence = "medium"
        else:
            confidence = "low"

        # Update profile stats
        best_match.last_verified = time.time()
        best_match.verification_count += 1
        self._save_profile(best_match)

        # Track active session
        self._active_sessions[best_match.profile_id] = time.time()

        self._successful_verifications += 1

        return VerificationResult(
            verified=True,
            profile_id=best_match.profile_id,
            tenant_id=best_match.tenant_id,
            similarity=best_similarity,
            confidence=confidence,
        )

    def needs_re_verification(self, profile_id: str) -> bool:
        """Check if a profile needs re-verification based on time elapsed."""
        last = self._active_sessions.get(profile_id, 0.0)
        return (time.time() - last) > self.re_verify_interval_s

    # ── PIN Fallback ────────────────────────────────────────────────────

    def set_pin(self, profile_id: str, pin: str) -> None:
        """Set a fallback PIN for a profile."""
        pin_hash = hashlib.sha256(pin.encode()).hexdigest()
        self._pin_hashes[profile_id] = pin_hash

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO pin_auth (profile_id, pin_hash) VALUES (?, ?)",
                (profile_id, pin_hash),
            )
            conn.commit()
        finally:
            conn.close()

    def verify_pin(self, profile_id: str, pin: str) -> VerificationResult:
        """Verify a user by PIN (fallback when voice confidence is low)."""
        self._pin_fallbacks += 1
        pin_hash = hashlib.sha256(pin.encode()).hexdigest()

        if self._pin_hashes.get(profile_id) == pin_hash:
            profile = self._profiles.get(profile_id)
            if profile:
                self._active_sessions[profile_id] = time.time()
                return VerificationResult(
                    verified=True,
                    profile_id=profile_id,
                    tenant_id=profile.tenant_id,
                    similarity=1.0,
                    confidence="high",
                    method="pin",
                )

        return VerificationResult(verified=False, method="pin")

    # ── Profile Management ──────────────────────────────────────────────

    def get_profile(self, profile_id: str) -> VoiceProfile | None:
        """Get a voice profile by ID."""
        return self._profiles.get(profile_id)

    def list_profiles(self, tenant_id: str | None = None) -> list[VoiceProfile]:
        """List enrolled profiles, optionally filtered by tenant."""
        profiles = list(self._profiles.values())
        if tenant_id:
            profiles = [p for p in profiles if p.tenant_id == tenant_id]
        return profiles

    def delete_profile(self, profile_id: str) -> bool:
        """Delete a voice profile."""
        if profile_id not in self._profiles:
            return False

        del self._profiles[profile_id]
        self._pin_hashes.pop(profile_id, None)
        self._active_sessions.pop(profile_id, None)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM voice_profiles WHERE profile_id = ?", (profile_id,))
            conn.execute("DELETE FROM pin_auth WHERE profile_id = ?", (profile_id,))
            conn.commit()
        finally:
            conn.close()

        return True

    # ── Internal Helpers ────────────────────────────────────────────────

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding vector."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return vec
        return vec / norm

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized vectors."""
        return float(np.dot(a, b))

    def _save_profile(self, profile: VoiceProfile) -> None:
        """Persist a profile to SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO voice_profiles "
                "(profile_id, tenant_id, display_name, embedding, "
                "sample_count, enrolled_at, last_verified, verification_count) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    profile.profile_id,
                    profile.tenant_id,
                    profile.display_name,
                    profile.embedding.astype(np.float32).tobytes(),
                    profile.sample_count,
                    profile.enrolled_at,
                    profile.last_verified,
                    profile.verification_count,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict[str, Any]:
        """Voice authenticator statistics."""
        return {
            "enrolled_profiles": len(self._profiles),
            "active_sessions": len(self._active_sessions),
            "total_verifications": self._total_verifications,
            "successful_verifications": self._successful_verifications,
            "failed_verifications": self._failed_verifications,
            "pin_fallbacks": self._pin_fallbacks,
            "success_rate": (
                self._successful_verifications / self._total_verifications
                if self._total_verifications > 0
                else 0.0
            ),
        }
