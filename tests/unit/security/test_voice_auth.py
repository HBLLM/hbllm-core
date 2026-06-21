"""Tests for VoiceAuthenticator — voice-based identity verification."""

import numpy as np
import pytest

from hbllm.security.voice_auth import (
    VoiceAuthConfig,
    VoiceAuthenticator,
)


def _random_embedding(dim: int = 256) -> np.ndarray:
    """Generate a random unit-normalised embedding."""
    v = np.random.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _similar_embedding(base: np.ndarray, noise: float = 0.02) -> np.ndarray:
    """Generate an embedding similar to *base* with small noise."""
    v = base + np.random.randn(*base.shape).astype(np.float32) * noise
    return v / np.linalg.norm(v)


@pytest.fixture
def auth():
    return VoiceAuthenticator()


@pytest.fixture
def enrolled_auth():
    """Authenticator with one enrolled profile."""
    a = VoiceAuthenticator()
    emb = _random_embedding()
    a.enroll(emb, tenant_id="t1", user_id="alice", speaker_id="spk_alice")
    return a, emb


# ── Enrollment Tests ─────────────────────────────────────────────────────────


class TestEnrollment:
    def test_enroll(self, auth):
        emb = _random_embedding()
        profile = auth.enroll(emb, tenant_id="t1", user_id="alice")
        assert profile.tenant_id == "t1"
        assert profile.user_id == "alice"
        assert profile.speaker_id is not None

    def test_enroll_with_explicit_id(self, auth):
        emb = _random_embedding()
        profile = auth.enroll(emb, tenant_id="t1", user_id="bob", speaker_id="custom_id")
        assert profile.speaker_id == "custom_id"

    def test_unenroll(self, auth):
        emb = _random_embedding()
        profile = auth.enroll(emb, tenant_id="t1", user_id="alice")
        assert auth.unenroll(profile.speaker_id)
        assert not auth.unenroll(profile.speaker_id)  # Already removed

    def test_list_profiles(self, auth):
        auth.enroll(_random_embedding(), tenant_id="t1", user_id="a")
        auth.enroll(_random_embedding(), tenant_id="t2", user_id="b")
        assert len(auth.list_profiles()) == 2
        assert len(auth.list_profiles(tenant_id="t1")) == 1

    def test_profile_to_dict(self, auth):
        profile = auth.enroll(_random_embedding(), "t1", "alice", "spk1")
        d = profile.to_dict()
        assert d["speaker_id"] == "spk1"
        assert d["tenant_id"] == "t1"


# ── Authentication Tests ─────────────────────────────────────────────────────


class TestAuthentication:
    def test_successful_auth(self, enrolled_auth):
        auth, emb = enrolled_auth
        # Same embedding should match
        result = auth.authenticate(emb)
        assert result.authenticated
        assert result.speaker_id == "spk_alice"
        assert result.tenant_id == "t1"
        assert result.similarity >= 0.85

    def test_similar_embedding_matches(self, enrolled_auth):
        auth, emb = enrolled_auth
        similar = _similar_embedding(emb, noise=0.02)
        result = auth.authenticate(similar)
        assert result.authenticated

    def test_different_embedding_fails(self, enrolled_auth):
        auth, _ = enrolled_auth
        different = _random_embedding()
        result = auth.authenticate(different)
        assert not result.authenticated
        assert result.fallback_required

    def test_no_profiles_returns_fallback(self, auth):
        result = auth.authenticate(_random_embedding())
        assert not result.authenticated
        assert result.fallback_required
        assert "No enrolled" in result.reason

    def test_auth_result_to_dict(self, enrolled_auth):
        auth, emb = enrolled_auth
        result = auth.authenticate(emb)
        d = result.to_dict()
        assert "authenticated" in d
        assert "similarity" in d


# ── Lockout Tests ────────────────────────────────────────────────────────────


class TestLockout:
    def test_lockout_after_max_failures(self):
        auth = VoiceAuthenticator(config=VoiceAuthConfig(max_failed_attempts=3))
        auth.enroll(_random_embedding(), "t1", "alice")

        for _ in range(3):
            auth.authenticate(_random_embedding(), session_id="sess1")

        result = auth.authenticate(_random_embedding(), session_id="sess1")
        assert not result.authenticated
        assert "Locked out" in result.reason

    def test_successful_auth_clears_failures(self, enrolled_auth):
        auth, emb = enrolled_auth
        # Fail once
        auth.authenticate(_random_embedding(), session_id="s1")
        # Succeed
        result = auth.authenticate(emb, session_id="s1")
        assert result.authenticated
        # Should not be locked
        result2 = auth.authenticate(emb, session_id="s1")
        assert result2.authenticated


# ── Continuous Verification Tests ────────────────────────────────────────────


class TestContinuousVerification:
    def test_continuous_verify_same_speaker(self, enrolled_auth):
        auth, emb = enrolled_auth
        auth.authenticate(emb, session_id="s1")  # Initial auth

        similar = _similar_embedding(emb, noise=0.01)
        assert auth.verify_continuous(similar, session_id="s1")

    def test_continuous_verify_different_speaker(self, enrolled_auth):
        auth, emb = enrolled_auth
        auth.authenticate(emb, session_id="s1")

        different = _random_embedding()
        result = auth.verify_continuous(different, session_id="s1")
        assert not result

    def test_continuous_verify_no_session(self, auth):
        assert not auth.verify_continuous(_random_embedding(), session_id="nonexistent")


# ── Anti-Spoofing Tests ──────────────────────────────────────────────────────


class TestAntiSpoofing:
    def test_spoof_detection_invalidates_session(self, enrolled_auth):
        auth, emb = enrolled_auth
        auth.authenticate(emb, session_id="s1")

        # Sudden embedding drift (very different voice)
        spoof = _random_embedding()
        auth.verify_continuous(spoof, session_id="s1")

        # Session should be invalidated
        assert auth.get_session_identity("s1") is None
        assert auth.stats()["spoof_detections"] >= 1


# ── Session Management Tests ────────────────────────────────────────────────


class TestSessionManagement:
    def test_end_session(self, enrolled_auth):
        auth, emb = enrolled_auth
        auth.authenticate(emb, session_id="s1")
        assert auth.get_session_identity("s1") is not None

        auth.end_session("s1")
        assert auth.get_session_identity("s1") is None

    def test_get_session_identity(self, enrolled_auth):
        auth, emb = enrolled_auth
        auth.authenticate(emb, session_id="s1")

        profile = auth.get_session_identity("s1")
        assert profile is not None
        assert profile.speaker_id == "spk_alice"


# ── Stats Tests ──────────────────────────────────────────────────────────────


class TestStats:
    def test_stats_tracking(self, enrolled_auth):
        auth, emb = enrolled_auth
        auth.authenticate(emb)
        auth.authenticate(_random_embedding())

        stats = auth.stats()
        assert stats["enrolled_profiles"] == 1
        assert stats["total_auth_attempts"] == 2
        assert stats["successful_auths"] >= 1
        assert stats["failed_auths"] >= 1
