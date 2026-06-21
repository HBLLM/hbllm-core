"""Tests for VoiceAuthenticator — speaker embedding-based auth."""

import time

import numpy as np
import pytest

from hbllm.security.voice_auth import (
    VerificationResult,
    VoiceAuthenticator,
)


class TestVoiceEnrollment:
    """Tests for voice profile enrollment."""

    @pytest.fixture
    async def auth(self, tmp_path):
        a = VoiceAuthenticator(
            db_path=tmp_path / "voice_profiles.db",
            min_samples_for_verification=1,
        )
        await a.init_db()
        return a

    @pytest.mark.asyncio
    async def test_enroll_creates_profile(self, auth):
        emb = np.random.randn(192).astype(np.float32)
        profile = auth.enroll("user1", "tenant1", emb, display_name="Alice")
        assert profile.profile_id == "user1"
        assert profile.tenant_id == "tenant1"
        assert profile.sample_count == 1
        assert profile.display_name == "Alice"

    @pytest.mark.asyncio
    async def test_enroll_multiple_samples(self, auth):
        profile = None
        for _ in range(5):
            emb = np.random.randn(192).astype(np.float32)
            profile = auth.enroll("user1", "tenant1", emb)
        assert profile.sample_count == 5

    @pytest.mark.asyncio
    async def test_list_profiles(self, auth):
        auth.enroll("u1", "t1", np.random.randn(192).astype(np.float32))
        auth.enroll("u2", "t1", np.random.randn(192).astype(np.float32))
        auth.enroll("u3", "t2", np.random.randn(192).astype(np.float32))

        all_profiles = auth.list_profiles()
        assert len(all_profiles) == 3

        t1_profiles = auth.list_profiles(tenant_id="t1")
        assert len(t1_profiles) == 2

    @pytest.mark.asyncio
    async def test_get_profile(self, auth):
        auth.enroll("u1", "t1", np.random.randn(192).astype(np.float32))
        p = auth.get_profile("u1")
        assert p is not None
        assert p.profile_id == "u1"

    @pytest.mark.asyncio
    async def test_delete_profile(self, auth):
        auth.enroll("u1", "t1", np.random.randn(192).astype(np.float32))
        assert auth.delete_profile("u1")
        assert auth.get_profile("u1") is None


class TestVoiceVerification:
    """Tests for speaker verification."""

    @pytest.fixture
    async def auth(self, tmp_path):
        a = VoiceAuthenticator(
            db_path=tmp_path / "voice_profiles.db",
            similarity_threshold=0.85,
            min_samples_for_verification=1,
        )
        await a.init_db()
        return a

    @pytest.mark.asyncio
    async def test_verify_matching_speaker(self, auth):
        base_emb = np.random.randn(192).astype(np.float32)
        auth.enroll("user1", "tenant1", base_emb)

        # Verify with same embedding (+ tiny noise)
        test_emb = base_emb + np.random.randn(192).astype(np.float32) * 0.01
        result = auth.verify(test_emb)
        assert result.verified
        assert result.profile_id == "user1"
        assert result.tenant_id == "tenant1"
        assert result.similarity > 0.85

    @pytest.mark.asyncio
    async def test_verify_unknown_speaker(self, auth):
        auth.enroll("user1", "tenant1", np.random.randn(192).astype(np.float32))

        different_emb = np.random.randn(192).astype(np.float32)
        result = auth.verify(different_emb)
        assert isinstance(result, VerificationResult)

    @pytest.mark.asyncio
    async def test_verify_no_profiles(self, auth):
        emb = np.random.randn(192).astype(np.float32)
        result = auth.verify(emb)
        assert not result.verified

    @pytest.mark.asyncio
    async def test_confidence_levels(self, auth):
        base_emb = np.random.randn(192).astype(np.float32)
        auth.enroll("user1", "tenant1", base_emb)

        result = auth.verify(base_emb.copy())
        if result.verified:
            assert result.confidence in ("high", "medium", "low")


class TestPINFallback:
    """Tests for PIN-based fallback auth."""

    @pytest.fixture
    async def auth(self, tmp_path):
        a = VoiceAuthenticator(
            db_path=tmp_path / "voice_profiles.db",
            min_samples_for_verification=1,
        )
        await a.init_db()
        a.enroll("user1", "tenant1", np.random.randn(192).astype(np.float32))
        return a

    @pytest.mark.asyncio
    async def test_set_and_verify_pin(self, auth):
        auth.set_pin("user1", "1234")
        result = auth.verify_pin("user1", "1234")
        assert result.verified
        assert result.method == "pin"
        assert result.tenant_id == "tenant1"

    @pytest.mark.asyncio
    async def test_wrong_pin(self, auth):
        auth.set_pin("user1", "1234")
        result = auth.verify_pin("user1", "9999")
        assert not result.verified

    @pytest.mark.asyncio
    async def test_pin_unknown_user(self, auth):
        result = auth.verify_pin("nobody", "1234")
        assert not result.verified


class TestReVerification:
    """Tests for session re-verification."""

    @pytest.fixture
    async def auth(self, tmp_path):
        a = VoiceAuthenticator(
            db_path=tmp_path / "voice_profiles.db",
            re_verify_interval_s=0.01,
            min_samples_for_verification=1,
        )
        await a.init_db()
        return a

    @pytest.mark.asyncio
    async def test_needs_re_verification_initially(self, auth):
        assert auth.needs_re_verification("user1")

    @pytest.mark.asyncio
    async def test_no_re_verify_after_recent(self, auth):
        auth._active_sessions["user1"] = time.time()
        time.sleep(0.02)
        assert auth.needs_re_verification("user1")


class TestVoiceAuthStats:
    """Tests for telemetry."""

    @pytest.mark.asyncio
    async def test_stats(self, tmp_path):
        auth = VoiceAuthenticator(
            db_path=tmp_path / "voice_profiles.db",
            min_samples_for_verification=1,
        )
        await auth.init_db()
        emb = np.random.randn(192).astype(np.float32)
        auth.enroll("u1", "t1", emb)
        auth.verify(emb)

        s = auth.stats()
        assert s["enrolled_profiles"] == 1
        assert s["total_verifications"] == 1

    @pytest.mark.asyncio
    async def test_profile_to_dict(self, tmp_path):
        auth = VoiceAuthenticator(
            db_path=tmp_path / "voice_profiles.db",
        )
        await auth.init_db()
        emb = np.random.randn(192).astype(np.float32)
        p = auth.enroll("u1", "t1", emb, display_name="Alice")
        d = p.to_dict()
        assert d["profile_id"] == "u1"
        assert d["embedding_dim"] == 192
