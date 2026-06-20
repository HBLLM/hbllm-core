"""Integration tests for Security subsystem — Encryption, TenantGuard, CSRF."""

import base64
import secrets

import pytest

from hbllm.security.encryption import EncryptionVault, _derive_key, _generate_key
from hbllm.security.tenant_guard import (
    TenantContext,
    TenantGuardMode,
    TenantIsolationError,
    get_current_identity,
    get_current_tenant,
    require_identity,
    require_tenant,
)

# ── Encryption Vault Integration ─────────────────────────────────────────────


class TestEncryptionVaultIntegration:
    """Test EncryptionVault encrypt/decrypt lifecycle and key management."""

    def test_encrypt_decrypt_round_trip(self):
        vault = EncryptionVault()
        plaintext = "sensitive-api-key-abc123"
        encrypted = vault.encrypt(plaintext)

        assert encrypted != plaintext
        assert vault.decrypt(encrypted) == plaintext

    def test_different_keys_produce_different_ciphertext(self):
        v1 = EncryptionVault()
        v2 = EncryptionVault()

        plaintext = "same-input"
        e1 = v1.encrypt(plaintext)
        e2 = v2.encrypt(plaintext)

        assert e1 != e2

    def test_wrong_key_raises(self):
        v1 = EncryptionVault()
        v2 = EncryptionVault()

        encrypted = v1.encrypt("secret")
        with pytest.raises(ValueError, match="tampered"):
            v2.decrypt(encrypted)

    def test_encrypt_decrypt_dict(self):
        vault = EncryptionVault()
        data = {"host": "db.example.com", "port": 5432, "password": "s3cr3t"}
        encrypted = vault.encrypt_dict(data)
        decrypted = vault.decrypt_dict(encrypted)

        assert decrypted["host"] == "db.example.com"
        assert decrypted["port"] == 5432
        assert decrypted["password"] == "s3cr3t"

    def test_key_fingerprint(self):
        vault = EncryptionVault()
        fp = vault.key_fingerprint
        assert isinstance(fp, str)
        assert len(fp) == 12

    def test_from_key_file_creates_and_loads(self, tmp_path):
        key_file = tmp_path / "encryption.key"
        v1 = EncryptionVault.from_key_file(str(key_file))
        assert key_file.exists()

        # Ensure file permissions are restrictive (owner-only)
        mode = key_file.stat().st_mode
        assert (mode & 0o077) == 0  # No group or other permissions

        # Reload from same file
        v2 = EncryptionVault.from_key_file(str(key_file))
        encrypted = v1.encrypt("test-persistence")
        assert v2.decrypt(encrypted) == "test-persistence"

    def test_from_password(self):
        salt = secrets.token_bytes(16)
        v1 = EncryptionVault.from_password("my-password", salt=salt)
        v2 = EncryptionVault.from_password("my-password", salt=salt)

        encrypted = v1.encrypt("password-derived-encryption")
        assert v2.decrypt(encrypted) == "password-derived-encryption"

    def test_from_password_different_passwords_differ(self):
        salt = secrets.token_bytes(16)
        v1 = EncryptionVault.from_password("password-a", salt=salt)
        v2 = EncryptionVault.from_password("password-b", salt=salt)

        encrypted = v1.encrypt("test")
        with pytest.raises(ValueError):
            v2.decrypt(encrypted)

    def test_from_env(self, monkeypatch):
        key = _generate_key()
        monkeypatch.setenv("HBLLM_ENCRYPTION_KEY", key.decode())

        vault = EncryptionVault.from_env()
        encrypted = vault.encrypt("env-test")
        assert vault.decrypt(encrypted) == "env-test"

    def test_from_env_missing_raises(self, monkeypatch):
        monkeypatch.delenv("HBLLM_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ValueError, match="not set"):
            EncryptionVault.from_env()

    def test_key_rotation(self):
        old_vault = EncryptionVault()
        encrypted_old = old_vault.encrypt("rotate-me")

        new_vault = old_vault.rotate_key()
        # Old vault can still decrypt old data
        assert old_vault.decrypt(encrypted_old) == "rotate-me"

        # New vault can't decrypt old data
        with pytest.raises(ValueError):
            new_vault.decrypt(encrypted_old)

        # But new vault encrypts with new key
        encrypted_new = new_vault.encrypt("new-data")
        assert new_vault.decrypt(encrypted_new) == "new-data"

    def test_rotate_and_reencrypt(self):
        old_vault = EncryptionVault()
        tokens = [old_vault.encrypt(f"secret-{i}") for i in range(5)]

        new_vault, reencrypted_gen = old_vault.rotate_and_reencrypt(tokens)
        reencrypted = list(reencrypted_gen)

        assert len(reencrypted) == 5
        # Verify new vault can decrypt re-encrypted data
        for i, token in enumerate(reencrypted):
            assert new_vault.decrypt(token) == f"secret-{i}"

    def test_empty_string(self):
        vault = EncryptionVault()
        encrypted = vault.encrypt("")
        assert vault.decrypt(encrypted) == ""

    def test_unicode_content(self):
        vault = EncryptionVault()
        text = "日本語テスト 🔐 مرحبا"
        encrypted = vault.encrypt(text)
        assert vault.decrypt(encrypted) == text

    def test_large_payload(self):
        vault = EncryptionVault()
        text = "x" * 100_000
        encrypted = vault.encrypt(text)
        assert vault.decrypt(encrypted) == text


# ── Key Derivation Tests ─────────────────────────────────────────────────────


class TestKeyDerivation:
    """Test PBKDF2 key derivation."""

    def test_deterministic(self):
        salt = b"fixed-salt-for-test"
        k1 = _derive_key("password", salt)
        k2 = _derive_key("password", salt)
        assert k1 == k2

    def test_different_salts_differ(self):
        k1 = _derive_key("password", b"salt-a")
        k2 = _derive_key("password", b"salt-b")
        assert k1 != k2

    def test_key_length(self):
        key = _derive_key("test", b"salt")
        assert len(key) == 32

    def test_generated_key_valid_base64(self):
        key = _generate_key()
        decoded = base64.urlsafe_b64decode(key)
        assert len(decoded) == 32


# ── TenantGuard Integration ─────────────────────────────────────────────────


class TestTenantGuardIntegration:
    """Test multi-tenant isolation via decorators and context managers."""

    def test_require_tenant_allows_valid(self):
        @require_tenant
        def store(tenant_id: str, data: str):
            return f"stored for {tenant_id}"

        with TenantContext("acme", mode=TenantGuardMode.STRICT):
            result = store(tenant_id="acme", data="test")
            assert result == "stored for acme"

    def test_require_tenant_rejects_empty_strict(self):
        @require_tenant
        def store(tenant_id: str, data: str):
            return "stored"

        with TenantContext("acme", mode=TenantGuardMode.STRICT):
            with pytest.raises(TenantIsolationError):
                store(tenant_id="", data="test")

    def test_require_tenant_rejects_cross_tenant_strict(self):
        @require_tenant
        def store(tenant_id: str, data: str):
            return "stored"

        with TenantContext("acme", mode=TenantGuardMode.STRICT):
            with pytest.raises(TenantIsolationError):
                store(tenant_id="other_corp", data="test")

    def test_require_tenant_warn_mode_allows(self):
        @require_tenant
        def store(tenant_id: str, data: str):
            return "stored"

        with TenantContext("acme", mode=TenantGuardMode.WARN):
            # Should warn but not raise
            result = store(tenant_id="other_corp", data="test")
            assert result == "stored"

    def test_require_tenant_off_mode(self):
        @require_tenant
        def store(tenant_id: str, data: str):
            return "stored"

        with TenantContext("acme", mode=TenantGuardMode.OFF):
            result = store(tenant_id="", data="test")  # Would normally fail
            assert result == "stored"

    def test_tenant_context_sets_identity(self):
        with TenantContext("acme", user_id="alice", device_id="phone_1"):
            assert get_current_tenant() == "acme"
            identity = get_current_identity()
            assert identity == ("acme", "alice", "phone_1")

        # After context exits, should be cleared
        assert get_current_tenant() is None

    def test_nested_tenant_contexts(self):
        with TenantContext("outer"):
            assert get_current_tenant() == "outer"

            with TenantContext("inner"):
                assert get_current_tenant() == "inner"

            # Restored to outer
            assert get_current_tenant() == "outer"

    @pytest.mark.asyncio
    async def test_async_require_tenant(self):
        @require_tenant
        async def async_store(tenant_id: str, data: str):
            return f"async stored for {tenant_id}"

        async with TenantContext("acme", mode=TenantGuardMode.STRICT):
            result = await async_store(tenant_id="acme", data="test")
            assert result == "async stored for acme"

    @pytest.mark.asyncio
    async def test_async_cross_tenant_rejected(self):
        @require_tenant
        async def async_store(tenant_id: str, data: str):
            return "stored"

        async with TenantContext("acme", mode=TenantGuardMode.STRICT):
            with pytest.raises(TenantIsolationError):
                await async_store(tenant_id="evil_corp", data="test")

    def test_require_identity_validates_full_triplet(self):
        @require_identity
        def store(tenant_id: str, user_id: str, device_id: str, data: str):
            return "stored"

        with TenantContext("acme", user_id="alice", device_id="phone", mode=TenantGuardMode.STRICT):
            result = store(
                tenant_id="acme",
                user_id="alice",
                device_id="phone",
                data="test",
            )
            assert result == "stored"

    def test_require_identity_cross_user_rejected(self):
        @require_identity
        def store(tenant_id: str, user_id: str, device_id: str, data: str):
            return "stored"

        with TenantContext("acme", user_id="alice", device_id="phone", mode=TenantGuardMode.STRICT):
            with pytest.raises(TenantIsolationError):
                store(
                    tenant_id="acme",
                    user_id="bob",  # Wrong user
                    device_id="phone",
                    data="test",
                )

    def test_custom_param_name(self):
        @require_tenant(param="org_id")
        def store(org_id: str, data: str):
            return f"stored for {org_id}"

        with TenantContext("acme", mode=TenantGuardMode.STRICT):
            result = store(org_id="acme", data="test")
            assert result == "stored for acme"
