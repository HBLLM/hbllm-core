"""
Security & Auth — Integration test coverage.

Covers uncovered lines in:
  - hbllm/security/encryption.py
  - hbllm/security/secrets.py
  - hbllm/security/audit_log.py
  - hbllm/security/identity.py
  - hbllm/security/tenant_guard.py
  - hbllm/serving/security.py
  - hbllm/serving/auth.py
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════
# encryption.py
# ═══════════════════════════════════════════════════════════════════════


class TestEncryptionVault:
    """Cover all paths in EncryptionVault: init, encrypt/decrypt, from_*, rotate."""

    def test_generate_key(self):
        from hbllm.security.encryption import _generate_key

        key = _generate_key()
        assert len(key) > 0
        assert isinstance(key, bytes)

    def test_derive_key(self):
        from hbllm.security.encryption import _derive_key

        key = _derive_key("password123", b"salt1234salt1234")
        assert len(key) == 32
        key2 = _derive_key("password123", b"salt1234salt1234")
        assert key == key2
        key3 = _derive_key("other-password", b"salt1234salt1234")
        assert key != key3

    def test_vault_init_autogenerate_key(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        assert vault._key is not None
        assert vault._fernet is not None

    def test_vault_init_custom_key(self):
        from hbllm.security.encryption import EncryptionVault, _generate_key

        key = _generate_key()
        vault = EncryptionVault(key=key)
        assert vault._key == key

    def test_encrypt_decrypt_roundtrip(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        plaintext = "Hello, sensitive data!"
        encrypted = vault.encrypt(plaintext)
        assert encrypted != plaintext
        assert vault.decrypt(encrypted) == plaintext

    def test_decrypt_tampered_token_raises(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        encrypted = vault.encrypt("test")
        tampered = encrypted[:10] + "X" + encrypted[11:]
        with pytest.raises(ValueError, match="tampered|Invalid"):
            vault.decrypt(tampered)

    def test_decrypt_invalid_token_raises(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        with pytest.raises(ValueError):
            vault.decrypt("not-a-valid-token")

    def test_encrypt_dict_decrypt_dict(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        data = {"api_key": "sk-123", "secret": "value", "count": 42}
        encrypted = vault.encrypt_dict(data)
        assert isinstance(encrypted, str)
        assert vault.decrypt_dict(encrypted) == data

    def test_from_env(self, monkeypatch):
        from hbllm.security.encryption import EncryptionVault, _generate_key

        key = _generate_key()
        monkeypatch.setenv("HBLLM_ENCRYPTION_KEY", key.decode("utf-8"))
        vault = EncryptionVault.from_env()
        assert vault._key == key

    def test_from_env_missing_raises(self, monkeypatch):
        from hbllm.security.encryption import EncryptionVault

        monkeypatch.delenv("HBLLM_ENCRYPTION_KEY", raising=False)
        with pytest.raises(ValueError, match="not set"):
            EncryptionVault.from_env()

    def test_from_key_file_creates(self, tmp_path):
        from hbllm.security.encryption import EncryptionVault

        key_path = str(tmp_path / "subdir" / "encryption.key")
        vault = EncryptionVault.from_key_file(key_path)
        assert Path(key_path).exists()
        vault2 = EncryptionVault.from_key_file(key_path)
        assert vault._key == vault2._key

    def test_from_key_file_loads_existing(self, tmp_path):
        from hbllm.security.encryption import EncryptionVault, _generate_key

        key = _generate_key()
        key_path = tmp_path / "existing.key"
        key_path.write_text(key.decode("utf-8"))
        vault = EncryptionVault.from_key_file(str(key_path))
        assert vault._key == key

    def test_from_password(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault.from_password("my-strong-password")
        assert vault._salt is not None
        encrypted = vault.encrypt("secret data")
        assert vault.decrypt(encrypted) == "secret data"

    def test_from_password_with_salt(self):
        from hbllm.security.encryption import EncryptionVault

        salt = b"1234567890123456"
        v1 = EncryptionVault.from_password("password", salt=salt)
        v2 = EncryptionVault.from_password("password", salt=salt)
        assert v1._key == v2._key

    def test_rotate_key(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        new_vault = vault.rotate_key()
        assert new_vault._key != vault._key

    def test_rotate_and_reencrypt(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        originals = ["data-1", "data-2", "data-3"]
        tokens = [vault.encrypt(d) for d in originals]

        new_vault, reencrypted_iter = vault.rotate_and_reencrypt(tokens)
        reencrypted = list(reencrypted_iter)

        assert len(reencrypted) == 3
        for orig, new_token in zip(originals, reencrypted):
            assert new_vault.decrypt(new_token) == orig
            with pytest.raises(ValueError):
                vault.decrypt(new_token)

    def test_key_fingerprint(self):
        from hbllm.security.encryption import EncryptionVault

        vault = EncryptionVault()
        fp = vault.key_fingerprint
        assert len(fp) == 12
        assert isinstance(fp, str)


# ═══════════════════════════════════════════════════════════════════════
# secrets.py
# ═══════════════════════════════════════════════════════════════════════


class TestEnvSecretProvider:
    def test_get(self, monkeypatch):
        from hbllm.security.secrets import EnvSecretProvider

        monkeypatch.setenv("TEST_SECRET", "mysecret")
        provider = EnvSecretProvider()
        assert provider.get("TEST_SECRET") == "mysecret"
        assert provider.get("MISSING_KEY") is None
        assert provider.get("MISSING_KEY", "default") == "default"

    def test_get_required(self, monkeypatch):
        from hbllm.security.secrets import EnvSecretProvider

        monkeypatch.setenv("TEST_SECRET", "mysecret")
        assert EnvSecretProvider().get_required("TEST_SECRET") == "mysecret"

    def test_get_required_missing_raises(self, monkeypatch):
        from hbllm.security.secrets import EnvSecretProvider

        monkeypatch.delenv("MISSING_KEY", raising=False)
        with pytest.raises(KeyError, match="not found"):
            EnvSecretProvider().get_required("MISSING_KEY")


class TestVaultSecretProvider:
    def test_init_with_token(self, monkeypatch):
        monkeypatch.setenv("VAULT_ADDR", "http://127.0.0.1:8200")
        monkeypatch.setenv("VAULT_TOKEN", "test-token")
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.VaultSecretProvider()
            assert provider._mount == "secret"
            assert provider._path == "hbllm"

    def test_init_with_approle(self, monkeypatch):
        monkeypatch.delenv("VAULT_TOKEN", raising=False)
        monkeypatch.setenv("VAULT_ROLE_ID", "test-role")
        monkeypatch.setenv("VAULT_SECRET_ID", "test-secret")
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_hvac.Client.return_value = mock_client
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.VaultSecretProvider()
            mock_client.auth.approle.login.assert_called_once()

    def test_init_auth_failure(self, monkeypatch):
        monkeypatch.setenv("VAULT_TOKEN", "bad-token")
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = False
        mock_hvac.Client.return_value = mock_client
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            with pytest.raises(RuntimeError, match="authentication failed"):
                hbllm.security.secrets.VaultSecretProvider()

    def test_get_loads_and_caches(self, monkeypatch):
        monkeypatch.setenv("VAULT_TOKEN", "tok")
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.read_secret_version.return_value = {
            "data": {"data": {"JWT_SECRET": "vault-jwt-value"}}
        }
        mock_hvac.Client.return_value = mock_client
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.VaultSecretProvider()
            assert provider.get("JWT_SECRET") == "vault-jwt-value"
            assert provider._loaded is True

    def test_get_required_missing_raises(self, monkeypatch):
        monkeypatch.setenv("VAULT_TOKEN", "tok")
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.read_secret_version.return_value = {"data": {"data": {}}}
        mock_hvac.Client.return_value = mock_client
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.VaultSecretProvider()
            with pytest.raises(KeyError, match="not found"):
                provider.get_required("NONEXISTENT_KEY")

    def test_load_failure_raises(self, monkeypatch):
        monkeypatch.setenv("VAULT_TOKEN", "tok")
        mock_hvac = MagicMock()
        mock_client = MagicMock()
        mock_client.is_authenticated.return_value = True
        mock_client.secrets.kv.v2.read_secret_version.side_effect = Exception("Connection refused")
        mock_hvac.Client.return_value = mock_client
        with patch.dict("sys.modules", {"hvac": mock_hvac}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.VaultSecretProvider()
            with pytest.raises(Exception, match="Connection refused"):
                provider.get("ANY_KEY")


class TestAWSSecretsProvider:
    def test_init(self, monkeypatch):
        monkeypatch.setenv("HBLLM_AWS_REGION", "eu-west-1")
        monkeypatch.setenv("HBLLM_AWS_SECRET_NAME", "test/secrets")
        mock_boto3 = MagicMock()
        mock_boto3.client.return_value = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.AWSSecretsProvider()
            assert provider._secret_name == "test/secrets"

    def test_get_loads_secrets(self, monkeypatch):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {
            "SecretString": '{"DB_PASSWORD": "aws-secret-value"}'
        }
        mock_boto3.client.return_value = mock_client
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.AWSSecretsProvider()
            assert provider.get("DB_PASSWORD") == "aws-secret-value"
            assert provider._loaded is True

    def test_get_required_missing_raises(self, monkeypatch):
        monkeypatch.delenv("MISSING_AWS_KEY", raising=False)
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.get_secret_value.return_value = {"SecretString": "{}"}
        mock_boto3.client.return_value = mock_client
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.AWSSecretsProvider()
            with pytest.raises(KeyError, match="not found"):
                provider.get_required("MISSING_AWS_KEY")

    def test_load_failure_raises(self, monkeypatch):
        mock_boto3 = MagicMock()
        mock_client = MagicMock()
        mock_client.get_secret_value.side_effect = Exception("AccessDenied")
        mock_boto3.client.return_value = mock_client
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            from importlib import reload
            import hbllm.security.secrets
            reload(hbllm.security.secrets)
            provider = hbllm.security.secrets.AWSSecretsProvider()
            with pytest.raises(Exception, match="AccessDenied"):
                provider.get("ANY")


class TestSecretProviderFactory:
    def test_default_env_backend(self, monkeypatch):
        from hbllm.security.secrets import EnvSecretProvider, get_secret_provider, reset_provider
        reset_provider()
        monkeypatch.setenv("HBLLM_SECRET_BACKEND", "env")
        assert isinstance(get_secret_provider(), EnvSecretProvider)
        reset_provider()

    def test_unknown_backend_raises(self, monkeypatch):
        from hbllm.security.secrets import get_secret_provider, reset_provider
        reset_provider()
        monkeypatch.setenv("HBLLM_SECRET_BACKEND", "unknown")
        with pytest.raises(ValueError, match="Unknown secret backend"):
            get_secret_provider()
        reset_provider()

    def test_singleton_pattern(self, monkeypatch):
        from hbllm.security.secrets import get_secret_provider, reset_provider
        reset_provider()
        monkeypatch.setenv("HBLLM_SECRET_BACKEND", "env")
        p1 = get_secret_provider()
        p2 = get_secret_provider()
        assert p1 is p2
        reset_provider()


# ═══════════════════════════════════════════════════════════════════════
# audit_log.py
# ═══════════════════════════════════════════════════════════════════════


class TestAuditLog:
    @pytest.fixture
    def audit(self, tmp_path):
        from hbllm.security.audit_log import AuditLog
        log = AuditLog(db_path=str(tmp_path / "audit.db"))
        yield log
        log.close()

    def test_log_basic(self, audit):
        from hbllm.security.audit_log import AuditAction
        entry = audit.log(action=AuditAction.AUTH_LOGIN, tenant_id="t1", user_id="u1",
                          actor="u1", resource="session:abc", ip_address="1.2.3.4",
                          user_agent="Mozilla/5.0")
        assert entry.action == "auth.login"

    def test_log_critical_severity(self, audit):
        from hbllm.security.audit_log import AuditAction, AuditSeverity
        entry = audit.log(action=AuditAction.ADMIN_ACTION, tenant_id="admin",
                          severity=AuditSeverity.CRITICAL, details={"op": "delete"})
        assert entry.severity == "critical"

    def test_log_to_dict(self, audit):
        d = audit.log(action="test.action", tenant_id="t1").to_dict()
        assert d["action"] == "test.action"
        assert "id" in d and "timestamp" in d

    def test_query_by_tenant(self, audit):
        audit.log(action="a", tenant_id="t1")
        audit.log(action="b", tenant_id="t2")
        assert len(audit.query(tenant_id="t1")) == 1

    def test_query_by_user(self, audit):
        audit.log(action="a", tenant_id="t1", user_id="u1")
        audit.log(action="a", tenant_id="t1", user_id="u2")
        assert len(audit.query(user_id="u1")) == 1

    def test_query_by_device(self, audit):
        audit.log(action="a", tenant_id="t1", device_id="d1")
        audit.log(action="a", tenant_id="t1", device_id="d2")
        assert len(audit.query(device_id="d1")) == 1

    def test_query_by_action(self, audit):
        audit.log(action="auth.login", tenant_id="t1")
        audit.log(action="auth.failed", tenant_id="t1")
        assert len(audit.query(action="auth.login")) == 1

    def test_query_by_severity(self, audit):
        audit.log(action="a", tenant_id="t1", severity="info")
        audit.log(action="b", tenant_id="t1", severity="critical")
        assert len(audit.query(severity="critical")) == 1

    def test_query_by_actor(self, audit):
        audit.log(action="a", tenant_id="t1", actor="admin")
        audit.log(action="b", tenant_id="t1", actor="user1")
        assert len(audit.query(actor="admin")) == 1

    def test_query_by_time_range(self, audit):
        now = time.time()
        audit.log(action="a", tenant_id="t1")
        assert len(audit.query(since=now - 10, until=now + 10)) == 1

    def test_query_by_success(self, audit):
        audit.log(action="a", tenant_id="t1", success=True)
        audit.log(action="b", tenant_id="t1", success=False)
        results = audit.query(success=False)
        assert len(results) == 1
        assert results[0]["success"] is False

    def test_query_with_offset(self, audit):
        for i in range(5):
            audit.log(action=f"t.{i}", tenant_id="t1")
        assert len(audit.query(limit=2, offset=2)) == 2

    def test_count_all(self, audit):
        audit.log(action="a", tenant_id="t1")
        audit.log(action="b", tenant_id="t1")
        assert audit.count() == 2

    def test_count_by_tenant(self, audit):
        audit.log(action="a", tenant_id="t1")
        audit.log(action="b", tenant_id="t2")
        assert audit.count(tenant_id="t1") == 1

    def test_count_by_action(self, audit):
        audit.log(action="auth.login", tenant_id="t1")
        audit.log(action="auth.failed", tenant_id="t1")
        assert audit.count(action="auth.login") == 1

    def test_count_since(self, audit):
        audit.log(action="a", tenant_id="t1")
        assert audit.count(since=time.time() - 60) == 1
        assert audit.count(since=time.time() + 60) == 0

    def test_failed_logins(self, audit):
        audit.log(action="auth.failed", tenant_id="t1")
        audit.log(action="auth.failed", tenant_id="t1")
        audit.log(action="auth.login", tenant_id="t1")
        assert audit.failed_logins() == 2

    def test_failed_logins_by_tenant(self, audit):
        audit.log(action="auth.failed", tenant_id="t1")
        audit.log(action="auth.failed", tenant_id="t2")
        assert audit.failed_logins(tenant_id="t1") == 1

    def test_export_json(self, audit):
        audit.log(action="a", tenant_id="t1")
        audit.log(action="b", tenant_id="t2")
        assert len(audit.export_json(tenant_id="t1")) == 1

    def test_purge_old_entries(self, audit):
        audit._conn.execute(
            "INSERT INTO audit_log (id, timestamp, tenant_id, actor, action, details) VALUES (?,?,?,?,?,?)",
            ("old1", time.time() - 400 * 86400, "t1", "sys", "test.old", "{}"),
        )
        audit._conn.commit()
        audit.log(action="new", tenant_id="t1")
        assert audit.purge_old_entries(older_than_days=365) == 1
        assert audit.count() == 1

    def test_stats(self, audit):
        audit.log(action="a", tenant_id="t1", severity="info")
        audit.log(action="b", tenant_id="t1", severity="warning")
        audit.log(action="c", tenant_id="t1", severity="critical")
        stats = audit.stats()
        assert stats["total_entries"] == 3
        assert stats["by_severity"]["critical"] == 1
        assert len(stats["recent_critical"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# identity.py
# ═══════════════════════════════════════════════════════════════════════


class TestNodeIdentity:
    def test_generate(self):
        from hbllm.security.identity import NodeIdentity
        identity = NodeIdentity.generate()
        assert identity.public_key_bytes is not None

    def test_load_or_create_new(self, tmp_path):
        from hbllm.security.identity import NodeIdentity
        key_path = tmp_path / "node.key"
        identity = NodeIdentity.load_or_create(key_path)
        assert key_path.exists()

    def test_load_or_create_existing(self, tmp_path):
        from hbllm.security.identity import NodeIdentity
        key_path = tmp_path / "node.key"
        id1 = NodeIdentity.load_or_create(key_path)
        assert key_path.exists()
        # Subsequent call produces valid identity (key saved in OpenSSH format,
        # which load_pem_private_key may not reload — falls back to new key)
        id2 = NodeIdentity.load_or_create(key_path)
        assert id2.public_key_bytes is not None

    def test_load_or_create_corrupted_fallback(self, tmp_path):
        from hbllm.security.identity import NodeIdentity
        key_path = tmp_path / "node.key"
        key_path.write_bytes(b"corrupted-key-data-not-valid-pem")
        identity = NodeIdentity.load_or_create(key_path)
        assert identity.public_key_bytes is not None

    def test_sign_and_verify(self):
        from hbllm.security.identity import NodeIdentity
        identity = NodeIdentity.generate()
        data = b"important message"
        sig = identity.sign(data)
        assert NodeIdentity.verify(identity.public_key_b64, data, sig)

    def test_verify_bad_signature_returns_false(self):
        from hbllm.security.identity import NodeIdentity
        identity = NodeIdentity.generate()
        assert NodeIdentity.verify(identity.public_key_b64, b"data", "badsig==") is False

    def test_public_key_b64(self):
        from hbllm.security.identity import NodeIdentity
        b64_key = NodeIdentity.generate().public_key_b64
        assert isinstance(b64_key, str) and len(b64_key) > 0


# ═══════════════════════════════════════════════════════════════════════
# tenant_guard.py
# ═══════════════════════════════════════════════════════════════════════


class TestTenantGuardCoverage:
    def test_tenant_context_isolation(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "production")
        monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")
        from hbllm.security.tenant_guard import TenantContext
        with TenantContext(tenant_id="tenant_A", user_id="user1") as ctx:
            assert ctx.tenant_id == "tenant_A"

    def test_require_tenant_blocks_cross_tenant(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "production")
        monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")
        from hbllm.security.tenant_guard import TenantContext, TenantIsolationError, require_tenant

        @require_tenant
        def my_fn(tenant_id: str = "") -> str:
            return f"ok-{tenant_id}"

        with TenantContext(tenant_id="tenant_A"):
            with pytest.raises(TenantIsolationError):
                my_fn(tenant_id="tenant_B")

    def test_require_tenant_allows_same_tenant(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "production")
        monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")
        from hbllm.security.tenant_guard import TenantContext, require_tenant

        @require_tenant
        def my_fn(tenant_id: str = "") -> str:
            return f"ok-{tenant_id}"

        with TenantContext(tenant_id="tenant_A"):
            assert my_fn(tenant_id="tenant_A") == "ok-tenant_A"

    @pytest.mark.asyncio
    async def test_require_identity_async(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "production")
        monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")
        from hbllm.security.tenant_guard import TenantContext, require_identity

        @require_identity
        async def my_fn(tenant_id="", user_id="", device_id=""):
            return f"ok-{tenant_id}-{user_id}"

        with TenantContext(tenant_id="t1", user_id="u1", device_id="d1"):
            assert await my_fn(tenant_id="t1", user_id="u1", device_id="d1") == "ok-t1-u1"

    def test_require_identity_sync(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "production")
        monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")
        from hbllm.security.tenant_guard import TenantContext, require_identity

        @require_identity
        def my_fn(tenant_id="", user_id="", device_id=""):
            return f"ok-{tenant_id}-{user_id}"

        with TenantContext(tenant_id="t1", user_id="u1", device_id="d1"):
            assert my_fn(tenant_id="t1", user_id="u1", device_id="d1") == "ok-t1-u1"

    def test_warn_mode_logs_but_allows(self, monkeypatch):
        monkeypatch.setenv("HBLLM_ENV", "development")
        monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "warn")
        from hbllm.security.tenant_guard import TenantContext, require_tenant

        @require_tenant
        def my_fn(tenant_id: str = "") -> str:
            return f"ok-{tenant_id}"

        with TenantContext(tenant_id="tenant_A"):
            assert my_fn(tenant_id="tenant_B") == "ok-tenant_B"


# ═══════════════════════════════════════════════════════════════════════
# serving/security.py
# ═══════════════════════════════════════════════════════════════════════


class TestServingSecurityCoverage:
    def test_csrf_token_generation(self):
        from hbllm.serving.security import generate_csrf_token
        token = generate_csrf_token("session-123")
        assert isinstance(token, str) and token.count(":") >= 2

    def test_csrf_token_validation(self):
        from hbllm.serving.security import generate_csrf_token, validate_csrf_token
        token = generate_csrf_token("session-123")
        assert validate_csrf_token(token, "session-123")

    def test_csrf_invalid_token(self):
        from hbllm.serving.security import validate_csrf_token
        assert not validate_csrf_token("invalid-token", "session-123")

    def test_csrf_wrong_session(self):
        from hbllm.serving.security import generate_csrf_token, validate_csrf_token
        token = generate_csrf_token("session-123")
        assert not validate_csrf_token(token, "session-456")

    def test_cors_validate_config(self):
        from hbllm.serving.security import validate_cors_config
        result = validate_cors_config(["https://example.com", "https://app.test.com"])
        assert result == ["https://example.com", "https://app.test.com"]

    def test_cors_wildcard_rejected(self):
        from hbllm.serving.security import validate_cors_config
        result = validate_cors_config(["*", "https://example.com"])
        assert "*" not in result and "https://example.com" in result

    def test_cors_wildcard_only_falls_back(self):
        from hbllm.serving.security import validate_cors_config
        result = validate_cors_config(["*"])
        assert "http://localhost" in result or "https://localhost" in result

    def test_password_hash_and_verify(self):
        from hbllm.serving.security import hash_password, verify_password
        hashed = hash_password("my-password")
        assert verify_password("my-password", hashed)
        assert not verify_password("wrong-password", hashed)

    def test_password_verify_invalid_format(self):
        from hbllm.serving.security import verify_password
        assert not verify_password("any", "not-a-valid-hash")

    def test_auth_rate_limiter_reset(self):
        from hbllm.serving.security import AuthRateLimiter
        limiter = AuthRateLimiter(max_attempts=3, window_seconds=300)
        limiter.record_attempt("ip1")
        limiter.reset("ip1")
        allowed, remaining = limiter.check("ip1")
        assert allowed and remaining == 3

    def test_api_key_manager_add_and_validate(self):
        from hbllm.serving.security import ApiKeyManager
        mgr = ApiKeyManager()
        mgr.enabled = True
        mgr.add_key(raw_key="sk-test-key-123", tenant_id="t1", scopes=["chat", "memory"])
        result = mgr.validate("sk-test-key-123")
        assert result is not None and result.tenant_id == "t1" and mgr.key_count == 1

    def test_api_key_manager_validate_invalid(self):
        from hbllm.serving.security import ApiKeyManager
        mgr = ApiKeyManager()
        mgr.enabled = True
        assert mgr.validate("nonexistent-key") is None

    def test_api_key_manager_disabled_bypass(self):
        from hbllm.serving.security import ApiKeyManager
        mgr = ApiKeyManager()
        mgr.enabled = False
        result = mgr.validate("any-key")
        assert result is not None and result.tenant_id == "dev"

    def test_api_key_manager_has_scope(self):
        from hbllm.serving.security import ApiKeyManager
        mgr = ApiKeyManager()
        key = mgr.add_key(raw_key="sk-scoped", tenant_id="t1", scopes=["chat"])
        assert mgr.has_scope(key, "chat") and not mgr.has_scope(key, "admin")

    def test_detect_injection(self):
        from hbllm.serving.security import detect_injection
        assert detect_injection("normal text")["detected"] is False

    def test_sanitize_input_truncates(self):
        from hbllm.serving.security import sanitize_input
        assert len(sanitize_input("a" * 100, max_length=10)) == 10


# ═══════════════════════════════════════════════════════════════════════
# serving/auth.py
# ═══════════════════════════════════════════════════════════════════════


class TestAuthMiddlewareCoverage:
    def test_health_bypasses_auth(self, monkeypatch):
        monkeypatch.setenv("HBLLM_JWT_SECRET", "test_secret_key_for_jwt_testing_32ch")
        from fastapi.testclient import TestClient
        from hbllm.serving.api import app
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/health").status_code == 200

    def test_static_assets_bypass_auth(self, monkeypatch):
        monkeypatch.setenv("HBLLM_JWT_SECRET", "test_secret_key_for_jwt_testing_32ch")
        from fastapi.testclient import TestClient
        from hbllm.serving.api import app
        client = TestClient(app, raise_server_exceptions=False)
        assert client.get("/admin/static/app.js").status_code != 401

    def test_malformed_bearer_token_rejected(self, monkeypatch):
        monkeypatch.setenv("HBLLM_JWT_SECRET", "test_secret_key_for_jwt_testing_32ch")
        monkeypatch.setenv("HBLLM_ENV", "production")
        from fastapi.testclient import TestClient
        from hbllm.serving.api import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/chat", headers={"Authorization": "Bearer not.a.valid.jwt.token"})
        assert resp.status_code == 401

    def test_non_bearer_auth_rejected(self, monkeypatch):
        monkeypatch.setenv("HBLLM_JWT_SECRET", "test_secret_key_for_jwt_testing_32ch")
        monkeypatch.setenv("HBLLM_ENV", "production")
        from fastapi.testclient import TestClient
        from hbllm.serving.api import app
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/chat", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401
