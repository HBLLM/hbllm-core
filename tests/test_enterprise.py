"""
Unit tests for the enterprise hardening modules:
  - Secret Provider (secrets.py)
  - RBAC Guard (rbac.py)
  - RBAC Middleware (middleware/rbac.py)
  - Audit Middleware (middleware/audit.py)
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock

import pytest

# ── Secret Provider ──────────────────────────────────────────────────


class TestEnvSecretProvider(unittest.TestCase):
    """Test the default environment variable secret backend."""

    def test_get_returns_env_value(self):
        from hbllm.security.secrets import EnvSecretProvider

        os.environ["_TEST_SECRET_KEY"] = "test_value"
        provider = EnvSecretProvider()
        assert provider.get("_TEST_SECRET_KEY") == "test_value"
        del os.environ["_TEST_SECRET_KEY"]

    def test_get_returns_default_when_missing(self):
        from hbllm.security.secrets import EnvSecretProvider

        provider = EnvSecretProvider()
        assert provider.get("_NONEXISTENT_KEY") is None
        assert provider.get("_NONEXISTENT_KEY", "fallback") == "fallback"

    def test_get_required_raises_on_missing(self):
        from hbllm.security.secrets import EnvSecretProvider

        provider = EnvSecretProvider()
        with pytest.raises(KeyError, match="Required secret"):
            provider.get_required("_DEFINITELY_NOT_SET")

    def test_get_required_returns_value(self):
        from hbllm.security.secrets import EnvSecretProvider

        os.environ["_TEST_REQ"] = "required_value"
        provider = EnvSecretProvider()
        assert provider.get_required("_TEST_REQ") == "required_value"
        del os.environ["_TEST_REQ"]


class TestSecretProviderFactory(unittest.TestCase):
    """Test the get_secret_provider factory."""

    def setUp(self):
        from hbllm.security.secrets import reset_provider

        reset_provider()

    def tearDown(self):
        from hbllm.security.secrets import reset_provider

        reset_provider()
        os.environ.pop("HBLLM_SECRET_BACKEND", None)

    def test_default_is_env(self):
        from hbllm.security.secrets import EnvSecretProvider, get_secret_provider

        provider = get_secret_provider()
        assert isinstance(provider, EnvSecretProvider)

    def test_explicit_env_backend(self):
        from hbllm.security.secrets import EnvSecretProvider, get_secret_provider

        os.environ["HBLLM_SECRET_BACKEND"] = "env"
        provider = get_secret_provider()
        assert isinstance(provider, EnvSecretProvider)

    def test_unknown_backend_raises(self):
        from hbllm.security.secrets import get_secret_provider, reset_provider

        reset_provider()
        os.environ["HBLLM_SECRET_BACKEND"] = "nonexistent"
        with pytest.raises(ValueError, match="Unknown secret backend"):
            get_secret_provider()

    def test_singleton_returns_same_instance(self):
        from hbllm.security.secrets import get_secret_provider

        p1 = get_secret_provider()
        p2 = get_secret_provider()
        assert p1 is p2


# ── RBAC Guard ───────────────────────────────────────────────────────


class TestRBACGuard(unittest.TestCase):
    """Test the RBAC permission enforcement."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from hbllm.security.rbac import RBACGuard

        self.guard = RBACGuard(db_path=f"{self.tmpdir}/rbac.db")

    def tearDown(self):
        self.guard.close()

    def test_default_role_is_viewer(self):
        from hbllm.security.rbac import Role

        role = self.guard.get_role("t1", "unknown_user")
        assert role == Role.VIEWER

    def test_assign_and_get_role(self):
        from hbllm.security.rbac import Role

        self.guard.assign_role("t1", "u1", Role.ADMIN)
        assert self.guard.get_role("t1", "u1") == Role.ADMIN

    def test_role_upgrade(self):
        from hbllm.security.rbac import Role

        self.guard.assign_role("t1", "u1", Role.MEMBER)
        self.guard.assign_role("t1", "u1", Role.OWNER)
        assert self.guard.get_role("t1", "u1") == Role.OWNER

    def test_viewer_cannot_send_chat(self):
        from hbllm.security.rbac import Permission

        assert not self.guard.check("t1", "viewer_user", Permission.CHAT_SEND)

    def test_viewer_can_read_chat(self):
        from hbllm.security.rbac import Permission

        assert self.guard.check("t1", "viewer_user", Permission.CHAT_READ)

    def test_member_can_send_chat(self):
        from hbllm.security.rbac import Permission, Role

        self.guard.assign_role("t1", "u1", Role.MEMBER)
        assert self.guard.check("t1", "u1", Permission.CHAT_SEND)

    def test_member_cannot_manage_users(self):
        from hbllm.security.rbac import Permission, Role

        self.guard.assign_role("t1", "u1", Role.MEMBER)
        assert not self.guard.check("t1", "u1", Permission.ADMIN_MANAGE_USERS)

    def test_admin_can_manage_users(self):
        from hbllm.security.rbac import Permission, Role

        self.guard.assign_role("t1", "u1", Role.ADMIN)
        assert self.guard.check("t1", "u1", Permission.ADMIN_MANAGE_USERS)

    def test_owner_has_all_permissions(self):
        from hbllm.security.rbac import Permission, Role

        self.guard.assign_role("t1", "u1", Role.OWNER)
        for perm in Permission:
            assert self.guard.check("t1", "u1", perm), f"Owner should have {perm}"

    def test_require_raises_on_insufficient(self):
        from hbllm.security.rbac import Permission

        with pytest.raises(PermissionError, match="lacks permission"):
            self.guard.require("t1", "u_viewer", Permission.ADMIN_MANAGE_USERS)

    def test_revoke_role(self):
        from hbllm.security.rbac import Role

        self.guard.assign_role("t1", "u1", Role.ADMIN)
        assert self.guard.revoke_role("t1", "u1")
        assert self.guard.get_role("t1", "u1") == Role.VIEWER

    def test_list_users(self):
        from hbllm.security.rbac import Role

        self.guard.assign_role("t1", "u1", Role.MEMBER)
        self.guard.assign_role("t1", "u2", Role.ADMIN)
        users = self.guard.list_users("t1")
        assert len(users) == 2
        user_ids = {u["user_id"] for u in users}
        assert user_ids == {"u1", "u2"}

    def test_get_permissions(self):
        from hbllm.security.rbac import Permission, Role

        self.guard.assign_role("t1", "u1", Role.MEMBER)
        perms = self.guard.get_permissions("t1", "u1")
        assert Permission.CHAT_SEND in perms
        assert Permission.ADMIN_MANAGE_USERS not in perms

    def test_tenant_isolation(self):
        from hbllm.security.rbac import Role

        self.guard.assign_role("t1", "u1", Role.OWNER)
        self.guard.assign_role("t2", "u1", Role.VIEWER)
        assert self.guard.get_role("t1", "u1") == Role.OWNER
        assert self.guard.get_role("t2", "u1") == Role.VIEWER


# ── RBAC Middleware ──────────────────────────────────────────────────


class TestRBACMiddleware(unittest.TestCase):
    """Test RBAC route matching logic."""

    def test_match_chat_send(self):
        from hbllm.serving.middleware.rbac import Permission, RBACMiddleware

        perm = RBACMiddleware._match_permission("POST", "/v1/chat/completions")
        assert perm == Permission.CHAT_SEND

    def test_match_admin_audit(self):
        from hbllm.serving.middleware.rbac import Permission, RBACMiddleware

        perm = RBACMiddleware._match_permission("GET", "/v1/admin/audit")
        assert perm == Permission.ADMIN_VIEW_AUDIT

    def test_unprotected_path_returns_none(self):
        from hbllm.serving.middleware.rbac import RBACMiddleware

        perm = RBACMiddleware._match_permission("GET", "/health/live")
        assert perm is None

    def test_unknown_path_returns_none(self):
        from hbllm.serving.middleware.rbac import RBACMiddleware

        perm = RBACMiddleware._match_permission("GET", "/some/random/path")
        assert perm is None

    def test_memory_write(self):
        from hbllm.serving.middleware.rbac import Permission, RBACMiddleware

        perm = RBACMiddleware._match_permission("POST", "/v1/memory")
        assert perm == Permission.MEMORY_WRITE


# ── Audit Middleware ─────────────────────────────────────────────────


class TestAuditMiddleware(unittest.TestCase):
    """Test audit action classification."""

    def test_classify_chat(self):
        from hbllm.security.audit_log import AuditAction
        from hbllm.serving.middleware.audit import AuditMiddleware

        action = AuditMiddleware._classify_action("POST", "/v1/chat/completions")
        assert action == AuditAction.CHAT_MESSAGE

    def test_classify_admin(self):
        from hbllm.security.audit_log import AuditAction
        from hbllm.serving.middleware.audit import AuditMiddleware

        action = AuditMiddleware._classify_action("POST", "/v1/admin/config")
        assert action == AuditAction.ADMIN_ACTION

    def test_classify_unknown_defaults(self):
        from hbllm.security.audit_log import AuditAction
        from hbllm.serving.middleware.audit import AuditMiddleware

        action = AuditMiddleware._classify_action("GET", "/unknown")
        assert action == AuditAction.DATA_ACCESSED

    def test_classify_data_export(self):
        from hbllm.security.audit_log import AuditAction
        from hbllm.serving.middleware.audit import AuditMiddleware

        action = AuditMiddleware._classify_action("POST", "/v1/data/export")
        assert action == AuditAction.DATA_EXPORTED

    def test_classify_tool_execution(self):
        from hbllm.security.audit_log import AuditAction
        from hbllm.serving.middleware.audit import AuditMiddleware

        action = AuditMiddleware._classify_action("POST", "/v1/tools/search")
        assert action == AuditAction.TOOL_EXECUTED


# ── Security Headers Middleware ──────────────────────────────────────


class TestSecurityHeadersDefaults(unittest.TestCase):
    """Test default security header values."""

    def test_default_headers_include_hsts(self):
        from hbllm.serving.middleware.security_headers import DEFAULT_SECURITY_HEADERS

        assert "Strict-Transport-Security" in DEFAULT_SECURITY_HEADERS

    def test_default_headers_include_csp(self):
        from hbllm.serving.middleware.security_headers import DEFAULT_SECURITY_HEADERS

        assert "Content-Security-Policy" in DEFAULT_SECURITY_HEADERS

    def test_default_headers_include_x_frame(self):
        from hbllm.serving.middleware.security_headers import DEFAULT_SECURITY_HEADERS

        assert DEFAULT_SECURITY_HEADERS["X-Frame-Options"] == "DENY"

    def test_default_headers_include_nosniff(self):
        from hbllm.serving.middleware.security_headers import DEFAULT_SECURITY_HEADERS

        assert DEFAULT_SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"

    def test_hsts_can_be_disabled(self):
        from unittest.mock import MagicMock

        from hbllm.serving.middleware.security_headers import SecurityHeadersMiddleware

        mw = SecurityHeadersMiddleware(MagicMock(), hsts=False)
        assert "Strict-Transport-Security" not in mw._headers

    def test_custom_headers_merged(self):
        from unittest.mock import MagicMock

        from hbllm.serving.middleware.security_headers import SecurityHeadersMiddleware

        mw = SecurityHeadersMiddleware(MagicMock(), custom_headers={"X-Custom": "value"})
        assert mw._headers["X-Custom"] == "value"
        # Default headers still present
        assert "X-Frame-Options" in mw._headers
