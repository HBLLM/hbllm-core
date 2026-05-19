"""Tests for Trust Boundaries and Permissions."""

import time

import pytest

from hbllm.brain.control.permissions import ActionClass, PermissionRegistry, TrustGrant


def test_permission_registry():
    registry = PermissionRegistry()
    registry.register_tool("file.read", ActionClass.SAFE)
    registry.register_tool("file.write", ActionClass.USER_AWARE)
    registry.register_tool("email.send", ActionClass.SENSITIVE)

    # SAFE actions
    assert registry.check_permission("file.read", "") is True

    # USER_AWARE actions
    assert registry.check_permission("file.write", "") is True

    # SENSITIVE actions
    assert registry.check_permission("email.send", "") is False
    assert registry.check_permission("unknown_tool", "") is False  # Defaults to SENSITIVE


def test_trust_grant_scoping():
    registry = PermissionRegistry()
    registry.register_tool("email.send", ActionClass.SENSITIVE)

    grant = TrustGrant(tool_name="email.send", allowed_scope="finance", max_actions=2)
    registry.issue_grant(grant)

    # Matching scope
    assert registry.check_permission("email.send", "finance/invoice_1") is True
    # Non-matching scope
    assert registry.check_permission("email.send", "personal/friend") is False

    # Consume actions
    assert registry.consume_grant("email.send", "finance/invoice_1") is True
    assert registry.consume_grant("email.send", "finance/invoice_2") is True
    # Grant exhausted
    assert registry.consume_grant("email.send", "finance/invoice_3") is False
    assert registry.check_permission("email.send", "finance/invoice_3") is False
