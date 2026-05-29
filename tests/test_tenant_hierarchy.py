"""
Tests for the hierarchical tenant registry and tenant guard clearances.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hbllm.security.tenant_guard import (
    TenantContext,
    TenantGuardMode,
    TenantIsolationError,
    _ctx_guard_mode,
    require_tenant,
)
from hbllm.security.tenant_registry import TenantRegistry


@pytest.fixture
def temp_registry() -> TenantRegistry:
    """Fixture to provide a clean temporary TenantRegistry."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "tenants.db"
        registry = TenantRegistry(db_path=db_path)
        yield registry


def test_registry_operations(temp_registry: TenantRegistry) -> None:
    """Verify registration, parent resolution, and child lookup."""
    parent_id = "org_acme"
    child_id_1 = "workspace_prod"
    child_id_2 = "workspace_staging"

    # Register parent & children
    temp_registry.register_tenant(parent_id, parent_id=None, name="Acme Org")
    temp_registry.register_tenant(child_id_1, parent_id=parent_id, name="Acme Prod Workspace")
    temp_registry.register_tenant(child_id_2, parent_id=parent_id, name="Acme Staging Workspace")

    # Verify parent resolution
    assert temp_registry.get_parent_id(child_id_1) == parent_id
    assert temp_registry.get_parent_id(child_id_2) == parent_id
    assert temp_registry.get_parent_id(parent_id) is None

    # Verify children resolution
    children = temp_registry.get_children_ids(parent_id)
    assert len(children) == 2
    assert child_id_1 in children
    assert child_id_2 in children


@require_tenant
def sample_guarded_function(tenant_id: str) -> str:
    """A dummy function guarded by the tenant validation wrapper."""
    return f"Access Granted to {tenant_id}"


def test_guard_hierarchical_clearance(temp_registry: TenantRegistry) -> None:
    """Verify that TenantGuard allows cross-tenant query access matching the hierarchy."""
    parent_id = "developer_alice"
    child_id = "code_project_xyz"
    unrelated_id = "code_project_abc"

    temp_registry.register_tenant(parent_id, parent_id=None, name="Alice Profile")
    temp_registry.register_tenant(child_id, parent_id=parent_id, name="Project XYZ")
    temp_registry.register_tenant(unrelated_id, parent_id=None, name="Project ABC")

    # Patch the global registry hook where defined to use our temp DB registry instance
    with patch("hbllm.security.tenant_registry.get_tenant_registry", return_value=temp_registry):
        # 1. Run in STRICT enforcement mode
        with TenantContext(child_id, mode=TenantGuardMode.STRICT):
            # Accessing its own tenant ID -> Allowed
            assert sample_guarded_function(tenant_id=child_id) == f"Access Granted to {child_id}"

            # Accessing its hierarchical parent -> Allowed!
            assert sample_guarded_function(tenant_id=parent_id) == f"Access Granted to {parent_id}"

            # Accessing an unrelated tenant -> Raises TenantIsolationError!
            with pytest.raises(TenantIsolationError):
                sample_guarded_function(tenant_id=unrelated_id)

        # 2. Context is parent, accessing child -> Allowed!
        with TenantContext(parent_id, mode=TenantGuardMode.STRICT):
            assert sample_guarded_function(tenant_id=child_id) == f"Access Granted to {child_id}"

            # Accessing unrelated tenant -> Raises
            with pytest.raises(TenantIsolationError):
                sample_guarded_function(tenant_id=unrelated_id)
