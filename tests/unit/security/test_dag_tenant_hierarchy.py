"""
Tests for the Multi-Parent Directed Acyclic Graph (DAG) Tenancy hierarchy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from hbllm.security.tenant_guard import (
    TenantContext,
    TenantGuardMode,
    TenantIsolationError,
    require_tenant,
)
from hbllm.security.tenant_registry import TenantRegistry


@pytest.fixture
def temp_registry() -> TenantRegistry:
    """Fixture to provide a clean temporary TenantRegistry with DAG support."""
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "tenants.db"
        registry = TenantRegistry(db_path=db_path)
        yield registry


@require_tenant
def guarded_function(tenant_id: str) -> str:
    """A dummy function guarded by the tenant validation wrapper."""
    return f"Access Granted to {tenant_id}"


def test_dag_hierarchy_operations(temp_registry: TenantRegistry) -> None:
    """Verify registration, multi-parent retrieval, and DFS/BFS ancestor walks."""
    # Define a DAG tenancy structure:
    #   home_network_A  office_network_B
    #         \              /
    #        developer_laptop_C
    #                |
    #            project_D
    #                |
    #             task_E

    parent_home = "home_network_A"
    parent_office = "office_network_B"
    child_laptop = "developer_laptop_C"
    child_project = "project_D"
    child_task = "task_E"

    # Register nodes
    temp_registry.register_tenant(parent_home, parent_id=None, name="Home Server")
    temp_registry.register_tenant(parent_office, parent_id=None, name="Office SaaS Platform")

    # laptop_C has two parents: home_A and office_B
    temp_registry.register_tenant(child_laptop, parent_id=parent_home, name="Dev Laptop")
    temp_registry.register_parent_relationship(child_laptop, parent_office)

    temp_registry.register_tenant(child_project, parent_id=child_laptop, name="HBLLM Core Project")
    temp_registry.register_tenant(child_task, parent_id=child_project, name="Refactor CLI Task")

    # Assert get_parents resolves both parents for laptop_C
    parents = temp_registry.get_parents(child_laptop)
    assert len(parents) == 2
    assert parent_home in parents
    assert parent_office in parents

    with patch("hbllm.security.tenant_registry.get_tenant_registry", return_value=temp_registry):
        # 1. Accessing ancestors from a deep child context -> Allowed!
        with TenantContext(child_task, mode=TenantGuardMode.STRICT):
            # Self
            assert guarded_function(child_task) == f"Access Granted to {child_task}"
            # Deep parent path 1 (Home)
            assert guarded_function(parent_home) == f"Access Granted to {parent_home}"
            # Deep parent path 2 (Office)
            assert guarded_function(parent_office) == f"Access Granted to {parent_office}"
            # Intermediate ancestor
            assert guarded_function(child_laptop) == f"Access Granted to {child_laptop}"

        # 2. Accessing descendants from an ancestor context -> Allowed!
        with TenantContext(parent_home, mode=TenantGuardMode.STRICT):
            # Descendant laptop
            assert guarded_function(child_laptop) == f"Access Granted to {child_laptop}"
            # Deep descendant task
            assert guarded_function(child_task) == f"Access Granted to {child_task}"

        with TenantContext(parent_office, mode=TenantGuardMode.STRICT):
            # Deep descendant project
            assert guarded_function(child_project) == f"Access Granted to {child_project}"

        # 3. Sibling-level blocks (Parents cannot access each other's sibling domains)
        with TenantContext(parent_home, mode=TenantGuardMode.STRICT):
            with pytest.raises(TenantIsolationError):
                guarded_function(parent_office)

        with TenantContext(parent_office, mode=TenantGuardMode.STRICT):
            with pytest.raises(TenantIsolationError):
                guarded_function(parent_home)

        # 4. Unrelated tenant is strictly blocked
        temp_registry.register_tenant("unrelated_X", parent_id=None, name="External Entity")
        with TenantContext(child_task, mode=TenantGuardMode.STRICT):
            with pytest.raises(TenantIsolationError):
                guarded_function("unrelated_X")
