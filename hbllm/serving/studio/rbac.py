"""
Studio RBAC / Permissions Endpoints.

Exposes role-based access control management: status, user listing,
role assignment/revocation, permission checks, and audit log queries.

Extracted from ``_legacy.py`` — see Work Stream 1.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hbllm.serving.studio.helpers import get_data_dir, get_tenant_id, get_user_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/studio/rbac/status")
async def studio_rbac_status(request: Request) -> Any:
    """RBAC system status and current user's role/permissions."""
    tenant_id = get_tenant_id(request)
    user_id = get_user_id(request)

    data_dir = get_data_dir()
    db_path = os.path.join(data_dir, "rbac.db")

    try:
        from hbllm.security.rbac import ROLE_PERMISSIONS, Permission, RBACGuard, Role

        guard = RBACGuard(db_path=db_path)
        role = guard.get_role(tenant_id, user_id)
        permissions = guard.get_permissions(tenant_id, user_id)

        return {
            "status": "active",
            "current_user": {
                "user_id": user_id,
                "tenant_id": tenant_id,
                "role": role.value,
                "permissions": sorted([p.value for p in permissions]),
            },
            "available_roles": [r.value for r in Role],
            "available_permissions": [p.value for p in Permission],
            "role_matrix": {
                r.value: sorted([p.value for p in perms]) for r, perms in ROLE_PERMISSIONS.items()
            },
        }
    except ImportError:
        return {"status": "not_available", "error": "RBAC module not installed"}
    except Exception as e:
        logger.error("RBAC status failed: %s", e)
        return {"status": "error", "error": str(e)}


@router.get("/studio/rbac/users")
async def studio_rbac_list_users(request: Request) -> Any:
    """List all users with role assignments in the current tenant."""
    tenant_id = get_tenant_id(request)
    data_dir = get_data_dir()
    db_path = os.path.join(data_dir, "rbac.db")

    try:
        from hbllm.security.rbac import ROLE_PERMISSIONS, RBACGuard, Role

        guard = RBACGuard(db_path=db_path)
        users = guard.list_users(tenant_id)

        # Enrich with permission count
        for user in users:
            try:
                role = Role(user["role"])
                user["permission_count"] = len(ROLE_PERMISSIONS.get(role, set()))
            except ValueError:
                user["permission_count"] = 0

        return {"tenant_id": tenant_id, "users": users, "count": len(users)}
    except Exception as e:
        logger.error("RBAC list users failed: %s", e)
        return {"tenant_id": tenant_id, "users": [], "count": 0, "error": str(e)}


@router.post("/studio/rbac/assign")
async def studio_rbac_assign_role(request: Request) -> Any:
    """Assign a role to a user within the current tenant.

    Body:
        {
            "user_id": "user_42",
            "role": "member"
        }
    """
    body = await request.json()
    tenant_id = get_tenant_id(request)
    acting_user = get_user_id(request)
    target_user = body.get("user_id")
    role_name = body.get("role")

    if not target_user or not role_name:
        raise HTTPException(status_code=400, detail="user_id and role are required")

    data_dir = get_data_dir()
    db_path = os.path.join(data_dir, "rbac.db")

    try:
        from hbllm.security.rbac import Permission, RBACGuard, Role

        guard = RBACGuard(db_path=db_path)

        # Check that acting user has permission to manage users
        if not guard.check(tenant_id, acting_user, Permission.ADMIN_MANAGE_USERS):
            raise HTTPException(
                status_code=403,
                detail=f"User '{acting_user}' lacks admin:manage_users permission",
            )

        role = Role(role_name)
        guard.assign_role(tenant_id, target_user, role, assigned_by=acting_user)

        return {
            "status": "assigned",
            "tenant_id": tenant_id,
            "user_id": target_user,
            "role": role.value,
            "assigned_by": acting_user,
        }
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {role_name}. Valid: owner, admin, member, viewer, api_key",
        )
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/studio/rbac/revoke")
async def studio_rbac_revoke_role(request: Request) -> Any:
    """Revoke a user's role assignment (resets to viewer).

    Body:
        {"user_id": "user_42"}
    """
    body = await request.json()
    tenant_id = get_tenant_id(request)
    acting_user = get_user_id(request)
    target_user = body.get("user_id")

    if not target_user:
        raise HTTPException(status_code=400, detail="user_id is required")

    data_dir = get_data_dir()
    db_path = os.path.join(data_dir, "rbac.db")

    try:
        from hbllm.security.rbac import Permission, RBACGuard

        guard = RBACGuard(db_path=db_path)

        if not guard.check(tenant_id, acting_user, Permission.ADMIN_MANAGE_USERS):
            raise HTTPException(
                status_code=403,
                detail=f"User '{acting_user}' lacks admin:manage_users permission",
            )

        removed = guard.revoke_role(tenant_id, target_user)
        return {
            "status": "revoked" if removed else "not_found",
            "tenant_id": tenant_id,
            "user_id": target_user,
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@router.post("/studio/rbac/check")
async def studio_rbac_check_permission(request: Request) -> Any:
    """Check if a user has a specific permission.

    Body:
        {
            "user_id": "user_42",
            "permission": "chat:send"
        }
    """
    body = await request.json()
    tenant_id = get_tenant_id(request)
    target_user = body.get("user_id", get_user_id(request))
    permission_name = body.get("permission")

    if not permission_name:
        raise HTTPException(status_code=400, detail="permission is required")

    data_dir = get_data_dir()
    db_path = os.path.join(data_dir, "rbac.db")

    try:
        from hbllm.security.rbac import Permission, RBACGuard

        guard = RBACGuard(db_path=db_path)
        permission = Permission(permission_name)
        allowed = guard.check(tenant_id, target_user, permission)
        role = guard.get_role(tenant_id, target_user)

        return {
            "tenant_id": tenant_id,
            "user_id": target_user,
            "permission": permission_name,
            "allowed": allowed,
            "role": role.value,
        }
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid permission: {permission_name}",
        )


@router.get("/studio/rbac/audit")
async def studio_rbac_audit(request: Request, limit: int = 50) -> Any:
    """Get recent audit log entries for the current tenant."""
    tenant_id = get_tenant_id(request)
    data_dir = get_data_dir()
    db_path = os.path.join(data_dir, "audit.db")

    try:
        from hbllm.security.audit_log import AuditLog

        audit = AuditLog(db_path=db_path)
        entries = audit.query(tenant_id=tenant_id, limit=limit)
        return {
            "tenant_id": tenant_id,
            "entries": entries,
            "count": len(entries),
        }
    except ImportError:
        return {"tenant_id": tenant_id, "entries": [], "error": "AuditLog not available"}
    except Exception as e:
        logger.error("Audit log query failed: %s", e)
        return {"tenant_id": tenant_id, "entries": [], "error": str(e)}
