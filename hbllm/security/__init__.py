"""
HBLLM Security Package — Defense-in-depth for multi-tenant data isolation.

Provides:
  - TenantGuard: Configurable tenant isolation enforcement
  - AuditLog: Append-only security audit trail
  - EncryptionVault: Field-level encryption at rest
"""

from hbllm.security.tenant_guard import (
    TenantContext,
    TenantGuardMode,
    TenantIsolationError,
    get_current_tenant,
    require_tenant,
)

__all__ = [
    "TenantContext",
    "TenantGuardMode",
    "TenantIsolationError",
    "get_current_tenant",
    "require_tenant",
]
