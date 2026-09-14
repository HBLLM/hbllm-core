"""
HBLLM Security Package — Defense-in-depth for multi-tenant data isolation.

Provides:
  - TenantGuard: Configurable tenant isolation enforcement
  - TenantInterceptor: Bus-level ambient tenant context propagation
  - AuditLog: Append-only security audit trail
  - EncryptionVault: Field-level encryption at rest
"""

from hbllm.security.audit_trail import (
    AuditAction,
    AuditEntry,
    AuditLog,
    AuditSeverity,
    AuditTrail,
    ComplianceAuditEntry,
    SafetyAuditEntry,
)
from hbllm.security.repository import TenantSQLiteRepository
from hbllm.security.tenant_guard import (
    SystemContext,
    TenantContext,
    TenantGuardMode,
    TenantIsolationError,
    get_current_tenant,
    require_capability,
    require_tenant,
)
from hbllm.security.tenant_interceptor import (
    TenantInterceptor,
    restore_tenant_context,
)

__all__ = [
    "AuditAction",
    "AuditEntry",
    "AuditLog",
    "AuditSeverity",
    "AuditTrail",
    "ComplianceAuditEntry",
    "SafetyAuditEntry",
    "SystemContext",
    "TenantContext",
    "TenantGuardMode",
    "TenantIsolationError",
    "TenantInterceptor",
    "get_current_tenant",
    "require_capability",
    "require_tenant",
    "restore_tenant_context",
    "TenantSQLiteRepository",
]
