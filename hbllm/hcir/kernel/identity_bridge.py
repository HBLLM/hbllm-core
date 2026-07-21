"""
Cognitive Identity Bridge — Bridges TenantContext ↔ HCIRExecutionContext.

Maps legacy TenantContext (tenant_id, user_id, device_id) into HCIR
HCIRExecutionContext and Scope for kernel-level tenant isolation.
"""

from __future__ import annotations

import logging

from hbllm.hcir.context import HCIRExecutionContext
from hbllm.hcir.types import Scope
from hbllm.security.tenant_guard import TenantContext

logger = logging.getLogger(__name__)


class IdentityBridge:
    """Bidirectional bridge between security TenantContext and HCIRExecutionContext."""

    @staticmethod
    def context_from_tenant(
        tenant_context: TenantContext | None = None,
        tenant_id: str = "default",
        user_id: str = "system",
        device_id: str | None = None,
        process_id_str: str = "proc_main",
        thread_id_str: str = "thr_main",
    ) -> HCIRExecutionContext:
        """Construct an HCIRExecutionContext from a TenantContext or explicit identities."""
        effective_tenant = tenant_id
        effective_user = user_id
        effective_device = device_id

        if tenant_context is not None:
            if getattr(tenant_context, "tenant_id", None):
                effective_tenant = tenant_context.tenant_id
            if getattr(tenant_context, "user_id", None):
                effective_user = tenant_context.user_id
            if getattr(tenant_context, "device_id", None):
                effective_device = tenant_context.device_id

        scope = Scope(
            tenant_id=effective_tenant,
            user_id=effective_user or "system",
            device_id=effective_device or "device_local",
        )

        exec_context = HCIRExecutionContext(
            tenant_scope=scope,
            metadata={
                "tenant_id": effective_tenant,
                "user_id": effective_user,
                "device_id": effective_device,
                "process_id_str": process_id_str,
                "thread_id_str": thread_id_str,
            },
        )
        return exec_context
