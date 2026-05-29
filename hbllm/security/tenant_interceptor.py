"""
Tenant Interceptor — Bus-level ambient tenant context propagation.

Bridges contextvars (TenantContext) ↔ Message fields so that:
  - On publish: ambient tenant identity is stamped onto messages automatically
  - On delivery: handlers run inside the correct TenantContext

This gives ALL subsystems (mesh, plugins, memory, actions) automatic
tenant isolation without per-component `tenant_id` threading.

Usage::

    # Wire on startup (done by BrainFactory):
    bus.add_interceptor(TenantInterceptor())

    # Handlers automatically get correct tenant scope via bus dispatch.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from hbllm.security.tenant_guard import (
    TenantContext,
    _ctx_device_id,
    _ctx_tenant_id,
    _ctx_user_id,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class TenantInterceptor:
    """
    MessageBus interceptor that auto-stamps ambient TenantContext onto messages.

    When code publishes a message while inside a TenantContext, this interceptor
    copies the ambient (tenant_id, user_id, device_id) onto the Message if the
    Message still has default values. This ensures messages always carry the
    correct tenant identity without manual threading.

    Rules:
      - Only stamps fields that are still ``"default"`` (preserves explicit values).
      - If no ambient context is set, the message passes through untouched.
      - Never blocks or drops messages (returns the message unconditionally).
    """

    async def __call__(self, message: Message) -> Message | None:
        """Stamp ambient tenant context onto the message if not already set."""
        # Read ambient context from contextvars
        ctx_tenant = _ctx_tenant_id.get(None)
        ctx_user = _ctx_user_id.get(None)
        ctx_device = _ctx_device_id.get(None)

        # Only stamp if message still has default values
        if ctx_tenant and message.tenant_id == "default":
            message.tenant_id = ctx_tenant

        if ctx_user and message.user_id == "default":
            message.user_id = ctx_user

        if ctx_device and message.device_id == "default":
            message.device_id = ctx_device

        return message


@contextmanager
def restore_tenant_context(message: Message) -> Generator[None, None, None]:
    """
    Restore TenantContext from a Message's fields for handler execution.

    Used by bus dispatch implementations to ensure every handler runs inside
    the correct tenant scope, enabling ``get_current_tenant()`` and
    ``@require_tenant`` to work correctly within handlers.

    Usage::

        with restore_tenant_context(message):
            response = await handler(message)
            # get_current_tenant() == message.tenant_id here
    """
    tenant_id = message.tenant_id if message.tenant_id != "default" else None
    user_id = message.user_id if message.user_id != "default" else None
    device_id = message.device_id if message.device_id != "default" else None

    if tenant_id:
        with TenantContext(tenant_id, user_id=user_id, device_id=device_id):
            yield
    else:
        yield
