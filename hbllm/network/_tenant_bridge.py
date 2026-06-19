"""
Tenant Bridge — Lazy-import bridge between network and security layers.

Provides a helper that bus dispatch implementations call to restore
TenantContext from Message fields without hard-coupling the network
package to the security package.

If the security package is not available (e.g., minimal install),
the helper gracefully falls back to a no-op context manager.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


@contextmanager
def restore_tenant_ctx(message: Any) -> Generator[None, None, None]:
    """
    Restore ambient TenantContext from a Message during handler dispatch.

    Lazy-imports ``restore_tenant_context`` from the security package.
    Falls back to a no-op if the security package is unavailable.

    Usage in bus dispatch::

        with restore_tenant_ctx(m):
            response = await handler(m)
    """
    try:
        from hbllm.security.tenant_interceptor import restore_tenant_context

        with restore_tenant_context(message):
            yield
    except ImportError:
        yield
