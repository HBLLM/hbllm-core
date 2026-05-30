"""
Tenant Guard — Defense-in-depth for multi-tenant data isolation.

Provides configurable enforcement levels:
  - OFF:    No enforcement (development only)
  - WARN:   Log violations but allow operations to proceed
  - STRICT: Raise TenantIsolationError on any violation

Features:
  - @require_tenant decorator: validates tenant_id is non-empty
  - @require_identity decorator: validates full (tenant_id, user_id, device_id) triplet
  - TenantContext: async-safe context manager for cross-tenant detection
  - IsolationLevel: configurable data separation strategy

Usage::

    from hbllm.security.tenant_guard import require_tenant, TenantContext

    @require_tenant
    def store(self, tenant_id: str, data: dict): ...

    async with TenantContext("acme", user_id="alice"):
        await store(tenant_id="acme", data={...})   # OK
        await store(tenant_id="other", data={...})   # Blocked/warned
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import logging
import os
from collections.abc import Callable
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── Enums ──────────────────────────────────────────────────────────────────


class TenantGuardMode(str, Enum):
    """Enforcement level for tenant isolation checks."""

    OFF = "off"  # No enforcement (dev only)
    WARN = "warn"  # Log violations, don't block
    STRICT = "strict"  # Raise on violation


class IsolationLevel(str, Enum):
    """Data separation strategy for multi-tenant storage."""

    SHARED = "shared"  # Logical separation (WHERE tenant_id = ?)
    NAMESPACE = "namespace"  # Per-tenant database files
    DEDICATED = "dedicated"  # Fully isolated storage backends


# ─── Exceptions ─────────────────────────────────────────────────────────────


class TenantIsolationError(Exception):
    """Raised when a tenant isolation violation is detected."""

    def __init__(
        self,
        message: str,
        tenant_id: str = "",
        user_id: str = "",
        device_id: str = "",
        operation: str = "",
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.device_id = device_id
        self.operation = operation
        super().__init__(message)


# ─── Async-safe Context (contextvars) ───────────────────────────────────────

_ctx_tenant_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tenant_id", default=None
)
_ctx_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_id", default=None)
_ctx_device_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "device_id", default=None
)
_ctx_guard_mode: contextvars.ContextVar[TenantGuardMode] = contextvars.ContextVar("guard_mode")


def _get_guard_mode() -> TenantGuardMode:
    """Get the current guard mode from context or environment."""
    try:
        return _ctx_guard_mode.get()
    except LookupError:
        pass

    env_mode = os.environ.get("HBLLM_TENANT_GUARD_MODE", "warn").lower()
    try:
        return TenantGuardMode(env_mode)
    except ValueError:
        return TenantGuardMode.WARN


class TenantContext:
    """
    Async-safe tenant context manager using contextvars.

    Sets the current identity scope for a block of operations.
    All guarded operations within this block validate against the set identity.

    Usage::

        async with TenantContext("acme", user_id="alice"):
            # All guarded operations validate against tenant="acme", user="alice"
            await store(tenant_id="acme", ...)   # OK
            await store(tenant_id="other", ...)  # Violation

        # Also works synchronously:
        with TenantContext("acme"):
            store(tenant_id="acme", ...)
    """

    def __init__(
        self,
        tenant_id: str,
        user_id: str | None = None,
        device_id: str | None = None,
        *,
        mode: TenantGuardMode | None = None,
    ):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.device_id = device_id
        self.mode = mode
        self._tokens: list[contextvars.Token] = []

    def __enter__(self) -> TenantContext:
        self._tokens.append(_ctx_tenant_id.set(self.tenant_id))
        if self.user_id is not None:
            self._tokens.append(_ctx_user_id.set(self.user_id))
        if self.device_id is not None:
            self._tokens.append(_ctx_device_id.set(self.device_id))
        if self.mode is not None:
            self._tokens.append(_ctx_guard_mode.set(self.mode))
        return self

    def __exit__(self, *exc: Any) -> None:
        for token in reversed(self._tokens):
            token.var.reset(token)
        self._tokens.clear()

    async def __aenter__(self) -> TenantContext:
        return self.__enter__()

    async def __aexit__(self, *exc: Any) -> None:
        self.__exit__()


def get_current_tenant() -> str | None:
    """Get the current tenant_id from context, or None if not set."""
    return _ctx_tenant_id.get(None)


def get_current_identity() -> tuple[str | None, str | None, str | None]:
    """Get the current (tenant_id, user_id, device_id) from context."""
    return (
        _ctx_tenant_id.get(None),
        _ctx_user_id.get(None),
        _ctx_device_id.get(None),
    )


# ─── Decorators ─────────────────────────────────────────────────────────────


def require_tenant(func: Callable | None = None, *, param: str = "tenant_id"):
    """
    Decorator that enforces tenant_id is present and valid.

    Validates:
      1. tenant_id argument is not empty, None, or whitespace
      2. If inside a TenantContext, tenant_id matches the context

    Behavior depends on TenantGuardMode:
      - OFF:    skip all checks
      - WARN:   log violations, allow through
      - STRICT: raise TenantIsolationError

    Usage::

        @require_tenant
        def search(self, tenant_id: str, query: str): ...

        @require_tenant(param="tid")
        def find(self, tid: str, query: str): ...
    """

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _validate_tenant(fn, args, kwargs, param)
                return await fn(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                _validate_tenant(fn, args, kwargs, param)
                return fn(*args, **kwargs)

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


def require_identity(
    func: Callable | None = None,
    *,
    tenant_param: str = "tenant_id",
    user_param: str = "user_id",
    device_param: str = "device_id",
):
    """
    Decorator that enforces the full identity triplet.

    Validates all three identity components are present and match context.
    """

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _validate_tenant(fn, args, kwargs, tenant_param)
                _validate_identity_field(fn, args, kwargs, user_param, _ctx_user_id)
                _validate_identity_field(fn, args, kwargs, device_param, _ctx_device_id)
                return await fn(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                _validate_tenant(fn, args, kwargs, tenant_param)
                _validate_identity_field(fn, args, kwargs, user_param, _ctx_user_id)
                _validate_identity_field(fn, args, kwargs, device_param, _ctx_device_id)
                return fn(*args, **kwargs)

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator


# ─── Internal Validation ────────────────────────────────────────────────────


def _validate_tenant(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    param: str,
) -> None:
    """Core validation logic for tenant_id."""
    mode = _get_guard_mode()
    if mode == TenantGuardMode.OFF:
        return

    tid = _extract_param(fn, args, kwargs, param)

    # Validate not empty
    if not tid or not str(tid).strip():
        msg = f"{fn.__qualname__}() requires a non-empty '{param}' argument"
        if mode == TenantGuardMode.STRICT:
            raise TenantIsolationError(msg, operation=fn.__qualname__)
        else:
            logger.warning("TENANT_GUARD_WARN: %s", msg)
            return

    # Cross-tenant context check
    ctx_tenant = _ctx_tenant_id.get(None)
    if ctx_tenant is not None and str(tid) != ctx_tenant:
        try:
            from hbllm.security.tenant_registry import get_tenant_registry

            registry = get_tenant_registry()

            # Perform deep recursive ancestor-descendant relation path check (supports DAGs)
            if _is_hierarchically_related(registry, ctx_tenant, str(tid)):
                return
        except Exception as e:
            logger.debug("Recursive hierarchy check failed for %s -> %s: %s", ctx_tenant, tid, e)

        msg = (
            f"Cross-tenant access denied in {fn.__qualname__}(): "
            f"context={ctx_tenant}, requested={tid}"
        )
        if mode == TenantGuardMode.STRICT:
            raise TenantIsolationError(msg, tenant_id=str(tid), operation=fn.__qualname__)
        else:
            logger.warning("TENANT_GUARD_WARN: %s", msg)


def _get_all_ancestors(registry: Any, tenant_id: str) -> set[str]:
    """Perform a Graph Search (BFS) walk up the lineage tree to find all ancestors.

    Fully supports multi-parent DAG hierarchies (e.g. Home and Office networks).
    """
    visited: set[str] = set()
    queue = [tenant_id]

    # Fallback to get_parent_id if get_parents helper is not yet available
    get_parents_fn = getattr(
        registry,
        "get_parents",
        lambda t: [registry.get_parent_id(t)] if registry.get_parent_id(t) else [],
    )

    while queue:
        curr = queue.pop(0)
        if curr not in visited:
            if curr != tenant_id:
                visited.add(curr)
            parents = get_parents_fn(curr)
            for p in parents:
                if p and p not in visited:
                    queue.append(p)
    return visited


def _is_hierarchically_related(registry: Any, ctx_tenant: str, req_tenant: str) -> bool:
    """Verify if two tenants are related along any multi-parent DAG lineage path.

    1. Enforces strict sibling-level isolation.
    2. Allows deep upward/downward path clearances (Platform -> Org -> Device -> Project -> Task).
    3. Natively acts as a flat multi-tenant model if no parent hierarchies exist.
    """
    if ctx_tenant == req_tenant:
        return True

    # 1. Downward path (Is req_tenant a parent/ancestor of ctx_tenant?)
    if req_tenant in _get_all_ancestors(registry, ctx_tenant):
        return True

    # 2. Upward path (Is ctx_tenant a parent/ancestor of req_tenant?)
    if ctx_tenant in _get_all_ancestors(registry, req_tenant):
        return True

    return False


def _validate_identity_field(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    param: str,
    ctx_var: contextvars.ContextVar[str | None],
) -> None:
    """Validate a single identity field against its context var."""
    mode = _get_guard_mode()
    if mode == TenantGuardMode.OFF:
        return

    value = _extract_param(fn, args, kwargs, param)
    ctx_value = ctx_var.get(None)

    if ctx_value is not None and value and str(value) != ctx_value:
        msg = (
            f"Cross-identity access denied in {fn.__qualname__}(): "
            f"context {param}={ctx_value}, requested={value}"
        )
        if mode == TenantGuardMode.STRICT:
            raise TenantIsolationError(msg, operation=fn.__qualname__)
        else:
            logger.warning("TENANT_GUARD_WARN: %s", msg)


def _extract_param(
    fn: Callable,
    args: tuple,
    kwargs: dict,
    param: str,
) -> str | None:
    """Extract a named parameter from function arguments."""
    if param in kwargs:
        return kwargs[param]

    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    if param in params:
        idx = params.index(param)
        if idx < len(args):
            return args[idx]

    # Special handling: if we are looking for tenant/user/device identifier,
    # and one of the arguments is a Message-like object, extract from it.
    for arg in list(args) + list(kwargs.values()):
        if hasattr(arg, "tenant_id") and hasattr(arg, "topic"):
            # This is a Message-like object
            if "tenant" in param:
                return getattr(arg, "tenant_id", None)
            elif "user" in param:
                return getattr(arg, "user_id", None)
            elif "device" in param:
                return getattr(arg, "device_id", None)

    return None
