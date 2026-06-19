"""
Environment Sanitization Utilities.

Module-level side effects (like proxy env var cleanup) should happen once
in a single canonical location, not duplicated across modules.
"""

from __future__ import annotations

import os

_SANITIZED = False


def sanitize_proxy_env() -> None:
    """Remove IPv6 ``::1`` entries from ``NO_PROXY``/``no_proxy`` env vars.

    httpx and urllib crash when these contain bare ``::1`` addresses.
    This function is idempotent — safe to call multiple times.
    """
    global _SANITIZED
    if _SANITIZED:
        return

    for env_var in ("NO_PROXY", "no_proxy"):
        env_val = os.environ.get(env_var, "")
        if env_val:
            cleaned = ",".join(part for part in env_val.split(",") if "::1" not in part)
            os.environ[env_var] = cleaned

    _SANITIZED = True
