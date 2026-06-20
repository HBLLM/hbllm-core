"""
Tests for the Hardware-Linked Sovereign Mesh Identity resolver.
"""

from __future__ import annotations

import getpass
import socket

from hbllm.security.identity_resolver import resolve_sovereign_identity


def test_resolve_sovereign_identity() -> None:
    """Verify that the hardware identity resolver functions correctly and consistently."""
    tenant_id, device_id = resolve_sovereign_identity()

    # 1. Assert non-empty and type correct
    assert isinstance(tenant_id, str)
    assert isinstance(device_id, str)
    assert len(tenant_id) > 0
    assert len(device_id) > 0

    # 2. Check tenant format: dev_{username}_{mac_hash}
    username = getpass.getuser()
    assert tenant_id.startswith(f"dev_{username}_")
    assert len(tenant_id.split("_")[-1]) == 8  # 8-char hex hash suffix

    # 3. Check device format: host_{hostname}_{host_hash}
    hostname = socket.gethostname()
    assert device_id.startswith(f"host_{hostname}_")
    assert len(device_id.split("_")[-1]) == 8  # 8-char hex hash suffix

    # 4. Assert stability (invoked multiple times yields identical results)
    t2, d2 = resolve_sovereign_identity()
    assert tenant_id == t2
    assert device_id == d2
