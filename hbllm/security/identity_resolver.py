"""
Identity Resolver — Hardware-linked identity generation for sovereign mesh nodes.

Generates stable, cryptographically unique identity boundaries for mesh
execution by combining system username, physical MAC address, and hostname.
"""

from __future__ import annotations

import getpass
import hashlib
import socket
import uuid


def resolve_sovereign_identity() -> tuple[str, str]:
    """Resolve a globally unique, stable, hardware-linked identity pair.

    Returns:
        tuple[str, str]: A pair of (tenant_id, device_id).
          - tenant_id: Represents the sovereign identity (e.g. dev_{username}_{mac_hash})
          - device_id: Represents the specific physical device (e.g. host_{hostname}_{host_hash})
    """
    # 1. System Username
    username = getpass.getuser()

    # 2. Hardware MAC Address (returns 48-bit integer)
    mac_node = uuid.getnode()
    mac_hash = hashlib.sha256(str(mac_node).encode("utf-8")).hexdigest()[:8]

    # 3. Hostname and network details
    hostname = socket.gethostname()
    host_hash = hashlib.sha256(hostname.encode("utf-8")).hexdigest()[:8]

    # Format stable signatures
    tenant_id = f"dev_{username}_{mac_hash}"
    device_id = f"host_{hostname}_{host_hash}"

    return tenant_id, device_id
