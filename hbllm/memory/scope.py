from __future__ import annotations

from enum import StrEnum


class MemoryScope(StrEnum):
    """
    Defines the synchronization and visibility scope of a memory entry.
    """

    WORKING = "working"  # Ephemeral, device-only, task-specific.
    EPISODIC = "episodic"  # User-specific history, synced across user's devices.
    SEMANTIC = "semantic"  # Tenant-level shared knowledge / facts.
    SENSITIVE = "sensitive"  # PII, credentials, local-only, NEVER SYNCED.
