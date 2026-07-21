"""
World State Snapshot — Immutable State Container for Deterministic Simulation & Replay.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorldStateSnapshot:
    """Immutable snapshot of environmental variables and physical entity states."""

    world_id: str
    timestamp: float = field(default_factory=time.time)
    variables: dict[str, Any] = field(default_factory=dict)
    entity_states: dict[str, str] = field(default_factory=dict)
    state_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Compute deterministic SHA256 state hash for simulation and replay matching."""
        raw_payload = {
            "world_id": self.world_id,
            "variables": sorted(self.variables.items()),
            "entity_states": sorted(self.entity_states.items()),
        }
        encoded = json.dumps(raw_payload, sort_keys=True).encode("utf-8")
        computed_hash = hashlib.sha256(encoded).hexdigest()[:16]
        object.__setattr__(self, "state_hash", computed_hash)
