from __future__ import annotations

import time
from typing import Any


class VectorClock:
    """
    Implements a simple Vector Clock for causal ordering in distributed HBLLM.
    Each node tracks the 'logical time' of all other nodes it interacts with.
    """

    def __init__(self, node_id: str, counters: dict[str, int] | None = None):
        self.node_id = node_id
        self.counters = counters or {node_id: 0}

    def increment(self) -> None:
        """Increment the local counter."""
        self.counters[self.node_id] = self.counters.get(self.node_id, 0) + 1

    def update(self, other: VectorClock) -> None:
        """Merge another vector clock into this one (Max of all counters)."""
        for node_id, count in other.counters.items():
            self.counters[node_id] = max(self.counters.get(node_id, 0), count)

    def compare(self, other: VectorClock) -> str:
        """
        Compare this clock with another.
        Returns: 'before', 'after', 'concurrent', or 'equal'.
        """
        if self.counters == other.counters:
            return "equal"

        greater = False
        less = False

        all_nodes = set(self.counters.keys()) | set(other.counters.keys())
        for node_id in all_nodes:
            c1 = self.counters.get(node_id, 0)
            c2 = other.counters.get(node_id, 0)
            if c1 > c2:
                greater = True
            if c1 < c2:
                less = True

        if greater and not less:
            return "after"
        if less and not greater:
            return "before"
        return "concurrent"

    def to_dict(self) -> dict[str, int]:
        return dict(self.counters)

    @classmethod
    def from_dict(cls, node_id: str, data: dict[str, int]) -> VectorClock:
        return cls(node_id, data)
