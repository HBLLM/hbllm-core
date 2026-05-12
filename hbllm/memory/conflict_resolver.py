"""
Memory Conflict Resolver — handles contradictory information in distributed HBLLM.

Uses a combination of:
1. Causal ordering (Vector Clocks)
2. Authority score (Who is more trusted?)
3. Recency (If all else fails)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from hbllm.network.clocks import VectorClock


class MemoryConflictResolver:
    """
    Decides between two memory fragments when a conflict is detected.
    """

    @staticmethod
    def resolve(
        fragment_a: dict[str, Any],
        fragment_b: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Compare two memory fragments and return the 'winning' one.

        Expected fragment dict structure:
            content: str
            vector_clock: dict[str, int] | None
            authority_score: int
            timestamp: str (ISO format)
        """
        # 1. Causal Check
        clock_a_dict = fragment_a.get("vector_clock")
        clock_b_dict = fragment_b.get("vector_clock")

        if clock_a_dict and clock_b_dict:
            # We need a dummy node_id to compare
            node_id = "resolver"
            vc_a = VectorClock.from_dict(node_id, clock_a_dict)
            vc_b = VectorClock.from_dict(node_id, clock_b_dict)

            relation = vc_a.compare(vc_b)
            if relation == "after":
                return fragment_a
            if relation == "before":
                return fragment_b

        # 2. Authority Check (Conflict or Concurrent)
        auth_a = fragment_a.get("authority_score", 50)
        auth_b = fragment_b.get("authority_score", 50)

        if auth_a > auth_b:
            return fragment_a
        if auth_b > auth_a:
            return fragment_b

        # 3. Recency Check (Fallback)
        try:
            ts_a = datetime.fromisoformat(fragment_a["timestamp"].replace("Z", "+00:00"))
            ts_b = datetime.fromisoformat(fragment_b["timestamp"].replace("Z", "+00:00"))

            if ts_a > ts_b:
                return fragment_a
            return fragment_b
        except (KeyError, ValueError):
            # If no timestamp, just return A
            return fragment_a
