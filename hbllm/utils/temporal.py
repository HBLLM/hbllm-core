"""
Temporal Reasoning Utilities — time-aware parsing logic.
"""

from __future__ import annotations

from datetime import timedelta

# Keywords that indicate temporal references in user queries
_TEMPORAL_KEYWORDS = {
    "yesterday": timedelta(days=1),
    "last week": timedelta(weeks=1),
    "last month": timedelta(days=30),
    "earlier today": timedelta(hours=6),
    "earlier": timedelta(hours=2),
    "recently": timedelta(hours=24),
    "before": timedelta(hours=1),
    "previously": timedelta(hours=24),
    "last time": timedelta(days=7),
    "a while ago": timedelta(days=3),
}


def parse_temporal_references(text: str) -> list[tuple[str, timedelta]]:
    """Extract temporal references from text."""
    text_lower = text.lower()
    found = []
    for keyword, delta in _TEMPORAL_KEYWORDS.items():
        if keyword in text_lower:
            found.append((keyword, delta))
    return found
