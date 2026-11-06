from __future__ import annotations

"""Workspace tier taxonomy (WP6) — leaf module to avoid import cycles."""

from enum import StrEnum


class WorkspaceTier(StrEnum):
    """The four tiers of the cognitive workspace hierarchy."""

    WORKING = "working"  # Task frames — ephemeral scratch per goal
    BRAIN = "brain"  # Session-level cognitive state
    PERSISTENT = "persistent"  # Long-term memory graph (forever)
    META = "meta"  # Cognitive performance & self-model
