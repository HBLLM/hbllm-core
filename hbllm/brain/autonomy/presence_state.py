"""Presence State & Activity Abstractions.

Defines the passive state representing user and environment presence,
along with abstract activity sources that feed presence signals.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PresenceState:
    """Passive representation of user presence and engagement state.

    This state is updated by activity sources and read by opportunity detectors.
    """

    last_user_input: float = 0.0
    last_ai_output: float = 0.0
    last_sensor_activity: dict[str, float] = field(default_factory=dict)
    engagement_level: float = 0.5
    interaction_score: float = 0.5

    def update_user_activity(self, timestamp: float) -> None:
        """Update last user input timestamp and boost engagement."""
        self.last_user_input = timestamp
        self.engagement_level = min(1.0, self.engagement_level + 0.15)
        self.interaction_score = min(1.0, self.interaction_score + 0.10)

    def update_ai_activity(self, timestamp: float) -> None:
        """Update last AI output timestamp."""
        self.last_ai_output = timestamp

    def update_sensor_activity(self, source: str, timestamp: float) -> None:
        """Update activity timestamp for a given sensor source."""
        self.last_sensor_activity[source] = timestamp
        # Subtle boost for environmental activity
        self.engagement_level = min(1.0, self.engagement_level + 0.02)

    def decay_engagement(self, elapsed: float, decay_rate: float = 0.01) -> None:
        """Decay engagement metrics during periods of silence."""
        if elapsed > 0:
            self.engagement_level = max(0.0, self.engagement_level - (decay_rate * elapsed))
            self.interaction_score = max(0.0, self.interaction_score - (decay_rate * elapsed * 0.5))


class ActivitySource:
    """Base class for activity sources reporting presence telemetry."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    def report_activity(self) -> dict[str, Any]:
        """Generate a standardized activity report payload."""
        return {
            "source": self.source_name,
            "timestamp": time.time(),
        }
