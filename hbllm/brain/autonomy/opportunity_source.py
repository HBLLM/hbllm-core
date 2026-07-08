"""Proactive Opportunity Sources & Scorer.

Defines the abstract interface for opportunity detection sources,
the Silence and Battery sources, and the contextual Opportunity Scorer.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from hbllm.brain.autonomy.opportunity import Opportunity
from hbllm.brain.autonomy.presence_state import PresenceState


class OpportunitySource:
    """Base interface for proactive opportunity detection sources."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    def detect(self, presence_state: PresenceState, cognitive_state: Any) -> list[Opportunity]:
        """Scan state to identify potential proactive opportunities.

        Args:
            presence_state: The current passive PresenceState.
            cognitive_state: The current CognitiveStateSnapshot.

        Returns:
            A list of candidate Opportunity objects.
        """
        raise NotImplementedError


class SilenceSource(OpportunitySource):
    """Detects opportunities based on user silence and interaction policies."""

    def __init__(self, policies: dict[str, float] | None = None) -> None:
        super().__init__("silence_monitor")
        # Default policies: category -> silence threshold in seconds
        self.policies = policies or {
            "task": 300.0,          # 5 minutes
            "conversation": 1800.0, # 30 minutes
            "relationship": 28800.0,# 8 hours
            "elderly": 900.0,       # 15 minutes
        }

    def detect(self, presence_state: PresenceState, cognitive_state: Any) -> list[Opportunity]:
        now = time.time()

        # Decide which policy applies based on interaction state
        policy_name = "relationship"
        if presence_state.engagement_level > 0.8:
            policy_name = "conversation"
        elif presence_state.interaction_score > 0.4:
            policy_name = "task"

        threshold = self.policies.get(policy_name, 1800.0)

        # Compute silence duration
        last_activity = max(presence_state.last_user_input, presence_state.last_ai_output)
        if last_activity == 0.0:
            return []

        silence_duration = now - last_activity
        if silence_duration >= threshold:
            opp = Opportunity(
                id=f"opp_silence_{uuid.uuid4().hex[:8]}",
                source=self.source_name,
                category="silence",
                priority=0.4,  # base priority
                urgency=0.3,
                confidence=0.9,
                created_at=now,
                expires_at=now + 600.0,  # expires in 10 minutes
                reason=f"User silent for {int(silence_duration // 60)}m under '{policy_name}' policy.",
                context={
                    "silence_duration": silence_duration,
                    "applied_policy": policy_name,
                    "threshold": threshold,
                },
                suggested_actions=["check_in", "suggest_activity"],
                aging_strategy="escalate",
                aging_rate=0.0001,  # increases priority slowly as silence drags on
            )
            return [opp]

        return []


class BatterySource(OpportunitySource):
    """Detects opportunities relating to low battery under upcoming event pressure."""

    def __init__(self) -> None:
        super().__init__("battery_sensor")

    def detect(self, presence_state: PresenceState, cognitive_state: Any) -> list[Opportunity]:
        sensor_data = presence_state.last_sensor_activity
        battery_level = sensor_data.get("battery_level", 1.0)
        minutes_to_meeting = sensor_data.get("minutes_to_next_meeting", 999.0)

        if battery_level < 0.20 and minutes_to_meeting < 60.0:
            now = time.time()
            opp = Opportunity(
                id=f"opp_battery_{uuid.uuid4().hex[:8]}",
                source=self.source_name,
                category="hardware",
                priority=0.8,  # high base priority
                urgency=0.9,
                confidence=0.95,
                created_at=now,
                expires_at=now + 1200.0,
                reason=f"Battery level is low ({battery_level:.0%}) with a meeting in {int(minutes_to_meeting)}m.",
                context={
                    "battery_level": battery_level,
                    "minutes_to_meeting": minutes_to_meeting,
                },
                suggested_actions=["remind_charge"],
                aging_strategy="none",
            )
            return [opp]

        return []


class OpportunityScorer:
    """Adjusts opportunity priorities based on global cognitive context (stress, goals)."""

    @staticmethod
    def score(opp: Opportunity, cognitive_state: Any) -> Opportunity:
        """Apply cognitive state context to grade an opportunity's priority.

        Args:
            opp: The Opportunity to score.
            cognitive_state: The current CognitiveStateSnapshot.

        Returns:
            The graded Opportunity object.
        """
        stress = getattr(cognitive_state, "stress", 0.0)
        fatigue = getattr(cognitive_state, "fatigue", 0.0)
        focus_target = getattr(cognitive_state, "focus_target", "")

        # If user is stressed, lower the priority of conversational check-ins to prevent annoyance
        if opp.category == "silence":
            if stress > 0.7:
                opp.priority = max(0.1, opp.priority - 0.4)
            elif stress > 0.4:
                opp.priority = max(0.1, opp.priority - 0.2)

        # If focus target is active (working on a task), penalize proactive suggestions (high interruption cost)
        if focus_target and focus_target != "general":
            opp.priority = max(0.1, opp.priority - 0.3)
            opp.urgency = max(0.1, opp.urgency - 0.2)

        # Boost critical warnings if fatigue is high (wellness checks)
        if fatigue > 0.8 and opp.category == "wellness":
            opp.priority = min(1.0, opp.priority + 0.2)

        return opp
