"""Proactive Opportunity Sources & Scorer.

Defines the abstract interface for opportunity detection sources,
the Silence and Battery sources, and the contextual Opportunity Scorer.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from hbllm.brain.autonomy.opportunity import Opportunity
from hbllm.brain.autonomy.presence_state import PresenceState


class OpportunitySource(ABC):
    """Base interface for proactive opportunity detection sources."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    @abstractmethod
    def detect(self, presence_state: PresenceState, cognitive_state: Any) -> list[Opportunity]:
        """Scan state to identify potential proactive opportunities.

        Args:
            presence_state: The current passive PresenceState.
            cognitive_state: The current CognitiveStateSnapshot.

        Returns:
            A list of candidate Opportunity objects.
        """
        pass


class SilenceSource(OpportunitySource):
    """Detects opportunities based on user silence and interaction policies."""

    def __init__(self, policies: dict[str, float] | None = None) -> None:
        super().__init__("silence_monitor")
        # Default policies: category -> silence threshold in seconds
        self.policies = policies or {
            "task": 300.0,  # 5 minutes
            "conversation": 1800.0,  # 30 minutes
            "relationship": 28800.0,  # 8 hours
            "elderly": 900.0,  # 15 minutes
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


class ReflectionSource(OpportunitySource):
    """Generates reflection opportunities based on internal state triggers."""

    def __init__(self) -> None:
        super().__init__("internal_reflection")
        # List of active triggers that have fired
        self.active_triggers: list[dict[str, Any]] = []

    def add_trigger(self, trigger_type: str, payload: dict[str, Any] | None = None) -> None:
        """Register a new trigger event from the system."""
        self.active_triggers.append(
            {
                "type": trigger_type,
                "payload": payload or {},
                "timestamp": time.time(),
            }
        )

    def detect(self, presence_state: PresenceState, cognitive_state: Any) -> list[Opportunity]:
        now = time.time()
        opportunities: list[Opportunity] = []

        # We consume all pending triggers and turn them into Reflection Opportunities
        # To avoid duplicated reflections, we clear triggers as we process them.
        triggers_to_process = list(self.active_triggers)
        self.active_triggers.clear()

        for trigger in triggers_to_process:
            trigger_type = trigger["type"]
            payload = trigger["payload"]

            # Map the trigger type to specific objectives, expected values, resource/interruption costs
            objective = "memory_consolidation"
            priority = 0.5
            urgency = 0.3
            expected_value = 0.5
            resource_cost = 0.2
            interruption_cost = 0.1
            confidence = 0.9
            requires: list[str] = []
            blocks: list[str] = []
            conflicts: list[str] = []

            if trigger_type == "user_idle":
                level = payload.get("level", "idle")
                if level == "deep_idle":
                    objective = "deep_memory_consolidation"
                    priority = 0.7
                    urgency = 0.4
                    expected_value = 0.8
                    resource_cost = 0.7  # heavy computation allowed when user is deeply idle
                    interruption_cost = 0.05
                    blocks = ["conversation"]  # don't start conversations while consolidating
                else:
                    objective = "lightweight_reflection"
                    priority = 0.5
                    urgency = 0.3
                    expected_value = 0.5
                    resource_cost = 0.3
                    interruption_cost = 0.1

            elif trigger_type == "task_completed":
                objective = "knowledge_review"
                priority = 0.6
                urgency = 0.5
                expected_value = 0.7
                resource_cost = 0.4
                interruption_cost = 0.2

            elif trigger_type == "goal_failed":
                objective = "error_analysis"
                priority = 0.8
                urgency = 0.8
                expected_value = 0.9
                resource_cost = 0.5
                interruption_cost = 0.3
                confidence = 0.95

            elif trigger_type == "goal_achieved":
                objective = "strategy_evaluation"
                priority = 0.6
                urgency = 0.4
                expected_value = 0.7
                resource_cost = 0.3
                interruption_cost = 0.2

            elif trigger_type == "memory_conflict":
                objective = "conflict_resolution"
                priority = 0.75
                urgency = 0.6
                expected_value = 0.8
                resource_cost = 0.4
                interruption_cost = 0.2

            elif trigger_type == "knowledge_gap":
                objective = "knowledge_retrieval"
                priority = 0.55
                urgency = 0.4
                expected_value = 0.6
                resource_cost = 0.3
                interruption_cost = 0.1

            elif trigger_type == "scheduled":
                objective = "self_improvement"
                priority = 0.5
                urgency = 0.2
                expected_value = 0.6
                resource_cost = 0.5
                interruption_cost = 0.1

            elif trigger_type == "sleep":
                objective = "sleep_cycle_consolidation"
                priority = 0.85
                urgency = 0.9
                expected_value = 0.95
                resource_cost = 0.9  # sleep allows full resource dedication
                interruption_cost = 0.0
                blocks = ["conversation", "task"]

            opp = Opportunity(
                id=f"opp_reflection_{trigger_type}_{uuid.uuid4().hex[:8]}",
                source=self.source_name,
                category="reflection",
                priority=priority,
                urgency=urgency,
                confidence=confidence,
                created_at=now,
                expires_at=now + 1800.0,  # expires in 30 minutes
                reason=f"Internal reflection triggered by {trigger_type}. Objective: {objective}.",
                context={
                    "trigger_type": trigger_type,
                    "objective": objective,
                    "trigger_payload": payload,
                },
                suggested_actions=["consolidate_memory", "update_knowledge_graph"],
                expected_value=expected_value,
                resource_cost=resource_cost,
                interruption_cost=interruption_cost,
                requires=requires,
                blocks=blocks,
                conflicts=conflicts,
            )
            opportunities.append(opp)

        return opportunities
