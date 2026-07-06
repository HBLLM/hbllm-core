"""
Competition Engine — WTA-based event selection for the cognitive loop.

Implements ``ICompetition``.  Bridges the gap between CognitiveEvent-level
saliency scores and the SNN-level ``HierarchicalWTA``:

    1. Groups events by domain (derived from event type)
    2. Converts event saliency to ``SpikeEvent`` for WTA
    3. Runs local WTA per domain → executive WTA across domains
    4. Maps winning spikes back to ``CognitiveEvent`` objects

This allows the SNN WTA machinery to be reused for cognitive-level
competition without coupling event types to SNN internals.

Usage::

    from hbllm.brain.competition_engine import CompetitionEngine

    engine = CompetitionEngine()
    winners = await engine.compete(scored_events)
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.cognitive_event import CognitiveEvent, CognitiveEventType
from hbllm.brain.cognitive_interfaces import ICompetition
from hbllm.brain.snn.lif import SpikeEvent
from hbllm.brain.snn.wta import HierarchicalWTA, WTAConfig

logger = logging.getLogger(__name__)


# ── Domain mapping: event type → cognitive domain ────────────────────────

_EVENT_DOMAIN: dict[CognitiveEventType, str] = {
    CognitiveEventType.USER_SPOKE: "perception",
    CognitiveEventType.MEMORY_UPDATED: "memory",
    CognitiveEventType.MEMORY_CONFLICT: "memory",
    CognitiveEventType.PREDICTION_FAILED: "prediction",
    CognitiveEventType.ATTENTION_SPIKE: "attention",
    CognitiveEventType.GOAL_ADDED: "goals",
    CognitiveEventType.GOAL_COMPLETED: "goals",
    CognitiveEventType.GOAL_FAILED: "goals",
    CognitiveEventType.EMOTION_CHANGED: "emotion",
    CognitiveEventType.REWARD_RECEIVED: "emotion",
    CognitiveEventType.TASK_COMPLETED: "execution",
    CognitiveEventType.IDLE_DETECTED: "maintenance",
    CognitiveEventType.SIMULATION_COMPLETE: "simulation",
}


class CompetitionEngine(ICompetition):
    """WTA-based competition for cognitive events.

    Uses ``HierarchicalWTA`` from the SNN layer:
        - Local WTA per cognitive domain (memory, goals, etc.)
        - Executive WTA across domain winners

    Args:
        local_k_winners: Number of winners per local domain.
        executive_k_winners: Number of overall winners.
        soft_wta: If True, use soft (graded) competition instead
            of hard winner-take-all.
    """

    def __init__(
        self,
        local_k_winners: int = 2,
        executive_k_winners: int = 3,
        soft_wta: bool = True,
    ) -> None:
        self._local_k = local_k_winners
        self._executive_k = executive_k_winners

        # Build hierarchical WTA with per-domain configs
        local_configs: dict[str, WTAConfig] = {}
        for domain in set(_EVENT_DOMAIN.values()):
            local_configs[domain] = WTAConfig(
                k_winners=local_k_winners,
                soft_wta=soft_wta,
            )

        self._hwta = HierarchicalWTA(
            local_configs=local_configs,
            executive_config=WTAConfig(
                k_winners=executive_k_winners,
                soft_wta=soft_wta,
            ),
        )

        self._compete_count = 0

    async def compete(self, scored_events: list[Any]) -> list[Any]:
        """Run hierarchical WTA competition on scored events.

        Pipeline:
            1. Group events by cognitive domain
            2. Convert each event to a SpikeEvent (saliency → strength)
            3. Run local WTA per domain
            4. Run executive WTA across domain winners
            5. Map winning spikes back to CognitiveEvent objects

        Args:
            scored_events: Events with ``snn_saliency`` already set.

        Returns:
            Winning ``CognitiveEvent`` objects that survived competition.
        """
        if not scored_events:
            return []

        # Group events by domain, tracking original indices
        domain_events: dict[str, list[tuple[int, CognitiveEvent]]] = {}
        for idx, event in enumerate(scored_events):
            if not isinstance(event, CognitiveEvent):
                continue
            domain = _EVENT_DOMAIN.get(event.type, "other")
            domain_events.setdefault(domain, [])
            domain_events[domain].append((idx, event))

        # Run local WTA per domain — collect surviving events
        # We track which events survived local competition by checking
        # which spikes still have fired=True after WTA
        local_surviving_events: list[CognitiveEvent] = []
        local_surviving_spikes_by_domain: dict[str, list[SpikeEvent]] = {}

        for domain, indexed_events in domain_events.items():
            spike_events = [
                SpikeEvent(
                    fired=True,
                    strength=e.effective_priority,
                    timestamp=e.timestamp,
                )
                for _, e in indexed_events
            ]

            result = self._hwta.compete_local(domain, spike_events)

            # Identify survivors: spikes that still have fired=True
            domain_survivors: list[SpikeEvent] = []
            for i, spike in enumerate(result):
                if spike.fired and spike.strength > 0:
                    local_surviving_events.append(indexed_events[i][1])
                    domain_survivors.append(spike)

            if domain_survivors:
                local_surviving_spikes_by_domain[domain] = domain_survivors

        if not local_surviving_events:
            self._compete_count += 1
            return []

        # Run executive WTA across domain winners
        executive_result = self._hwta.compete_executive(local_surviving_spikes_by_domain)

        # Map executive winners back to events using index alignment
        # Build a flat list matching the order events were added to executive
        flat_events: list[CognitiveEvent] = []
        flat_spikes: list[SpikeEvent] = []
        for domain in local_surviving_spikes_by_domain:
            for spike in local_surviving_spikes_by_domain[domain]:
                flat_spikes.append(spike)

        # Rebuild flat_events in same order as they appear in
        # local_surviving_events (which parallels the spike order)
        flat_events = list(local_surviving_events)

        # Executive winners are a subset — find matching events
        # Use timestamp as a stable key (not modified by WTA)
        winner_timestamps = {round(s.timestamp, 9) for s in executive_result if s.fired}

        all_mapped_events: list[CognitiveEvent] = []
        used_timestamps: set[float] = set()
        for event in flat_events:
            ts_key = round(event.timestamp, 9)
            if ts_key in winner_timestamps and ts_key not in used_timestamps:
                all_mapped_events.append(event)
                used_timestamps.add(ts_key)

        # If timestamp matching missed any (e.g., same-timestamp events),
        # fall back to including all local survivors up to executive_k limit
        if not all_mapped_events and local_surviving_events:
            all_mapped_events = local_surviving_events[: self._executive_k]

        # Sort by effective priority descending
        all_mapped_events.sort(key=lambda e: e.effective_priority, reverse=True)
        self._compete_count += 1

        logger.debug(
            "Competition: %d events → %d domain groups → %d winners",
            len(scored_events),
            len(domain_events),
            len(all_mapped_events),
        )

        return all_mapped_events

    def stats(self) -> dict[str, Any]:
        """Competition engine statistics."""
        return {
            "compete_rounds": self._compete_count,
            "local_k": self._local_k,
            "executive_k": self._executive_k,
            "wta_stats": self._hwta.get_stats(),
        }
