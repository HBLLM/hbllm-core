"""World State Engine — live probabilistic graph of reality.

This replaces the old WorldSimulator. Instead of just simulating scenarios,
the WorldStateEngine fuses incoming perception events into a real-time,
cross-device state graph.

It contains:
1. State Fusion Engine (in-memory live graph)
2. Temporal Memory Buffer (confidence decay)
3. Simulation Interface (legacy planning capability retained for "what-if")
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.perception.event_log import EventLog
from hbllm.perception.reality_bus import PerceptionEvent

logger = logging.getLogger(__name__)


# --- Live Graph & Entities ---


@dataclass
class EntityState:
    """Probabilistic state of an entity in the real world."""

    entity_id: str
    properties: dict[str, Any] = field(default_factory=dict)

    # Confidence Tracking (0.0 - 1.0)
    confidence: float = 1.0
    last_updated: float = field(default_factory=time.time)

    # Provenance
    source_set: set[str] = field(default_factory=set)

    def update(self, event: PerceptionEvent, now: float | None = None) -> None:
        """Apply a perception event using probabilistic fusion rules."""
        if now is None:
            now = time.time()

        # Decay existing confidence before fusing
        self.decay_confidence(now)

        # Calculate event weight
        # trust_weight (SYSTEM > APP > SENSOR > INFERRED)
        trust_weights = {"system": 1.0, "app": 0.8, "sensor": 0.5, "inferred": 0.2}
        trust_w = trust_weights.get(event.modality.value, 0.5)

        # recency_weight (fresh events are weighted higher)
        age = now - event.event_timestamp
        recency_w = max(0.1, 1.0 - (age / 3600.0))  # Decays over an hour

        event_weight = trust_w + recency_w + event.confidence
        current_weight = self.confidence * 2.0  # Heuristic scaling

        # If the incoming event is "stronger" than current belief, overwrite properties.
        # Otherwise, probabilistically blend or reject.
        if event_weight >= current_weight:
            for key, val in event.payload.items():
                self.properties[key] = val

            # Boost confidence but cap at 1.0
            self.confidence = min(1.0, event.confidence * event.source_trust + 0.1)
        else:
            # Event is weaker, but still contributes to overall confidence if it aligns
            self.confidence = min(1.0, self.confidence + 0.05)

        self.last_updated = now
        self.source_set.add(event.origin.value)

    def decay_confidence(self, current_time: float, half_life_s: float = 3600.0) -> None:
        """Exponential decay of state confidence over time."""
        age = current_time - self.last_updated
        if age > 0:
            decay_factor = 0.5 ** (age / half_life_s)
            self.confidence *= decay_factor


class WorldStateEngine:
    """The Live Graph of Reality.

    Maintains an in-memory graph of entities. Automatically decays confidence
    and resolves conflicts based on modality trust tiers.
    """

    def __init__(self, event_log: EventLog | None = None) -> None:
        self.event_log = event_log
        self._graph: dict[str, EntityState] = {}
        self._lock = asyncio.Lock()

        # Background tasks
        self._running = False
        self._decay_task: asyncio.Task[Any] | None = None

    def start(self) -> None:
        if not self._running:
            self._running = True
            self._decay_task = asyncio.create_task(self._decay_loop())

    async def stop(self) -> None:
        self._running = False
        if self._decay_task:
            self._decay_task.cancel()
            try:
                await self._decay_task
            except asyncio.CancelledError:
                pass

    async def handle_normalized_event(self, event: PerceptionEvent) -> None:
        """Ingest a normalized event from the EventNormalizer."""
        async with self._lock:
            entity = self._graph.get(event.entity_id)
            if not entity:
                entity = EntityState(entity_id=event.entity_id)
                self._graph[event.entity_id] = entity

            entity.update(event)

    async def _decay_loop(self) -> None:
        """Periodically decay confidence of all entities in the graph."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Check every minute
                now = time.time()
                async with self._lock:
                    for entity in self._graph.values():
                        entity.decay_confidence(now)

                    # Prune entities with near-zero confidence (< 0.05)
                    self._graph = {k: v for k, v in self._graph.items() if v.confidence > 0.05}
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in WorldStateEngine decay loop: %s", e)

    async def boot_recovery(self) -> None:
        """Replay the event log to rebuild the in-memory graph."""
        if not self.event_log:
            return

        logger.info("Rebuilding WorldStateEngine graph from EventLog...")
        async with self._lock:
            self._graph.clear()
            # Fast synchronous replay for boot
            for event in self.event_log.replay():
                entity = self._graph.get(event.entity_id)
                if not entity:
                    entity = EntityState(entity_id=event.entity_id)
                    self._graph[event.entity_id] = entity

                entity.update(event)

            # Apply time decay based on current time
            now = time.time()
            for entity in self._graph.values():
                entity.decay_confidence(now)

        logger.info("Rebuilt WorldStateEngine graph with %d entities.", len(self._graph))

    def get_entity_state(self, entity_id: str) -> EntityState | None:
        """Synchronous read access to an entity's current state."""
        return self._graph.get(entity_id)


# --- Simulation Interface (Retained for Planner) ---


@dataclass
class Scenario:
    """A simulated scenario with predicted outcomes."""

    scenario_id: str
    strategy: str
    steps: list[str]
    predicted_outcome: str
    confidence: float
    expected_reward: float
    risks: list[str] = field(default_factory=list)
    resource_cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Result of simulating multiple scenarios."""

    best_scenario: Scenario
    all_scenarios: list[Scenario]
    simulation_time_ms: float
    consensus_confidence: float


class SimulationInterface:
    """Legacy projection tool retained for 'what-if' planning.

    This is used by the AutonomyCore/Planner to simulate futures without
    affecting the live reality graph.
    """

    def __init__(self, max_scenarios: int = 5, risk_weight: float = 0.3):
        self.max_scenarios = max_scenarios
        self.risk_weight = risk_weight

    async def simulate(
        self,
        goal: str,
        strategies: list[dict[str, Any]],
        predict_fn: Any = None,
    ) -> SimulationResult:
        """Simulate multiple strategies and select the best."""
        start = time.monotonic()

        scenarios: list[Scenario] = []
        for i, strategy in enumerate(strategies[: self.max_scenarios]):
            scenario = await self._simulate_strategy(f"scenario_{i}", goal, strategy, predict_fn)
            scenarios.append(scenario)

        scenarios.sort(
            key=lambda s: s.expected_reward * s.confidence - len(s.risks) * self.risk_weight,
            reverse=True,
        )

        best = (
            scenarios[0]
            if scenarios
            else Scenario(
                scenario_id="fallback",
                strategy="direct",
                steps=["Execute directly"],
                predicted_outcome="Unknown",
                confidence=0.3,
                expected_reward=0.0,
            )
        )

        consensus = best.confidence
        if len(scenarios) >= 2:
            top_scores = [s.expected_reward for s in scenarios[:3]]
            spread = max(top_scores) - min(top_scores)
            consensus = max(0.0, 1.0 - spread)

        elapsed = (time.monotonic() - start) * 1000

        return SimulationResult(
            best_scenario=best,
            all_scenarios=scenarios,
            simulation_time_ms=elapsed,
            consensus_confidence=round(consensus, 3),
        )

    async def _simulate_strategy(
        self,
        scenario_id: str,
        goal: str,
        strategy: dict[str, Any],
        predict_fn: Any = None,
    ) -> Scenario:
        name = strategy.get("name", "unnamed")
        steps = strategy.get("steps", [])
        tools = strategy.get("tools", [])

        if predict_fn:
            prediction = await predict_fn(strategy)
            outcome = prediction.get("outcome", "Unknown")
            confidence = prediction.get("confidence", 0.5)
            reward = prediction.get("reward", 0.0)
            risks = prediction.get("risks", [])
        else:
            outcome, confidence, reward, risks = self._heuristic_predict(goal, steps, tools)

        return Scenario(
            scenario_id=scenario_id,
            strategy=name,
            steps=steps,
            predicted_outcome=outcome,
            confidence=confidence,
            expected_reward=reward,
            risks=risks,
            resource_cost=len(steps) * 0.1,
        )

    def _heuristic_predict(
        self,
        goal: str,
        steps: list[str],
        tools: list[str],
    ) -> tuple[str, float, float, list[str]]:
        risks: list[str] = []
        confidence = 0.6
        reward = 0.5

        if len(steps) > 5:
            risks.append("Complex multi-step plan may have cascading failures")
            confidence -= 0.1
        if len(steps) > 10:
            risks.append("Very long plan — consider breaking into sub-goals")
            confidence -= 0.15

        external_tools = [t for t in tools if t in {"api", "browser", "database"}]
        if external_tools:
            risks.append(f"Depends on external tools: {external_tools}")
            confidence -= 0.05

        if len(steps) <= 3:
            confidence += 0.1
            reward += 0.1

        confidence = max(0.1, min(1.0, confidence))
        reward = max(-1.0, min(1.0, reward))
        outcome = f"Execute {len(steps)} steps to achieve: {goal[:100]}"

        return outcome, confidence, reward, risks
