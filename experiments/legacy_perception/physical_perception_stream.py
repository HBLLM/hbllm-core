"""Physical Perception Stream — Embodied sensorimotor bus bridge.

Bridges real-time physical observations, embodied deliberative thoughts,
and physical motor actions into the HBLLM MessageBus and live CognitiveStream.
Supports Crafter, Overcooked, NetHack, Sokoban, BabyAI, and robotics environments.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from hbllm.network.bus import MessageBus
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


@dataclass
class PhysicalObservationEvent:
    """Structured physical observation captured from an embodied environment."""

    domain: str
    step: int
    data: dict[str, Any]
    correlation_id: str
    timestamp: float = field(default_factory=time.time)
    observation_id: str = field(default_factory=lambda: f"phys_obs_{uuid.uuid4().hex[:10]}")


class PhysicalPerceptionStream:
    """Continuous stream for physical/sensorimotor perceptions and embodied coordination.

    Publishes normalized observations, deliberative thoughts, and motor actions
    to the bus topics:
      - 'embodied.observation'
      - 'embodied.thought'
      - 'embodied.action'
    """

    def __init__(
        self,
        bus: MessageBus,
        domain: str = "general",
        default_correlation_id: str | None = None,
    ) -> None:
        self._bus = bus
        self.domain = domain
        self._default_corr_id = default_correlation_id or str(uuid.uuid4())
        self.step_counter = 0

    def _normalize_observation(self, obs: Any) -> Any:
        """Convert arbitrary environment observations to JSON-serializable structures."""
        if isinstance(obs, dict):
            return {k: self._normalize_observation(v) for k, v in obs.items()}
        if hasattr(obs, "__dataclass_fields__"):
            return asdict(obs)
        if hasattr(obs, "_asdict"):
            return obs._asdict()
        if hasattr(obs, "tolist"):
            return obs.tolist()
        if isinstance(obs, (list, tuple)):
            return [self._normalize_observation(item) for item in obs]
        if isinstance(obs, (int, float, str, bool)) or obs is None:
            return obs
        return str(obs)

    async def ingest_observation(
        self,
        obs: Any,
        step: int | None = None,
        correlation_id: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> PhysicalObservationEvent:
        """Ingest a physical observation, publish to 'embodied.observation', and return event."""
        if step is None:
            step = self.step_counter
            self.step_counter += 1

        corr_id = correlation_id or self._default_corr_id
        normalized = self._normalize_observation(obs)
        if extra_data:
            if isinstance(normalized, dict):
                normalized.update(extra_data)
            else:
                normalized = {"observation": normalized, **extra_data}

        event = PhysicalObservationEvent(
            domain=self.domain,
            step=step,
            data=normalized if isinstance(normalized, dict) else {"raw": normalized},
            correlation_id=corr_id,
        )

        msg = Message.model_construct(
            id=str(uuid.uuid4()),
            type=MessageType.EVENT,
            source_node_id=f"embodied.{self.domain}.sensor",
            topic="embodied.observation",
            payload={
                "domain": self.domain,
                "step": step,
                "observation": event.data,
                "observation_id": event.observation_id,
                "timestamp": event.timestamp,
            },
            correlation_id=corr_id,
        )

        await self._bus.publish("embodied.observation", msg)
        return event

    async def emit_thought(
        self,
        thought: str,
        correlation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish an embodied epistemic reasoning step to 'embodied.thought'."""
        corr_id = correlation_id or self._default_corr_id
        payload = {"text": thought, "domain": self.domain, **(metadata or {})}
        msg = Message.model_construct(
            id=str(uuid.uuid4()),
            type=MessageType.EVENT,
            source_node_id=f"embodied.{self.domain}.cortex",
            topic="embodied.thought",
            payload=payload,
            correlation_id=corr_id,
        )
        await self._bus.publish("embodied.thought", msg)

    async def emit_action(
        self,
        action: Any,
        payload: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        """Publish an embodied motor execution or decision to 'embodied.action'."""
        corr_id = correlation_id or self._default_corr_id
        action_repr = action.name if hasattr(action, "name") else str(action)
        body = {
            "action": action_repr,
            "domain": self.domain,
            **(payload or {}),
        }
        msg = Message.model_construct(
            id=str(uuid.uuid4()),
            type=MessageType.EVENT,
            source_node_id=f"embodied.{self.domain}.motor",
            topic="embodied.action",
            payload=body,
            correlation_id=corr_id,
        )
        await self._bus.publish("embodied.action", msg)
