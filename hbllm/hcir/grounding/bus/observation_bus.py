from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from hbllm.hcir.grounding.bus.channel import ChannelMetrics, ObservationChannel
from hbllm.hcir.grounding.bus.policies import ChannelPolicy
from hbllm.hcir.grounding.contracts.enums import ModalitySource, TrafficClass
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope

logger = logging.getLogger(__name__)

SubscriberCallback = Callable[[ObservationEnvelope], None]


class ObservationBus:
    """Central multimodal sensory multiplexer and provenance-preserving ingestion hub.

    Multiplexes channels with traffic-class specific QoS policies, routes envelopes
    to subscribers by modality or channel ID, and maintains an ordered historical replay log.
    """

    def __init__(self, replay_capacity: int = 5000) -> None:
        self._channels: dict[str, ObservationChannel] = {}
        self._lock = threading.RLock()
        self._subscribers: list[dict[str, Any]] = []
        self._replay_capacity = replay_capacity
        self._replay_log: list[ObservationEnvelope] = []

    def register_channel(
        self,
        channel_id: str,
        policy: ChannelPolicy | None = None,
        traffic_class: TrafficClass = TrafficClass.CLASS_HIGH_FREQ,
    ) -> ObservationChannel:
        """Register a new sensor channel with an explicit policy or traffic class default."""
        with self._lock:
            if channel_id in self._channels:
                return self._channels[channel_id]

            effective_policy = policy or ChannelPolicy.default_for(traffic_class)
            channel = ObservationChannel(channel_id=channel_id, policy=effective_policy)
            self._channels[channel_id] = channel
            logger.debug(
                "Registered observation channel %s with policy %s", channel_id, effective_policy
            )
            return channel

    def get_channel(self, channel_id: str) -> ObservationChannel | None:
        with self._lock:
            return self._channels.get(channel_id)

    def publish(self, envelope: ObservationEnvelope, timeout: float | None = None) -> bool:
        """Publish an observation envelope into its registered channel.

        Auto-registers channel if not yet registered using envelope's traffic_class.
        Dispatches to all matching registered subscribers and appends to replay log.
        """
        with self._lock:
            channel = self._channels.get(envelope.source_channel_id)
            if channel is None:
                channel = self.register_channel(
                    envelope.source_channel_id,
                    traffic_class=envelope.traffic_class,
                )

        # Ingest into channel (enforcing channel policies and sequence checks)
        success = channel.ingest(envelope, timeout=timeout)
        if not success:
            return False

        with self._lock:
            # Store in replay log
            self._replay_log.append(envelope)
            if len(self._replay_log) > self._replay_capacity:
                self._replay_log.pop(0)

            # Match subscribers
            matched_callbacks: list[SubscriberCallback] = []
            for sub in self._subscribers:
                if sub["channel_id"] and sub["channel_id"] != envelope.source_channel_id:
                    continue
                if sub["modality"] and sub["modality"] != envelope.source_modality:
                    continue
                if sub["traffic_class"] and sub["traffic_class"] != envelope.traffic_class:
                    continue
                matched_callbacks.append(sub["callback"])

        # Invoke callbacks outside lock to prevent deadlock
        for cb in matched_callbacks:
            try:
                cb(envelope)
            except Exception as e:
                logger.error("Error in observation subscriber callback: %s", e, exc_info=True)

        return True

    def publish_dialogue(
        self,
        text: str,
        speaker_id: str = "user",
        channel_id: str = "user_dialogue",
        dialogue_act: str = "command",
        named_entities: dict[str, list[str]] | list[dict[str, Any]] | None = None,
        embeddings: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ObservationEnvelope:
        """Publishes a natural language dialogue utterance as an SEMANTIC_LLM_ADAPTER envelope."""
        from hbllm.hcir.grounding.features.composite_observation_adapter import (
            SemanticLLMObservationAdapter,
        )

        adapter = SemanticLLMObservationAdapter(channel_id=channel_id)
        envelope, _ = adapter.adapt(
            text=text,
            speaker_id=speaker_id,
            dialogue_act=dialogue_act,
            named_entities=named_entities,
            embeddings=embeddings,
            metadata=metadata,
        )
        self.publish(envelope, timeout=timeout)
        return envelope

    def subscribe(
        self,
        callback: SubscriberCallback,
        channel_id: str | None = None,
        modality: ModalitySource | None = None,
        traffic_class: TrafficClass | None = None,
    ) -> None:
        """Subscribe to observation envelopes filtered by channel, modality, or traffic class."""
        with self._lock:
            self._subscribers.append(
                {
                    "callback": callback,
                    "channel_id": channel_id,
                    "modality": modality,
                    "traffic_class": traffic_class,
                }
            )

    def replay(
        self,
        since_wall_time: float = 0.0,
        modality: ModalitySource | None = None,
    ) -> list[ObservationEnvelope]:
        """Retrieve ordered historical observations from the replay buffer."""
        with self._lock:
            return [
                env
                for env in self._replay_log
                if env.wall_timestamp >= since_wall_time
                and (modality is None or env.source_modality == modality)
            ]

    def get_all_metrics(self) -> dict[str, ChannelMetrics]:
        """Collect metrics across all registered channels."""
        with self._lock:
            return {cid: ch.metrics for cid, ch in self._channels.items()}

    def clear(self) -> None:
        """Clear all channels and the replay buffer."""
        with self._lock:
            for ch in self._channels.values():
                ch.clear()
            self._replay_log.clear()
