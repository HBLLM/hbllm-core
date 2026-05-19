"""
Routing Intelligence Layer (RIL) — The decision engine for message routing.

The RIL evaluates the best transport for each message using a probabilistic
scoring model. It implements the MessageBus protocol so existing nodes can
use it as a drop-in replacement, while internally dispatching across multiple
transports (InProcess, WebSocket, Redis, WebRTC).

The RIL:
  - Chooses the best transport per message.
  - Maintains fallback chains.
  - Attaches ExecutionContext for traceability.

The RIL MUST NOT:
  - Manage memory sync logic.
  - Handle cognitive decisions.
  - Run discovery logic directly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

from hbllm.network.bus import MessageHandler, Subscription
from hbllm.network.messages import Message
from hbllm.network.routing.context import ExecutionContext
from hbllm.network.transports.base import Transport, TransportState

logger = logging.getLogger(__name__)


class RoutingIntelligenceLayer:
    """
    Central routing engine that dispatches messages across multiple transports.

    Implements a scoring model to select the optimal transport per message:

        score = capability_match + latency_score + reliability_score
                + trust_score - load_penalty

    For Phase 1, the scoring is simplified to:
      1. Local (InProcess) is always preferred if it has subscribers.
      2. Fall back to the next connected transport with lowest latency.

    The RIL exposes the same publish/subscribe/request interface as
    MessageBus, so existing nodes can use it transparently.
    """

    def __init__(self, node_id: str = "local") -> None:
        self.node_id = node_id
        self._transports: dict[str, Transport] = {}
        self._transport_priority: list[str] = []  # Ordered by preference
        self._interceptors: list[Callable[[Message], Coroutine[Any, Any, Message | None]]] = []
        self._running = False

    # ── Transport Management ──────────────────────────────────────────

    def register_transport(self, transport: Transport, priority: int = 0) -> None:
        """
        Register a transport with the RIL.

        Args:
            transport: The transport instance.
            priority: Lower = higher preference. InProcess should be 0,
                      Redis 10, WebSocket 20, WebRTC 5 (for local peers).
        """
        self._transports[transport.transport_id] = transport
        # Re-sort priority list
        self._transport_priority.append(transport.transport_id)
        self._transport_priority.sort(
            key=lambda tid: self._get_transport_score(tid),
            reverse=True,
        )
        logger.info(
            "RIL registered transport '%s' (type=%s)",
            transport.transport_id,
            transport.transport_type,
        )

    def deregister_transport(self, transport_id: str) -> None:
        """Remove a transport from the RIL."""
        self._transports.pop(transport_id, None)
        if transport_id in self._transport_priority:
            self._transport_priority.remove(transport_id)
        logger.info("RIL deregistered transport '%s'", transport_id)

    def get_transport(self, transport_id: str) -> Transport | None:
        return self._transports.get(transport_id)

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all registered transports."""
        self._running = True
        for transport in self._transports.values():
            try:
                await transport.start()
            except Exception as e:
                logger.error("Failed to start transport '%s': %s", transport.transport_id, e)

        logger.info(
            "RIL started with %d transports: %s",
            len(self._transports),
            list(self._transports.keys()),
        )

    async def stop(self) -> None:
        """Stop all transports."""
        self._running = False
        for transport in self._transports.values():
            try:
                await transport.stop()
            except Exception as e:
                logger.error("Failed to stop transport '%s': %s", transport.transport_id, e)

        logger.info("RIL stopped")

    # ── MessageBus-compatible Interface ───────────────────────────────

    def add_interceptor(
        self, interceptor: Callable[[Message], Coroutine[Any, Any, Message | None]]
    ) -> None:
        """Add a proactive message interceptor."""
        self._interceptors.append(interceptor)

    async def publish(self, topic: str, message: Message) -> None:
        """
        Route a message to the best available transport.

        Routing decision (Phase 1 — deterministic):
        1. If InProcess transport has subscribers for the topic → use it.
        2. Otherwise, use the first connected transport.
        """
        if not self._running:
            logger.warning("RIL not running, dropping message %s", message.id)
            return

        # Run interceptors
        for interceptor in self._interceptors:
            try:
                result = await interceptor(message)
                if result is None:
                    return  # Blocked
                message = result
            except Exception as e:
                logger.error("RIL interceptor failed: %s", e)
                return

        # Create execution context
        ctx = ExecutionContext(
            origin_node=self.node_id,
            origin_transport="ril",
        )

        # Select transport
        selected = self._select_transport(topic, message, ctx)
        if selected is None:
            logger.warning(
                "RIL: no transport available for topic '%s', message %s dropped",
                topic,
                message.id,
            )
            return

        ctx.selected_transport = selected.transport_id
        ctx.routing_score = self._get_transport_score(selected.transport_id)

        # Record hop
        ctx.add_hop(
            transport_id=selected.transport_id,
            transport_type=selected.transport_type,
            node_id=self.node_id,
        )

        # Store context in message metadata for traceability
        message.payload.setdefault("_execution_context", ctx.model_dump())

        start = time.monotonic()
        try:
            await selected.send(topic, message)
            latency = (time.monotonic() - start) * 1000
            logger.debug(
                "RIL routed '%s' via %s (%.1fms)",
                topic,
                selected.transport_id,
                latency,
            )
        except Exception as e:
            logger.error(
                "RIL: transport '%s' failed for topic '%s': %s. Attempting fallback.",
                selected.transport_id,
                topic,
                e,
            )
            # Attempt fallback
            await self._fallback_send(topic, message, ctx, exclude=selected.transport_id)

    async def request(self, topic: str, message: Message, timeout: float = 90.0) -> Message:
        """
        Route a request to the best transport and wait for a response.
        """
        if not self._running:
            raise RuntimeError("RIL is not running")

        # Run interceptors
        for interceptor in self._interceptors:
            try:
                result = await interceptor(message)
                if result is None:
                    raise RuntimeError("Message blocked by interceptor")
                message = result
            except Exception as e:
                raise RuntimeError(f"RIL interceptor failed: {e}") from e

        # Select transport
        ctx = ExecutionContext(
            origin_node=self.node_id,
            origin_transport="ril",
        )
        selected = self._select_transport(topic, message, ctx)
        if selected is None:
            raise RuntimeError(f"No transport available for topic '{topic}'")

        ctx.selected_transport = selected.transport_id
        ctx.add_hop(
            transport_id=selected.transport_id,
            transport_type=selected.transport_type,
            node_id=self.node_id,
        )

        message.payload.setdefault("_execution_context", ctx.model_dump())
        return await selected.send_request(topic, message, timeout=timeout)

    async def subscribe(
        self, topic: str, handler: MessageHandler, tenant_id: str | None = None
    ) -> Subscription:
        """
        Subscribe to a topic. Subscribes on ALL transports that support it.
        This ensures messages arriving from any transport are handled.
        """
        subscriptions: list[Subscription] = []
        for transport in self._transports.values():
            if hasattr(transport, "subscribe"):
                sub = await transport.subscribe(topic, handler, tenant_id=tenant_id)
                subscriptions.append(sub)

        # Return the first subscription as the handle
        # (in practice, nodes only need one handle to unsubscribe)
        if subscriptions:
            return subscriptions[0]
        raise RuntimeError(f"No transport supports subscriptions for topic '{topic}'")

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Unsubscribe from a topic on all transports."""
        for transport in self._transports.values():
            if hasattr(transport, "unsubscribe"):
                try:
                    await transport.unsubscribe(subscription)
                except Exception:
                    pass  # Subscription may not exist on this transport

    def has_subscribers(self, topic: str) -> bool:
        """Check if any transport has subscribers for a topic."""
        return any(t.has_subscribers(topic) for t in self._transports.values())

    # ── Transport Scoring & Selection ─────────────────────────────────

    def _select_transport(
        self,
        topic: str,
        message: Message,
        ctx: ExecutionContext,
    ) -> Transport | None:
        """
        Select the best transport for a message.

        Phase 1 scoring (deterministic):
          1. Prefer InProcess if it has local subscribers for the topic.
          2. Otherwise, pick the connected transport with the best score.

        Future phases will use the full probabilistic model:
          score = capability_match + latency_score + reliability_score
                  + trust_score - load_penalty
        """
        best: Transport | None = None
        best_score = -1.0

        for transport in self._transports.values():
            if transport.state not in (TransportState.CONNECTED, TransportState.DEGRADED):
                continue

            score = self._score_transport(transport, topic)

            if score > best_score:
                best_score = score
                best = transport

        return best

    def _score_transport(self, transport: Transport, topic: str) -> float:
        """
        Compute a routing score for a transport.

        Phase 1 scoring weights:
          - Local subscribers bonus: +100 (strong local preference)
          - Low latency bonus: up to +50 (inversely proportional to latency)
          - Reliability bonus: up to +30 (based on error rate)
          - Type bonus: inprocess=+20, webrtc=+15, redis=+10, websocket=+5
        """
        score = 0.0
        metrics = transport.get_metrics()

        # ── Local preference (Execution Authority) ──
        if transport.has_subscribers(topic):
            score += 100.0

        # ── Latency score ──
        # Lower latency = higher score. Cap at 50 points.
        if metrics.avg_latency_ms > 0:
            latency_score = max(0, 50.0 - metrics.avg_latency_ms)
        else:
            # No data yet — assume fast for local, moderate for others
            latency_score = 50.0 if transport.transport_type == "inprocess" else 25.0
        score += latency_score

        # ── Reliability score ──
        reliability = 1.0 - metrics.error_rate
        score += reliability * 30.0

        # ── Transport type bonus ──
        type_bonuses = {
            "inprocess": 20.0,
            "webrtc": 15.0,
            "redis": 10.0,
            "websocket": 5.0,
        }
        score += type_bonuses.get(transport.transport_type, 0.0)

        return score

    def _get_transport_score(self, transport_id: str) -> float:
        """Get the current routing score for a transport."""
        transport = self._transports.get(transport_id)
        if transport is None:
            return 0.0
        return self._score_transport(transport, "*")

    async def _fallback_send(
        self,
        topic: str,
        message: Message,
        ctx: ExecutionContext,
        exclude: str,
    ) -> None:
        """Attempt to send via an alternative transport."""
        for transport in self._transports.values():
            if transport.transport_id == exclude:
                continue
            if transport.state not in (TransportState.CONNECTED, TransportState.DEGRADED):
                continue
            try:
                await transport.send(topic, message)
                logger.info(
                    "RIL fallback succeeded via '%s' for topic '%s'",
                    transport.transport_id,
                    topic,
                )
                return
            except Exception as e:
                logger.error("RIL fallback '%s' also failed: %s", transport.transport_id, e)

        logger.error("RIL: all transports exhausted for topic '%s'. Message lost.", topic)
