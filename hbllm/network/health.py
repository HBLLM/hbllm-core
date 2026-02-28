"""
Health Monitor — periodic heartbeat and health checking for all nodes.

Runs as a background task that pings nodes, collects health reports,
and updates the service registry. Integrates with circuit breakers
to detect and isolate failing nodes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from hbllm.network.messages import HeartbeatPayload, Message, MessageType
from hbllm.network.node import HealthStatus, NodeHealth

if TYPE_CHECKING:
    from hbllm.network.bus import MessageBus
    from hbllm.network.circuit_breaker import CircuitBreakerRegistry
    from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class HealthMonitor:
    """
    Monitors health of all registered nodes via heartbeats.

    Periodically sends heartbeat requests and marks nodes as
    unhealthy if they don't respond within the timeout.
    """

    def __init__(
        self,
        registry: ServiceRegistry,
        circuit_breakers: CircuitBreakerRegistry,
        bus: MessageBus,
        check_interval: float = 10.0,
        heartbeat_timeout: float = 5.0,
    ):
        self._registry = registry
        self._circuit_breakers = circuit_breakers
        self._bus = bus
        self._check_interval = check_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._running = False
        self._monitor_task: asyncio.Task[None] | None = None
        self._on_node_unhealthy_callbacks: list[callable] = []

    def on_node_unhealthy(self, callback: callable) -> None:
        """Register a callback for when a node becomes unhealthy."""
        self._on_node_unhealthy_callbacks.append(callback)

    async def start(self) -> None:
        """Start the health monitoring loop."""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "HealthMonitor started (interval=%.1fs, timeout=%.1fs)",
            self._check_interval,
            self._heartbeat_timeout,
        )

    async def stop(self) -> None:
        """Stop the health monitor."""
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("HealthMonitor stopped")

    async def check_node(self, node_id: str) -> NodeHealth:
        """Check health of a specific node."""
        heartbeat_msg = Message(
            type=MessageType.HEARTBEAT,
            source_node_id="health_monitor",
            target_node_id=node_id,
            topic=f"node.{node_id}.heartbeat",
            payload=HeartbeatPayload(node_id="health_monitor").model_dump(),
        )

        start_time = time.monotonic()
        try:
            response = await self._bus.request(
                f"node.{node_id}.heartbeat",
                heartbeat_msg,
                timeout=self._heartbeat_timeout,
            )
            latency = (time.monotonic() - start_time) * 1000  # ms

            health = NodeHealth(
                node_id=node_id,
                status=HealthStatus.HEALTHY,
                last_heartbeat=time.monotonic(),
                latency_ms=latency,
                capabilities_available=response.payload.get("capabilities", []),
            )

            # Record success in circuit breaker
            breaker = self._circuit_breakers.get(node_id)
            breaker.record_success()

            return health

        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("Heartbeat timeout for node '%s'", node_id)

            health = NodeHealth(
                node_id=node_id,
                status=HealthStatus.UNHEALTHY,
                last_heartbeat=time.monotonic(),
                message=f"Heartbeat timeout ({self._heartbeat_timeout}s)",
            )

            # Record failure in circuit breaker
            breaker = self._circuit_breakers.get(node_id)
            breaker.record_failure()

            return health

        except Exception as e:
            logger.error("Health check error for node '%s': %s", node_id, e)

            health = NodeHealth(
                node_id=node_id,
                status=HealthStatus.UNHEALTHY,
                last_heartbeat=time.monotonic(),
                message=str(e),
            )

            breaker = self._circuit_breakers.get(node_id)
            breaker.record_failure()

            return health

    async def _monitor_loop(self) -> None:
        """Main monitoring loop — checks all nodes periodically."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval)

                all_health = await self._registry.get_all_health()

                for node_id, prev_health in all_health.items():
                    if not self._running:
                        break

                    new_health = await self.check_node(node_id)
                    await self._registry.update_health(new_health)

                    # Detect transitions to unhealthy
                    if (
                        prev_health.status != HealthStatus.UNHEALTHY
                        and new_health.status == HealthStatus.UNHEALTHY
                    ):
                        logger.warning("Node '%s' became UNHEALTHY", node_id)
                        for callback in self._on_node_unhealthy_callbacks:
                            try:
                                await callback(node_id, new_health)
                            except Exception:
                                logger.exception("Error in unhealthy callback")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in health monitor loop")
