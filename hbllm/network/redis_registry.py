"""
Redis-backed Service Registry — distributed node discovery.

Stores node registrations and health in Redis so all servers in the
cluster see the same nodes.  Falls back to in-process dict when Redis
is unavailable or in single-server mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from hbllm.network.node import HealthStatus, NodeHealth, NodeInfo, NodeType
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)

# Key prefixes in Redis
_PREFIX = "hbllm:registry"
_NODES_KEY = f"{_PREFIX}:nodes"        # Hash: node_id → JSON(NodeInfo)
_HEALTH_KEY = f"{_PREFIX}:health"      # Hash: node_id → JSON(NodeHealth)
_CHANNEL = f"{_PREFIX}:events"         # PubSub channel for join/leave events
_DEFAULT_TTL = 60                       # seconds before a node is auto-expired


class RedisRegistry(ServiceRegistry):
    """
    Distributed ServiceRegistry backed by Redis.

    Node registrations are stored as Redis hashes with TTL-based expiry.
    Nodes must periodically refresh their registration (via health updates)
    or be evicted automatically.

    Publishes join/leave events on a PubSub channel so other instances
    can react immediately to topology changes.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        ttl: int = _DEFAULT_TTL,
        health_check_interval: float = 10.0,
        node_timeout: float = 30.0,
    ):
        super().__init__(
            health_check_interval=health_check_interval,
            node_timeout=node_timeout,
        )
        self.redis_url = redis_url
        self.ttl = ttl
        self._redis = None
        self._listener_task: asyncio.Task | None = None

    async def start(self, bus=None) -> None:
        """Connect to Redis and start health check loop."""
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("RedisRegistry connected to %s", self.redis_url)
        except Exception as e:
            logger.warning(
                "RedisRegistry failed to connect to Redis (%s), "
                "falling back to in-process registry: %s",
                self.redis_url, e,
            )
            self._redis = None

        await super().start(bus)

    async def stop(self) -> None:
        """Disconnect from Redis."""
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._redis:
            await self._redis.aclose()
            self._redis = None

        await super().stop()

    async def register(self, node_info: NodeInfo) -> None:
        """Register a node — stores in Redis and local cache."""
        # Always update local cache
        await super().register(node_info)

        if self._redis:
            try:
                data = node_info.model_dump(mode="json")
                await self._redis.hset(_NODES_KEY, node_info.node_id, json.dumps(data))

                # Set per-node TTL key for auto-expiry
                ttl_key = f"{_PREFIX}:ttl:{node_info.node_id}"
                await self._redis.set(ttl_key, "1", ex=self.ttl)

                # Announce join
                await self._redis.publish(_CHANNEL, json.dumps({
                    "event": "join",
                    "node_id": node_info.node_id,
                    "node_type": node_info.node_type.value,
                    "capabilities": node_info.capabilities,
                }))
                logger.info("Registered '%s' in Redis", node_info.node_id)
            except Exception as e:
                logger.error("Failed to register '%s' in Redis: %s", node_info.node_id, e)

    async def deregister(self, node_id: str) -> None:
        """Remove a node from Redis and local cache."""
        await super().deregister(node_id)

        if self._redis:
            try:
                await self._redis.hdel(_NODES_KEY, node_id)
                await self._redis.hdel(_HEALTH_KEY, node_id)
                await self._redis.delete(f"{_PREFIX}:ttl:{node_id}")

                await self._redis.publish(_CHANNEL, json.dumps({
                    "event": "leave",
                    "node_id": node_id,
                }))
                logger.info("Deregistered '%s' from Redis", node_id)
            except Exception as e:
                logger.error("Failed to deregister '%s' from Redis: %s", node_id, e)

    async def update_health(self, health: NodeHealth) -> None:
        """Update health in both local cache and Redis, refresh TTL."""
        await super().update_health(health)

        if self._redis:
            try:
                data = health.model_dump(mode="json")
                await self._redis.hset(_HEALTH_KEY, health.node_id, json.dumps(data))

                # Refresh TTL
                ttl_key = f"{_PREFIX}:ttl:{health.node_id}"
                await self._redis.set(ttl_key, "1", ex=self.ttl)
            except Exception:
                logger.exception("Failed to update health in Redis for '%s'", health.node_id)

    async def discover(
        self,
        node_type: NodeType | None = None,
        capability: str | None = None,
        healthy_only: bool = True,
    ) -> list[NodeInfo]:
        """
        Discover nodes — prefers Redis for cross-server visibility,
        falls back to local cache.
        """
        if not self._redis:
            return await super().discover(node_type, capability, healthy_only)

        try:
            return await self._discover_from_redis(node_type, capability, healthy_only)
        except Exception as e:
            logger.warning("Redis discover failed, using local cache: %s", e)
            return await super().discover(node_type, capability, healthy_only)

    async def _discover_from_redis(
        self,
        node_type: NodeType | None,
        capability: str | None,
        healthy_only: bool,
    ) -> list[NodeInfo]:
        """Query Redis for matching nodes."""
        all_nodes_raw = await self._redis.hgetall(_NODES_KEY)
        all_health_raw = await self._redis.hgetall(_HEALTH_KEY)

        results = []
        for node_id, info_json in all_nodes_raw.items():
            info = NodeInfo.model_validate_json(info_json)

            # Filter by type
            if node_type and info.node_type != node_type:
                continue

            # Filter by capability
            if capability and capability not in info.capabilities:
                continue

            # Filter by health
            if healthy_only:
                health_json = all_health_raw.get(node_id)
                if health_json:
                    health = NodeHealth.model_validate_json(health_json)
                    if health.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN):
                        continue

            results.append(info)

        results.sort(key=lambda n: n.priority, reverse=True)
        return results

    async def sync_from_redis(self) -> int:
        """
        Pull all nodes from Redis into local cache.
        Returns the number of nodes synced.
        """
        if not self._redis:
            return 0

        try:
            all_nodes_raw = await self._redis.hgetall(_NODES_KEY)
            count = 0
            for node_id, info_json in all_nodes_raw.items():
                if node_id not in self._nodes:
                    info = NodeInfo.model_validate_json(info_json)
                    self._nodes[node_id] = info
                    self._health[node_id] = NodeHealth(
                        node_id=node_id,
                        status=HealthStatus.UNKNOWN,
                        last_heartbeat=time.monotonic(),
                    )
                    count += 1
            if count:
                logger.info("Synced %d new nodes from Redis", count)
            return count
        except Exception as e:
            logger.error("Failed to sync from Redis: %s", e)
            return 0
