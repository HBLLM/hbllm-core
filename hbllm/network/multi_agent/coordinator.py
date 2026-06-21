"""Multi-Agent Coordinator — orchestrates task delegation across agents.

Manages a network of HBLLM agents, enabling:
    1. Agent discovery and capability registration
    2. Task delegation to the best-suited agent
    3. Load balancing across available agents
    4. Heartbeat monitoring and failure detection
    5. Result aggregation from delegated tasks

Architecture:
    - Each HBLLM instance registers with the coordinator
    - Tasks are routed based on required capabilities + current load
    - Failed delegations are retried or returned to the caller
    - Agents communicate via AgentMessage protocol

Usage::

    coordinator = MultiAgentCoordinator(identity=my_identity)
    coordinator.register_peer(other_agent_identity)
    result = await coordinator.delegate_task(task)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.network.multi_agent.protocol import (
    AgentIdentity,
    AgentMessage,
    AgentMessageType,
    DelegationTask,
)

logger = logging.getLogger(__name__)


class MultiAgentCoordinator:
    """Orchestrates task delegation across a network of AI agents.

    The coordinator maintains a registry of known agents and their
    capabilities, delegating tasks to the best-suited agent.
    """

    def __init__(
        self,
        identity: AgentIdentity | None = None,
        heartbeat_interval_s: float = 30.0,
        peer_timeout_s: float = 120.0,
    ) -> None:
        self.identity = identity or AgentIdentity()
        self.heartbeat_interval_s = heartbeat_interval_s
        self.peer_timeout_s = peer_timeout_s

        # Known peers
        self._peers: dict[str, AgentIdentity] = {}
        self._peer_last_seen: dict[str, float] = {}

        # Active delegations
        self._delegations: dict[str, DelegationTask] = {}
        self._delegation_futures: dict[str, asyncio.Future[dict[str, Any]]] = {}

        # Message handlers
        self._message_handlers: dict[AgentMessageType, Any] = {
            AgentMessageType.DISCOVER: self._handle_discover,
            AgentMessageType.DELEGATE: self._handle_delegate,
            AgentMessageType.RESULT: self._handle_result,
            AgentMessageType.HEARTBEAT: self._handle_heartbeat,
            AgentMessageType.CAPABILITY: self._handle_capability,
        }

        # Transport layer (pluggable)
        self._transport: Any = None

        # Telemetry
        self._tasks_delegated = 0
        self._tasks_received = 0
        self._tasks_completed = 0
        self._tasks_failed = 0

    def register_peer(self, peer: AgentIdentity) -> None:
        """Register a known peer agent."""
        self._peers[peer.agent_id] = peer
        self._peer_last_seen[peer.agent_id] = time.time()
        logger.info(
            "Registered peer: %s (%s) — capabilities: %s",
            peer.name,
            peer.agent_id[:8],
            peer.capabilities,
        )

    def remove_peer(self, agent_id: str) -> None:
        """Remove a peer from the registry."""
        self._peers.pop(agent_id, None)
        self._peer_last_seen.pop(agent_id, None)

    def find_capable_peers(
        self,
        required_capabilities: list[str],
        exclude_self: bool = True,
    ) -> list[AgentIdentity]:
        """Find peers that have all required capabilities.

        Returns agents sorted by load (least loaded first).
        """
        now = time.time()
        candidates: list[AgentIdentity] = []

        for agent_id, peer in self._peers.items():
            # Skip self
            if exclude_self and agent_id == self.identity.agent_id:
                continue

            # Check if peer is alive
            last_seen = self._peer_last_seen.get(agent_id, 0)
            if now - last_seen > self.peer_timeout_s:
                continue

            # Check capabilities
            if all(cap in peer.capabilities for cap in required_capabilities):
                candidates.append(peer)

        # Sort by load (least loaded first)
        candidates.sort(key=lambda p: p.load)
        return candidates

    async def delegate_task(
        self,
        task: DelegationTask,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Delegate a task to the best-suited peer.

        Finds a capable peer, sends the delegation message,
        and waits for the result.

        Args:
            task: The task to delegate.
            timeout_s: Maximum time to wait. Defaults to task.max_duration_s.

        Returns:
            The task result dict.

        Raises:
            RuntimeError: If no capable peer is found.
            asyncio.TimeoutError: If the task times out.
        """
        # Find a peer
        candidates = self.find_capable_peers(task.required_capabilities)
        if not candidates:
            raise RuntimeError(
                f"No capable peer found for capabilities: {task.required_capabilities}"
            )

        # Select the best candidate (least loaded)
        target = candidates[0]
        task.delegated_to = target.agent_id
        task.status = "delegated"

        self._delegations[task.task_id] = task
        self._tasks_delegated += 1

        # Create a future for the result
        loop = asyncio.get_event_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._delegation_futures[task.task_id] = future

        # Send delegation message
        msg = AgentMessage(
            message_type=AgentMessageType.DELEGATE,
            sender_id=self.identity.agent_id,
            recipient_id=target.agent_id,
            payload=task.to_dict(),
        )

        await self._send_message(msg)

        logger.info(
            "Delegated task '%s' to %s (%s)",
            task.description[:50],
            target.name,
            target.agent_id[:8],
        )

        # Wait for result
        timeout = timeout_s or task.max_duration_s
        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            task.status = "completed"
            task.result = result
            self._tasks_completed += 1
            return result
        except asyncio.TimeoutError:
            task.status = "failed"
            self._tasks_failed += 1
            self._delegation_futures.pop(task.task_id, None)
            raise

    async def handle_incoming(self, msg: AgentMessage) -> AgentMessage | None:
        """Process an incoming agent message."""
        if msg.is_expired:
            logger.debug("Dropping expired message from %s", msg.sender_id)
            return None

        handler = self._message_handlers.get(msg.message_type)
        if handler:
            return await handler(msg)

        logger.debug("No handler for message type: %s", msg.message_type)
        return None

    async def _handle_discover(self, msg: AgentMessage) -> AgentMessage | None:
        """Handle discovery broadcast — respond with our identity."""
        peer_data = msg.payload
        peer = AgentIdentity(
            agent_id=msg.sender_id,
            name=peer_data.get("name", "Unknown"),
            capabilities=peer_data.get("capabilities", []),
            endpoint=peer_data.get("endpoint", ""),
            load=peer_data.get("load", 0),
        )
        self.register_peer(peer)

        # Reply with our identity
        return msg.create_reply(
            AgentMessageType.DISCOVER,
            self.identity.to_dict(),
        )

    async def _handle_delegate(self, msg: AgentMessage) -> AgentMessage | None:
        """Handle a delegated task from another agent."""
        self._tasks_received += 1
        task_data = msg.payload

        logger.info(
            "Received delegated task: %s (from %s)",
            task_data.get("description", "")[:50],
            msg.sender_id[:8],
        )

        # Process the task (subclasses override this)
        try:
            result = await self._execute_delegated_task(task_data)
            return msg.create_reply(
                AgentMessageType.RESULT,
                {"task_id": task_data.get("task_id"), "status": "completed", "result": result},
            )
        except Exception as e:
            return msg.create_reply(
                AgentMessageType.ERROR,
                {"task_id": task_data.get("task_id"), "status": "failed", "error": str(e)},
            )

    async def _handle_result(self, msg: AgentMessage) -> AgentMessage | None:
        """Handle a result from a delegated task."""
        task_id = msg.payload.get("task_id", "")
        future = self._delegation_futures.pop(task_id, None)
        if future and not future.done():
            future.set_result(msg.payload.get("result", {}))
        return None

    async def _handle_heartbeat(self, msg: AgentMessage) -> AgentMessage | None:
        """Handle heartbeat — update peer last-seen time."""
        self._peer_last_seen[msg.sender_id] = time.time()
        load = msg.payload.get("load", 0)
        if msg.sender_id in self._peers:
            self._peers[msg.sender_id].load = load
        return None

    async def _handle_capability(self, msg: AgentMessage) -> AgentMessage | None:
        """Handle capability update from a peer."""
        if msg.sender_id in self._peers:
            self._peers[msg.sender_id].capabilities = msg.payload.get("capabilities", [])
            logger.info(
                "Peer %s updated capabilities: %s",
                msg.sender_id[:8],
                self._peers[msg.sender_id].capabilities,
            )
        return None

    async def _execute_delegated_task(self, task_data: dict[str, Any]) -> dict[str, Any]:
        """Execute a delegated task. Override in subclasses.

        Default implementation returns a stub result.
        """
        return {"status": "completed", "message": "Task processed by default handler"}

    async def _send_message(self, msg: AgentMessage) -> None:
        """Send a message via the transport layer."""
        if self._transport:
            await self._transport.send(msg)
        else:
            logger.debug(
                "No transport configured — message to %s dropped",
                msg.recipient_id[:8] if msg.recipient_id else "broadcast",
            )

    async def broadcast_discovery(self) -> None:
        """Broadcast our identity to discover peers."""
        msg = AgentMessage(
            message_type=AgentMessageType.DISCOVER,
            sender_id=self.identity.agent_id,
            payload=self.identity.to_dict(),
        )
        await self._send_message(msg)

    def get_peer_list(self) -> list[dict[str, Any]]:
        """Get list of known peers with status."""
        now = time.time()
        return [
            {
                **peer.to_dict(),
                "alive": now - self._peer_last_seen.get(peer.agent_id, 0) < self.peer_timeout_s,
                "last_seen_s_ago": now - self._peer_last_seen.get(peer.agent_id, 0),
            }
            for peer in self._peers.values()
        ]

    def stats(self) -> dict[str, Any]:
        """Coordinator statistics."""
        now = time.time()
        alive_peers = sum(
            1 for aid in self._peers if now - self._peer_last_seen.get(aid, 0) < self.peer_timeout_s
        )
        return {
            "agent_id": self.identity.agent_id[:8],
            "total_peers": len(self._peers),
            "alive_peers": alive_peers,
            "tasks_delegated": self._tasks_delegated,
            "tasks_received": self._tasks_received,
            "tasks_completed": self._tasks_completed,
            "tasks_failed": self._tasks_failed,
            "active_delegations": len(self._delegation_futures),
        }
