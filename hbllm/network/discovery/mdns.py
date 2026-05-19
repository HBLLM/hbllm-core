"""
mDNS Discovery — Local peer discovery via ZeroConf/mDNS.

This module broadcasts and discovers HBLLM nodes on the local network
(LAN / WiFi). It answers the question:

    "Who else is running HBLLM on this network?"

mDNS is DISCOVERY ONLY. It does not:
  - Synchronize state (that's Gossip's job).
  - Route messages (that's the RIL's job).
  - Understand capabilities (that's the Registry's job).

It simply announces presence and discovers peers, then feeds them into
the NodeState engine and CapabilityRegistry.

Service format:
    _hbllm._tcp.local.
    Properties: node_id, role, device_tier, capabilities (comma-separated),
                api_port, ws_port
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hbllm.network.discovery.registry import CapabilityRegistry
    from hbllm.network.node_state import NodeStateEngine

# mDNS service type for HBLLM instances
HBLLM_SERVICE_TYPE = "_hbllm._tcp.local."
HBLLM_SERVICE_NAME_PREFIX = "hbllm-"


class DiscoveredPeer:
    """Represents a peer discovered via mDNS."""

    def __init__(
        self,
        node_id: str,
        host: str,
        port: int,
        role: str = "edge",
        device_tier: str = "server",
        capabilities: list[str] | None = None,
        ws_port: int | None = None,
        properties: dict[str, str] | None = None,
    ) -> None:
        self.node_id = node_id
        self.host = host
        self.port = port
        self.role = role
        self.device_tier = device_tier
        self.capabilities = capabilities or []
        self.ws_port = ws_port
        self.properties = properties or {}
        self.discovered_at = time.monotonic()
        self.last_seen = time.monotonic()

    def __repr__(self) -> str:
        return (
            f"<DiscoveredPeer {self.node_id} at {self.host}:{self.port} "
            f"role={self.role} caps={len(self.capabilities)}>"
        )


class MDNSDiscovery:
    """
    Broadcasts and discovers HBLLM nodes on the local network via mDNS/ZeroConf.

    Usage:
        discovery = MDNSDiscovery(
            node_id="homeserver",
            role="coordinator",
            capabilities=["llm_inference", "semantic_search"],
            api_port=8000,
        )
        discovery.on_peer_found = my_callback
        await discovery.start()
        # ... later ...
        await discovery.stop()
    """

    def __init__(
        self,
        node_id: str,
        role: str = "edge",
        device_tier: str = "server",
        capabilities: list[str] | None = None,
        api_port: int = 8000,
        ws_port: int = 9833,
        broadcast_interval: float = 30.0,
        peer_timeout: float = 90.0,
    ) -> None:
        self.node_id = node_id
        self.role = role
        self.device_tier = device_tier
        self.capabilities = capabilities or []
        self.api_port = api_port
        self.ws_port = ws_port
        self.broadcast_interval = broadcast_interval
        self.peer_timeout = peer_timeout

        # Known peers
        self._peers: dict[str, DiscoveredPeer] = {}

        # Callbacks
        self.on_peer_found: Any | None = None   # async fn(DiscoveredPeer)
        self.on_peer_lost: Any | None = None    # async fn(str node_id)

        # ZeroConf internals
        self._zeroconf: Any | None = None
        self._service_info: Any | None = None
        self._browser: Any | None = None
        self._running = False
        self._prune_task: asyncio.Task[None] | None = None

    @property
    def peers(self) -> dict[str, DiscoveredPeer]:
        """Currently known peers."""
        return dict(self._peers)

    async def start(self) -> None:
        """Start mDNS broadcasting and listening."""
        try:
            from zeroconf import IPVersion, ServiceBrowser, ServiceInfo, Zeroconf
            from zeroconf.asyncio import AsyncZeroconf
        except ImportError:
            logger.warning(
                "zeroconf package not installed. mDNS discovery disabled. "
                "Install with: pip install zeroconf"
            )
            return

        self._running = True

        # Get local IP
        local_ip = self._get_local_ip()

        # Create service properties
        properties = {
            "node_id": self.node_id,
            "role": self.role,
            "device_tier": self.device_tier,
            "capabilities": ",".join(self.capabilities),
            "ws_port": str(self.ws_port),
        }

        # Register our service
        service_name = f"{HBLLM_SERVICE_NAME_PREFIX}{self.node_id}.{HBLLM_SERVICE_TYPE}"
        self._service_info = ServiceInfo(
            HBLLM_SERVICE_TYPE,
            service_name,
            addresses=[socket.inet_aton(local_ip)],
            port=self.api_port,
            properties=properties,
            server=f"{self.node_id}.local.",
        )

        self._zeroconf = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await self._zeroconf.async_register_service(self._service_info)

        logger.info(
            "mDNS: broadcasting '%s' on %s:%d (capabilities: %s)",
            self.node_id, local_ip, self.api_port,
            ", ".join(self.capabilities) or "none",
        )

        # Browse for other HBLLM services
        self._browser = ServiceBrowser(
            self._zeroconf.zeroconf,
            HBLLM_SERVICE_TYPE,
            handlers=[self._on_service_state_change],
        )

        # Start peer pruning loop
        self._prune_task = asyncio.create_task(self._prune_loop())

    async def stop(self) -> None:
        """Stop mDNS broadcasting and listening."""
        self._running = False

        if self._prune_task and not self._prune_task.done():
            self._prune_task.cancel()
            try:
                await self._prune_task
            except asyncio.CancelledError:
                pass

        if self._browser:
            self._browser.cancel()
            self._browser = None

        if self._zeroconf and self._service_info:
            await self._zeroconf.async_unregister_service(self._service_info)
            await self._zeroconf.async_close()

        self._peers.clear()
        logger.info("mDNS: discovery stopped for '%s'", self.node_id)

    def _on_service_state_change(
        self, zeroconf: Any, service_type: str, name: str, state_change: Any
    ) -> None:
        """Callback from ZeroConf when services are added/removed."""
        from zeroconf import ServiceStateChange

        if state_change == ServiceStateChange.Added:
            asyncio.ensure_future(self._handle_service_added(zeroconf, service_type, name))
        elif state_change == ServiceStateChange.Removed:
            asyncio.ensure_future(self._handle_service_removed(name))

    async def _handle_service_added(
        self, zeroconf: Any, service_type: str, name: str
    ) -> None:
        """Process a newly discovered service."""
        from zeroconf import ServiceInfo

        info = ServiceInfo(service_type, name)
        info.request(zeroconf, 3000)  # Timeout 3s

        if not info.addresses:
            return

        # Decode properties
        props: dict[str, str] = {}
        if info.properties:
            for k, v in info.properties.items():
                key = k.decode("utf-8") if isinstance(k, bytes) else str(k)
                val = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                props[key] = val

        node_id = props.get("node_id", "")
        if not node_id or node_id == self.node_id:
            return  # Ignore self

        host = socket.inet_ntoa(info.addresses[0])
        capabilities_str = props.get("capabilities", "")
        capabilities = [c.strip() for c in capabilities_str.split(",") if c.strip()]

        peer = DiscoveredPeer(
            node_id=node_id,
            host=host,
            port=info.port or 0,
            role=props.get("role", "edge"),
            device_tier=props.get("device_tier", "server"),
            capabilities=capabilities,
            ws_port=int(props.get("ws_port", "0")) or None,
            properties=props,
        )

        is_new = node_id not in self._peers
        self._peers[node_id] = peer

        if is_new:
            logger.info(
                "mDNS: discovered peer '%s' at %s:%d (role=%s, caps=%s)",
                node_id, host, peer.port, peer.role,
                ", ".join(capabilities) or "none",
            )
            if self.on_peer_found:
                try:
                    await self.on_peer_found(peer)
                except Exception as e:
                    logger.error("mDNS on_peer_found callback error: %s", e)
        else:
            # Refresh last_seen
            peer.last_seen = time.monotonic()

    async def _handle_service_removed(self, name: str) -> None:
        """Handle a service being removed from the network."""
        # Extract node_id from service name
        for node_id, peer in list(self._peers.items()):
            if node_id in name:
                self._peers.pop(node_id, None)
                logger.info("mDNS: peer '%s' left the network", node_id)
                if self.on_peer_lost:
                    try:
                        await self.on_peer_lost(node_id)
                    except Exception as e:
                        logger.error("mDNS on_peer_lost callback error: %s", e)
                break

    async def _prune_loop(self) -> None:
        """Periodically remove peers that haven't been seen recently."""
        while self._running:
            try:
                await asyncio.sleep(self.peer_timeout / 2)
                now = time.monotonic()
                stale = [
                    nid for nid, peer in self._peers.items()
                    if (now - peer.last_seen) > self.peer_timeout
                ]
                for nid in stale:
                    self._peers.pop(nid, None)
                    logger.info("mDNS: peer '%s' timed out (stale)", nid)
                    if self.on_peer_lost:
                        try:
                            await self.on_peer_lost(nid)
                        except Exception:
                            pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in mDNS prune loop")

    @staticmethod
    def _get_local_ip() -> str:
        """Get the local network IP address (not loopback)."""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
