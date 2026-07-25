"""
WebRTC Transport — Fast neural links for edge-to-edge P2P communication.

Wraps WebRTC data channels into the Transport abstraction for low-latency
peer-to-peer communication between edge devices on the same LAN.

This transport:
  - Establishes RTCPeerConnections via SDP offer/answer exchange.
  - Sends/receives HBLLM Messages over RTCDataChannels.
  - Reports latency metrics per-peer.

This transport does NOT:
  - Implement STUN/TURN for NAT traversal (LAN-only for Phase 3).
  - Route messages (that's the RIL's job).
  - Discover peers (that's mDNS's job — signaling comes from the caller).

The signaling channel (SDP exchange) is handled externally via the
WebSocket transport or mDNS properties. This transport only manages
established data channels.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from typing import Any

from hbllm.network.bus import MessageHandler, Subscription
from hbllm.network.messages import Message
from hbllm.network.transports.base import Transport, TransportState

logger = logging.getLogger(__name__)


class WebRTCPeer:
    """Represents an active WebRTC peer connection."""

    def __init__(
        self,
        node_id: str,
        pc: Any,  # RTCPeerConnection
        channel: Any | None = None,  # RTCDataChannel
    ) -> None:
        self.node_id = node_id
        self.pc = pc
        self.channel = channel
        self.connected_at = time.monotonic()
        self.last_message_at = 0.0
        self.messages_sent = 0
        self.messages_received = 0


class WebRTCTransport(Transport):
    """
    WebRTC-based transport for low-latency P2P edge communication.

    Manages multiple peer connections, each identified by node_id.
    Messages are sent as JSON over RTCDataChannels.

    Usage:
        transport = WebRTCTransport(transport_id="webrtc_mesh", local_node_id="car")
        await transport.start()

        # Initiate a connection to a peer (offer side)
        offer_sdp = await transport.create_offer("mobile")
        # ... send offer_sdp to peer via signaling channel ...
        # ... receive answer_sdp from peer ...
        await transport.handle_answer("mobile", answer_sdp)

        # Or accept a connection (answer side)
        answer_sdp = await transport.handle_offer("laptop", offer_sdp)
        # ... send answer_sdp back via signaling channel ...
    """

    def __init__(
        self,
        transport_id: str = "webrtc_mesh",
        local_node_id: str = "local",
        tenant_id: str = "default",
    ) -> None:
        super().__init__(transport_id=transport_id, transport_type="webrtc")
        self.local_node_id = local_node_id
        self.tenant_id = tenant_id

        # Active peer connections: node_id -> WebRTCPeer
        self._peers: dict[str, WebRTCPeer] = {}

        # Local subscriptions (mirroring the InProcess/Redis pattern)
        self._subscriptions: dict[str, list[Subscription]] = defaultdict(list)
        self._pending_requests: dict[str, asyncio.Future[Message]] = {}
        self._sub_counter = 0
        self._active_tasks: set[asyncio.Task[Any]] = set()

    async def start(self) -> None:
        """Start the WebRTC transport."""
        self._state = TransportState.CONNECTED
        self.metrics.uptime_start = time.monotonic()
        logger.info("WebRTCTransport '%s' started (node=%s)", self.transport_id, self.local_node_id)

    async def stop(self) -> None:
        """Stop all peer connections."""
        self._state = TransportState.STOPPED

        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()

        # Close all peer connections
        for peer in list(self._peers.values()):
            try:
                await peer.pc.close()
            except Exception as e:
                logger.debug("[Webrtc] non-critical error: %s", e)
        self._peers.clear()
        self._subscriptions.clear()

        logger.info("WebRTCTransport '%s' stopped", self.transport_id)

    # ── Signaling: Offer / Answer ─────────────────────────────────────

    async def create_offer(self, remote_node_id: str) -> dict[str, str]:
        """
        Create an SDP offer to initiate a P2P connection with a remote peer.

        Returns:
            {"sdp": "...", "type": "offer"}
        """
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: F401
        except ImportError:
            logger.error("aiortc not installed. WebRTC transport unavailable.")
            return {"sdp": "", "type": "error"}

        pc = RTCPeerConnection()
        channel = pc.createDataChannel("hbllm")

        peer = WebRTCPeer(node_id=remote_node_id, pc=pc, channel=channel)
        self._peers[remote_node_id] = peer

        # Set up channel event handlers
        self._setup_data_channel(peer, channel)

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            await self._handle_connection_state(peer)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        logger.info("WebRTC: created offer for peer '%s'", remote_node_id)
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def handle_answer(self, remote_node_id: str, answer: dict[str, str]) -> None:
        """Apply an SDP answer from a remote peer (completes the offer side)."""
        try:
            from aiortc import RTCSessionDescription
        except ImportError:
            return

        peer = self._peers.get(remote_node_id)
        if not peer:
            logger.warning("WebRTC: no pending offer for peer '%s'", remote_node_id)
            return

        desc = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        await peer.pc.setRemoteDescription(desc)
        logger.info("WebRTC: applied answer from peer '%s'", remote_node_id)

    async def handle_offer(self, remote_node_id: str, offer: dict[str, str]) -> dict[str, str]:
        """
        Handle an SDP offer from a remote peer and return an answer.

        Returns:
            {"sdp": "...", "type": "answer"}
        """
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
        except ImportError:
            return {"sdp": "", "type": "error"}

        # Clean up existing connection if reconnecting
        if remote_node_id in self._peers:
            old = self._peers.pop(remote_node_id)
            await old.pc.close()

        pc = RTCPeerConnection()
        peer = WebRTCPeer(node_id=remote_node_id, pc=pc)
        self._peers[remote_node_id] = peer

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            peer.channel = channel
            self._setup_data_channel(peer, channel)
            logger.info("WebRTC: datachannel '%s' from peer '%s'", channel.label, remote_node_id)

        @pc.on("connectionstatechange")
        async def on_state() -> None:
            await self._handle_connection_state(peer)

        desc = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        await pc.setRemoteDescription(desc)

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info("WebRTC: created answer for peer '%s'", remote_node_id)
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    # ── Message Sending ───────────────────────────────────────────────

    async def send(self, topic: str, message: Message) -> None:
        """Send a message to all connected peers via data channels."""
        wire = json.dumps(
            {
                "topic": topic,
                "message": message.model_dump(mode="json"),
            }
        )

        sent = False
        for peer in self._peers.values():
            if peer.channel and peer.channel.readyState == "open":
                try:
                    peer.channel.send(wire)
                    peer.messages_sent += 1
                    sent = True
                except Exception as e:
                    logger.error("WebRTC: send to '%s' failed: %s", peer.node_id, e)
                    self.metrics.record_error()

        if sent:
            self.metrics.record_send(len(wire))
        else:
            self.metrics.record_drop()

    async def send_to_peer(self, peer_node_id: str, topic: str, message: Message) -> None:
        """Send a message to a specific peer."""
        peer = self._peers.get(peer_node_id)
        if not peer or not peer.channel or peer.channel.readyState != "open":
            self.metrics.record_drop()
            return

        wire = json.dumps(
            {
                "topic": topic,
                "message": message.model_dump(mode="json"),
            }
        )
        try:
            peer.channel.send(wire)
            peer.messages_sent += 1
            self.metrics.record_send(len(wire))
        except Exception as e:
            self.metrics.record_error()
            logger.error("WebRTC: send to '%s' failed: %s", peer_node_id, e)

    async def send_request(self, topic: str, message: Message, timeout: float = 30.0) -> Message:
        """Send a request and wait for a correlated response."""
        future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()
        self._pending_requests[message.id] = future
        await self.send(topic, message)

        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except (TimeoutError, asyncio.TimeoutError):
            raise TimeoutError(
                f"WebRTC request {message.id} to '{topic}' timed out after {timeout}s"
            )
        finally:
            self._pending_requests.pop(message.id, None)

    # ── Subscriptions ─────────────────────────────────────────────────

    async def subscribe(
        self, topic: str, handler: MessageHandler, tenant_id: str | None = None
    ) -> Subscription:
        """Subscribe to messages received via WebRTC data channels."""
        self._sub_counter += 1
        sub = Subscription(
            topic=topic,
            handler=handler,
            sub_id=f"webrtc-sub-{self._sub_counter}",
            tenant_id=tenant_id,
        )
        self._subscriptions[topic].append(sub)
        return sub

    async def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription."""
        subscription.cancel()
        subs = self._subscriptions[subscription.topic]
        if subscription in subs:
            subs.remove(subscription)

    def has_subscribers(self, topic: str) -> bool:
        return bool(self._subscriptions.get(topic))

    # ── Internal ──────────────────────────────────────────────────────

    def _setup_data_channel(self, peer: WebRTCPeer, channel: Any) -> None:
        """Set up message handlers on a data channel."""

        @channel.on("message")
        async def on_message(raw: str | bytes) -> None:
            start = time.monotonic()
            try:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                data = json.loads(raw)
                topic = data.get("topic", "webrtc.unknown")
                msg_data = data.get("message", {})

                message = Message.model_validate(msg_data)
                peer.messages_received += 1
                self.metrics.record_receive(len(raw))

                # Check for pending request correlation
                if message.correlation_id and message.correlation_id in self._pending_requests:
                    future = self._pending_requests.pop(message.correlation_id)
                    if not future.done():
                        future.set_result(message)
                    return

                # Dispatch to local subscribers
                await self._dispatch_to_subscribers(topic, message)

                latency_ms = (time.monotonic() - start) * 1000
                self.metrics.record_latency(latency_ms)

            except json.JSONDecodeError:
                logger.warning("WebRTC: invalid JSON from peer '%s'", peer.node_id)
            except Exception as e:
                logger.error("WebRTC: error handling message from '%s': %s", peer.node_id, e)
                self.metrics.record_error()

    async def _dispatch_to_subscribers(self, topic: str, message: Message) -> None:
        """Dispatch to matching local subscribers."""
        matching = self._get_matching_topics(topic)

        for match_topic in matching:
            for sub in self._subscriptions.get(match_topic, []):
                if not sub.active:
                    continue
                if sub.tenant_id and message.tenant_id and sub.tenant_id != message.tenant_id:
                    continue

                async def _run(s: Subscription = sub, t: str = topic, m: Message = message) -> None:
                    try:
                        from hbllm.network._tenant_bridge import restore_tenant_ctx

                        with restore_tenant_ctx(m):
                            response = await s.handler(m)
                        if response is not None:
                            if response.correlation_id is None:
                                response.correlation_id = m.id
                            await self.send(response.topic, response)
                    except Exception:
                        self.metrics.record_error()
                        logger.exception("WebRTC handler error for '%s'", t)

                task = asyncio.create_task(_run())
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

    async def _handle_connection_state(self, peer: WebRTCPeer) -> None:
        """Handle connection state changes for a peer."""
        state = peer.pc.connectionState
        logger.info("WebRTC: peer '%s' state → %s", peer.node_id, state)
        if state in ("failed", "closed", "disconnected"):
            self._peers.pop(peer.node_id, None)
            try:
                await peer.pc.close()
            except Exception as e:
                logger.debug("[Webrtc] non-critical error: %s", e)

    def _get_matching_topics(self, topic: str) -> list[str]:
        """Match exact topics and wildcards."""
        matches = []
        for registered in self._subscriptions:
            if registered == topic:
                matches.append(registered)
            elif registered.endswith(".*"):
                prefix = registered[:-2]
                if topic.startswith(prefix):
                    matches.append(registered)
            elif registered == "*":
                matches.append(registered)
        return matches

    # ── Peer Info ─────────────────────────────────────────────────────

    @property
    def connected_peers(self) -> list[str]:
        """List of currently connected peer node IDs."""
        return [
            p.node_id for p in self._peers.values() if p.channel and p.channel.readyState == "open"
        ]

    def get_peer_stats(self) -> dict[str, dict[str, Any]]:
        """Get per-peer statistics."""
        return {
            p.node_id: {
                "connected_at": p.connected_at,
                "messages_sent": p.messages_sent,
                "messages_received": p.messages_received,
                "channel_state": p.channel.readyState if p.channel else "none",
            }
            for p in self._peers.values()
        }
