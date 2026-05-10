"""
WebRTC Gateway — High-bandwidth perception channels over UDP.

Allows edge devices to establish WebRTC data channels with the central Core
for ultra-low latency streaming (e.g., live video frames for VisionNode).
"""

import asyncio
import logging
import orjson
from typing import Any

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel

from hbllm.network.bus import MessageBus
from hbllm.network.messages import Message, MessageType

logger = logging.getLogger(__name__)


class WebRTCGateway:
    """
    Manages WebRTC PeerConnections for edge devices.
    Bridges RTCDataChannels directly to the internal MessageBus.
    """

    def __init__(self, bus: MessageBus | None = None):
        self.bus = bus
        # Map: (tenant_id, user_id, device_id) -> RTCPeerConnection
        self.active_peers: dict[tuple[str, str, str], RTCPeerConnection] = {}

    async def handle_offer(
        self,
        tenant_id: str,
        user_id: str,
        device_id: str,
        sdp: str,
        sdp_type: str,
    ) -> dict[str, str]:
        """
        Process an SDP offer from an edge device and return an SDP answer.
        """
        key = (tenant_id, user_id, device_id)

        # Clean up existing connection if reconnecting
        if key in self.active_peers:
            old_pc = self.active_peers.pop(key)
            await old_pc.close()

        pc = RTCPeerConnection()
        self.active_peers[key] = pc

        @pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel) -> None:
            logger.info(f"RTCDataChannel established: {channel.label} for {device_id}")

            @channel.on("message")
            async def on_message(message: str | bytes) -> None:
                if not self.bus:
                    return

                try:
                    # We expect JSON messages over the data channel for perception frames
                    if isinstance(message, bytes):
                        data = orjson.loads(message)
                    else:
                        data = orjson.loads(message.encode("utf-8"))

                    topic = data.get("topic", f"perception.{channel.label}")
                    
                    msg = Message(
                        type=MessageType.EVENT,
                        source_node_id=f"webrtc_{device_id}",
                        tenant_id=tenant_id,
                        user_id=user_id,
                        device_id=device_id,
                        topic=topic,
                        payload=data.get("payload", {}),
                    )
                    await self.bus.publish(topic, msg)

                except Exception as e:
                    logger.error(f"Error handling WebRTC message from {device_id}: {e}")

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info(f"WebRTC connection state for {device_id} is {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                self.active_peers.pop(key, None)
                await pc.close()

        # Apply remote offer
        offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
        await pc.setRemoteDescription(offer)

        # Create answer
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)

        logger.info(f"WebRTC SDP Answer created for device {device_id}")
        return {
            "sdp": pc.localDescription.sdp,
            "type": pc.localDescription.type,
        }

    async def stop(self) -> None:
        """Close all active PeerConnections."""
        coros = [pc.close() for pc in self.active_peers.values()]
        await asyncio.gather(*coros, return_exceptions=True)
        self.active_peers.clear()
