import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType, TaskAssignmentPayload
from hbllm.network.uplink_node import UplinkNode
from hbllm.serving.synapse_gateway import SynapseGateway


@pytest.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest.mark.asyncio
async def test_synapse_gateway_instruction_bridging(bus):
    """Test that SynapseGateway bridges TASK_ASSIGNMENT messages from bus to WebSocket."""
    gateway = SynapseGateway(bus=bus)
    await gateway.start()

    # Mock a connected device
    mock_ws = AsyncMock()
    tenant_id, user_id, device_id = "t1", "u1", "d1"
    await gateway.connect(mock_ws, tenant_id, user_id, device_id)

    # Publish an instruction to the bus targeted at this device
    task_payload = TaskAssignmentPayload(instruction="Analyze logs").model_dump()
    instruction_msg = Message(
        type=MessageType.TASK_ASSIGNMENT,
        source_node_id="planner",
        tenant_id=tenant_id,
        user_id=user_id,
        device_id=device_id,
        topic="edge.task_assignment",
        payload=task_payload,
    )

    # The gateway should be listening to "edge.task_assignment"
    await bus.publish("edge.task_assignment", instruction_msg)
    await asyncio.sleep(0.1) # Wait for async processing

    # Verify WebSocket received the bridged message
    assert mock_ws.send_text.called
    sent_data = json.loads(mock_ws.send_text.call_args[0][0])
    
    assert sent_data["type"] == "bridge_message"
    assert sent_data["msg_type"] == MessageType.TASK_ASSIGNMENT.value
    assert sent_data["payload"]["instruction"] == "Analyze logs"

    await gateway.stop()


@pytest.mark.asyncio
async def test_uplink_node_outbound_forwarding(bus):
    """Test that UplinkNode forwards local 'uplink.send' messages upstream."""
    mock_ws = AsyncMock()
    
    with patch("websockets.connect") as mock_connect:
        mock_ws.__aenter__.return_value = mock_ws
        mock_connect.return_value = mock_ws

        node = UplinkNode(
            node_id="edge_uplink",
            upstream_url="ws://hub.local",
            tenant_id="t1",
            user_id="u1",
            device_id="d1"
        )
        await node.start(bus)
        await asyncio.sleep(0.1)

        # Local node wants to send an instruction upstream
        local_msg = Message(
            type=MessageType.INSTRUCTION,
            source_node_id="local_manager",
            topic="hub.instruction",
            payload={"action": "report_status"}
        )
        await bus.publish("uplink.send", local_msg)
        await asyncio.sleep(0.1)

        # Verify it was sent via the WebSocket
        assert mock_ws.send.called
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "bridge_message"
        assert sent_data["topic"] == "hub.instruction"
        assert sent_data["payload"]["action"] == "report_status"

        await node.on_stop()


@pytest.mark.asyncio
async def test_uplink_node_inbound_bridging(bus):
    """Test that UplinkNode receives instructions from Hub and publishes to local bus."""
    mock_ws = AsyncMock()
    
    # Hub sends a bridge_message down to the edge
    hub_payload = {
        "type": "bridge_message",
        "msg_type": "task_assignment",
        "topic": "local.executor",
        "payload": {"task": "cleanup_cache"},
        "correlation_id": "hub_corr_1"
    }
    mock_ws.__aiter__.return_value = [json.dumps(hub_payload)]

    # Listen for the message on the local bus
    received_msgs = []
    async def catcher(msg):
        received_msgs.append(msg)

    await bus.subscribe("local.executor", catcher)

    with patch("websockets.connect") as mock_connect:
        mock_ws.__aenter__.return_value = mock_ws
        mock_connect.return_value = mock_ws

        node = UplinkNode(
            node_id="edge_uplink",
            upstream_url="ws://hub.local",
            tenant_id="t1",
            user_id="u1",
            device_id="d1"
        )
        await node.start(bus)
        await asyncio.sleep(0.2) # Wait for read loop

        # Verify local bus received the message
        assert len(received_msgs) == 1
        msg = received_msgs[0]
        assert msg.type == MessageType.TASK_ASSIGNMENT
        assert msg.payload["task"] == "cleanup_cache"
        assert msg.correlation_id == "hub_corr_1"
        assert msg.source_node_id == "hub"

        await node.on_stop()


@pytest.mark.asyncio
async def test_synapse_gateway_legacy_tool_compatibility(bus):
    """Ensure legacy tool_call format is still preserved for older clients."""
    gateway = SynapseGateway(bus=bus)
    await gateway.start()

    mock_ws = AsyncMock()
    await gateway.connect(mock_ws, "t1", "u1", "d1")

    # Legacy-style tool call
    tool_msg = Message(
        type=MessageType.QUERY,
        source_node_id="planner",
        tenant_id="t1",
        user_id="u1",
        device_id="d1",
        topic="edge.tool_call",
        payload={"tool_name": "web_search", "args": {"q": "HBLLM"}}
    )

    await bus.publish("edge.tool_call", tool_msg)
    await asyncio.sleep(0.1)

    assert mock_ws.send_text.called
    sent_data = json.loads(mock_ws.send_text.call_args[0][0])
    
    # Check that both new and legacy fields exist
    assert sent_data["type"] == "tool_call" # Legacy type
    assert sent_data["tool_name"] == "web_search"
    assert sent_data["args"]["q"] == "HBLLM"
    assert "msg_type" in sent_data # New bridge metadata also present

    await gateway.stop()
