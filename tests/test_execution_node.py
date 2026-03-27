import pytest
import asyncio
from hbllm.network.messages import Message, MessageType
from hbllm.actions.execution_node import ExecutionNode

@pytest.fixture
def execution_node():
    return ExecutionNode(node_id="test_exec", timeout=5.0)

@pytest.mark.asyncio
async def test_execution_node_success(execution_node):
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "print('hello world'); print(2+2)"}
    )
    resp = await execution_node.handle_message(msg)
    assert resp is not None
    assert resp.payload["status"] == "SUCCESS"
    assert "hello world" in resp.payload["output"]
    assert "4" in resp.payload["output"]

@pytest.mark.asyncio
async def test_execution_node_syntax_error(execution_node):
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "print('missing paren"}
    )
    resp = await execution_node.handle_message(msg)
    assert resp.payload["status"] == "FAILURE"
    assert "SyntaxError" in resp.payload["error"]

@pytest.mark.asyncio
async def test_execution_node_timeout(execution_node):
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "import time\ntime.sleep(10)"}
    )
    resp = await execution_node.handle_message(msg)
    assert resp.payload["status"] == "FAILURE"
    assert "timed out" in resp.payload["output"]

@pytest.mark.asyncio
async def test_execution_node_extraction(execution_node):
    text = "Here is an approach:\n```python\nx=10\nprint(x*2)\n```\nHope it works!"
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"text": text}
    )
    resp = await execution_node.handle_message(msg)
    assert resp.payload["status"] == "SUCCESS"
    assert resp.payload["output"] == "20"
