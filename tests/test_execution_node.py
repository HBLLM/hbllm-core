import pytest

from hbllm.actions.execution_node import ExecutionNode, validate_code
from hbllm.network.messages import Message, MessageType


@pytest.fixture
def execution_node():
    return ExecutionNode(node_id="test_exec", timeout=5.0)


@pytest.mark.asyncio
async def test_execution_node_success(execution_node):
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "print('hello world'); print(2+2)"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp is not None
    assert resp.payload["status"] == "SUCCESS"
    assert "hello world" in resp.payload["output"]
    assert "4" in resp.payload["output"]


@pytest.mark.asyncio
async def test_execution_node_syntax_error(execution_node):
    """Syntax errors are now caught by AST validation before execution."""
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "print('missing paren"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp is not None
    assert resp.type == MessageType.ERROR
    assert "Syntax error" in resp.payload["error"]


@pytest.mark.asyncio
async def test_execution_node_timeout(execution_node):
    # time module is not blocked — only dangerous system modules are
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "import time\ntime.sleep(10)"},
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
        payload={"text": text},
    )
    resp = await execution_node.handle_message(msg)
    assert resp.payload["status"] == "SUCCESS"
    assert resp.payload["output"] == "20"


# ── Sandbox security tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execution_node_blocks_os_import(execution_node):
    """AST validator should reject 'import os'."""
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "import os\nprint(os.listdir('.'))"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp.type == MessageType.ERROR
    assert "security policy" in resp.payload["error"]


@pytest.mark.asyncio
async def test_execution_node_blocks_subprocess(execution_node):
    """AST validator should reject 'import subprocess'."""
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "import subprocess\nsubprocess.run(['ls'])"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp.type == MessageType.ERROR
    assert "security policy" in resp.payload["error"]


@pytest.mark.asyncio
async def test_execution_node_blocks_eval(execution_node):
    """AST validator should reject calls to 'eval()'."""
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "result = eval('1+1')\nprint(result)"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp.type == MessageType.ERROR
    assert "security policy" in resp.payload["error"]


@pytest.mark.asyncio
async def test_execution_node_blocks_open(execution_node):
    """AST validator should reject calls to 'open()'."""
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "f = open('/etc/passwd')\nprint(f.read())"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp.type == MessageType.ERROR
    assert "security policy" in resp.payload["error"]


@pytest.mark.asyncio
async def test_execution_node_blocks_dunder(execution_node):
    """AST validator should reject dunder attribute access."""
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "x = ''.__class__.__subclasses__()"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp.type == MessageType.ERROR
    assert "security policy" in resp.payload["error"]


@pytest.mark.asyncio
async def test_execution_node_blocks_from_import(execution_node):
    """AST validator should reject 'from os import ...'."""
    msg = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="action.execute_code",
        payload={"code": "from os.path import join\nprint(join('a', 'b'))"},
    )
    resp = await execution_node.handle_message(msg)
    assert resp.type == MessageType.ERROR
    assert "security policy" in resp.payload["error"]


# ── validate_code unit tests ────────────────────────────────────────────────


def test_validate_code_safe():
    """Pure math code should pass validation."""
    violations = validate_code("x = 1 + 2\nprint(x)")
    assert violations == []


def test_validate_code_blocked_import():
    violations = validate_code("import sys")
    assert len(violations) == 1
    assert "sys" in violations[0]


def test_validate_code_multiple_violations():
    code = "import os\nimport subprocess\nresult = eval('1+1')"
    violations = validate_code(code)
    assert len(violations) == 3
