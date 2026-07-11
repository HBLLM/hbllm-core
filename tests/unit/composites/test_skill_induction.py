"""Tests for SkillInductionNode — autonomous tool generation with AST security scanning."""

import pytest
import pytest_asyncio

from hbllm.brain.skills.skill_induction_node import SecurityInterceptor, SkillInductionNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

# ── SecurityInterceptor Tests ─────────────────────────────────────────────────


class TestSecurityInterceptor:
    """Test AST-based security validation for induced code."""

    def _check(self, code: str) -> list[str]:
        import ast

        interceptor = SecurityInterceptor()
        tree = ast.parse(code)
        interceptor.visit(tree)
        return interceptor.errors

    def test_safe_code_passes(self):
        """Clean utility code should pass security scan."""
        code = '''
def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """Calculate BMI."""
    return weight_kg / (height_m ** 2)
'''
        errors = self._check(code)
        assert errors == []

    def test_blocks_os_import(self):
        """Import os should be blocked."""
        errors = self._check("import os")
        assert len(errors) == 1
        assert "os" in errors[0]

    def test_blocks_subprocess_import(self):
        """Import subprocess should be blocked."""
        errors = self._check("import subprocess")
        assert len(errors) == 1
        assert "subprocess" in errors[0]

    def test_blocks_from_import(self):
        """From shutil import should be blocked."""
        errors = self._check("from shutil import rmtree")
        assert len(errors) == 1
        assert "shutil" in errors[0]

    def test_blocks_eval(self):
        """eval() calls should be blocked."""
        errors = self._check("result = eval('1+1')")
        assert len(errors) == 1
        assert "eval" in errors[0]

    def test_blocks_exec(self):
        """exec() calls should be blocked."""
        errors = self._check("exec('print(1)')")
        assert len(errors) == 1
        assert "exec" in errors[0]

    def test_blocks_open(self):
        """open() calls should be blocked."""
        errors = self._check("f = open('file.txt')")
        assert len(errors) == 1
        assert "open" in errors[0]

    def test_blocks_getattr(self):
        """getattr() calls should be blocked."""
        errors = self._check("x = getattr(obj, 'method')")
        assert len(errors) == 1
        assert "getattr" in errors[0]

    def test_multiple_violations(self):
        """Multiple violations should all be caught."""
        code = """
import os
import subprocess
result = eval("1+1")
"""
        errors = self._check(code)
        assert len(errors) == 3

    def test_safe_builtins_allowed(self):
        """Safe builtins like len, range, print should pass."""
        code = """
def process(items: list) -> int:
    for i in range(len(items)):
        print(items[i])
    return len(items)
"""
        errors = self._check(code)
        assert errors == []


# ── SkillInductionNode Integration Tests ──────────────────────────────────────


class MockLLM:
    """Mock LLM that returns a tool definition."""

    def __init__(self, response: dict):
        self._response = response

    async def generate_json(self, prompt: str) -> dict:
        return self._response


@pytest_asyncio.fixture
async def induction_env():
    bus = InProcessBus()
    await bus.start()
    yield bus
    await bus.stop()


@pytest.mark.asyncio
async def test_induction_safe_tool(induction_env):
    """Safe induced code should be accepted and published."""
    bus = induction_env
    llm = MockLLM(
        {
            "name": "calculate_area",
            "code": "def calculate_area(length: float, width: float) -> float:\n    return length * width",
            "description": "Calculate rectangle area",
        }
    )

    node = SkillInductionNode(node_id="inductor_safe", llm=llm)
    await node.start(bus)

    induced_events = []

    async def _collect(msg: Message) -> None:
        induced_events.append(msg)

    await bus.subscribe("system.skill.induced", _collect)

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.induction.request",
        payload={"gap": "Calculate the area of rectangles"},
    )
    response = await node.handle_message(request)

    assert response is not None
    assert response.payload["status"] == "SUCCESS"
    assert response.payload["skill_name"] == "calculate_area"

    await node.stop()


@pytest.mark.asyncio
async def test_induction_blocks_dangerous_code(induction_env):
    """Induced code with dangerous imports should be rejected."""
    bus = induction_env
    llm = MockLLM(
        {
            "name": "hacker_tool",
            "code": "import os\ndef hack(): os.system('rm -rf /')",
            "description": "Dangerous tool",
        }
    )

    node = SkillInductionNode(node_id="inductor_danger", llm=llm)
    await node.start(bus)

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.induction.request",
        payload={"gap": "Hack the system"},
    )
    response = await node.handle_message(request)

    assert response is not None
    assert "Security validation failed" in response.payload.get("error", "")

    await node.stop()


@pytest.mark.asyncio
async def test_induction_no_llm(induction_env):
    """Without an LLM, induction should return an error."""
    bus = induction_env
    node = SkillInductionNode(node_id="inductor_nollm", llm=None)
    await node.start(bus)

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.induction.request",
        payload={"gap": "Some gap"},
    )
    response = await node.handle_message(request)

    assert response is not None
    assert "No LLM available" in response.payload.get("error", "")

    await node.stop()


@pytest.mark.asyncio
async def test_induction_empty_gap(induction_env):
    """Empty gap description should return an error."""
    bus = induction_env
    node = SkillInductionNode(node_id="inductor_empty", llm=MockLLM({}))
    await node.start(bus)

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.induction.request",
        payload={"gap": ""},
    )
    response = await node.handle_message(request)

    assert response is not None
    assert "No gap description" in response.payload.get("error", "")

    await node.stop()


@pytest.mark.asyncio
async def test_induction_syntax_error(induction_env):
    """Induced code with syntax errors should be rejected."""
    bus = induction_env
    llm = MockLLM(
        {
            "name": "broken_tool",
            "code": "def broken(:\n    return",
            "description": "Broken tool",
        }
    )

    node = SkillInductionNode(node_id="inductor_syntax", llm=llm)
    await node.start(bus)

    request = Message(
        type=MessageType.QUERY,
        source_node_id="test",
        topic="system.induction.request",
        payload={"gap": "Some gap"},
    )
    response = await node.handle_message(request)

    assert response is not None
    assert "syntax errors" in response.payload.get("error", "")

    await node.stop()
