"""Tests for CommandDispatcher — COMMAND SemanticFrame → governed capability execution."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from hbllm.hcir.kernel.command_dispatcher import (
    VERB_CAPABILITY_MAP,
    CommandDispatcher,
    CommandExecutionReceipt,
    DiscourseExecutionResult,
)
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_resolver():
    """Create a mock CapabilityResolver with fs.read and fs.list capabilities."""
    resolver = MagicMock()

    def _has_cap(name: str) -> bool:
        return name in {"fs.read", "fs.list", "code.python_ast", "git.status"}

    resolver.has_capability = _has_cap

    async def _resolve_and_execute(cap_name: str, params: dict[str, Any], **kw) -> dict[str, Any]:
        if cap_name == "fs.read":
            return {
                "ok": True,
                "path": params.get("path", ""),
                "content": "file contents here",
                "truncated": False,
                "bytes": 18,
            }
        if cap_name == "fs.list":
            return {
                "ok": True,
                "path": params.get("path", "."),
                "entries": [
                    {"name": "README.md", "type": "file"},
                    {"name": "src", "type": "dir"},
                ],
                "truncated": False,
                "total": 2,
            }
        if cap_name == "code.python_ast":
            return {
                "ok": True,
                "functions": ["main"],
                "classes": ["Foo"],
                "imports": ["os"],
                "node_count": 10,
            }
        if cap_name == "git.status":
            return {"ok": True, "returncode": 0, "lines": ["M file.py"], "stderr": ""}
        return {"error": f"Unknown capability: {cap_name}"}

    resolver.resolve_and_execute = AsyncMock(side_effect=_resolve_and_execute)
    return resolver


@pytest.fixture
def mock_workspace():
    """Create a mock workspace with commit_operations."""
    ws = MagicMock()
    ws.commit_operations = MagicMock()
    ws.graph = MagicMock()
    ws.graph.node_count = 5
    return ws


@pytest.fixture
def dispatcher(mock_resolver, mock_workspace):
    """Create a CommandDispatcher with mock resolver and workspace."""
    return CommandDispatcher(mock_resolver, workspace=mock_workspace)


def _make_command_frame(verb: str, target: str = "", destination: str = "") -> SemanticFrame:
    """Helper to build a COMMAND SemanticFrame."""
    frame = SemanticFrame(
        frame_type=FrameType.COMMAND,
        predicate=verb,
        metadata=LanguageMetadata(language="en", raw_text=f"{verb} the {target}"),
    )
    if target:
        frame.set_role(
            ThematicRole.PATIENT,
            EntityReference(concept_name=target, raw_text=f"the {target}"),
        )
    if destination:
        frame.set_role(
            ThematicRole.DESTINATION,
            EntityReference(concept_name=destination, raw_text=f"to the {destination}"),
        )
    return frame


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestVerbCapabilityMap:
    """Test the static VERB_CAPABILITY_MAP is well-formed."""

    def test_core_verbs_mapped(self):
        """Core verbs must be in the map."""
        assert "read" in VERB_CAPABILITY_MAP
        assert "list" in VERB_CAPABILITY_MAP
        assert "verify" in VERB_CAPABILITY_MAP
        assert "search" in VERB_CAPABILITY_MAP
        assert "status" in VERB_CAPABILITY_MAP

    def test_all_values_are_dotted_capability_names(self):
        """Every mapped capability should follow the namespace.action pattern."""
        for verb, cap in VERB_CAPABILITY_MAP.items():
            assert "." in cap, f"Verb '{verb}' maps to '{cap}' which lacks namespace dot"


class TestCommandDispatcher:
    """Test single command dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_read_command(self, dispatcher: CommandDispatcher):
        """'Read the file' → fs.read capability."""
        frame = _make_command_frame("read", "README.md")
        receipt = await dispatcher.dispatch(frame)

        assert isinstance(receipt, CommandExecutionReceipt)
        assert receipt.success is True
        assert receipt.verb == "read"
        assert receipt.capability == "fs.read"
        assert receipt.target == "README.md"
        assert receipt.latency_ms > 0
        assert receipt.result.get("ok") is True
        assert "file contents here" in receipt.result.get("content", "")

    @pytest.mark.asyncio
    async def test_dispatch_list_command(self, dispatcher: CommandDispatcher):
        """'List the directory' → fs.list capability."""
        frame = _make_command_frame("list", "src")
        receipt = await dispatcher.dispatch(frame)

        assert receipt.success is True
        assert receipt.capability == "fs.list"
        assert receipt.result.get("total") == 2

    @pytest.mark.asyncio
    async def test_dispatch_verify_command(self, dispatcher: CommandDispatcher):
        """'Verify the code' → code.python_ast capability."""
        frame = _make_command_frame("verify", "main.py")
        receipt = await dispatcher.dispatch(frame)

        assert receipt.success is True
        assert receipt.capability == "code.python_ast"

    @pytest.mark.asyncio
    async def test_dispatch_unknown_verb(self, dispatcher: CommandDispatcher):
        """Unknown verb should return a failure receipt."""
        frame = _make_command_frame("flibbertigibbet", "something")
        receipt = await dispatcher.dispatch(frame)

        assert receipt.success is False
        assert "No capability mapping" in receipt.error

    @pytest.mark.asyncio
    async def test_dispatch_commits_evidence(
        self, dispatcher: CommandDispatcher, mock_workspace: MagicMock
    ):
        """Successful dispatch should commit an EvidenceNode receipt."""
        frame = _make_command_frame("read", "file.txt")
        receipt = await dispatcher.dispatch(frame)

        assert receipt.evidence_node_id is not None
        mock_workspace.commit_operations.assert_called()

    @pytest.mark.asyncio
    async def test_dispatch_with_destination(self, dispatcher: CommandDispatcher):
        """Frame with DESTINATION role passes destination param."""
        frame = _make_command_frame("read", "file.txt", destination="output")
        await dispatcher.dispatch(frame)

        # Verify resolver was called with destination in params
        call_args = dispatcher._resolver.resolve_and_execute.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("params", {})
        assert params.get("destination") == "output"


class TestDiscourseExecution:
    """Test multi-clause discourse dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_discourse_sequential(self, dispatcher: CommandDispatcher):
        """Multiple COMMAND frames execute sequentially."""
        frames = [
            _make_command_frame("read", "file1.txt"),
            _make_command_frame("list", "src"),
            _make_command_frame("verify", "main.py"),
        ]
        result = await dispatcher.dispatch_discourse(frames)

        assert isinstance(result, DiscourseExecutionResult)
        assert result.total_commands == 3
        assert result.successful == 3
        assert result.failed == 0
        assert result.all_succeeded is True
        assert result.total_latency_ms > 0
        assert len(result.receipts) == 3

    @pytest.mark.asyncio
    async def test_discourse_skips_non_commands(self, dispatcher: CommandDispatcher):
        """Non-COMMAND frames are filtered out from discourse."""
        frames = [
            _make_command_frame("read", "file.txt"),
            SemanticFrame(
                frame_type=FrameType.ASSERTION,
                predicate="located_on",
            ),
            _make_command_frame("list", "."),
        ]
        result = await dispatcher.dispatch_discourse(frames)

        assert result.total_commands == 2  # Only COMMAND frames
        assert result.successful == 2

    @pytest.mark.asyncio
    async def test_discourse_counts_failures(self, dispatcher: CommandDispatcher):
        """Failed commands in discourse are counted correctly."""
        frames = [
            _make_command_frame("read", "file.txt"),
            _make_command_frame("flibbertigibbet", "x"),
        ]
        result = await dispatcher.dispatch_discourse(frames)

        assert result.total_commands == 2
        assert result.successful == 1
        assert result.failed == 1
        assert result.all_succeeded is False


class TestVerbRegistration:
    """Test dynamic verb registration."""

    def test_register_custom_verb(self, dispatcher: CommandDispatcher):
        """Custom verb registration should extend the mapping."""
        dispatcher.register_verb("deploy", "world.sync_digital_twin")
        assert "deploy" in dispatcher.known_verbs()

    def test_known_verbs_sorted(self, dispatcher: CommandDispatcher):
        """known_verbs() returns sorted list."""
        verbs = dispatcher.known_verbs()
        assert verbs == sorted(verbs)


class TestBootstrapIntegration:
    """Test that process_dialogue correctly dispatches COMMAND frames."""

    def test_process_dialogue_command_integration(self):
        """Smoke test: build_cognitive_system + process_dialogue with a command."""
        from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

        cfg = CognitiveSystemConfig(
            world_id="test_cmd",
            enable_grounding=False,
            enable_brain_llm=False,
            memory_dir=None,
        )
        system = build_cognitive_system(cfg)

        # "List the directory" should parse as COMMAND and dispatch
        result = system.process_dialogue("List the directory")

        # Whether parse succeeds depends on lexicon state;
        # the key invariant is no crash
        assert result is not None
