"""Tests for governed kernel tool capabilities, native HCIR dialogue, and LLM scaffold isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from plugins.llm_scaffold import (
    DEFAULT_LOCAL_CHECKPOINT,
    AutonomyTracker,
    LLMTeacherScaffold,
)


@pytest.mark.asyncio
async def test_governed_tool_capabilities_registered_and_sandboxed(tmp_path: Path) -> None:
    sample = tmp_path / "hello.py"
    sample.write_text("import os\n\ndef greet(name: str) -> str:\n    return name\n")

    ws = HCIRWorkspaceState()
    services = KernelServices.create_default(ws)
    resolver = services.capability_resolver

    read_impl = await resolver.resolve("fs.read")
    assert read_impl is not None
    read_res = await read_impl.execute({"root_dir": str(tmp_path), "path": "hello.py"})
    assert read_res["ok"] is True
    assert "def greet" in read_res["content"]

    traversal_res = await read_impl.execute({"root_dir": str(tmp_path), "path": "../secret.txt"})
    assert traversal_res["ok"] is False
    assert "traversal" in traversal_res["error"].lower()

    list_impl = await resolver.resolve("fs.list")
    assert list_impl is not None
    list_res = await list_impl.execute({"root_dir": str(tmp_path), "path": "."})
    assert list_res["ok"] is True
    assert any(entry["name"] == "hello.py" for entry in list_res["entries"])

    ast_impl = await resolver.resolve("code.python_ast")
    assert ast_impl is not None
    ast_res = await ast_impl.execute({"root_dir": str(tmp_path), "path": "hello.py"})
    assert ast_res["ok"] is True
    assert "greet" in ast_res["functions"]
    assert "os" in ast_res["imports"]

    git_impl = await resolver.resolve("git.status")
    assert git_impl is not None


def test_cognitive_system_native_hcir_dialogue_commits_via_transactions() -> None:
    from hbllm.hcir.nodes import PhysicalEntityNode
    from hbllm.hcir.transactions import TransactionOp, TransactionOperation

    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_pure_hcir_dialogue",
            memory_dir=None,
            enable_grounding=False,
        )
    )
    ball = PhysicalEntityNode(
        id="ball_1", entity_name="ball", entity_type="ball", properties={"color": "red"}
    )
    table = PhysicalEntityNode(id="table_1", entity_name="table", entity_type="table")
    system.workspace.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=ball.id, node_data=ball.model_dump()
            ),
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=table.id, node_data=table.model_dump()
            ),
        ],
        author="test_setup",
    )

    res = system.process_dialogue("The red ball is on the table.")
    assert res.is_success is True
    assert res.hcir_node_id is not None
    assert system.workspace.get_node(res.hcir_node_id) is not None

    metrics = system.services.extensions["autonomy_metrics"]
    assert metrics["parse_attempts"] == 1
    assert metrics["parse_hcir_success"] == 1


def test_llm_teacher_scaffold_verifier_gated_and_self_retiring() -> None:
    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="test_scaffold_isolation",
            memory_dir=None,
            enable_grounding=False,
        )
    )
    tracker = AutonomyTracker(min_turns_for_retirement=3, retirement_threshold=0.98)

    def fake_teacher(utterance: str) -> dict[str, object]:
        if "unverified" in utterance:
            return {
                "response_text": "guess",
                "claim": "unverified guess",
                "confidence": 0.9,
                "verified": False,
            }
        return {
            "response_text": "learned",
            "claim": "keys unlock matching doors",
            "confidence": 0.9,
            "verified": True,
        }

    scaffold = LLMTeacherScaffold(tracker=tracker, teacher_fn=fake_teacher)
    assert scaffold.checkpoint_path == DEFAULT_LOCAL_CHECKPOINT

    r_bad = scaffold.advise_and_distill("unverified prompt", system.workspace)
    assert r_bad["distilled"] is False
    assert tracker.rejected_lessons == 1

    r_good = scaffold.advise_and_distill("how do doors open", system.workspace)
    assert r_good["distilled"] is True
    assert tracker.distilled_lessons == 1

    for _ in range(3):
        tracker.record_parse(True)
        tracker.record_realize(True)

    r_ret = scaffold.advise_and_distill("anything", system.workspace)
    assert r_ret["retired"] is True
    assert scaffold.retired is True
