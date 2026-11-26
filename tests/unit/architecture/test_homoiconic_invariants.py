"""Permanent Phase M6 architecture invariants and homoiconic VM gates (ADR 005)."""

from __future__ import annotations

import ast
import json
import time
from pathlib import Path
from typing import Any

import jwt
import numpy as np
import pytest
from fastapi.testclient import TestClient

from hbllm.drivers.base import DriverAction, DriverInput, DriverModality
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.hcir.abi import ExecutionMetrics
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import ConceptNode
from hbllm.hcir.interpreter import SyscallAllowlist, sys_execute
from hbllm.hcir.kernel.capability_resolver import CapabilityImplementation, ICapabilityExecutor
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.learning_loop import SkillCompiler
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.seed.loader import load_core_priors_spec
from hbllm.hcir.skills.induction import SkillInductionEngine
from hbllm.hcir.skills.library_learner import anti_unify_streams
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from hbllm.serving.api import _state, app
from hbllm.serving.cognitive_registry import CognitiveSystemRegistry

ROOT_DIR = Path(__file__).resolve().parents[3]
HCIR_DIR = ROOT_DIR / "hbllm" / "hcir"
WORLD_DIR = HCIR_DIR / "world"

_FORBIDDEN_VERBS = frozenset({"go", "reach", "navigate", "move", "push", "grab", "pull", "collect"})
_FORBIDDEN_COLORS = frozenset({"red", "blue", "green", "yellow"})
_FORBIDDEN_ONTOLOGY_LITERALS = frozenset(
    {"barrier", "walkable", "cargo", "receptacle", "key", "door"}
)


def _iter_py(base: Path) -> list[Path]:
    return [p for p in sorted(base.rglob("*.py")) if "__pycache__" not in p.parts]


def _extract_execute_capability_literal(call: ast.Call) -> str | None:
    """Return literal capability string if *call* constructs an `Opcode.EXECUTE` instruction."""
    func = call.func
    is_ins = isinstance(func, ast.Name) and func.id in ("Instruction", "_ins")
    if not is_ins:
        return None
    has_exec_op = False
    for arg in call.args:
        if "Opcode.EXECUTE" in ast.unparse(arg):
            has_exec_op = True
    for kw in call.keywords:
        if kw.arg == "opcode" and "Opcode.EXECUTE" in ast.unparse(kw.value):
            has_exec_op = True
    if not has_exec_op:
        return None
    for kw in call.keywords:
        if (
            kw.arg == "capability"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value
        if kw.arg == "params" and isinstance(kw.value, ast.Dict):
            for k, v in zip(kw.value.keys, kw.value.values, strict=False):
                if (
                    isinstance(k, ast.Constant)
                    and k.value == "capability"
                    and isinstance(v, ast.Constant)
                    and isinstance(v.value, str)
                ):
                    return v.value
    return None


def test_zero_hardcoded_verb_or_color_tables_in_compiler_and_world() -> None:
    """Compiler and world modules must contain zero hardcoded lexical verb/color tables."""
    targets = [HCIR_DIR / "compiler.py", *_iter_py(WORLD_DIR)]
    violations: list[str] = []
    for path in targets:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
                elts = node.keys if isinstance(node, ast.Dict) else node.elts
                words = {
                    e.value.strip().lower()
                    for e in elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                }
                if len(words & _FORBIDDEN_VERBS) >= 3:
                    violations.append(
                        f"{path.relative_to(ROOT_DIR)}:{node.lineno} verb table {sorted(words)}"
                    )
                if len(words & _FORBIDDEN_COLORS) >= 3:
                    violations.append(
                        f"{path.relative_to(ROOT_DIR)}:{node.lineno} color table {sorted(words)}"
                    )
    assert not violations, "Hardcoded lexical tables in compiler/world:\n" + "\n".join(violations)

    tkb_path = WORLD_DIR / "causal" / "transition_kb.py"
    tkb_tree = ast.parse(
        tkb_path.read_text(encoding="utf-8", errors="ignore"), filename=str(tkb_path)
    )
    ont_hits = [
        f"{node.lineno}:{node.value!r}"
        for node in ast.walk(tkb_tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.strip().lower() in _FORBIDDEN_ONTOLOGY_LITERALS
    ]
    assert not ont_hits, f"Forbidden ontology literals in transition_kb.py: {ont_hits}"

    spec = load_core_priors_spec()
    assert spec.get("skills") and spec.get("concepts")


@pytest.mark.asyncio
async def test_strict_syscall_allowlist_and_zero_escaped_execute_literals() -> None:
    """SyscallAllowlist defaults to strict=True and hcir contains zero escaped EXECUTE literals."""
    default_allow = SyscallAllowlist()
    assert default_allow.strict is True

    ws = HCIRWorkspaceState(tenant_id="tenant_sec")
    services = KernelServices.create_default(
        ws,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_sec")),
    )

    class _MockExec(ICapabilityExecutor):
        @property
        def is_available(self) -> bool:
            return True

        async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
            return {"executed": True}

    for cap in ("planning.decompose", "reasoning.compare", "world_prediction"):
        services.capability_resolver.register(
            CapabilityImplementation(
                capability_name=cap, implementation_id=f"mock_{cap}", executor=_MockExec()
            )
        )
        res = await sys_execute(
            Instruction(opcode=Opcode.EXECUTE, params={"capability": cap, "params": {}}),
            ws,
            services,
        )
        assert res.get("blocked_by_allowlist") is True, f"Capability {cap} was not blocked"

    bad_literals: list[str] = []
    for path in _iter_py(HCIR_DIR):
        src = path.read_text(encoding="utf-8", errors="ignore")
        if "Opcode.EXECUTE" not in src:
            continue
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                cap = _extract_execute_capability_literal(node)
                if cap is not None and default_allow.classify(cap) != "allowed":
                    bad_literals.append(f"{path.relative_to(ROOT_DIR)}:{node.lineno} -> {cap!r}")
    assert not bad_literals, "Non-allowlisted Opcode.EXECUTE literals:\n" + "\n".join(bad_literals)


def test_all_skill_producers_emit_executable_bytecode_streams() -> None:
    """Every homoiconic SkillNode producer in hcir passes a non-None executable bytecode_stream."""
    producers = (
        "hbllm/hcir/learning_loop.py",
        "hbllm/hcir/skills/declarative.py",
        "hbllm/hcir/skills/library_learner.py",
        "hbllm/hcir/reasoning/operators/skill_chunker.py",
        "hbllm/hcir/world/option_discovery.py",
        "hbllm/hcir/language/reading_curriculum.py",
        "hbllm/hcir/seed/loader.py",
        "hbllm/hcir/stdlib.py",
        "hbllm/hcir/world/cortical_programs.py",
    )
    missing_kw: list[str] = []
    for rel in producers:
        path = ROOT_DIR / rel
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
                )
                if func_name != "SkillNode":
                    continue
                bs_kw = next((kw for kw in node.keywords if kw.arg == "bytecode_stream"), None)
                if bs_kw is None or (
                    isinstance(bs_kw.value, ast.Constant) and bs_kw.value.value is None
                ):
                    missing_kw.append(f"{rel}:{node.lineno}")
    assert not missing_kw, f"SkillNode calls missing bytecode_stream: {missing_kw}"

    receipt = ExecutionReceipt(
        execution_id="rec_m6",
        process_id="p",
        thread_id="t",
        instruction_stream_hash="s",
        input_snapshot_version=0,
        final_snapshot_version=1,
        capabilities_used=["motor.interact"],
        metrics=ExecutionMetrics(),
        success=True,
    )
    compiled = SkillCompiler().compile_receipt_to_skill(receipt, skill_name="probe_skill")
    assert compiled.bytecode_stream is not None
    induced = SkillInductionEngine().induce_skill_node_from_trajectory([1, 2], skill_name="traj_m6")
    assert induced.bytecode_stream is not None

    def _make_trace(target: str) -> InstructionStream:
        return InstructionStream(
            author="w",
            instructions=[
                Instruction(
                    opcode=Opcode.UNIFY,
                    params={"pattern": {"role": "cargo", "id": target}, "bind": "$r0"},
                    cost_estimate=10,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={"capability": "motor.move", "params": {"target": target}},
                    cost_estimate=15,
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={"capability": "motor.interact", "params": {"target": target}},
                    cost_estimate=15,
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            ],
        )

    sa = _make_trace("box_1")
    sb = _make_trace("box_2")
    cand = anti_unify_streams(sa, sb, skill_name="move_macro", support_count=2)
    assert cand is not None and cand.to_skill_node().bytecode_stream is not None


def test_fake_predictors_absent_and_plugins_use_two_tier_sandbox() -> None:
    """Removed stub predictors stay deleted and terminal/compiler plugins enforce two-tier sandboxing."""
    for stub_name in ("llm.py", "neural.py", "statistical.py"):
        assert not (WORLD_DIR / "predictors" / stub_name).exists(), (
            f"Stub {stub_name} must be absent"
        )

    for rel in (
        "plugins/hbllm_terminal/__init__.py",
        "plugins/compiler-verification/compiler_verification.py",
    ):
        path = ROOT_DIR / rel
        src = path.read_text(encoding="utf-8", errors="ignore")
        assert "CapabilitySandboxManager" in src and "run_sandboxed_command" in src
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                "create_subprocess_exec",
                "create_subprocess_shell",
                "Popen",
                "system",
            ):
                raise AssertionError(f"Direct subprocess call disallowed in {rel}:{node.lineno}")


def test_serving_and_blackbox_unified_over_cognitive_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All chat/memory/rules HTTP+WS endpoints and CognitiveBlackbox run through CognitiveSystem."""
    secret = "m6_serving_unification_secret_32ch"
    monkeypatch.setenv("HBLLM_JWT_SECRET", secret)
    monkeypatch.setenv("HBLLM_ENV", "production")
    monkeypatch.delenv("HBLLM_UNSAFE_DEV_AUTH", raising=False)

    def _headers(tenant: str) -> dict[str, str]:
        tok = jwt.encode(
            {"tenant_id": tenant, "exp": int(time.time()) + 3600},
            secret,
            algorithm="HS256",
        )
        return {"Authorization": f"Bearer {tok}"}

    def _token(tenant: str) -> str:
        return jwt.encode(
            {"tenant_id": tenant, "exp": int(time.time()) + 3600},
            secret,
            algorithm="HS256",
        )

    registry = CognitiveSystemRegistry(base_memory_dir=tmp_path / "tenants")
    prev_reg = _state.get("cognitive_registry")
    prev_sys = _state.pop("cognitive_system", None)
    prev_brain = _state.pop("brain", None)
    _state["cognitive_registry"] = registry
    try:
        client = TestClient(app)

        # 1. POST /v1/chat -> Pure-HCIR direct assertion
        r1 = client.post(
            "/v1/chat",
            headers=_headers("tenant_m6_a"),
            json={"session_id": "sess_m6", "text": "The red ball is on the table."},
        )
        assert r1.status_code == 200
        b1 = r1.json()
        assert b1["source_node"] == "pure-hcir"
        assert b1["provider_used"] == "pure-hcir"

        # 2. POST /v1/chat/stream -> Pure-HCIR SSE stream
        r2 = client.post(
            "/v1/chat/stream",
            headers=_headers("tenant_m6_a"),
            json={"session_id": "sess_m6", "text": "The blue cube is in the box."},
        )
        assert r2.status_code == 200
        assert r2.headers.get("X-HBLLM-Engine") == "pure-hcir"
        sse_lines = [ln for ln in r2.text.splitlines() if ln.startswith("data: ")]
        assert any(json.loads(ln[6:]).get("done") is True for ln in sse_lines)

        # 3. POST /v1/chat/completions (non-streaming & streaming) -> Pure-HCIR recall
        r3 = client.post(
            "/v1/chat/completions",
            headers=_headers("tenant_m6_a"),
            json={"messages": [{"role": "user", "content": "Where is the red ball?"}]},
        )
        assert r3.status_code == 200
        assert "table" in r3.json()["choices"][0]["message"]["content"].lower()

        r4 = client.post(
            "/v1/chat/completions",
            headers=_headers("tenant_m6_a"),
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "Where is the blue cube?"}],
            },
        )
        assert r4.status_code == 200
        assert r4.headers.get("X-HBLLM-Engine") == "pure-hcir"
        assert "box" in r4.text.lower() and "[DONE]" in r4.text

        # 4. WS /v1/chat/ws -> Pure-HCIR streaming without brain.bus
        with client.websocket_connect(f"/v1/chat/ws?token={_token('tenant_m6_a')}") as ws_conn:
            ws_conn.send_text(
                json.dumps({"session_id": "sess_m6", "text": "Where is the blue cube?"})
            )
            chunks: list[str] = []
            while True:
                pkt = ws_conn.receive_json()
                chunks.append(pkt.get("token", ""))
                if pkt.get("done"):
                    assert pkt.get("engine") == "pure-hcir"
                    break
            assert "box" in "".join(chunks).lower()

        # 5. GET /v1/memory/{session_id} & GET /v1/rules serve tenant HCIR state without brain.bus
        rmem = client.get("/v1/memory/sess_m6", headers=_headers("tenant_m6_a"))
        assert rmem.status_code == 200
        assert len(rmem.json()["turns"]) >= 4

        rrules = client.get("/v1/rules", headers=_headers("tenant_m6_a"))
        assert rrules.status_code == 200
        assert rrules.json()["total"] >= 1

        # 6. Cross-tenant isolation: tenant_m6_b does not see tenant_m6_a's "red ball -> table"
        rb = client.post(
            "/v1/chat/completions",
            headers=_headers("tenant_m6_b"),
            json={"messages": [{"role": "user", "content": "Where is the red ball?"}]},
        )
        assert rb.status_code == 503

        # 7. CognitiveBlackbox delegates step() directly over build_cognitive_system()
        bb = CognitiveBlackbox(
            tenant_id="tenant_m6_a", memory_dir=tmp_path / "bb", enable_grounding=True
        )
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[1, 1] = 2
        grid[3, 3] = 9
        act = bb.step(
            DriverInput(raw_data=grid, modality=DriverModality.GRID_2D, source_id="grid_m6"),
            [
                DriverAction(action_id=1),
                DriverAction(action_id=2),
                DriverAction(action_id=3),
                DriverAction(action_id=4),
            ],
        )
        assert act.action_id in (1, 2, 3, 4)
        assert "grid_m6" in bb._cognitive_systems
    finally:
        registry.clear()
        if prev_reg is None:
            _state.pop("cognitive_registry", None)
        else:
            _state["cognitive_registry"] = prev_reg
        if prev_sys is not None:
            _state["cognitive_system"] = prev_sys
        if prev_brain is not None:
            _state["brain"] = prev_brain


def test_node_budget_consolidation_preserves_system_priors_and_tenant_scope(tmp_path: Path) -> None:
    """Node budget compaction skips SYSTEM/core_prior nodes and preserves tenant scope."""
    tenant = "tenant_budget_m6"
    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id=tenant,
            tenant_id=tenant,
            memory_dir=tmp_path / "budget_mem",
            enable_grounding=False,
        )
    )
    ws = system.tiered_workspace.brain
    assert ws.graph.get_node("prior_skill_count_entities") is not None

    prior_ids = [n.id for n in ws.graph.all_nodes()]
    low_ops: list[TransactionOperation] = []
    for idx in range(15):
        node = ConceptNode(
            id=f"ephemeral_concept_{idx}",
            label=f"ephemeral_{idx}",
            definition="Low-salience test concept",
            domain="test_compaction",
            properties={"activation_energy": 0.05},
            scope=Scope(tenant_id=tenant),
        )
        low_ops.append(
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id=node.id,
                node_data=node.model_dump(),
            )
        )
    grant = system.tiered_workspace._system_grant
    commit_res = ws.commit_operations(
        low_ops, author="core_priors_loader", grant=grant, tenant_id=tenant
    )
    assert commit_res.is_committed

    cap = len(prior_ids) + 4
    evicted = ws.consolidate_node_budget(max_nodes=cap, grant=grant)
    assert len(evicted) >= 12
    assert "prior_skill_count_entities" not in evicted
    assert ws.graph.get_node("prior_skill_count_entities") is not None
    assert ws.graph.node_count <= cap

    summary_node = ws.graph.get_node(f"consolidated_summary_{ws.tier.value}")
    assert summary_node is not None
    assert summary_node.scope.tenant_id == tenant
