"""Phase M0 verification tests: CoW fork isolation, transaction tenant_id, verified adapters, and wheel packaging."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest

from hbllm.hcir.graph import CognitiveGraph, HCIREdge, HCIREdgeType, PhysicalEntityNode
from hbllm.hcir.grounding.concepts.registry import GroundedConceptRegistry
from hbllm.hcir.language.core.gateway import HCIRGateway
from hbllm.hcir.language.core.grounding import GroundingResolver
from hbllm.hcir.language.core.semantic_frame import (
    EntityReference,
    FrameType,
    GroundedSemanticFrame,
    LanguageMetadata,
    SemanticFrame,
    ThematicRole,
)
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState

ROOT_DIR = Path(__file__).resolve().parents[3]


def test_fork_copy_on_write_isolates_node_and_edge_mutations() -> None:
    """Mutating nodes or edges on a forked CognitiveGraph never bleeds into the parent."""
    graph = CognitiveGraph(enable_native_substrate=False)
    n1 = PhysicalEntityNode(
        id="ent_door",
        entity_name="door",
        properties={"status": "closed", "nested": {"locked": True}},
        tags=["barrier"],
    )
    n2 = PhysicalEntityNode(
        id="ent_room",
        entity_name="room",
        properties={"kind": "space"},
    )
    graph.add_node(n1)
    graph.add_node(n2)
    edge = HCIREdge(
        id="edge_loc",
        edge_type=HCIREdgeType.LOCATED_IN,
        sources=["ent_door"],
        targets=["ent_room"],
        properties={"weight": 1},
    )
    graph.add_edge(edge)

    forked = graph.fork()
    fork_node = forked.get_node("ent_door")
    assert fork_node is not None
    fork_node.properties["status"] = "open"
    fork_node.properties["nested"]["locked"] = False
    forked.nodes["ent_door"].tags.append("mutated")

    fork_edge = forked.get_edge("edge_loc")
    assert fork_edge is not None
    fork_edge.properties["weight"] = 99

    parent_node = graph.get_node("ent_door")
    parent_edge = graph.get_edge("edge_loc")
    assert parent_node is not None and parent_edge is not None
    assert parent_node.properties["status"] == "closed"
    assert parent_node.properties["nested"]["locked"] is True
    assert parent_node.tags == ["barrier"]
    assert parent_edge.properties["weight"] == 1

    # Fork over 2,000 nodes remains O(1) (< 15ms)
    big = CognitiveGraph(enable_native_substrate=False)
    for i in range(2000):
        big.add_node(PhysicalEntityNode(id=f"n_{i}", entity_name=f"e_{i}"))
    gc.collect()
    t0 = time.perf_counter()
    big_fork = big.fork()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert big_fork.node_count == 2000
    assert elapsed_ms < 15.0, f"CognitiveGraph.fork() over 2k nodes took {elapsed_ms:.2f}ms"
    big_fork.clear()
    assert big_fork.node_count == 0
    assert big.node_count == 2000
    assert len(big.nodes_by_type(n1.node_type)) == 2000


def test_transaction_first_class_tenant_id_and_verifier_enforcement() -> None:
    """HCIRTransaction promotes tenant_id as a first-class synced field enforced by ConstitutionalVerifier."""
    tx_direct = HCIRTransaction(author="agent_a", tenant_id="tenant_a")
    assert tx_direct.tenant_id == "tenant_a"
    assert tx_direct.metadata["tenant_id"] == "tenant_a"

    tx_meta = HCIRTransaction(author="agent_a", metadata={"tenant_id": "tenant_a"})
    assert tx_meta.tenant_id == "tenant_a"

    with pytest.raises(ValueError, match="Conflicting transaction tenant_id"):
        HCIRTransaction(
            author="agent_a",
            tenant_id="tenant_a",
            metadata={"tenant_id": "tenant_b"},
        )

    ws = HCIRWorkspaceState(tenant_id="tenant_a")
    node_a = PhysicalEntityNode(id="ent_a", entity_name="alpha")
    node_a.scope.tenant_id = "tenant_a"
    rejected = ws.commit_operations(
        [TransactionOperation(op=TransactionOp.ADD_NODE, node=node_a)],
        author="agent_a",
        tenant_id="tenant_b",
    )
    assert rejected.is_rejected

    committed = ws.commit_operations(
        [TransactionOperation(op=TransactionOp.ADD_NODE, node=node_a)],
        author="agent_a",
        tenant_id="tenant_a",
    )
    assert committed.is_committed


def test_standalone_gateway_grounding_and_concept_registry_use_verified_transactions() -> None:
    """HCIRGateway, GroundingResolver, and GroundedConceptRegistry route all writes through TransactionManager."""
    graph = CognitiveGraph(enable_native_substrate=False)
    ent = PhysicalEntityNode(id="ent_box", entity_name="box", properties={"color": "blue"})
    graph.add_node(ent)

    gateway = HCIRGateway(graph)
    resolver = GroundingResolver(graph)
    registry = GroundedConceptRegistry(graph)

    assert gateway._workspace.transaction_manager.unverified is False
    assert resolver._workspace.transaction_manager.unverified is False
    assert registry._workspace.transaction_manager.unverified is False

    # Exercise GroundingResolver._auto_create_entity
    created = resolver._auto_create_entity(
        EntityReference(concept_name="cube", properties={"color": "red"})
    )
    assert graph.get_node(created.id) is not None
    assert len(resolver._workspace.transaction_manager.committed_log) >= 1

    # Exercise HCIRGateway.process_assertion (without mutating prior ent object in-place)
    frame = SemanticFrame(
        frame_type=FrameType.ASSERTION,
        predicate="has_property",
        arguments={
            ThematicRole.THEME: EntityReference(concept_name="box", properties={"mass": 5}),
        },
        metadata=LanguageMetadata(language="en", utterance_id="utt_m0"),
    )
    grounded = GroundedSemanticFrame(
        frame=frame,
        grounded_entities={ThematicRole.THEME: "ent_box"},
    )
    gateway.process_assertion(grounded)
    assert ent.properties.get("mass") is None  # original instance not mutated in-place
    updated_ent = graph.get_node("ent_box")
    assert updated_ent is not None and updated_ent.properties.get("mass") == 5
    assert len(gateway._workspace.transaction_manager.committed_log) >= 2

    # Exercise GroundedConceptRegistry.register & record_prediction
    cid = registry.register(
        feature_prototype={"color": "blue"},
        member_ids=["ent_box"],
        domain="spatial",
        formation_source="feature",
    )
    before_concept = graph.get_node(cid)
    assert before_concept is not None
    registry.record_prediction(cid, correct=True)
    after_concept = graph.get_node(cid)
    assert after_concept is not None and after_concept.prediction_count == 1
    assert before_concept.prediction_count == 0  # original snapshot not mutated in-place
    assert len(registry._workspace.transaction_manager.committed_log) >= 2


def test_clean_wheel_extract_imports_and_fastapi_app_boot(tmp_path: Path) -> None:
    """Built wheel installs cleanly outside repo with zero experiments/ dependency and boots FastAPI."""
    wheel_dir = tmp_path / "dist"
    site_dir = tmp_path / "site"
    wheel_dir.mkdir(parents=True)
    site_dir.mkdir(parents=True)

    build_script = (
        f"from setuptools import build_meta; print(build_meta.build_wheel({str(wheel_dir)!r}))"
    )
    build_res = subprocess.run(
        [sys.executable, "-c", build_script],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    assert build_res.returncode == 0, f"Wheel build failed:\n{build_res.stdout}\n{build_res.stderr}"

    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1, f"Expected 1 built wheel, got {wheels}"
    with zipfile.ZipFile(wheels[0], "r") as zf:
        zf.extractall(site_dir)

    probe_script = f"""
import importlib, os, pkgutil, sys
from pathlib import Path

site_dir = Path({str(site_dir)!r}).resolve()
repo_dir = Path({str(ROOT_DIR)!r}).resolve()
venv_dir = (repo_dir / ".venv").resolve()
sys.path = [str(site_dir)] + [
    p
    for p in sys.path
    if p
    and (
        Path(p).resolve().is_relative_to(venv_dir)
        or not Path(p).resolve().is_relative_to(repo_dir)
    )
]

import hbllm
hbllm_file = Path(hbllm.__file__).resolve()
assert hbllm_file.is_relative_to(site_dir), f"Loaded hbllm from {{hbllm_file}} instead of wheel!"

critical_modules = [
    "hbllm.hcir",
    "hbllm.hcir.kernel.subprocess_sandbox",
    "hbllm.hcir.kernel.capability_sandboxing",
    "hbllm.runtime.bootstrap",
    "hbllm.runtime.adapters.perception.audio_event_adapter",
    "hbllm.runtime.adapters.perception.speech_adapter",
    "hbllm.runtime.adapters.perception.vision_adapter",
    "hbllm.serving.api",
]
for mod_name in critical_modules:
    importlib.import_module(mod_name)

from hbllm.serving.api import app
assert app.title

# Walk every shipped submodule in the wheel and verify no hbllm.*/experiments.* import is missing
failures = []
for mod_info in pkgutil.walk_packages([str(site_dir / "hbllm")], prefix="hbllm."):
    try:
        importlib.import_module(mod_info.name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if missing_name.startswith("hbllm") or missing_name.startswith("experiments"):
            failures.append(f"{{mod_info.name}}: {{exc}}")
    except Exception:
        # Non-import runtime errors (e.g. optional hardware drivers) are ignored; missing hbllm modules are fatal
        pass

assert not failures, "Missing hbllm/experiments imports in wheel:\\n" + "\\n".join(failures)
print("WHEEL_CLEAN_BOOT_OK")
"""
    env = dict(os.environ)
    env["HBLLM_DISABLE_EXPERIMENTS_SHIM"] = "1"
    env.pop("PYTHONPATH", None)
    probe_res = subprocess.run(
        [sys.executable, "-c", probe_script],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe_res.returncode == 0, (
        f"Clean wheel probe failed:\nSTDOUT:\n{probe_res.stdout}\nSTDERR:\n{probe_res.stderr}"
    )
    assert "WHEEL_CLEAN_BOOT_OK" in probe_res.stdout
