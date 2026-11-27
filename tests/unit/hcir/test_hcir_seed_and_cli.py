"""Tests for Phase 6 (.hcir seed loader & canonical seeds) and Phase 7 (`hbllm hcir` CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hbllm.cli.main import app
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.compiler import HCIRCompiler
from hbllm.hcir.graph import SkillNode
from hbllm.hcir.interpreter import HCIRInterpreter, SyscallAllowlist
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.seed.loader import (
    CONSTRUCTION_HCIR_SEED_PATHS,
    CONSTRUCTION_YAML_SEED_PATHS,
    DEFAULT_CORE_PRIORS_HCIR_PATH,
    DEFAULT_CORE_PRIORS_YAML_PATH,
    build_construction_operations,
    build_core_prior_operations,
    hydrate_skills_seed,
    load_constructions_spec,
    load_core_priors_spec,
)
from hbllm.hcir.sexpr import format_hcir, parse_hcir, semantic_digest
from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope, SecurityLevel
from hbllm.hcir.workspace_tiers import TieredWorkspace

runner = CliRunner()


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — Dual .hcir / YAML Seed Loader & Canonical Seeds
# ═══════════════════════════════════════════════════════════════════════════


def test_canonical_seed_files_exist_and_match_yaml_hydration() -> None:
    assert DEFAULT_CORE_PRIORS_HCIR_PATH.exists()
    for lang in ("en", "si", "ta"):
        assert CONSTRUCTION_HCIR_SEED_PATHS[lang].exists()

    hcir_core = load_core_priors_spec(DEFAULT_CORE_PRIORS_HCIR_PATH)
    yaml_core = load_core_priors_spec(DEFAULT_CORE_PRIORS_YAML_PATH)

    tw_hcir = TieredWorkspace()
    tw_yaml = TieredWorkspace()
    svc_h = KernelServices.create_default(tw_hcir.working_state, tiered_workspace=tw_hcir)
    svc_y = KernelServices.create_default(tw_yaml.working_state, tiered_workspace=tw_yaml)
    assert (
        svc_h.commit_authority_registry is not None and svc_y.commit_authority_registry is not None
    )
    grant = svc_h.commit_authority_registry.issue("seed_test", {tw_hcir.brain.tier})
    grant_y = svc_y.commit_authority_registry.issue("seed_test", {tw_yaml.brain.tier})

    tw_hcir.brain.commit_operations(
        build_core_prior_operations(hcir_core), author="seed_test", grant=grant
    )
    tw_yaml.brain.commit_operations(
        build_core_prior_operations(yaml_core), author="seed_test", grant=grant_y
    )

    assert set(tw_hcir.brain.graph.nodes.keys()) == set(tw_yaml.brain.graph.nodes.keys())
    assert set(tw_hcir.brain.graph.edges.keys()) == set(tw_yaml.brain.graph.edges.keys())
    assert semantic_digest(tw_hcir.brain.graph) == semantic_digest(tw_yaml.brain.graph)

    for lang in ("en", "si", "ta"):
        hcir_cx = load_constructions_spec(lang, CONSTRUCTION_HCIR_SEED_PATHS[lang])
        yaml_cx = load_constructions_spec(lang, CONSTRUCTION_YAML_SEED_PATHS[lang])
        assert hcir_cx["lexical_rules"] == yaml_cx["lexical_rules"]
        assert [c["id"] for c in hcir_cx["constructions"]] == [
            c["id"] for c in yaml_cx["constructions"]
        ]

    # Hydrate constructions into both workspaces and verify compiler parity
    tw_hcir.brain.commit_operations(
        build_construction_operations(), author="seed_test", grant=grant
    )
    tw_yaml.brain.commit_operations(
        build_construction_operations(), author="seed_test", grant=grant_y
    )
    assert semantic_digest(tw_hcir.brain.graph) == semantic_digest(tw_yaml.brain.graph)

    comp_h = HCIRCompiler(tiered_workspace=tw_hcir, language="en")
    comp_y = HCIRCompiler(tiered_workspace=tw_yaml, language="en")
    s_h = comp_h.compile_text("What is the status?")
    s_y = comp_y.compile_text("What is the status?")
    assert [ins.opcode for ins in s_h.instructions] == [ins.opcode for ins in s_y.instructions]


def test_core_priors_hcir_domain_purity_enforced_and_allows_keywords(tmp_path: Path) -> None:
    impure = tmp_path / "core_priors_bad.hcir"
    impure.write_text(
        '(hcir :version 1 :type seed-bundle\n  (seed-bundle :kind core-priors\n    (concepts\n      (concept "c1" :label "bad" :definition "key opens door"))))',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="priors.md"):
        load_core_priors_spec(impure)


@pytest.mark.asyncio
async def test_hydrate_skills_seed_hcir_and_call_in_vm(tmp_path: Path) -> None:
    skill = SkillNode(
        id="skill_seed_card",
        skill_name="seed_card_check",
        description="Check set cardinality via Layer-0 ALU",
        parameters=[{"name": "items_a", "required": True}, {"name": "items_b", "required": True}],
        postcondition_pattern={"intent": "seed_card", "items_a": "?items_a"},
        bytecode_stream=InstructionStream(
            author="seed",
            instructions=[
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "math.set_cardinality",
                        "params": {"items_a": "$r0", "items_b": "$r1"},
                        "bind": "$r2",
                    },
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r2"]}),
            ],
        ).model_dump(),
        alpha_success=4.0,
        beta_failure=1.0,
    )
    skills_file = tmp_path / "skills.hcir"
    skills_file.write_text(
        format_hcir({"kind": "skills", "skills": [skill]}, mode="canonical"), encoding="utf-8"
    )

    tw = TieredWorkspace()
    services = KernelServices.create_default(tw.working_state, tiered_workspace=tw)
    assert services.commit_authority_registry is not None
    grant = services.commit_authority_registry.issue("skills_seed_loader", {tw.brain.tier})
    hydrated = hydrate_skills_seed(tw, skills_file, grant=grant)
    assert hydrated == 1

    interp = HCIRInterpreter(
        tw.working_state, services, syscall_allowlist=SyscallAllowlist(strict=True)
    )
    res = await interp.execute(
        InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "goal_pattern": {"intent": "seed_card", "items_a": [1, 2]},
                        "args": {"items_b": [1, 2, 3]},
                        "bind": "$r0",
                    },
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            ]
        )
    )
    assert res.success, res.error
    assert res.events[0]["registers"]["$r0"]["subset"] is True


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — `hbllm hcir` Inspection CLI
# ═══════════════════════════════════════════════════════════════════════════


def test_cli_hcir_inspect_and_print_from_booted_and_persistent_stores(tmp_path: Path) -> None:
    res = runner.invoke(
        app,
        ["hcir", "inspect", "skill", "prior_skill_geodesic_field", "--digest"],
    )
    assert res.exit_code == 0, res.output
    assert "; semantic_digest:" in res.output
    assert "; exact_digest:" in res.output
    body_lines = [ln for ln in res.output.splitlines() if not ln.startswith(";")]
    parsed = parse_hcir("\n".join(body_lines))
    assert isinstance(parsed, SkillNode)
    assert parsed.id == "prior_skill_geodesic_field"

    # Persist a custom skill in SQLite persistent.db and inspect via --memory-dir
    mem_dir = tmp_path / "world_mem"
    mem_dir.mkdir()
    store = SQLiteEventStore(mem_dir / "persistent.db", tenant_id="tenant_cli")
    tw = TieredWorkspace(persistent_store=store, tenant_id="tenant_cli")
    services = KernelServices.create_default(
        tw.working_state,
        tiered_workspace=tw,
        verification_pipeline=create_default_pipeline(default_scope=Scope(tenant_id="tenant_cli")),
    )
    assert services.commit_authority_registry is not None
    custom = SkillNode(
        id="skill_persisted_cli",
        skill_name="persisted_cli_skill",
        description="Persisted custom skill",
        bytecode_stream=InstructionStream(
            instructions=[Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]})]
        ).model_dump(),
        scope=Scope(tenant_id="tenant_cli", security_level=SecurityLevel.TENANT),
    )
    commit_res = tw.persistent.commit_operations(
        [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE, node_id=custom.id, node_data=custom.model_dump()
            )
        ],
        author="cli_test",
        grant=services.commit_authority_registry.issue("cli_test", {tw.persistent.tier}),
        tenant_id="tenant_cli",
    )
    assert commit_res.is_committed
    store.close()

    res_persisted = runner.invoke(
        app,
        [
            "hcir",
            "inspect",
            "skill",
            "persisted_cli_skill",
            "--tenant-id",
            "tenant_cli",
            "--memory-dir",
            str(mem_dir),
        ],
    )
    assert res_persisted.exit_code == 0, res_persisted.output
    parsed_custom = parse_hcir(res_persisted.output)
    assert isinstance(parsed_custom, SkillNode)
    assert parsed_custom.id == "skill_persisted_cli"

    # Print trace from node and from JSON receipt
    res_trace = runner.invoke(app, ["hcir", "print", "trace", "prior_skill_collision_cone"])
    assert res_trace.exit_code == 0, res_trace.output
    stream = parse_hcir(res_trace.output)
    assert isinstance(stream, InstructionStream)
    assert stream.instructions[0].params["capability"] == "math.collision_cone"

    rcpt_path = tmp_path / "receipt.json"
    rcpt_path.write_text(
        json.dumps({"execution_id": "rcpt_1", "capabilities_used": ["motor.interact"]}),
        encoding="utf-8",
    )
    res_rcpt = runner.invoke(app, ["hcir", "print", "trace", str(rcpt_path)])
    assert res_rcpt.exit_code == 0, res_rcpt.output
    assert isinstance(parse_hcir(res_rcpt.output), InstructionStream)


def test_cli_hcir_validate_and_roundtrip(tmp_path: Path) -> None:
    res_val = runner.invoke(app, ["hcir", "validate", str(DEFAULT_CORE_PRIORS_HCIR_PATH)])
    assert res_val.exit_code == 0, res_val.output
    assert "VALID:" in res_val.output

    bad_file = tmp_path / "bad.hcir"
    bad_file.write_text("(if (empty? $r0) (return))", encoding="utf-8")
    res_bad = runner.invoke(app, ["hcir", "validate", str(bad_file)])
    assert res_bad.exit_code == 1

    res_rt = runner.invoke(app, ["hcir", "roundtrip", str(DEFAULT_CORE_PRIORS_HCIR_PATH)])
    assert res_rt.exit_code == 0, res_rt.output

    messy = tmp_path / "messy.hcir"
    messy.write_text('; comment\n(stream   :author "x" (return   $r0   $r1))\n', encoding="utf-8")
    res_inplace = runner.invoke(app, ["hcir", "roundtrip", str(messy), "--in-place"])
    assert res_inplace.exit_code == 0, res_inplace.output
    assert messy.read_text(encoding="utf-8") == format_hcir(
        InstructionStream(
            author="x",
            instructions=[Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0", "$r1"]})],
        ),
        mode="canonical",
    )
