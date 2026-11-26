"""Phase M2 Exit-Gate & Verification Tests: SymbolicExpr, math.* ALUs, ISA v2 stdlib, SQLite skill persistence, and .hcir core_priors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from hbllm.hcir.abi import ExecutionMetrics
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import GoalNode, PhysicalEntityNode, SkillNode
from hbllm.hcir.interpreter import HCIRInterpreter, SyscallAllowlist
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.learning_loop import SkillCompiler
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.seed.loader import load_core_priors_spec
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    AllOf,
    AnyOf,
    ColorDistributionPredicate,
    CutSetPredicate,
    DeclarativeNeuroSymbolicSkill,
    EntityBoundingBoxPredicate,
    EntityCountPredicate,
    ExprPredicate,
    GridDimensionPredicate,
    HomogeneousRegionPredicate,
    MetadataPredicate,
    Not,
    PearlCausalDoPredicate,
    PixelDensityPredicate,
    PredictiveConfidencePredicate,
    SkillEvaluationContext,
    SubgoalSequence,
    SymbolicExpr,
    SymbolicPredicate,
    SymbolicSubgoal,
    SymmetryPredicate,
)
from hbllm.hcir.skills.induction import SkillInductionEngine
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.stdlib import stdlib
from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.option_discovery import OptionDiscoveryEngine
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


class TestSymbolicExprAndMathALUs:
    def test_symbolic_expr_round_trip_equivalence(self) -> None:
        grid = np.zeros((20, 20), dtype=int)
        grid[2:18, 2:8] = 1
        grid[2:18, 12:18] = 1
        ctx = SkillEvaluationContext(
            grid=grid,
            available_actions=[1, 2, 4],
            metadata={
                "switch_state": "OPEN",
                "total_free_energy": 0.2,
                "calibrated_confidence": 0.9,
            },
        )
        expr_pred = AllOf(
            ActionAffordancePredicate(required={1, 4}, forbidden={5}),
            GridDimensionPredicate(exact_shape=(20, 20)),
            EntityBoundingBoxPredicate(min_width=4, min_height=10, min_count=2),
            EntityCountPredicate(min_count=2, max_count=2),
            ColorDistributionPredicate(present_colors={0, 1}, absent_colors={9}),
            CutSetPredicate(min_partitions=2),
            HomogeneousRegionPredicate(min_row_ratio=0.0, max_row_ratio=0.1),
            SymmetryPredicate(
                axis="vertical", min_entities=2, min_area=10, max_area=200, ignore_top_colors=1
            ),
            MetadataPredicate(key="switch_state", expected_value="OPEN"),
            PearlCausalDoPredicate(variable="switch_state", expected_value="OPEN"),
            PredictiveConfidencePredicate(max_free_energy=0.5, min_confidence=0.8),
            PixelDensityPredicate(min_count=10),
            AnyOf(Not(EntityCountPredicate(min_count=99)), MetadataPredicate("never", False)),
        )
        assert expr_pred.evaluate(ctx) is True
        raw_dict = expr_pred.to_dict()
        reloaded = SymbolicPredicate.from_dict(raw_dict)
        assert isinstance(reloaded, ExprPredicate)
        assert isinstance(reloaded.to_expr(), SymbolicExpr)
        assert reloaded.evaluate(ctx) is True

    @pytest.mark.asyncio
    async def test_math_alus_execute_under_strict_allowlist(self) -> None:
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        interp = HCIRInterpreter(ws, services, syscall_allowlist=SyscallAllowlist(strict=True))

        grid = np.zeros((10, 10), dtype=int)
        grid[1:5, 1:4] = 2
        grid[1:5, 6:9] = 2
        stream = InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "math.connected_components",
                        "params": {"grid": grid.tolist()},
                        "bind": "$r0",
                    },
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "math.topology_cuts",
                        "params": {"grid": grid.tolist()},
                        "bind": "$r1",
                    },
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "math.symmetry_score",
                        "params": {
                            "grid": grid.tolist(),
                            "min_area": 4,
                            "max_area": 30,
                            "ignore_top_colors": 1,
                        },
                        "bind": "$r2",
                    },
                ),
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={
                        "capability": "math.set_cardinality",
                        "params": {"items_a": [1, 2], "items_b": [1, 2, 3]},
                        "bind": "$r3",
                    },
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r1"]}),
            ]
        )
        res = await interp.execute(stream)
        assert res.success
        assert not [a for a in res.annotations if a.severity == "warning"]
        assert res.events[0]["registers"]["$r0"]["count"] == 2
        assert res.events[0]["registers"]["$r1"]["has_cut_set"] is True
        assert res.events[0]["registers"]["$r2"]["symmetric"] is True
        assert res.events[0]["registers"]["$r3"]["subset"] is True


class TestStdlibPureISAv2:
    @pytest.mark.asyncio
    async def test_all_stdlib_streams_run_under_strict_allowlist(self) -> None:
        ws = HCIRWorkspaceState()
        services = KernelServices.create_default(ws)
        ws.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=GoalNode(id="g_s1", description="solar turbine").model_dump(),
                ),
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=PhysicalEntityNode(
                        id="e_py", entity_name="Python runtime"
                    ).model_dump(),
                ),
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=PhysicalEntityNode(
                        id="e_rs", entity_name="Rust runtime"
                    ).model_dump(),
                ),
            ],
            author="test",
        )
        interp = HCIRInterpreter(ws, services, syscall_allowlist=SyscallAllowlist(strict=True))
        streams = [
            stdlib.memory.search("solar"),
            stdlib.memory.store(GoalNode(id="g_s2", description="stored via stdlib").model_dump()),
            stdlib.reasoning.compare("Python", "Rust"),
            stdlib.reasoning.evaluate("e_py"),
            stdlib.planning.decompose("Build solar turbine"),
            stdlib.planning.prioritize(["g_s1", "g_s2"]),
            stdlib.learning.extract_skill("ep_missing"),
            stdlib.learning.consolidate(),
            stdlib.simulation.hypothesize("Copper transfers heat quickly"),
            stdlib.simulation.counterfactual("What if valve closes"),
            stdlib.memory.forget("g_s2"),
        ]
        for s in streams:
            res = await interp.execute(s)
            assert res.success, f"{s.description} failed: {res.error}"
            assert not any("deprecated" in a.assertion.lower() for a in res.annotations), (
                f"{s.description} emitted deprecated capability warning: {res.annotations}"
            )


class TestPersistentHomoiconicSkillRoundTrip:
    @pytest.mark.asyncio
    async def test_induced_skills_persist_sqlite_and_call_in_fresh_vm(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m2_persistent.db"

        # 1. OptionDiscoveryEngine must never emit dynamic AutonomousOptionSkill subclass
        opt = OptionDiscoveryEngine().induce_option(
            bottleneck=(2, 4), subgoal_sequence=["NAVIGATE_TO"]
        )
        decl_opt = opt.to_declarative_skill()
        assert type(decl_opt) is DeclarativeNeuroSymbolicSkill
        assert decl_opt.__class__.__name__ == "DeclarativeNeuroSymbolicSkill"
        opt_skill_node = opt.to_skill_node()

        # 2. SkillCompiler compiles receipt to homoiconic SkillNode with bytecode_stream
        receipt = ExecutionReceipt(
            execution_id="rcpt_m2",
            author="agent",
            success=True,
            capabilities_used=["motor.interact"],
            metrics=ExecutionMetrics(tokens_consumed=25),
        )
        compiled_skill = SkillCompiler().compile_receipt_to_skill(
            receipt, skill_name="compiled_touch"
        )

        # 3. SkillInductionEngine produces parameterized SkillNode + DeclarativeNeuroSymbolicSkill conversion
        tx_trace = HCIRTransaction(
            author="trace",
            operations=[
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_id="n_nav",
                    node_data={
                        "action_intent": "navigate_to_portal",
                        "action_id": 1,
                        "role": "portal",
                    },
                )
            ],
        )
        induced_skill = SkillInductionEngine().induce_skill_node_from_transactions(
            [tx_trace], skill_name="induced_portal"
        )
        assert induced_skill is not None and induced_skill.bytecode_stream is not None

        custom_decl = DeclarativeNeuroSymbolicSkill()
        custom_decl.skill_name = "decl_custom"
        custom_decl.signature = ActionAffordancePredicate(required={1, 4})
        custom_decl.program = SubgoalSequence(
            SymbolicSubgoal(intent=SpatialActionIntent.ACTUATE, target_query={"role": "portal"})
        )
        custom_skill_node = custom_decl.to_skill_node()

        # Persist all SkillNodes to SQLiteEventStore on TieredWorkspace.persistent
        store1 = SQLiteEventStore(db_path)
        tw1 = TieredWorkspace(persistent_store=store1)
        services1 = KernelServices.create_default(tw1.working_state, tiered_workspace=tw1)
        assert services1.commit_authority_registry is not None
        tw1.persistent.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE, node_id=n.id, node_data=n.model_dump()
                )
                for n in (opt_skill_node, compiled_skill, induced_skill, custom_skill_node)
            ],
            author="m2_test",
            grant=services1.commit_authority_registry.issue("m2_test", {tw1.persistent.tier}),
        )
        store1.close()

        # Boot a fresh TieredWorkspace & strict HCIRInterpreter from SQLite alone
        store2 = SQLiteEventStore(db_path)
        tw2 = TieredWorkspace(persistent_store=store2)
        services2 = KernelServices.create_default(tw2.working_state, tiered_workspace=tw2)
        tw2.working_state.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_data=PhysicalEntityNode(
                        id="portal_1", entity_name="gate", properties={"role": "portal"}
                    ).model_dump(),
                )
            ],
            author="scene",
        )
        interp2 = HCIRInterpreter(
            tw2.working_state, services2, syscall_allowlist=SyscallAllowlist(strict=True)
        )

        # Reconstruct DeclarativeNeuroSymbolicSkill directly from persisted SkillNode
        loaded_skill = tw2.persistent.get_node(custom_skill_node.id)
        assert isinstance(loaded_skill, SkillNode)
        rehydrated_decl = DeclarativeNeuroSymbolicSkill.from_skill_node(loaded_skill)
        assert rehydrated_decl.can_handle(np.zeros((4, 4), dtype=int), [1, 4]) is True
        assert (
            rehydrated_decl.compile_to_instruction_stream(
                np.zeros((4, 4), dtype=int), [1, 4]
            ).length
            > 0
        )

        # Execute each persisted SkillNode via Opcode.CALL (by ID and by postcondition goal_pattern)
        call_stream = InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "skill_id": compiled_skill.id,
                        "args": {"target": "portal_1"},
                        "bind": "$r0",
                    },
                ),
                Instruction(
                    opcode=Opcode.CALL, params={"skill_id": induced_skill.id, "bind": "$r1"}
                ),
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "goal_pattern": {"intent": "reach_bottleneck", "target": [2, 4]},
                        "bind": "$r2",
                    },
                ),
                Instruction(
                    opcode=Opcode.CALL, params={"skill_id": custom_skill_node.id, "bind": "$r3"}
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            ]
        )
        res = await interp2.execute(call_stream)
        assert res.success, f"Fresh SQLite SkillNode CALL failed: {res.error} {res.annotations}"
        regs = res.events[0]["registers"]
        assert regs["$r0"] == "portal_1"
        assert regs["$r1"] == "portal_1"
        assert regs["$r2"] == [2, 4]
        assert regs["$r3"] == "portal_1"
        store2.close()


class TestCorePriorsGenesisSeed:
    def test_core_priors_domain_purity_rejects_prohibited_terms(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "bad_priors.yaml"
        bad_yaml.write_text(
            "concepts:\n  - id: bad\n    label: bad\n    definition: color 0 is lava and key opens door\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="priors.md"):
            load_core_priors_spec(bad_yaml)

    @pytest.mark.asyncio
    async def test_bootstrap_hydrates_core_priors_into_brain_and_calls_innate_skills(self) -> None:
        system = build_cognitive_system(
            CognitiveSystemConfig(
                world_id="m2_priors_boot", memory_dir=None, enable_grounding=False
            )
        )
        assert system.innate_priors_count >= 25
        assert system.hydrated_count == 0
        assert system.tiered_workspace.persistent.graph.node_count == 0

        skill_node = system.tiered_workspace.brain.get_node("prior_skill_count_entities")
        assert isinstance(skill_node, SkillNode) and skill_node.bytecode_stream is not None

        grid = np.zeros((8, 8), dtype=int)
        grid[1:3, 1:3] = 3
        grid[5:7, 5:7] = 4

        interp = HCIRInterpreter(
            system.workspace, system.services, syscall_allowlist=SyscallAllowlist(strict=True)
        )
        stream = InstructionStream(
            instructions=[
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "goal_pattern": {"intent": "count_entities", "target": grid.tolist()},
                        "bind": "$r0",
                    },
                ),
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "goal_pattern": {"intent": "detect_topology_cuts", "target": grid.tolist()},
                        "bind": "$r1",
                    },
                ),
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "goal_pattern": {"intent": "compare_cardinality", "items_a": [1, 2]},
                        "args": {"items_b": [1, 2, 3]},
                        "bind": "$r2",
                    },
                ),
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "goal_pattern": {"intent": "counterfactual_probe", "scenario": "test_do"},
                        "bind": "$r3",
                    },
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            ]
        )
        res = await interp.execute(stream)
        assert res.success, f"Core prior skill CALL failed: {res.error} {res.annotations}"
        regs: dict[str, Any] = res.events[0]["registers"]
        assert regs["$r0"]["count"] == 2
        assert regs["$r1"]["has_cut_set"] is True
        assert regs["$r2"]["subset"] is True
        assert regs["$r3"]["predicted"] is True
