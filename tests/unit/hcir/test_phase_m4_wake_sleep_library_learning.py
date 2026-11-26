"""Phase M4 tests: Wake-Sleep Anti-Unification, MDL Library Learning & Cross-Seed Isolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmarks.envs.grid_nav import GridNavEnvironment
from benchmarks.envs.symbolic_rules import SymbolicRulesEnvironment
from benchmarks.suites.runner import run_closed_loop_task
from hbllm.hcir.abi import ExecutionMetrics
from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.graph import ActionModality, ActionNode, HCIRNodeType, SkillNode
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.capability_resolver import CapabilityImplementation
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.learning_loop import LearningLoopEngine
from hbllm.hcir.nodes import PhysicalEntityNode
from hbllm.hcir.reasoning.operators.skill_chunker import SkillMacroChunker
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.skills.library_learner import (
    LibraryLearner,
    anti_unify_streams,
    compute_mdl_gain,
    rewrite_stream_with_library,
)
from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.episodic.models import EpisodicTransition, ProceduralMacro
from hbllm.hcir.world.episodic.replay import HippocampalReplayConsolidator
from hbllm.hcir.world.percept import EpisodeBoundary, EpisodeBoundaryKind
from hbllm.hcir.world.working_memory.memory import PrefrontalWorkingMemory
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def _wake_unlock_stream(
    entity_id: str, coord: tuple[int, int], target_role: str = "key"
) -> InstructionStream:
    return InstructionStream(
        author="agent_wake",
        description=f"Acquire {entity_id}",
        instructions=[
            Instruction(
                opcode=Opcode.UNIFY,
                params={"pattern": {"role": target_role, "id": entity_id}, "bind": "$r0"},
                cost_estimate=10,
            ),
            Instruction(
                opcode=Opcode.EXECUTE,
                params={
                    "capability": "motor.navigate",
                    "params": {"target": entity_id, "pos": list(coord)},
                },
                cost_estimate=15,
            ),
            Instruction(
                opcode=Opcode.EXECUTE,
                params={"capability": "motor.interact", "params": {"target": entity_id}},
                cost_estimate=15,
            ),
            Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
        ],
    )


class TestAntiUnificationAndMDLGate:
    """Verify Plotkin LGG variable binding consistency and MDL compression gate."""

    def test_plotkin_lgg_consistent_bindings_and_surface_stripping(self) -> None:
        sa = _wake_unlock_stream("key_gold_1", (2, 3))
        sb = _wake_unlock_stream("key_silver_9", (6, 4))
        cand = anti_unify_streams(sa, sb, skill_name="unlock_key_macro", support_count=2)
        assert cand is not None
        assert cand.mdl_gain > 0.0
        assert cand.compressed_cost < cand.raw_cost
        # Plotkin LGG must bind repeated concrete IDs (key_gold_1/key_silver_9) to the SAME ?var.
        body_targets = [
            ins.params.get("params", {}).get("target")
            for ins in cand.stream.instructions
            if ins.opcode == Opcode.EXECUTE
        ]
        assert len(set(body_targets)) == 1
        assert str(body_targets[0]).startswith("?")
        dumped = str(cand.stream.model_dump())
        assert "key_gold_1" not in dumped and "key_silver_9" not in dumped
        sk_node = cand.to_skill_node(tenant_id="tenant_alpha")
        assert isinstance(sk_node, SkillNode)
        assert sk_node.scope.tenant_id == "tenant_alpha"
        assert sk_node.bytecode_stream is not None

    def test_mdl_gate_rejects_single_step_trivial_stream(self) -> None:
        trivial = InstructionStream(
            author="w",
            instructions=[
                Instruction(
                    opcode=Opcode.EXECUTE,
                    params={"capability": "motor.navigate", "params": {"target": "x"}},
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            ],
        )
        assert anti_unify_streams(trivial, trivial) is None
        raw_c, comp_c, gain = compute_mdl_gain(
            trivial.instructions[:-1], param_count=1, support_count=2
        )
        assert gain < 0.0 and comp_c > raw_c


class TestWakeSleepConsolidationAndCallExecution:
    """Verify offline sleep consolidation persists homoiconic SkillNodes and compresses wake traces."""

    @pytest.mark.asyncio
    async def test_offline_sleep_persists_skill_and_executes_via_call(self, tmp_path: Path) -> None:
        system = build_cognitive_system(
            CognitiveSystemConfig(
                world_id="m4_sleep_test",
                memory_dir=tmp_path / "mem",
                tenant_id="tenant_m4",
                enable_grounding=False,
                load_core_priors=False,
            )
        )
        assert system.runtime is not None
        assert system.engine.library_learner is not None
        system.runtime._state.tenant_id = "tenant_m4"
        learner: LibraryLearner = system.engine.library_learner

        loop = LearningLoopEngine(
            system.workspace, system.services.transaction_manager, library_learner=learner
        )
        for idx, (eid, rc) in enumerate((("key_gold_1", (1, 2)), ("key_silver_2", (4, 5)))):
            receipt = ExecutionReceipt(
                execution_id=f"rcpt_{idx}",
                author="tenant_m4",
                success=True,
                capabilities_used=["motor.navigate", "motor.interact"],
                metrics=ExecutionMetrics(tokens_consumed=40),
            )
            tx = loop.process_and_learn(
                receipt, user_reward=1.0, source_stream=_wake_unlock_stream(eid, rc)
            )
            assert tx is not None and tx.is_committed

        await system.runtime.run_offline_consolidation()
        assert len(learner.promoted_skills) >= 1

        persisted_skills = system.tiered_workspace.persistent.graph.nodes_by_type(
            HCIRNodeType.SKILL
        )
        ws_macros = [
            n for n in persisted_skills if isinstance(n, SkillNode) and n.id.startswith("skill_ws_")
        ]
        assert len(ws_macros) >= 1
        assert all(
            n.scope.tenant_id == "tenant_m4" and n.bytecode_stream is not None for n in ws_macros
        )

        # Verify survival in a fresh SQLite-backed TieredWorkspace + tenant isolation
        reopened = TieredWorkspace(
            persistent_store=SQLiteEventStore(
                tmp_path / "mem" / "persistent.db", tenant_id="tenant_m4"
            ),
            tenant_id="tenant_m4",
        )
        assert any(
            n.id.startswith("skill_ws_") and n.scope.tenant_id == "tenant_m4"
            for n in reopened.persistent.graph.nodes_by_type(HCIRNodeType.SKILL)
        )
        # Cross-tenant write isolation: a different tenant cannot overwrite or commit into tenant_m4
        other_tenant = TieredWorkspace(
            persistent_store=SQLiteEventStore(
                tmp_path / "mem_other" / "persistent.db", tenant_id="tenant_other"
            ),
            tenant_id="tenant_other",
        )
        assert [
            n
            for n in other_tenant.persistent.graph.nodes_by_type(HCIRNodeType.SKILL)
            if n.id.startswith("skill_ws_")
        ] == []
        rejected = system.workspace.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=ws_macros[0].id,
                    node_data=ws_macros[0]
                    .model_copy(update={"scope": Scope(tenant_id="tenant_other")})
                    .model_dump(),
                )
            ],
            author="tenant_other",
            tenant_id="tenant_other",
        )
        assert rejected.is_rejected

        # Compress a novel wake trace into an Opcode.CALL and execute through HCIRInterpreter
        raw_novel = _wake_unlock_stream("key_emerald_7", (7, 3))
        compressed = rewrite_stream_with_library(raw_novel, ws_macros)
        assert any(ins.opcode == Opcode.CALL for ins in compressed.instructions)
        assert len(compressed.instructions) < len(raw_novel.instructions)

        key_node = PhysicalEntityNode(
            id="key_emerald_7",
            name="key emerald",
            entity_type="key",
            properties={"role": "key"},
            scope=Scope(tenant_id="tenant_m4"),
        )
        key_tx = system.workspace.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=key_node.id,
                    node_data=key_node.model_dump(),
                )
            ],
            author="test_setup",
            tenant_id="tenant_m4",
        )
        assert key_tx.is_committed
        executed_caps: list[str] = []

        class _TrackMotorExecutor:
            def __init__(self, cap_name: str) -> None:
                self._cap_name = cap_name

            @property
            def is_available(self) -> bool:
                return True

            async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
                executed_caps.append(self._cap_name)
                return {"ok": True, "capability": self._cap_name, "target": params.get("target")}

        for cap_name in ("motor.navigate", "motor.interact"):
            system.services.capability_resolver.register(
                CapabilityImplementation(
                    capability_name=cap_name,
                    implementation_id=f"test_{cap_name}",
                    executor=_TrackMotorExecutor(cap_name),
                    priority=100,
                )
            )
        interp = HCIRInterpreter(system.workspace, services=system.services)
        res = await interp.execute(compressed)
        assert res.success is True
        assert executed_caps == ["motor.navigate", "motor.interact"]


class TestCrossSeedIsolationAndConsolidators:
    """Verify EpisodeBoundary(same_level=False) sanitization and chunker/replay SkillNodes."""

    def test_cross_seed_boundary_strips_surface_ids_and_coords(self) -> None:
        ws = TieredWorkspace(tenant_id="tenant_iso")
        learner = LibraryLearner(tiered_workspace=ws, tenant_id="tenant_iso")
        learner.record_stream(_wake_unlock_stream("key_seed1", (8, 9)))
        learner.record_stream(_wake_unlock_stream("key_seed2", (3, 1)))
        learner.on_boundary(
            EpisodeBoundary(kind=EpisodeBoundaryKind.LEVEL_COMPLETE, same_level=False)
        )

        for rec in learner.wake_traces:
            assert rec.sanitized_for_transfer is True
            dumped = str(rec.stream.model_dump())
            assert "key_seed1" not in dumped and "key_seed2" not in dumped
            assert "[8, 9]" not in dumped and "[3, 1]" not in dumped

        wm = PrefrontalWorkingMemory()
        macro = ProceduralMacro(
            macro_id="m_surface",
            action_sequence=[1, 2],
            action_data_sequence=[None, None],
            waypoints=[type("WP", (), {"role": "goal", "feature": 99, "target_pos": (6, 6)})()],
        )
        wm.register_procedural_skill("m_surface", macro.to_declarative_skill())
        wm.sanitize_procedural_library_for_transfer()
        sg_query = wm.procedural_library["m_surface"].program.subgoals[0].target_query
        assert sg_query.get("role") == "goal"
        assert "pos" not in sg_query and "feature" not in sg_query

    def test_skill_chunker_and_replay_emit_constitutional_skill_nodes(self) -> None:
        ws = HCIRWorkspaceState(tenant_id="tenant_chunk")
        KernelServices.create_default(
            ws,
            verification_pipeline=create_default_pipeline(
                default_scope=Scope(tenant_id="tenant_chunk")
            ),
        )

        chunker = SkillMacroChunker(min_trace_occurrences=2)
        for suffix in ("gold", "silver"):
            a1 = ActionNode(
                id=f"n_{suffix}",
                intent=f"navigate_to(key_{suffix})",
                modality=ActionModality.LOCOMOTION,
                requirements=[f"visible(key_{suffix})"],
                produces=[f"adjacent(agent, key_{suffix})"],
            )
            a2 = ActionNode(
                id=f"p_{suffix}",
                intent=f"pickup(key_{suffix})",
                modality=ActionModality.MANIPULATION,
                requirements=[f"adjacent(agent, key_{suffix})"],
                produces=[f"holds(agent, key_{suffix})"],
            )
            chunker.record_trace(
                [a1, a2], goal_achieved=f"holds(agent, key_{suffix})", success=True
            )

        tx_ops = chunker.propose_skill_transactions(tenant_id="tenant_chunk")
        assert len(tx_ops) == 1
        sk_data = tx_ops[0].node_data
        assert sk_data["node_type"] == "skill" and sk_data["bytecode_stream"] is not None
        res = ws.commit_operations(tx_ops, author="skill_chunker", tenant_id="tenant_chunk")
        assert res.is_committed

        replay = HippocampalReplayConsolidator(min_motif_support=1)
        traj = [
            EpisodicTransition(step=0, avatar_pos=(0, 0), action_id=1),
            EpisodicTransition(step=1, avatar_pos=(0, 1), action_id=2, is_success=True),
            EpisodicTransition(step=2, avatar_pos=(0, 2), action_id=1),
            EpisodicTransition(step=3, avatar_pos=(0, 3), action_id=2, is_success=True),
        ]
        schemas = replay.distill_semantic_schemas([traj])
        replay_ops = replay.propose_consolidation_transactions(
            schemas[:1], tenant_id="tenant_chunk"
        )
        assert len(replay_ops) == 1
        assert replay_ops[0].node_data["node_type"] == "skill"
        assert replay_ops[0].node_data["bytecode_stream"] is not None
        assert ws.commit_operations(
            replay_ops, author="HippocampalReplayConsolidator", tenant_id="tenant_chunk"
        ).is_committed


class TestCrossSeedBenchmarkObservability:
    """Verify runner cross-seed consolidation reports non-zero skills_consolidated."""

    @pytest.mark.parametrize(
        "env_cls,task_id,seeds",
        [(GridNavEnvironment, "gn_m4", [10, 11]), (SymbolicRulesEnvironment, "sr_m4", [70, 71])],
    )
    def test_cross_seed_runner_reports_nonzero_skills_consolidated(
        self, env_cls: type, task_id: str, seeds: list[int]
    ) -> None:
        ok, steps, lats, errors, extra = run_closed_loop_task(
            env_cls,
            {"id": task_id, "seeds": seeds},
            max_budget=60,
            episodes=1,
            cross_seed=True,
        )
        assert errors == []
        assert extra["cross_seed_transfer"]["skills_consolidated"] > 0
