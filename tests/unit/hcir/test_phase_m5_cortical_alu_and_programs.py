"""Phase M5 tests: Layer-0 Cortical ALUs, Dirichlet/Beta Posteriors, & Homoiconic World Programs."""

from __future__ import annotations

import time
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.capabilities.cortical_alus import (
    compute_bayesian_posterior_update,
    compute_bellman_risk_backup,
    compute_collision_cone,
    compute_geodesic_field,
)
from hbllm.hcir.graph import ConceptNode, HCIRNodeType, SkillNode
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.reasoning.operators.registry import create_default_operator_registry
from hbllm.hcir.reasoning.unified_runtime import UnifiedReasoningRuntime
from hbllm.hcir.seed.loader import hydrate_core_priors
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.hcir.world.causal.theory import HCIRSymbolicWorldTheory
from hbllm.hcir.world.causal.transition_kb import (
    DirichletOutcomePosterior,
    TransitionKnowledgeBase,
    TransitionOutcome,
)
from hbllm.hcir.world.cortical_pipeline import CorticalPipeline
from hbllm.hcir.world.cortical_programs import (
    build_ba10_subgoal_mediation_program,
    build_curiosity_probe_program,
    build_default_cortical_skill_nodes,
    build_deliberation_safety_gating_program,
    build_epistemic_phase_arbitration_program,
)
from hbllm.hcir.world.percept import EmbodiedObservation, EpistemicPhase
from hbllm.hcir.world.state.context import CognitiveContext
from hbllm.hcir.world.state.models import FrozenSymbolicTheory
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


class TestLayer0CorticalALUs:
    """Verify stateless math.* ALUs both directly and via HCIRInterpreter + core_priors."""

    def test_geodesic_bellman_collision_cone_and_bayesian_alus(self) -> None:
        grid = np.zeros((5, 5), dtype=np.int32)
        impassable = [[1, 1], [2, 1], [3, 1]]
        geo = compute_geodesic_field(
            grid,
            [[2, 4]],
            impassable_cells=impassable,
            start_cell=[2, 0],
        )
        assert geo["start_distance"] > 4.0
        assert geo["best_step"] in ([-1, 0], [1, 0])

        det = compute_bellman_risk_backup(
            (5, 5), [[0, 2]], lethal_cells=[[1, 1]], transition_noise=0.0, start_cell=[2, 1]
        )
        stoch = compute_bellman_risk_backup(
            (5, 5), [[0, 2]], lethal_cells=[[1, 1]], transition_noise=0.3, start_cell=[2, 1]
        )
        assert stoch["start_value"] < det["start_value"]
        assert stoch["best_delta"] is not None

        cone_hit = compute_collision_cone(
            (3, 5),
            [[1, 1], [1, 2], [1, 3]],
            [{"pos": [1, 3], "facing": [0, -1]}],
        )
        assert cone_hit["is_safe"] is False
        assert cone_hit["first_collision_step"] == 0
        assert cone_hit["recommended_wait_steps"] == 1

        cone_clear = compute_collision_cone(
            (3, 5),
            [[0, 1], [0, 2]],
            [{"pos": [2, 3], "facing": [0, -1]}],
        )
        assert cone_clear["is_safe"] is True

        post = compute_bayesian_posterior_update(
            observations=["pass", "pass", "lethal", "goal"], risk_weight=50.0
        )
        assert post["alpha_pass"] == pytest.approx(2.5)
        assert post["gamma_lethal"] == pytest.approx(1.05)
        assert post["delta_goal"] == pytest.approx(1.05)
        assert 0.0 < post["p_pass"] < 1.0 and post["risk_cost"] > 0.0

    @pytest.mark.asyncio
    async def test_core_priors_alu_skills_execute_via_interpreter(self) -> None:
        tiered = TieredWorkspace(tenant_id="tenant_m5")
        services = KernelServices.create_default(
            tiered.working_state,
            tiered_workspace=tiered,
            verification_pipeline=create_default_pipeline(
                default_scope=Scope(tenant_id="tenant_m5")
            ),
        )
        hydrated = hydrate_core_priors(tiered, grant=tiered._system_grant, tenant_id="tenant_m5")
        assert hydrated > 0

        skill_ids = {n.id for n in tiered.brain.graph.nodes_by_type(HCIRNodeType.SKILL)}
        assert {
            "prior_skill_geodesic_field",
            "prior_skill_bellman_risk_backup",
            "prior_skill_collision_cone",
            "prior_skill_bayesian_posterior",
        } <= skill_ids

        interp = HCIRInterpreter(tiered.working_state, services=services)
        caller_stream = InstructionStream(
            author="test_m5",
            instructions=[
                Instruction(
                    opcode=Opcode.CALL,
                    params={
                        "skill_id": "prior_skill_bellman_risk_backup",
                        "args": {"grid_shape": [4, 4], "goal_cells": [[3, 3]]},
                        "bind": "$r0",
                    },
                ),
                Instruction(opcode=Opcode.RETURN, params={"yield": ["$r0"]}),
            ],
        )
        res = await interp.execute(caller_stream)
        assert res.success is True
        ret_val = res.events[0].get("return_value") if res.events else None
        assert isinstance(ret_val, dict)
        assert ret_val["values"][3][3] == pytest.approx(10.0)


class TestDirichletAndToolBarrierPosteriors:
    """Verify Dirichlet transition posteriors and conjugate Beta tool-barrier learning."""

    def test_dirichlet_pos_action_and_belief_node_roundtrip(self) -> None:
        kb = TransitionKnowledgeBase()
        kb.record_action_outcome((2, 2), 1, TransitionOutcome.TERMINAL_FAIL, evidence="trap_up")
        kb.record_action_outcome((2, 2), 3, TransitionOutcome.TRAVERSED, evidence="safe_left")

        b_up = kb.get_action_belief((2, 2), 1)
        assert b_up is not None
        post = b_up.dirichlet_posterior()
        assert isinstance(post, DirichletOutcomePosterior)
        assert post.gamma_lethal > 1.0
        assert post.p_lethal > post.p_pass

        ev_up = kb.evaluate_action_transition((2, 2), 1, feature_id=0, locus=(1, 2))
        ev_left = kb.evaluate_action_transition((2, 2), 3, feature_id=0, locus=(2, 1))
        assert ev_up.traversable is False and ev_up.risk_cost >= 40.0
        assert ev_left.traversable is True and ev_left.risk_cost == 0.0

        bnode = b_up.to_belief_node()
        assert "dirichlet_posterior" in bnode.properties

        # Persist through transactions and rehydrate into fresh KB
        ops = kb.to_transaction_operations()
        rehydrated = TransitionKnowledgeBase()
        from hbllm.hcir.graph import BeliefNode, LearnedRuleNode

        nodes: list[Any] = []
        for op in ops:
            assert op.node_data is not None
            if op.node_data.get("node_type") == "belief":
                nodes.append(BeliefNode.model_validate(op.node_data))
            elif op.node_data.get("node_type") == "learned_rule":
                nodes.append(LearnedRuleNode.model_validate(op.node_data))
        assert rehydrated.hydrate_from_nodes(nodes) == 2
        ev_re = rehydrated.evaluate_action_transition((2, 2), 1, feature_id=0, locus=(1, 2))
        assert ev_re.traversable is False and ev_re.risk_cost >= 40.0

    def test_tool_barrier_beta_posteriors_and_frozen_theory(self) -> None:
        theory = HCIRSymbolicWorldTheory()
        theory.record_tool_barrier_observation(5, 9, unlocked=True, weight=2.0)
        theory.record_tool_barrier_observation(6, 9, unlocked=False, weight=2.0)

        a5, b5, m5 = theory.tool_barrier_posterior(5, 9)
        a6, b6, m6 = theory.tool_barrier_posterior(6, 9)
        assert a5 == pytest.approx(3.0) and b5 == pytest.approx(1.0) and m5 == pytest.approx(0.75)
        assert a6 == pytest.approx(1.0) and b6 == pytest.approx(3.0) and m6 == pytest.approx(0.25)
        assert theory.has_tool_barrier_affinity(5, 9) is True
        assert theory.is_tool_barrier_incompatible(6, 9) is True

        frozen = FrozenSymbolicTheory(
            tool_barrier_posteriors=MappingProxyType(dict(theory._tool_barrier_posteriors))
        )
        assert frozen.has_tool_barrier_affinity(5, 9) is True
        assert frozen.is_tool_barrier_incompatible(6, 9) is True


class TestHomoiconicCorticalProgramsAndPurePipeline:
    """Verify cortical control programs, pure CorticalPipeline, & non-mutating ReasoningRuntime."""

    @pytest.mark.asyncio
    async def test_cortical_programs_and_non_mutating_reasoning_runtime(self) -> None:
        ws = HCIRWorkspaceState(tenant_id="tenant_ctrl")
        services = KernelServices.create_default(
            ws,
            verification_pipeline=create_default_pipeline(
                default_scope=Scope(tenant_id="tenant_ctrl")
            ),
        )
        runtime = UnifiedReasoningRuntime(create_default_operator_registry())

        g_node = ConceptNode(id="g_ctrl", label="goal_cell", scope=Scope(tenant_id="tenant_ctrl"))
        receipt = ws.commit_operations(
            [
                TransactionOperation(
                    op=TransactionOp.ADD_NODE,
                    node_id=g_node.id,
                    node_data=g_node.model_dump(),
                )
            ],
            author="test_ctrl",
            tenant_id="tenant_ctrl",
        )
        assert receipt.is_committed
        snapshot_before = ws.graph.nodes.keys()

        for builder in (
            build_epistemic_phase_arbitration_program,
            build_ba10_subgoal_mediation_program,
            build_deliberation_safety_gating_program,
            build_curiosity_probe_program,
        ):
            trace = await runtime.execute_homoiconic_program(
                ws.graph,
                builder(author="test_ctrl"),
                services,
                scope=Scope(tenant_id="tenant_ctrl"),
            )
            assert trace.final_result is not None
            assert trace.final_result.status.value == "success"
            assert set(ws.graph.nodes.keys()) == set(snapshot_before)

        nodes = build_default_cortical_skill_nodes(tenant_id="tenant_ctrl")
        assert len(nodes) == 4
        assert all(isinstance(n, SkillNode) and n.bytecode_stream is not None for n in nodes)

    def test_pure_cortical_pipeline_selects_bellman_geodesic_action(self) -> None:
        pipeline = CorticalPipeline()
        grid = np.zeros((4, 4), dtype=np.int32)
        grid[0, 3] = 7
        theory = FrozenSymbolicTheory(goal_features=frozenset({7}))
        ctx = CognitiveContext(
            curr_grid=grid,
            avatar_pos=(2, 3),
            learned_goal_positions=frozenset({(0, 3)}),
            action_dynamics=MappingProxyType({1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}),
            symbolic_theory=theory,
        )
        res = pipeline.run(ctx)
        assert not res.rejected
        act = res.final_action[0] if isinstance(res.final_action, tuple) else res.final_action
        assert act == 1
        assert res.signals.get("target_phase") == EpistemicPhase.EXPLOITATION.value
        assert any(r.wm_deltas for r in res.stage_results)


class TestGroundedLatencyInvariant:
    """Verify grounded 32x32 step budget stays within CPU p95 <= 45ms."""

    def test_grounded_32x32_step_within_45ms_cpu_budget(self) -> None:
        sys = build_cognitive_system(
            CognitiveSystemConfig(
                world_id="m5_grounded_perf",
                memory_dir=None,
                seed=7,
                enable_grounding=True,
                enable_logging=False,
                load_core_priors=False,
            )
        )
        grid = np.zeros((32, 32), dtype=np.int32)
        grid[1, 1] = 3
        grid[28, 28] = 8
        grid[10, 10:20] = 2

        # Warm-up
        last_act: Any = None
        last_act, _ = sys.engine.step(
            EmbodiedObservation(lattice=grid.copy(), available_actions=[1, 2, 3, 4]),
            last_action=last_act,
        )

        cpu_ms: list[float] = []
        for idx in range(20):
            grid[1, 1] = 0
            grid[1 + (idx % 3), 1] = 3
            obs = EmbodiedObservation(lattice=grid.copy(), available_actions=[1, 2, 3, 4])
            t0 = time.process_time()
            last_act, _ = sys.engine.step(obs, last_action=last_act)
            cpu_ms.append((time.process_time() - t0) * 1000.0)

        p95_cpu = float(np.percentile(cpu_ms, 95))
        assert p95_cpu <= 45.0, f"Grounded 32x32 CPU p95 {p95_cpu:.2f}ms exceeds 45ms"
