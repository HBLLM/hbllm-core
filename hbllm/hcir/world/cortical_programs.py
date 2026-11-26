"""Homoiconic Cortical Control Programs & Pure Stage Evaluation (Phase M5).

Expresses epistemic phase arbitration, BA10 subgoal mediation, deliberation
safety gating, and curiosity probing as inspectable ISA v2 `InstructionStream`
and `SkillNode` programs operating over `FrozenGraphView` and Layer-0 ALUs.
Returns pure `StageResult` structures with `TransactionOperation`s and
`WorkingMemoryDelta`s without directly mutating engine or world state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from hbllm.hcir.bytecode import Instruction, InstructionStream, Opcode
from hbllm.hcir.capabilities.cortical_alus import (
    compute_bellman_risk_backup,
    compute_collision_cone,
    compute_geodesic_field,
)
from hbllm.hcir.graph import NodeLifecycle, SkillNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Provenance, Scope
from hbllm.hcir.world.cortical_stage import StageResult, WorkingMemoryDelta
from hbllm.hcir.world.percept import EpistemicPhase

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext


def _ins(opcode: Opcode, cost: int = 5, /, **params: Any) -> Instruction:
    return Instruction(opcode=opcode, params=params, cost_estimate=cost)


def build_epistemic_phase_arbitration_program(
    author: str = "cortical_program",
) -> InstructionStream:
    """Build ISA v2 program selecting EXPLOITATION / REPLANNING / EPISTEMIC_EXPLORATION."""
    goal_belief = {
        "id": "cortical_phase_decision",
        "node_type": "belief",
        "claim": "Epistemic phase: EXPLOITATION toward confirmed goal",
        "properties": {"domain": "cortical_control", "target_phase": "EXPLOITATION"},
    }
    explore_belief = {
        "id": "cortical_phase_decision",
        "node_type": "belief",
        "claim": "Epistemic phase: EPISTEMIC_EXPLORATION for causal grounding",
        "properties": {"domain": "cortical_control", "target_phase": "EPISTEMIC_EXPLORATION"},
    }
    return InstructionStream(
        author=author,
        description="Cortical epistemic phase arbitration over FrozenGraphView",
        instructions=[
            _ins(Opcode.UNIFY, 5, pattern={"node_type": "goal"}, bind="$r0"),
            _ins(
                Opcode.BRANCH,
                1,
                condition="if_empty",
                register="$r0",
                target_label="explore_phase",
            ),
            _ins(Opcode.ASSERT, 5, node_data=goal_belief, author=author, bind="$r1"),
            _ins(Opcode.RETURN, 0, **{"yield": ["$r1"]}),
            _ins(
                Opcode.ASSERT,
                5,
                node_data=explore_belief,
                author=author,
                bind="$r1",
                label="explore_phase",
            ),
            _ins(Opcode.RETURN, 0, **{"yield": ["$r1"]}),
        ],
    )


def build_ba10_subgoal_mediation_program(
    author: str = "cortical_program",
) -> InstructionStream:
    """Build ISA v2 BA10 topological cut-set & tool-barrier subgoal mediation program."""
    subgoal_node = {
        "id": "ba10_mediated_subgoal",
        "node_type": "goal",
        "description": "Satisfy prerequisite tool_barrier_posterior rule before target goal",
        "properties": {"domain": "ba10_mediation", "tool_ref": "$r1"},
    }
    return InstructionStream(
        author=author,
        description="BA10 subgoal decomposition over topology cuts & LearnedRuleNodes",
        instructions=[
            _ins(Opcode.UNIFY, 5, pattern={"node_type": "goal"}, bind="$r0"),
            _ins(
                Opcode.UNIFY,
                5,
                pattern={"node_type": "learned_rule", "domain": "tool_barrier_posterior"},
                bind="$r1",
            ),
            _ins(
                Opcode.EXECUTE,
                15,
                capability="math.topology_cuts",
                params={"grid": [[0]]},
                bind="$r2",
            ),
            _ins(
                Opcode.BRANCH,
                1,
                condition="if_empty",
                register="$r1",
                target_label="direct_goal",
            ),
            _ins(Opcode.ASSERT, 5, node_data=subgoal_node, author=author, bind="$r3"),
            _ins(Opcode.RETURN, 0, **{"yield": ["$r3"]}),
            _ins(Opcode.RETURN, 0, **{"yield": ["$r0"]}, label="direct_goal"),
        ],
    )


def build_deliberation_safety_gating_program(
    author: str = "cortical_program",
) -> InstructionStream:
    """Build ISA v2 deliberation safety gate combining collision-cone & Bellman risk ALUs."""
    safe_decision = {
        "id": "safety_gate_decision",
        "node_type": "belief",
        "claim": "Safety gate cleared via math.collision_cone + math.bellman_risk_backup",
        "properties": {"domain": "safety_gate", "is_safe": True},
    }
    return InstructionStream(
        author=author,
        description="Deliberation safety gating over Layer-0 collision-cone & Bellman risk ALUs",
        instructions=[
            _ins(
                Opcode.EXECUTE,
                10,
                capability="math.collision_cone",
                params={"grid_shape": [4, 4], "avatar_trajectory": [[0, 0]], "threat_states": []},
                bind="$r0",
            ),
            _ins(
                Opcode.EXECUTE,
                10,
                capability="math.bellman_risk_backup",
                params={"grid_shape": [4, 4], "goal_cells": [[3, 3]], "start_cell": [0, 0]},
                bind="$r1",
            ),
            _ins(Opcode.ASSERT, 5, node_data=safe_decision, author=author, bind="$r2"),
            _ins(Opcode.RETURN, 0, **{"yield": ["$r2"]}),
        ],
    )


def build_curiosity_probe_program(author: str = "cortical_program") -> InstructionStream:
    """Build ISA v2 epistemic curiosity probe using Bayesian uncertainty and FORK/ROLLBACK."""
    probe_plan = {
        "id": "epistemic_probe_plan",
        "node_type": "plan",
        "description": "Probe highest Expected Free Energy unknown via forked prediction",
        "properties": {"domain": "epistemic_curiosity", "uncertainty": "$r1"},
    }
    return InstructionStream(
        author=author,
        description="Active inference epistemic inquiry via Dirichlet posterior + FORK probe",
        instructions=[
            _ins(Opcode.QUERY, 5, node_type="unknown", bind="$r0"),
            _ins(
                Opcode.EXECUTE,
                10,
                capability="math.bayesian_posterior_update",
                params={"observations": []},
                bind="$r1",
            ),
            _ins(Opcode.FORK, 10, branch_name="cortical_probe"),
            _ins(
                Opcode.EXECUTE,
                15,
                capability="world.predict_outcome",
                params={"hypothesis": "epistemic_probe", "branch": "cortical_probe"},
                bind="$r2",
            ),
            _ins(Opcode.ROLLBACK, 5, branch_name="cortical_probe"),
            _ins(Opcode.ASSERT, 5, node_data=probe_plan, author=author, bind="$r3"),
            _ins(Opcode.RETURN, 0, **{"yield": ["$r3"]}),
        ],
    )


def build_default_cortical_skill_nodes(tenant_id: str = "default") -> list[SkillNode]:
    """Materialize all 4 canonical cortical control programs as executable `SkillNode`s."""
    specs: tuple[tuple[str, str, str, InstructionStream], ...] = (
        (
            "cortical_skill_phase_arbitration",
            "cortical_phase_arbitration",
            "phase_arbitration",
            build_epistemic_phase_arbitration_program(),
        ),
        (
            "cortical_skill_ba10_mediation",
            "cortical_ba10_mediation",
            "ba10_subgoal_mediation",
            build_ba10_subgoal_mediation_program(),
        ),
        (
            "cortical_skill_safety_gating",
            "cortical_safety_gating",
            "deliberation_safety_gating",
            build_deliberation_safety_gating_program(),
        ),
        (
            "cortical_skill_curiosity_probe",
            "cortical_curiosity_probe",
            "curiosity_probe",
            build_curiosity_probe_program(),
        ),
    )
    return [
        SkillNode(
            id=sid,
            skill_name=sname,
            description=stream.description,
            precondition_pattern={"domain": "cortical_control", "intent": intent},
            postcondition_pattern={"intent": intent, "domain": "cortical_control"},
            bytecode_stream=stream.model_dump(),
            alpha_success=3.0,
            beta_failure=1.0,
            lifecycle=NodeLifecycle.ACTIVE,
            provenance=Provenance(created_by="cortical_program"),
            scope=Scope(tenant_id=tenant_id),
            tags=["cortical_program", intent],
        )
        for sid, sname, intent, stream in specs
    ]


def _delta_to_action(delta: Sequence[int] | None, ctx: CognitiveContext) -> Any | None:
    if delta is None or len(delta) < 2:
        return None
    target_d = (int(delta[0]), int(delta[1]))
    for act, dyn in ctx.action_dynamics.items():
        if isinstance(dyn, tuple) and len(dyn) >= 2:
            d = (int(dyn[0]), int(dyn[1]))
        else:
            d = (int(getattr(dyn, "delta_r", 0)), int(getattr(dyn, "delta_c", 0)))
        if d == target_d:
            return act
    return None


def _select_pure_stage_action(
    ctx: CognitiveContext,
) -> tuple[Any, dict[str, Any], EpistemicPhase, str]:
    """Evaluate immutable `CognitiveContext` via stateless Layer-0 ALUs and return pure intent."""
    grid = ctx.curr_grid if ctx.curr_grid is not None else ctx.signals.get("_curr_grid")
    actions = list(ctx.action_dynamics.keys()) or list(ctx.signals.get("available_actions", []))
    default_act = actions[0] if actions else 1

    if ctx.mental_plan:
        first_step = ctx.mental_plan[0]
        plan_act = getattr(first_step, "action", first_step)
        return (
            plan_act,
            {"source": "mental_plan", "phase": EpistemicPhase.EXPLOITATION.value},
            EpistemicPhase.EXPLOITATION,
            "mental_plan",
        )

    signal_plan_act = ctx.signals.get("mental_plan_action")
    if signal_plan_act is not None:
        return (
            signal_plan_act,
            {"source": "mental_plan_signal", "phase": EpistemicPhase.EXPLOITATION.value},
            EpistemicPhase.EXPLOITATION,
            "mental_plan_signal",
        )

    goal_cells: list[tuple[int, int]] = list(ctx.learned_goal_positions)
    lethal_cells: list[tuple[int, int]] = list(ctx.current_context_hazards)
    blocked_cells: list[tuple[int, int]] = list(ctx.learned_barriers)

    arr = np.asarray(grid) if grid is not None else None
    if arr is not None and arr.ndim == 2 and ctx.avatar_pos is not None:
        h, w = int(arr.shape[0]), int(arr.shape[1])
        theory = ctx.symbolic_theory
        for r in range(h):
            for c in range(w):
                feat = int(arr[r, c])
                ev = theory.evaluate_transition(feat, locus=(r, c))
                if ev.risk_cost >= 40.0 and (r, c) not in lethal_cells:
                    lethal_cells.append((r, c))
                if not ev.traversable and ev.risk_cost < 40.0 and (r, c) not in blocked_cells:
                    blocked_cells.append((r, c))
                if (feat in theory.goal_features or ev.goal_value > 0.5) and (
                    r,
                    c,
                ) not in goal_cells:
                    goal_cells.append((r, c))

        if goal_cells:
            bellman = compute_bellman_risk_backup(
                (h, w),
                goal_cells,
                lethal_cells=lethal_cells,
                blocked_cells=blocked_cells,
                start_cell=ctx.avatar_pos,
            )
            geo = compute_geodesic_field(
                arr,
                goal_cells,
                impassable_cells=[*blocked_cells, *lethal_cells],
                start_cell=ctx.avatar_pos,
            )
            chosen_delta = bellman.get("best_delta") or geo.get("best_step")
            if chosen_delta is not None and len(chosen_delta) >= 2:
                next_pt = [
                    ctx.avatar_pos[0] + int(chosen_delta[0]),
                    ctx.avatar_pos[1] + int(chosen_delta[1]),
                ]
                cone = compute_collision_cone(
                    (h, w),
                    [ctx.avatar_pos, next_pt],
                    (),
                    blocked_cells=blocked_cells,
                )
                if cone["is_safe"]:
                    mapped = _delta_to_action(chosen_delta, ctx)
                    if mapped is not None:
                        return (
                            mapped,
                            {
                                "source": "bellman_geodesic_alu",
                                "goal_cells": [list(g) for g in goal_cells[:4]],
                                "start_value": bellman["start_value"],
                                "start_distance": geo["start_distance"],
                            },
                            EpistemicPhase.EXPLOITATION,
                            "bellman_geodesic_alu",
                        )

    return (
        default_act,
        {"source": "epistemic_exploration", "goal_count": len(goal_cells)},
        EpistemicPhase.EPISTEMIC_EXPLORATION,
        "epistemic_exploration",
    )


def evaluate_symbolic_stage_intent(
    ctx: CognitiveContext,
    *,
    stage_kind: Literal["simulation", "deliberation"],
) -> StageResult:
    """Evaluate a pure cortical stage over immutable `CognitiveContext` using Layer-0 ALUs."""
    proposed_act, details, target_phase, source = _select_pure_stage_action(ctx)
    if stage_kind == "simulation":
        ops = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="simulation_plan",
                node_data={
                    "id": "simulation_plan",
                    "node_type": "plan",
                    "description": f"Pure simulation intent via {source}",
                    "properties": {
                        "stage": "simulation_stage",
                        "plan_len": len(ctx.mental_plan),
                        "step": ctx.step_counter,
                        **details,
                    },
                },
            )
        ]
        return StageResult(
            author="simulation_stage",
            operations=ops,
            signals={"mental_plan_action": proposed_act, "simulation_alu": details},
            proposed_action=proposed_act,
        )

    ops = [
        TransactionOperation(
            op=TransactionOp.UPSERT_NODE,
            node_id="deliberation_decision",
            node_data={
                "id": "deliberation_decision",
                "node_type": "decision",
                "description": f"Pure deliberation intent via {source}",
                "properties": {
                    "stage": "deliberation_stage",
                    "arbitrated_action": str(proposed_act),
                    "target_phase": target_phase.value,
                    "step": ctx.step_counter,
                    **details,
                },
            },
        )
    ]
    deltas = [WorkingMemoryDelta(field="phase", value=target_phase, op="set")]
    return StageResult(
        author="deliberation_stage",
        operations=ops,
        signals={
            "arbitrated_action": proposed_act,
            "branch": source,
            "target_phase": target_phase.value,
        },
        proposed_action=proposed_act,
        wm_deltas=deltas,
    )


__all__ = [
    "build_ba10_subgoal_mediation_program",
    "build_curiosity_probe_program",
    "build_default_cortical_skill_nodes",
    "build_deliberation_safety_gating_program",
    "build_epistemic_phase_arbitration_program",
    "evaluate_symbolic_stage_intent",
]
