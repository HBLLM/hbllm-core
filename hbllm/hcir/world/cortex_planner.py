"""Orbitofrontal Cortex (OFC) & Prefrontal Mental Simulation Planner.

Biologically modeled on mammalian OFC counterfactual preplay and frontopolar (BA10) cognitive branching:
1. Counterfactual Forward Simulation (MentalSimulationPlanner)
2. Topological Deadlock Pruning & Spatial Affordances (SpatialAffordanceModel)
3. BA10 Hierarchical Subgoal Scheduling (EpistemicGoalScheduler, FrontopolarSubgoalMCTSPlanner)
4. Dynamic Plan Repair & Metric Kinematics (DynamicReplanner, ContinuousKinematicPlanner)
5. SimulationStage (Cortical Stage 3 Pipeline Owner)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hbllm.hcir.world.cortical_programs import evaluate_symbolic_stage_intent
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.frontopolar_subgoal_stack import FrontopolarSubgoal, SubgoalType
from hbllm.hcir.world.planning import (
    DynamicReplanner,
    EpistemicGoalProposal,
    EpistemicGoalScheduler,
    FrontopolarSubgoalMCTSNode,
    FrontopolarSubgoalMCTSPlanner,
    FrontopolarSubgoalMCTSResult,
    MentalSimulationPlanner,
    MentalSimulationStep,
    ReplanResult,
    SpatialAffordanceModel,
    SpatialAffordanceSnapshot,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext

logger = logging.getLogger(__name__)


@dataclass
class SimulationStage:
    """Cortical Stage 3: Counterfactual Simulation & Subgoal Scheduling.

    OFC mental simulation rollouts; dead-end pruning; BA10 recursive subgoal decomposition.
    """

    name: str = "simulation_stage"
    engine: Any | None = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        engine = getattr(self, "engine", None)
        if engine is not None or ctx.signals.get("_unified_deliberation"):
            plan_len = len(getattr(engine, "mental_plan", ctx.mental_plan) or ())
            sg_obj = getattr(engine, "subgoal_stack", None) or ctx.subgoal_stack
            sg_depth = len(getattr(sg_obj, "stack", ()) or ())
            active_goals = len(
                getattr(engine, "learned_goal_positions", ctx.learned_goal_positions) or ()
            )
            return StageResult(
                author=self.name,
                signals={
                    "simulation_summary": {
                        "plan_len": plan_len,
                        "subgoal_stack_depth": sg_depth,
                        "active_goals": active_goals,
                    }
                },
            )
        return evaluate_symbolic_stage_intent(ctx, stage_kind="simulation")


__all__ = [
    "DynamicReplanner",
    "EpistemicGoalProposal",
    "EpistemicGoalScheduler",
    "FrontopolarSubgoal",
    "FrontopolarSubgoalMCTSNode",
    "FrontopolarSubgoalMCTSPlanner",
    "FrontopolarSubgoalMCTSResult",
    "MentalSimulationPlanner",
    "MentalSimulationStep",
    "ReplanResult",
    "SimulationStage",
    "SpatialAffordanceModel",
    "SpatialAffordanceSnapshot",
    "SubgoalType",
]
