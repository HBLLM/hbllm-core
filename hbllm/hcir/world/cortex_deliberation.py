"""Prefrontal Cortex Deliberation & Executive Arbitration Faculty (Facade).

Biologically modeled on mammalian dorsolateral / ventromedial prefrontal cortex (dlPFC / vmPFC)
and striatal gating:
1. Multi-Faculty Cognitive Arbitration (safety_predicates.py)
2. Basal Ganglia Tri-Pathway Gating (basal_ganglia.py)
3. Dual Neuromodulation Governor (neuromodulation.py)
4. Prefrontal Deliberation Engine (engine.py)
5. DeliberationStage (Cortical Stage 4 Pipeline Owner)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from hbllm.hcir.world.cortical_programs import evaluate_symbolic_stage_intent
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.deliberation import (
    ActionProposal,
    AdaptiveNeuromodulationGovernor,
    AnteriorCingulateConflictPredicate,
    BasalGangliaDecision,
    BasalGangliaGatingCircuit,
    BasalGangliaPathway,
    CausalUndoSafetyPredicate,
    DynamicPatrolTrajectoryPredicate,
    HabenularInhibitionPredicate,
    NeuromodulatoryState,
    OFCCounterfactualMentalSimulationPredicate,
    PlanSafetyEvaluator,
    PlanSafetyPredicate,
    PrefrontalDeliberationEngine,
    SafetyGateResult,
    StaticLethalHazardPredicate,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext

logger = logging.getLogger(__name__)


@dataclass
class DeliberationStage:
    """Cortical Stage 4: Prefrontal Deliberation & Arbitration.

    Receives CognitiveContext containing candidate proposals from upstream stages
    (such as simulation, macro actions, or episodic habit) and outputs a pure StageResult
    with arbitrated action and proposed transaction operations.
    """

    name: str = "deliberation_stage"
    engine: Any | None = None
    _current_journal: Any = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        engine = getattr(self, "engine", None)
        if engine is not None:
            from hbllm.hcir.world.engine_state import (
                FacultyWriteJournal,
                JournaledEngineView,
            )

            journal = FacultyWriteJournal(author="deliberation")
            self._current_journal = journal
            try:
                proposal = engine.deliberation_engine.deliberate(
                    JournaledEngineView(engine, journal),
                    curr_grid=ctx.signals.get("_curr_grid"),
                    available_actions=ctx.signals.get("_available_actions", []),
                    terminal=ctx.signals.get("_terminal"),
                    is_success=ctx.signals.get("_is_success", False),
                    is_failure=ctx.signals.get("_is_failure", False),
                    action_schemas=ctx.signals.get("_action_schemas"),
                    objects=ctx.signals.get("_objects"),
                    micro_actions=ctx.signals.get("_micro_actions"),
                    **ctx.signals.get("_kwargs", {}),
                )
            except Exception:
                engine.rollback_faculty_journal(journal)
                raise
            finally:
                self._current_journal = None
            meta = dict(getattr(proposal, "meta", {}))
            if "simulation_summary" not in meta:
                meta["simulation_summary"] = {
                    "plan_len": len(getattr(engine, "mental_plan", ()) or ()),
                    "subgoal_stack_depth": len(
                        getattr(getattr(engine, "subgoal_stack", None), "stack", ()) or ()
                    ),
                    "active_goals": len(getattr(engine, "learned_goal_positions", ()) or ()),
                }
            return StageResult(
                author=self.name,
                journal=journal,
                wm_deltas=list(journal.deltas),
                signals={
                    "_proposal": proposal,
                    "branch": getattr(proposal, "branch", "unknown"),
                    "meta": meta,
                },
            )
        return evaluate_symbolic_stage_intent(ctx, stage_kind="deliberation")


__all__ = [
    "ActionProposal",
    "AdaptiveNeuromodulationGovernor",
    "AnteriorCingulateConflictPredicate",
    "BasalGangliaDecision",
    "BasalGangliaGatingCircuit",
    "BasalGangliaPathway",
    "CausalUndoSafetyPredicate",
    "DeliberationStage",
    "DynamicPatrolTrajectoryPredicate",
    "HabenularInhibitionPredicate",
    "NeuromodulatoryState",
    "OFCCounterfactualMentalSimulationPredicate",
    "PlanSafetyEvaluator",
    "PlanSafetyPredicate",
    "PrefrontalDeliberationEngine",
    "SafetyGateResult",
    "StaticLethalHazardPredicate",
]
