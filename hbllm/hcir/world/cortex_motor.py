"""Basal Ganglia & Anterior Cingulate Cortex (ACC) Motor Arbitration Faculty (Facade).

Biologically modeled on mammalian striatal action selection, dopamine RPE gating,
and ACC conflict monitoring:
1. Conflict Monitoring & Limit-Cycle Breaking (conflict.py)
2. Epistemic Curiosity Explorer (curiosity.py)
3. Effector Selection & Spatial Targeting (effector.py)
4. Egocentric Parietal Motor Dispatch (dispatcher.py)
5. MotorStage (Cortical Stage 5 Pipeline Owner)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.hcir.transactions import TransactionOperation
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.motor import (
    ACCConflictMonitor,
    ACCConflictRecord,
    EffectorTargetScorer,
    EgocentricDispatchResult,
    EgocentricMotorIntent,
    EpistemicCuriosityExplorer,
    EpistemicProbeProposal,
    MotorCortexEffector,
    ParietalMotorDispatcher,
)

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext

logger = logging.getLogger(__name__)


class MotorStage:
    """Pure CorticalStage for Stage 5: Motor Dispatch & Transaction Execution.

    In accordance with Invariant 1 and WP1, this stage observes an immutable CognitiveContext
    and emits a pure StageResult containing the selected motor action and state mutation intent.
    """

    name: str = "motor_stage"
    engine: Any | None = None
    _current_journal: Any = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        engine = getattr(self, "engine", None)
        if engine is not None:
            # WP1: real motor dispatch — finalise DeliberationStage's
            # ActionProposal through _finalize_step (grid already normalized).
            proposal = ctx.signals.get("_proposal")
            if proposal is None or not hasattr(proposal, "action"):
                return StageResult(author=self.name)
            if getattr(proposal, "finalized", False):
                meta = dict(getattr(proposal, "meta", {}))
                meta["chosen_action"] = proposal.action
                meta["action_data"] = proposal.action_data
                return StageResult(
                    author=self.name,
                    signals={"branch": proposal.branch, "meta": meta},
                    proposed_action=(proposal.action, meta),
                )
            from hbllm.hcir.world.engine_state import (
                FacultyWriteJournal,
                JournaledEngineView,
            )

            journal = FacultyWriteJournal(author="motor_stage")
            self._current_journal = journal
            try:
                act, data = engine.deliberation_engine._finalize_step(
                    JournaledEngineView(engine, journal),
                    proposal.grid,
                    proposal.action,
                    proposal.action_data,
                    predicted_pos=proposal.predicted_pos,
                )
            except Exception:
                engine.rollback_faculty_journal(journal)
                raise
            finally:
                self._current_journal = None
            meta = dict(getattr(proposal, "meta", {}))
            meta["chosen_action"] = act
            meta["action_data"] = data
            return StageResult(
                author=self.name,
                journal=journal,
                signals={"branch": proposal.branch, "meta": meta},
                proposed_action=(act, meta),
            )
        proposed_action = ctx.signals.get("proposed_action")
        ops: list[TransactionOperation] = []
        signals: dict[str, Any] = {}
        if proposed_action is None:
            actions = (
                list(ctx.action_dynamics.keys())
                or list(ctx.signals.get("available_actions", []))
                or [1, 2, 3, 4]
            )
            proposed_action = actions[0] if actions else None
        meta = dict(ctx.signals.get("meta", {}))
        meta.setdefault("chosen_action", proposed_action)
        meta.setdefault("goal_target", None)
        meta.setdefault("goal_source", "none")
        meta.setdefault("subgoal_stack", [])
        meta.setdefault("plan", None)
        meta.setdefault("plan_source", "arbitration")
        meta.setdefault(
            "candidates",
            {
                int(a): (1.0 if a == proposed_action else 0.0)
                for a in (actions if proposed_action is not None else [1, 2, 3, 4])
            },
        )
        meta.setdefault("veto_reasons", [])
        meta.setdefault("phase", "unknown")
        meta.setdefault("branch", "fallback")
        meta.setdefault("action_data", None)
        signals["meta"] = meta
        return StageResult(
            author=self.name,
            operations=ops,
            signals=signals,
            proposed_action=(proposed_action, meta),
        )


__all__ = [
    "ACCConflictMonitor",
    "ACCConflictRecord",
    "EffectorTargetScorer",
    "EgocentricDispatchResult",
    "EgocentricMotorIntent",
    "EpistemicCuriosityExplorer",
    "EpistemicProbeProposal",
    "MotorCortexEffector",
    "MotorStage",
    "ParietalMotorDispatcher",
]
