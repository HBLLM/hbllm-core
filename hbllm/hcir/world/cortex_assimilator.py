"""Epistemic Feedback Assimilator Faculty (Facade).

Assimilates sensory feedback (pixel diffs, kinetic flow, reward signals)
into motor dynamics, avatar identity consensus, and empirical neuro-symbolic theory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


def get_sub_attr(target: Any, attr: str, default: Any = None) -> Any:
    return getattr(target, attr, default) if target is not None else default


from dataclasses import dataclass

from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.assimilation.assimilator import EpistemicFeedbackAssimilator
from hbllm.hcir.world.cortical_stage import StageResult
from hbllm.hcir.world.percept import EpistemicPhase

if TYPE_CHECKING:
    from hbllm.hcir.world.engine_state import CognitiveContext


@dataclass
class AssimilationStage:
    """Cortical Stage 2: Assimilation & Surprise Induction.

    Binds motor diffs to actions; updates EpistemicPhase; detects high surprise
    to invalidate stale causal hypotheses. Emits transactions updating
    dynamic theories and working memory.
    """

    name: str = "assimilation_stage"
    engine: Any | None = None
    _current_journal: Any = None

    def bind_engine(self, engine: Any) -> None:
        self.engine = engine

    def observe(self, ctx: CognitiveContext) -> StageResult:
        engine = getattr(self, "engine", None)
        if engine is not None:
            # WP1: real assimilation — the full prepare() preamble (terminal
            # resolution, step_counter, normalize, strict-budget, bg estimate,
            # assimilate_feedback) runs here; DeliberationStage then calls
            # deliberate() starting from the already-assimilated state.
            from hbllm.hcir.world.engine_state import (
                FacultyWriteJournal,
                JournaledEngineView,
            )

            journal = FacultyWriteJournal(author="assimilation_stage")
            self._current_journal = journal
            view = JournaledEngineView(engine, journal)
            try:
                g, term = engine.deliberation_engine.prepare(
                    view,
                    ctx.signals.get("_curr_grid"),
                    ctx.signals.get("_available_actions", []),
                    terminal=ctx.signals.get("_terminal"),
                    is_success=ctx.signals.get("_is_success", False),
                    is_failure=ctx.signals.get("_is_failure", False),
                    action_schemas=ctx.signals.get("_action_schemas"),
                    **ctx.signals.get("_kwargs", {}),
                )
            except Exception:
                engine.rollback_faculty_journal(journal)
                raise
            finally:
                self._current_journal = None
            return StageResult(
                author=self.name,
                journal=journal,
                wm_deltas=list(journal.deltas),
                signals={"_curr_grid": g, "_terminal": term},
            )
        ops: list[TransactionOperation] = [
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="assimilation_state",
                node_data={
                    "stage": self.name,
                    "phase": str(ctx.phase),
                    "step": ctx.step_counter,
                },
            )
        ]
        signals = {
            "assimilated_phase": str(ctx.phase),
            "step": ctx.step_counter,
        }
        return StageResult(
            author=self.name,
            operations=ops,
            signals=signals,
            proposed_action=None,
        )


__all__ = [
    "AssimilationStage",
    "EpistemicFeedbackAssimilator",
    "EpistemicPhase",
    "get_sub_attr",
]
