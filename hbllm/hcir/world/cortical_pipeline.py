from __future__ import annotations

"""Canonical 5-Stage Cortical Pipeline.

Executes the biological active inference loop from AGENTS.md §2:
1. SENSORY PERCEPTION & COORDINATE LIFTING (PerceptionStage)
2. ASSIMILATION & SURPRISE INDUCTION (AssimilationStage)
3. COUNTERFACTUAL SIMULATION & SUBGOAL SCHEDULING (SimulationStage)
4. PREFRONTAL DELIBERATION & SAFETY GATING (DeliberationStage)
5. MOTOR DISPATCH & TRANSACTION EXECUTION (MotorStage)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.hcir.transactions import HCIRTransaction
from hbllm.hcir.world.cortex_assimilator import AssimilationStage
from hbllm.hcir.world.cortex_deliberation import DeliberationStage
from hbllm.hcir.world.cortex_motor import MotorStage
from hbllm.hcir.world.cortex_perception import PerceptionStage
from hbllm.hcir.world.cortex_planner import SimulationStage
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult

if TYPE_CHECKING:
    from hbllm.hcir.kernel.transaction_manager import TransactionManager
    from hbllm.hcir.world.engine_state import CognitiveContext


@dataclass
class PipelineResult:
    """The result of executing the full 5-stage cortical pipeline."""

    final_action: Any
    stage_results: list[StageResult] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    operations: list[Any] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str | None = None


class CorticalPipeline:
    """The authoritative 5-stage epistemic active inference pipeline."""

    def __init__(
        self,
        stages: Sequence[CorticalStage] | None = None,
        transaction_manager: TransactionManager | None = None,
        engine: Any = None,
        grounding_bridge: Any = None,
    ) -> None:
        if stages is not None:
            self.stages: list[CorticalStage] = list(stages)
        else:
            self.stages = [
                PerceptionStage(),
                AssimilationStage(),
                SimulationStage(),
                DeliberationStage(),
                MotorStage(),
            ]
        self.transaction_manager = transaction_manager
        self.engine = engine
        self.grounding_bridge = grounding_bridge

    def run(self, ctx: CognitiveContext) -> PipelineResult:
        """Run the active inference cycle across all stages.

        Each stage observes the (immutable) context augmented with signals
        from upstream stages. Accumulates operations for atomic commit.
        """
        current_ctx = ctx
        stage_results: list[StageResult] = []
        all_ops: list[Any] = []
        final_action: Any = None

        for stage in self.stages:
            result = stage.observe(current_ctx)
            stage_results.append(result)
            all_ops.extend(result.operations)
            if result.signals:
                current_ctx = current_ctx.with_signals(result.signals)
            if result.proposed_action is not None:
                final_action = result.proposed_action

            # Stage 1: SGE Perception Bridge Hook
            if (
                self.grounding_bridge is not None
                and getattr(stage, "name", "") == "perception_stage"
            ):
                extracted = current_ctx.signals.get(
                    "extracted_entities"
                ) or current_ctx.signals.get("_objects")
                p_signals = self.grounding_bridge.process_perception(
                    current_ctx,
                    extracted_entities=extracted if isinstance(extracted, list) else None,
                )
                current_ctx = current_ctx.with_signals(p_signals)

            # Stage 2: SGE Assimilation Bridge Hook
            elif (
                self.grounding_bridge is not None
                and getattr(stage, "name", "") == "assimilation_stage"
            ):
                states = current_ctx.signals.get("sge_entity_states", {})
                a_signals = self.grounding_bridge.process_assimilation(
                    current_ctx, current_entity_states=states if isinstance(states, dict) else {}
                )
                current_ctx = current_ctx.with_signals(a_signals)

            # WP1: each stage's journaled faculty writes commit before the next stage.
            if getattr(result, "journal", None) is not None and self.engine is not None:
                if not self.engine.commit_faculty_journal(result.journal):
                    return PipelineResult(
                        final_action=None,
                        stage_results=stage_results,
                        signals=dict(current_ctx.signals),
                        operations=all_ops,
                        rejected=True,
                        rejection_reason=f"{result.author}: journal commit rejected",
                    )

        # Stage 5: SGE Motor Gateway Authorization Hook
        if self.grounding_bridge is not None and final_action is not None:
            act = final_action[0] if isinstance(final_action, tuple) else final_action
            act_data = (
                final_action[1]
                if isinstance(final_action, tuple)
                and len(final_action) > 1
                and isinstance(final_action[1], dict)
                else {}
            )
            auth_act, rej_info = self.grounding_bridge.process_motor(
                current_ctx, proposed_action=act, action_data=act_data
            )
            if rej_info and rej_info.get("action_rejected_by_gateway"):
                return PipelineResult(
                    final_action=None,
                    stage_results=stage_results,
                    signals=dict(current_ctx.signals),
                    operations=all_ops,
                    rejected=True,
                    rejection_reason=f"ActionValidationGateway rejected: {rej_info.get('rejection_reasons')}",
                )

        # If a transaction manager is attached, batch-commit all operations
        if self.transaction_manager is not None and all_ops:
            tx = HCIRTransaction(author="cortical_pipeline", operations=all_ops)
            try:
                self.transaction_manager.commit(tx)
            except Exception as exc:
                return PipelineResult(
                    final_action=None,
                    stage_results=stage_results,
                    signals=dict(current_ctx.signals),
                    operations=all_ops,
                    rejected=True,
                    rejection_reason=str(exc),
                )

        return PipelineResult(
            final_action=final_action,
            stage_results=stage_results,
            signals=dict(current_ctx.signals),
            operations=all_ops,
        )
