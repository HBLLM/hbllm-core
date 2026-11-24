from __future__ import annotations

"""Canonical 5-Stage Cortical Pipeline.

Executes the biological active inference loop from AGENTS.md §2:
1. SENSORY PERCEPTION & COORDINATE LIFTING (PerceptionStage)
2. ASSIMILATION & SURPRISE INDUCTION (AssimilationStage)
3. COUNTERFACTUAL SIMULATION & SUBGOAL SCHEDULING (SimulationStage)
4. PREFRONTAL DELIBERATION & SAFETY GATING (DeliberationStage)
5. MOTOR DISPATCH & TRANSACTION EXECUTION (MotorStage)
"""

import time
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
    stage_timings_ms: dict[str, float] = field(default_factory=dict)
    total_duration_ms: float = 0.0


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
        self.last_stage_timings_ms: dict[str, float] = {}
        self.last_duration_ms: float = 0.0
        if self.engine is not None:
            for stage in self.stages:
                if hasattr(stage, "bind_engine"):
                    stage.bind_engine(self.engine)
                elif hasattr(stage, "engine"):
                    stage.engine = self.engine

    def run(self, ctx: CognitiveContext) -> PipelineResult:
        """Run the active inference cycle across all stages.

        Each stage observes the (immutable) context augmented with signals
        from upstream stages. Accumulates operations for atomic commit.
        """
        t_run0 = time.perf_counter()
        stage_timings_ms: dict[str, float] = {}
        current_ctx = ctx
        stage_results: list[StageResult] = []
        all_ops: list[Any] = []
        final_action: Any = None
        bridge = self.grounding_bridge
        if self.engine is not None and hasattr(self.engine, "grounding_bridge"):
            eng_bridge = getattr(self.engine, "grounding_bridge", None)
            if eng_bridge is not None:
                bridge = eng_bridge

        grid_data = (
            current_ctx.curr_grid
            if current_ctx.curr_grid is not None
            else current_ctx.signals.get("_curr_grid")
        )
        if self.engine is not None and grid_data is not None:
            theory = getattr(self.engine, "symbolic_theory", None)
            wm = getattr(self.engine, "working_memory", None)
            if theory is not None and hasattr(theory, "observe_grid_structure"):
                bg_feat = getattr(self.engine, "bg_feature", 0)
                av_feats = set(getattr(self.engine, "avatar_features", ()) or ())
                if getattr(self.engine, "avatar_feature", None) is not None:
                    av_feats.add(self.engine.avatar_feature)
                shifted = theory.observe_grid_structure(
                    grid_data, bg_feature=bg_feat, avatar_features=av_feats
                )
                if shifted:
                    from hbllm.hcir.world.cortex_causal import (
                        invalidate_stale_context_ontology,
                    )

                    invalidate_stale_context_ontology(
                        theory,
                        hazard_tracker=getattr(self.engine, "hazard_tracker", None),
                        episodic_cortex=getattr(self.engine, "episodic_cortex", None),
                        working_memory=wm,
                        engine_or_view=self.engine,
                        bg_feature=bg_feat,
                        preserve_current_context=False,
                        reason="structural_label_permutation",
                    )
                    self.engine.active_arbitrated_goal = None

        for idx, stage in enumerate(self.stages):
            t_stage0 = time.perf_counter()
            try:
                result = stage.observe(current_ctx)
            except Exception:
                cur_journal = getattr(stage, "_current_journal", None)
                if self.engine is not None and cur_journal is not None:
                    self.engine.rollback_faculty_journal(cur_journal)
                raise
            stage_name = getattr(stage, "name", "")
            stage_key = stage_name or type(stage).__name__ or f"stage_{idx}"
            stage_timings_ms[stage_key] = round((time.perf_counter() - t_stage0) * 1000.0, 4)
            stage_results.append(result)
            all_ops.extend(result.operations)
            if result.signals:
                current_ctx = current_ctx.with_signals(result.signals)
            if result.proposed_action is not None:
                final_action = result.proposed_action

            # Stage 1: SGE Perception Bridge + G8/G10/G11 Hooks
            if stage_name == "perception_stage":
                t_hook0 = time.perf_counter()
                extracted = current_ctx.signals.get(
                    "extracted_entities"
                ) or current_ctx.signals.get("_objects")
                if bridge:
                    p_signals = bridge.process_perception(
                        current_ctx,
                        extracted_entities=extracted if isinstance(extracted, list) else None,
                    )
                    current_ctx = current_ctx.with_signals(p_signals)
                else:
                    p_signals = {}
                if self.engine is not None and self.engine.working_memory is not None:
                    wm_ref = self.engine.working_memory
                    merged_priors: dict[int, dict[str, float]] = (
                        dict(p_signals.get("sge_role_priors", {}))
                        if bridge
                        else dict(wm_ref.concept_role_priors)
                    )
                    obs_meta = wm_ref.last_observation_metadata
                    if isinstance(obs_meta, dict):
                        hint = (
                            obs_meta.get("rule_hint")
                            or obs_meta.get("instruction")
                            or obs_meta.get("mission")
                        )
                        if isinstance(hint, str) and hint.strip():
                            clean_hint = hint.strip()
                            av_s = set(self.engine.avatar_features or ())
                            if self.engine.avatar_feature is not None:
                                av_s.add(self.engine.avatar_feature)
                            bg_f = (
                                int(self.engine.bg_feature)
                                if self.engine.bg_feature is not None
                                else 0
                            )
                            if bridge:
                                if clean_hint not in wm_ref.compiled_linguistic_hints:
                                    d_priors = bridge.ground_mission_instruction(
                                        clean_hint,
                                        grid_data,
                                        bg_f,
                                        av_s,
                                        self.engine.symbolic_theory,
                                    )
                                    wm_ref.compiled_linguistic_hints.add(clean_hint)
                                else:
                                    d_priors = bridge.state.dialogue_role_priors
                            elif clean_hint not in wm_ref.compiled_linguistic_hints:
                                import numpy as np

                                from hbllm.hcir.compiler import HCIRCompiler

                                comp = HCIRCompiler()
                                th = self.engine.symbolic_theory
                                low_h = clean_hint.lower()
                                g_verbs = (
                                    "go ",
                                    "reach ",
                                    "navigate ",
                                    "move ",
                                    "find ",
                                    "touch ",
                                    "select ",
                                    "collect ",
                                )
                                is_g = low_h.startswith(g_verbs) or any(
                                    f" {v}" in low_h for v in g_verbs
                                )
                                tf = (
                                    comp._resolve_feature_for_label(clean_hint, th)
                                    if is_g
                                    else None
                                )
                                d_priors = {}
                                if isinstance(tf, int):
                                    d_priors[tf] = {"goal": 0.96}
                                    if th is not None:
                                        for gf in list(th.goal_features):
                                            if gf != tf:
                                                th.retract_goal(gf, reason="mission_override")
                                        th.induce_candidate_goal(tf, evidence="grounded_dialogue")
                                    if grid_data is not None:
                                        arr = np.asarray(grid_data)
                                        if arr.ndim == 2:
                                            for v_np, c_np in zip(
                                                *np.unique(arr, return_counts=True), strict=False
                                            ):
                                                v_i = int(v_np)
                                                if (
                                                    v_i != tf
                                                    and v_i != bg_f
                                                    and v_i not in av_s
                                                    and int(c_np) <= 4
                                                    and (th is None or not th.is_barrier(v_i))
                                                ):
                                                    d_priors[v_i] = {"hazard": 0.78}
                                elif th is not None:
                                    comp.compile_utterance_to_causal_hypothesis(
                                        clean_hint, theory=th
                                    )
                                wm_ref.compiled_linguistic_hints.add(clean_hint)
                            else:
                                d_priors = {}
                            for av_f in av_s:
                                d_priors.pop(av_f, None)
                                merged_priors.pop(av_f, None)
                            for rc_d, rp_d in d_priors.items():
                                dst_d = merged_priors.setdefault(rc_d, {})
                                dst_d.update(rp_d)
                    if bridge or merged_priors:
                        wm_ref.concept_role_priors = merged_priors

                if self.engine is not None:
                    current_ctx = self._run_perception_faculty_hooks(current_ctx, extracted)
                stage_timings_ms["perception_hooks"] = round(
                    (time.perf_counter() - t_hook0) * 1000.0, 4
                )

            # Stage 2: SGE Assimilation Bridge Hook & Goal Arbitration (WP10, Gap 5)
            elif stage_name == "assimilation_stage":
                if bridge:
                    t_ab0 = time.perf_counter()
                    states = current_ctx.signals.get("sge_entity_states", {})
                    a_signals = bridge.process_assimilation(
                        current_ctx,
                        current_entity_states=states if isinstance(states, dict) else {},
                    )
                    current_ctx = current_ctx.with_signals(a_signals)
                    if (
                        self.engine is not None
                        and getattr(self.engine, "symbolic_theory", None) is not None
                        and hasattr(bridge, "sync_theory_roles")
                    ):
                        hz = getattr(self.engine, "hazard_tracker", None)
                        hz_feats = (
                            set(getattr(hz, "known_lethal_features", ()) or ())
                            if hz is not None
                            else set()
                        )
                        bridge.sync_theory_roles(
                            self.engine.symbolic_theory,
                            hazard_features=hz_feats,
                        )
                    stage_timings_ms["assimilation_bridge"] = round(
                        (time.perf_counter() - t_ab0) * 1000.0, 4
                    )

            # WP1: each stage's journaled faculty writes commit before the next stage.
            if getattr(result, "journal", None) is not None and self.engine is not None:
                t_j0 = time.perf_counter()
                committed = self.engine.commit_faculty_journal(result.journal)
                stage_timings_ms["journal_commit_ms"] = round(
                    stage_timings_ms.get("journal_commit_ms", 0.0)
                    + (time.perf_counter() - t_j0) * 1000.0,
                    4,
                )
                if not committed:
                    total_ms = round((time.perf_counter() - t_run0) * 1000.0, 4)
                    self.last_stage_timings_ms = stage_timings_ms
                    self.last_duration_ms = total_ms
                    return PipelineResult(
                        final_action=None,
                        stage_results=stage_results,
                        signals=dict(current_ctx.signals),
                        operations=all_ops,
                        rejected=True,
                        rejection_reason=f"{result.author}: journal commit rejected",
                        stage_timings_ms=stage_timings_ms,
                        total_duration_ms=total_ms,
                    )

        # Stage 5: SGE Motor Gateway Authorization Hook
        if bridge and final_action is not None:
            act = final_action[0] if isinstance(final_action, tuple) else final_action
            act_data = (
                final_action[1]
                if isinstance(final_action, tuple)
                and len(final_action) > 1
                and isinstance(final_action[1], dict)
                else {}
            )
            auth_act, rej_info = bridge.process_motor(
                current_ctx, proposed_action=act, action_data=act_data
            )
            if rej_info and rej_info.get("action_rejected_by_gateway"):
                total_ms = round((time.perf_counter() - t_run0) * 1000.0, 4)
                self.last_stage_timings_ms = stage_timings_ms
                self.last_duration_ms = total_ms
                return PipelineResult(
                    final_action=None,
                    stage_results=stage_results,
                    signals=dict(current_ctx.signals),
                    operations=all_ops,
                    rejected=True,
                    rejection_reason=f"ActionValidationGateway rejected: {rej_info.get('rejection_reasons')}",
                    stage_timings_ms=stage_timings_ms,
                    total_duration_ms=total_ms,
                )

        # If a transaction manager is attached, batch-commit all operations in bounded chunks
        if self.transaction_manager is not None and all_ops:
            t_tx0 = time.perf_counter()
            max_batch = 250
            for i in range(0, len(all_ops), max_batch):
                batch_ops = all_ops[i : i + max_batch]
                tx = HCIRTransaction(author="cortical_pipeline", operations=batch_ops)
                try:
                    self.transaction_manager.commit(tx)
                except Exception as exc:
                    total_ms = round((time.perf_counter() - t_run0) * 1000.0, 4)
                    self.last_stage_timings_ms = stage_timings_ms
                    self.last_duration_ms = total_ms
                    return PipelineResult(
                        final_action=None,
                        stage_results=stage_results,
                        signals=dict(current_ctx.signals),
                        operations=all_ops,
                        rejected=True,
                        rejection_reason=str(exc),
                        stage_timings_ms=stage_timings_ms,
                        total_duration_ms=total_ms,
                    )
            stage_timings_ms["tx_commit"] = round((time.perf_counter() - t_tx0) * 1000.0, 4)

        total_ms = round((time.perf_counter() - t_run0) * 1000.0, 4)
        self.last_stage_timings_ms = stage_timings_ms
        self.last_duration_ms = total_ms
        return PipelineResult(
            final_action=final_action,
            stage_results=stage_results,
            signals=dict(current_ctx.signals),
            operations=all_ops,
            stage_timings_ms=stage_timings_ms,
            total_duration_ms=total_ms,
        )

    def _run_perception_faculty_hooks(
        self,
        current_ctx: CognitiveContext,
        extracted: Any,
    ) -> CognitiveContext:
        """Execute G8 language grounding, G10 social cognition, and G11 option discovery."""
        import numpy as np

        engine = self.engine
        if engine is None:
            return current_ctx

        theory = engine.symbolic_theory
        wm = engine.working_memory

        # G8: Compile natural-language rule hints into CausalPredicates
        obs_meta = wm.last_observation_metadata
        if isinstance(obs_meta, dict):
            rule_hint = obs_meta.get("rule_hint")
            if isinstance(rule_hint, str) and rule_hint.strip():
                clean_rule = rule_hint.strip()
                if clean_rule not in wm.compiled_linguistic_hints:
                    from hbllm.hcir.compiler import HCIRCompiler

                    HCIRCompiler().compile_utterance_to_causal_hypothesis(clean_rule, theory=theory)
                    wm.compiled_linguistic_hints.add(clean_rule)

        raw_grid = (
            current_ctx.curr_grid
            if current_ctx.curr_grid is not None
            else current_ctx.signals.get("_curr_grid")
        )
        grid_arr = np.asarray(raw_grid) if raw_grid is not None else None
        has_2d_grid = grid_arr is not None and grid_arr.ndim == 2

        # G11: Topological Bottleneck & Option Discovery (gated off by default on live steps)
        opt_eng = engine.option_discovery
        if (
            opt_eng is not None
            and bool(opt_eng.live_pipeline_enabled)
            and has_2d_grid
            and grid_arr is not None
            and 4 <= grid_arr.size <= 256
        ):
            bg_val = int(engine.estimate_background(grid_arr))
            av_feats = set(engine.avatar_features or ())
            if engine.avatar_feature is not None:
                av_feats.add(engine.avatar_feature)
            passable_mask = grid_arr == bg_val
            if av_feats:
                passable_mask = passable_mask | np.isin(grid_arr, list(av_feats))
            grid_sig = (int(grid_arr.shape[0]), int(grid_arr.shape[1]), passable_mask.tobytes())
            if opt_eng.last_scanned_grid_sig != grid_sig:
                opt_eng.last_scanned_grid_sig = grid_sig
                # Treat avatar cells as traversable floor so static layout bottlenecks stay cached across steps
                topo_grid = np.where(passable_mask, bg_val, grid_arr) if av_feats else grid_arr
                bottlenecks = opt_eng.discover_bottlenecks_from_grid(topo_grid, walkable_val=bg_val)
                for b in bottlenecks:
                    opt_eng.induce_option(bottleneck=b)

            av_pos = engine.avatar_pos
            matched_opt = None
            if av_pos is not None:
                for opt in opt_eng.discovered_options.values():
                    if opt.bottleneck_state == av_pos:
                        matched_opt = opt
                        break
            if matched_opt is not None:
                engine.record_option_usage(matched_opt.option_id, steps=1, success=True)
            else:
                opt_eng.record_primitive_execution(steps=1)

        return current_ctx
