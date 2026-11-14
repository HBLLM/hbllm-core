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
        current_ctx = ctx
        stage_results: list[StageResult] = []
        all_ops: list[Any] = []
        final_action: Any = None
        bridge = self.grounding_bridge
        if self.engine is not None and hasattr(self.engine, "grounding_bridge"):
            eng_bridge = getattr(self.engine, "grounding_bridge", None)
            if eng_bridge is not None:
                bridge = eng_bridge

        for stage in self.stages:
            try:
                result = stage.observe(current_ctx)
            except Exception:
                cur_journal = getattr(stage, "_current_journal", None)
                if self.engine is not None and cur_journal is not None:
                    self.engine.rollback_faculty_journal(cur_journal)
                raise
            stage_results.append(result)
            all_ops.extend(result.operations)
            if result.signals:
                current_ctx = current_ctx.with_signals(result.signals)
            if result.proposed_action is not None:
                final_action = result.proposed_action

            stage_name = getattr(stage, "name", "")

            # Stage 1: SGE Perception Bridge + G8/G10/G11 Hooks
            if stage_name == "perception_stage":
                extracted = current_ctx.signals.get(
                    "extracted_entities"
                ) or current_ctx.signals.get("_objects")
                if bridge:
                    p_signals = bridge.process_perception(
                        current_ctx,
                        extracted_entities=extracted if isinstance(extracted, list) else None,
                    )
                    current_ctx = current_ctx.with_signals(p_signals)
                    if (
                        self.engine is not None
                        and getattr(self.engine, "working_memory", None) is not None
                        and "sge_role_priors" in p_signals
                    ):
                        self.engine.working_memory.concept_role_priors = dict(
                            p_signals["sge_role_priors"]
                        )

                if self.engine is not None:
                    current_ctx = self._run_perception_faculty_hooks(current_ctx, extracted)

            # Stage 2: SGE Assimilation Bridge Hook & Goal Arbitration (WP10, Gap 5)
            elif stage_name == "assimilation_stage":
                if bridge:
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

                # Goal Arbitration Hook: Selects active TaskFrame using Expected Free Energy
                if self.engine is not None and getattr(self.engine, "goal_arbiter", None):
                    candidates = self.engine.goal_arbiter.extract_candidate_goals(
                        self.engine, current_ctx
                    )
                    if candidates:
                        arb_result = self.engine.goal_arbiter.arbitrate(
                            candidates=candidates,
                            engine=self.engine,
                            homeostatic_state=getattr(self.engine, "homeostatic_state", None),
                            neuromodulator_system=getattr(
                                self.engine, "neuromodulatory_system", None
                            ),
                            tiered_workspace=getattr(self.engine, "tiered_workspace", None),
                        )
                        if arb_result is not None:
                            current_ctx = current_ctx.with_signals(
                                {
                                    "arbitrated_goal": arb_result.selected_goal,
                                    "arbitration_result": arb_result,
                                }
                            )
                            if hasattr(self.engine, "consume_energy"):
                                self.engine.consume_energy(0.1)

            # Stage 3: G15 Goal Provenance Hook
            elif stage_name == "simulation_stage" and self.engine is not None:
                gp = getattr(self.engine, "goal_provenance", None)
                if gp:
                    arb_goal = current_ctx.signals.get("arbitrated_goal")
                    active_goals = list(getattr(self.engine, "goals", ()) or ())
                    target_str = (
                        str(arb_goal)
                        if arb_goal is not None
                        else (str(active_goals[0]) if active_goals else None)
                    )
                    gp.register_goal(
                        goal_id=f"step_{current_ctx.step_counter}",
                        intent="achieve_arbitrated_goal" if arb_goal else "epistemic_explore",
                        target_entity=target_str,
                        justification="simulation_stage active inference selection",
                    )

            # Stage 4: G15 Value-Uncertainty Safety Evaluation Hook
            elif stage_name == "deliberation_stage" and self.engine is not None:
                arbiter = getattr(self.engine, "value_uncertainty_arbiter", None)
                if arbiter:
                    prop = result.signals.get("_proposal")
                    act = getattr(prop, "action", None) if prop is not None else final_action
                    if act is not None:
                        dyn = getattr(self.engine, "action_dynamics", {}) or {}
                        epistemic_var = 0.05 if act in dyn else 0.25
                        approved, req = self.engine.evaluate_action_value_safety(
                            action_id=act,
                            action_type=str(act),
                            expected_value=1.0,
                            epistemic_variance=epistemic_var,
                            is_hazard_cone=False,
                        )
                        current_ctx = current_ctx.with_signals(
                            {
                                "value_safety_approved": approved,
                                "value_deferral_reason": (
                                    req.deferral_reason.value if req is not None else None
                                ),
                            }
                        )

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

        theory = getattr(engine, "symbolic_theory", None)
        wm = getattr(engine, "working_memory", None)

        # G8: Compile any natural-language hint metadata into CausalPredicates
        if theory is not None and wm is not None:
            obs_meta = getattr(wm, "last_observation_metadata", None)
            if isinstance(obs_meta, dict):
                hint = (
                    obs_meta.get("rule_hint")
                    or obs_meta.get("instruction")
                    or obs_meta.get("mission")
                )
                if isinstance(hint, str) and hint.strip():
                    clean_hint = hint.strip()
                    compiled_set = getattr(wm, "compiled_linguistic_hints", None)
                    if not isinstance(compiled_set, set) or clean_hint not in compiled_set:
                        from hbllm.hcir.compiler import HCIRCompiler

                        compiler = HCIRCompiler(
                            concept_store=getattr(engine, "working_concept_store", None)
                        )
                        compiler.compile_utterance_to_causal_hypothesis(clean_hint, theory=theory)
                        if isinstance(compiled_set, set):
                            compiled_set.add(clean_hint)

        raw_grid = (
            current_ctx.curr_grid
            if current_ctx.curr_grid is not None
            else current_ctx.signals.get("_curr_grid")
        )
        grid_arr = np.asarray(raw_grid) if raw_grid is not None else None
        has_2d_grid = grid_arr is not None and grid_arr.ndim == 2

        # G10: Theory of Mind tracking for external moving / intentional entities
        soc = getattr(engine, "social_cognition", None)
        if soc and has_2d_grid and grid_arr is not None:
            mobile_feats = set(getattr(engine, "mobile_threat_features", ()) or ())
            av_feats = set(getattr(engine, "avatar_features", ()) or ())
            av_pos = getattr(engine, "avatar_pos", None)
            dorsal = getattr(engine, "dorsal_kinetic_stream", None)
            ext_agents = getattr(dorsal, "external_agents", {}) if dorsal else {}
            cand_goals = list(getattr(engine, "goals", ()) or ())

            predicted_hazards: list[tuple[int, int]] = []
            sge_states = current_ctx.signals.get("sge_entity_states")
            if isinstance(sge_states, dict) and sge_states:
                items = list(sge_states.items())
                small_scene = len(items) <= 8
                for ent_id, st in items:
                    pos = st.get("position")
                    feat = int(st.get("color", 0) or 0)
                    size = int(st.get("size", 1) or 1)
                    if not isinstance(pos, (tuple, list)) or len(pos) < 2:
                        continue
                    pos_2d = (int(round(pos[0])), int(round(pos[1])))
                    if pos_2d == av_pos or feat in av_feats:
                        continue
                    if theory is not None and theory.is_barrier(feat):
                        continue
                    is_dynamic = (
                        feat in mobile_feats
                        or ent_id in ext_agents
                        or (small_scene and size == 1 and bool(st.get("moved")))
                    )
                    if is_dynamic:
                        model = engine.track_and_infer_agent_intents(
                            entity_id=str(ent_id),
                            feature_id=feat,
                            position=pos_2d,
                            avatar_pos=av_pos,
                            candidate_goals=cand_goals,
                            grid=grid_arr,
                        )
                        if (
                            getattr(model, "agency_status", None) == "INTENTIONAL_AGENT"
                            or getattr(model, "inferred_intent", None) == "PURSUING_AVATAR"
                        ):
                            predicted_hazards.extend(
                                soc.project_intent_trajectory(str(ent_id), horizon=3, grid=grid_arr)
                            )
            elif isinstance(extracted, list) and extracted:
                small_scene = len(extracted) <= 8
                for ent in extracted:
                    props = getattr(ent, "properties", None)
                    feat_raw = getattr(ent, "feature_id", None) or getattr(ent, "color", None)
                    if feat_raw is None and isinstance(props, dict):
                        feat_raw = props.get("feature_id") or props.get("color")
                    elif feat_raw is None and isinstance(ent, dict):
                        feat_raw = ent.get("feature_id") or ent.get("color")
                    if feat_raw is None:
                        continue
                    feat = int(feat_raw)
                    pos = getattr(ent, "grid_pos", None) or getattr(ent, "centroid", None)
                    if pos is None and isinstance(ent, dict):
                        pos = ent.get("position")
                    if not isinstance(pos, (tuple, list)) or len(pos) < 2:
                        continue
                    pos_2d = (int(round(pos[0])), int(round(pos[1])))
                    if pos_2d == av_pos or feat in av_feats:
                        continue
                    if theory is not None and theory.is_barrier(feat):
                        continue
                    area = int(
                        getattr(ent, "area", 1) if not isinstance(ent, dict) else ent.get("size", 1)
                    )
                    ent_id = str(
                        getattr(ent, "id", None)
                        or (ent.get("id") if isinstance(ent, dict) else None)
                        or f"agent_f{feat}"
                    )
                    if feat in mobile_feats or ent_id in ext_agents or (small_scene and area == 1):
                        model = engine.track_and_infer_agent_intents(
                            entity_id=ent_id,
                            feature_id=feat,
                            position=pos_2d,
                            avatar_pos=av_pos,
                            candidate_goals=cand_goals,
                            grid=grid_arr,
                        )
                        if (
                            getattr(model, "agency_status", None) == "INTENTIONAL_AGENT"
                            or getattr(model, "inferred_intent", None) == "PURSUING_AVATAR"
                        ):
                            predicted_hazards.extend(
                                soc.project_intent_trajectory(ent_id, horizon=3, grid=grid_arr)
                            )

            if predicted_hazards:
                current_ctx = current_ctx.with_signals(
                    {"social_predicted_hazards": predicted_hazards}
                )

        # G11: Topological Bottleneck & Option Discovery
        opt_eng = getattr(engine, "option_discovery", None)
        if opt_eng and has_2d_grid and grid_arr is not None and 4 <= grid_arr.size <= 256:
            grid_sig = (int(grid_arr.shape[0]), int(grid_arr.shape[1]), grid_arr.tobytes())
            if getattr(opt_eng, "last_scanned_grid_sig", None) != grid_sig:
                opt_eng.last_scanned_grid_sig = grid_sig
                bg_val = int(engine.estimate_background(grid_arr))
                bottlenecks = opt_eng.discover_bottlenecks_from_grid(grid_arr, walkable_val=bg_val)
                for b in bottlenecks:
                    opt_eng.induce_option(bottleneck=b)

            av_pos = getattr(engine, "avatar_pos", None)
            matched_opt = None
            if av_pos is not None:
                for opt in getattr(opt_eng, "discovered_options", {}).values():
                    if getattr(opt, "bottleneck_state", None) == av_pos:
                        matched_opt = opt
                        break
            if matched_opt is not None:
                engine.record_option_usage(matched_opt.option_id, steps=1, success=True)
                lib = getattr(engine, "library_learner", None)
                if lib is not None:
                    opt_eng.consolidate_options_to_library(lib)
            else:
                opt_eng.record_primitive_execution(steps=1)

        return current_ctx
