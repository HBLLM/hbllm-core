from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.hcir.grounding.actions.gateway import ActionValidationGateway
from hbllm.hcir.grounding.actions.renderers import (
    ActionRenderer,
    DialogueRenderer,
    GuiRenderer,
    KinematicRenderer,
)
from hbllm.hcir.grounding.bus.observation_bus import ObservationBus
from hbllm.hcir.grounding.bus.policies import ChannelPolicy
from hbllm.hcir.grounding.causal.schema import SchemaInducer
from hbllm.hcir.grounding.committer.committer import HCIRCommitter
from hbllm.hcir.grounding.concepts.concept_learner import ConceptAssignment, ConceptLearner
from hbllm.hcir.grounding.concepts.store import WorkingConceptStore
from hbllm.hcir.grounding.contracts.actions import (
    ActionExecutionRecord,
    ActionValidationResult,
    SymbolicActionIntent,
)
from hbllm.hcir.grounding.contracts.dialogue import GoalProposal
from hbllm.hcir.grounding.contracts.enums import (
    ActionLifecycleState,
    DecisionOutcome,
    ModalitySource,
    TrafficClass,
)
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.contracts.references import (
    EntityCandidateRef,
    GroundedEntityRef,
    TrackletRef,
)
from hbllm.hcir.grounding.epistemics.evaluator import EpistemicEvaluator
from hbllm.hcir.grounding.features.feature_extractor import (
    FeatureExtractorRegistry,
    PerceptualFeatureVector,
)
from hbllm.hcir.grounding.predictive.assimilator import (
    GroundingSurpriseAlert,
    GroundingSurpriseAssimilator,
)
from hbllm.hcir.grounding.predictive.feedback import ActiveInferenceFeedback
from hbllm.hcir.grounding.predictive.predictor import (
    ForwardWorldPredictor,
    PredictedEntityState,
)
from hbllm.hcir.grounding.relations.relation_learner import RelationLearner
from hbllm.hcir.grounding.tracking.tracker import ThreeTierEntityTracker
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.percept import RetentionPolicy, apply_boundary_retention

if TYPE_CHECKING:
    from hbllm.hcir.kernel.services import KernelServices
    from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
    from hbllm.hcir.world.engine_state import CognitiveContext
    from hbllm.hcir.world.surprise_engine import SurpriseEngine

logger = logging.getLogger(__name__)


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class CorticalGroundingState:
    """Working state maintained by the CorticalGroundingBridge across cognitive cycles."""

    FIELD_RETENTION = {
        "active_predictions": RetentionPolicy.LEVEL_LOCAL,
        "last_surprise_alerts": RetentionPolicy.LEVEL_LOCAL,
        "pending_epistemic_goals": RetentionPolicy.LEVEL_LOCAL,
    }

    step_counter: int = 0
    active_predictions: dict[str, PredictedEntityState] = field(default_factory=dict)
    last_dispatched_intent: SymbolicActionIntent | None = None
    last_execution_record: ActionExecutionRecord | None = None
    last_surprise_alerts: list[GroundingSurpriseAlert] = field(default_factory=list)
    pending_epistemic_goals: list[GoalProposal] = field(default_factory=list)
    last_feature_vectors: list[PerceptualFeatureVector] = field(default_factory=list)
    last_concept_assignments: list[ConceptAssignment] = field(default_factory=list)
    verified_outcomes_count: int = 0
    rejected_actions_count: int = 0


class CorticalGroundingBridge:
    """The authoritative bridge integrating SGE into the Cortical Epistemic Loop.

    Upholds Invariants 1, 3, 4, 7, and 8:
    1. SGE -> Cortical Perception: Converts sensory arrays into timestamped ObservationEnvelopes
       on the ObservationBus; tracks persistent candidate identities via ThreeTierEntityTracker.
    2. SGE -> Assimilation: Evaluates predictions vs observations via GroundingSurpriseAssimilator;
       modulates SurpriseEngine, degrades fragile causal schemas, and triggers epistemic goals.
    3. Cortical Motor -> ActionValidationGateway: Intercepts all proposed actions, enforces
       precondition & safety authorization, dispatches, and verifies real environmental outcomes.
    4. General Intelligence: Task-agnostic, LLM-independent, and transactionally governed.
    """

    FIELD_RETENTION = {
        "active_predictions": RetentionPolicy.LEVEL_LOCAL,
        "last_surprise_alerts": RetentionPolicy.LEVEL_LOCAL,
        "pending_epistemic_goals": RetentionPolicy.LEVEL_LOCAL,
    }

    def __init__(
        self,
        workspace: HCIRWorkspaceState | None = None,
        transaction_manager: TransactionManager | None = None,
        symbolic_theory: HCIRSymbolicWorldTheory | None = None,
        surprise_engine: SurpriseEngine | None = None,
        acceptance_threshold: float = 0.65,
        surprise_threshold: float = 0.35,
        services: KernelServices | None = None,
    ) -> None:
        self.services = services
        self.workspace = (
            workspace
            or (services.workspace if services is not None else None)
            or HCIRWorkspaceState()
        )
        self.tx_manager = (
            transaction_manager
            or (services.transaction_manager if services is not None else None)
            or TransactionManager(self.workspace)
        )
        self.symbolic_theory = symbolic_theory
        self.surprise_engine = surprise_engine

        # SGE Core Subsystems
        self.bus = ObservationBus()
        self.perception_channel = self.bus.register_channel(
            "cortical_perception",
            ChannelPolicy.default_for(TrafficClass.CLASS_HIGH_FREQ),
            TrafficClass.CLASS_HIGH_FREQ,
        )
        self.tracker = ThreeTierEntityTracker(spatial_distance_threshold=1.5)
        self.concept_store = WorkingConceptStore()
        self.concept_learner = ConceptLearner(concept_store=self.concept_store)
        self.feature_registry = FeatureExtractorRegistry.default()
        self.relation_learner = RelationLearner(proximity_threshold=3.0)
        self.evaluator = EpistemicEvaluator(acceptance_threshold=acceptance_threshold)
        self.committer = HCIRCommitter(self.tx_manager)
        self.schema_inducer = SchemaInducer()
        self.predictor = ForwardWorldPredictor()
        self.assimilator = GroundingSurpriseAssimilator(surprise_threshold=surprise_threshold)
        self.feedback = ActiveInferenceFeedback(schema_inducer=self.schema_inducer)
        self.gateway = ActionValidationGateway()
        self.renderer: ActionRenderer = KinematicRenderer()
        self.gui_renderer = GuiRenderer()
        self.dialogue_renderer = DialogueRenderer()

        self.state = CorticalGroundingState()
        self._committed_beliefs: dict[str, tuple[str, float]] = {}

    def boundary_retention(self, boundary: Any) -> None:
        """Apply boundary retention to grounding state on episode transitions."""
        apply_boundary_retention(self.state, boundary)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: SGE -> Cortical Perception
    # ─────────────────────────────────────────────────────────────────────────

    def process_perception(
        self,
        ctx: CognitiveContext | None = None,
        extracted_entities: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Stage 1: Converts incoming observation into SGE envelope and updates entity tracking."""
        self.state.step_counter += 1
        now = time.time()
        step = self.state.step_counter

        # 1. Ingest into ObservationBus
        obs_id = f"obs_cortex_s{step}_{uuid.uuid4().hex[:6]}"
        envelope = ObservationEnvelope(
            observation_id=obs_id,
            source_modality=ModalitySource.VISION,
            source_channel_id="cortical_perception",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=step,
            wall_timestamp=now,
            monotonic_timestamp=time.monotonic(),
            payload_ref=f"hcir://cortex/grid/{step}",
            clock_domain="cortex_clock",
            metadata={"step": step, "entity_count": len(extracted_entities or [])},
        )
        self.perception_channel.ingest(envelope)

        # 2. Update Entity Tracking (Tracklet -> EntityCandidateRef)
        entity_states: dict[str, dict[str, Any]] = {}
        entities = extracted_entities or []

        # Find existing active candidates to enable permutation-invariant tracking
        active_cands = self.tracker.get_active_candidates()
        matched_cand_ids: set[str] = set()

        for idx, ent in enumerate(entities):
            # Extract coordinates, role, and identity from SpatialEntity or dict
            pos: tuple[float, ...] = (0.0, 0.0)
            color: int = 1
            bbox: tuple[float, float, float, float] | None = None
            role: str = "entity"
            ent_id: str | None = None

            if hasattr(ent, "id"):
                ent_id = str(ent.id)
            elif hasattr(ent, "entity_id"):
                ent_id = str(ent.entity_id)

            if hasattr(ent, "coords"):
                coords = list(ent.coords)
                if coords:
                    r_coords = [c[0] for c in coords]
                    c_coords = [c[1] for c in coords]
                    min_r, max_r = min(r_coords), max(r_coords)
                    min_c, max_c = min(c_coords), max(c_coords)
                    pos = (float(sum(r_coords)) / len(coords), float(sum(c_coords)) / len(coords))
                    bbox = (
                        float(min_r),
                        float(min_c),
                        float(max_r - min_r + 1),
                        float(max_c - min_c + 1),
                    )
                color = getattr(ent, "color", 1)
                role = str(getattr(ent, "role", "entity"))
            elif isinstance(ent, dict):
                ent_id = ent.get("id") or ent_id
                pos = tuple(float(x) for x in ent.get("position", (0.0, 0.0)))
                color = int(ent.get("color", 1))
                role = str(ent.get("role", "entity"))
                b = ent.get("bbox")
                if b and len(b) == 4:
                    bbox = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))

            # Determine stable tracklet ID (WP3: permutation-invariant association)
            if ent_id:
                tracklet_id = TrackletRef(f"trk_{ent_id}")
            else:
                best_trk_id: str | None = None
                best_dist = float("inf")
                for c in active_cands:
                    cid = c.candidate_ref.id
                    if cid in matched_cand_ids:
                        continue
                    if c.spatial_trajectory:
                        last_p = c.spatial_trajectory[-1]
                        dist = math.sqrt((pos[0] - last_p[0]) ** 2 + (pos[1] - last_p[1]) ** 2)
                        if dist < best_dist and dist <= self.tracker._spatial_distance_threshold:
                            best_dist = dist
                            if c.associated_tracklet_ids:
                                best_trk_id = next(iter(c.associated_tracklet_ids))
                                matched_cand_ids.add(cid)

                if best_trk_id is not None:
                    tracklet_id = TrackletRef(best_trk_id)
                else:
                    tracklet_id = TrackletRef(_gen_id(f"trk_{color}"))

            self.tracker.update_tracklet(
                tracklet_id,
                channel_id="cortical_perception",
                spatial_coords=pos,
                metadata={"color": color, "role": role, "bbox": bbox},
            )
            cand = self.tracker.auto_associate(tracklet_id, spatial_coords=pos)
            matched_cand_ids.add(cand.id)
            entity_states[cand.id] = {
                "position": pos,
                "bbox": bbox,
                "color": color,
                "role": role,
            }

        # 3. Dynamic Relation Induction
        self.relation_learner.observe_entity_states(entity_states, timestamp=now)
        claims, ev_records = self.relation_learner.propose_claims(
            source_channel_id="cortical_perception",
            observation_id=obs_id,
            dependency_group=f"dep_cortex_s{step}",
        )

        # 4. Epistemic Evaluation & Commit of Accepted Relations (bounded growth check)
        committed_claims: list[str] = []
        for claim in claims:
            # Match supporting evidence
            claim_ev = [e for e in ev_records if e.evidence_id in claim.evidence_ids]
            decision, belief = self.evaluator.evaluate_claim(claim, claim_ev)
            if decision.outcome == DecisionOutcome.ACCEPT:
                b_key = f"{claim.subject_ref.id}:{claim.predicate_ref}:{claim.object_ref}"
                prev = self._committed_beliefs.get(b_key)
                if (
                    prev is not None
                    and prev[0] == claim.polarity.value
                    and abs(decision.posterior_belief - prev[1]) < 0.05
                ):
                    continue  # Bounded state growth: skip redundant transaction
                try:
                    self.committer.commit_decision(decision, claim, belief)
                    self._committed_beliefs[b_key] = (
                        claim.polarity.value,
                        decision.posterior_belief,
                    )
                    committed_claims.append(claim.claim_id)
                except Exception as exc:
                    logger.debug("Failed to commit claim %s: %s", claim.claim_id, exc)

        active_cands = self.tracker.get_active_candidates()

        # 5. Feature Extraction via FeatureExtractorRegistry
        if entity_states:
            envelope.metadata["entities"] = [
                {"id": cand_id, **state} for cand_id, state in entity_states.items()
            ]

        feature_vectors: list[PerceptualFeatureVector] = []
        try:
            extracted = self.feature_registry.extract_all(envelope)
            feature_vectors.extend(extracted)
        except Exception as exc:
            logger.debug("FeatureExtractorRegistry.extract_all failed: %s", exc)

        # 6. Concept Learning via ConceptLearner
        concept_assignments: list[ConceptAssignment] = []
        for fv in feature_vectors:
            try:
                entity_id: str | None = fv.feature_metadata.get("entity_id")
                if not entity_id and active_cands:
                    entity_id = active_cands[0].candidate_ref.id
                assignment = self.concept_learner.learn(fv, exemplar_entity_id=entity_id)
                concept_assignments.append(assignment)
            except Exception as exc:
                logger.debug("ConceptLearner.learn failed for %s: %s", fv.feature_id, exc)

        # Run concept maintenance (split/merge detection) periodically
        split_count = 0
        merge_count = 0
        if self.state.step_counter % 5 == 0 and self.state.step_counter > 0:
            try:
                splits, merges = self.concept_learner.maintain()
                split_count = len(splits)
                merge_count = len(merges)
            except Exception as exc:
                logger.debug("ConceptLearner.maintain failed: %s", exc)

        # Promote eligible concepts to authoritative HCIR state (WP3: grounded concept promotion)
        promoted_concepts: list[str] = []
        for concept in self.concept_store.get_all():
            if self.concept_store.lifecycle_manager.should_propose_promotion(concept):
                cid = concept.concept_ref.id
                if not self.workspace.graph.has_node(cid):
                    prop_id = f"prop_{cid}_{uuid.uuid4().hex[:6]}"
                    proposal = self.concept_store.lifecycle_manager.create_promotion_proposal(
                        concept=concept,
                        proposal_id=prop_id,
                    )
                    try:
                        self.committer.commit_concept_promotion(proposal, concept)
                        promoted_concepts.append(cid)
                    except Exception as exc:
                        logger.debug("Failed to commit concept promotion for %s: %s", cid, exc)

        self.state.last_feature_vectors = feature_vectors
        self.state.last_concept_assignments = concept_assignments

        # 7. Post-Action Outcome Verification (from previous cycle)
        outcome_verified = False
        if self.state.last_execution_record is not None and entity_states:
            try:
                verified = self.verify_post_action_outcome(entity_states)
                outcome_verified = verified is not None and getattr(
                    verified, "outcome_verified", False
                )
            except Exception as exc:
                logger.debug("Post-action outcome verification failed: %s", exc)

        return {
            "sge_observation_id": obs_id,
            "sge_active_candidates": [c.candidate_ref.id for c in active_cands],
            "sge_entity_states": entity_states,
            "sge_proposed_claims_count": len(claims),
            "sge_committed_claims_count": len(committed_claims),
            "sge_feature_vectors_count": len(feature_vectors),
            "sge_concept_assignments": [
                {
                    "concept_id": a.concept_ref.id,
                    "distance": a.distance,
                    "is_new": a.is_new_concept,
                }
                for a in concept_assignments
            ],
            "sge_concept_splits": split_count,
            "sge_concept_merges": merge_count,
            "sge_outcome_verified": outcome_verified,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2: SGE -> Assimilation & Surprise Induction
    # ─────────────────────────────────────────────────────────────────────────

    def process_assimilation(
        self,
        ctx: CognitiveContext | None = None,
        current_entity_states: dict[str, dict[str, Any]] | None = None,
        terminal: Any = None,
    ) -> dict[str, Any]:
        """Stage 2: Compares forward predictions with actual observations to detect surprise."""
        alerts: list[GroundingSurpriseAlert] = []
        epistemic_goals: list[GoalProposal] = []
        surprise_score = 0.0

        entity_states = current_entity_states or {}
        # If previous predictions exist, assimilate surprise
        if self.state.active_predictions:
            alerts = self.assimilator.evaluate_observation(
                predicted_states=self.state.active_predictions,
                observed_entities=entity_states,
            )
            self.state.last_surprise_alerts = alerts

            for alert in alerts:
                surprise_score = max(surprise_score, alert.surprise_score)
                # Active inference feedback degrades fragile schemas & emits curiosity goals
                goal = self.feedback.process_surprise_alert(alert)
                if goal is not None:
                    epistemic_goals.append(goal)
                    self.state.pending_epistemic_goals.append(goal)

                # Feed surprise into cortical SurpriseEngine unconditionally (WP3: connected surprise path)
                if self.surprise_engine is not None:
                    try:
                        self.surprise_engine.ingest_external_prediction_error(
                            score=alert.surprise_score,
                            source="grounding",
                            is_surprising=alert.surprise_score
                            >= self.assimilator._surprise_threshold,
                        )
                    except Exception as exc:
                        logger.debug(
                            "SurpriseEngine ingest_external_prediction_error failed: %s", exc
                        )

        # Route epistemic curiosity goals into PrefrontalWorkingMemory if available
        if epistemic_goals and self.services is not None:
            wm = getattr(self.services, "working_memory", None)
            if wm is not None and hasattr(wm, "pending_epistemic_goals"):
                wm.pending_epistemic_goals.extend(epistemic_goals)

        # Clear predictions once assimilated
        self.state.active_predictions.clear()

        return {
            "sge_surprise_alerts": len(alerts),
            "sge_max_surprise": surprise_score,
            "sge_epistemic_goals": [g.goal_intent for g in epistemic_goals],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: Cortical Motor -> ActionValidationGateway
    # ─────────────────────────────────────────────────────────────────────────

    def process_motor(
        self,
        ctx: CognitiveContext | None = None,
        proposed_action: Any = None,
        action_data: dict[str, Any] | None = None,
        world_state: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any] | None]:
        """Stage 3: Validates proposed cortical action through ActionValidationGateway."""
        if proposed_action is None:
            return None, None

        act_type = str(proposed_action)
        params = dict(action_data or {})
        target_id = params.get("target_id")

        effective_world_state = world_state or {}
        if "tracked_entities" not in effective_world_state:
            active_cands = self.tracker.get_active_candidates()
            tracked: dict[str, Any] = {}
            for c in active_cands:
                pos = c.spatial_trajectory[-1] if c.spatial_trajectory else (0.0, 0.0)
                info = {"confidence": 0.95, "position": pos}
                tracked[c.candidate_ref.id] = info
                if c.grounded_entity_ref:
                    tracked[c.grounded_entity_ref.id] = info
                for trk_id in c.associated_tracklet_ids:
                    tracked[trk_id] = info
                    if trk_id.startswith("trk_"):
                        raw_id = trk_id[4:]
                        tracked[raw_id] = info
                        if not raw_id.startswith("e_"):
                            tracked[f"e_{raw_id}"] = info
            if "e_avatar" not in tracked:
                tracked["e_avatar"] = {"confidence": 1.0, "position": (0.0, 0.0)}
            effective_world_state["tracked_entities"] = tracked

        if not target_id:
            # Fallback to first tracked entity or default self-avatar
            active = self.tracker.get_active_candidates()
            target_id = active[0].candidate_ref.id if active else "e_avatar"

        if target_id.startswith("ecand_"):
            target_ref = EntityCandidateRef(target_id)
        elif target_id.startswith("e_"):
            target_ref = GroundedEntityRef(target_id)
        else:
            target_ref = GroundedEntityRef(f"e_{target_id}")

        # Ensure target_ref is present in tracked_entities for gateway verification
        if target_ref.id not in effective_world_state["tracked_entities"]:
            # Check if underlying id is present
            if target_id in effective_world_state["tracked_entities"]:
                effective_world_state["tracked_entities"][target_ref.id] = effective_world_state[
                    "tracked_entities"
                ][target_id]

        intent = SymbolicActionIntent(
            intent_id=f"intent_act_{self.state.step_counter}_{uuid.uuid4().hex[:4]}",
            action_type=act_type,
            target_ref=target_ref,
            parameters=params,
            preconditions=tuple(params.get("preconditions", ())),
            expected_outcomes=tuple(params.get("expected_outcomes", ())),
            context={"step": self.state.step_counter},
        )

        renderer: ActionRenderer = self.renderer
        if self.gui_renderer.can_render(intent):
            renderer = self.gui_renderer
        elif self.dialogue_renderer.can_render(intent):
            renderer = self.dialogue_renderer

        payload = renderer.render(
            intent,
            context={
                "speed": params.get("speed", 1.0),
                "target_coordinates": params.get("target_coordinates"),
            },
        )

        # Execute Gateway Authorization
        val_result: ActionValidationResult = self.gateway.validate_action(
            intent=intent,
            payload=payload,
            world_state=effective_world_state,
        )

        if not val_result.is_authorized:
            self.state.rejected_actions_count += 1
            logger.warning(
                "Action %s rejected by ActionValidationGateway: %s",
                act_type,
                val_result.rejection_reasons,
            )
            return None, {
                "action_rejected_by_gateway": True,
                "rejection_reasons": list(val_result.rejection_reasons),
            }

        # Dispatch
        exec_record = self.gateway.dispatch_action(intent, payload, val_result)
        self.state.last_dispatched_intent = intent
        self.state.last_execution_record = exec_record

        # Generate forward predictions for next cycle assimilation (WP3: velocity-aware kinematic simulation)
        active_entities = {}
        for c in self.tracker.get_active_candidates():
            traj = c.spatial_trajectory
            pos = traj[-1] if traj else (0.0, 0.0)
            if len(traj) >= 2:
                vel = (traj[-1][0] - traj[-2][0], traj[-1][1] - traj[-2][1])
            else:
                vel = (0.0, 0.0)
            active_entities[c.candidate_ref.id] = {
                "ref": c.candidate_ref,
                "position": pos,
                "velocity": vel,
            }
        self.state.active_predictions = self.predictor.predict_next_state(
            current_entities=active_entities,
            action_type=act_type,
            target_entity_ref=target_ref,
        )

        return proposed_action, None

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4: Outcome Verification
    # ─────────────────────────────────────────────────────────────────────────

    def verify_post_action_outcome(
        self,
        post_observation: dict[str, Any],
        driver_feedback: Any | None = None,
    ) -> ActionExecutionRecord | None:
        """Verifies whether the dispatched action actually produced the expected real-world outcome."""
        if not self.state.last_execution_record or not self.state.last_dispatched_intent:
            return None

        success = True
        if driver_feedback is not None:
            success = bool(getattr(driver_feedback, "success", True))
        elif isinstance(post_observation, dict) and "success" in post_observation:
            success = bool(post_observation["success"])

        # Report execution completed with actual feedback
        reported = self.gateway.report_execution(self.state.last_execution_record, success=success)
        # Verify post-observation against expectations
        verified = self.gateway.verify_outcome(
            record=reported,
            intent=self.state.last_dispatched_intent,
            post_observation=post_observation,
        )

        if verified.state == ActionLifecycleState.COMPLETED and verified.outcome_verified:
            self.state.verified_outcomes_count += 1

        self.state.last_execution_record = verified
        return verified
