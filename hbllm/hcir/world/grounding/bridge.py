"""Cortical Grounding Bridge — Integrates SGE Grounding into the Cortical Epistemic Loop."""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np

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
from hbllm.hcir.grounding.contracts.dialogue import DialogueInterpretationResult, GoalProposal
from hbllm.hcir.grounding.contracts.enums import (
    ActionLifecycleState,
    DecisionOutcome,
    ModalitySource,
    TrafficClass,
)
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.contracts.quality import GroundingQuality
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
)
from hbllm.hcir.grounding.relations.relation_learner import RelationLearner
from hbllm.hcir.grounding.tracking.tracker import ThreeTierEntityTracker
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.world.grounding.state import CorticalGroundingState, _gen_id
from hbllm.hcir.world.percept import RetentionPolicy, apply_boundary_retention

if TYPE_CHECKING:
    from hbllm.hcir.kernel.services import KernelServices
    from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory
    from hbllm.hcir.world.engine_state import CognitiveContext
    from hbllm.hcir.world.surprise_engine import SurpriseEngine

logger = logging.getLogger(__name__)


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
        "active_predictions": RetentionPolicy.EPISODE_LOCAL,
        "last_surprise_alerts": RetentionPolicy.EPISODE_LOCAL,
        "pending_epistemic_goals": RetentionPolicy.EPISODE_LOCAL,
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
        perception_mode: str = "handcrafted",
        enable_fuzzy_grounding: bool = False,
    ) -> None:
        self.services = services
        self.perception_mode = perception_mode
        self.enable_fuzzy_grounding = bool(enable_fuzzy_grounding)
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
        self.concept_learner = ConceptLearner(
            concept_store=self.concept_store,
            assignment_threshold=0.32,
            split_dispersion_threshold=0.48,
            merge_distance_threshold=0.14,
            enable_fuzzy=self.enable_fuzzy_grounding,
        )
        self.feature_registry = FeatureExtractorRegistry.default(perception_mode=perception_mode)
        self.relation_learner = RelationLearner(
            proximity_threshold=3.0, enable_fuzzy=self.enable_fuzzy_grounding
        )
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
        self.state.last_role_priors.clear()
        self.state.dialogue_role_priors.clear()
        self.state.last_scene_sig = None
        self.state.episode_feature_memory.clear()
        self.state.cached_entity_features.clear()
        self.state.cached_entity_concepts.clear()
        self.state.cached_entity_priors.clear()
        self.state.cached_entity_multi_concepts.clear()
        self.state.last_fuzzy_relations.clear()
        self.state.committed_fuzzy_assertions.clear()
        self.state.adapted_role_pairs.clear()
        self.state.promoted_concept_ids.clear()
        self.relation_learner.reset()
        self.tracker.reset()
        self._committed_beliefs.clear()
        if (
            not boundary.is_same_context
            and self.symbolic_theory is not None
            and self.symbolic_theory.labels_non_stationary
        ):
            self.concept_store._concepts.clear()
        for ext in self.feature_registry.extractors:
            if hasattr(ext, "reset_traces"):
                ext.reset_traces()

    def on_boundary(self, boundary: Any) -> None:
        """Lifecycle boundary handler called by AutonomousEpistemicEngine."""
        self.boundary_retention(boundary)

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

        # Find existing active candidates indexed by color for permutation-invariant O(1) lookup
        active_cands = self.tracker.get_active_candidates()
        cands_by_color: dict[int, list[Any]] = {}
        uncolored_cands: list[Any] = []
        for c in active_cands:
            c_col = c.properties.get("color")
            if c_col is None:
                uncolored_cands.append(c)
            else:
                cands_by_color.setdefault(int(c_col), []).append(c)
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

            sample_pos: tuple[float, ...] | None = None
            if hasattr(ent, "centroid") and ent.centroid is not None:
                pos = tuple(float(x) for x in ent.centroid)
                if hasattr(ent, "properties") and isinstance(ent.properties, dict):
                    cells = ent.properties.get("cells")
                    if isinstance(cells, (list, tuple)) and cells:
                        r_rnd = int(round(pos[0]))
                        c_rnd = int(round(pos[1]))
                        if (r_rnd, c_rnd) not in cells:
                            best_cell = min(
                                cells,
                                key=lambda pt: (pt[0] - pos[0]) ** 2 + (pt[1] - pos[1]) ** 2,
                            )
                            sample_pos = (float(best_cell[0]), float(best_cell[1]))
                if hasattr(ent, "bounding_box") and ent.bounding_box is not None:
                    bb = ent.bounding_box
                    if len(bb) == 4:
                        bbox = (
                            float(bb[0]),
                            float(bb[2]),
                            float(bb[1] - bb[0] + 1),
                            float(bb[3] - bb[2] + 1),
                        )
                color = (
                    getattr(ent, "color", None)
                    or getattr(ent, "feature_id", None)
                    or (
                        ent.properties.get("color")
                        if hasattr(ent, "properties") and isinstance(ent.properties, dict)
                        else None
                    )
                    or 1
                )
                role = str(getattr(ent, "role", "entity"))
            elif hasattr(ent, "coords"):
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
            elif hasattr(ent, "grid_pos") and ent.grid_pos is not None:
                pos = tuple(float(x) for x in ent.grid_pos)
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

            ent_size = 1
            if hasattr(ent, "area") and ent.area:
                ent_size = int(ent.area)
            elif hasattr(ent, "coords"):
                ent_size = max(1, len(list(ent.coords)))
            elif isinstance(ent, dict) and "size" in ent:
                ent_size = max(1, int(ent["size"]))

            # Determine stable tracklet ID (WP3: permutation-invariant association)
            prev_coords = None
            if ent_id:
                tracklet_id = TrackletRef(f"trk_{ent_id}")
            else:
                best_trk_id: str | None = None
                best_cand_id: str | None = None
                best_dist = float("inf")
                cand_pool = list(cands_by_color.get(int(color), ()))
                if uncolored_cands:
                    cand_pool.extend(uncolored_cands)
                for c in cand_pool:
                    cid = c.candidate_ref.id
                    if cid in matched_cand_ids:
                        continue
                    c_color = c.properties.get("color")
                    if c_color is not None and int(c_color) != int(color):
                        continue
                    if c.spatial_trajectory:
                        last_p = c.spatial_trajectory[-1]
                        dist = math.sqrt((pos[0] - last_p[0]) ** 2 + (pos[1] - last_p[1]) ** 2)
                        if dist < best_dist and dist <= self.tracker._spatial_distance_threshold:
                            best_dist = dist
                            prev_coords = last_p
                            if c.associated_tracklet_ids:
                                best_trk_id = next(iter(c.associated_tracklet_ids))
                                best_cand_id = cid
                if best_cand_id is not None:
                    matched_cand_ids.add(best_cand_id)

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
            cand_state = self.tracker.get_candidate(cand.id)
            if cand_state is not None:
                cand_state.properties["color"] = int(color)
            if (
                prev_coords is None
                and cand_state is not None
                and len(cand_state.spatial_trajectory) >= 2
            ):
                prev_coords = cand_state.spatial_trajectory[-2]
            vel = (
                (float(pos[0]) - float(prev_coords[0]), float(pos[1]) - float(prev_coords[1]))
                if prev_coords is not None
                else (0.0, 0.0)
            )
            moved = bool(prev_coords is not None and (vel[0] != 0.0 or vel[1] != 0.0))
            entity_states[cand.id] = {
                "position": pos,
                "sample_pos": sample_pos or pos,
                "bbox": bbox,
                "color": color,
                "size": ent_size,
                "velocity": vel,
                "moved": moved,
                "role": role,
            }

        scene_sig = tuple(
            (
                int(round(st["position"][0])),
                int(round(st["position"][1])),
                int(st["color"]),
                int(st["size"]),
            )
            for _, st in sorted(entity_states.items())
        )
        if scene_sig == self.state.last_scene_sig and self.state.last_feature_vectors:
            merged_cached: dict[int, dict[str, float]] = {
                rc: dict(rp) for rc, rp in self.state.last_role_priors.items()
            }
            for rc, drp in self.state.dialogue_role_priors.items():
                dst = merged_cached.setdefault(rc, {})
                for k_r, v_r in drp.items():
                    if v_r > dst.get(k_r, 0.0):
                        dst[k_r] = v_r
            active_cands_cached = self.tracker.get_active_candidates()
            return {
                "sge_observation_id": obs_id,
                "sge_active_candidates": [c.candidate_ref.id for c in active_cands_cached],
                "sge_entity_states": entity_states,
                "sge_proposed_claims_count": 0,
                "sge_committed_claims_count": 0,
                "sge_feature_vectors_count": len(self.state.last_feature_vectors),
                "sge_role_priors": merged_cached,
                "sge_composite_matches": [],
                "sge_concept_assignments": [],
                "sge_fuzzy_relations": list(self.state.last_fuzzy_relations),
                "sge_concept_splits": 0,
                "sge_concept_merges": 0,
                "sge_outcome_verified": False,
            }
        self.state.last_scene_sig = scene_sig

        # 3. Dynamic Relation Induction (evaluate only dyads whose participants moved/changed)
        self.relation_learner.observe_entity_states(entity_states, timestamp=now)
        claims, ev_records = self.relation_learner.propose_claims(
            source_channel_id="cortical_perception",
            observation_id=obs_id,
            dependency_group=f"dep_cortex_s{step}",
            candidate_pairs=self.relation_learner.last_changed_pairs,
        )

        # 3b. Fuzzy Adaptive Relation Evaluation & Dual-Channel Commit (ADR 006 Phase F1)
        fuzzy_assertions_to_commit: list[Any] = []
        if self.enable_fuzzy_grounding:
            nav_diam = 10.0
            curr_grid_ctx = ctx.curr_grid if ctx is not None else None
            if isinstance(curr_grid_ctx, np.ndarray) and curr_grid_ctx.ndim >= 2:
                nav_diam = max(
                    1.0, math.hypot(float(curr_grid_ctx.shape[-2]), float(curr_grid_ctx.shape[-1]))
                )
            fuzzy_rel_summaries: list[dict[str, Any]] = []
            for pair_key in self.relation_learner.last_changed_pairs:
                history = self.relation_learner._dyad_histories.get(pair_key)
                latest_dyad = history.latest() if history is not None else None
                if latest_dyad is None:
                    continue
                sub_st = entity_states.get(pair_key[0], {})
                obj_st = entity_states.get(pair_key[1], {})
                sub_bb = sub_st.get("bbox")
                obj_bb = obj_st.get("bbox")
                sub_ext = (
                    math.hypot(float(sub_bb[2]), float(sub_bb[3]))
                    if sub_bb and len(sub_bb) >= 4
                    else math.sqrt(max(1.0, float(sub_st.get("size", 1))))
                )
                obj_ext = (
                    math.hypot(float(obj_bb[2]), float(obj_bb[3]))
                    if obj_bb and len(obj_bb) >= 4
                    else math.sqrt(max(1.0, float(obj_st.get("size", 1))))
                )
                f_rels = self.relation_learner.evaluate_fuzzy_relations(
                    latest_dyad,
                    history,
                    subject_extent=sub_ext,
                    object_extent=obj_ext,
                    navigable_diameter=nav_diam,
                    contextual_scope="manipulation",
                    snapshot_sequence=step,
                    evidence_refs=(obs_id,),
                )
                for fa in f_rels:
                    fuzzy_rel_summaries.append(
                        {
                            "subject": fa.subject_ref,
                            "target": fa.target_ref,
                            "semantics": fa.semantics,
                            "degree": round(float(fa.degree), 4),
                            "model_id": fa.model_id,
                            "model_version": fa.model_version,
                        }
                    )
                    f_key = f"{fa.subject_ref}:{fa.semantics}:{fa.target_ref}"
                    prev_fa = self.state.committed_fuzzy_assertions.get(f_key)
                    if (
                        prev_fa is None
                        or abs(float(fa.degree) - prev_fa[0]) >= 0.05
                        or fa.model_version != prev_fa[1]
                    ):
                        fuzzy_assertions_to_commit.append(fa)
                        self.state.committed_fuzzy_assertions[f_key] = (
                            float(fa.degree),
                            int(fa.model_version),
                        )
            self.state.last_fuzzy_relations = fuzzy_rel_summaries

        # 4. Epistemic Evaluation & Commit of Accepted Relations (bounded growth check)
        committed_claims: list[str] = []
        to_commit: list[tuple[Any, Any, Any]] = []
        ev_by_id = {e.evidence_id: e for e in ev_records}
        for claim in claims:
            # Match supporting evidence
            claim_ev = [ev_by_id[eid] for eid in claim.evidence_ids if eid in ev_by_id]
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
                to_commit.append((decision, claim, belief))
                self._committed_beliefs[b_key] = (
                    claim.polarity.value,
                    decision.posterior_belief,
                )
                committed_claims.append(claim.claim_id)

        if to_commit:
            try:
                eff_scope = Scope(tenant_id=self.workspace.tenant_id or "default")
                self.committer.commit_decisions(to_commit, scope=eff_scope)
            except Exception as exc:
                logger.debug("Failed to commit batch claims: %s", exc)

        active_cands = self.tracker.get_active_candidates()

        # 5. Feature Extraction via FeatureExtractorRegistry with Unmoved Entity Caching
        if ctx is not None:
            curr_grid = getattr(ctx, "curr_grid", None)
            if curr_grid is not None:
                envelope.metadata["grid"] = curr_grid
                envelope.metadata["bg_feature"] = getattr(ctx, "bg_feature", 0)

        needs_extract_entities: list[dict[str, Any]] = []
        for cand_id, state in entity_states.items():
            if (
                state.get("moved", False)
                or cand_id not in self.state.cached_entity_features
                or cand_id not in self.state.cached_entity_concepts
            ):
                needs_extract_entities.append({"id": cand_id, **state})

        feature_vectors: list[PerceptualFeatureVector] = []
        if needs_extract_entities:
            envelope.metadata["entities"] = needs_extract_entities
            try:
                extracted = self.feature_registry.extract_all(envelope)
                for fv in extracted:
                    ent_id = fv.feature_metadata.get("entity_id")
                    if ent_id:
                        self.state.cached_entity_features[ent_id] = fv
            except Exception as exc:
                logger.debug("FeatureExtractorRegistry.extract_all failed: %s", exc)

        for cand_id in entity_states:
            cached_fv = self.state.cached_entity_features.get(cand_id)
            if cached_fv is not None:
                feature_vectors.append(cached_fv)

        # 6. Concept Learning & Role Prior Inference via ConceptLearner (active across all modes)
        concept_assignments: list[ConceptAssignment] = []
        feature_concept_ids: dict[int, str] = {}
        role_priors: dict[int, dict[str, float]] = {}
        infer_priors = (
            self.enable_fuzzy_grounding
            or self.perception_mode in ("learned", "snn")
            or (
                self.symbolic_theory is not None
                and self.symbolic_theory.has_prior_context_transition
            )
        ) and not (self.symbolic_theory is not None and self.symbolic_theory.labels_non_stationary)
        for fv in feature_vectors:
            try:
                entity_id: str | None = fv.feature_metadata.get("entity_id")
                if not entity_id and active_cands:
                    entity_id = active_cands[0].candidate_ref.id

                is_unmoved_cached = bool(
                    entity_id
                    and not entity_states.get(entity_id, {}).get("moved", False)
                    and entity_id in self.state.cached_entity_concepts
                )
                if infer_priors and "raw_color" in fv.feature_metadata:
                    if (
                        is_unmoved_cached
                        and entity_id is not None
                        and entity_id in self.state.cached_entity_priors
                    ):
                        priors = self.state.cached_entity_priors[entity_id]
                    else:
                        if self.enable_fuzzy_grounding:
                            priors = self.concept_learner.infer_role_priors_fuzzy(
                                fv, graph=self.workspace.graph
                            )
                        else:
                            priors = self.concept_learner.infer_role_priors(
                                fv, max_distance=0.35, distance_scale=0.30
                            )
                        if entity_id:
                            self.state.cached_entity_priors[entity_id] = dict(priors)
                    if priors:
                        rc = int(fv.feature_metadata["raw_color"])
                        existing_rp = role_priors.setdefault(rc, {})
                        for r_name, r_conf in priors.items():
                            if r_conf > existing_rp.get(r_name, 0.0):
                                existing_rp[r_name] = r_conf

                if is_unmoved_cached and entity_id is not None:
                    assignment = self.state.cached_entity_concepts[entity_id]
                else:
                    if self.enable_fuzzy_grounding:
                        multi = self.concept_learner.learn_fuzzy(
                            fv, exemplar_entity_id=entity_id, snapshot_sequence=step
                        )
                        assignment = multi.primary_assignment
                        if entity_id:
                            self.state.cached_entity_multi_concepts[entity_id] = multi.memberships
                        for fa in multi.assertions:
                            f_key = f"{fa.subject_ref}:{fa.semantics}:*"
                            prev_fa = self.state.committed_fuzzy_assertions.get(f_key)
                            if (
                                prev_fa is None
                                or abs(float(fa.degree) - prev_fa[0]) >= 0.05
                                or fa.model_version != prev_fa[1]
                            ):
                                fuzzy_assertions_to_commit.append(fa)
                                self.state.committed_fuzzy_assertions[f_key] = (
                                    float(fa.degree),
                                    int(fa.model_version),
                                )
                    else:
                        assignment = self.concept_learner.learn(fv, exemplar_entity_id=entity_id)
                    if entity_id:
                        self.state.cached_entity_concepts[entity_id] = assignment

                concept_assignments.append(assignment)
                if "raw_color" in fv.feature_metadata:
                    rc_feat = int(fv.feature_metadata["raw_color"])
                    feature_concept_ids[rc_feat] = assignment.concept_ref.id
                    self.state.episode_feature_memory[rc_feat] = fv
            except Exception as exc:
                logger.debug("ConceptLearner.learn failed for %s: %s", fv.feature_id, exc)

        if fuzzy_assertions_to_commit:
            try:
                eff_scope = Scope(tenant_id=self.workspace.tenant_id or "default")
                self.committer.commit_fuzzy_assertions(
                    fuzzy_assertions_to_commit,
                    scope=eff_scope,
                )
            except Exception as exc:
                logger.debug("Failed to commit batched fuzzy assertions: %s", exc)

        # Run concept maintenance (split/merge detection) periodically
        split_count = 0
        merge_count = 0
        if self.state.step_counter % 12 == 0 and self.state.step_counter > 0:
            try:
                splits, merges = self.concept_learner.maintain()
                split_count = len(splits)
                merge_count = len(merges)
                if split_count > 0 or merge_count > 0:
                    self.state.cached_entity_concepts.clear()
                    self.state.cached_entity_priors.clear()
            except Exception as exc:
                logger.debug("ConceptLearner.maintain failed: %s", exc)

        # Promote eligible concepts to authoritative HCIR state (WP3: grounded concept promotion)
        promoted_concepts: list[str] = []
        eff_scope = Scope(tenant_id=self.workspace.tenant_id or "default")
        for concept in self.concept_store.get_all():
            cid = concept.concept_ref.id
            if cid in self.state.promoted_concept_ids:
                continue
            if self.concept_store.lifecycle_manager.should_propose_promotion(concept):
                if self.workspace.graph.has_node(cid):
                    self.state.promoted_concept_ids.add(cid)
                    continue
                prop_id = f"prop_{cid}_{uuid.uuid4().hex[:6]}"
                proposal = self.concept_store.lifecycle_manager.create_promotion_proposal(
                    concept=concept,
                    proposal_id=prop_id,
                )
                try:
                    tx = self.committer.commit_concept_promotion(proposal, concept, scope=eff_scope)
                    if tx.is_committed:
                        promoted_concepts.append(cid)
                        self.state.promoted_concept_ids.add(cid)
                except Exception as exc:
                    logger.debug("Failed to commit concept promotion for %s: %s", cid, exc)

        self.state.last_feature_vectors = feature_vectors
        self.state.last_concept_assignments = concept_assignments
        self.state.last_feature_concept_ids = feature_concept_ids

        has_composite = any(c.is_composite for c in self.concept_store.get_all_concepts())
        composite_matches: list[dict[str, Any]] = []
        if has_composite and self.state.episode_feature_memory:
            composite_matches = self.concept_learner.infer_composite_role_priors(
                self.state.episode_feature_memory
            )
            for match in composite_matches:
                bindings = match.get("part_feature_bindings", {})
                scores = match.get("part_role_scores", {})
                for part_role, rc_feat in bindings.items():
                    boost = float(scores.get(part_role, 0.0))
                    if boost > 0.0:
                        existing_rp = role_priors.setdefault(int(rc_feat), {})
                        if boost > existing_rp.get(part_role, 0.0):
                            existing_rp[part_role] = boost

        self.state.last_role_priors = {rc: dict(rp) for rc, rp in role_priors.items()}
        for rc, drp in self.state.dialogue_role_priors.items():
            dst = role_priors.setdefault(rc, {})
            for k_r, v_r in drp.items():
                if v_r > dst.get(k_r, 0.0):
                    dst[k_r] = v_r

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
            "sge_role_priors": role_priors,
            "sge_composite_matches": composite_matches,
            "sge_concept_assignments": [
                {
                    "concept_id": a.concept_ref.id,
                    "distance": a.distance,
                    "is_new": a.is_new_concept,
                }
                for a in concept_assignments
            ],
            "sge_fuzzy_relations": list(self.state.last_fuzzy_relations),
            "sge_concept_splits": split_count,
            "sge_concept_merges": merge_count,
            "sge_outcome_verified": outcome_verified,
        }

    def sync_theory_roles(
        self,
        theory: Any,
        hazard_features: set[int] | frozenset[int] | None = None,
    ) -> int:
        """Bind empirically discovered causal roles onto pure GroundedConcepts and adapt weights."""
        if theory is None and not hazard_features:
            return 0
        haz_set = set(hazard_features or ())
        if theory is not None:
            th_lethal = getattr(theory, "lethal_features", None)
            if th_lethal:
                haz_set |= set(th_lethal)

        candidate_fvs: list[PerceptualFeatureVector] = list(self.state.last_feature_vectors)
        seen_raw_feats = {
            int(fv.feature_metadata["raw_color"])
            for fv in candidate_fvs
            if "raw_color" in fv.feature_metadata
        }
        for rc, mem_fv in self.state.episode_feature_memory.items():
            if rc not in seen_raw_feats:
                candidate_fvs.append(mem_fv)

        if not candidate_fvs:
            return 0

        bound = 0
        newly_bound = False
        adapted_any = False
        seen_pairs: set[tuple[int, str]] = set()
        role_to_concept_id: dict[str, str] = {}
        actuator_feats: set[int] = set()
        affinity_barriers: set[int] = set()
        if theory is not None:
            actuator_feats = {
                int(f)
                for f in (getattr(theory, "actuator_features", ()) or ())
                if isinstance(f, int)
            } | {
                int(f)
                for f in (getattr(theory, "tool_barrier_affinities", {}) or {}).keys()
                if isinstance(f, int)
            }
            for b_set in (getattr(theory, "tool_barrier_affinities", {}) or {}).values():
                affinity_barriers |= {int(b) for b in b_set if isinstance(b, int)}
        for fv in candidate_fvs:
            entity_id = fv.feature_metadata.get("entity_id")
            raw_color = fv.feature_metadata.get("raw_color")
            if raw_color is None:
                continue
            feat = int(raw_color)
            cid = self.state.last_feature_concept_ids.get(feat)
            roles: list[str] = []
            if feat in haz_set:
                roles.append("hazard")
            if theory is not None:
                if theory.is_cargo(feat):
                    roles.append("cargo")
                if feat in theory.goal_features:
                    roles.append("goal")
                if feat in theory.receptacle_features:
                    roles.append("receptacle")
                if feat in actuator_feats:
                    roles.append("actuator")
                if (theory.is_barrier(feat) or feat in affinity_barriers) and feat not in haz_set:
                    roles.append("barrier")
            for r_name in roles:
                if (feat, r_name) in seen_pairs:
                    continue
                seen_pairs.add((feat, r_name))
                existing_c = self.concept_store.get_concept(cid) if cid else None
                already_had_role = (
                    existing_c is not None and f"role:{r_name}" in existing_c.affordances
                )
                refreshed_fv = fv
                pair_key = (feat, r_name)
                if (
                    self.perception_mode in ("learned", "snn")
                    and pair_key not in self.state.adapted_role_pairs
                ):
                    self.state.adapted_role_pairs.add(pair_key)
                    for ext in self.feature_registry.extractors:
                        adapt_fn = getattr(ext, "adapt_role", None)
                        if callable(adapt_fn):
                            adapted = adapt_fn(fv.feature_id, r_name)
                            if isinstance(adapted, tuple) and adapted:
                                adapted_any = True
                                refreshed_fv = PerceptualFeatureVector(
                                    feature_id=fv.feature_id,
                                    observation_id=fv.observation_id,
                                    modality=fv.modality,
                                    channel_id=fv.channel_id,
                                    embedding=adapted,
                                    spatial_extent=fv.spatial_extent,
                                    semantic_labels=fv.semantic_labels,
                                    feature_metadata=fv.feature_metadata,
                                    provenance_hash=fv.provenance_hash,
                                    timestamp=fv.timestamp,
                                )
                                break
                target_promoted_c = None
                pure_concept = self.concept_learner.bind_role_pure(
                    cid,
                    refreshed_fv,
                    r_name,
                    exemplar_entity_id=str(entity_id) if entity_id else None,
                    confidence=1.0,
                )
                if pure_concept is not None:
                    if not already_had_role or pure_concept.concept_ref.id != cid:
                        newly_bound = True
                    self.state.last_feature_concept_ids[feat] = pure_concept.concept_ref.id
                    cid = pure_concept.concept_ref.id
                    role_to_concept_id[r_name] = cid
                    bound += 1
                    target_promoted_c = pure_concept
                elif entity_id:
                    bound_concept = self.concept_learner.bind_role_to_exemplar(
                        str(entity_id), r_name
                    )
                    if bound_concept is not None:
                        if not already_had_role:
                            newly_bound = True
                        role_to_concept_id[r_name] = bound_concept.concept_ref.id
                        bound += 1
                        target_promoted_c = bound_concept

                if (
                    self.enable_fuzzy_grounding
                    and target_promoted_c is not None
                    and target_promoted_c.feature_distributions
                    and (
                        not already_had_role
                        or target_promoted_c.concept_ref.id not in self.state.promoted_concept_ids
                        or not self.workspace.graph.has_node(target_promoted_c.concept_ref.id)
                    )
                ):
                    q = target_promoted_c.grounding_quality
                    if q.predictive_utility < 0.85 or q.interaction_support < 0.85:
                        target_promoted_c.update_grounding_quality(
                            GroundingQuality(
                                perceptual_stability=q.perceptual_stability,
                                temporal_persistence=q.temporal_persistence,
                                relational_consistency=q.relational_consistency,
                                cross_modal_support=q.cross_modal_support,
                                predictive_utility=max(q.predictive_utility, 0.85),
                                interaction_support=max(q.interaction_support, 0.85),
                                boundary_stability=q.boundary_stability,
                                measurements=dict(q.measurements),
                            )
                        )
                    tcid = target_promoted_c.concept_ref.id
                    prop = self.concept_store.lifecycle_manager.create_promotion_proposal(
                        concept=target_promoted_c,
                        proposal_id=f"prop_{tcid}_{uuid.uuid4().hex[:6]}",
                        rationale=f"Empirical causal role:{r_name} grounded",
                    )
                    try:
                        eff_scope = Scope(tenant_id=self.workspace.tenant_id or "default")
                        tx = self.committer.commit_concept_promotion(
                            prop, target_promoted_c, scope=eff_scope
                        )
                        if tx.is_committed:
                            self.state.promoted_concept_ids.add(tcid)
                    except Exception as exc:
                        logger.debug(
                            "Failed to promote causal role concept %s (%s): %s",
                            tcid,
                            r_name,
                            exc,
                        )
        if "cargo" in role_to_concept_id and "receptacle" in role_to_concept_id:
            self.concept_learner.induce_composite_concept(
                parts={
                    "cargo": role_to_concept_id["cargo"],
                    "receptacle": role_to_concept_id["receptacle"],
                },
                relation_name="docks_into",
                composite_role="push_docking_system",
            )
        if "actuator" in role_to_concept_id and "barrier" in role_to_concept_id:
            self.concept_learner.induce_composite_concept(
                parts={
                    "actuator": role_to_concept_id["actuator"],
                    "barrier": role_to_concept_id["barrier"],
                },
                relation_name="unlocks_barrier",
                composite_role="lock_mechanism_system",
            )
        if newly_bound:
            self.state.cached_entity_priors.clear()
            self.state.cached_entity_concepts.clear()
        if adapted_any:
            self.state.cached_entity_features.clear()
        return bound

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

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5: Dialogue & Grounded Language Interaction
    # ─────────────────────────────────────────────────────────────────────────

    def process_dialogue(
        self,
        dialogue: ObservationEnvelope | str,
        speaker_id: str = "user",
    ) -> DialogueInterpretationResult:
        """Processes a natural language dialogue utterance or envelope, resolving grounded entities

        against the active ConceptStore without requiring hand-seeded aliases.
        """
        from hbllm.hcir.compiler import HCIRCompiler

        if isinstance(dialogue, ObservationEnvelope):
            text = str(dialogue.metadata.get("text", ""))
            speaker = str(dialogue.metadata.get("speaker_id", speaker_id))
        else:
            text = str(dialogue)
            speaker = speaker_id

        compiler = HCIRCompiler(concept_store=self.concept_store)
        res = compiler.interpret_dialogue(
            text, speaker_id=speaker, concept_store=self.concept_store
        )

        for goal in res.goal_proposals:
            self.state.pending_epistemic_goals.append(goal)

        return res

    def ground_mission_instruction(
        self,
        instruction: str,
        curr_grid: Any,
        bg_feature: int,
        avatar_features: set[int],
        theory: Any,
    ) -> dict[int, dict[str, float]]:
        """Ground a natural-language mission or rule hint against active concepts and scene grid."""
        import numpy as np

        from hbllm.hcir.compiler import HCIRCompiler, IntentType

        clean = instruction.strip()
        if not clean:
            return self.state.dialogue_role_priors

        self.process_dialogue(clean)
        compiler = HCIRCompiler(concept_store=self.concept_store, workspace=self.workspace)
        parsed_ast = compiler.parse_utterance(clean)
        is_goal_cmd = parsed_ast.intent == IntentType.ACT
        target_feat = compiler._resolve_feature_for_label(clean, theory) if is_goal_cmd else None

        if isinstance(target_feat, int):
            self.state.dialogue_role_priors[target_feat] = {"goal": 0.96}
            if theory is not None:
                for gf in list(theory.goal_features):
                    if gf != target_feat:
                        theory.retract_goal(gf, reason="mission_override")
                theory.induce_candidate_goal(target_feat, evidence="grounded_dialogue")
            if curr_grid is not None:
                arr = np.asarray(curr_grid)
                if arr.ndim == 2:
                    vals, cnts = np.unique(arr, return_counts=True)
                    for v_np, c_np in zip(vals.tolist(), cnts.tolist(), strict=False):
                        v_int = int(v_np)
                        if (
                            v_int != target_feat
                            and v_int != bg_feature
                            and v_int not in avatar_features
                            and int(c_np) <= 4
                            and (theory is None or not theory.is_barrier(v_int))
                        ):
                            self.state.dialogue_role_priors[v_int] = {"hazard": 0.78}
        elif theory is not None:
            compiler.compile_utterance_to_causal_hypothesis(clean, theory=theory)

        return self.state.dialogue_role_priors
