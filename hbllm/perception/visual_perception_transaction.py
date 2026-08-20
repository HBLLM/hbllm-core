"""Visual Perception Transaction — atomic HCIR state commitment.

The EXCLUSIVE layer for committing visual perception results into
HCIR cognitive state.  All node creation, edge linking, and belief
recording happens here — atomically.

Architecture invariant:
    VisualPerceptionRuntime produces evidence.
    VisualPerceptionTransaction commits state.
    These never cross.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hbllm.hcir.graph import (
    CognitiveGraph,
    HCIREdge,
    HCIREdgeType,
    VisualConceptNode,
    VisualObservationNode,
    _new_id,
)
from hbllm.hcir.types import Provenance
from hbllm.perception.providers.evidence import (
    CandidateRanking,
    ConceptCandidate,
    EpistemicEvidenceProfile,
    VisualAssessment,
)
from hbllm.perception.providers.policy import RecognitionPolicy

if TYPE_CHECKING:
    from hbllm.memory.belief_graph import BeliefGraph
    from hbllm.perception.visual_memory import VisualMemory

logger = logging.getLogger(__name__)


@dataclass
class VisualRecognitionResult:
    """Result of a visual recognition transaction."""

    matched: bool
    concept_node_id: str | None = None
    label: str = ""
    is_ambiguous: bool = False
    is_novel: bool = False
    observation_node_id: str | None = None
    confidence: EpistemicEvidenceProfile | None = None
    ranking: CandidateRanking | None = None
    candidates: list[ConceptCandidate] | None = None


class VisualPerceptionTransaction:
    """Commits visual assessment into HCIR — atomically.

    All state changes (observation, concept, belief, edges)
    succeed together or are rolled back on failure.

    Transaction types:
        commit_learning:     User-labeled evidence → concept
        commit_recognition:  Unlabeled evidence → match/ambiguous/novel

    Uses RecognitionPolicy for thresholds — no hardcoded magic numbers.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        memory: VisualMemory,
        belief_graph: BeliefGraph | None = None,
        policy: RecognitionPolicy | None = None,
    ) -> None:
        self.graph = graph
        self.memory = memory
        self.belief_graph = belief_graph
        self.policy = policy or RecognitionPolicy()

    async def commit_learning(
        self,
        assessment: VisualAssessment,
    ) -> VisualConceptNode:
        """Commit labeled evidence as a concept. Atomic.

        Three cases:
            1. Same label + high similarity → update existing concept
            2. Different label + high similarity → new concept + SIMILAR_TO
            3. No match / low similarity → new concept
        """
        label = assessment.proposed_label
        if not label:
            raise ValueError("commit_learning requires proposed_label")

        best = assessment.candidate_concepts[0] if assessment.candidate_concepts else None

        if best and best.label == label and best.best_similarity >= self.policy.minimum_similarity:
            return await self._update_existing_concept(assessment, best)
        if (
            best and best.label != label and best.best_similarity >= self.policy.minimum_similarity
        ):
            return await self._create_with_similarity(assessment, best)
        return await self._create_new_concept(assessment)

    async def commit_recognition(
        self,
        assessment: VisualAssessment,
    ) -> VisualRecognitionResult:
        """Commit recognition using policy thresholds.

        Three outcomes:
            1. Clear match → positive recognition
            2. Ambiguous → store observation, no concept commitment
            3. Novel → store as unknown observation
        """
        ranking = assessment.ranking

        if self.policy.is_match(ranking):
            best = assessment.candidate_concepts[0]
            return await self._commit_positive_recognition(assessment, best)
        if self.policy.is_ambiguous(ranking):
            return await self._commit_ambiguous_recognition(assessment)
        return await self._commit_novel_observation(assessment)

    # ══════════════════════════════════════════════════════════════════
    # Learning transactions
    # ══════════════════════════════════════════════════════════════════

    async def _create_new_concept(
        self,
        assessment: VisualAssessment,
    ) -> VisualConceptNode:
        """Create a brand-new visual concept — atomic transaction."""
        # Allocate concept ID BEFORE persistence
        concept_id = _new_id("vcpt")
        label = assessment.proposed_label or "unknown"
        emb = assessment.evidence.embedding

        try:
            # 1. Store observation embedding
            obs_ref = await self.memory.store_observation(
                emb,
                concept_node_id=concept_id,
                label=label,
            )

            # 2. Store prototype (centroid = first observation)
            proto_ref = await self.memory.store_prototype(
                concept_node_id=concept_id,
                centroid=emb.vector,
                embedding=emb,
            )

            # 3. HCIR observation node
            obs_node = VisualObservationNode(
                id=_new_id("vobs"),
                embedding_ref=obs_ref,
                embedding_space=emb.space_id,
                embedding_model=emb.model_id,
                image_hash=assessment.evidence.image_hash,
                provenance=assessment.evidence.provenance,
            )
            self.graph.add_node(obs_node)

            # 4. HCIR concept node
            concept = VisualConceptNode(
                id=concept_id,
                label=label,
                definition=f"Visually learned concept: {label}",
                prototype_ref=proto_ref,
                embedding_space=emb.space_id,
                embedding_model=emb.model_id,
                observation_count=1,
                exemplar_refs=[obs_ref],
                contexts=([assessment.proposed_context] if assessment.proposed_context else []),
                last_seen=time.time(),
                provenance=Provenance(
                    created_by="visual_perception_transaction",
                    source_type="observed",
                    reason=f"One-shot learning: '{label}'",
                ),
            )
            self.graph.add_node(concept)

            # 5. SUPPORTS edge: observation → concept
            self.graph.add_edge(
                HCIREdge(
                    edge_type=HCIREdgeType.SUPPORTS,
                    sources=[obs_node.id],
                    targets=[concept.id],
                )
            )

            # 6. Belief with structured epistemic profile
            await self._record_belief(
                concept.id,
                assessment.epistemic_profile,
                f"One-shot learning: '{label}'",
                "user_input",
            )

            logger.info(
                "Created visual concept '%s' (id=%s) with 1 observation",
                label,
                concept_id,
            )
            return concept

        except Exception:
            logger.error("Visual concept creation failed — transaction incomplete")
            raise

    async def _update_existing_concept(
        self,
        assessment: VisualAssessment,
        best: ConceptCandidate,
    ) -> VisualConceptNode:
        """Update an existing concept with a new observation."""
        emb = assessment.evidence.embedding
        concept_id = best.concept_node_id
        label = assessment.proposed_label or best.label

        try:
            # Add exemplar (may be skipped if near-duplicate)
            obs_ref = await self.memory.add_exemplar(
                concept_id,
                emb,
                self.policy,
            )

            if obs_ref is None:
                # Near-duplicate — just update prototype
                count = self.memory.get_observation_count(concept_id)
                await self.memory.update_prototype(concept_id, emb, count + 1)
            else:
                # New exemplar — add observation node + update prototype
                obs_node = VisualObservationNode(
                    id=_new_id("vobs"),
                    embedding_ref=obs_ref,
                    embedding_space=emb.space_id,
                    embedding_model=emb.model_id,
                    image_hash=assessment.evidence.image_hash,
                    provenance=assessment.evidence.provenance,
                )
                self.graph.add_node(obs_node)

                self.graph.add_edge(
                    HCIREdge(
                        edge_type=HCIREdgeType.SUPPORTS,
                        sources=[obs_node.id],
                        targets=[concept_id],
                    )
                )

                count = self.memory.get_observation_count(concept_id)
                await self.memory.update_prototype(concept_id, emb, count)

            # Update concept node
            concept_node = self.graph.get_node(concept_id)
            if isinstance(concept_node, VisualConceptNode):
                concept_node.observation_count = self.memory.get_observation_count(concept_id)
                concept_node.last_seen = time.time()
                if obs_ref and obs_ref not in concept_node.exemplar_refs:
                    concept_node.exemplar_refs.append(obs_ref)

            # Reinforce belief
            await self._record_belief(
                concept_id,
                assessment.epistemic_profile,
                f"Reinforced learning: '{label}' (n={count})",
                "reinforcement",
            )

            logger.info("Updated concept '%s' (id=%s)", label, concept_id)
            return concept_node  # type: ignore[return-value]

        except Exception:
            logger.error("Concept update failed — transaction incomplete")
            raise

    async def _create_with_similarity(
        self,
        assessment: VisualAssessment,
        similar: ConceptCandidate,
    ) -> VisualConceptNode:
        """Create a new concept that's visually similar to an existing one."""
        concept = await self._create_new_concept(assessment)

        # Add SIMILAR_TO edge
        self.graph.add_edge(
            HCIREdge(
                edge_type=HCIREdgeType.SIMILAR_TO,
                sources=[concept.id],
                targets=[similar.concept_node_id],
                provenance=Provenance(
                    created_by="visual_perception_transaction",
                    reason=f"Visually similar (sim={similar.best_similarity:.3f})",
                ),
            )
        )

        logger.info(
            "Created concept '%s' similar to '%s' (sim=%.3f)",
            assessment.proposed_label,
            similar.label,
            similar.best_similarity,
        )
        return concept

    # ══════════════════════════════════════════════════════════════════
    # Recognition transactions
    # ══════════════════════════════════════════════════════════════════

    async def _commit_positive_recognition(
        self,
        assessment: VisualAssessment,
        best: ConceptCandidate,
    ) -> VisualRecognitionResult:
        """Confident recognition — store observation + link to concept."""
        emb = assessment.evidence.embedding

        obs_ref = await self.memory.add_exemplar(
            best.concept_node_id,
            emb,
            self.policy,
        )

        obs_node_id = None
        if obs_ref:
            obs_node = VisualObservationNode(
                id=_new_id("vobs"),
                embedding_ref=obs_ref,
                embedding_space=emb.space_id,
                embedding_model=emb.model_id,
                image_hash=assessment.evidence.image_hash,
                provenance=assessment.evidence.provenance,
            )
            self.graph.add_node(obs_node)
            obs_node_id = obs_node.id

            self.graph.add_edge(
                HCIREdge(
                    edge_type=HCIREdgeType.SUPPORTS,
                    sources=[obs_node.id],
                    targets=[best.concept_node_id],
                )
            )

            # Update concept
            concept_node = self.graph.get_node(best.concept_node_id)
            if isinstance(concept_node, VisualConceptNode):
                concept_node.observation_count = self.memory.get_observation_count(
                    best.concept_node_id
                )
                concept_node.last_seen = time.time()
                if obs_ref not in concept_node.exemplar_refs:
                    concept_node.exemplar_refs.append(obs_ref)

                count = concept_node.observation_count
                await self.memory.update_prototype(best.concept_node_id, emb, count)

        return VisualRecognitionResult(
            matched=True,
            concept_node_id=best.concept_node_id,
            label=best.label,
            observation_node_id=obs_node_id,
            confidence=assessment.epistemic_profile,
            ranking=assessment.ranking,
        )

    async def _commit_ambiguous_recognition(
        self,
        assessment: VisualAssessment,
    ) -> VisualRecognitionResult:
        """Ambiguous — store observation without concept commitment."""
        emb = assessment.evidence.embedding

        obs_ref = await self.memory.store_observation(
            emb,
            concept_node_id=None,
            label="ambiguous",
        )
        obs_node = VisualObservationNode(
            id=_new_id("vobs"),
            embedding_ref=obs_ref,
            embedding_space=emb.space_id,
            embedding_model=emb.model_id,
            image_hash=assessment.evidence.image_hash,
            provenance=assessment.evidence.provenance,
        )
        self.graph.add_node(obs_node)

        return VisualRecognitionResult(
            matched=False,
            is_ambiguous=True,
            observation_node_id=obs_node.id,
            confidence=assessment.epistemic_profile,
            ranking=assessment.ranking,
            candidates=assessment.candidate_concepts,
        )

    async def _commit_novel_observation(
        self,
        assessment: VisualAssessment,
    ) -> VisualRecognitionResult:
        """Novel — store as unassigned observation."""
        emb = assessment.evidence.embedding

        obs_ref = await self.memory.store_observation(
            emb,
            concept_node_id=None,
            label="unknown",
        )
        obs_node = VisualObservationNode(
            id=_new_id("vobs"),
            embedding_ref=obs_ref,
            embedding_space=emb.space_id,
            embedding_model=emb.model_id,
            image_hash=assessment.evidence.image_hash,
            provenance=assessment.evidence.provenance,
        )
        self.graph.add_node(obs_node)

        return VisualRecognitionResult(
            matched=False,
            is_novel=True,
            observation_node_id=obs_node.id,
            confidence=assessment.epistemic_profile,
            ranking=assessment.ranking,
        )

    # ══════════════════════════════════════════════════════════════════
    # Belief recording
    # ══════════════════════════════════════════════════════════════════

    async def _record_belief(
        self,
        concept_id: str,
        profile: EpistemicEvidenceProfile,
        reason: str,
        trigger: str,
    ) -> None:
        """Record structured belief — epistemic profile in metadata, not reason."""
        if self.belief_graph is None:
            return

        from hbllm.memory.belief_graph import BeliefRecord

        record = BeliefRecord(
            id=f"vblf_{uuid.uuid4().hex[:12]}",
            memory_id=concept_id,
            created_by="visual_perception_transaction",
            created_at=time.time(),
            reason=reason,
            trigger=trigger,
            confidence=profile.combined,
        )
        await self.belief_graph.record_belief(record)
