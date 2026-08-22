"""Perception-Epistemic Bridge — pure structural adapter for HCIR.

Materializes sensory observations, normalized evidence, and cross-modal
correlation hyperedges into the canonical HCIR graph.

Master Architectural Invariant:
    Perception may create observations and evidence.
    Correlation may establish relationships.
    Epistemics may evaluate evidence, generate hypotheses, and revise beliefs.
    No perceptual component may directly create or mutate a belief.

Architecture::

    Perception Assessments (AudioAssessment, VisualAssessment)
               │
               ▼
    PerceptionEpistemicBridge
         ├── Materialize AudioObservationNode / VisualObservationNode
         ├── Materialize EvidenceNodes (with EpistemicProfile & Provenance)
         └── Run CorrelationEngine → commit CORRELATES_WITH edges
               │
               ▼
          HCIR Graph
"""

from __future__ import annotations

import logging
import time
import uuid

from hbllm.hcir.graph import (
    AudioObservationNode,
    CognitiveGraph,
    EvidenceNode,
    HCIREdge,
    HCIREdgeType,
    PerceptualEvidenceNode,
    VisualObservationNode,
)
from hbllm.hcir.types import (
    CorrelationCandidate,
    EvidenceStrength,
    PerceptualEpistemicProfile,
    PerceptualModality,
)
from hbllm.perception.correlation_engine import CorrelationEngine, ObservationEnvelope
from hbllm.perception.providers.audio_evidence import AudioAssessment
from hbllm.perception.providers.evidence import VisualAssessment

logger = logging.getLogger(__name__)


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class PerceptionEpistemicBridge:
    """Pure structural bridge from perception assessments to HCIR graph entities.

    Contains NO epistemic logic or belief revision triggers.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        correlation_engine: CorrelationEngine | None = None,
    ) -> None:
        self._graph = graph
        self._correlation_engine = correlation_engine or CorrelationEngine(max_temporal_gap=5.0)

    def ingest_audio_assessment(
        self,
        assessment: AudioAssessment,
    ) -> list[str]:
        """Materialize an AudioAssessment into HCIR ObservationNode and EvidenceNodes.

        Args:
            assessment: Normalized AudioAssessment from AudioPerceptionRuntime.

        Returns:
            List of node IDs committed to the graph.
        """
        committed_ids: list[str] = []
        obs_id = assessment.observation.observation_id or _generate_id("aud_obs")

        # 1. Map epistemic profile
        audio_profile = assessment.epistemic_profile
        sensory_clarity = float(
            getattr(
                audio_profile,
                "sensory_clarity",
                getattr(audio_profile, "perceptual_confidence", 0.85),
            )
        )
        model_conf = float(
            getattr(
                audio_profile,
                "model_confidence",
                getattr(audio_profile, "classification_confidence", 0.85),
            )
        )
        temp_stab = float(
            getattr(
                audio_profile,
                "temporal_stability",
                getattr(audio_profile, "temporal_confidence", 0.85),
            )
        )
        epistemic_profile = PerceptualEpistemicProfile(
            sensory_clarity=sensory_clarity,
            model_confidence=model_conf,
            temporal_stability=temp_stab,
        )

        # 2. Materialize AudioObservationNode
        start_t = float(
            getattr(
                assessment.observation.temporal,
                "start_time",
                getattr(assessment.observation.temporal, "start_seconds", 0.0),
            )
        )
        end_t = float(
            getattr(
                assessment.observation.temporal,
                "end_time",
                getattr(assessment.observation.temporal, "end_seconds", start_t),
            )
        )
        dur = float(
            getattr(
                assessment.observation.temporal,
                "duration",
                getattr(assessment.observation.temporal, "duration_seconds", end_t - start_t),
            )
        )

        transcript_text = assessment.speech.transcript if assessment.speech else ""
        label_text = ""
        first_event_type = ""
        if assessment.events:
            first_event_type = getattr(
                assessment.events[0],
                "event_type",
                getattr(assessment.events[0], "label", "sound_event"),
            )
            label_text = first_event_type
        elif transcript_text:
            label_text = transcript_text[:50]
        elif assessment.scene:
            label_text = assessment.scene.scene_tags[0] if assessment.scene.scene_tags else "scene"

        emb_ref = (
            getattr(
                assessment.observation,
                "embedding_ref",
                getattr(assessment.observation, "embedding_id", ""),
            )
            or ""
        )
        aud_node = AudioObservationNode(
            id=obs_id,
            modality="audio",
            label=label_text,
            embedding_ref=emb_ref,
            event_type=first_event_type
            if first_event_type
            else "speech"
            if transcript_text
            else "audio",
            start_time=start_t,
            end_time=end_t,
            duration=dur,
            transcript=transcript_text,
            temporal_span={"start_time": start_t, "end_time": end_t, "duration": dur},
            provider_provenance=assessment.observation.provenance.__dict__
            if hasattr(assessment.observation.provenance, "__dict__")
            else None,
        )
        self._graph.upsert_node(aud_node)
        committed_ids.append(obs_id)

        # 3. Materialize Speech Evidence (if present)
        if assessment.speech and assessment.speech.transcript:
            speech_evi_id = _generate_id("evi_speech")
            prov_dict = (
                assessment.speech.provider_provenance.to_dict()
                if hasattr(assessment.speech.provider_provenance, "to_dict")
                else None
            )
            from hbllm.hcir.proposition import Proposition, TemporalValidity

            speech_evi = PerceptualEvidenceNode(
                id=speech_evi_id,
                proposition=Proposition(
                    subject=obs_id,
                    predicate="transcribed_as",
                    object_value=assessment.speech.transcript,
                    object_type="transcript",
                ),
                temporal_validity=TemporalValidity(
                    observed_at=start_t,
                    received_at=time.time(),
                ),
                modality="audio",
                evidence_type=EvidenceStrength.OBSERVATIONAL,
                strength=float(assessment.speech.confidence),
                epistemic_profile=epistemic_profile,
                provider_provenance=prov_dict,
                candidates=[
                    {
                        "label": assessment.speech.transcript,
                        "confidence": float(assessment.speech.confidence),
                    }
                ],
                payload={"transcript": assessment.speech.transcript},
            )
            self._graph.upsert_node(speech_evi)
            committed_ids.append(speech_evi_id)

            # Link Observation -> Evidence
            self._graph.add_edge(
                HCIREdge(
                    edge_type=HCIREdgeType.DERIVED_FROM,
                    sources=[obs_id],
                    targets=[speech_evi_id],
                )
            )

        # 4. Materialize Sound Event Evidence (if present)
        if assessment.events:
            event_evi_id = _generate_id("evi_sound_event")
            cand_list = [
                {
                    "label": getattr(ev, "event_type", getattr(ev, "label", "sound_event")),
                    "confidence": float(ev.confidence),
                }
                for ev in assessment.events
            ]
            prov_dict = (
                assessment.events[0].provider_provenance.to_dict()
                if hasattr(assessment.events[0].provider_provenance, "to_dict")
                else None
            )
            first_label = getattr(assessment.events[0], "event_type", getattr(assessment.events[0], "label", "sound_event"))
            from hbllm.hcir.proposition import Proposition, TemporalValidity

            event_evi = PerceptualEvidenceNode(
                id=event_evi_id,
                proposition=Proposition(
                    subject=obs_id,
                    predicate="classified_as",
                    object_value=first_label,
                    object_type="sound_event",
                ),
                temporal_validity=TemporalValidity(
                    observed_at=start_t,
                    received_at=time.time(),
                ),
                modality="audio",
                evidence_type=EvidenceStrength.OBSERVATIONAL,
                strength=float(assessment.events[0].confidence),
                epistemic_profile=epistemic_profile,
                provider_provenance=prov_dict,
                candidates=cand_list,
                payload={"event_type": first_label},
            )
            self._graph.upsert_node(event_evi)
            committed_ids.append(event_evi_id)

            # Link Observation -> Evidence
            self._graph.add_edge(
                HCIREdge(
                    edge_type=HCIREdgeType.DERIVED_FROM,
                    sources=[obs_id],
                    targets=[event_evi_id],
                )
            )

        logger.debug(
            "Ingested audio assessment into HCIR: obs=%s, total_nodes=%d",
            obs_id,
            len(committed_ids),
        )
        return committed_ids

    def ingest_visual_assessment(
        self,
        assessment: VisualAssessment,
    ) -> list[str]:
        """Materialize a VisualAssessment into HCIR ObservationNode and EvidenceNodes.

        Args:
            assessment: VisualAssessment from visual runtime.

        Returns:
            List of node IDs committed to the graph.
        """
        committed_ids: list[str] = []
        obs_id = (
            assessment.observation.observation_id
            if hasattr(assessment, "observation") and assessment.observation
            else _generate_id("vis_obs")
        )

        # 1. Map epistemic profile
        vis_profile = getattr(assessment, "epistemic_profile", None)
        if vis_profile:
            sensory_clarity = float(
                getattr(
                    vis_profile, "source_reliability", getattr(vis_profile, "sensory_clarity", 0.85)
                )
            )
            model_conf = float(
                getattr(
                    vis_profile,
                    "perceptual_similarity",
                    getattr(vis_profile, "model_confidence", 0.85),
                )
            )
            temp_stab = float(
                getattr(
                    vis_profile,
                    "evidence_strength",
                    getattr(vis_profile, "temporal_stability", 0.85),
                )
            )
            epistemic_profile = PerceptualEpistemicProfile(
                sensory_clarity=sensory_clarity,
                model_confidence=model_conf,
                temporal_stability=temp_stab,
            )
        else:
            epistemic_profile = PerceptualEpistemicProfile(
                sensory_clarity=0.85,
                model_confidence=0.85,
                temporal_stability=0.85,
            )

        # 2. Materialize VisualObservationNode
        caption_text = getattr(assessment, "caption", "") or ""
        if (
            not caption_text
            and hasattr(assessment, "candidate_concepts")
            and assessment.candidate_concepts
        ):
            caption_text = assessment.candidate_concepts[0].label

        vis_node = VisualObservationNode(
            id=obs_id,
            modality="visual",
            caption=caption_text,
            temporal_span={"start_time": time.time(), "end_time": time.time(), "duration": 0.1},
        )
        self._graph.upsert_node(vis_node)
        committed_ids.append(obs_id)

        # 3. Materialize Visual Evidence
        vis_evi_id = _generate_id("evi_visual")
        candidates = []
        if hasattr(assessment, "candidate_concepts") and assessment.candidate_concepts:
            candidates = [
                {
                    "label": c.label,
                    "score": float(
                        getattr(
                            c,
                            "best_similarity",
                            getattr(c, "mean_similarity", getattr(c, "similarity", 0.8)),
                        )
                    ),
                }
                for c in assessment.candidate_concepts
            ]
        elif hasattr(assessment, "candidates") and assessment.candidates:
            candidates = [
                {
                    "label": getattr(c, "label", str(c)),
                    "score": float(getattr(c, "confidence", 0.8)),
                }
                for c in assessment.candidates
            ]
        elif caption_text:
            candidates = [{"label": caption_text, "score": 0.85}]

        from hbllm.hcir.proposition import Proposition, TemporalValidity

        vis_evi = PerceptualEvidenceNode(
            id=vis_evi_id,
            proposition=Proposition(
                subject=obs_id,
                predicate="classified_as",
                object_value=caption_text or "visual_concept",
                object_type="visual_concept",
            ),
            temporal_validity=TemporalValidity(
                observed_at=time.time(),
                received_at=time.time(),
            ),
            modality="visual",
            evidence_type=EvidenceStrength.OBSERVATIONAL,
            strength=0.85,
            epistemic_profile=epistemic_profile,
            candidates=candidates,
            payload={"caption": caption_text},
        )
        self._graph.upsert_node(vis_evi)
        committed_ids.append(vis_evi_id)

        # Link Observation -> Evidence
        self._graph.add_edge(
            HCIREdge(
                edge_type=HCIREdgeType.DERIVED_FROM,
                sources=[obs_id],
                targets=[vis_evi_id],
            )
        )

        logger.debug(
            "Ingested visual assessment into HCIR: obs=%s, total_nodes=%d",
            obs_id,
            len(committed_ids),
        )
        return committed_ids

    def correlate_and_commit(
        self,
        window_seconds: float = 5.0,
    ) -> list[CorrelationCandidate]:
        """Find and commit cross-modal correlation edges for active observations.

        Uses CorrelationEngine to establish measurable, neutral CORRELATES_WITH
        hyperedges without asserting causal relationships.
        """
        candidates: list[CorrelationCandidate] = []
        visual_nodes: list[VisualObservationNode] = []
        audio_nodes: list[AudioObservationNode] = []

        # Collect recent visual and audio observations
        for _node in self._graph.all_nodes():
            node = self._graph.get_node(_node.id)
            if isinstance(node, VisualObservationNode):
                visual_nodes.append(node)
            elif isinstance(node, AudioObservationNode):
                audio_nodes.append(node)

        # Check all cross-modal pairs
        for vis in visual_nodes:
            vis_start = vis.temporal_span.get("start_time", 0.0)
            vis_end = vis.temporal_span.get("end_time", vis_start)
            vis_env = ObservationEnvelope(
                observation_id=vis.id,
                modality="visual",
                start_time=vis_start,
                end_time=vis_end,
            )

            for aud in audio_nodes:
                aud_start = aud.start_time or aud.temporal_span.get("start_time", 0.0)
                aud_end = aud.end_time or aud.temporal_span.get("end_time", aud_start)
                aud_env = ObservationEnvelope(
                    observation_id=aud.id,
                    modality="audio",
                    start_time=aud_start,
                    end_time=aud_end,
                )

                corr = self._correlation_engine.correlate(vis_env, aud_env)
                if corr and corr.score > 0.3:
                    cand = CorrelationCandidate(
                        source_obs_id=vis.id,
                        target_obs_id=aud.id,
                        source_modality=PerceptualModality.VISUAL,
                        target_modality=PerceptualModality.AUDIO,
                        temporal_overlap=corr.temporal_overlap,
                        spatial_overlap=corr.spatial_overlap,
                        delta_time_ms=corr.delta_time_ms,
                        confidence=corr.score,
                        rationale=f"Cross-modal alignment (score={corr.score:.2f}, Δt={corr.delta_time_ms:.1f}ms)",
                    )
                    candidates.append(cand)

                    # Commit CORRELATES_WITH hyperedge to HCIR
                    edge = HCIREdge(
                        edge_type=HCIREdgeType.CORRELATES_WITH,
                        sources=[vis.id],
                        targets=[aud.id],
                        weight=corr.score,
                        properties={
                            "confidence": corr.score,
                            "temporal_overlap": corr.temporal_overlap,
                            "spatial_overlap": corr.spatial_overlap,
                            "delta_time_ms": corr.delta_time_ms,
                            "created_by": "PerceptionEpistemicBridge",
                        },
                    )
                    try:
                        self._graph.add_edge(edge)
                    except ValueError:
                        pass  # Edge may already exist

        logger.debug(
            "Correlated cross-modal observations: found %d candidate edges",
            len(candidates),
        )
        return candidates

    # ── Universal Modality-Neutral Ingestion ─────────────────────────────

    def ingest_perceptual_evidence(
        self,
        evidence: PerceptualEvidenceNode,
    ) -> list[str]:
        """Ingest a modality-neutral PerceptualEvidenceNode into HCIR.

        This is the UNIVERSAL ingestion path.  Any perception provider
        can produce a ``PerceptualEvidenceNode`` (via the
        ``EvidenceNormalizer``), and this method commits it to HCIR.

        The existing ``ingest_audio_assessment()`` and
        ``ingest_visual_assessment()`` methods remain for backward
        compatibility with the existing modality-specific pipeline.

        Architecture::

            Provider → Observation → EvidenceNormalizer →
            PerceptualEvidenceNode → ingest_perceptual_evidence() → HCIR

        Args:
            evidence: A ``PerceptualEvidenceNode`` produced by the
                ``EvidenceNormalizer``.

        Returns:
            List of node IDs committed to the graph (typically one).
        """
        committed_ids: list[str] = []

        # Commit PerceptualEvidenceNode to HCIR
        self._graph.upsert_node(evidence)
        committed_ids.append(evidence.id)

        logger.debug(
            "Ingested PerceptualEvidenceNode: id=%s modality=%s proposition=(%s %s %s)",
            evidence.id,
            evidence.modality,
            evidence.proposition.subject,
            evidence.proposition.predicate,
            evidence.proposition.object_value,
        )

        return committed_ids
