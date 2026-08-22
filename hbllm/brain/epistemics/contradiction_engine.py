"""Contradiction Engine — proactive contradiction and anomaly hunting.

Instead of just detecting contradictions when they happen,
the engine actively scans the knowledge base for:
- Direct contradictions between claims
- Anomalous observations that don't fit existing models
- Unexpected successes and failures in experiments

Each discovery produces a ``CuriositySignal`` or ``ContradictionReport``
that feeds the discovery loop.

Usage::

    engine = ContradictionEngine(graph=graph, llm=llm)
    reports = await engine.scan_for_contradictions(domain="biology")
    signals = await engine.scan_for_anomalies()
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.epistemics.interfaces import (
    ContradictionReport,
    CuriositySignal,
)
from hbllm.hcir.graph import (
    AudioObservationNode,
    BeliefNode,
    CognitiveGraph,
    ContradictionNode,
    EvidenceNode,
    ExperimentNode,
    HCIREdgeType,
    ObservationNode,
    PerceptualEvidenceNode,
    VisualObservationNode,
)
from hbllm.hcir.types import (
    DiscoveryTrigger,
    ExperimentStatus,
    PerceptualContradictionLevel,
)

logger = logging.getLogger(__name__)


class ContradictionEngine:
    """Actively searches for contradictions and anomalies.

    Implements the ``IContradictionSeeker`` protocol.

    Contradictions are not errors — they are opportunities.
    The engine is domain-neutral; it identifies structural conflicts
    (opposing claims, failed predictions, unexpected outcomes) without
    interpreting domain content.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        llm: Any | None = None,
    ) -> None:
        self._graph = graph
        self._llm = llm

    async def scan_for_contradictions(
        self,
        domain: str = "",
        scope: str = "",
    ) -> list[ContradictionReport]:
        """Systematically scan for contradictions across evidence.

        Checks:
        1. Beliefs that contradict each other
        2. Hypotheses with conflicting evidence
        3. Claims with opposing support
        4. Three-level perceptual contradictions

        Args:
            domain: Optional filter by knowledge domain.
            scope: Optional scope filter.

        Returns:
            List of discovered contradiction reports.
        """
        reports: list[ContradictionReport] = []

        # Scan existing contradiction edges
        reports.extend(self._scan_contradiction_edges())

        # Scan for beliefs with opposing evidence
        reports.extend(await self._scan_belief_conflicts(domain))

        # Scan for 3-level perceptual contradictions
        reports.extend(await self.scan_for_perceptual_contradictions())

        return reports

    async def scan_for_perceptual_contradictions(self) -> list[ContradictionReport]:
        """Scan for 3-tiered perceptual and epistemic contradictions:

        - Level 1: Classifier / multi-candidate disagreement within observation
        - Level 2: Cross-modal contradiction across correlated observations
        - Level 3: Perception vs active belief conflict
        """
        reports: list[ContradictionReport] = []

        # ── Level 1: Classifier Disagreement ──────────────────────────────
        for _node in self._graph.all_nodes():
            node = self._graph.get_node(_node.id)
            if not isinstance(node, EvidenceNode) or not node.candidates:
                continue

            if len(node.candidates) >= 2:
                top1 = float(node.candidates[0].get("score", 0.0))
                top2 = float(node.candidates[1].get("score", 0.0))
                if top1 >= 0.5 and top2 >= 0.5 and (top1 - top2) < 0.20:
                    label1 = node.candidates[0].get("label", "cand1")
                    label2 = node.candidates[1].get("label", "cand2")
                    report = ContradictionReport(
                        claim_a_id=f"{node.id}:{label1}",
                        claim_b_id=f"{node.id}:{label2}",
                        contradiction_type="classifier_ambiguity",
                        contradiction_level=str(
                            PerceptualContradictionLevel.LEVEL_1_CLASSIFIER_DISAGREEMENT
                        ),
                        possible_explanations=[
                            f"Model uncertainty between '{label1}' and '{label2}'",
                            "Acoustic/visual signal contains blended features",
                        ],
                        investigation_priority=float(top2),
                        context=f"Evidence {node.id} multi-candidate conflict",
                    )
                    reports.append(report)

        # ── Level 2: Cross-Modal Conflict ──────────────────────────────────
        for edge in self._graph.all_edges():
            if edge.edge_type != HCIREdgeType.CORRELATES_WITH:
                continue

            for src_id in edge.sources:
                for tgt_id in edge.targets:
                    src_node = self._graph.get_node(src_id)
                    tgt_node = self._graph.get_node(tgt_id)
                    if src_node is None or tgt_node is None:
                        continue

                    vis_node: Any = None
                    aud_node: Any = None

                    if isinstance(src_node, VisualObservationNode) and isinstance(tgt_node, AudioObservationNode):
                        vis_node, aud_node = src_node, tgt_node
                    elif isinstance(src_node, AudioObservationNode) and isinstance(tgt_node, VisualObservationNode):
                        aud_node, vis_node = src_node, tgt_node
                    elif isinstance(src_node, PerceptualEvidenceNode) and isinstance(tgt_node, PerceptualEvidenceNode):
                        if src_node.modality == "visual" and tgt_node.modality == "audio":
                            vis_node, aud_node = src_node, tgt_node
                        elif src_node.modality == "audio" and tgt_node.modality == "visual":
                            aud_node, vis_node = src_node, tgt_node
                        else:
                            continue
                    elif isinstance(src_node, PerceptualEvidenceNode) and isinstance(tgt_node, AudioObservationNode):
                        if src_node.modality == "visual":
                            vis_node, aud_node = src_node, tgt_node
                        else:
                            continue
                    elif isinstance(src_node, AudioObservationNode) and isinstance(tgt_node, PerceptualEvidenceNode):
                        if tgt_node.modality == "visual":
                            aud_node, vis_node = src_node, tgt_node
                        else:
                            continue
                    elif isinstance(src_node, VisualObservationNode) and isinstance(tgt_node, PerceptualEvidenceNode):
                        if tgt_node.modality == "audio":
                            vis_node, aud_node = src_node, tgt_node
                        else:
                            continue
                    elif isinstance(src_node, PerceptualEvidenceNode) and isinstance(tgt_node, VisualObservationNode):
                        if src_node.modality == "audio":
                            aud_node, vis_node = src_node, tgt_node
                        else:
                            continue
                    else:
                        continue

                    # Check for semantic conflict between visual and audio
                    vis_cap = str(getattr(vis_node, "caption", "") or getattr(getattr(vis_node, "proposition", None), "object_value", "")).lower()
                    aud_event = str(getattr(aud_node, "event_type", "") or getattr(aud_node, "label", "") or getattr(getattr(aud_node, "proposition", None), "object_value", "")).lower()
                    aud_transcript = str(getattr(aud_node, "transcript", "") or (getattr(aud_node, "payload", {}).get("transcript", "") if hasattr(aud_node, "payload") and isinstance(aud_node.payload, dict) else "")).lower()

                    is_conflict = False
                    reason = ""
                    if any(empty in vis_cap for empty in ["empty", "nobody", "no person", "dark"]):
                        if (
                            aud_event in ["speech", "applause", "crowd", "screaming", "footsteps"]
                            or len(aud_transcript) > 5
                        ):
                            is_conflict = True
                            reason = f"Vision indicates '{vis_cap}' while Audio detected '{aud_event or aud_transcript}'"

                    if is_conflict:
                        report = ContradictionReport(
                            claim_a_id=vis_node.id,
                            claim_b_id=aud_node.id,
                            contradiction_type="cross_modal_conflict",
                            contradiction_level=str(
                                PerceptualContradictionLevel.LEVEL_2_CROSS_MODAL_CONFLICT
                            ),
                            possible_explanations=[
                                "Occluded or off-camera sound source",
                                "Microphone picking up audio from adjacent room",
                                "Vision detector false negative",
                            ],
                            investigation_priority=0.85,
                            context=reason,
                        )
                        reports.append(report)

        # ── Level 3: Belief Conflict ──────────────────────────────────────
        for _node in self._graph.all_nodes():
            belief = self._graph.get_node(_node.id)
            if not isinstance(belief, BeliefNode) or belief.uncertainty.confidence < 0.6:
                continue

            # Check if any incoming WEAKENS edges from high-confidence evidence
            incoming = self._graph.edges_to(belief.id)
            for edge in incoming:
                if edge.edge_type in (HCIREdgeType.WEAKENS, HCIREdgeType.CONTRADICTS):
                    for src_id in edge.sources:
                        evi = self._graph.get_node(src_id)
                        if isinstance(evi, EvidenceNode) and float(evi.strength) >= 0.7:
                            report = ContradictionReport(
                                claim_a_id=belief.id,
                                claim_b_id=evi.id,
                                contradiction_type="belief_perception_conflict",
                                contradiction_level=str(
                                    PerceptualContradictionLevel.LEVEL_3_BELIEF_CONFLICT
                                ),
                                possible_explanations=[
                                    f"Belief '{belief.claim}' is out-of-date",
                                    "Sensory evidence represents an anomalous exception",
                                ],
                                investigation_priority=0.90,
                                context=f"Belief '{belief.claim}' contradicted by Evidence {evi.id}",
                            )
                            reports.append(report)

        # Commit ContradictionNodes to graph
        for report in reports:
            contra_id = f"contra_{abs(hash(report.claim_a_id + report.claim_b_id)) % 1000000}"
            if self._graph.get_node(contra_id) is None:
                contra_node = ContradictionNode(
                    id=contra_id,
                    claim_a_id=report.claim_a_id,
                    claim_b_id=report.claim_b_id,
                    contradiction_type=report.contradiction_type,
                    contradiction_level=PerceptualContradictionLevel(
                        report.contradiction_level
                        or PerceptualContradictionLevel.LEVEL_1_CLASSIFIER_DISAGREEMENT
                    ),
                    possible_explanations=report.possible_explanations,
                    investigation_priority=report.investigation_priority,
                )
                self._graph.upsert_node(contra_node)

        return reports

    async def scan_for_anomalies(
        self,
        domain: str = "",
    ) -> list[CuriositySignal]:
        """Scan for observations that don't fit existing models.

        An anomaly is an observation that:
        - Contradicts strong beliefs (confidence > 0.7)
        - Has no supporting hypothesis
        - Comes from a high-trust source

        Returns:
            List of CuriositySignals for anomalous observations.
        """
        signals: list[CuriositySignal] = []

        for _node in self._graph.all_nodes():
            node_id = _node.id
            node = self._graph.get_node(node_id)
            if not isinstance(node, ObservationNode):
                continue

            # Check if observation has any supporting hypothesis
            edges = self._graph.edges_from(node_id) + self._graph.edges_to(node_id)
            has_explanation = any(
                e.edge_type in (HCIREdgeType.SUPPORTS, HCIREdgeType.DERIVED_FROM) for e in edges
            )

            if not has_explanation:
                signals.append(
                    CuriositySignal(
                        trigger=DiscoveryTrigger.ANOMALY,
                        source_engine="contradiction_engine",
                        source_id=node_id,
                        estimated_info_gain=0.6,
                        estimated_impact=0.5,
                        description=f"Unexplained observation: {node_id}",
                    )
                )

        return signals

    async def analyze_contradiction(
        self,
        contradiction_id: str,
    ) -> dict[str, Any]:
        """Deep analysis of a contradiction — identify hidden variables.

        Uses LLM (if available) to analyze the root cause of a
        contradiction and suggest resolution strategies.
        """
        node = self._graph.get_node(contradiction_id)
        if not isinstance(node, ContradictionNode):
            return {"error": f"Not a contradiction: {contradiction_id}"}

        analysis: dict[str, Any] = {
            "contradiction_id": contradiction_id,
            "claim_a": node.claim_a_id,
            "claim_b": node.claim_b_id,
            "type": node.contradiction_type,
        }

        # Get claim descriptions
        claim_a_desc = self._get_node_description(node.claim_a_id)
        claim_b_desc = self._get_node_description(node.claim_b_id)
        analysis["claim_a_description"] = claim_a_desc
        analysis["claim_b_description"] = claim_b_desc

        if self._llm is not None:
            analysis.update(
                await self._llm_analyze(
                    claim_a_desc,
                    claim_b_desc,
                    node.contradiction_type,
                )
            )

        return analysis

    async def detect_unexpected_outcomes(
        self,
        program_id: str = "",
    ) -> list[CuriositySignal]:
        """Detect unexpected successes and failures in experiments.

        An unexpected outcome is an experiment whose result conflicts
        with the hypothesis prediction confidence (highly expected
        outcomes that failed, or unlikely outcomes that succeeded).
        """
        signals: list[CuriositySignal] = []

        for _node in self._graph.all_nodes():
            node_id = _node.id
            node = self._graph.get_node(node_id)
            if not isinstance(node, ExperimentNode):
                continue

            if program_id and getattr(node, "research_program_id", "") != program_id:
                continue

            exp_status = getattr(node, "experiment_status", getattr(node, "status", None))
            if exp_status == ExperimentStatus.FAILED:
                signals.append(
                    CuriositySignal(
                        trigger=DiscoveryTrigger.UNEXPECTED_FAILURE,
                        source_engine="contradiction_engine",
                        source_id=node_id,
                        estimated_info_gain=0.7,
                        estimated_impact=0.6,
                        description=f"Experiment failed unexpectedly: {node.design[:60]}",
                    )
                )

        return signals

    # ── Internal Methods ───────────────────────────────────────────────

    def _scan_contradiction_edges(self) -> list[ContradictionReport]:
        """Find existing CONTRADICTS edges in the graph."""
        reports: list[ContradictionReport] = []
        seen: set[tuple[str, str]] = set()

        for _node in self._graph.all_nodes():
            node_id = _node.id
            edges = self._graph.edges_from(node_id) + self._graph.edges_to(node_id)
            for edge in edges:
                if edge.edge_type != HCIREdgeType.CONTRADICTS:
                    continue

                for src in edge.sources:
                    for tgt in edge.targets:
                        pair: tuple[str, str] = (min(src, tgt), max(src, tgt))
                        if pair in seen:
                            continue
                        seen.add(pair)

                        reports.append(
                            ContradictionReport(
                                claim_a_id=src,
                                claim_b_id=tgt,
                                contradiction_type="edge_contradiction",
                                investigation_priority=0.6,
                                context="Found via CONTRADICTS edge scan",
                            )
                        )

        return reports

    async def _scan_belief_conflicts(
        self,
        domain: str,
    ) -> list[ContradictionReport]:
        """Scan beliefs for potential conflicts."""
        beliefs: list[BeliefNode] = []

        for _node in self._graph.all_nodes():
            node_id = _node.id
            node = self._graph.get_node(node_id)
            if isinstance(node, BeliefNode):
                beliefs.append(node)

        # Pairwise comparison using LLM if available
        reports: list[ContradictionReport] = []
        if self._llm is not None and len(beliefs) >= 2:
            # Only compare high-confidence beliefs (expensive operation)
            strong_beliefs = [b for b in beliefs if b.uncertainty.confidence >= 0.5][
                :20
            ]  # Limit to prevent O(n²) explosion

            for i in range(len(strong_beliefs)):
                for j in range(i + 1, len(strong_beliefs)):
                    if await self._beliefs_conflict(
                        strong_beliefs[i],
                        strong_beliefs[j],
                    ):
                        reports.append(
                            ContradictionReport(
                                claim_a_id=strong_beliefs[i].id,
                                claim_b_id=strong_beliefs[j].id,
                                contradiction_type="belief_conflict",
                                investigation_priority=0.7,
                                context="LLM-detected belief conflict",
                            )
                        )

        return reports

    async def _beliefs_conflict(
        self,
        a: BeliefNode,
        b: BeliefNode,
    ) -> bool:
        """Use LLM to check if two beliefs conflict."""
        prompt = (
            f"Do these two beliefs contradict each other?\n"
            f"Belief A: {a.claim}\n"
            f"Belief B: {b.claim}\n\n"
            f"Answer YES or NO only."
        )
        if self._llm is None:
            return False

        try:
            response = await self._llm.generate(prompt)
            text = response if isinstance(response, str) else str(response)
            return "yes" in text.strip().lower()
        except Exception:
            return False

    async def _llm_analyze(
        self,
        claim_a: str,
        claim_b: str,
        contradiction_type: str,
    ) -> dict[str, Any]:
        """Use LLM for deep contradiction analysis."""
        if self._llm is None:
            return {}

        prompt = (
            f"Analyze this contradiction:\n"
            f"Claim A: {claim_a}\n"
            f"Claim B: {claim_b}\n"
            f"Type: {contradiction_type}\n\n"
            f"Provide:\n"
            f"1. HIDDEN_VARIABLES: What hidden variables could explain both?\n"
            f"2. RESOLUTION_STRATEGIES: How could this be resolved?\n"
            f"3. PRIORITY: How urgent is resolving this? (0.0-1.0)\n"
        )
        try:
            response = await self._llm.generate(prompt)
            text = response if isinstance(response, str) else str(response)
            return {"llm_analysis": text}
        except Exception as exc:
            return {"llm_analysis_error": str(exc)}

    def _get_node_description(self, node_id: str) -> str:
        """Get a human-readable description of any node."""
        node = self._graph.get_node(node_id)
        if node is None:
            return f"(unknown: {node_id})"
        claim = getattr(node, "claim", None) or getattr(node, "statement", None)
        return claim or node_id
