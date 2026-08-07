"""Curiosity Engine — self-directed investigation with discovery economics.

Decides *what* to investigate when there is no user prompt.
Integrates discovery economics to optimize for value, not just uncertainty::

    Priority = (Info Gain × Impact × Novelty × Strategic Relevance)
               ──────────────────────────────────────────────────────
               (CPU Cost + Time Cost + Risk + Attention Cost)

Sometimes the optimal action is: **don't investigate**.

An autonomous scientist that spends 3 days improving a belief by 0.2%
is irrational.  The CuriosityEngine prevents this.

Architecture::

    CuriosityEngine
        ├── Scan unknowns              → CuriositySignal per unknown
        ├── Scan contradictions        → CuriositySignal per contradiction
        ├── Scan anomalies             → CuriositySignal per anomaly
        ├── Estimate value of knowing  → KnowledgeValue analysis
        ├── Apply budget constraints   → InvestigationBudget
        └── Rank by EV / EC           → sorted investigation list

Usage::

    curiosity = CuriosityEngine(graph=graph, reputation=tracker)
    signals = await curiosity.prioritize_investigations(budget)
    if signals:
        print(f"Investigate: {signals[0].description}")
    else:
        print("Nothing worth investigating right now")
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.epistemics.interfaces import CuriositySignal, InvestigationBudget
from hbllm.hcir.graph import (
    CognitiveGraph,
    HypothesisNode,
    UnknownNode,
)
from hbllm.hcir.types import DiscoveryTrigger, KnowledgeValue

logger = logging.getLogger(__name__)

#: Minimum investigation score to be worth pursuing.
_MIN_INVESTIGATION_SCORE = 0.15


class CuriosityEngine:
    """Self-directed investigation prioritization with discovery economics.

    Implements the ``ICuriosityEngine`` protocol.

    The engine scans all known unknowns, contradictions, and anomalies,
    assigns each an investigation priority based on expected value
    divided by expected cost, and returns a ranked list.

    The engine is domain-neutral — it uses ``KnowledgeValue`` and
    ``InvestigationBudget`` to prioritize, never domain-specific content.
    """

    def __init__(
        self,
        graph: CognitiveGraph,
        contradiction_engine: Any | None = None,
        reputation_tracker: Any | None = None,
    ) -> None:
        """Initialize the curiosity engine.

        Args:
            graph: The shared HCIR cognitive graph.
            contradiction_engine: Optional ContradictionEngine for
                scanning anomalies and unexpected outcomes.
            reputation_tracker: Optional for source trust context.
        """
        self._graph = graph
        self._contradiction_engine = contradiction_engine
        self._reputation = reputation_tracker

    async def prioritize_investigations(
        self,
        budget: InvestigationBudget | None = None,
    ) -> list[CuriositySignal]:
        """Rank all known unknowns and signals by investigation priority.

        Priority formula::

            score = (info_gain × impact × novelty × strategic_relevance)
                    ────────────────────────────────────────────────────
                    (cost + attention_cost + risk_penalty)

        Signals below ``_MIN_INVESTIGATION_SCORE`` are filtered out.
        Sometimes the best action is to do nothing.

        Args:
            budget: Resource constraints.  Signals exceeding the
                budget are filtered.

        Returns:
            Sorted list of CuriositySignals (highest priority first).
        """
        budget = budget or InvestigationBudget()
        signals: list[CuriositySignal] = []

        # Scan unknowns
        signals.extend(self._scan_unknowns())

        # Scan hypotheses needing testing
        signals.extend(self._scan_untested_hypotheses())

        # Scan for anomalies via contradiction engine
        if self._contradiction_engine is not None:
            try:
                anomalies = await self._contradiction_engine.scan_for_anomalies()
                signals.extend(anomalies)

                unexpected = await self._contradiction_engine.detect_unexpected_outcomes()
                signals.extend(unexpected)
            except Exception as exc:
                logger.warning("Contradiction engine scan failed: %s", exc)

        # Score each signal
        scored: list[tuple[float, CuriositySignal]] = []
        for signal in signals:
            score = self._compute_investigation_score(signal, budget)
            if score >= _MIN_INVESTIGATION_SCORE:
                scored.append((score, signal))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        result = [signal for _, signal in scored]
        logger.info(
            "Prioritized %d investigations (%d candidates, %d filtered)",
            len(result),
            len(signals),
            len(signals) - len(result),
        )
        return result

    async def estimate_value_of_knowing(
        self,
        unknown_id: str,
    ) -> float:
        """Estimate the value of resolving a specific unknown.

        Uses the ``KnowledgeValue`` attached to the unknown node.

        Returns:
            A value score [0.0, 1.0] where higher = more valuable.
        """
        node = self._graph.get_node(unknown_id)
        if not isinstance(node, UnknownNode):
            return 0.0
        return node.knowledge_value.derived_value

    async def generate_spontaneous_unknowns(self) -> list[str]:
        """Scan for uncertainty hotspots and generate new UnknownNodes.

        Identifies hypotheses with high uncertainty that lack
        corresponding unknowns.

        Returns:
            List of newly created UnknownNode IDs.
        """
        new_unknowns: list[str] = []

        for _node in self._graph.all_nodes():
            node_id = _node.id
            node = self._graph.get_node(node_id)
            if not isinstance(node, HypothesisNode):
                continue

            # Hypothesis with high uncertainty and no linked experiments
            if (
                node.uncertainty.confidence < 0.5
                and not node.linked_experiments
                and not node.linked_predictions
            ):
                unknown = UnknownNode(
                    question=f"Why is '{node.claim[:80]}' uncertain? "
                    f"What evidence would resolve it?",
                    context=f"Hypothesis {node.id} has confidence "
                    f"{node.uncertainty.confidence:.2f} but no "
                    f"experiments or predictions.",
                    importance=0.6,
                    research_program_id=node.research_program_id,
                    trigger=DiscoveryTrigger.CURIOSITY,
                    knowledge_value=KnowledgeValue(
                        novelty=node.novelty,
                        impact=node.predicted_impact,
                        strategic_relevance=0.6,
                    ),
                )
                self._graph.upsert_node(unknown)
                new_unknowns.append(unknown.id)

        if new_unknowns:
            logger.info(
                "Generated %d spontaneous unknowns from uncertainty hotspots",
                len(new_unknowns),
            )

        return new_unknowns

    # ── Scanning Methods ───────────────────────────────────────────────

    def _scan_unknowns(self) -> list[CuriositySignal]:
        """Scan all UnknownNodes for investigation signals."""
        signals: list[CuriositySignal] = []

        for _node in self._graph.all_nodes():
            node_id = _node.id
            node = self._graph.get_node(node_id)
            if not isinstance(node, UnknownNode):
                continue

            kv = node.knowledge_value
            signals.append(
                CuriositySignal(
                    unknown_id=node.id,
                    trigger=node.trigger,
                    source_engine="curiosity_engine",
                    source_id=node.id,
                    estimated_info_gain=1.0 - node.estimated_difficulty,
                    estimated_impact=kv.impact,
                    estimated_cost=kv.cost,
                    description=f"Knowledge gap: {node.question[:80]}",
                )
            )

        return signals

    def _scan_untested_hypotheses(self) -> list[CuriositySignal]:
        """Scan for hypotheses that need testing."""
        signals: list[CuriositySignal] = []

        for _node in self._graph.all_nodes():
            node_id = _node.id
            node = self._graph.get_node(node_id)
            if not isinstance(node, HypothesisNode):
                continue

            # Only consider hypotheses without experiments
            if node.linked_experiments:
                continue

            kv = node.knowledge_value
            signals.append(
                CuriositySignal(
                    trigger=DiscoveryTrigger.KNOWLEDGE_GAP,
                    source_engine="curiosity_engine",
                    source_id=node.id,
                    estimated_info_gain=node.testability,
                    estimated_impact=kv.impact,
                    estimated_cost=1.0 - node.testability,  # Hard to test = expensive
                    description=f"Untested hypothesis: {node.claim[:80]}",
                )
            )

        return signals

    # ── Scoring ────────────────────────────────────────────────────────

    def _compute_investigation_score(
        self,
        signal: CuriositySignal,
        budget: InvestigationBudget,
    ) -> float:
        """Compute the investigation priority score.

        score = (info_gain × impact) / (cost + attention + risk_penalty)

        This implements discovery economics: expected value / expected cost.
        """
        # Numerator: expected value of investigating
        info_gain = max(0.01, signal.estimated_info_gain)
        impact = max(0.01, signal.estimated_impact)
        value = info_gain * impact

        # Denominator: expected cost
        investigation_cost = max(0.01, signal.estimated_cost)
        attention = budget.attention_cost
        risk_penalty = 0.0

        if budget.risk_tolerance < 0.3:
            risk_penalty = 0.2  # Conservative budget penalizes risky investigations

        total_cost = investigation_cost + attention + risk_penalty

        score = value / total_cost

        # Normalize to [0.0, 1.0] range
        return min(1.0, max(0.0, score))
