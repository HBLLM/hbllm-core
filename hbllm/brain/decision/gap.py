"""Epistemic Gaps, Observation-Grounded Hypotheses, and Shannon Entropy for A19.

Represents uncertainty over structured hypotheses, calculates Shannon entropy,
and performs Bayesian posterior updates based on generative observation likelihoods.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode


@dataclass
class HypothesisOption:
    """An observationally grounded hypothesis candidate within an EpistemicGap."""

    hypothesis_id: str
    label: str
    prior: float = 0.5
    # Generative model: P(observation_token | this_hypothesis, probe_action)
    # e.g. {"roll": 0.95, "no_roll": 0.05}
    predicted_observations: dict[str, float] = field(default_factory=dict)
    posterior: float = 0.5
    evidence_count: int = 0
    grounded_properties: dict[str, Any] = field(default_factory=dict)

    def likelihood(self, observation: str) -> float:
        """P(observation | hypothesis). Defaults to uniform epsilon if unspecified."""
        return self.predicted_observations.get(observation, 0.01)


@dataclass
class EpistemicGap:
    """An explicit uncertainty representation in the HCIR World Model."""

    gap_id: str = field(default_factory=lambda: f"gap_{uuid.uuid4().hex[:8]}")
    domain: str = "geometry"  # geometry, containment, identity, relation, concept
    source_node_ids: list[str] = field(default_factory=list)
    hypotheses: list[HypothesisOption] = field(default_factory=list)
    decision_relevance: float = 0.5  # 0.0 (irrelevant) to 1.0 (critical path)
    resolution_threshold: float = 0.90  # Required posterior confidence to commit
    status: str = "UNCERTAIN"  # UNCERTAIN -> TENTATIVE -> SUPPORTED -> GROUNDED -> COMMITTED

    def __post_init__(self) -> None:
        self._normalize_priors()

    def _normalize_priors(self) -> None:
        """Ensure hypothesis priors sum to 1.0."""
        total = sum(h.prior for h in self.hypotheses)
        if total > 0.0:
            for h in self.hypotheses:
                h.prior = h.prior / total
                h.posterior = h.prior

    @property
    def entropy(self) -> float:
        """Shannon entropy H(H) = -sum P(h) * log2(P(h))."""
        h_val = 0.0
        for h in self.hypotheses:
            p = h.posterior
            if p > 1e-9:
                h_val -= p * math.log2(p)
        return max(0.0, h_val)

    @property
    def leading_hypothesis(self) -> HypothesisOption | None:
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.posterior)

    @property
    def is_resolved(self) -> bool:
        leading = self.leading_hypothesis
        return leading is not None and leading.posterior >= self.resolution_threshold

    def update_with_observation(self, observation: str) -> None:
        """Perform Bayesian posterior update: P(h|o) = P(o|h)*P(h) / P(o)."""
        # 1. Compute marginal likelihood P(o) = sum P(o|h_i) * P(h_i)
        marginal_p = sum(h.likelihood(observation) * h.posterior for h in self.hypotheses)
        if marginal_p < 1e-9:
            return  # Observation had zero probability under all hypotheses

        # 2. Update posteriors
        for h in self.hypotheses:
            like = h.likelihood(observation)
            unnorm_post = like * h.posterior
            h.posterior = unnorm_post / marginal_p
            h.evidence_count += 1

        # 3. Update lifecycle status
        leading = self.leading_hypothesis
        if leading is not None:
            if leading.posterior >= self.resolution_threshold:
                self.status = "GROUNDED"
            elif leading.posterior >= 0.75:
                self.status = "SUPPORTED"
            elif leading.posterior >= 0.60:
                self.status = "TENTATIVE"
            else:
                self.status = "UNCERTAIN"


class EpistemicGapScanner:
    """Scans canonical HCIR CognitiveGraph to identify epistemic gaps and uncertainties."""

    def scan_graph(self, graph: CognitiveGraph, active_goal_nodes: list[str] | None = None) -> list[EpistemicGap]:
        """Identify entities with ungrounded geometry, occluded contents, or tentative states."""
        gaps: list[EpistemicGap] = []
        active_goals = set(active_goal_nodes or [])

        for node in graph.all_nodes():
            if isinstance(node, PhysicalEntityNode):
                props = getattr(node, "properties", None) or getattr(node, "observed_properties", {}) or {}

                # 1. Geometry / Shape uncertainty (only if shape is explicitly unknown, or missing on non-standard object)
                shape = props.get("shape", "")
                is_unknown_geom = (
                    "geometry" not in props
                    and "surface" not in props
                    and (shape == "unknown" or (not shape and node.entity_type not in ("box", "table", "shelf", "tray", "floor", "cube", "block")))
                )
                if is_unknown_geom:
                    relevance = 0.95 if node.id in active_goals else 0.50
                    gap = EpistemicGap(
                        domain="geometry",
                        source_node_ids=[node.id],
                        decision_relevance=relevance,
                        resolution_threshold=0.90,
                        hypotheses=[
                            HypothesisOption(
                                hypothesis_id=f"{node.id}_flat",
                                label="FLAT",
                                prior=0.5,
                                predicted_observations={"roll": 0.05, "no_roll": 0.95},
                                grounded_properties={"geometry": "flat", "surface": "flat"},
                            ),
                            HypothesisOption(
                                hypothesis_id=f"{node.id}_convex",
                                label="CONVEX",
                                prior=0.5,
                                predicted_observations={"roll": 0.95, "no_roll": 0.05},
                                grounded_properties={"geometry": "convex", "surface": "convex"},
                            ),
                        ],
                    )
                    gaps.append(gap)

                # 2. Container containment / lid state uncertainty
                if node.entity_type in ("box", "container") and "is_closed" not in props:
                    relevance = 0.85 if node.id in active_goals else 0.30
                    gap = EpistemicGap(
                        domain="containment",
                        source_node_ids=[node.id],
                        decision_relevance=relevance,
                        resolution_threshold=0.85,
                        hypotheses=[
                            HypothesisOption(
                                hypothesis_id=f"{node.id}_open",
                                label="OPEN",
                                prior=0.5,
                                predicted_observations={"interior_visible": 0.90, "lid_detected": 0.10},
                                grounded_properties={"is_closed": False},
                            ),
                            HypothesisOption(
                                hypothesis_id=f"{node.id}_closed",
                                label="CLOSED",
                                prior=0.5,
                                predicted_observations={"interior_visible": 0.05, "lid_detected": 0.95},
                                grounded_properties={"is_closed": True},
                            ),
                        ],
                    )
                    gaps.append(gap)

        return gaps
