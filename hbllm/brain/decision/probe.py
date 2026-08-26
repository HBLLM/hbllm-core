"""Discriminative Epistemic Probes and Value-of-Information (VoI) for A19.

Evaluates candidate probes by calculating exact expected information gain (entropy reduction)
and scaling by decision-relevance to compute the true Value of Information (VoI).
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any

from hbllm.brain.decision.gap import EpistemicGap


@dataclass
class EpistemicProbe:
    """A physical action sequence designed to generate observations that discriminate hypotheses."""

    probe_id: str = field(default_factory=lambda: f"probe_{uuid.uuid4().hex[:8]}")
    target_gap_id: str = ""
    description: str = ""
    action_sequence: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    possible_observations: list[str] = field(default_factory=list)
    cost: float = 0.10  # Action cost / effort (0.0 to 1.0)
    risk: float = 0.05  # Inherent safety risk (0.0 to 1.0)
    reversibility: float = 0.90  # 1.0 = fully reversible (e.g. looking), 0.1 = destructive

    def compute_expected_information_gain(self, gap: EpistemicGap) -> float:
        """Compute exact expected information gain: IG = H(H) - sum_o P(o) * H(H|o)."""
        current_h = gap.entropy
        if current_h < 1e-6 or not self.possible_observations:
            return 0.0

        expected_conditional_entropy = 0.0

        for obs in self.possible_observations:
            # P(o) = sum_h P(o|h) * P(h)
            p_obs = sum(h.likelihood(obs) * h.posterior for h in gap.hypotheses)
            if p_obs < 1e-9:
                continue

            # Compute posterior P(h|o)
            cond_h = 0.0
            for h in gap.hypotheses:
                p_h_given_o = (h.likelihood(obs) * h.posterior) / p_obs
                if p_h_given_o > 1e-9:
                    cond_h -= p_h_given_o * math.log2(p_h_given_o)

            expected_conditional_entropy += p_obs * cond_h

        ig = current_h - expected_conditional_entropy
        return max(0.0, ig)

    def compute_value_of_information(self, gap: EpistemicGap) -> float:
        """Compute Value of Information (VoI): Decision Relevance * Expected IG - Cost."""
        ig = self.compute_expected_information_gain(gap)
        decision_value = ig * gap.decision_relevance
        return max(0.0, decision_value - (self.cost * 0.1))


class ProbeGenerator:
    """Generates grounded physical candidate probes to resolve specific EpistemicGaps."""

    def generate_probes_for_gap(self, gap: EpistemicGap) -> list[EpistemicProbe]:
        """Generate targeted discriminative probes for a given gap."""
        probes: list[EpistemicProbe] = []

        if gap.domain == "geometry":
            target_node = gap.source_node_ids[0] if gap.source_node_ids else "target"

            # 1. Gentle Nudge Probe (Highly discriminative, low risk, low cost, reversible)
            probes.append(
                EpistemicProbe(
                    probe_id=f"probe_nudge_{target_node}",
                    target_gap_id=gap.gap_id,
                    description=f"Gentle lateral nudge on {target_node} to observe rolling behavior",
                    action_sequence=[("PUSH", {"target_id": target_node, "dx": 0.5, "dy": 0.0})],
                    possible_observations=["roll", "no_roll"],
                    cost=0.05,
                    risk=0.03,
                    reversibility=0.95,
                )
            )

            # 2. Heavy Strike Probe (Discriminative, but high risk / cost / low reversibility)
            probes.append(
                EpistemicProbe(
                    probe_id=f"probe_strike_{target_node}",
                    target_gap_id=gap.gap_id,
                    description=f"High-force shove on {target_node}",
                    action_sequence=[("PUSH", {"target_id": target_node, "dx": 5.0, "dy": 0.0})],
                    possible_observations=["roll", "no_roll"],
                    cost=0.40,
                    risk=0.85,  # Dangerous!
                    reversibility=0.20,
                )
            )

        elif gap.domain == "containment":
            target_box = gap.source_node_ids[0] if gap.source_node_ids else "box"
            probes.append(
                EpistemicProbe(
                    probe_id=f"probe_inspect_{target_box}",
                    target_gap_id=gap.gap_id,
                    description=f"Move near {target_box} to inspect interior",
                    action_sequence=[("MOVE", {"entity_id": "agent", "target_x": 2.0, "target_y": 2.0})],
                    possible_observations=["interior_visible", "lid_detected"],
                    cost=0.08,
                    risk=0.02,
                    reversibility=0.98,
                )
            )

        return probes
