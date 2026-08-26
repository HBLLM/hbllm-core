"""Active Discovery Loop Orchestrator for A19.

Coordinates the full epistemic discovery cycle:
Canonical HCIR -> Scan Gaps -> Propose Candidates -> Simulate in A18 ->
Decide in DecisionEngine -> Execute Physical Probe -> Observe ->
Bayesian Posterior Update -> Commit Grounded Knowledge to HCIR.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.decision.gap import EpistemicGap, EpistemicGapScanner
from hbllm.brain.decision.policy import (
    CandidateKind,
    DecisionCandidate,
    DecisionEngine,
    DecisionResult,
)
from hbllm.brain.decision.probe import ProbeGenerator
from hbllm.brain.simulation.counterfactual_engine import MentalSandbox
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode

logger = logging.getLogger(__name__)


class ActiveDiscoveryLoop:
    """The central orchestrator for goal pursuit and active epistemic discovery."""

    def __init__(
        self,
        scanner: EpistemicGapScanner | None = None,
        probe_generator: ProbeGenerator | None = None,
        sandbox: MentalSandbox | None = None,
        decision_engine: DecisionEngine | None = None,
    ) -> None:
        self.scanner = scanner or EpistemicGapScanner()
        self.probe_generator = probe_generator or ProbeGenerator()
        self.sandbox = sandbox or MentalSandbox()
        self.decision_engine = decision_engine or DecisionEngine()

    def propose_and_decide(
        self,
        graph: CognitiveGraph,
        goal_actions: list[tuple[str, list[tuple[str, dict[str, Any]]]]] | None = None,
        active_goal_nodes: list[str] | None = None,
    ) -> tuple[DecisionResult, list[EpistemicGap]]:
        """Scan world state, simulate candidates in A18, and select the optimal decision."""
        # 1. Scan for active epistemic gaps
        gaps = self.scanner.scan_graph(graph, active_goal_nodes=active_goal_nodes)
        candidates: list[DecisionCandidate] = []

        # 2. Add goal-directed candidates
        if goal_actions:
            for desc, action_seq in goal_actions:
                # Simulate in A18 sandbox to evaluate feasibility and risk
                branch, sim_res = self.sandbox.simulate_trajectory(graph, action_seq)
                success = all(r.is_success for r in sim_res)
                risk = branch.accumulated_risk

                cand = DecisionCandidate(
                    candidate_kind=CandidateKind.GOAL_ACTION,
                    action_sequence=action_seq,
                    description=desc,
                    target_goal_ids=active_goal_nodes or [],
                    goal_progress=1.0 if success else 0.2,
                    information_gain=0.0,
                    value_of_information=0.0,
                    predicted_risk=risk,
                    action_cost=len(action_seq) * 0.1,
                    reversibility=0.80,
                )
                candidates.append(cand)

        # 3. Add epistemic probe candidates for each gap
        for gap in gaps:
            probes = self.probe_generator.generate_probes_for_gap(gap)
            for probe in probes:
                # Simulate probe in A18 sandbox
                branch, sim_res = self.sandbox.simulate_trajectory(graph, probe.action_sequence)
                risk = max(probe.risk, branch.accumulated_risk)

                ig = probe.compute_expected_information_gain(gap)
                voi = probe.compute_value_of_information(gap)

                cand = DecisionCandidate(
                    candidate_kind=CandidateKind.EPISTEMIC_PROBE,
                    action_sequence=probe.action_sequence,
                    description=probe.description,
                    target_gap_ids=[gap.gap_id],
                    goal_progress=0.10,  # Probes may have minor goal utility
                    information_gain=ig,
                    value_of_information=voi,
                    predicted_risk=risk,
                    action_cost=probe.cost,
                    reversibility=probe.reversibility,
                )
                candidates.append(cand)

        # 4. Evaluate all candidates through DecisionEngine
        decision = self.decision_engine.select_best_decision(candidates)
        return decision, gaps

    def process_probe_observation_and_commit(
        self,
        graph: CognitiveGraph,
        gap: EpistemicGap,
        observation: str,
    ) -> bool:
        """Update gap with physical observation, and commit to canonical HCIR if resolved.

        Returns:
            True if committed to canonical HCIR graph.
        """
        # 1. Bayesian posterior update
        gap.update_with_observation(observation)

        # 2. Check if grounded resolution threshold reached
        if gap.is_resolved:
            leading = gap.leading_hypothesis
            if leading is not None and gap.source_node_ids:
                target_id = gap.source_node_ids[0]
                node = graph.get_node(target_id)
                if isinstance(node, PhysicalEntityNode):
                    props = dict(getattr(node, "properties", None) or getattr(node, "observed_properties", {}) or {})
                    # Update properties with grounded properties from winning hypothesis
                    props.update(leading.grounded_properties)
                    node.properties = props
                    graph.upsert_node(node)  # Update canonical graph node
                    gap.status = "COMMITTED"
                    logger.info("Committed resolved epistemic hypothesis %s to node %s", leading.label, target_id)
                    return True

        return False
