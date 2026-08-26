"""A19 Intrinsic Curiosity, Active Epistemic Discovery & Decision Policy Benchmark Suite (20 Scenarios).

Evaluates EpistemicGap scanning, Shannon entropy quantification, Bayesian posterior updates,
discriminative EpistemicProbes, expected information gain, Value-of-Information (VoI),
multi-criteria DecisionEngine, rational inaction, and the Flagship Epistemic Discovery Trial.
"""

from __future__ import annotations

import sys
from typing import Any

from hbllm.brain.decision import (
    ActiveDiscoveryLoop,
    CandidateKind,
    DecisionCandidate,
    DecisionEngine,
    DecisionType,
    EpistemicGap,
    EpistemicGapScanner,
    EpistemicProbe,
    HypothesisOption,
    ProbeGenerator,
)
from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode

# ═══════════════════════════════════════════════════════════════════════════
# Scenario 1: A19-01 Shannon Entropy Quantification
# ═══════════════════════════════════════════════════════════════════════════


class TestShannonEntropy:
    """Calculates exact Shannon entropy H(H) = -sum P(h)*log2(P(h)) over hypotheses."""

    def test_entropy_calculation_equal_prior(self) -> None:
        gap = EpistemicGap(
            domain="geometry",
            hypotheses=[
                HypothesisOption(hypothesis_id="h1", label="FLAT", prior=0.5),
                HypothesisOption(hypothesis_id="h2", label="CONVEX", prior=0.5),
            ],
        )
        # H(0.5, 0.5) = 1.0 bit
        assert abs(gap.entropy - 1.0) < 1e-4

        # Skewed posterior H(0.9, 0.1) ≈ 0.469 bits
        gap.hypotheses[0].posterior = 0.9
        gap.hypotheses[1].posterior = 0.1
        assert abs(gap.entropy - 0.469) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 2: A19-02 Epistemic Gap Scanner
# ═══════════════════════════════════════════════════════════════════════════


class TestEpistemicGapScanner:
    """Scans canonical HCIR graph to identify ungrounded geometry and container states."""

    def test_scans_geometry_and_containment_gaps(self) -> None:
        graph = CognitiveGraph()
        unknown_obj = PhysicalEntityNode(id="obj_unknown", entity_type="object", properties={"shape": "unknown"})
        box = PhysicalEntityNode(id="box_1", entity_type="box", properties={})  # Missing is_closed
        graph.add_node(unknown_obj)
        graph.add_node(box)

        scanner = EpistemicGapScanner()
        gaps = scanner.scan_graph(graph)

        assert len(gaps) == 2
        domains = {g.domain for g in gaps}
        assert "geometry" in domains
        assert "containment" in domains


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 3: A19-03 Decision Relevance Weighting
# ═══════════════════════════════════════════════════════════════════════════


class TestDecisionRelevance:
    """Entities involved in active goals receive higher decision relevance."""

    def test_active_goal_node_boosts_relevance(self) -> None:
        graph = CognitiveGraph()
        obj_goal = PhysicalEntityNode(id="obj_goal", entity_type="object", properties={"shape": "unknown"})
        obj_distant = PhysicalEntityNode(id="obj_distant", entity_type="object", properties={"shape": "unknown"})
        graph.add_node(obj_goal)
        graph.add_node(obj_distant)

        scanner = EpistemicGapScanner()
        gaps = scanner.scan_graph(graph, active_goal_nodes=["obj_goal"])

        goal_gap = next(g for g in gaps if "obj_goal" in g.source_node_ids)
        distant_gap = next(g for g in gaps if "obj_distant" in g.source_node_ids)

        assert goal_gap.decision_relevance == 0.95
        assert distant_gap.decision_relevance == 0.50


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 4: A19-04 Expected Information Gain
# ═══════════════════════════════════════════════════════════════════════════


class TestExpectedInformationGain:
    """Calculates IG = H(H) - sum_o P(o)*H(H|o) accurately for discriminative probes."""

    def test_discriminative_probe_high_ig(self) -> None:
        gap = EpistemicGap(
            domain="geometry",
            hypotheses=[
                HypothesisOption(
                    hypothesis_id="h1",
                    label="FLAT",
                    prior=0.5,
                    predicted_observations={"roll": 0.05, "no_roll": 0.95},
                ),
                HypothesisOption(
                    hypothesis_id="h2",
                    label="CONVEX",
                    prior=0.5,
                    predicted_observations={"roll": 0.95, "no_roll": 0.05},
                ),
            ],
        )
        probe = EpistemicProbe(
            probe_id="p1",
            possible_observations=["roll", "no_roll"],
        )

        ig = probe.compute_expected_information_gain(gap)
        # Expected conditional entropy is very low -> IG should be near 1.0 (approx 0.71+)
        assert ig > 0.70


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 5: A19-05 Discriminative vs Uninformative Probes
# ═══════════════════════════════════════════════════════════════════════════


class TestDiscriminativeVsUninformativeProbe:
    """Uninformative probe (both hypotheses predict same observation) yields ~0 IG."""

    def test_uninformative_probe_zero_ig(self) -> None:
        gap = EpistemicGap(
            domain="geometry",
            hypotheses=[
                HypothesisOption(
                    hypothesis_id="h1",
                    label="FLAT",
                    prior=0.5,
                    predicted_observations={"sound": 1.0},
                ),
                HypothesisOption(
                    hypothesis_id="h2",
                    label="CONVEX",
                    prior=0.5,
                    predicted_observations={"sound": 1.0},
                ),
            ],
        )
        probe = EpistemicProbe(
            probe_id="p_uninformative",
            possible_observations=["sound"],
        )

        ig = probe.compute_expected_information_gain(gap)
        assert abs(ig - 0.0) < 1e-4


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 6: A19-06 Value of Information (VoI)
# ═══════════════════════════════════════════════════════════════════════════


class TestValueOfInformation:
    """VoI scales expected IG by decision relevance minus cost."""

    def test_voi_scales_with_decision_relevance(self) -> None:
        gap_critical = EpistemicGap(
            decision_relevance=0.90,
            hypotheses=[
                HypothesisOption(hypothesis_id="h1", label="FLAT", prior=0.5, predicted_observations={"roll": 0.05, "no_roll": 0.95}),
                HypothesisOption(hypothesis_id="h2", label="CONVEX", prior=0.5, predicted_observations={"roll": 0.95, "no_roll": 0.05}),
            ],
        )
        gap_irrelevant = EpistemicGap(
            decision_relevance=0.10,
            hypotheses=[
                HypothesisOption(hypothesis_id="h1", label="FLAT", prior=0.5, predicted_observations={"roll": 0.05, "no_roll": 0.95}),
                HypothesisOption(hypothesis_id="h2", label="CONVEX", prior=0.5, predicted_observations={"roll": 0.95, "no_roll": 0.05}),
            ],
        )

        probe = EpistemicProbe(possible_observations=["roll", "no_roll"], cost=0.05)

        voi_crit = probe.compute_value_of_information(gap_critical)
        voi_irrel = probe.compute_value_of_information(gap_irrelevant)

        assert voi_crit > voi_irrel
        assert voi_crit > 0.60
        assert voi_irrel < 0.10


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 7: A19-07 Multi-Criteria Utility Ranking
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiCriteriaUtility:
    """DecisionEngine ranks candidates according to composite EU."""

    def test_composite_utility_ranking(self) -> None:
        engine = DecisionEngine()
        c1 = DecisionCandidate(
            candidate_kind=CandidateKind.GOAL_ACTION,
            description="Direct Goal Action",
            goal_progress=0.8,
            action_cost=0.1,
            predicted_risk=0.05,
            reversibility=0.9,
        )
        c2 = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Dangerous Probe",
            value_of_information=0.9,
            predicted_risk=0.85,  # Too dangerous!
            action_cost=0.3,
            reversibility=0.1,
        )

        eu1 = engine.evaluate_candidate_utility(c1)
        eu2 = engine.evaluate_candidate_utility(c2)

        assert eu1 > eu2
        res = engine.select_best_decision([c1, c2])
        assert res.selected_candidate == c1
        assert res.decision_type == DecisionType.ACTION


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 8: A19-08 Goal Dominance
# ═══════════════════════════════════════════════════════════════════════════


class TestGoalDominance:
    """When a clear high-utility goal action exists, goal pursuit dominates exploration."""

    def test_high_progress_goal_beats_pure_exploration(self) -> None:
        engine = DecisionEngine()
        goal_cand = DecisionCandidate(
            candidate_kind=CandidateKind.GOAL_ACTION,
            description="Complete Delivery",
            goal_progress=1.0,
            predicted_risk=0.02,
            action_cost=0.1,
        )
        probe_cand = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Inspect Distant Room",
            value_of_information=0.5,
            goal_progress=0.0,
            predicted_risk=0.02,
            action_cost=0.1,
        )

        res = engine.select_best_decision([goal_cand, probe_cand])
        assert res.selected_candidate == goal_cand
        assert res.decision_type == DecisionType.ACTION


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 9: A19-09 Uncertainty Dominance
# ═══════════════════════════════════════════════════════════════════════════


class TestUncertaintyDominance:
    """When direct goal action has low probability/high risk due to unknown state, probe is selected first."""

    def test_critical_gap_prioritizes_probe_before_action(self) -> None:
        engine = DecisionEngine()
        # Blindly stacking without knowing geometry has high risk (0.80)
        blind_action = DecisionCandidate(
            candidate_kind=CandidateKind.GOAL_ACTION,
            description="Blind Stack on Unknown Object",
            goal_progress=0.4,
            predicted_risk=0.80,
            action_cost=0.1,
        )
        # Nudge probe to check geometry has low risk and high VoI
        nudge_probe = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Nudge to Verify Geometry",
            value_of_information=0.85,
            goal_progress=0.1,
            predicted_risk=0.03,
            action_cost=0.05,
            reversibility=0.95,
        )

        res = engine.select_best_decision([blind_action, nudge_probe])
        assert res.selected_candidate == nudge_probe
        assert res.decision_type == DecisionType.PROBE


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 10: A19-10 Risk Gating on Probes
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskGatingOnProbes:
    """High-risk probe is penalized by risk weight and rejected."""

    def test_high_risk_strike_probe_penalized(self) -> None:
        engine = DecisionEngine()
        gentle_probe = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Gentle Nudge",
            value_of_information=0.75,
            predicted_risk=0.03,
            action_cost=0.05,
        )
        violent_probe = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Violent Push",
            value_of_information=0.80,
            predicted_risk=0.85,  # Too risky!
            action_cost=0.40,
        )

        res = engine.select_best_decision([gentle_probe, violent_probe])
        assert res.selected_candidate == gentle_probe


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 11: A19-11 Cost & Effort Penalization
# ═══════════════════════════════════════════════════════════════════════════


class TestCostEffortPenalization:
    """Between two probes with equal info gain, the lower cost probe is chosen."""

    def test_low_cost_nudge_beats_expensive_probe(self) -> None:
        engine = DecisionEngine()
        cheap_probe = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Cheap Local Nudge",
            value_of_information=0.70,
            action_cost=0.05,
        )
        expensive_probe = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Expensive Long Trajectory",
            value_of_information=0.70,
            action_cost=0.80,
        )

        res = engine.select_best_decision([cheap_probe, expensive_probe])
        assert res.selected_candidate == cheap_probe


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 12: A19-12 Reversibility Valuation
# ═══════════════════════════════════════════════════════════════════════════


class TestReversibilityValuation:
    """Reversible actions/probes receive a bonus over destructive ones."""

    def test_reversible_probe_favored(self) -> None:
        engine = DecisionEngine()
        reversible_cand = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Visual Inspection",
            value_of_information=0.60,
            reversibility=1.0,
        )
        destructive_cand = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Break Container Seal",
            value_of_information=0.60,
            reversibility=0.1,
        )

        res = engine.select_best_decision([reversible_cand, destructive_cand])
        assert res.selected_candidate == reversible_cand


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 13: A19-13 Rational Inaction: High Risk
# ═══════════════════════════════════════════════════════════════════════════


class TestRationalInactionHighRisk:
    """When all candidate actions have excessive risk (>=0.80), the system rationally chooses INACTION."""

    def test_rational_inaction_when_risk_exceeds_threshold(self) -> None:
        engine = DecisionEngine()
        risky_cand = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Walk over fragile glass",
            value_of_information=0.9,
            predicted_risk=0.95,
        )

        res = engine.select_best_decision([risky_cand])
        assert res.decision_type == DecisionType.INACTION
        assert res.selected_candidate is None
        assert "excessive risk" in res.rationale or "insufficient net expected utility" in res.rationale


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 14: A19-14 Rational Inaction: Negligible Value
# ═══════════════════════════════════════════════════════════════════════════


class TestRationalInactionNegligibleValue:
    """When probe cost exceeds marginal information gain, system chooses INACTION."""

    def test_rational_inaction_when_net_utility_negative(self) -> None:
        engine = DecisionEngine(inaction_threshold=0.10)
        trivial_cand = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Inspect trivial speck",
            value_of_information=0.02,
            action_cost=0.20,
            reversibility=0.9,
        )

        res = engine.select_best_decision([trivial_cand])
        assert res.decision_type == DecisionType.INACTION


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 15: A19-15 Stability Probe Generation
# ═══════════════════════════════════════════════════════════════════════════


class TestStabilityProbeGeneration:
    """ProbeGenerator creates targeted PUSH nudge action for geometry uncertainty."""

    def test_probe_generator_nudge_action(self) -> None:
        gap = EpistemicGap(
            domain="geometry",
            source_node_ids=["mystery_cylinder"],
        )
        generator = ProbeGenerator()
        probes = generator.generate_probes_for_gap(gap)

        assert len(probes) >= 1
        nudge = probes[0]
        assert nudge.action_sequence[0][0] == "PUSH"
        assert nudge.action_sequence[0][1]["target_id"] == "mystery_cylinder"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 16: A19-16 Bayesian Posterior Update & Entropy Collapse
# ═══════════════════════════════════════════════════════════════════════════


class TestBayesianPosteriorUpdate:
    """Observing probe outcome collapses hypothesis entropy."""

    def test_observation_collapses_entropy(self) -> None:
        gap = EpistemicGap(
            domain="geometry",
            hypotheses=[
                HypothesisOption(hypothesis_id="h1", label="FLAT", prior=0.5, predicted_observations={"roll": 0.05, "no_roll": 0.95}),
                HypothesisOption(hypothesis_id="h2", label="CONVEX", prior=0.5, predicted_observations={"roll": 0.95, "no_roll": 0.05}),
            ],
        )
        assert gap.entropy == 1.0

        # Physical probe executed -> observation = "roll"
        gap.update_with_observation("roll")

        # Posterior for CONVEX should now be 0.95 / (0.95 + 0.05) = 0.95
        assert gap.hypotheses[1].posterior >= 0.94
        assert gap.hypotheses[0].posterior <= 0.06
        assert gap.entropy < 0.35
        assert gap.is_resolved


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 17: A19-17 Grounded HCIR Commitment
# ═══════════════════════════════════════════════════════════════════════════


class TestGroundedHCIRCommitment:
    """Resolved gap commits winning grounded properties directly into canonical graph node."""

    def test_resolved_gap_commits_property_to_canonical_graph(self) -> None:
        graph = CognitiveGraph()
        node = PhysicalEntityNode(id="mystery_obj", entity_type="object", properties={"shape": "unknown"})
        graph.add_node(node)

        gap = EpistemicGap(
            domain="geometry",
            source_node_ids=["mystery_obj"],
            hypotheses=[
                HypothesisOption(hypothesis_id="h1", label="FLAT", prior=0.5, predicted_observations={"roll": 0.05, "no_roll": 0.95}, grounded_properties={"geometry": "flat", "surface": "flat"}),
                HypothesisOption(hypothesis_id="h2", label="CONVEX", prior=0.5, predicted_observations={"roll": 0.95, "no_roll": 0.05}, grounded_properties={"geometry": "convex", "surface": "convex"}),
            ],
        )

        loop = ActiveDiscoveryLoop()
        committed = loop.process_probe_observation_and_commit(graph, gap, observation="no_roll")

        assert committed
        updated_node = graph.get_node("mystery_obj")
        assert isinstance(updated_node, PhysicalEntityNode)
        assert updated_node.properties["geometry"] == "flat"
        assert gap.status == "COMMITTED"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 18: A19-18 The Flagship Epistemic Discovery Trial
# ═══════════════════════════════════════════════════════════════════════════


class TestFlagshipEpistemicDiscoveryTrial:
    """The Flagship Acceptance Gate: Agent encounters partially observed scene,

    detects epistemic gap, selects low-risk discriminative probe over blind action,
    simulates probe in A18, executes physical probe, observes outcome,
    collapses entropy, commits grounded geometry to HCIR, and enables safe goal execution.
    """

    def test_autonomous_epistemic_probe_and_goal_redirection(self) -> None:
        graph = CognitiveGraph()
        cup = PhysicalEntityNode(id="cup", entity_type="cup", properties={"x": 0.0, "y": 0.0, "shape": "cylinder", "geometry": "flat"})
        mystery_base = PhysicalEntityNode(id="mystery_base", entity_type="object", properties={"x": 2.0, "y": 0.0, "shape": "unknown"})
        graph.add_node(cup)
        graph.add_node(mystery_base)

        loop = ActiveDiscoveryLoop()

        # 1. Propose and decide: Blind stack vs Nudge probe
        goal_actions: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = [
            ("Stack Cup on Mystery Base", [("STACK", {"item_id": "cup", "base_id": "mystery_base"})])
        ]
        decision, gaps = loop.propose_and_decide(
            graph,
            goal_actions=goal_actions,
            active_goal_nodes=["mystery_base"],
        )

        # System detects the uncertainty and selects the PROBE
        assert decision.decision_type == DecisionType.PROBE
        assert decision.selected_candidate is not None
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap.source_node_ids == ["mystery_base"]

        # 2. Execute probe in physical reality -> observe "no_roll" (meaning it is FLAT!)
        committed = loop.process_probe_observation_and_commit(graph, gap, observation="no_roll")
        assert committed

        # 3. Canonical HCIR is now grounded with flat geometry
        base_node = graph.get_node("mystery_base")
        assert isinstance(base_node, PhysicalEntityNode)
        assert base_node.properties["geometry"] == "flat"

        # 4. Subsequent decision: now stack action is safe and selected as ACTION!
        decision_after, _ = loop.propose_and_decide(
            graph,
            goal_actions=goal_actions,
            active_goal_nodes=["mystery_base"],
        )
        assert decision_after.decision_type == DecisionType.ACTION
        assert decision_after.selected_candidate is not None
        assert decision_after.selected_candidate.description == "Stack Cup on Mystery Base"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 19: A19-19 Zero-LLM Invariant
# ═══════════════════════════════════════════════════════════════════════════


class TestZeroLLM:
    """Decision engine and discovery loop run with 100% deterministic code and zero neural/LLM imports."""

    def test_zero_llm_imports(self) -> None:
        llm_markers = ["openai", "anthropic", "litellm", "langchain", "transformers"]
        loaded = set(sys.modules.keys())
        for marker in llm_markers:
            assert marker not in loaded, f"LLM module loaded in decision runtime: {marker}"


# ═══════════════════════════════════════════════════════════════════════════
# Scenario 20: A19-20 Value-of-Information vs Information-Gain Divergence
# ═══════════════════════════════════════════════════════════════════════════


class TestVoIVsIGDivergence:
    """System selects a probe with lower raw IG but high decision relevance

    over a probe with high raw IG but near-zero decision relevance.
    """

    def test_high_ig_low_relevance_rejected_for_high_voi(self) -> None:
        engine = DecisionEngine()

        # Probe A: High raw IG (0.95), but on irrelevant distant object (relevance 0.02) -> VoI ≈ 0.02
        probe_a_irrelevant = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Probe Irrelevant Distant Speck",
            information_gain=0.95,
            value_of_information=0.02,
            action_cost=0.05,
            predicted_risk=0.01,
        )

        # Probe B: Moderate raw IG (0.55), but on critical path object (relevance 0.95) -> VoI ≈ 0.52
        probe_b_critical = DecisionCandidate(
            candidate_kind=CandidateKind.EPISTEMIC_PROBE,
            description="Probe Critical Path Obstacle",
            information_gain=0.55,
            value_of_information=0.52,
            action_cost=0.05,
            predicted_risk=0.01,
        )

        res = engine.select_best_decision([probe_a_irrelevant, probe_b_critical])
        assert res.selected_candidate == probe_b_critical
        assert res.decision_type == DecisionType.PROBE
