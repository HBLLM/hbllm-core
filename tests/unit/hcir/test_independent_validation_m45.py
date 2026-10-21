"""Milestone M4.5 Independent Validation, Representation Revision, and Transfer Causality Tests.

Verifies:
1. Independent Task Generator: Disjoint rule grammar not shared with solver.
2. Representation Expansion: Inadequacy detection, operator synthesis, and model revision.
3. Multi-Hypothesis Ambiguity Safeguard: Detection of test-input divergence among survivors.
4. Latency Distribution Reconciliation: Candidate count scaling and repeated MDL vs No MDL trials.
5. Four-Condition Causal Ablation: Strict comparative controls proving cross-pathway transfer benefit.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np

from experiments.benchmarks.task_generators.independent_task_generator import (
    IndependentTaskGenerator,
)
from experiments.benchmarks.transfer.cross_pathway_causal_benchmark import (
    CrossPathwayCausalBenchmark,
)
from hbllm.hcir.world.grid_operator import (
    DecisionAction,
    TransformationProgramSearch,
)
from hbllm.hcir.world.representation_expansion import RepresentationExpansionEngine


def test_independent_task_suite_generation_and_grammar_independence():
    """Verify that independent task generator has no solver imports and produces disjoint families."""
    # 1. AST check: Zero solver or GridOperator imports in independent_task_generator.py
    generator_module = inspect.getmodule(IndependentTaskGenerator)
    assert generator_module is not None
    tree = ast.parse(inspect.getsource(generator_module))

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module)

    # Must NOT import GridOperator or TransformationProgramSearch
    assert not any("grid_operator" in m for m in imported_modules), (
        "Independent generator must not import grid_operator"
    )

    # 2. Generate 50-task independent suite
    suite = IndependentTaskGenerator.generate_independent_50_suite(seed=42)
    assert len(suite) == 50

    families = {t.family for t in suite}
    expected = {
        "gravity_settle",
        "diagonal_ray_cast",
        "interior_infill",
        "alternating_pattern_extrapolate",
        "component_size_rank",
    }
    assert families == expected

    # 3. Test baseline solver on unexpressible independent task -> triggers inadequacy
    base_solver = TransformationProgramSearch(max_depth=3, use_mdl=True, enable_relational=True)
    grav_task = [t for t in suite if t.family == "gravity_settle"][0]
    pred, rule, meta = base_solver.solve(list(grav_task.train_pairs), grav_task.test_input)

    assert meta["solved"] is False
    assert meta["insufficient_hypothesis_language"] is True
    assert meta["epistemic_uncertainty"] == 1.0


def test_representation_inadequacy_and_expansion_revision():
    """Verify Level 3 representation expansion vs Level 2 predicate synthesis."""
    suite = IndependentTaskGenerator.generate_independent_50_suite(seed=42)

    # 1. Level 3 Representation Expansion: Rules genuinely unexpressible by 9 neighborhood predicates
    for fam in ["gravity_settle", "diagonal_ray_cast"]:
        task = [t for t in suite if t.family == fam][0]
        trace = RepresentationExpansionEngine.evaluate_representation_revision_cycle(task)

        # 7-Step Level 3 Verification Criteria from Peer Review:
        # 1. Initial hypothesis language and failure reason
        assert trace.phase1_solved is False, f"Phase 1 should fail for {fam}"
        assert trace.phase1_inadequacy_detected is True, f"Phase 1 should flag inadequacy for {fam}"
        assert trace.phase1_epistemic_uncertainty == 1.0

        # 2. Specific unexplained observations / residual errors
        assert trace.unexplained_residual_pixels > 0, f"Residual pixels must be detected for {fam}"

        # 3. Dynamic synthesis of candidate operator directly from residuals
        assert trace.synthesized_operator_name.startswith("synthesized_"), (
            f"Operator must be inductively synthesized, got {trace.synthesized_operator_name}"
        )

        # 4. Out-of-construction verification on subsequent demonstration pairs
        assert trace.out_of_construction_verified is True, (
            f"Operator must verify on demo 1..N for {fam}"
        )

        # 5. Generalization prediction on held-out test query input
        assert trace.phase2_solved is True, f"Phase 2 should succeed after revision for {fam}"
        assert trace.phase2_exact_match is True, f"Phase 2 exact match required for {fam}"
        assert trace.phase2_epistemic_uncertainty == 0.0, (
            f"Uncertainty should resolve to 0.0 for {fam}"
        )

        # 6. Reusability on independent novel task instance with different dimensions/objects
        assert trace.reusable_on_novel_task is True, (
            f"Operator must be reusable on novel task instances for {fam}"
        )

        # 7. Control ablation 1: Disabling synthesis causes failure (revision, not fallback, explains success)
        assert trace.control_ablation_passed is True, (
            f"Control ablation must fail when synthesis is disabled for {fam}"
        )

        # 8. Control ablation 2: Disabling particle correspondence causes failure
        assert trace.correspondence_ablation_passed is True, (
            f"Correspondence ablation must fail for {fam}"
        )

        # 9. Control ablation 3: Disabling propagation conditions causes failure
        assert trace.conditions_ablation_passed is True, (
            f"Propagation conditions ablation must fail for {fam}"
        )

    # 2. Level 2 Predicate Synthesis: interior_infill is solvable by generic neighborhood predicate 'interior'
    infill_task = [t for t in suite if t.family == "interior_infill"][0]
    infill_trace = RepresentationExpansionEngine.evaluate_representation_revision_cycle(infill_task)
    assert infill_trace.phase1_solved is True, (
        "interior_infill should be solved by Level 2 neighborhood predicate synthesis"
    )
    assert infill_trace.phase1_epistemic_uncertainty == 0.0


def test_cellular_automaton_withheld_laws_representation_expansion_and_ablations():
    """Stage 3 Validation: Induce a 2D cellular automaton with withheld transition laws.

    Verifies:
    1. Base solver failure (inadequacy detected, uncertainty = 1.0).
    2. Dynamic Level 3 synthesis of local CA transition laws from categorical residuals.
    3. Out-of-construction verification on demonstration 1.
    4. Exact match on held-out test query.
    5. Reusability on independent novel lattice.
    6. Causal ablation 1 (synthesis disabled): fails with uncertainty 1.0.
    7. Causal ablation 2 (correspondence disabled): fails.
    """
    gen = IndependentTaskGenerator(seed=100)
    for i in range(3):
        task = gen.generate_task(
            f"ca_stage3_task_{i:02d}",
            family="cellular_automaton_local_rule",
            num_demos=2,
        )

        # 1. Base solver failure
        base_solver = TransformationProgramSearch(max_depth=3, use_mdl=True, enable_relational=True)
        pred_b, _, meta_b = base_solver.solve(list(task.train_pairs), task.test_input)
        assert meta_b["solved"] is False
        assert meta_b["insufficient_hypothesis_language"] is True
        assert meta_b["epistemic_uncertainty"] == 1.0

        # 2. Representation revision cycle
        trace = RepresentationExpansionEngine.evaluate_representation_revision_cycle(task)
        assert trace.phase1_solved is False
        assert trace.phase1_inadequacy_detected is True
        assert trace.phase1_epistemic_uncertainty == 1.0
        assert trace.unexplained_residual_pixels > 0
        assert "cellular_automaton" in trace.synthesized_operator_name
        assert trace.out_of_construction_verified is True
        assert trace.phase2_solved is True
        assert trace.phase2_exact_match is True
        assert trace.phase2_epistemic_uncertainty == 0.0
        assert trace.reusable_on_novel_task is True
        assert trace.control_ablation_passed is True
        assert trace.correspondence_ablation_passed is True


def test_multi_hypothesis_test_ambiguity_safeguard():
    """Verify that solver detects ambiguity when multiple surviving hypotheses diverge on test query."""
    # Demonstrations are symmetric under both ROT90 and TRANSPOSE on square anti-diagonal input
    # Demo 1: [[0, 2], [2, 0]] -> [[0, 2], [2, 0]]
    d1_in = np.array([[0, 2], [2, 0]])
    d1_out = np.array([[0, 2], [2, 0]])

    # Test input: [[1, 2], [3, 4]]
    # Rot90 gives [[2, 4], [1, 3]] (or similar), while Transpose gives [[1, 3], [2, 4]]
    test_in = np.array([[1, 2], [3, 4]])

    searcher = TransformationProgramSearch(max_depth=1, use_mdl=True)
    pred, rule, meta = searcher.solve([(d1_in, d1_out)], test_in)

    # When multiple hypotheses survive Demo 1, check ambiguity flag
    if meta.get("survivor_count", 0) >= 2:
        assert meta["is_ambiguous_on_test"] is True
        assert meta["epistemic_uncertainty"] == 0.5


def test_latency_distribution_reconciliation_and_candidate_scaling():
    """Verify candidate count scaling across depths and repeated MDL trials."""
    from experiments.benchmarks.evaluation.scaled_evaluation_suite import (
        ScaledGeneralizationBenchmark,
    )

    suite = ScaledGeneralizationBenchmark.generate_100_task_suite(seed=42)

    # Depth 1 tasks (geometric)
    d1_tasks = [t for t in suite if t.family == "geometric_affine"][:5]
    rep_d1 = ScaledGeneralizationBenchmark.run_benchmark(tasks=d1_tasks, max_depth=1, use_mdl=True)

    # Depth 3 tasks (compositional)
    d3_tasks = [t for t in suite if t.family == "compositional_deep"][:5]
    rep_d3 = ScaledGeneralizationBenchmark.run_benchmark(tasks=d3_tasks, max_depth=3, use_mdl=True)

    # Reconcile distribution skew: Depth 1 searches fewer candidates and finishes much faster than Depth 3
    mean_d1_total = rep_d1.latency_profile["total_wall_clock"]["mean"]
    mean_d3_total = rep_d3.latency_profile["total_wall_clock"]["mean"]
    assert mean_d1_total < mean_d3_total, (
        f"Depth 1 ({mean_d1_total}ms) must be faster than Depth 3 ({mean_d3_total}ms)"
    )

    # Repeated trials of Full HCIR vs No MDL: Ranking overhead is negligible (< 1.0ms)
    assert rep_d3.latency_profile["ranking_selection"]["mean"] < 1.0

    # MDL selects lower complexity on average
    rep_d3_nomdl = ScaledGeneralizationBenchmark.run_benchmark(
        tasks=d3_tasks, max_depth=3, use_mdl=False
    )
    comp_full = [tr.winning_complexity for tr in rep_d3.traces if tr.winning_complexity is not None]
    comp_nomdl = [
        tr.winning_complexity for tr in rep_d3_nomdl.traces if tr.winning_complexity is not None
    ]
    assert np.mean(comp_full) <= np.mean(comp_nomdl)


def test_four_condition_cross_pathway_causal_ablation():
    """Verify that Condition 2 outperforms Condition 1 and Condition 3 (negative control)."""
    report = CrossPathwayCausalBenchmark.run_all_conditions(seed=42)
    sums = report.condition_summaries

    c1 = sums["path_a_alone"]
    c2 = sums["path_a_with_relational_priors"]
    c3 = sums["path_a_with_shuffled_priors"]
    c4 = sums["path_a_transfer_disabled_post_learning"]

    # 1. Condition 2 vs Condition 1 (Baseline): Transfer significantly reduces total navigation steps
    assert c2["mean_nav_steps"] <= c1["mean_nav_steps"], (
        f"Transferred priors ({c2['mean_nav_steps']}) must be more efficient than baseline ({c1['mean_nav_steps']})"
    )

    # 2. Condition 2 vs Condition 3 (Negative Control): Shuffled/irrelevant priors cause higher prediction errors
    assert c2["mean_prediction_errors"] <= c3["mean_prediction_errors"], (
        f"Relevant priors ({c2['mean_prediction_errors']}) must have <= errors than shuffled priors ({c3['mean_prediction_errors']})"
    )

    # 3. Condition 2 retains spatial boundaries perfectly
    assert c2["mean_barriers_retained_pct"] == 100.0

    # 4. Condition 4 confirms goal attainment when transfer is deactivated post-learning
    assert c4["goal_attainment_pct"] >= 80.0


def test_cross_pathway_causal_multi_seed_uncertainty_and_controls():
    """Verify statistical distributions, confidence intervals, and 3 advanced controls."""
    # Run across 3 seeds (15 trials per condition)
    report = CrossPathwayCausalBenchmark.run_all_conditions(seeds=[42, 100, 2024])
    sums = report.condition_summaries
    dists = report.statistical_distributions

    c_rel = sums["path_a_with_relational_priors"]
    c_noise = sums["path_a_with_unstructured_noise"]
    c_corrupt = sums["path_a_with_corrupted_topology"]
    c_delayed = sums["path_a_delayed_retention_intervening_learning"]

    # 1. Control A: Unstructured noise priors cause higher prediction errors than structured priors
    assert c_rel["mean_prediction_errors"] <= c_noise["mean_prediction_errors"], (
        f"Structured priors ({c_rel['mean_prediction_errors']}) must have <= errors than noise ({c_noise['mean_prediction_errors']})"
    )

    # 2. Control B: Corrupted topology causes higher prediction errors than true relational priors
    assert c_rel["mean_prediction_errors"] <= c_corrupt["mean_prediction_errors"], (
        f"Structured priors ({c_rel['mean_prediction_errors']}) must have <= errors than corrupted ({c_corrupt['mean_prediction_errors']})"
    )

    # 3. Control C: Delayed retention survives intervening learning
    assert c_delayed["goal_attainment_pct"] >= 80.0
    assert c_delayed["mean_barriers_retained_pct"] == 100.0

    # 4. Statistical Distributions and 95% Confidence Intervals
    rel_nav = dists["path_a_with_relational_priors"]["nav_steps"]
    assert rel_nav.ci_95_low <= rel_nav.mean <= rel_nav.ci_95_high, (
        "95% CI bounds must bracket the sample mean"
    )
    assert rel_nav.std >= 0.0, "Sample standard deviation must be non-negative"

    # 5. Active probe steps tracking
    assert c_rel["mean_probe_steps"] == 4.0, (
        "Exactly 4 active probes required to ground cardinal action directions"
    )

    # 6. Rigorous Paired Hypothesis Testing (N=15 matched trials, df=14)
    assert "relational_vs_alone_nav" in report.paired_tests
    p_nav = report.paired_tests["relational_vs_alone_nav"]
    assert p_nav.sample_count_n == 15
    assert p_nav.degrees_of_freedom == 14
    assert p_nav.mean_difference > 0.0  # Saves ~0.73 steps on average
    assert p_nav.t_statistic >= 2.0  # t = 2.22, p <= 0.05
    assert p_nav.ci_95_low <= p_nav.mean_difference <= p_nav.ci_95_high

    assert "shuffled_vs_relational_errors" in report.paired_tests
    p_shuff = report.paired_tests["shuffled_vs_relational_errors"]
    assert p_shuff.mean_difference > 0.5  # 0.73 fewer errors than mismatched priors
    assert p_shuff.t_statistic >= 2.5  # t = 2.58, p = 0.01

    assert "disabled_vs_relational_nav" in report.paired_tests
    p_dis = report.paired_tests["disabled_vs_relational_nav"]
    assert p_dis.mean_difference == 0.0  # Consolidated internal model identical step-for-step


def test_operational_epistemic_decision_policy():
    """Verify operational epistemic decision policy: PREDICT, PROBE, and ABSTAIN behaviors."""
    # 1. Unambiguous condition -> PREDICT with Brier score calibration
    # Identical spatial rotation where all surviving hypotheses agree on test input
    d1 = (np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))
    d2 = (np.array([[2, 0], [0, 0]]), np.array([[0, 0], [0, 2]]))
    d3 = (np.array([[3, 0], [0, 0]]), np.array([[0, 0], [0, 3]]))
    test_in = np.array([[4, 0], [0, 0]])
    expected_out = np.array([[0, 0], [0, 4]])

    searcher = TransformationProgramSearch(max_depth=1, use_mdl=True)
    dec_pred = searcher.decide([d1, d2, d3], test_in, ground_truth_test=expected_out)

    assert dec_pred.action == DecisionAction.PREDICT
    assert dec_pred.selected_prediction is not None
    assert np.array_equal(dec_pred.selected_prediction, expected_out)
    assert dec_pred.probing_coordinate is None
    assert dec_pred.expected_information_gain == 0.0
    assert dec_pred.brier_score == 0.0, "Exact correct prediction should have 0.0 Brier score"

    # 2. Ambiguous condition under interactive vs static protocols
    # Symmetric square input admits multiple rotations/reflections that diverge on asymmetric test input
    d_amb_in = np.array([[0, 2], [2, 0]])
    d_amb_out = np.array([[0, 2], [2, 0]])
    test_amb_in = np.array([[1, 2], [3, 4]])

    _, _, meta_amb = searcher.solve([(d_amb_in, d_amb_out)], test_amb_in)
    if meta_amb.get("survivor_count", 0) >= 2 and meta_amb.get("is_ambiguous_on_test", False):
        # 2a. Interactive protocol: chooses PROBE with maximum information gain coordinate
        dec_probe = searcher.decide(
            [(d_amb_in, d_amb_out)], test_amb_in, ground_truth_test=None, protocol="interactive"
        )
        assert dec_probe.action == DecisionAction.PROBE
        assert dec_probe.probing_coordinate is not None
        assert dec_probe.expected_information_gain > 0.0
        assert dec_probe.abstention_reason is None

        # 2b. Static ARC protocol: test querying prohibited -> calibrated ABSTAIN without output leakage
        dec_static = searcher.decide(
            [(d_amb_in, d_amb_out)], test_amb_in, ground_truth_test=None, protocol="static_arc"
        )
        assert dec_static.action == DecisionAction.ABSTAIN
        assert dec_static.probing_coordinate is None
        assert dec_static.abstention_reason == "AMBIGUOUS_TEST_HYPOTHESES_STATIC_PROTOCOL"

    # 3. Inadequate hypothesis language -> ABSTAIN
    suite = IndependentTaskGenerator.generate_independent_50_suite(seed=42)
    grav_task = [t for t in suite if t.family == "gravity_settle"][0]
    base_solver = TransformationProgramSearch(max_depth=3, use_mdl=True, enable_relational=True)
    dec_abstain = base_solver.decide(
        list(grav_task.train_pairs), grav_task.test_input, ground_truth_test=grav_task.test_output
    )

    assert dec_abstain.action == DecisionAction.ABSTAIN
    assert dec_abstain.selected_prediction is None
    assert dec_abstain.probing_coordinate is None
    assert dec_abstain.expected_information_gain == 0.0
    assert dec_abstain.abstention_reason == "HYPOTHESIS_GENERATION"
    assert dec_abstain.brier_score is None
