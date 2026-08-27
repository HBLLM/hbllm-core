"""Test Suite for the Scientific Comparison Experiment.

Validates the full comparative experimental harness:
1. Leakage Auditor verifies zero hidden environment state or preloaded task data.
2. Independent Oracle computes optimal utility and probes outside all cohorts.
3. Cohort interface parity (HBLLM-Core, HBLLM+LLM, LLM-Only, Ablations).
4. Task Battery (E1 through E7) execution under identical observation/action spaces.
5. Standardized Metrics (N_tau, Brier, ECE, BWT, FWT, full 5x5 R_{i,j} matrix).
6. Master End-to-End Experiment Runner and Markdown Report generation.
"""

from __future__ import annotations

from hbllm.experiment import (
    AblatedHBLLMCohort,
    CanonicalTaskEnvironment,
    E3_CounterfactualSimulationTask,
    E4_EpistemicCalibrationTask,
    E5_ActiveEpistemicDiscoveryTask,
    E6_RelationalTransferTask,
    E7_LifelongCurriculumTask,
    ExperimentMetricsCalculator,
    ExperimentRunner,
    ExperimentStatistics,
    HBLLMCoreCohort,
    HBLLMPlusLLMCohort,
    IndependentEnvironmentOracle,
    LeakageAuditor,
    LLMOnlyCohort,
    PhysicalEnvironmentState,
)


class TestLeakageAudit:
    """Verifies that no hidden simulator state, labels, or preloaded task solutions are leaked."""

    def test_audit_detects_hidden_field_leakage(self) -> None:
        auditor = LeakageAuditor()
        canonical_obs = {"step": 1, "entities": ["a", "b"]}
        leaked_obs = {"step": 1, "entities": ["a", "b"], "_true_mass": 5.2}

        violations = auditor.audit_observation_parity("test_cohort", canonical_obs, leaked_obs)
        assert len(violations) > 0
        assert any("_true_mass" in v for v in violations)

    def test_audit_detects_nested_hidden_field_leakage(self) -> None:
        auditor = LeakageAuditor()
        canonical_obs = {"step": 1, "entities": [{"id": "e1", "props": {"color": "red"}}]}
        leaked_obs = {
            "step": 1,
            "entities": [
                {"id": "e1", "props": {"color": "red", "_oracle_optimal_action": "STACK"}}
            ],
        }

        violations = auditor.audit_observation_parity("test_cohort", canonical_obs, leaked_obs)
        assert len(violations) > 0
        assert any("_oracle_optimal_action" in v for v in violations)

    def test_audit_passes_on_clean_initial_state(self) -> None:
        auditor = LeakageAuditor()
        clean_state = {"generic_rules": ["gravity_exists"]}
        report = auditor.run_full_audit({"HBLLM-Core": clean_state}, {})
        assert report.is_clean
        assert len(report.violations) == 0
        assert len(report.initial_knowledge_hash) == 64

    def test_audit_live_cohort_instances_clean(self) -> None:
        auditor = LeakageAuditor()
        cohorts = [HBLLMCoreCohort(), HBLLMPlusLLMCohort(), LLMOnlyCohort()]
        report = auditor.run_full_audit(cohorts)
        assert report.is_clean
        assert len(report.violations) == 0
        assert len(report.audited_cohorts) == 3

    def test_audit_catches_preloaded_task_token(self) -> None:
        auditor = LeakageAuditor()
        contaminated_cohort = HBLLMCoreCohort()
        contaminated_cohort.concept_memory = {"mepo": "cube"}  # Cheat preload
        report = auditor.run_full_audit([contaminated_cohort])
        assert not report.is_clean
        assert any("mepo" in v for v in report.violations)

    def test_audit_catches_oracle_reference_on_cohort(self) -> None:
        auditor = LeakageAuditor()
        contaminated_cohort = HBLLMCoreCohort()
        contaminated_cohort.oracle = IndependentEnvironmentOracle()  # Direct oracle cheat
        report = auditor.run_full_audit([contaminated_cohort])
        assert not report.is_clean
        assert any("oracle" in v.lower() for v in report.violations)


class TestIndependentOracle:
    """Verifies that the ground-truth oracle computes optimal action and probe utilities independently."""

    def test_oracle_computes_true_utility_and_probe(self) -> None:
        oracle = IndependentEnvironmentOracle()
        true_state = PhysicalEnvironmentState(
            entities={"support": {"surface_geometry": "flat", "is_rigid": True}},
            physics_rules={},
            target_goal={},
        )

        # Valid stacking on flat rigid support
        act_valid = {"name": "STACK", "parameters": {"base": "support"}}
        u_valid = oracle.evaluate_true_action_utility(act_valid, true_state)
        assert u_valid == 1.0

        # Optimal probe calculation
        probes = [
            {"name": "GENTLE_TAP", "cost": 0.2, "discriminative_power": 0.8},
            {"name": "EXPENSIVE_HEAVY_SLAM", "cost": 5.0, "discriminative_power": 0.9},
        ]
        best_probe, eff = oracle.compute_optimal_probe(probes, {"flat": 0.50, "curved": 0.50})
        assert best_probe is not None
        assert best_probe["name"] == "GENTLE_TAP"
        assert eff > 0.0


class TestCohortsInterface:
    """Verifies that all experimental cohorts implement structured output and resource tracking."""

    def test_all_cohorts_return_structured_output(self) -> None:
        env = CanonicalTaskEnvironment(domain="test_env")
        obs = env.reset()

        cohorts = [
            HBLLMCoreCohort(),
            HBLLMPlusLLMCohort(),
            LLMOnlyCohort(),
            AblatedHBLLMCohort("A18"),
        ]
        for c in cohorts:
            output = c.process_observation(obs)
            assert isinstance(output.prediction, str)
            assert 0.0 <= output.confidence <= 1.0
            assert isinstance(output.action, dict)

            resources = c.get_resource_usage()
            assert "wall_clock_ms" in resources
            assert "cpu_time_ms" in resources


class TestTaskParityAndMetrics:
    """Verifies that tasks E1 through E7 execute and metrics calculator computes exact formulas."""

    def test_metrics_calculator_formulas(self) -> None:
        calc = ExperimentMetricsCalculator()

        # N_tau calculation
        accuracies = [0.50, 0.70, 0.92, 0.94, 0.95, 0.95]
        n_tau = calc.calculate_episodes_to_threshold(accuracies, tau=0.90, consecutive_m=3)
        assert n_tau == 3

        # Brier score calculation
        brier = calc.calculate_brier_score([0.80, 0.90, 0.20], [True, True, False])
        assert abs(brier - 0.03) < 1e-2

        # BWT calculation from R_{i,j} matrix
        r_matrix = [
            [1.00, 0.00, 0.00],
            [1.00, 0.95, 0.00],
            [1.00, 0.95, 0.90],
        ]
        bwt = calc.calculate_bwt_from_matrix(r_matrix)
        assert bwt == 0.00  # Zero forgetting

    def test_statistics_aggregation(self) -> None:
        summary = ExperimentStatistics.summarize("test_metric", [0.90, 0.92, 0.94, 0.96, 0.98])
        assert summary.mean == 0.94
        assert summary.median == 0.94
        assert summary.ci_95_low < 0.94 < summary.ci_95_high


class TestAblationMatrixDegradation:
    """Verifies that disabling specific cognitive waves shows expected causal degradation."""

    def test_ablations_exhibit_targeted_performance_drops(self) -> None:
        # Ablating A18 impairs mental simulation
        a18_ablated = AblatedHBLLMCohort("A18")
        res_e3 = E3_CounterfactualSimulationTask().evaluate(a18_ablated)
        assert res_e3.simulation_error > 0.40

        # Ablating A19 impairs active probing
        a19_ablated = AblatedHBLLMCohort("A19")
        res_e5 = E5_ActiveEpistemicDiscoveryTask().evaluate(a19_ablated)
        assert res_e5.probing_regret > 0.30

        # Ablating A20 impairs structural transfer
        a20_ablated = AblatedHBLLMCohort("A20")
        res_e6 = E6_RelationalTransferTask().evaluate(a20_ablated)
        assert res_e6.structural_accuracy < 0.60

        # Ablating A21 impairs calibration
        a21_ablated = AblatedHBLLMCohort("A21")
        res_e4 = E4_EpistemicCalibrationTask().evaluate(a21_ablated)
        assert res_e4.brier_score > 0.20

        # Ablating A22 impairs continual retention
        a22_ablated = AblatedHBLLMCohort("A22")
        res_e7 = E7_LifelongCurriculumTask().evaluate(a22_ablated)
        assert res_e7.bwt < 0.00


class TestMasterScientificExperiment:
    """End-to-end execution of the Master Scientific Comparison Experiment."""

    def test_full_experiment_execution_and_reporting(self) -> None:
        runner = ExperimentRunner(random_seeds=[42, 101, 2024])
        report = runner.run_full_experiment()

        assert report.manifest.git_commit_hash != ""
        assert len(report.manifest.initial_knowledge_hash) == 64
        assert len(report.cohort_results) == 3
        assert "HBLLM-Core" in report.cohort_results
        assert "HBLLM+LLM" in report.cohort_results
        assert "LLM-Only" in report.cohort_results

        # Verify Markdown Summary renders properly
        md = report.render_markdown_summary()
        assert "# Scientific Comparison Report" in md
        assert "HBLLM-Core" in md
        assert "Continual Learning Task Matrix" in md
        assert "Ablation Analysis" in md

        # Verify JSON export
        json_str = report.to_json()
        assert "experiment_id" in json_str
        assert "primary_endpoints_table" in json_str

    def test_cohort_label_invariance(self) -> None:
        """Verifies that changing the cohort_id string does not alter evaluation metrics."""
        cohort_normal = LLMOnlyCohort(cohort_id="LLM-Only")
        cohort_relabeled = LLMOnlyCohort(cohort_id="HBLLM-Core-Fake-Label")

        res_normal = E4_EpistemicCalibrationTask().evaluate(cohort_normal)
        res_relabeled = E4_EpistemicCalibrationTask().evaluate(cohort_relabeled)

        assert res_normal.brier_score == res_relabeled.brier_score
        assert res_normal.accuracy == res_relabeled.accuracy
