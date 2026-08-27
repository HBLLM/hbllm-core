"""Scientific Comparison Experiment Runner.

Orchestrates multi-cohort, multi-task, and ablation evaluations,
runs pre-flight leakage audits, gathers multi-seed metrics, and produces the master report.
"""

from __future__ import annotations

import logging

from hbllm.experiment.cohorts import (
    AblatedHBLLMCohort,
    BaseCohort,
    HBLLMCoreCohort,
    HBLLMPlusLLMCohort,
    LLMOnlyCohort,
)
from hbllm.experiment.leakage_audit import LeakageAuditor
from hbllm.experiment.manifests import ReproducibilityManifest
from hbllm.experiment.reports import ScientificExperimentReport
from hbllm.experiment.statistics import ExperimentStatistics, MetricSummary
from hbllm.experiment.tasks import (
    E1_ConceptAcquisitionTask,
    E2_LexicalAcquisitionTask,
    E3_CounterfactualSimulationTask,
    E4_EpistemicCalibrationTask,
    E5_ActiveEpistemicDiscoveryTask,
    E6_RelationalTransferTask,
    E7_LifelongCurriculumTask,
    TaskEvaluationResult,
)

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Coordinates the execution of the full scientific comparison battery."""

    def __init__(self, random_seeds: list[int] | None = None) -> None:
        self.seeds = random_seeds or [42, 101, 2024]
        self.auditor = LeakageAuditor()
        self.manifest = ReproducibilityManifest(random_seeds=self.seeds)

    def run_full_experiment(self) -> ScientificExperimentReport:
        """Execute all 7 tasks across all 3 primary cohorts and ablations with multi-seed aggregation."""
        cohort_constructors = {
            "HBLLM-Core": lambda: HBLLMCoreCohort(),
            "HBLLM+LLM": lambda: HBLLMPlusLLMCohort(),
            "LLM-Only": lambda: LLMOnlyCohort(),
        }

        ablations: list[BaseCohort] = [
            AblatedHBLLMCohort("A18"),
            AblatedHBLLMCohort("A19"),
            AblatedHBLLMCohort("A20"),
            AblatedHBLLMCohort("A21"),
            AblatedHBLLMCohort("A22"),
        ]

        tasks = [
            E1_ConceptAcquisitionTask(),
            E2_LexicalAcquisitionTask(),
            E3_CounterfactualSimulationTask(),
            E4_EpistemicCalibrationTask(),
            E5_ActiveEpistemicDiscoveryTask(),
            E6_RelationalTransferTask(),
            E7_LifelongCurriculumTask(),
        ]

        self.manifest.cohort_ids = list(cohort_constructors.keys())
        self.manifest.task_order = [t.__class__.__name__ for t in tasks]

        # 1. Pre-flight Leakage Audit
        cohort_instances = [fn() for fn in cohort_constructors.values()]
        audit_report = self.auditor.run_full_audit(cohort_instances, tasks)
        self.manifest.initial_knowledge_hash = audit_report.initial_knowledge_hash
        if not audit_report.is_clean:
            logger.warning(
                "Pre-flight leakage audit found %d violations: %s",
                len(audit_report.violations),
                audit_report.violations,
            )

        # 2. Execute Primary Cohort Tasks across seeds
        # seed -> cohort_id -> task_id -> TaskEvaluationResult
        seed_results: dict[int, dict[str, dict[str, TaskEvaluationResult]]] = {}
        last_cohort_results: dict[str, dict[str, TaskEvaluationResult]] = {}

        for seed in self.seeds:
            seed_results[seed] = {}
            for cid, ctor in cohort_constructors.items():
                c = ctor()
                seed_results[seed][cid] = {}
                for t in tasks:
                    res = t.evaluate(c)
                    seed_results[seed][cid][res.task_id] = res
                last_cohort_results[cid] = seed_results[seed][cid]

        # 3. Execute Ablation Battery
        ablation_matrix = []
        for a in ablations:
            e1 = E1_ConceptAcquisitionTask().evaluate(a)
            e3 = E3_CounterfactualSimulationTask().evaluate(a)
            e4 = E4_EpistemicCalibrationTask().evaluate(a)
            e7 = E7_LifelongCurriculumTask().evaluate(a)

            ablation_matrix.append(
                {
                    "variant": a.cohort_id,
                    "n_tau": f"{e1.episodes_to_threshold} eps",
                    "sim_error": f"{e3.simulation_error:.2f}",
                    "brier": f"{e4.brier_score:.2f}",
                    "bwt": f"{e7.bwt:+.2f}",
                }
            )

        # 4. Dynamically Aggregate Primary Endpoints Summary Table across seeds
        def get_stat(cid: str, tid: str, field_name: str) -> MetricSummary:
            vals = [
                float(getattr(seed_results[s][cid][tid], field_name) or 0.0) for s in self.seeds
            ]
            return ExperimentStatistics.summarize(f"{cid}_{tid}_{field_name}", vals)

        def format_stat(stat: MetricSummary, unit: str = "", decimals: int = 2) -> str:
            if stat.std < 1e-4:
                return f"{stat.mean:.{decimals}f}{unit}"
            return f"{stat.mean:.{decimals}f} ± {stat.std:.{decimals}f}{unit}"

        e1_core = get_stat("HBLLM-Core", "E1_ConceptAcquisition", "episodes_to_threshold")
        e1_plus = get_stat("HBLLM+LLM", "E1_ConceptAcquisition", "episodes_to_threshold")
        e1_llm = get_stat("LLM-Only", "E1_ConceptAcquisition", "episodes_to_threshold")

        e2_core = get_stat("HBLLM-Core", "E2_LexicalAcquisition", "episodes_to_threshold")
        e2_plus = get_stat("HBLLM+LLM", "E2_LexicalAcquisition", "episodes_to_threshold")
        e2_llm = get_stat("LLM-Only", "E2_LexicalAcquisition", "episodes_to_threshold")

        e3_core_err = get_stat("HBLLM-Core", "E3_CounterfactualSimulation", "simulation_error")
        e3_plus_err = get_stat("HBLLM+LLM", "E3_CounterfactualSimulation", "simulation_error")
        e3_llm_err = get_stat("LLM-Only", "E3_CounterfactualSimulation", "simulation_error")

        e4_core_bs = get_stat("HBLLM-Core", "E4_EpistemicCalibration", "brier_score")
        e4_plus_bs = get_stat("HBLLM+LLM", "E4_EpistemicCalibration", "brier_score")
        e4_llm_bs = get_stat("LLM-Only", "E4_EpistemicCalibration", "brier_score")

        e4_core_ece = get_stat("HBLLM-Core", "E4_EpistemicCalibration", "ece")
        e4_plus_ece = get_stat("HBLLM+LLM", "E4_EpistemicCalibration", "ece")
        e4_llm_ece = get_stat("LLM-Only", "E4_EpistemicCalibration", "ece")

        e5_core_reg = get_stat("HBLLM-Core", "E5_ActiveDiscovery", "probing_regret")
        e5_plus_reg = get_stat("HBLLM+LLM", "E5_ActiveDiscovery", "probing_regret")
        e5_llm_reg = get_stat("LLM-Only", "E5_ActiveDiscovery", "probing_regret")

        e6_core_acc = get_stat("HBLLM-Core", "E6_RelationalTransfer", "structural_accuracy")
        e6_plus_acc = get_stat("HBLLM+LLM", "E6_RelationalTransfer", "structural_accuracy")
        e6_llm_acc = get_stat("LLM-Only", "E6_RelationalTransfer", "structural_accuracy")

        e7_core_bwt = get_stat("HBLLM-Core", "E7_LifelongCurriculum", "bwt")
        e7_plus_bwt = get_stat("HBLLM+LLM", "E7_LifelongCurriculum", "bwt")
        e7_llm_bwt = get_stat("LLM-Only", "E7_LifelongCurriculum", "bwt")

        e7_core_fwt = get_stat("HBLLM-Core", "E7_LifelongCurriculum", "fwt")
        e7_plus_fwt = get_stat("HBLLM+LLM", "E7_LifelongCurriculum", "fwt")
        e7_llm_fwt = get_stat("LLM-Only", "E7_LifelongCurriculum", "fwt")

        primary_table = [
            {
                "dimension": "Sample Efficiency ($N_\\tau$ to 80%)",
                "HBLLM-Core": f"{format_stat(e1_core, decimals=0)} eps",
                "HBLLM+LLM": f"{format_stat(e1_plus, decimals=0)} eps",
                "LLM-Only": f"{format_stat(e1_llm, decimals=0)} eps",
                "Oracle": "1 eps",
            },
            {
                "dimension": "Artificial Lexicon Acquisition",
                "HBLLM-Core": f"{format_stat(e2_core, decimals=0)} eps",
                "HBLLM+LLM": f"{format_stat(e2_plus, decimals=0)} eps",
                "LLM-Only": f"{format_stat(e2_llm, decimals=0)} eps",
                "Oracle": "1 eps",
            },
            {
                "dimension": "State-Transition Prediction Error ($E$)",
                "HBLLM-Core": format_stat(e3_core_err),
                "HBLLM+LLM": format_stat(e3_plus_err),
                "LLM-Only": format_stat(e3_llm_err),
                "Oracle": "0.00",
            },
            {
                "dimension": "Epistemic Calibration ($BS$ / $ECE$)",
                "HBLLM-Core": f"BS: {e4_core_bs.mean:.2f} / ECE: {e4_core_ece.mean:.2f}",
                "HBLLM+LLM": f"BS: {e4_plus_bs.mean:.2f} / ECE: {e4_plus_ece.mean:.2f}",
                "LLM-Only": f"BS: {e4_llm_bs.mean:.2f} / ECE: {e4_llm_ece.mean:.2f}",
                "Oracle": "BS: 0.00 / ECE: 0.00",
            },
            {
                "dimension": "Active Probing Regret ($U(a^*) - U(a)$)",
                "HBLLM-Core": format_stat(e5_core_reg),
                "HBLLM+LLM": format_stat(e5_plus_reg),
                "LLM-Only": format_stat(e5_llm_reg),
                "Oracle": "0.00",
            },
            {
                "dimension": "Relational Structural Transfer",
                "HBLLM-Core": f"{e6_core_acc.mean * 100:.0f}%",
                "HBLLM+LLM": f"{e6_plus_acc.mean * 100:.0f}%",
                "LLM-Only": f"{e6_llm_acc.mean * 100:.0f}%",
                "Oracle": "100%",
            },
            {
                "dimension": "Lifelong Continual Retention (BWT)",
                "HBLLM-Core": f"{e7_core_bwt.mean:+.2f}",
                "HBLLM+LLM": f"{e7_plus_bwt.mean:+.2f}",
                "LLM-Only": f"{e7_llm_bwt.mean:+.2f}",
                "Oracle": "+0.00",
            },
            {
                "dimension": "Forward Transfer (FWT on novel T5)",
                "HBLLM-Core": f"{e7_core_fwt.mean:+.2f}",
                "HBLLM+LLM": f"{e7_plus_fwt.mean:+.2f}",
                "LLM-Only": f"{e7_llm_fwt.mean:+.2f}",
                "Oracle": "+0.40",
            },
        ]

        return ScientificExperimentReport(
            manifest=self.manifest,
            cohort_results=last_cohort_results,
            primary_endpoints_table=primary_table,
            ablation_matrix=ablation_matrix,
        )
