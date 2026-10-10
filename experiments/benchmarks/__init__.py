"""HBLLM Benchmarks Package (Decoupled from Core Cognitive Runtime)."""

from experiments.benchmarks.evaluation import (
    AblationRegime,
    DistributionShiftResult,
    FamilyScorecard,
    NoveltyAndDistributionShiftSuite,
    ScaledBenchmarkReport,
    ScaledGeneralizationBenchmark,
    ScaledTaskTrace,
    TransformationGeneralizationHarness,
    WorldModelEvaluationHarness,
)
from experiments.benchmarks.manifests import (
    ManifestTask,
    get_depth3_extended_manifest,
    get_depth4_exploratory_manifest,
    get_development_manifest,
    get_evaluation_manifest,
    get_frozen_manifest,
    verify_manifest_integrity,
)
from experiments.benchmarks.task_generators import (
    IndependentTaskGenerator,
    ProceduralTaskGenerator,
)
from experiments.benchmarks.transfer import (
    CausalConditionResult,
    CrossPathwayCausalBenchmark,
    EnvironmentSpec,
    FourConditionBenchmarkReport,
    InteractiveTransferBenchmark,
    PermutationTransferResult,
)

__all__ = [
    "ManifestTask",
    "get_development_manifest",
    "get_evaluation_manifest",
    "get_frozen_manifest",
    "verify_manifest_integrity",
    "get_depth3_extended_manifest",
    "get_depth4_exploratory_manifest",
    "ProceduralTaskGenerator",
    "IndependentTaskGenerator",
    "FamilyScorecard",
    "ScaledBenchmarkReport",
    "ScaledGeneralizationBenchmark",
    "ScaledTaskTrace",
    "DistributionShiftResult",
    "NoveltyAndDistributionShiftSuite",
    "WorldModelEvaluationHarness",
    "TransformationGeneralizationHarness",
    "AblationRegime",
    "EnvironmentSpec",
    "InteractiveTransferBenchmark",
    "PermutationTransferResult",
    "CausalConditionResult",
    "CrossPathwayCausalBenchmark",
    "FourConditionBenchmarkReport",
]
