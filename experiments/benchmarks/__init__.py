"""HBLLM Benchmarks Package (Decoupled from Core Cognitive Runtime)."""

from __future__ import annotations

from typing import Any

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

_EVAL_EXPORTS = {
    "AblationRegime",
    "DistributionShiftResult",
    "FamilyScorecard",
    "NoveltyAndDistributionShiftSuite",
    "ScaledBenchmarkReport",
    "ScaledGeneralizationBenchmark",
    "ScaledTaskTrace",
    "TransformationGeneralizationHarness",
    "WorldModelEvaluationHarness",
}

_TRANSFER_EXPORTS = {
    "CausalConditionResult",
    "CrossPathwayCausalBenchmark",
    "EnvironmentSpec",
    "FourConditionBenchmarkReport",
    "InteractiveTransferBenchmark",
    "PermutationTransferResult",
}


def __getattr__(name: str) -> Any:
    if name in _EVAL_EXPORTS:
        from experiments.benchmarks import evaluation

        return getattr(evaluation, name)
    if name in _TRANSFER_EXPORTS:
        from experiments.benchmarks import transfer

        return getattr(transfer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
