"""Task Manifests subpackage for immutable frozen benchmark tasks."""

from experiments.benchmarks.manifests.deep_composition_manifest import (
    get_depth3_extended_manifest,
    get_depth4_exploratory_manifest,
)
from experiments.benchmarks.manifests.transformation_task_manifest import (
    ManifestTask,
    get_development_manifest,
    get_evaluation_manifest,
    get_frozen_manifest,
    verify_manifest_integrity,
)

__all__ = [
    "ManifestTask",
    "get_development_manifest",
    "get_evaluation_manifest",
    "get_frozen_manifest",
    "verify_manifest_integrity",
    "get_depth3_extended_manifest",
    "get_depth4_exploratory_manifest",
]
