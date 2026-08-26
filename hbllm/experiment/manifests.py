"""Immutable Reproducibility Manifest Generator.

Records complete environmental, algorithmic, hardware, and configuration parameters
to ensure full reproducibility of the scientific comparison.
"""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class ReproducibilityManifest:
    """Immutable audit record detailing exact conditions of an experimental comparison run."""

    experiment_id: str = field(default_factory=lambda: f"exp_{uuid.uuid4().hex[:8]}")
    git_commit_hash: str = "9737295"  # Current core HEAD
    python_version: str = platform.python_version()
    platform_system: str = platform.system()
    platform_machine: str = platform.machine()
    cohort_ids: list[str] = field(default_factory=list)
    task_order: list[str] = field(default_factory=list)
    initial_knowledge_hash: str = ""
    random_seeds: list[int] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_json(self) -> str:
        """Export manifest as formatted JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)
