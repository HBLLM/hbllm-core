"""Evaluation Protocol and Generality Harness.

Enforces strict evaluation discipline:
1. Frozen agent build with config hashing and Git SHA recording.
2. Standardized metrics: steps-to-first-success, steps-to-stable-policy, success@budget,
   and decisional latency quantiles (p50/p95).
3. Anti-overfitting tamper gate: Held-out suites write once to reports/heldout/<sha>_<date>.json
   and refuse to execute twice on the same Git commit SHA.
4. Multi-family evaluation (grid, continuous, symbolic) with normalized generality scoring.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class TaskResult:
    """Detailed performance metrics for a single task."""

    task_id: str
    family: str  # grid | continuous | symbolic
    split: str  # dev | heldout
    steps: int = 0
    max_budget: int = 200
    steps_to_first_success: int | None = None
    steps_to_stable_policy: int | None = None
    success_at_budget: bool = False
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


@dataclass
class EvaluationReport:
    """Full evaluation report across tasks and families."""

    git_sha: str
    config_hash: str
    timestamp: str
    split: str
    tasks: list[TaskResult] = field(default_factory=list)
    family_scores: dict[str, float] = field(default_factory=dict)
    generality_score: float = 0.0
    mean_p50_latency_ms: float = 0.0
    mean_p95_latency_ms: float = 0.0
    baselines: dict[str, Any] = field(default_factory=dict)
    env_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationProtocol:
    """Harness managing agent evaluation, verification gating, and scoring."""

    def __init__(
        self,
        reports_dir: Path | str = "reports",
        git_sha: str | None = None,
        config: dict[str, Any] | None = None,
        perception_mode: str = "handcrafted",
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.heldout_dir = self.reports_dir / "heldout"
        self.heldout_dir.mkdir(parents=True, exist_ok=True)
        self.git_sha = git_sha or self._get_current_git_sha()
        self.perception_mode = perception_mode
        self.config = config or {
            "model": "deterministic_hcir",
            "version": "1.0.0",
            "perception_mode": perception_mode,
        }
        self.config_hash = self._compute_config_hash(self.config)

    @staticmethod
    def _get_current_git_sha() -> str:
        """Retrieve HEAD Git SHA or fallback if unversioned."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except Exception:
            return "unversioned_git_sha"

    @staticmethod
    def _compute_config_hash(cfg: dict[str, Any]) -> str:
        """Deterministic sha256 hash of agent configuration and benchmark definition files."""
        hasher = hashlib.sha256()
        dumped = json.dumps(cfg, sort_keys=True)
        hasher.update(dumped.encode("utf-8"))

        # Include SHA-256 of benchmarks/envs/*.py
        envs_dir = Path("benchmarks/envs")
        if envs_dir.exists():
            for p in sorted(envs_dir.glob("*.py")):
                try:
                    hasher.update(p.name.encode("utf-8"))
                    hasher.update(p.read_bytes())
                except Exception:
                    pass

        # Include SHA-256 of benchmarks/suites/heldout.yaml
        heldout_yaml = Path("benchmarks/suites/heldout.yaml")
        if heldout_yaml.exists():
            try:
                hasher.update(heldout_yaml.read_bytes())
            except Exception:
                pass

        return hasher.hexdigest()[:16]

    def verify_heldout_gate(self) -> None:
        """Anti-tamper rule: Refuse to execute held-out evaluation twice on the same Git commit SHA and perception mode.
        Also verifies integrity of benchmark environments and held-out suite definition.
        """
        # Verify no uncommitted modifications to benchmark definition files
        try:
            res = subprocess.run(
                [
                    "git",
                    "diff",
                    "--name-only",
                    "HEAD",
                    "--",
                    "benchmarks/envs/",
                    "benchmarks/suites/heldout.yaml",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            modified = [line.strip() for line in res.stdout.splitlines() if line.strip()]
            if modified:
                raise RuntimeError(
                    f"Held-out anti-tamper violation: uncommitted changes detected in benchmark definition files: {modified}"
                )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise

        if not self.heldout_dir.exists():
            return

        # Check for existing report for this git sha and perception mode
        pattern = f"{self.git_sha}_{self.perception_mode}_*.json"
        existing = list(self.heldout_dir.glob(pattern))
        if not existing:
            # Also check legacy pattern without perception mode
            legacy = list(self.heldout_dir.glob(f"{self.git_sha}_*.json"))
            if legacy and self.perception_mode == "handcrafted":
                existing = legacy
        if existing:
            raise RuntimeError(
                f"Held-out evaluation already exists for git commit {self.git_sha} [{self.perception_mode}]: {existing[0].name}. "
                "Held-out tasks cannot be re-evaluated on the same commit to prevent overfitting."
            )

    def evaluate_task(
        self,
        task_id: str,
        family: str,
        split: str,
        run_fn: Callable[[], tuple[bool, int, list[float], list[str]]],
        max_budget: int = 200,
    ) -> TaskResult:
        """Run a single task under protocol measurement."""
        result = TaskResult(
            task_id=task_id,
            family=family,
            split=split,
            max_budget=max_budget,
        )

        try:
            success, steps, latencies, errors = run_fn()
            result.steps = steps
            result.errors = errors
            result.success_at_budget = success
            if success:
                result.steps_to_first_success = steps
                result.steps_to_stable_policy = steps  # In deterministic tasks, success is stable

            if latencies:
                sorted_lats = sorted(latencies)
                n = len(sorted_lats)
                result.p50_latency_ms = round(sorted_lats[int(n * 0.50)], 2)
                p95_idx = min(n - 1, int(n * 0.95))
                result.p95_latency_ms = round(sorted_lats[p95_idx], 2)

        except Exception as e:
            result.errors.append(f"Task execution failure: {e}")
            result.success_at_budget = False

        return result

    @staticmethod
    def _compute_env_hash() -> str:
        """Deterministic sha256 hash of benchmark environment definitions and suites."""
        hasher = hashlib.sha256()
        envs_dir = Path("benchmarks/envs")
        if envs_dir.exists():
            for p in sorted(envs_dir.glob("*.py")):
                try:
                    hasher.update(p.name.encode("utf-8"))
                    hasher.update(p.read_bytes())
                except Exception:
                    pass
        heldout_yaml = Path("benchmarks/suites/heldout.yaml")
        if heldout_yaml.exists():
            try:
                hasher.update(heldout_yaml.read_bytes())
            except Exception:
                pass
        return hasher.hexdigest()[:16]

    def compute_report(
        self,
        split: str,
        tasks: list[TaskResult],
        baselines: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """Aggregate per-task results into family scores and generality score."""
        families: dict[str, list[TaskResult]] = {}
        for t in tasks:
            families.setdefault(t.family, []).append(t)

        family_scores: dict[str, float] = {}
        for fam, t_list in families.items():
            if not t_list:
                family_scores[fam] = 0.0
                continue
            # Score per family is fraction of tasks solved normalized by step efficiency
            solved = [t for t in t_list if t.success_at_budget]
            if not solved:
                family_scores[fam] = 0.0
            else:
                success_rate = len(solved) / len(t_list)
                step_efficiency = sum(
                    max(0.0, 1.0 - (t.steps / max(1, t.max_budget))) for t in solved
                ) / len(solved)
                family_scores[fam] = round(0.5 * success_rate + 0.5 * step_efficiency, 4)

        generality_score = (
            round(sum(family_scores.values()) / len(family_scores), 4) if family_scores else 0.0
        )

        all_p50 = [t.p50_latency_ms for t in tasks if t.p50_latency_ms > 0]
        all_p95 = [t.p95_latency_ms for t in tasks if t.p95_latency_ms > 0]
        mean_p50 = round(sum(all_p50) / len(all_p50), 2) if all_p50 else 0.0
        mean_p95 = round(sum(all_p95) / len(all_p95), 2) if all_p95 else 0.0

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report = EvaluationReport(
            git_sha=self.git_sha,
            config_hash=self.config_hash,
            timestamp=ts,
            split=split,
            tasks=tasks,
            family_scores=family_scores,
            generality_score=generality_score,
            mean_p50_latency_ms=mean_p50,
            mean_p95_latency_ms=mean_p95,
            baselines=baselines or {},
            env_hash=self._compute_env_hash(),
        )

        # Write output file
        if split == "heldout":
            out_file = self.heldout_dir / f"{self.git_sha}_{self.perception_mode}_{ts}.json"
        else:
            dev_dir = self.reports_dir / "dev"
            dev_dir.mkdir(parents=True, exist_ok=True)
            out_file = dev_dir / f"dev_{self.perception_mode}_{ts}.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        return report
