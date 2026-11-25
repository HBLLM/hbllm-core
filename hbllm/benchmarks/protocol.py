"""Evaluation Protocol and Generality Harness.

Enforces strict evaluation discipline:
1. Frozen agent build with config hashing and Git SHA recording.
2. All-seed zero-shot evaluation with Wilson 95% confidence intervals, cross-seed transfer,
   same-layout replay curves, exploration death tracking, and latency quantiles (p50/p95).
3. Anti-overfitting tamper gate: Held-out v2 requires an explicit `eval-*` git tag on HEAD,
   refuses re-execution on the same commit, and appends an audit row to reports/heldout/LEDGER.md.
4. Multi-family evaluation with normalized generality scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute exact Wilson 95% score confidence interval (low, high) for a binomial rate."""
    if total <= 0:
        return (0.0, 0.0)
    k = max(0, min(successes, total))
    p = k / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    rad = math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    half = (z * rad) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


@dataclass
class TaskResult:
    """Detailed performance metrics for a single task across all manifest seeds."""

    task_id: str
    family: str
    split: str  # dev | heldout
    steps: int = 0
    max_budget: int = 200
    seed_count: int = 1
    steps_to_first_success: int | None = None
    steps_to_stable_policy: int | None = None
    success_at_budget: bool = False
    total_deaths: int = 0
    exploration_deaths: int = 0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    episode_successes: list[bool] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    zero_shot: dict[str, Any] = field(default_factory=dict)
    same_layout_replay: dict[str, Any] = field(default_factory=dict)
    lifelong: dict[str, Any] = field(default_factory=dict)
    one_shot_imitation: dict[str, Any] = field(default_factory=dict)
    cross_seed_transfer: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Full evaluation report across tasks and families."""

    git_sha: str
    config_hash: str
    timestamp: str
    split: str
    suite_name: str | None = None
    eval_tag: str | None = None
    tasks: list[TaskResult] = field(default_factory=list)
    family_scores: dict[str, float] = field(default_factory=dict)
    generality_score: float = 0.0  # Zero-shot (all seeds) generality — primary headline
    cross_seed_transfer_score: float = 0.0  # Cross-seed transfer solve rate — secondary headline
    lifelong_generality_score: float = 0.0  # Same-layout replay after N episodes (legacy alias)
    total_deaths: int = 0
    exploration_deaths: int = 0
    mean_p50_latency_ms: float = 0.0
    mean_p95_latency_ms: float = 0.0
    baselines: dict[str, Any] = field(default_factory=dict)
    env_hash: str = ""
    zero_shot: dict[str, Any] = field(default_factory=dict)
    same_layout_replay: dict[str, Any] = field(default_factory=dict)
    lifelong: dict[str, Any] = field(default_factory=dict)
    one_shot_imitation: dict[str, Any] = field(default_factory=dict)
    cross_seed_transfer: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvaluationProtocol:
    """Harness managing agent evaluation, verification gating, ledger recording, and scoring."""

    def __init__(
        self,
        reports_dir: Path | str = "reports",
        git_sha: str | None = None,
        config: dict[str, Any] | None = None,
        perception_mode: str = "handcrafted",
        eval_tag: str | None = None,
    ) -> None:
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.heldout_dir = self.reports_dir / "heldout"
        self.heldout_dir.mkdir(parents=True, exist_ok=True)
        self.git_sha = git_sha or self._get_current_git_sha()
        self.eval_tag = (
            eval_tag if eval_tag is not None else self._get_eval_tag_for_sha(self.git_sha)
        )
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
    def _get_eval_tag_for_sha(git_sha: str) -> str | None:
        """Return the first `eval-*` git tag pointing at `git_sha`, or None."""
        if not git_sha or git_sha == "unversioned_git_sha":
            return None
        try:
            res = subprocess.run(
                ["git", "tag", "--points-at", git_sha],
                capture_output=True,
                text=True,
                check=False,
            )
            for raw in res.stdout.splitlines():
                tag = raw.strip()
                if tag.startswith("eval-"):
                    return tag
        except Exception:
            return None
        return None

    @staticmethod
    def _compute_config_hash(cfg: dict[str, Any]) -> str:
        """Deterministic sha256 hash of agent configuration and benchmark definition files."""
        hasher = hashlib.sha256()
        dumped = json.dumps(cfg, sort_keys=True)
        hasher.update(dumped.encode("utf-8"))

        envs_dir = Path("benchmarks/envs")
        if envs_dir.exists():
            for p in sorted(envs_dir.glob("*.py")):
                hasher.update(p.name.encode("utf-8"))
                hasher.update(p.read_bytes())

        heldout_yaml = Path("benchmarks/suites/heldout.yaml")
        if heldout_yaml.exists():
            hasher.update(heldout_yaml.read_bytes())

        return hasher.hexdigest()[:16]

    def verify_heldout_gate(self, require_eval_tag: bool = True) -> None:
        """Anti-tamper rule:
        1. Refuse re-execution on the same Git SHA + perception mode.
        2. Require an explicit `eval-*` git tag on the evaluated commit.
        3. Refuse uncommitted modifications to benchmark envs or heldout.yaml.
        """
        if self.heldout_dir.exists():
            pattern = f"{self.git_sha}_{self.perception_mode}_*.json"
            existing = list(self.heldout_dir.glob(pattern))
            if not existing and self.perception_mode == "handcrafted":
                existing = list(self.heldout_dir.glob(f"{self.git_sha}_*.json"))
            if existing:
                raise RuntimeError(
                    f"Held-out evaluation already exists for git commit {self.git_sha} [{self.perception_mode}]: {existing[0].name}. "
                    "Held-out tasks cannot be re-evaluated on the same commit to prevent overfitting."
                )

        if require_eval_tag and not (self.eval_tag and self.eval_tag.startswith("eval-")):
            raise RuntimeError(
                f"Held-out anti-tamper violation: commit {self.git_sha} is not tagged with 'eval-*'. "
                "Tag milestone commits explicitly (`git tag eval-<milestone>`) before running held-out."
            )

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

    def evaluate_task(
        self,
        task_id: str,
        family: str,
        split: str,
        run_fn: Callable[
            [],
            tuple[bool, int, list[float], list[str]]
            | tuple[bool, int, list[float], list[str], dict[str, Any]],
        ],
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
            fn_res = run_fn()
            extra: dict[str, Any] = {}
            if len(fn_res) >= 5:
                success, steps, latencies, errors, extra = (
                    fn_res[0],
                    fn_res[1],
                    fn_res[2],
                    fn_res[3],
                    fn_res[4],
                )  # type: ignore[misc]
            else:
                success, steps, latencies, errors = fn_res[0], fn_res[1], fn_res[2], fn_res[3]  # type: ignore[misc]

            result.steps = steps
            result.errors = errors
            result.success_at_budget = bool(success) and not errors
            result.seed_count = int(extra.get("seed_count", 1))
            result.exploration_deaths = int(extra.get("exploration_deaths", 0))
            result.total_deaths = int(extra.get("total_deaths", result.exploration_deaths))
            result.episode_successes = extra.get("episode_successes", [result.success_at_budget])
            result.episode_steps = extra.get("episode_steps", [steps])
            result.steps_to_first_success = extra.get(
                "steps_to_first_success", steps if result.success_at_budget else None
            )
            result.steps_to_stable_policy = extra.get(
                "steps_to_stable_policy", steps if result.success_at_budget else None
            )
            result.zero_shot = extra.get(
                "zero_shot",
                {
                    "n": 1,
                    "success": result.success_at_budget,
                    "steps": steps,
                    "solve_rate": 1.0 if result.success_at_budget else 0.0,
                    "wilson_95_ci": list(wilson_ci(1 if result.success_at_budget else 0, 1)),
                },
            )
            replay_default = {
                "success": result.success_at_budget,
                "episodes": len(result.episode_successes),
                "successes": result.episode_successes,
                "steps": result.episode_steps,
                "steps_to_stable_policy": result.steps_to_stable_policy,
            }
            result.same_layout_replay = extra.get(
                "same_layout_replay", extra.get("lifelong", replay_default)
            )
            result.lifelong = extra.get("lifelong", result.same_layout_replay)
            result.one_shot_imitation = extra.get("one_shot_imitation", {})
            result.cross_seed_transfer = extra.get("cross_seed_transfer", {})

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
                hasher.update(p.name.encode("utf-8"))
                hasher.update(p.read_bytes())
        heldout_yaml = Path("benchmarks/suites/heldout.yaml")
        if heldout_yaml.exists():
            hasher.update(heldout_yaml.read_bytes())
        return hasher.hexdigest()[:16]

    @staticmethod
    def extract_zero_shot_trials(t: TaskResult) -> list[tuple[bool, int, int]]:
        """Return list of (success, steps, max_budget) zero-shot trials for a task."""
        zs = t.zero_shot or {}
        succs = zs.get("successes")
        steps_list = zs.get("steps_per_seed")
        if isinstance(succs, list) and succs:
            if not isinstance(steps_list, list) or len(steps_list) != len(succs):
                default_step = int(zs.get("steps", t.steps))
                steps_list = [default_step] * len(succs)
            return [
                (bool(ok), int(st), t.max_budget) for ok, st in zip(succs, steps_list, strict=False)
            ]
        ok_single = bool(zs.get("success", t.success_at_budget))
        st_single = int(zs.get("steps", t.steps))
        return [(ok_single, st_single, t.max_budget)]

    def _append_heldout_ledger(self, report: EvaluationReport, out_file: Path) -> None:
        """Append an audit row to reports/heldout/LEDGER.md."""
        ledger_path = self.heldout_dir / "LEDGER.md"
        if not ledger_path.exists():
            ledger_path.write_text(
                "# Held-Out Evaluation Ledger\n\n"
                "| Timestamp (UTC) | Git SHA | Eval Tag | Config Hash | Env Hash | Mode | Zero-Shot (Solve Rate, 95% CI) | Cross-Seed Transfer (95% CI) | Report |\n"
                "|---|---|---|---|---|---|---|---|---|\n",
                encoding="utf-8",
            )
        zs_sr = report.zero_shot.get("solve_rate", 0.0)
        zs_ci = report.zero_shot.get("wilson_95_ci", [0.0, 0.0])
        xs_sr = report.cross_seed_transfer.get("transfer_solve_rate", 0.0)
        xs_ci = report.cross_seed_transfer.get("transfer_wilson_95_ci", [0.0, 0.0])
        row = (
            f"| {report.timestamp} | `{report.git_sha[:8]}` | `{report.eval_tag or 'untagged'}` "
            f"| `{report.config_hash}` | `{report.env_hash}` | {self.perception_mode} "
            f"| {report.generality_score:.4f} (sr={zs_sr:.4f}, CI={zs_ci}) "
            f"| {xs_sr:.4f} (CI={xs_ci}) | `{out_file.name}` |\n"
        )
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(row)

    def _append_dev_ledger(self, report: EvaluationReport, out_file: Path, suite_slug: str) -> None:
        """Append an audit row to reports/dev/LEDGER.md."""
        dev_dir = self.reports_dir / "dev"
        dev_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = dev_dir / "LEDGER.md"
        if not ledger_path.exists():
            ledger_path.write_text(
                "# Dev Evaluation Ledger\n\n"
                "| Timestamp (UTC) | Git SHA | Suite | Mode | Zero-Shot Solve | Generality | Exploration Deaths | Mean p50 / p95 (ms) | Report |\n"
                "|---|---|---|---|---|---|---|---|---|\n",
                encoding="utf-8",
            )
        zs_sr = report.zero_shot.get("solve_rate", 0.0)
        row = (
            f"| {report.timestamp} | `{report.git_sha[:8]}` | `{suite_slug}` "
            f"| {self.perception_mode} | {zs_sr:.4f} | {report.generality_score:.4f} "
            f"| {report.exploration_deaths} | {report.mean_p50_latency_ms:.2f} / {report.mean_p95_latency_ms:.2f} "
            f"| `{out_file.name}` |\n"
        )
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(row)

    def compute_report(
        self,
        split: str,
        tasks: list[TaskResult],
        baselines: dict[str, Any] | None = None,
        suite_name: str | None = None,
    ) -> EvaluationReport:
        """Aggregate per-seed zero-shot, cross-seed transfer, and replay metrics."""
        families: dict[str, list[TaskResult]] = {}
        for t in tasks:
            families.setdefault(t.family, []).append(t)

        zs_fam_scores: dict[str, float] = {}
        zs_fam_solve_rates: dict[str, float] = {}
        zs_fam_ci: dict[str, list[float]] = {}
        lf_fam_scores: dict[str, float] = {}

        total_zs_trials = 0
        total_zs_solved = 0

        for fam, t_list in families.items():
            # Zero-shot across all seeds in family
            fam_trials: list[tuple[bool, int, int]] = []
            for t in t_list:
                fam_trials.extend(self.extract_zero_shot_trials(t))

            n_fam = len(fam_trials)
            solved_trials = [tr for tr in fam_trials if tr[0]]
            k_fam = len(solved_trials)
            total_zs_trials += n_fam
            total_zs_solved += k_fam

            zs_fam_solve_rates[fam] = round(k_fam / max(1, n_fam), 4)
            zs_fam_ci[fam] = list(wilson_ci(k_fam, n_fam))
            if not solved_trials:
                zs_fam_scores[fam] = 0.0
            else:
                sr = k_fam / n_fam
                eff = (
                    sum(max(0.0, 1.0 - (st / max(1, budget))) for _, st, budget in solved_trials)
                    / k_fam
                )
                zs_fam_scores[fam] = round(0.5 * sr + 0.5 * eff, 4)

            # Same-layout replay (legacy lifelong)
            lf_solved = [
                t
                for t in t_list
                if t.same_layout_replay.get(
                    "success", t.lifelong.get("success", t.success_at_budget)
                )
            ]
            if not lf_solved:
                lf_fam_scores[fam] = 0.0
            else:
                sr_lf = len(lf_solved) / len(t_list)
                eff_lf = sum(
                    max(
                        0.0,
                        1.0
                        - (
                            t.same_layout_replay.get(
                                "final_steps", t.lifelong.get("steps_final", t.steps)
                            )
                            if isinstance(
                                t.same_layout_replay.get(
                                    "final_steps", t.lifelong.get("steps_final", t.steps)
                                ),
                                int,
                            )
                            else t.steps
                        )
                        / max(1, t.max_budget),
                    )
                    for t in lf_solved
                ) / len(lf_solved)
                lf_fam_scores[fam] = round(0.5 * sr_lf + 0.5 * eff_lf, 4)

        zs_gen = (
            round(sum(zs_fam_scores.values()) / max(1, len(zs_fam_scores)), 4)
            if zs_fam_scores
            else 0.0
        )
        lf_gen = (
            round(sum(lf_fam_scores.values()) / max(1, len(lf_fam_scores)), 4)
            if lf_fam_scores
            else 0.0
        )
        lf_solved_count = sum(
            1
            for t in tasks
            if t.same_layout_replay.get("success", t.lifelong.get("success", t.success_at_budget))
        )

        stable_policies = [
            t.steps_to_stable_policy for t in tasks if t.steps_to_stable_policy is not None
        ]
        mean_stable = (
            round(sum(stable_policies) / len(stable_policies), 2) if stable_policies else None
        )
        total_deaths = sum(int(t.exploration_deaths) for t in tasks)
        total_all_deaths = sum(max(int(t.total_deaths), int(t.exploration_deaths)) for t in tasks)

        # One-shot imitation (demonstration mode)
        osi_fam_scores: dict[str, float] = {}
        for fam, t_list in families.items():
            osi_solved = [t for t in t_list if t.one_shot_imitation.get("success", False)]
            if not osi_solved:
                osi_fam_scores[fam] = 0.0
            else:
                sr = len(osi_solved) / len(t_list)
                eff = sum(
                    max(
                        0.0,
                        1.0
                        - (t.one_shot_imitation.get("steps", t.max_budget) / max(1, t.max_budget)),
                    )
                    for t in osi_solved
                ) / len(osi_solved)
                osi_fam_scores[fam] = round(0.5 * sr + 0.5 * eff, 4)

        osi_gen = (
            round(sum(osi_fam_scores.values()) / max(1, len(osi_fam_scores)), 4)
            if osi_fam_scores
            else 0.0
        )
        osi_solved_count = sum(1 for t in tasks if t.one_shot_imitation.get("success", False))

        zero_shot_dict = {
            "generality_score": zs_gen,
            "family_scores": zs_fam_scores,
            "family_solve_rates": zs_fam_solve_rates,
            "family_wilson_95_ci": zs_fam_ci,
            "n_trials": total_zs_trials,
            "n_solved": total_zs_solved,
            "solve_rate": round(total_zs_solved / max(1, total_zs_trials), 4),
            "wilson_95_ci": list(wilson_ci(total_zs_solved, total_zs_trials)),
            "total_deaths": total_all_deaths,
            "death_rate": round(total_all_deaths / max(1, total_zs_trials), 4),
            "exploration_deaths": total_deaths,
            "exploration_death_rate": round(total_deaths / max(1, total_zs_trials), 4),
        }
        replay_dict = {
            "generality_score": lf_gen,
            "family_scores": lf_fam_scores,
            "solve_rate": round(lf_solved_count / max(1, len(tasks)), 4),
            "mean_steps_to_stable_policy": mean_stable,
        }
        one_shot_imitation_dict = {
            "generality_score": osi_gen,
            "family_scores": osi_fam_scores,
            "solve_rate": round(osi_solved_count / max(1, len(tasks)), 4),
        }

        # Cross-seed procedural transfer (distinct layouts, shared engine)
        xs_tasks = [t for t in tasks if t.cross_seed_transfer.get("successes")]
        cross_seed_dict: dict[str, Any] = {}
        xs_headline = 0.0
        if xs_tasks:
            xs_fam_ok: dict[str, int] = {}
            xs_fam_tot: dict[str, int] = {}
            for t in xs_tasks:
                s_list = [bool(x) for x in t.cross_seed_transfer["successes"]]
                tr_list = s_list[1:] if len(s_list) > 1 else s_list
                xs_fam_ok[t.family] = xs_fam_ok.get(t.family, 0) + sum(tr_list)
                xs_fam_tot[t.family] = xs_fam_tot.get(t.family, 0) + len(tr_list)
            all_eps = sum(len(t.cross_seed_transfer["successes"]) for t in xs_tasks)
            all_ok = sum(sum(t.cross_seed_transfer["successes"]) for t in xs_tasks)
            tr_eps = sum(max(0, len(t.cross_seed_transfer["successes"]) - 1) for t in xs_tasks)
            tr_ok = sum(sum(t.cross_seed_transfer["successes"][1:]) for t in xs_tasks)
            xs_headline = round(tr_ok / max(1, tr_eps), 4)
            cross_seed_dict = {
                "n_episodes": all_eps,
                "n_transfer_episodes": tr_eps,
                "solve_rate": round(all_ok / max(1, all_eps), 4),
                "solve_rate_wilson_95_ci": list(wilson_ci(all_ok, all_eps)),
                "transfer_solve_rate": xs_headline,
                "transfer_wilson_95_ci": list(wilson_ci(tr_ok, tr_eps)),
                "family_solve_rates": {
                    f: round(xs_fam_ok[f] / max(1, xs_fam_tot[f]), 4) for f in xs_fam_tot
                },
                "family_wilson_95_ci": {
                    f: list(wilson_ci(xs_fam_ok[f], xs_fam_tot[f])) for f in xs_fam_tot
                },
                "total_deaths": sum(
                    max(
                        int(t.cross_seed_transfer.get("total_deaths", 0)),
                        int(t.cross_seed_transfer.get("exploration_deaths", 0)),
                    )
                    for t in xs_tasks
                ),
                "exploration_deaths": sum(
                    int(t.cross_seed_transfer.get("exploration_deaths", 0)) for t in xs_tasks
                ),
                "skills_consolidated": sum(
                    int(t.cross_seed_transfer.get("skills_consolidated", 0)) for t in xs_tasks
                ),
            }

        all_p50 = [t.p50_latency_ms for t in tasks if t.p50_latency_ms > 0]
        all_p95 = [t.p95_latency_ms for t in tasks if t.p95_latency_ms > 0]
        mean_p50 = round(sum(all_p50) / len(all_p50), 2) if all_p50 else 0.0
        mean_p95 = round(sum(all_p95) / len(all_p95), 2) if all_p95 else 0.0

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        suite_slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (suite_name or split).strip()).strip("_")
        if not suite_slug:
            suite_slug = split
        report = EvaluationReport(
            git_sha=self.git_sha,
            config_hash=self.config_hash,
            timestamp=ts,
            split=split,
            suite_name=suite_name or split,
            eval_tag=self.eval_tag,
            tasks=tasks,
            family_scores=zs_fam_scores,
            generality_score=zs_gen,
            cross_seed_transfer_score=xs_headline,
            lifelong_generality_score=lf_gen,
            total_deaths=total_all_deaths,
            exploration_deaths=total_deaths,
            mean_p50_latency_ms=mean_p50,
            mean_p95_latency_ms=mean_p95,
            baselines=baselines or {},
            env_hash=self._compute_env_hash(),
            zero_shot=zero_shot_dict,
            same_layout_replay=replay_dict,
            lifelong=replay_dict,
            one_shot_imitation=one_shot_imitation_dict,
            cross_seed_transfer=cross_seed_dict,
        )

        if split == "heldout":
            out_file = self.heldout_dir / f"{self.git_sha}_{self.perception_mode}_{ts}.json"
        else:
            dev_dir = self.reports_dir / "dev"
            dev_dir.mkdir(parents=True, exist_ok=True)
            prefix = f"{split}_{suite_slug}" if suite_slug != split else split
            out_file = dev_dir / f"{prefix}_{self.perception_mode}_{ts}.json"

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)

        if split == "heldout":
            self._append_heldout_ledger(report, out_file)
        else:
            self._append_dev_ledger(report, out_file, suite_slug)

        return report
