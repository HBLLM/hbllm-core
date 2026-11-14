"""Closed-loop perceptual representation ablation benchmark.

Measures the causal impact of perception modes (``no_grounding``,
``handcrafted``, ``learned``, ``snn``) on unseen-palette zero-shot transfer in
:class:`~benchmarks.envs.visual_transfer.VisualMorphologyTransferEnv`.

In each seed, an agent completes Episode 1 on palette offset ``0`` (allowing up
to 3 attempts to observe environmental hazard/goal consequences via active
exploration), and is then evaluated zero-shot on Attempt 1 of Episode 2 under
an unseen palette offset (``10 + seed``), where spatial cues alone cannot
disambiguate hazard from goal without morphology/dynamics representation
transfer.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from benchmarks.envs.visual_transfer import VisualMorphologyTransferEnv
from hbllm.hcir.world.autonomous_epistemic_engine import AutonomousEpistemicEngine
from hbllm.hcir.world.cortical_grounding_bridge import CorticalGroundingBridge


@dataclass(frozen=True, slots=True)
class TrialResult:
    """Single-seed evaluation outcome for a perception mode."""

    seed: int
    mode: str
    ep1_attempts: int
    ep1_learned_hazard: bool
    ep1_learned_goal: bool
    ep2_solved: bool
    ep2_hazard_contact: bool
    ep2_reason: str
    ep2_steps: int
    mean_step_ms: float


@dataclass(frozen=True, slots=True)
class ModeSummary:
    """Aggregated transfer metrics across all seeds for a single mode."""

    mode: str
    seeds: int
    ep2_solve_rate: float
    ep2_hazard_death_rate: float
    ep2_timeout_rate: float
    ep2_mean_steps: float
    step_p50_ms: float
    step_p95_ms: float
    conditional_solve_rate_when_hazard_learned: float | None


def run_transfer_trial(mode: str, seed: int, max_steps: int = 35) -> TrialResult:
    """Run Episode 1 adaptation followed by Attempt-1 Episode 2 unseen-palette transfer."""
    engine = AutonomousEpistemicEngine(enable_logging=False, seed=seed)
    bridge: CorticalGroundingBridge | None = None
    if mode != "no_grounding":
        bridge = CorticalGroundingBridge(
            symbolic_theory=engine.symbolic_theory,
            surprise_engine=engine.surprise_engine,
            perception_mode=mode,
        )
        engine.attach_grounding_bridge(bridge)

    e1 = VisualMorphologyTransferEnv(palette_offset=0, max_steps=max_steps, dynamic_hazard=False)
    ep1_attempts = 0
    for _attempt in range(3):
        ep1_attempts += 1
        obs = e1.reset(seed=seed)
        engine.reset_episode()
        last_action: Any | None = None
        info: dict[str, Any] = {}
        for _step in range(max_steps):
            action, _meta = engine.step(obs, last_action=last_action)
            obs, _reward, _done, info = e1.step(action)
            last_action = action
            if obs.terminal is not None:
                engine.step(obs, last_action=last_action)
                break
        if info.get("success"):
            break

    learned_hazard = bool(engine.hazard_tracker.known_lethal_features)
    learned_goal = bool(engine.symbolic_theory.goal_features)

    e2 = VisualMorphologyTransferEnv(
        palette_offset=10 + seed,
        max_steps=max_steps,
        dynamic_hazard=False,
    )
    obs2 = e2.reset(seed=seed)
    engine.reset_episode()
    last_action2: Any | None = None
    step_times_ms: list[float] = []
    ep2_info: dict[str, Any] = {"success": False, "reason": "timeout", "steps": max_steps}
    for _step in range(max_steps):
        t0 = time.perf_counter()
        action2, _meta2 = engine.step(obs2, last_action=last_action2)
        step_times_ms.append((time.perf_counter() - t0) * 1000.0)
        obs2, _reward2, _done2, ep2_info = e2.step(action2)
        last_action2 = action2
        if obs2.terminal is not None:
            break

    reason = str(ep2_info.get("reason", "timeout"))
    return TrialResult(
        seed=seed,
        mode=mode,
        ep1_attempts=ep1_attempts,
        ep1_learned_hazard=learned_hazard,
        ep1_learned_goal=learned_goal,
        ep2_solved=bool(ep2_info.get("success", False)),
        ep2_hazard_contact=(reason == "hazard_contact"),
        ep2_reason=reason,
        ep2_steps=int(ep2_info.get("steps", max_steps)),
        mean_step_ms=float(statistics.mean(step_times_ms)) if step_times_ms else 0.0,
    )


def summarize_mode(mode: str, trials: list[TrialResult]) -> ModeSummary:
    """Aggregate per-seed trial metrics into a summary report."""
    n = len(trials)
    step_latencies = sorted(t.mean_step_ms for t in trials)
    p50_idx = int(0.50 * (n - 1)) if n > 0 else 0
    p95_idx = min(n - 1, int(0.95 * (n - 1))) if n > 0 else 0
    haz_trials = [t for t in trials if t.ep1_learned_hazard and t.ep1_learned_goal]
    cond_rate = sum(1 for t in haz_trials if t.ep2_solved) / len(haz_trials) if haz_trials else None
    return ModeSummary(
        mode=mode,
        seeds=n,
        ep2_solve_rate=round(sum(1 for t in trials if t.ep2_solved) / max(1, n), 4),
        ep2_hazard_death_rate=round(sum(1 for t in trials if t.ep2_hazard_contact) / max(1, n), 4),
        ep2_timeout_rate=round(sum(1 for t in trials if t.ep2_reason == "timeout") / max(1, n), 4),
        ep2_mean_steps=round(statistics.mean(t.ep2_steps for t in trials), 2),
        step_p50_ms=round(step_latencies[p50_idx], 2) if step_latencies else 0.0,
        step_p95_ms=round(step_latencies[p95_idx], 2) if step_latencies else 0.0,
        conditional_solve_rate_when_hazard_learned=(
            round(cond_rate, 4) if cond_rate is not None else None
        ),
    )


def run_ablation(num_seeds: int = 20, output_path: Path | None = None) -> dict[str, Any]:
    """Execute the full perception representation ablation across all four modes."""
    modes = ("no_grounding", "handcrafted", "learned", "snn")
    summaries: dict[str, dict[str, Any]] = {}
    all_trials: dict[str, list[dict[str, Any]]] = {}

    for mode in modes:
        mode_trials = [run_transfer_trial(mode, seed=s) for s in range(1, num_seeds + 1)]
        summary = summarize_mode(mode, mode_trials)
        summaries[mode] = asdict(summary)
        all_trials[mode] = [asdict(t) for t in mode_trials]

    report: dict[str, Any] = {
        "benchmark": "VisualMorphologyTransferEnv",
        "num_seeds": num_seeds,
        "summary": summaries,
        "trials": all_trials,
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """CLI entrypoint for the perception representation ablation."""
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Run unseen-palette perceptual representation transfer ablation."
    )
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to evaluate.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ablation/perception_representation_ablation.json"),
        help="Path to write JSON ablation report.",
    )
    args = parser.parse_args()
    report = run_ablation(num_seeds=args.seeds, output_path=args.output)
    print(f"Perception Representation Ablation ({args.seeds} seeds, VisualMorphologyTransferEnv)")
    for mode, stats in report["summary"].items():
        print(
            f"  {mode:12s} | ep2_solve={stats['ep2_solve_rate'] * 100:5.1f}% "
            f"| haz_death={stats['ep2_hazard_death_rate'] * 100:5.1f}% "
            f"| steps={stats['ep2_mean_steps']:5.1f} "
            f"| cond_solve={stats['conditional_solve_rate_when_hazard_learned']} "
            f"| p50={stats['step_p50_ms']:5.2f}ms"
        )
    print(f"Saved JSON report to: {args.output}")


if __name__ == "__main__":
    main()
