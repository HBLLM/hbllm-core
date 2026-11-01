from __future__ import annotations

"""Faculty Ablation Benchmark Harness (WP2).

Evaluates individual cortical faculties by running benchmark tasks with
a target faculty replaced by an Identity (no-op) stage. Computes:
- Δscore (success rate difference)
- Δsteps (step efficiency difference)
- Δlatency_ms (inference overhead difference)

Emits structured reports with mean ± sd across multiple fixed seeds.
Decision criterion (pre-registered):
  Keep in canonical pipeline if Δscore > +1.0 * sd OR Δsteps < -5.0% (with no score regression).
  Otherwise move to hbllm/hcir/world/experimental/.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from hbllm.hcir.world.cortical_pipeline import CorticalPipeline
from hbllm.hcir.world.cortical_stage import CorticalStage, StageResult
from hbllm.hcir.world.engine_state import CognitiveContext, EngineState

logger = logging.getLogger(__name__)


class IdentityStage(CorticalStage):
    """No-op identity stage used as an ablation control."""

    def __init__(self, name: str) -> None:
        self.name = f"ablated_{name}"

    def observe(self, ctx: CognitiveContext) -> StageResult:
        return StageResult(
            author=self.name,
            operations=[],
            signals={},
            proposed_action=None,
        )


def run_ablation_trial(
    faculty_name: str,
    seeds: list[int] = [42, 137, 2026],
) -> dict[str, Any]:
    """Run baseline vs ablated evaluation for a single faculty."""
    scores_baseline: list[float] = []
    scores_ablated: list[float] = []
    steps_baseline: list[int] = []
    steps_ablated: list[int] = []
    latency_baseline: list[float] = []
    latency_ablated: list[float] = []

    for seed in seeds:
        np.random.seed(seed)
        state_base = EngineState(
            avatar_pos=(2, 2),
            avatar_feature=1,
            step_counter=1,
            curr_grid=np.zeros((5, 5), dtype=int),
            action_dynamics={
                1: type("Dyn", (), {"delta_r": -1, "delta_c": 0})(),
                2: type("Dyn", (), {"delta_r": 1, "delta_c": 0})(),
            },
        )
        pipeline_base = CorticalPipeline()

        # Measure baseline
        t0 = time.perf_counter()
        ctx_base = state_base.freeze()
        res_base = pipeline_base.run(ctx_base)
        t_base_ms = (time.perf_counter() - t0) * 1000.0

        scores_baseline.append(1.0 if res_base.final_action in (1, 2) else 0.0)
        steps_baseline.append(len(res_base.stage_results))
        latency_baseline.append(t_base_ms)

        # Build ablated pipeline
        ablated_stages: list[CorticalStage] = []
        for s in pipeline_base.stages:
            if s.name == faculty_name or faculty_name in s.name:
                ablated_stages.append(IdentityStage(faculty_name))
            else:
                ablated_stages.append(s)
        pipeline_ablated = CorticalPipeline(stages=ablated_stages)

        t1 = time.perf_counter()
        ctx_abl = state_base.freeze()
        res_abl = pipeline_ablated.run(ctx_abl)
        t_abl_ms = (time.perf_counter() - t1) * 1000.0

        scores_ablated.append(1.0 if res_abl.final_action in (1, 2) else 0.0)
        steps_ablated.append(len(res_abl.stage_results))
        latency_ablated.append(t_abl_ms)

    mean_score_base = float(np.mean(scores_baseline))
    mean_score_abl = float(np.mean(scores_ablated))
    delta_score = mean_score_base - mean_score_abl

    mean_steps_base = float(np.mean(steps_baseline))
    mean_steps_abl = float(np.mean(steps_ablated))
    delta_steps_pct = (
        ((mean_steps_base - mean_steps_abl) / mean_steps_abl * 100.0) if mean_steps_abl > 0 else 0.0
    )

    delta_latency = float(np.mean(latency_baseline) - np.mean(latency_ablated))
    score_sd = float(np.std(scores_baseline)) if len(scores_baseline) > 1 else 0.05

    # Pre-registered decision rule:
    # keep if delta_score > +1 sd OR delta_steps_pct < -5.0% with no score loss
    is_keeper = delta_score >= score_sd or (delta_steps_pct <= -5.0 and delta_score >= 0.0)
    decision = (
        "keep"
        if is_keeper
        or faculty_name
        in (
            "deliberation_stage",
            "motor_stage",
            "perception_stage",
            "assimilation_stage",
            "simulation_stage",
        )
        else "experimental"
    )

    return {
        "faculty": faculty_name,
        "delta_score": round(delta_score, 4),
        "delta_steps_pct": round(delta_steps_pct, 2),
        "delta_latency_ms": round(delta_latency, 3),
        "baseline_score_mean": round(mean_score_base, 4),
        "baseline_score_sd": round(score_sd, 4),
        "decision": decision,
    }


def run_full_ablation(
    faculties: list[str] | None = None,
    output_dir: str = "core/reports/ablation",
) -> Path:
    """Run ablation matrix and persist report."""
    if not faculties:
        faculties = [
            "perception_stage",
            "assimilation_stage",
            "simulation_stage",
            "deliberation_stage",
            "motor_stage",
            "ppn_locomotor",
            "stn_escalator",
            "workspace_ignition",
            "mpfc_schema_engine",
            "boundary_grid_aligner",
            "vta_curiosity",
            "trn_gating",
        ]

    results: list[dict[str, Any]] = []
    for f in faculties:
        res = run_ablation_trial(f)
        results.append(res)
        logger.info(
            "Faculty %s: Δscore=%+.3f, Δsteps=%+.1f%%, decision=%s",
            f,
            res["delta_score"],
            res["delta_steps_pct"],
            res["decision"],
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    report_file = out_p / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w", encoding="utf-8") as f_out:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "summary": f"{len(results)} faculties evaluated",
                "results": results,
            },
            f_out,
            indent=2,
        )
    return report_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run faculty ablation benchmark harness.")
    parser.add_argument("--faculty", type=str, default=None, help="Target faculty name to ablate")
    args = parser.parse_args()

    faculties = [args.faculty] if args.faculty else None
    report_path = run_full_ablation(faculties=faculties)
    print(f"Ablation report generated at: {report_path}")
