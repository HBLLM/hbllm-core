#!/usr/bin/env python3
"""Executable Experiment Battery: Milestone A23 Wave 2.

Evaluates:
1. Stage D4: Affordance Discovery (ROLLABLE, SLIDABLE, GRASPABLE) & Out-of-Distribution Transfer
2. Stage D2: Spatial Containment & Transport Invariance (INSIDE(x, y) ∧ MOVE(y) => MOVE(x))
3. Stage D5: Tool Use & Compositional Action Chains (Indirect Reach via Stick)
Outputs complete discrete empirical distributions and saves structured reports.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

CORE_ROOT = Path(__file__).resolve().parents[3]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from plugins.developmental_adapter.affordance_discovery import AffordanceDiscoveryEngine
from plugins.developmental_adapter.blank_brain import create_blank_brain_substrate
from plugins.developmental_adapter.environment import BabyWorldEnvironment
from plugins.developmental_adapter.perception import DevelopmentalPerceptionAdapter
from plugins.developmental_adapter.spatial_containment import SpatialContainmentEngine
from plugins.developmental_adapter.tool_learning import ToolLearningEngine

logger = logging.getLogger("A23_Wave2_Battery")


def run_wave2_experiment_battery(n_trials: int = 15) -> dict[str, Any]:
    """Execute Wave 2 developmental experiment battery across 15 randomized trials."""
    reports: dict[str, Any] = {
        "stage_d4_affordances": {},
        "stage_d2_containment": {},
        "stage_d5_tool_use": {},
    }

    # ─────────────────────────────────────────────────────────────────────────────
    # STAGE D4: AFFORDANCE DISCOVERY & TRANSFER
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING MILESTONE A23.6 / STAGE D4: AFFORDANCE DISCOVERY & TRANSFER")
    print("=" * 85)

    d4_discovery_counts: list[int] = []
    d4_transfer_accs: list[float] = []

    held_out_entities = [
        {
            "id": "unseen_sphere_1",
            "shape": "ellipsoid",
            "base_shape": "ball",
            "ground_truth_affordances": ["ROLLABLE", "SLIDABLE", "GRASPABLE"],
        },
        {
            "id": "unseen_cube_1",
            "shape": "polyhedron",
            "base_shape": "block",
            "ground_truth_affordances": ["SLIDABLE", "GRASPABLE"],
        },
    ]

    for trial in range(n_trials):
        env = BabyWorldEnvironment(seed=500 + trial)
        env.reset("affordance_discovery_world")
        substrate = create_blank_brain_substrate()
        perception = DevelopmentalPerceptionAdapter()
        engine = AffordanceDiscoveryEngine(substrate, perception, env)

        engine.discover_affordances(max_interventions=20)
        acc, _ = engine.evaluate_novel_entity_transfer(held_out_entities)

        d4_discovery_counts.append(engine.interventions_count)
        d4_transfer_accs.append(acc)

    d4_median_ntau = sorted(d4_discovery_counts)[len(d4_discovery_counts) // 2]
    d4_mean_transfer = sum(d4_transfer_accs) / len(d4_transfer_accs)

    reports["stage_d4_affordances"] = {
        "trials": n_trials,
        "median_interventions_n_tau": d4_median_ntau,
        "mean_novel_transfer_acc": d4_mean_transfer,
        "all_n_tau": d4_discovery_counts,
        "all_transfer_acc": d4_transfer_accs,
    }
    print(
        f"Stage D4 Results: Median N_tau = {d4_median_ntau}, Transfer Acc = {d4_mean_transfer * 100:.1f}%"
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # STAGE D2: SPATIAL CONTAINMENT & TRANSPORT
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING MILESTONE A23.7 / STAGE D2: SPATIAL CONTAINMENT & TRANSPORT")
    print("=" * 85)

    d2_confirmed_count = 0
    d2_permanence_count = 0

    for trial in range(n_trials):
        env = BabyWorldEnvironment(seed=600 + trial)
        env.reset("containment_world")
        substrate = create_blank_brain_substrate()
        perception = DevelopmentalPerceptionAdapter()
        engine = SpatialContainmentEngine(substrate, perception, env)

        schema = engine.discover_containment_transport_schema(
            container_id="obj_container_box",
            contained_id="obj_toy_ball",
            outside_id="obj_outside_ball",
        )
        if schema["confirmed"]:
            d2_confirmed_count += 1

        perm = engine.verify_object_permanence_during_transport(
            container_id="obj_container_box",
            contained_id="obj_toy_ball",
        )
        if perm["permanence_preserved"]:
            d2_permanence_count += 1

    reports["stage_d2_containment"] = {
        "trials": n_trials,
        "transport_invariance_discovery_rate": d2_confirmed_count / n_trials,
        "object_permanence_tracking_rate": d2_permanence_count / n_trials,
    }
    print(
        f"Stage D2 Results: Transport Invariance = {d2_confirmed_count / n_trials * 100:.1f}%, Permanence Tracking = {d2_permanence_count / n_trials * 100:.1f}%"
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # STAGE D5: TOOL USE & COMPOSITIONAL CAUSAL CHAINS
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING MILESTONE A23.8 / STAGE D5: TOOL USE & COMPOSITIONAL CHAINS")
    print("=" * 85)

    d5_success_count = 0
    d5_transfer_accs: list[float] = []

    held_out_scenarios = [
        {
            "id": "long_rake_test",
            "tool_length": 1.2,
            "target_distance": 2.5,
            "tool_mass": 1.2,
            "actual_success": True,
        },
        {
            "id": "short_fork_test",
            "tool_length": 0.3,
            "target_distance": 2.5,
            "tool_mass": 0.5,
            "actual_success": False,
        },
        {
            "id": "heavy_bar_test",
            "tool_length": 2.0,
            "target_distance": 2.0,
            "tool_mass": 30.0,
            "actual_success": False,
        },
    ]

    candidate_tools = ["obj_heavy_boulder", "obj_short_twig", "obj_stick_tool"]

    for trial in range(n_trials):
        env = BabyWorldEnvironment(seed=700 + trial)
        env.reset("tool_use_world")
        substrate = create_blank_brain_substrate()
        perception = DevelopmentalPerceptionAdapter()
        engine = ToolLearningEngine(substrate, perception, env)

        res = engine.discover_and_execute_tool_chain(
            target_id="obj_distant_reward",
            candidate_tool_ids=candidate_tools,
        )
        if res["success"] and res["effective_tool_id"] == "obj_stick_tool":
            d5_success_count += 1

        t_acc, _ = engine.evaluate_novel_tool_transfer(held_out_scenarios)
        d5_transfer_accs.append(t_acc)

    d5_mean_transfer = sum(d5_transfer_accs) / len(d5_transfer_accs)

    reports["stage_d5_tool_use"] = {
        "trials": n_trials,
        "tool_chain_synthesis_rate": d5_success_count / n_trials,
        "novel_tool_transfer_acc": d5_mean_transfer,
        "all_transfer_acc": d5_transfer_accs,
    }
    print(
        f"Stage D5 Results: Tool Chain Synthesis = {d5_success_count / n_trials * 100:.1f}%, Novel Tool Transfer = {d5_mean_transfer * 100:.1f}%"
    )

    return reports


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    results = run_wave2_experiment_battery(n_trials=15)

    out_file = reports_dir / "a23_wave2_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nWave 2 Report saved to: {out_file}")


if __name__ == "__main__":
    main()
