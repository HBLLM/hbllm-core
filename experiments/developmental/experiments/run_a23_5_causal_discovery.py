#!/usr/bin/env python3
"""Executable Experiment Runner: A23.5 Developmental Causal Discovery Battery.

Runs:
1. Experiment E1: Active Interventional Causal Discovery under Confounding
2. Experiment E2: Causal Variable Invariance across Randomized Confounded Worlds
Outputs complete discrete empirical distributions and saves structured reports.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[3]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from plugins.developmental_adapter.benchmark import run_a23_5_benchmark

logger = logging.getLogger("A23.5_ExperimentBattery")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # EXPERIMENT E1: FIXED CONFOUNDED TRAINING WORLD
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING A23.5-E1: ACTIVE INTERVENTIONAL CAUSAL DISCOVERY")
    print("Observational Setup: Color ↔ Mass Confounder (15 randomized trials)")
    print("=" * 85)

    e1_results = run_a23_5_benchmark(
        n_trials=15,
        max_interventions=20,
        seed_base=200,
        scenario="confounded_train_world",
    )
    print("\n" + e1_results["ascii_table"])
    print("\n" + e1_results["discrete_table"])

    e1_file = reports_dir / "a23_5_e1_fixed_confounder_results.json"
    e1_file.write_text(json.dumps(e1_results["data"], indent=2))
    print(f"E1 report saved to: {e1_file}")

    # ─────────────────────────────────────────────────────────────────────────────
    # EXPERIMENT E2: CAUSAL VARIABLE INVARIANCE (DYNAMIC SURFACE CONVOLUTIONS)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING A23.5-E2: CAUSAL VARIABLE INVARIANCE")
    print("Observational Setup: Surface Confounders Vary Dynamically Across Episodes")
    print("=" * 85)

    e2_results = run_a23_5_benchmark(
        n_trials=15,
        max_interventions=20,
        seed_base=300,
        scenario="randomized_confounded_world",
    )
    print("\n" + e2_results["ascii_table"])
    print("\n" + e2_results["discrete_table"])

    e2_file = reports_dir / "a23_5_e2_variable_invariance_results.json"
    e2_file.write_text(json.dumps(e2_results["data"], indent=2))
    print(f"E2 report saved to: {e2_file}")


if __name__ == "__main__":
    main()
