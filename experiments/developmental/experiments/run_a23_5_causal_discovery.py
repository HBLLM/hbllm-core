#!/usr/bin/env python3
"""Executable Experiment Runner: A23.5-E1 Active Interventional Causal Discovery."""

from __future__ import annotations

import json
import logging

# Ensure plugins can be imported
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parents[3]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from plugins.developmental_adapter.benchmark import run_a23_5_benchmark

logger = logging.getLogger("A23.5_Experiment")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 85)
    print("EXECUTING MILESTONE A23.5-E1: ACTIVE INTERVENTIONAL CAUSAL DISCOVERY")
    print("Under Confounded Color/Mass Setup Across 5 Comparative Cohorts")
    print("=" * 85)

    results = run_a23_5_benchmark(n_trials=15, max_interventions=20, seed_base=200)

    print("\n" + results["ascii_table"] + "\n")

    # Save report
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "a23_5_results.json"
    report_file.write_text(json.dumps(results["data"], indent=2))
    print(f"Experimental report successfully written to:\n  {report_file}\n")


if __name__ == "__main__":
    main()
