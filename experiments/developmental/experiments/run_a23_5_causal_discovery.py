#!/usr/bin/env python3
"""Executable Experiment Runner: A23.5 Developmental Causal Discovery Battery.

Runs the complete publication-grade scientific battery:
1. Experiment E1: Active Interventional Causal Discovery under Confounding (Mass)
2. Experiment E2: Causal Variable Invariance across 8-Mode Multi-Attribute Confounders
3. Experiment E3: Novel Latent Causal Physical Mechanisms Battery (Friction, Static Force, Aperture Clearance)
4. Experiment E4: Compositional Causal Transfer into A20 Relational Structures

Outputs complete discrete empirical distributions and saves structured JSON reports.
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

from hbllm.hcir.graph import CognitiveGraph, PhysicalEntityNode
from plugins.developmental_adapter.a20_transfer_bridge import A20RelationalTransferBridge
from plugins.developmental_adapter.benchmark import run_a23_5_benchmark

logger = logging.getLogger("A23.5_ExperimentBattery")


def run_a20_relational_transfer_battery() -> dict[str, Any]:
    """Execute Experiment E4: Analogical and Compositional Transfer into A20 Relational Structures."""
    bridge = A20RelationalTransferBridge()
    transfer_results: list[dict[str, Any]] = []

    # 1. Containment Transport Schema -> Industrial Hopper Domain
    schema_containment = bridge.lift_containment_schema()
    target_hopper = CognitiveGraph()
    target_hopper.add_node(
        PhysicalEntityNode(
            id="industrial_hopper",
            entity_type="container",
            properties={
                "is_closed": False,
                "is_mobile": True,
                "is_container": True,
                "capacity": 50.0,
            },
        )
    )
    target_hopper.add_node(
        PhysicalEntityNode(
            id="ore_pellet",
            entity_type="physical_entity",
            properties={"mass": 0.5, "rollable": True},
        )
    )
    res_containment = bridge.transfer_to_target_domain(schema_containment, target_hopper)
    transfer_results.append(
        {
            "test_case": "containment_to_hopper",
            "source_schema": schema_containment.name,
            "target_domain": "industrial_hopper_transport",
            "is_applicable": res_containment["is_applicable"],
            "is_rejected": res_containment["is_rejected"],
            "alignment_score": res_containment["score"],
            "candidate_actions": res_containment["candidate_actions"],
            "role_mapping": res_containment["role_mapping"],
            "expected_applicable": True,
        }
    )

    # 2. Containment Negative Transfer Rejection -> Sealed Vault
    target_vault = CognitiveGraph()
    target_vault.add_node(
        PhysicalEntityNode(
            id="sealed_vault",
            entity_type="container",
            properties={"is_container": True, "is_closed": True},  # Violates is_closed == False
        )
    )
    target_vault.add_node(
        PhysicalEntityNode(
            id="gold_ingot",
            entity_type="physical_entity",
            properties={"mass": 2.0},
        )
    )
    res_vault = bridge.transfer_to_target_domain(schema_containment, target_vault)
    transfer_results.append(
        {
            "test_case": "containment_sealed_vault_negative_rejection",
            "source_schema": schema_containment.name,
            "target_domain": "sealed_vault_containment",
            "is_applicable": res_vault["is_applicable"],
            "is_rejected": res_vault["is_rejected"],
            "alignment_score": res_vault["score"],
            "violations": res_vault["violations"],
            "expected_applicable": False,
        }
    )

    # 3. Tool Reach Extension Schema -> Robotic Lever / Crowbar Domain
    schema_tool = bridge.lift_tool_reach_schema()
    target_robotic = CognitiveGraph()
    target_robotic.add_node(
        PhysicalEntityNode(id="robotic_arm", entity_type="agent", properties={})
    )
    target_robotic.add_node(
        PhysicalEntityNode(
            id="crowbar",
            entity_type="tool",
            properties={"is_rigid": True, "mass": 1.2},
        )
    )
    target_robotic.add_node(
        PhysicalEntityNode(id="stuck_crate", entity_type="physical_entity", properties={})
    )
    res_tool = bridge.transfer_to_target_domain(schema_tool, target_robotic)
    transfer_results.append(
        {
            "test_case": "tool_reach_to_robotic_lever",
            "source_schema": schema_tool.name,
            "target_domain": "robotic_lever_manipulation",
            "is_applicable": res_tool["is_applicable"],
            "is_rejected": res_tool["is_rejected"],
            "alignment_score": res_tool["score"],
            "candidate_actions": res_tool["candidate_actions"],
            "role_mapping": res_tool["role_mapping"],
            "expected_applicable": True,
        }
    )

    # 4. Discovered Causal Friction Rule -> Conveyor Pushing Domain
    causal_rule = {
        "rule_id": "rule_push_friction_discovered",
        "action": "PUSH",
        "precondition": {"property": "surface_friction", "operator": "<", "value": 1.0},
        "consequence": "MOVES",
        "empirical_support_count": 10,
    }
    schema_friction_causal = bridge.lift_causal_rule_schema(causal_rule)
    target_conveyor = CognitiveGraph()
    target_conveyor.add_node(PhysicalEntityNode(id="conveyor_pusher", entity_type="agent"))
    target_conveyor.add_node(
        PhysicalEntityNode(
            id="smooth_pallet",
            entity_type="physical_entity",
            properties={"surface_friction": 0.25},
        )
    )
    res_causal = bridge.transfer_to_target_domain(schema_friction_causal, target_conveyor)
    transfer_results.append(
        {
            "test_case": "causal_rule_to_conveyor_pusher",
            "source_schema": schema_friction_causal.name,
            "target_domain": "conveyor_system",
            "is_applicable": res_causal["is_applicable"],
            "is_rejected": res_causal["is_rejected"],
            "alignment_score": res_causal["score"],
            "candidate_actions": res_causal["candidate_actions"],
            "role_mapping": res_causal["role_mapping"],
            "expected_applicable": True,
        }
    )

    # Compute aggregate transfer performance metrics
    n_cases = len(transfer_results)
    positive_transfers = [r for r in transfer_results if r["expected_applicable"]]
    negative_transfers = [r for r in transfer_results if not r["expected_applicable"]]

    pos_success_rate = (
        sum(1 for r in positive_transfers if r["is_applicable"]) / len(positive_transfers)
        if positive_transfers
        else 0.0
    )
    neg_rejection_rate = (
        sum(1 for r in negative_transfers if r["is_rejected"]) / len(negative_transfers)
        if negative_transfers
        else 0.0
    )
    overall_accuracy = (
        sum(1 for r in transfer_results if r["is_applicable"] == r["expected_applicable"]) / n_cases
    )

    mean_alignment_score = sum(r["alignment_score"] for r in positive_transfers) / len(
        positive_transfers
    )

    return {
        "transfer_results": transfer_results,
        "metrics": {
            "total_test_cases": n_cases,
            "positive_transfer_success_rate": round(pos_success_rate, 4),
            "negative_transfer_rejection_rate": round(neg_rejection_rate, 4),
            "overall_analogical_accuracy": round(overall_accuracy, 4),
            "mean_positive_alignment_score": round(mean_alignment_score, 4),
        },
    }


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
    # EXPERIMENT E2: CAUSAL VARIABLE INVARIANCE (8-MODE MULTI-ATTRIBUTE CONVOLUTIONS)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING A23.5-E2: CAUSAL VARIABLE INVARIANCE")
    print("Observational Setup: 8-Mode Multi-Attribute Orthogonal Confounder Invariance")
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

    # ─────────────────────────────────────────────────────────────────────────────
    # EXPERIMENT E3: NOVEL CAUSAL MECHANISMS BATTERY
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING A23.5-E3: NOVEL CAUSAL MECHANISMS BATTERY")
    print("Mechanisms: E3.1 Friction | E3.2 Static Resistance Force | E3.3 Aperture Clearance")
    print("=" * 85)

    e3_mechanisms = [
        ("friction_confounded_world", "Surface Friction (μ < 1.0)"),
        ("force_confounded_world", "Static Force Threshold (Threshold < 5.0 N)"),
        ("aperture_confounded_world", "Geometric Aperture Clearance (Diameter <= 0.5 m)"),
    ]
    e3_combined_reports: dict[str, Any] = {}

    for scenario_key, label in e3_mechanisms:
        print(f"\n--- Sub-Mechanism: {label} ---")
        mech_res = run_a23_5_benchmark(
            n_trials=10,
            max_interventions=20,
            seed_base=400,
            scenario=scenario_key,
        )
        print(mech_res["ascii_table"])
        e3_combined_reports[scenario_key] = mech_res["data"]

    e3_file = reports_dir / "a23_5_e3_novel_causal_mechanisms_results.json"
    e3_file.write_text(json.dumps(e3_combined_reports, indent=2))
    print(f"\nE3 comprehensive report saved to: {e3_file}")

    # ─────────────────────────────────────────────────────────────────────────────
    # EXPERIMENT E4: COMPOSITIONAL CAUSAL TRANSFER INTO A20 RELATIONAL STRUCTURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 85)
    print("EXECUTING A23.5-E4: COMPOSITIONAL CAUSAL TRANSFER INTO A20 RELATIONAL STRUCTURES")
    print("Structure Mapping, Zero-Shot Action Synthesis, & Negative Transfer Rejection")
    print("=" * 85)

    e4_results = run_a20_relational_transfer_battery()
    print("\nA20 Relational Transfer Performance Summary:")
    print(f"  Total Domain Test Cases:            {e4_results['metrics']['total_test_cases']}")
    print(
        f"  Positive Transfer Success Rate:     {e4_results['metrics']['positive_transfer_success_rate'] * 100:.1f}%"
    )
    print(
        f"  Negative Transfer Rejection Rate:   {e4_results['metrics']['negative_transfer_rejection_rate'] * 100:.1f}%"
    )
    print(
        f"  Overall Analogical Accuracy:        {e4_results['metrics']['overall_analogical_accuracy'] * 100:.1f}%"
    )
    print(
        f"  Mean Alignment Score:               {e4_results['metrics']['mean_positive_alignment_score']:.4f}"
    )

    for case in e4_results["transfer_results"]:
        status_sym = "✓ PASS" if case["is_applicable"] == case["expected_applicable"] else "✗ FAIL"
        print(
            f"  [{status_sym}] {case['test_case']} -> Status: {'APPLICABLE' if case['is_applicable'] else 'REJECTED'}"
        )

    e4_file = reports_dir / "a23_5_e4_relational_transfer_results.json"
    e4_file.write_text(json.dumps(e4_results, indent=2))
    print(f"\nE4 report saved to: {e4_file}")
    print("\n" + "=" * 85)
    print("ALL A23.5 SCIENTIFIC BENCHMARKS SUCCESSFULLY EXECUTED AND RECORDED.")
    print("=" * 85)


if __name__ == "__main__":
    main()
