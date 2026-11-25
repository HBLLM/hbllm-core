"""Regression test proving grounding (enable_grounding=True) has positive causal utility (Δ > 0)."""

from __future__ import annotations

from benchmarks.perception_ablation import run_transfer_trial


def test_grounding_delivers_positive_transfer_delta_on_unseen_palettes() -> None:
    """Learned and SNN grounding transfer affordances across unseen palettes better than no_grounding."""
    seeds = range(5)
    baseline = [run_transfer_trial("no_grounding", s) for s in seeds]
    learned = [run_transfer_trial("learned", s) for s in seeds]
    snn = [run_transfer_trial("snn", s) for s in seeds]

    baseline_solved = sum(1 for t in baseline if t.ep2_solved)
    learned_solved = sum(1 for t in learned if t.ep2_solved)
    snn_solved = sum(1 for t in snn if t.ep2_solved)

    baseline_deaths = sum(1 for t in baseline if t.ep2_hazard_contact)
    learned_deaths = sum(1 for t in learned if t.ep2_hazard_contact)
    snn_deaths = sum(1 for t in snn if t.ep2_hazard_contact)

    assert learned_solved > baseline_solved, (
        f"Expected learned grounding to beat no_grounding ({learned_solved} vs {baseline_solved})"
    )
    assert snn_solved > baseline_solved, (
        f"Expected snn grounding to beat no_grounding ({snn_solved} vs {baseline_solved})"
    )
    assert learned_deaths < baseline_deaths
    assert snn_deaths < baseline_deaths
