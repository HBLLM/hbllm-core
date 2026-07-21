"""
Unit tests for Commit 6 of Phase 11 — Predictive World Kernel Integration.
Tests CounterfactualGraph and ActiveInferenceEngine.
"""

from __future__ import annotations

from hbllm.hcir.graph import ActionNode
from hbllm.hcir.world.active_inference import ActiveInferenceEngine
from hbllm.hcir.world.counterfactual_graph import CounterfactualGraph


def test_counterfactual_graph():
    graph = CounterfactualGraph(world_id="w1")
    branch = graph.create_counterfactual_branch("b1", action_intent="reduce_speed")
    assert branch.branch_id == "b1"
    assert branch.action_intent == "reduce_speed"
    assert graph.get_branch("b1") is not None


def test_active_inference_engine():
    engine = ActiveInferenceEngine()
    actions = [
        ActionNode(id="a1", intent="use_cooling", risk_factor=0.1, estimated_cost=5.0),
        ActionNode(id="a2", intent="ignore_warning", risk_factor=0.9, estimated_cost=10.0),
    ]

    evals = engine.evaluate_candidates(actions, information_gain_map={"a1": 0.8, "a2": 0.1})
    assert len(evals) == 2
    assert evals[0].action.id == "a1"
    assert evals[0].utility_score > evals[1].utility_score

    best = engine.select_best_action(actions)
    assert best.action.id == "a1"
