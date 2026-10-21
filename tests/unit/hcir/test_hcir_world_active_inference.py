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

    pred_br = branch.to_causal_predicate()
    assert pred_br.variable == "cf_branch:b1"
    assert pred_br.value == "reduce_speed"

    tx_br = branch.to_transaction_operations()
    assert len(tx_br) == 1
    assert tx_br[0].node_id == "cf_branch_b1"

    preds_g = graph.export_causal_predicates()
    assert len(preds_g) == 1

    txs_g = graph.to_transaction_operations()
    assert len(txs_g) == 1


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

    # Forward architecture neuro-symbolic and transaction exports
    pred_cand = evals[0].to_causal_predicate()
    assert pred_cand.variable == "candidate_utility:a1"
    assert pred_cand.value == evals[0].utility_score

    afford_cand = evals[0].to_affordance()
    assert afford_cand.action_id == "use_cooling"
    assert afford_cand.confidence == evals[0].utility_score

    tx_cand = evals[0].to_transaction_operations()
    assert len(tx_cand) == 1
    assert tx_cand[0].node_id == "act_cand_a1"

    best_res, txs_sel = engine.select_and_transact(actions)
    assert best_res.action.id == "a1"
    assert len(txs_sel) == 1

    all_cand_preds = engine.export_causal_predicates(actions)
    assert len(all_cand_preds) == 2
