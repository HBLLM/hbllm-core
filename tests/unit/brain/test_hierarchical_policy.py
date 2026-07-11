from hbllm.brain.core.cognitive_state import (
    CognitiveBudget,
    CognitivePolicy,
    HierarchicalCognitivePolicy,
)
from hbllm.brain.self_model.self_model import SelfModel


def test_hierarchical_policy_overrides():
    # 1. Simple Cascade
    global_policy = CognitivePolicy(reasoning_strategy="direct", simulation_depth=0)
    task_policy = CognitivePolicy(reasoning_strategy="GoT")

    policy = HierarchicalCognitivePolicy(global_policy=global_policy, task_policy=task_policy)
    resolved = policy.resolve()

    # Task policy overrides global reasoning strategy
    assert resolved.reasoning_strategy == "GoT"
    # Unspecified fields in task fall back to global
    assert resolved.simulation_depth == 0


def test_critical_safety_override():
    # 2. Safety Critical Override Check
    # If global has reflection enabled as a safety policy, task_policy trying to disable it should fail
    global_policy = CognitivePolicy(reflection_enabled=True)
    task_policy = CognitivePolicy(reflection_enabled=False)

    policy = HierarchicalCognitivePolicy(global_policy=global_policy, task_policy=task_policy)
    resolved = policy.resolve()

    assert resolved.reflection_enabled is True


def test_budget_resolution():
    global_budget = CognitiveBudget(simulation_budget=1, reasoning_budget=500)
    task_budget = CognitiveBudget(simulation_budget=5)

    global_policy = CognitivePolicy(budget=global_budget)
    task_policy = CognitivePolicy(budget=task_budget)

    policy = HierarchicalCognitivePolicy(global_policy=global_policy, task_policy=task_policy)
    resolved = policy.resolve()

    assert resolved.budget.simulation_budget == 5
    assert resolved.budget.reasoning_budget == 500


def test_self_model_policy_learning(tmp_path):
    self_model = SelfModel(data_dir=str(tmp_path))

    # Let's record positive outcomes for a deep policy on "math"
    deep_policy = CognitivePolicy(reasoning_strategy="GoT")

    # Verify insert
    self_model.record_policy_outcome(
        deep_policy, "math", success=True, cost_metrics={"latency_ms": 100.0}
    )
    self_model.record_policy_outcome(
        deep_policy, "math", success=True, cost_metrics={"latency_ms": 120.0}
    )

    # Record a failure for fast policy
    fast_policy = CognitivePolicy(reasoning_strategy="direct")
    self_model.record_policy_outcome(
        fast_policy, "math", success=False, cost_metrics={"latency_ms": 10.0}
    )

    # When select_policy is run under exploitation, it should pick the deep policy config (GoT strategy)
    # We loop to bypass the 15% exploration probability
    picked_got = False
    for _ in range(50):
        selected_policy = self_model.select_policy("math")
        effective = selected_policy.resolve()
        if effective.reasoning_strategy == "GoT":
            picked_got = True
            break

    assert picked_got is True
