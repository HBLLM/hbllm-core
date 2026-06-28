from dataclasses import FrozenInstanceError

import pytest

from hbllm.brain.autonomy.task_graph import Goal
from hbllm.brain.cognitive_state import CandidatePlan, CognitivePolicy, CognitiveState, Evidence


def test_cognitive_state_immutability():
    goal = Goal(name="Test Goal", description="A test goal")
    policy = CognitivePolicy()
    state = CognitiveState(goal=goal, policy=policy)

    # Trying to modify directly should fail
    with pytest.raises(FrozenInstanceError):
        state.version = 2  # type: ignore

    with pytest.raises(FrozenInstanceError):
        state.parent_state_id = "some_parent"  # type: ignore


def test_cognitive_state_derivation():
    goal = Goal(name="Original Goal", description="An original goal")
    policy = CognitivePolicy()
    state = CognitiveState(goal=goal, policy=policy)

    assert state.version == 1
    assert state.parent_state_id is None

    # Derive new state
    derived = state.derive_state(working_memory={"some_key": "some_value"})

    assert derived.version == 2
    assert derived.parent_state_id == state.state_id
    assert derived.state_id != state.state_id
    assert derived.working_memory == {"some_key": "some_value"}
    assert derived.goal == state.goal
    assert derived.policy == state.policy


def test_cognitive_state_forking():
    goal = Goal(name="Forking Goal", description="A goal for forking")
    policy = CognitivePolicy()
    state = CognitiveState(goal=goal, policy=policy)

    forked = state.fork()

    assert forked.parent_state_id == state.state_id
    assert forked.version == 2
    assert forked.state_id != state.state_id


def test_evidence_and_candidate_plan_immutability():
    evidence = Evidence(source="test_sensor", confidence=0.8)
    with pytest.raises(FrozenInstanceError):
        evidence.confidence = 0.9  # type: ignore

    plan = CandidatePlan(origin="planner", confidence=0.7)
    with pytest.raises(FrozenInstanceError):
        plan.confidence = 0.8  # type: ignore


def test_to_dict_serialization():
    goal = Goal(name="Serialization Goal", description="Serialization test")
    policy = CognitivePolicy(reasoning_strategy="direct")
    evidence = Evidence(source="system", confidence=1.0)
    plan = CandidatePlan(origin="planner", confidence=0.95)

    state = CognitiveState(
        goal=goal,
        policy=policy,
        evidence_ledger={"fact_1": evidence},
        candidate_plans=[plan],
        working_memory={"data": 123},
    )

    state_dict = state.to_dict()

    assert state_dict["state_id"] == state.state_id
    assert state_dict["version"] == 1
    assert state_dict["goal"]["name"] == "Serialization Goal"
    assert state_dict["policy"]["reasoning_strategy"] == "direct"
    assert state_dict["evidence_ledger"]["fact_1"]["source"] == "system"
    assert state_dict["evidence_ledger"]["fact_1"]["confidence"] == 1.0
    assert len(state_dict["candidate_plans"]) == 1
    assert state_dict["candidate_plans"][0]["origin"] == "planner"
    assert state_dict["working_memory"] == {"data": 123}
