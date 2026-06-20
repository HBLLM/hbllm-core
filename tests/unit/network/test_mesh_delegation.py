"""Tests for Contract Delegation and Conflict Resolution."""

from hbllm.brain.mesh.capsule import TaskCapsule
from hbllm.brain.mesh.delegator import ContractDelegator, ContractOffer, DelegationResponse
from hbllm.brain.mesh.registry import NodeType
from hbllm.brain.mesh.resolver import ConflictProposal, DistributedConflictResolver


def test_contract_evaluation():
    delegator = ContractDelegator("phone_1")
    capsule = TaskCapsule()
    offer = ContractOffer(capsule=capsule, offered_by="desktop_1")

    # Accept under low pressure
    assert (
        delegator.evaluate_contract(offer, current_memory_pressure=0.2) == DelegationResponse.ACCEPT
    )

    # Decline under high pressure
    assert (
        delegator.evaluate_contract(offer, current_memory_pressure=0.9)
        == DelegationResponse.DECLINE
    )


def test_conflict_resolution():
    resolver = DistributedConflictResolver()

    p_phone = ConflictProposal("phone_1", NodeType.PHONE, "vehicle", "value_A")
    p_car = ConflictProposal("car_1", NodeType.CAR, "vehicle", "value_B")

    # Domain Authority rule: Car wins over phone for "vehicle" domain
    winner = resolver.resolve([p_phone, p_car])
    assert winner is p_car

    # Safety Override rule:
    p_phone_safety = ConflictProposal(
        "phone_1", NodeType.PHONE, "vehicle", "value_A", is_safety_critical=True
    )
    winner_safety = resolver.resolve([p_phone_safety, p_car])
    assert winner_safety is p_phone_safety
