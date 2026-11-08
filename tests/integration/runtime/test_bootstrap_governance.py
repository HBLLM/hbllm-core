from __future__ import annotations

import pytest

from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
from plugins.arc_agi_adapter.my_agent import MyAgent


def test_bootstrap_creates_unified_system_with_verified_governance():
    """Verify that build_cognitive_system produces a unified system with verified governance."""
    config = CognitiveSystemConfig(world_id="test_world", seed=123)
    system = build_cognitive_system(config)

    assert system.services is not None
    assert system.engine is not None
    assert system.bridge is not None
    assert system.workspace is not None
    assert system.tiered_workspace is not None

    # Verify state ownership & shared services
    assert system.engine.services is system.services
    assert system.bridge.services is system.services
    assert system.engine.grounding_bridge is system.bridge

    # Verify ConstitutionalVerifier is attached and TM is verified
    tm = system.services.transaction_manager
    assert tm.unverified is False
    assert any(isinstance(stage, ConstitutionalVerifier) for stage in tm._verification_stages)


def test_my_agent_arc_plugin_enforces_constitutional_governance():
    """Verify that MyAgent boots via the single composition root and rejects policy-violating commits."""
    agent = MyAgent(card_id="local_suite", game_id="nav_basic", seed=42)

    assert hasattr(agent, "services")
    assert agent.services is not None
    tm = agent.services.transaction_manager
    assert tm.unverified is False

    # Find the ConstitutionalVerifier stage
    verifier = next(
        stage for stage in tm._verification_stages if isinstance(stage, ConstitutionalVerifier)
    )
    # Configure a forbidden node type constraint
    verifier.forbidden_node_types.add("ForbiddenHazardNode")

    # Propose a policy-violating transaction
    invalid_tx = HCIRTransaction(
        author="malicious_or_faulty_stage",
        operations=[
            TransactionOperation(
                op=TransactionOp.ADD_NODE,
                node_id="illegal_node_999",
                node_data={"node_type": "ForbiddenHazardNode", "properties": {"hazard": True}},
            )
        ],
    )

    committed = tm.commit(invalid_tx)

    # Assert that the transaction was REJECTED by ConstitutionalVerifier
    assert committed.is_committed is False
    assert committed.is_rejected is True
    assert "ForbiddenHazardNode" in (committed.rejection_reason or "")

    # Assert that the illegal node was NOT written to the workspace graph
    assert not agent.services.workspace.graph.has_node("illegal_node_999")


def test_unverified_transaction_manager_rejected_when_enforced():
    """Verify that a TransactionManager without verification stages is flagged and rejected."""
    ws = HCIRWorkspaceState()
    bare_tm = TransactionManager(ws)

    # Must be flagged as unverified
    assert bare_tm.unverified is True

    # Initializing KernelServices with require_verified=True must raise AssertionError
    from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
    from hbllm.hcir.kernel.scheduler import KernelInstructionScheduler

    with pytest.raises(AssertionError, match="unverified TransactionManager"):
        KernelServices(
            workspace=ws,
            transaction_manager=bare_tm,
            capability_resolver=CapabilityResolver(),
            scheduler=KernelInstructionScheduler(),
            require_verified=True,
        )
