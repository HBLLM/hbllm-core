from __future__ import annotations

from hbllm.hcir.kernel.governance.constitutional_verifier import (
    ConstitutionalVerifier,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import WorkspaceTier


def test_tier_commit_authority_working_open() -> None:
    verifier = ConstitutionalVerifier()
    ws = HCIRWorkspaceState()
    ws.tier = WorkspaceTier.WORKING

    tx = HCIRTransaction(author="random_untrusted_author", operations=[])
    assert verifier.verify(tx, ws) is True


def test_tier_commit_authority_persistent_rejection() -> None:
    verifier = ConstitutionalVerifier()
    ws = HCIRWorkspaceState()
    ws.tier = WorkspaceTier.PERSISTENT

    # Unauthorized author
    tx_unauth = HCIRTransaction(author="random_untrusted_author", operations=[])
    assert verifier.verify(tx_unauth, ws) is False
    assert any("Commit authority violation" in a.assertion for a in tx_unauth.annotations)

    # Authorized author
    tx_auth = HCIRTransaction(author="cortex_episodic", operations=[])
    assert verifier.verify(tx_auth, ws) is True


def test_tier_commit_authority_brain_tier() -> None:
    verifier = ConstitutionalVerifier()
    ws = HCIRWorkspaceState()
    ws.tier = WorkspaceTier.BRAIN

    tx_unauth = HCIRTransaction(author="external_sensor", operations=[])
    assert verifier.verify(tx_unauth, ws) is False

    tx_auth = HCIRTransaction(author="cortex_deliberation", operations=[])
    assert verifier.verify(tx_auth, ws) is True


def test_transaction_manager_rejects_unauthorized_tier_commit() -> None:
    ws = HCIRWorkspaceState()
    ws.tier = WorkspaceTier.PERSISTENT

    tm = TransactionManager(workspace=ws)
    tm.add_verification_stage(ConstitutionalVerifier())

    tx = HCIRTransaction(
        author="intruder",
        operations=[
            TransactionOperation(
                op=TransactionOp.UPSERT_NODE,
                node_id="malicious_node",
                node_data={"hacked": True},
            )
        ],
    )

    res_tx = tm.commit(tx)
    from hbllm.hcir.transactions import TransactionStatus

    assert res_tx.status == TransactionStatus.REJECTED
    # Ensure transaction was not applied to graph
    assert not ws._graph.has_node("malicious_node")
