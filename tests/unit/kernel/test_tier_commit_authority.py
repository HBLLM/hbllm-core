from __future__ import annotations

from hbllm.hcir.kernel.commit_authority import CommitAuthorityRegistry
from hbllm.hcir.kernel.governance.constitutional_verifier import (
    ConstitutionalVerifier,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.tier_types import WorkspaceTier
from hbllm.hcir.transactions import HCIRTransaction, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState


def test_tier_commit_authority_working_open() -> None:
    verifier = ConstitutionalVerifier(authority_registry=CommitAuthorityRegistry())
    ws = HCIRWorkspaceState(tier=WorkspaceTier.WORKING)

    tx = HCIRTransaction(author="random_untrusted_author", operations=[])
    assert verifier.verify(tx, ws) is True


def test_tier_commit_authority_persistent_rejection() -> None:
    registry = CommitAuthorityRegistry()
    verifier = ConstitutionalVerifier(authority_registry=registry)
    ws = HCIRWorkspaceState(tier=WorkspaceTier.PERSISTENT)

    # No grant
    tx_unauth = HCIRTransaction(author="random_untrusted_author", operations=[])
    assert verifier.verify(tx_unauth, ws) is False
    assert any("Commit authority violation" in a.assertion for a in tx_unauth.annotations)

    # Valid grant
    grant = registry.issue("cortex_episodic", {WorkspaceTier.PERSISTENT})
    tx_auth = HCIRTransaction(
        author="cortex_episodic",
        operations=[],
        metadata={"commit_grant": grant.grant_id},
    )
    assert verifier.verify(tx_auth, ws) is True


def test_tier_commit_authority_brain_tier() -> None:
    registry = CommitAuthorityRegistry()
    verifier = ConstitutionalVerifier(authority_registry=registry)
    ws = HCIRWorkspaceState(tier=WorkspaceTier.BRAIN)

    tx_unauth = HCIRTransaction(author="external_sensor", operations=[])
    assert verifier.verify(tx_unauth, ws) is False

    grant = registry.issue("cortex_deliberation", {WorkspaceTier.BRAIN})
    tx_auth = HCIRTransaction(
        author="cortex_deliberation",
        operations=[],
        metadata={"commit_grant": grant.grant_id},
    )
    assert verifier.verify(tx_auth, ws) is True


def test_tier_commit_authority_wrong_tier_grant_rejected() -> None:
    registry = CommitAuthorityRegistry()
    verifier = ConstitutionalVerifier(authority_registry=registry)
    ws = HCIRWorkspaceState(tier=WorkspaceTier.PERSISTENT)

    grant = registry.issue("worker", {WorkspaceTier.BRAIN})  # BRAIN-only grant
    tx = HCIRTransaction(author="worker", operations=[], metadata={"commit_grant": grant.grant_id})
    assert verifier.verify(tx, ws) is False


def test_tier_commit_authority_no_registry_backcompat() -> None:
    """Without a configured registry, non-WORKING tiers stay open (back-compat)."""
    verifier = ConstitutionalVerifier()
    ws = HCIRWorkspaceState(tier=WorkspaceTier.PERSISTENT)
    tx = HCIRTransaction(author="anyone", operations=[])
    assert verifier.verify(tx, ws) is True


def test_transaction_manager_rejects_unauthorized_tier_commit() -> None:
    registry = CommitAuthorityRegistry()
    ws = HCIRWorkspaceState(tier=WorkspaceTier.PERSISTENT)

    tm = TransactionManager(workspace=ws)
    tm.add_verification_stage(ConstitutionalVerifier(authority_registry=registry))

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
    assert not ws._graph.has_node("malicious_node")
