"""Tests verifying structural tenant isolation inside HCIR workspaces and ConstitutionalVerifier."""

from __future__ import annotations

from hbllm.hcir.graph import CognitiveCategory, HCIRNode, HCIRNodeType, Scope
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.transactions import HCIRTransaction, Provenance, TransactionOp, TransactionOperation
from hbllm.hcir.workspace import HCIRWorkspaceState
from hbllm.hcir.workspace_tiers import TieredWorkspace
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system


def test_tiered_workspace_and_forks_preserve_tenant_id() -> None:
    tw = TieredWorkspace(tenant_id="tenant_a")
    assert tw.tenant_id == "tenant_a"
    assert tw.working_state.tenant_id == "tenant_a"
    assert tw.brain.tenant_id == "tenant_a"
    assert tw.persistent.tenant_id == "tenant_a"
    assert tw.meta.tenant_id == "tenant_a"
    assert tw.working_state.fork("sim_a").tenant_id == "tenant_a"


def test_constitutional_verifier_blocks_cross_tenant_default_and_unscoped_writes() -> None:
    verifier = ConstitutionalVerifier(enforce_scope_isolation=True)
    ws = HCIRWorkspaceState(tenant_id="tenant_a")
    node = HCIRNode(
        id="n1",
        node_type=HCIRNodeType.OBSERVATION,
        category=CognitiveCategory.PERCEPTION,
        scope=Scope(tenant_id="tenant_a"),
    )

    # 1. Cross-tenant tx rejected
    tx_other = HCIRTransaction(
        author="agent_b",
        metadata={"tenant_id": "tenant_b"},
        operations=[TransactionOperation(op=TransactionOp.ADD_NODE, node_data=node.model_dump())],
    )
    assert verifier.verify(tx_other, ws) is False

    # 2. "default" tenant fallback rejected against tenant_a workspace
    tx_default = HCIRTransaction(
        author="agent_default",
        metadata={"tenant_id": "default"},
        operations=[TransactionOperation(op=TransactionOp.ADD_NODE, node_data=node.model_dump())],
    )
    assert verifier.verify(tx_default, ws) is False

    # 3. Unscoped non-system tx rejected against tenant_a workspace
    tx_unscoped = HCIRTransaction(
        author="external_plugin",
        operations=[TransactionOperation(op=TransactionOp.ADD_NODE, node_data=node.model_dump())],
    )
    assert verifier.verify(tx_unscoped, ws) is False

    # 4. Matching tenant_a tx succeeds
    tx_valid = HCIRTransaction(
        author="agent_a",
        provenance=Provenance(session_id="sess_1"),
        metadata={"tenant_id": "tenant_a"},
        operations=[TransactionOperation(op=TransactionOp.ADD_NODE, node_data=node.model_dump())],
    )
    assert verifier.verify(tx_valid, ws) is True


def test_bootstrap_propagates_tenant_id_to_workspaces() -> None:
    system = build_cognitive_system(
        CognitiveSystemConfig(
            world_id="tenant_iso_world",
            memory_dir=None,
            tenant_id="tenant_alpha",
            enable_logging=False,
        )
    )
    assert system.tiered_workspace.tenant_id == "tenant_alpha"
    assert system.workspace.tenant_id == "tenant_alpha"
    assert system.tiered_workspace.persistent.tenant_id == "tenant_alpha"
