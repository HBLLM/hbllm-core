"""
Phase 10 — HBLLM Cognitive Migration to HCIR Kernel Integration Tests.

Verifies end-to-end cognitive OS execution across migration modes, capability health,
identity boundary enforcement, memory dual-write consistency, and 100% Cognitive Authority Metric.
"""

from __future__ import annotations

import pytest

from hbllm.hcir.adapters.memory_adapter import MemoryAdapter
from hbllm.hcir.adapters.memory_consistency import MemoryConsistencyChecker
from hbllm.hcir.graph import GoalNode
from hbllm.hcir.kernel.capability_health import CapabilityHealthRegistry
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.cognitive_kernel import CognitiveKernel
from hbllm.hcir.kernel.executive_runtime import ExecutiveRuntime
from hbllm.hcir.kernel.governance.governance_engine import GovernanceEngine
from hbllm.hcir.kernel.governance.policies.migration_policy import MigrationMode, MigrationPolicy
from hbllm.hcir.kernel.identity_bridge import IdentityBridge
from hbllm.hcir.kernel.migration_metrics import MigrationMetrics
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_envelope import (
    CognitiveAuthorityChain,
    KernelTransactionEnvelope,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.workspace import HCIRWorkspaceState


class TestPhase10Migration:
    """Integration test suite for Phase 10 HBLLM Cognitive Kernel Migration."""

    @pytest.mark.asyncio
    async def test_end_to_end_kernel_runtime_cycle(self):
        ws = HCIRWorkspaceState()
        ws.add_node(GoalNode(id="g_test", description="Test migration pipeline", priority=0.9))

        tx_mgr = TransactionManager(ws)
        resolver = CapabilityResolver()
        scheduler = CognitiveScheduler()

        services = KernelServices(
            workspace=ws,
            transaction_manager=tx_mgr,
            capability_resolver=resolver,
            scheduler=scheduler,
        )

        gov_policy = MigrationPolicy(MigrationMode.HYBRID)
        gov_engine = GovernanceEngine(migration_policy=gov_policy)
        kernel = CognitiveKernel(workspace=ws, governance_engine=gov_engine)

        runtime = ExecutiveRuntime(services=services, kernel=kernel)
        await runtime.start()

        assert runtime.is_running
        result = await runtime.run_cycle()

        assert result.cycle_index == 1
        assert result.elapsed_ms >= 0
        assert result.goal_evaluated == "g_test"

        await runtime.stop()
        assert not runtime.is_running

    def test_identity_bridge_and_transaction_envelope(self):
        exec_ctx = IdentityBridge.context_from_tenant(
            tenant_id="tenant_alpha",
            user_id="user_test",
            device_id="device_edge_1",
        )
        assert exec_ctx.tenant_scope.tenant_id == "tenant_alpha"
        assert exec_ctx.tenant_scope.user_id == "user_test"

        auth_chain = CognitiveAuthorityChain(
            requestor="user_test",
            tenant="tenant_alpha",
            executive="ExecutiveController",
            kernel="CognitiveKernel",
            capability="planning.generate_candidates",
            actuator="device_edge_1",
        )
        envelope = KernelTransactionEnvelope(
            execution_context=exec_ctx,
            authority_chain=auth_chain,
            capability_name="planning.generate_candidates",
            migration_mode=MigrationMode.HCIR,
        )
        assert envelope.migration_mode == MigrationMode.HCIR
        assert envelope.authority_chain.capability == "planning.generate_candidates"

    def test_memory_dual_write_and_consistency(self):
        ws = HCIRWorkspaceState()
        checker = MemoryConsistencyChecker()
        adapter = MemoryAdapter(consistency_checker=checker)

        ep_node = adapter.import_episode(
            workspace=ws,
            session_id="sess_101",
            summary="Tested solar dehydrator efficiency",
            outcome="Efficiency improved by 15%",
            reward=0.9,
            tenant_id="tenant_alpha",
        )

        assert ep_node.id == "ep_sess_101"
        assert checker.bridge.get_hcir_id("ep_sess_101") == "ep_sess_101"

    def test_capability_health_and_migration_metrics(self):
        health_reg = CapabilityHealthRegistry()
        health_reg.register_health("planning.generate_candidates", status="READY", latency_ms=12)
        health_reg.register_health("routing.resolve_route", status="READY", latency_ms=8)

        assert health_reg.is_all_ready(["planning.generate_candidates", "routing.resolve_route"])

        metrics = MigrationMetrics()
        metrics.record_execution(
            "planning.generate_candidates", MigrationMode.HCIR, "hcir", elapsed_ms=12
        )
        metrics.record_execution("routing.resolve_route", MigrationMode.HCIR, "hcir", elapsed_ms=8)

        assert metrics.get_cognitive_authority_metric() == 100.0
        assert metrics.get_fallback_count() == 0
