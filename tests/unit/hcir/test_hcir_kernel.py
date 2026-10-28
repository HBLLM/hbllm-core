"""Unit tests for Kernel Services: Scheduler, CapabilityResolver, Validation."""

import pytest

from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    FactNode,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    NodeLifecycle,
)
from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
)
from hbllm.hcir.kernel.event_bus import KernelEvent, KernelEventBus
from hbllm.hcir.kernel.governance.constitutional_verifier import ConstitutionalVerifier
from hbllm.hcir.kernel.scheduler import (
    CognitiveProcess,
    CognitiveThread,
    KernelInstructionScheduler,
    ProcessState,
)
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.kernel.verification import create_default_pipeline
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.types import Scope, SecurityLevel
from hbllm.hcir.validation import (
    GraphValidator,
    ValidationSeverity,
)
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# Graph Validator Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphValidator:
    def test_valid_graph(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        graph.add_node(BeliefNode(id="b1", claim="b"))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"])
        )
        validator = GraphValidator()
        report = validator.validate(graph)
        assert report.is_valid

    def test_cross_tenant_edge_detected(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a", scope=Scope(tenant_id="tenant_a")))
        graph.add_node(BeliefNode(id="b1", claim="b", scope=Scope(tenant_id="tenant_b")))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"])
        )
        validator = GraphValidator()
        report = validator.validate(graph)
        assert not report.is_valid
        assert report.error_count >= 1
        cross_tenant = [i for i in report.issues if i.code == "CROSS_TENANT_EDGE"]
        assert len(cross_tenant) == 1

    def test_system_scoped_nodes_bypass_tenant_check(self):
        graph = CognitiveGraph()
        graph.add_node(
            GoalNode(
                id="g1",
                description="a",
                scope=Scope(tenant_id="tenant_a", security_level=SecurityLevel.SYSTEM),
            )
        )
        graph.add_node(BeliefNode(id="b1", claim="b", scope=Scope(tenant_id="tenant_b")))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"])
        )
        validator = GraphValidator()
        report = validator.validate(graph)
        # g1 is system-scoped, so it can cross boundaries
        cross_tenant = [i for i in report.issues if i.code == "CROSS_TENANT_EDGE"]
        assert len(cross_tenant) == 0

    def test_terminal_node_active_edges_warning(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a", lifecycle=NodeLifecycle.ARCHIVED))
        graph.add_node(BeliefNode(id="b1", claim="b"))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g1"], targets=["b1"])
        )
        validator = GraphValidator()
        report = validator.validate(graph)
        warnings = [i for i in report.issues if i.severity == ValidationSeverity.WARNING]
        assert len(warnings) >= 1

    def test_orphan_node_detection(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="connected_goal"))
        graph.add_node(BeliefNode(id="b1", claim="connected_belief"))
        graph.add_node(FactNode(id="f_orphan", claim="standalone_fact"))
        graph.add_edge(
            HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"])
        )
        validator = GraphValidator()
        report_default = validator.validate(graph)
        assert len([i for i in report_default.issues if i.code == "ORPHAN_NODE"]) == 0

        report_with_orphans = validator.validate(graph, check_orphans=True)
        orphans = [i for i in report_with_orphans.issues if i.code == "ORPHAN_NODE"]
        assert len(orphans) == 1
        assert orphans[0].node_id == "f_orphan"
        assert orphans[0].severity == ValidationSeverity.INFO


# ═══════════════════════════════════════════════════════════════════════════
# Capability Resolver Tests
# ═══════════════════════════════════════════════════════════════════════════


class MockExecutor:
    """Mock executor for testing."""

    def __init__(self, available: bool = True) -> None:
        self._available = available
        self.last_params: dict | None = None

    async def execute(self, params: dict) -> dict:
        self.last_params = params
        return {"result": "mock_success"}

    @property
    def is_available(self) -> bool:
        return self._available


class TestCapabilityResolver:
    def test_register_and_has(self):
        resolver = CapabilityResolver()
        impl = CapabilityImplementation(
            capability_name="search",
            implementation_id="local_search",
            executor=MockExecutor(),
        )
        resolver.register(impl)
        assert resolver.has_capability("search")
        assert not resolver.has_capability("nonexistent")

    @pytest.mark.asyncio
    async def test_resolve_returns_available(self):
        resolver = CapabilityResolver()
        executor = MockExecutor(available=True)
        resolver.register(
            CapabilityImplementation(
                capability_name="search",
                implementation_id="local",
                executor=executor,
                priority=10,
            )
        )
        resolved = await resolver.resolve("search")
        assert resolved is executor

    @pytest.mark.asyncio
    async def test_resolve_skips_unavailable(self):
        resolver = CapabilityResolver()
        unavailable = MockExecutor(available=False)
        available = MockExecutor(available=True)
        resolver.register(
            CapabilityImplementation(
                capability_name="search",
                implementation_id="docker",
                executor=unavailable,
                priority=20,  # Higher priority but unavailable
            )
        )
        resolver.register(
            CapabilityImplementation(
                capability_name="search",
                implementation_id="local",
                executor=available,
                priority=10,
            )
        )
        resolved = await resolver.resolve("search")
        assert resolved is available

    @pytest.mark.asyncio
    async def test_resolve_none_available(self):
        resolver = CapabilityResolver()
        resolved = await resolver.resolve("nonexistent")
        assert resolved is None

    def test_list_capabilities(self):
        resolver = CapabilityResolver()
        resolver.register(
            CapabilityImplementation(
                capability_name="search",
                implementation_id="a",
                executor=MockExecutor(),
            )
        )
        resolver.register(
            CapabilityImplementation(
                capability_name="execute_python",
                implementation_id="b",
                executor=MockExecutor(),
            )
        )
        caps = resolver.list_capabilities()
        assert "search" in caps
        assert "execute_python" in caps

    def test_unregister(self):
        resolver = CapabilityResolver()
        resolver.register(
            CapabilityImplementation(
                capability_name="search",
                implementation_id="local",
                executor=MockExecutor(),
            )
        )
        assert resolver.unregister("search", "local") is True
        assert resolver.unregister("search", "local") is False


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Scheduler Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKernelInstructionScheduler:
    def test_register_process(self):
        sched = KernelInstructionScheduler()
        proc = CognitiveProcess(process_id="p1", conversation_id="c1")
        sched.register_process(proc)
        assert sched.get_process("p1") is proc
        assert sched.process_count == 1

    def test_enqueue_and_dispatch(self):
        sched = KernelInstructionScheduler()
        proc = CognitiveProcess(process_id="p1")
        thread = CognitiveThread(thread_id="t1")
        proc.add_thread(thread)
        sched.register_process(proc)
        sched.enqueue("p1", "t1", salience=0.9, activation=0.8)
        entry = sched.dispatch()
        assert entry is not None
        assert entry.process_id == "p1"
        assert entry.thread_id == "t1"

    def test_dispatch_priority_ordering(self):
        sched = KernelInstructionScheduler()
        for pid, sal in [("p1", 0.3), ("p2", 0.9), ("p3", 0.6)]:
            proc = CognitiveProcess(process_id=pid)
            proc.add_thread(CognitiveThread(thread_id=f"t_{pid}"))
            sched.register_process(proc)
            sched.enqueue(pid, f"t_{pid}", salience=sal, activation=1.0)

        first = sched.dispatch()
        assert first.process_id == "p2"  # Highest salience
        second = sched.dispatch()
        assert second.process_id == "p3"

    def test_dispatch_respects_max_concurrent(self):
        sched = KernelInstructionScheduler(max_concurrent=1)
        proc = CognitiveProcess(process_id="p1")
        proc.add_thread(CognitiveThread(thread_id="t1"))
        sched.register_process(proc)
        sched.enqueue("p1", "t1", salience=0.9)
        sched.dispatch()  # Fills the slot
        assert sched.dispatch() is None  # Blocked by concurrency limit

    def test_complete_frees_slot(self):
        sched = KernelInstructionScheduler(max_concurrent=1)
        proc = CognitiveProcess(process_id="p1")
        proc.add_thread(CognitiveThread(thread_id="t1"))
        proc.add_thread(CognitiveThread(thread_id="t2"))
        sched.register_process(proc)
        sched.enqueue("p1", "t1", salience=0.9)
        sched.enqueue("p1", "t2", salience=0.5)
        sched.dispatch()
        sched.complete("p1", "t1")
        entry = sched.dispatch()
        assert entry is not None
        assert entry.thread_id == "t2"

    def test_thread_state_transitions(self):
        sched = KernelInstructionScheduler()
        proc = CognitiveProcess(process_id="p1")
        thread = CognitiveThread(thread_id="t1")
        proc.add_thread(thread)
        sched.register_process(proc)
        sched.enqueue("p1", "t1", salience=0.9)
        sched.dispatch()
        assert thread.state == ProcessState.RUNNING
        sched.complete("p1", "t1")
        assert thread.state == ProcessState.COMPLETED

    def test_process_completes_when_all_threads_done(self):
        sched = KernelInstructionScheduler()
        proc = CognitiveProcess(process_id="p1")
        proc.add_thread(CognitiveThread(thread_id="t1"))
        sched.register_process(proc)
        sched.enqueue("p1", "t1", salience=0.9)
        sched.dispatch()
        sched.complete("p1", "t1")
        assert proc.state == ProcessState.COMPLETED


# ═══════════════════════════════════════════════════════════════════════════
# Architectural Consolidation & Advanced Governance Tests
# ═══════════════════════════════════════════════════════════════════════════


class MockCapabilityExecutor:
    def __init__(self, available: bool = True, result: dict | None = None):
        self._available = available
        self._result = result or {"status": "ok"}

    @property
    def is_available(self) -> bool:
        return self._available

    async def execute(self, params):
        return self._result


class TestKernelArchitecturalConsolidation:
    @pytest.mark.asyncio
    async def test_transaction_manager_begin_transaction_context(self):
        ws = HCIRWorkspaceState()
        tx_mgr = TransactionManager(ws)

        new_node = GoalNode(id="g_test_begin", description="Test begin_transaction context")
        async with tx_mgr.begin_transaction() as tx:
            tx.add_operation(
                TransactionOperation(
                    op=TransactionOp.INSERT_NODE,
                    node=new_node,
                )
            )
            result = await tx.commit()

        assert result.is_committed
        assert ws.graph.has_node("g_test_begin")

    @pytest.mark.asyncio
    async def test_transaction_manager_acid_rollback_before_image(self):
        ws = HCIRWorkspaceState()
        tx_mgr = TransactionManager(ws)

        initial_node = GoalNode(id="g_initial", description="Original description")
        async with tx_mgr.begin_transaction() as tx:
            tx.add_operation(
                TransactionOperation(
                    op=TransactionOp.INSERT_NODE,
                    node=initial_node,
                )
            )
            await tx.commit()

        # Now mutate the node
        async with tx_mgr.begin_transaction() as tx:
            tx.add_operation(
                TransactionOperation(
                    op=TransactionOp.MODIFY_NODE,
                    node_id="g_initial",
                    node_data={"description": "Mutated description"},
                )
            )
            mutate_res = await tx.commit()

        assert ws.graph.get_node("g_initial").description == "Mutated description"

        # ACID Rollback should restore before-image
        rollback_res = tx_mgr.rollback(mutate_res.id, reason="test_recovery")
        assert rollback_res.is_committed
        assert ws.graph.get_node("g_initial").description == "Original description"

    @pytest.mark.asyncio
    async def test_event_bus_wait_for(self):
        bus = KernelEventBus()

        async def trigger_event():
            import asyncio

            await asyncio.sleep(0.01)
            bus.publish(KernelEvent(event_type="perception.stabilized", payload={"clarity": 0.99}))

        import asyncio

        asyncio.create_task(trigger_event())
        evt = await bus.wait_for("perception.stabilized", timeout_ms=500)
        assert evt.event_type == "perception.stabilized"
        assert evt.payload.get("clarity") == 0.99

    def test_default_verification_pipeline_includes_constitutional_verifier(self):
        pipeline = create_default_pipeline()
        verifier_types = [type(v) for v in pipeline]
        assert ConstitutionalVerifier in verifier_types

    def test_capability_resolver_find_best_and_health_integration(self):
        from hbllm.hcir.kernel.capability_health import CapabilityHealth, CapabilityHealthRegistry
        from hbllm.hcir.kernel.capability_resolver import (
            CapabilityImplementation,
            CapabilityResolver,
            PerformanceProfile,
        )

        health_reg = CapabilityHealthRegistry()
        resolver = CapabilityResolver(health_registry=health_reg)

        impl_fast = CapabilityImplementation(
            capability_name="translate",
            implementation_id="fast_model",
            priority=10,
            executor=MockCapabilityExecutor(available=True),
            performance=PerformanceProfile(avg_latency_ms=10, cost_per_call=0.05, accuracy=0.8),
        )
        impl_cheap = CapabilityImplementation(
            capability_name="translate",
            implementation_id="cheap_model",
            priority=5,
            executor=MockCapabilityExecutor(available=True),
            performance=PerformanceProfile(avg_latency_ms=50, cost_per_call=0.01, accuracy=0.85),
        )

        resolver.register(impl_fast)
        resolver.register(impl_cheap)

        best_fast = resolver.find_best("translate", strategy="fastest")
        assert best_fast.implementation_id == "fast_model"

        best_cheap = resolver.find_best("translate", strategy="cheapest")
        assert best_cheap.implementation_id == "cheap_model"

        # Marking translate UNAVAILABLE in health registry should filter it
        health_reg.update_health(
            CapabilityHealth(capability_name="translate", status="UNAVAILABLE")
        )
        assert resolver.find_best("translate") is None
        assert resolver.resolve_implementation("translate") is None
