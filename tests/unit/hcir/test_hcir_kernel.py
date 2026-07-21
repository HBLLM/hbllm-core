"""Unit tests for Kernel Services: Scheduler, CapabilityResolver, Validation."""

import pytest

from hbllm.hcir.graph import (
    BeliefNode,
    CognitiveGraph,
    GoalNode,
    HCIREdge,
    HCIREdgeType,
    NodeLifecycle,
)
from hbllm.hcir.kernel.capability_resolver import (
    CapabilityImplementation,
    CapabilityResolver,
    ICapabilityExecutor,
)
from hbllm.hcir.kernel.scheduler import (
    CognitiveProcess,
    CognitiveScheduler,
    CognitiveThread,
    ProcessState,
)
from hbllm.hcir.types import Scope, SecurityLevel
from hbllm.hcir.validation import (
    GraphValidator,
    ValidationSeverity,
)


# ═══════════════════════════════════════════════════════════════════════════
# Graph Validator Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGraphValidator:
    def test_valid_graph(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a"))
        graph.add_node(BeliefNode(id="b1", claim="b"))
        graph.add_edge(HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"]))
        validator = GraphValidator()
        report = validator.validate(graph)
        assert report.is_valid

    def test_cross_tenant_edge_detected(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a", scope=Scope(tenant_id="tenant_a")))
        graph.add_node(BeliefNode(id="b1", claim="b", scope=Scope(tenant_id="tenant_b")))
        graph.add_edge(HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"]))
        validator = GraphValidator()
        report = validator.validate(graph)
        assert not report.is_valid
        assert report.error_count >= 1
        cross_tenant = [i for i in report.issues if i.code == "CROSS_TENANT_EDGE"]
        assert len(cross_tenant) == 1

    def test_system_scoped_nodes_bypass_tenant_check(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(
            id="g1", description="a",
            scope=Scope(tenant_id="tenant_a", security_level=SecurityLevel.SYSTEM),
        ))
        graph.add_node(BeliefNode(id="b1", claim="b", scope=Scope(tenant_id="tenant_b")))
        graph.add_edge(HCIREdge(id="e1", edge_type=HCIREdgeType.SUPPORTS, sources=["b1"], targets=["g1"]))
        validator = GraphValidator()
        report = validator.validate(graph)
        # g1 is system-scoped, so it can cross boundaries
        cross_tenant = [i for i in report.issues if i.code == "CROSS_TENANT_EDGE"]
        assert len(cross_tenant) == 0

    def test_terminal_node_active_edges_warning(self):
        graph = CognitiveGraph()
        graph.add_node(GoalNode(id="g1", description="a", lifecycle=NodeLifecycle.ARCHIVED))
        graph.add_node(BeliefNode(id="b1", claim="b"))
        graph.add_edge(HCIREdge(id="e1", edge_type=HCIREdgeType.DEPENDS_ON, sources=["g1"], targets=["b1"]))
        validator = GraphValidator()
        report = validator.validate(graph)
        warnings = [i for i in report.issues if i.severity == ValidationSeverity.WARNING]
        assert len(warnings) >= 1


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
        resolver.register(CapabilityImplementation(
            capability_name="search",
            implementation_id="local",
            executor=executor,
            priority=10,
        ))
        resolved = await resolver.resolve("search")
        assert resolved is executor

    @pytest.mark.asyncio
    async def test_resolve_skips_unavailable(self):
        resolver = CapabilityResolver()
        unavailable = MockExecutor(available=False)
        available = MockExecutor(available=True)
        resolver.register(CapabilityImplementation(
            capability_name="search",
            implementation_id="docker",
            executor=unavailable,
            priority=20,  # Higher priority but unavailable
        ))
        resolver.register(CapabilityImplementation(
            capability_name="search",
            implementation_id="local",
            executor=available,
            priority=10,
        ))
        resolved = await resolver.resolve("search")
        assert resolved is available

    @pytest.mark.asyncio
    async def test_resolve_none_available(self):
        resolver = CapabilityResolver()
        resolved = await resolver.resolve("nonexistent")
        assert resolved is None

    def test_list_capabilities(self):
        resolver = CapabilityResolver()
        resolver.register(CapabilityImplementation(
            capability_name="search", implementation_id="a", executor=MockExecutor(),
        ))
        resolver.register(CapabilityImplementation(
            capability_name="execute_python", implementation_id="b", executor=MockExecutor(),
        ))
        caps = resolver.list_capabilities()
        assert "search" in caps
        assert "execute_python" in caps

    def test_unregister(self):
        resolver = CapabilityResolver()
        resolver.register(CapabilityImplementation(
            capability_name="search", implementation_id="local", executor=MockExecutor(),
        ))
        assert resolver.unregister("search", "local") is True
        assert resolver.unregister("search", "local") is False


# ═══════════════════════════════════════════════════════════════════════════
# Cognitive Scheduler Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCognitiveScheduler:
    def test_register_process(self):
        sched = CognitiveScheduler()
        proc = CognitiveProcess(process_id="p1", conversation_id="c1")
        sched.register_process(proc)
        assert sched.get_process("p1") is proc
        assert sched.process_count == 1

    def test_enqueue_and_dispatch(self):
        sched = CognitiveScheduler()
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
        sched = CognitiveScheduler()
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
        sched = CognitiveScheduler(max_concurrent=1)
        proc = CognitiveProcess(process_id="p1")
        proc.add_thread(CognitiveThread(thread_id="t1"))
        sched.register_process(proc)
        sched.enqueue("p1", "t1", salience=0.9)
        sched.dispatch()  # Fills the slot
        assert sched.dispatch() is None  # Blocked by concurrency limit

    def test_complete_frees_slot(self):
        sched = CognitiveScheduler(max_concurrent=1)
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
        sched = CognitiveScheduler()
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
        sched = CognitiveScheduler()
        proc = CognitiveProcess(process_id="p1")
        proc.add_thread(CognitiveThread(thread_id="t1"))
        sched.register_process(proc)
        sched.enqueue("p1", "t1", salience=0.9)
        sched.dispatch()
        sched.complete("p1", "t1")
        assert proc.state == ProcessState.COMPLETED
