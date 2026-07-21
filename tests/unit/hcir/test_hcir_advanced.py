"""Unit tests for SQLite Event Store, Compiler, Stdlib, Event Bus, and Capability Registry."""

import time

import pytest

from hbllm.hcir.compiler import HCIRCompiler, IntentType, SemanticAST, SemanticSlot
from hbllm.hcir.kernel.capability_registry import (
    CapabilityRegistry,
    CapabilitySpec,
    PerformanceProfile,
)
from hbllm.hcir.kernel.capability_resolver import ICapabilityExecutor
from hbllm.hcir.kernel.event_bus import KernelEvent, KernelEventBus
from hbllm.hcir.stdlib import stdlib
from hbllm.hcir.stores import EventType, GraphEvent
from hbllm.hcir.stores.sqlite_event_store import SQLiteEventStore

# ═══════════════════════════════════════════════════════════════════════════
# SQLite Event Store Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSQLiteEventStore:
    @pytest.fixture
    def store(self, tmp_path):
        db = tmp_path / "test_events.db"
        s = SQLiteEventStore(db)
        yield s
        s.close()

    def test_append_and_retrieve(self, store):
        event = GraphEvent(
            sequence=1,
            event_type=EventType.NODE_ADDED,
            timestamp=time.time(),
            author="test",
            data={"node_id": "g1"},
        )
        store.append(event)
        events = store.get_events()
        assert len(events) == 1
        assert events[0].event_type == EventType.NODE_ADDED
        assert events[0].data["node_id"] == "g1"

    def test_sequence_order(self, store):
        for i in range(1, 6):
            store.append(
                GraphEvent(
                    sequence=i,
                    event_type=EventType.NODE_ADDED,
                    timestamp=time.time(),
                    author="test",
                    data={"seq": i},
                )
            )
        events = store.get_events(from_sequence=3)
        assert len(events) == 3
        assert events[0].data["seq"] == 3

    def test_filter_by_event_type(self, store):
        store.append(
            GraphEvent(
                sequence=1,
                event_type=EventType.NODE_ADDED,
                timestamp=time.time(),
                author="t",
            )
        )
        store.append(
            GraphEvent(
                sequence=2,
                event_type=EventType.EDGE_ADDED,
                timestamp=time.time(),
                author="t",
            )
        )
        events = store.get_events(event_types=[EventType.EDGE_ADDED])
        assert len(events) == 1

    def test_hash_chain_integrity(self, store):
        for i in range(1, 11):
            store.append(
                GraphEvent(
                    sequence=i,
                    event_type=EventType.NODE_ADDED,
                    timestamp=time.time(),
                    author="test",
                    data={"i": i},
                )
            )
        assert store.verify_chain_integrity() is True

    def test_latest_sequence(self, store):
        assert store.latest_sequence() == 0
        store.append(
            GraphEvent(
                sequence=42,
                event_type=EventType.NODE_ADDED,
                timestamp=time.time(),
                author="t",
            )
        )
        assert store.latest_sequence() == 42

    def test_event_count(self, store):
        assert store.event_count == 0
        for i in range(1, 4):
            store.append(
                GraphEvent(
                    sequence=i,
                    event_type=EventType.NODE_ADDED,
                    timestamp=time.time(),
                    author="t",
                )
            )
        assert store.event_count == 3

    def test_clear(self, store):
        store.append(
            GraphEvent(
                sequence=1,
                event_type=EventType.NODE_ADDED,
                timestamp=time.time(),
                author="t",
            )
        )
        store.clear()
        assert store.event_count == 0
        assert store.latest_sequence() == 0

    def test_snapshot_bookmark(self, store):
        for i in range(1, 4):
            store.append(
                GraphEvent(
                    sequence=i,
                    event_type=EventType.NODE_ADDED,
                    timestamp=time.time(),
                    author="t",
                )
            )
        store.save_snapshot(version=1, node_count=3)
        assert store.get_latest_snapshot_version() == 1

    def test_node_index(self, store):
        store.index_node("g1", "goal", tenant_id="acme")
        store.index_node("g2", "belief", tenant_id="acme")
        store.remove_node_index("g1")
        # No assertion on count — just verifying no crash

    def test_persistence_across_reopen(self, tmp_path):
        db = tmp_path / "persist.db"
        s1 = SQLiteEventStore(db)
        s1.append(
            GraphEvent(
                sequence=1,
                event_type=EventType.NODE_ADDED,
                timestamp=time.time(),
                author="t",
                data={"persist": True},
            )
        )
        s1.close()

        s2 = SQLiteEventStore(db)
        events = s2.get_events()
        assert len(events) == 1
        assert events[0].data["persist"] is True
        s2.close()


# ═══════════════════════════════════════════════════════════════════════════
# Compiler Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHCIRCompiler:
    def test_compile_investigate(self):
        compiler = HCIRCompiler()
        ast = SemanticAST(
            intent=IntentType.INVESTIGATE,
            subject="battery_temperature",
            action="find_cause",
        )
        stream = compiler.compile(ast)
        assert stream.length == 4  # ASSERT + QUERY + QUERY + EXECUTE
        assert stream.instructions[0].opcode.value == "ASSERT"
        assert stream.instructions[3].opcode.value == "EXECUTE"

    def test_compile_query(self):
        compiler = HCIRCompiler()
        ast = SemanticAST(
            intent=IntentType.QUERY,
            subject="solar panels",
        )
        stream = compiler.compile(ast)
        assert stream.length == 1
        assert stream.instructions[0].opcode.value == "QUERY"

    def test_compile_simulate(self):
        compiler = HCIRCompiler()
        ast = SemanticAST(
            intent=IntentType.SIMULATE,
            subject="what if copper tubing",
        )
        stream = compiler.compile(ast)
        assert stream.length == 3  # FORK + EXECUTE + MERGE
        assert stream.instructions[0].opcode.value == "FORK"
        assert stream.instructions[2].opcode.value == "MERGE"

    def test_compile_all_intents(self):
        compiler = HCIRCompiler()
        for intent in IntentType:
            ast = SemanticAST(intent=intent, subject="test")
            stream = compiler.compile(ast)
            assert stream.length > 0, f"Intent {intent} produced empty stream"

    def test_semantic_slots(self):
        ast = SemanticAST(
            intent=IntentType.QUERY,
            slots=[
                SemanticSlot(name="node_type", value="belief"),
                SemanticSlot(name="limit", value=10),
            ],
        )
        assert ast.get_slot("node_type") == "belief"
        assert ast.get_slot("missing") is None

    def test_cost_estimates(self):
        compiler = HCIRCompiler()
        ast = SemanticAST(intent=IntentType.INVESTIGATE, subject="test")
        stream = compiler.compile(ast)
        total_cost = sum(i.cost_estimate for i in stream.instructions)
        assert total_cost > 0


# ═══════════════════════════════════════════════════════════════════════════
# Standard Library Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStdlib:
    def test_memory_search(self):
        stream = stdlib.memory.search("battery temperature")
        assert stream.length == 1
        assert stream.instructions[0].opcode.value == "QUERY"
        assert stream.instructions[0].params["text_contains"] == "battery temperature"

    def test_memory_store(self):
        stream = stdlib.memory.store({"id": "n1", "node_type": "belief"})
        assert stream.length == 1
        assert stream.instructions[0].opcode.value == "ASSERT"

    def test_memory_forget(self):
        stream = stdlib.memory.forget("n1")
        assert stream.length == 1
        assert stream.instructions[0].opcode.value == "RETRACT"

    def test_reasoning_compare(self):
        stream = stdlib.reasoning.compare("Python", "Rust")
        assert stream.length == 3  # QUERY + QUERY + EXECUTE
        assert stream.instructions[2].params["capability"] == "reasoning.compare"

    def test_reasoning_evaluate(self):
        stream = stdlib.reasoning.evaluate("hypothesis_42", criteria=["accuracy"])
        assert stream.length == 3

    def test_planning_decompose(self):
        stream = stdlib.planning.decompose("Build solar dehydrator")
        assert stream.length == 3  # ASSERT + QUERY + EXECUTE
        assert stream.instructions[0].opcode.value == "ASSERT"

    def test_planning_prioritize(self):
        stream = stdlib.planning.prioritize(["g1", "g2", "g3"])
        assert stream.length == 2

    def test_learning_extract_skill(self):
        stream = stdlib.learning.extract_skill("ep_session_42")
        assert stream.length == 2

    def test_learning_consolidate(self):
        stream = stdlib.learning.consolidate(memory_type="episodic")
        assert stream.length == 2

    def test_simulation_hypothesize(self):
        stream = stdlib.simulation.hypothesize("Copper is more efficient")
        assert stream.length == 4  # FORK + ASSERT + EXECUTE + MERGE
        assert stream.instructions[0].opcode.value == "FORK"
        assert stream.instructions[3].opcode.value == "MERGE"

    def test_simulation_counterfactual(self):
        stream = stdlib.simulation.counterfactual("What if we used aluminum")
        assert stream.length == 3  # FORK + EXECUTE + ROLLBACK
        assert stream.instructions[2].opcode.value == "ROLLBACK"


# ═══════════════════════════════════════════════════════════════════════════
# Kernel Event Bus Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestKernelEventBus:
    def test_subscribe_and_publish(self):
        bus = KernelEventBus()
        received = []
        bus.subscribe("transaction.committed", lambda e: received.append(e))
        bus.publish(
            KernelEvent(
                event_type="transaction.committed",
                source="tx_mgr",
            )
        )
        assert len(received) == 1

    def test_wildcard_subscription(self):
        bus = KernelEventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e))
        bus.publish(KernelEvent(event_type="anything", source="test"))
        bus.publish(KernelEvent(event_type="something.else", source="test"))
        assert len(received) == 2

    def test_prefix_subscription(self):
        bus = KernelEventBus()
        received = []
        bus.subscribe("transaction.*", lambda e: received.append(e))
        bus.publish(KernelEvent(event_type="transaction.committed", source="test"))
        bus.publish(KernelEvent(event_type="transaction.rejected", source="test"))
        bus.publish(KernelEvent(event_type="capability.bound", source="test"))
        assert len(received) == 2

    def test_no_duplicate_delivery(self):
        bus = KernelEventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe("transaction.committed", handler)
        bus.subscribe("transaction.*", handler)
        bus.publish(KernelEvent(event_type="transaction.committed", source="test"))
        assert len(received) == 1  # Same handler, delivered once

    def test_handler_exception_doesnt_crash(self):
        bus = KernelEventBus()

        def bad_handler(e):
            raise RuntimeError("boom")

        bus.subscribe("test", bad_handler)
        bus.publish(KernelEvent(event_type="test", source="test"))
        # Should not raise

    def test_history(self):
        bus = KernelEventBus(history_size=5)
        for i in range(10):
            bus.publish(KernelEvent(event_type=f"e{i}", source="test"))
        history = bus.get_history()
        assert len(history) == 5
        assert history[0].event_type == "e5"

    def test_unsubscribe(self):
        bus = KernelEventBus()
        received = []
        handler = lambda e: received.append(e)
        bus.subscribe("test", handler)
        bus.publish(KernelEvent(event_type="test", source="t"))
        bus.unsubscribe("test", handler)
        bus.publish(KernelEvent(event_type="test", source="t"))
        assert len(received) == 1

    def test_stats(self):
        bus = KernelEventBus()
        bus.subscribe("a", lambda e: None)
        bus.subscribe("b", lambda e: None)
        stats = bus.stats()
        assert stats["subscriptions"]["a"] == 1

    def test_is_error(self):
        e1 = KernelEvent(event_type="capability.failed", source="t")
        e2 = KernelEvent(event_type="transaction.committed", source="t")
        assert e1.is_error is True
        assert e2.is_error is False


# ═══════════════════════════════════════════════════════════════════════════
# Capability Registry Tests
# ═══════════════════════════════════════════════════════════════════════════


class _MockExecutor(ICapabilityExecutor):
    def __init__(self, available: bool = True):
        self._available = available

    @property
    def is_available(self) -> bool:
        return self._available

    async def execute(self, params):
        return {"status": "ok"}


class TestCapabilityRegistry:
    def test_register_and_list(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="image_understanding",
                provider_id="qwen-vl",
                executor=_MockExecutor(),
                performance=PerformanceProfile(cost_per_call=0.03, accuracy=0.92),
            )
        )
        assert reg.has_capability("image_understanding")
        assert "image_understanding" in reg.list_capabilities()

    def test_find_best_by_priority(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="llm",
                provider_id="gpt-4",
                executor=_MockExecutor(),
                priority=10,
                performance=PerformanceProfile(cost_per_call=0.10),
            )
        )
        reg.register(
            CapabilitySpec(
                capability_name="llm",
                provider_id="qwen-3",
                executor=_MockExecutor(),
                priority=20,
                performance=PerformanceProfile(cost_per_call=0.01),
            )
        )
        best = reg.find_best("llm", strategy="priority")
        assert best is not None
        assert best.provider_id == "qwen-3"

    def test_find_cheapest(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="embed",
                provider_id="openai",
                executor=_MockExecutor(),
                performance=PerformanceProfile(cost_per_call=0.05),
            )
        )
        reg.register(
            CapabilitySpec(
                capability_name="embed",
                provider_id="local",
                executor=_MockExecutor(),
                performance=PerformanceProfile(cost_per_call=0.00),
            )
        )
        best = reg.find_best("embed", strategy="cheapest")
        assert best is not None
        assert best.provider_id == "local"

    def test_find_fastest(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="ocr",
                provider_id="cloud",
                executor=_MockExecutor(),
                performance=PerformanceProfile(avg_latency_ms=2000),
            )
        )
        reg.register(
            CapabilitySpec(
                capability_name="ocr",
                provider_id="local",
                executor=_MockExecutor(),
                performance=PerformanceProfile(avg_latency_ms=100),
            )
        )
        best = reg.find_best("ocr", strategy="fastest")
        assert best is not None
        assert best.provider_id == "local"

    def test_find_most_accurate(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="classify",
                provider_id="small",
                executor=_MockExecutor(),
                performance=PerformanceProfile(accuracy=0.7),
            )
        )
        reg.register(
            CapabilitySpec(
                capability_name="classify",
                provider_id="large",
                executor=_MockExecutor(),
                performance=PerformanceProfile(accuracy=0.95),
            )
        )
        best = reg.find_best("classify", strategy="accurate")
        assert best is not None
        assert best.provider_id == "large"

    def test_filter_by_cost(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="gen",
                provider_id="expensive",
                executor=_MockExecutor(),
                performance=PerformanceProfile(cost_per_call=1.0),
            )
        )
        reg.register(
            CapabilitySpec(
                capability_name="gen",
                provider_id="cheap",
                executor=_MockExecutor(),
                performance=PerformanceProfile(cost_per_call=0.01),
            )
        )
        best = reg.find_best("gen", max_cost=0.05)
        assert best is not None
        assert best.provider_id == "cheap"

    def test_unavailable_provider_excluded(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="gpu_task",
                provider_id="offline",
                executor=_MockExecutor(available=False),
                priority=100,
            )
        )
        reg.register(
            CapabilitySpec(
                capability_name="gpu_task",
                provider_id="online",
                executor=_MockExecutor(available=True),
                priority=1,
            )
        )
        best = reg.find_best("gpu_task")
        assert best is not None
        assert best.provider_id == "online"

    def test_unregister(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="task",
                provider_id="p1",
                executor=_MockExecutor(),
            )
        )
        assert reg.unregister("task", "p1") is True
        assert len(reg.list_providers("task")) == 0

    def test_stats(self):
        reg = CapabilityRegistry()
        reg.register(
            CapabilitySpec(
                capability_name="a",
                provider_id="p1",
                executor=_MockExecutor(),
            )
        )
        reg.register(
            CapabilitySpec(
                capability_name="a",
                provider_id="p2",
                executor=_MockExecutor(),
            )
        )
        stats = reg.stats()
        assert stats["total_capabilities"] == 1
        assert stats["total_providers"] == 2
