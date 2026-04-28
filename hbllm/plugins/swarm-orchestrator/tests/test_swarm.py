"""Tests for SwarmEngine cognitive plugin."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pytest_asyncio
from swarm_engine import (
    SwarmEngine,
    SwarmExecution,
    SwarmTask,
    TaskDecomposer,
    TaskStatus,
)

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def bus():
    b = InProcessBus()
    await b.start()
    yield b
    await b.stop()


@pytest_asyncio.fixture
async def engine(bus):
    e = SwarmEngine(node_id="test_swarm", max_workers=2, task_timeout=5.0)
    await e.start(bus)
    yield e
    await e.stop()


class TestSwarmTask:
    def test_default_status(self):
        t = SwarmTask(task_id="t1", description="test")
        assert t.status == TaskStatus.PENDING

    def test_to_dict(self):
        t = SwarmTask(task_id="t1", description="test", priority=0.8)
        d = t.to_dict()
        assert d["task_id"] == "t1"
        assert d["priority"] == 0.8


class TestSwarmExecution:
    def test_progress_empty(self):
        e = SwarmExecution(execution_id="e1", original_task="test")
        assert e.progress == 0.0

    def test_progress_partial(self):
        e = SwarmExecution(
            execution_id="e1",
            original_task="test",
            tasks=[
                SwarmTask(task_id="t1", description="a", status=TaskStatus.COMPLETED),
                SwarmTask(task_id="t2", description="b", status=TaskStatus.PENDING),
            ],
        )
        assert e.progress == 0.5


class TestTaskDecomposer:
    def test_decompose_numbered_steps(self):
        task = "1. Install Python\n2. Create virtualenv\n3. Install packages"
        tasks = TaskDecomposer.decompose(task)
        assert len(tasks) == 3
        assert "Install Python" in tasks[0].description

    def test_decompose_step_prefix(self):
        task = "Step 1: Research\nStep 2: Implement\nStep 3: Test"
        tasks = TaskDecomposer.decompose(task)
        assert len(tasks) == 3

    def test_decompose_semicolons(self):
        task = "build the frontend; deploy to staging; run tests"
        tasks = TaskDecomposer.decompose(task)
        assert len(tasks) == 3

    def test_decompose_and_then(self):
        task = "compile the code and then run the tests and then deploy"
        tasks = TaskDecomposer.decompose(task)
        assert len(tasks) == 3

    def test_decompose_single_task(self):
        task = "What is Python?"
        tasks = TaskDecomposer.decompose(task)
        assert len(tasks) == 1

    def test_max_subtasks_limit(self):
        task = "\n".join(f"{i + 1}. Step {i + 1}" for i in range(10))
        tasks = TaskDecomposer.decompose(task, max_subtasks=3)
        assert len(tasks) <= 3

    def test_identify_dependencies(self):
        tasks = [
            SwarmTask(task_id="sub_1", description="Fetch data"),
            SwarmTask(task_id="sub_2", description="Process the result from above"),
        ]
        result = TaskDecomposer.identify_dependencies(tasks)
        assert result[1].dependencies == ["sub_1"]

    def test_no_false_dependencies(self):
        tasks = [
            SwarmTask(task_id="sub_1", description="Task A"),
            SwarmTask(task_id="sub_2", description="Task B independently"),
        ]
        result = TaskDecomposer.identify_dependencies(tasks)
        assert result[1].dependencies == []


class TestSwarmEngine:
    async def test_execute_without_worker(self, engine):
        result = await engine.execute("1. Step one\n2. Step two")
        assert result.status == TaskStatus.COMPLETED
        assert len(result.tasks) == 2

    async def test_execute_with_worker(self, engine):
        async def mock_worker(desc: str) -> str:
            return f"Done: {desc}"

        engine.set_worker(mock_worker)
        result = await engine.execute("build the app; test the app")
        assert result.status == TaskStatus.COMPLETED
        assert "Done:" in result.tasks[0].result

    async def test_worker_failure_handled(self, engine):
        async def failing_worker(desc: str) -> str:
            raise ValueError("Worker crashed")

        engine.set_worker(failing_worker)
        result = await engine.execute("do something")
        assert result.status == TaskStatus.FAILED
        assert result.tasks[0].error == "Worker crashed"

    async def test_worker_timeout(self, engine):
        engine.task_timeout = 0.1

        async def slow_worker(desc: str) -> str:
            await asyncio.sleep(5)
            return "never"

        engine.set_worker(slow_worker)
        result = await engine.execute("slow task")
        assert result.tasks[0].status == TaskStatus.FAILED
        assert "timed out" in result.tasks[0].error

    async def test_parallel_execution(self, engine):
        import time as _time

        call_times = []

        async def timed_worker(desc: str) -> str:
            call_times.append(_time.time())
            await asyncio.sleep(0.1)
            return f"Done: {desc}"

        engine.set_worker(timed_worker)
        result = await engine.execute("task A; task B")
        assert result.status == TaskStatus.COMPLETED
        if len(call_times) == 2:
            assert abs(call_times[0] - call_times[1]) < 0.05

    async def test_stats(self, engine):
        await engine.execute("simple task")
        stats = engine.stats()
        assert stats["total_executions"] == 1
        assert stats["completed"] == 1
        assert stats["max_workers"] == 2

    async def test_swarm_request_via_bus(self, engine, bus):
        received = []

        async def capture(msg: Message) -> None:
            received.append(msg)

        await bus.subscribe("swarm.complete", capture)
        msg = Message(
            type=MessageType.EVENT,
            source_node_id="test",
            topic="swarm.request",
            payload={"task": "1. Step one\n2. Step two"},
        )
        await bus.publish("swarm.request", msg)
        await asyncio.sleep(0.2)
        assert len(received) >= 1
        assert received[0].payload["status"] == "completed"

    async def test_aggregated_result_format(self, engine):
        result = await engine.execute("1. First\n2. Second")
        assert "✅" in result.aggregated_result
        assert "First" in result.aggregated_result
