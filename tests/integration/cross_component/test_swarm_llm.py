"""Tests for SwarmEngine LLM decomposition."""

from __future__ import annotations

import pytest

from hbllm.brain.planning.swarm_node import (
    SwarmNode as SwarmEngine,
)
from hbllm.brain.planning.swarm_node import (
    SwarmTask,
    TaskDecomposer,
    TaskStatus,
)

# ── Heuristic Decomposition ─────────────────────────────────────────────


def test_decompose_numbered_steps():
    """Should split numbered steps into subtasks."""
    task = """
    1. Set up the database schema
    2. Create the API endpoints
    3. Build the frontend UI
    4. Write integration tests
    """
    subtasks = TaskDecomposer.decompose(task)
    assert len(subtasks) == 4
    assert "database" in subtasks[0].description.lower()


def test_decompose_bullet_points():
    """Should split bullet points into subtasks."""
    task = """
    - Install dependencies
    - Configure environment variables
    - Run the migration script
    """
    subtasks = TaskDecomposer.decompose(task)
    assert len(subtasks) == 3


def test_decompose_single_task():
    """Single task without structure should be wrapped."""
    task = "Deploy the application to production"
    subtasks = TaskDecomposer.decompose(task)
    assert len(subtasks) == 1
    assert "deploy" in subtasks[0].description.lower()


def test_decompose_semicolons():
    """Should split by semicolons."""
    task = "build frontend; test everything; deploy"
    subtasks = TaskDecomposer.decompose(task)
    assert len(subtasks) == 3


def test_decompose_max_subtasks():
    """Should respect max_subtasks limit."""
    task = "\n".join(f"{i}. Step {i}" for i in range(1, 20))
    subtasks = TaskDecomposer.decompose(task, max_subtasks=3)
    assert len(subtasks) <= 3


def test_identify_dependencies():
    """Should detect dependency references."""
    subtasks = [
        SwarmTask(task_id="sub_1", description="Create the database"),
        SwarmTask(task_id="sub_2", description="Use the result from sub_1"),
    ]
    resolved = TaskDecomposer.identify_dependencies(subtasks)
    assert "sub_1" in resolved[1].dependencies


# ── LLM Decomposition ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_decompose_valid_json():
    """LLM returning valid JSON should produce subtasks."""

    async def mock_llm(prompt: str) -> str:
        return """[
            {"id": "sub_1", "description": "Setup database", "dependencies": [], "priority": 1.0},
            {"id": "sub_2", "description": "Build API", "dependencies": ["sub_1"], "priority": 0.9},
            {"id": "sub_3", "description": "Write tests", "dependencies": [], "priority": 0.8}
        ]"""

    subtasks = await TaskDecomposer.decompose_with_llm("Build a web app", mock_llm)
    assert len(subtasks) == 3
    assert subtasks[0].task_id == "sub_1"
    assert subtasks[1].dependencies == ["sub_1"]
    assert subtasks[2].priority == 0.8


@pytest.mark.asyncio
async def test_llm_decompose_markdown_wrapped():
    """LLM wrapping JSON in markdown code blocks should still parse."""

    async def mock_llm(prompt: str) -> str:
        return """Sure! Here's the decomposition:

```json
[
    {"id": "sub_1", "description": "Step one", "dependencies": [], "priority": 0.9}
]
```"""

    subtasks = await TaskDecomposer.decompose_with_llm("Do something", mock_llm)
    assert len(subtasks) == 1
    assert subtasks[0].description == "Step one"


@pytest.mark.asyncio
async def test_llm_decompose_fallback_on_error():
    """LLM failure should fall back to heuristic decomposition."""

    async def failing_llm(prompt: str) -> str:
        raise RuntimeError("LLM unavailable")

    task = "1. First step\n2. Second step"
    subtasks = await TaskDecomposer.decompose_with_llm(task, failing_llm)
    # Should fall back to heuristic and still return subtasks
    assert len(subtasks) >= 2


@pytest.mark.asyncio
async def test_llm_decompose_invalid_json():
    """Invalid JSON from LLM should fall back to heuristic."""

    async def bad_json_llm(prompt: str) -> str:
        return "I can't help with that."

    task = "Build everything"
    subtasks = await TaskDecomposer.decompose_with_llm(task, bad_json_llm)
    assert len(subtasks) >= 1  # Heuristic fallback


@pytest.mark.asyncio
async def test_llm_decompose_max_subtasks():
    """Should respect max_subtasks even with LLM output."""

    async def verbose_llm(prompt: str) -> str:
        items = [
            {"id": f"sub_{i}", "description": f"Step {i}", "dependencies": [], "priority": 0.5}
            for i in range(1, 20)
        ]
        import json

        return json.dumps(items)

    subtasks = await TaskDecomposer.decompose_with_llm("Big task", verbose_llm, max_subtasks=3)
    assert len(subtasks) <= 3


# ── SwarmEngine with LLM ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_swarm_engine_uses_llm_when_available():
    """SwarmEngine should use LLM decomposition when llm_generate is provided."""

    async def mock_llm(prompt: str) -> str:
        return """[
            {"id": "sub_1", "description": "Do the thing", "dependencies": [], "priority": 1.0}
        ]"""

    async def mock_worker(description: str) -> str:
        return f"Done: {description}"

    engine = SwarmEngine(llm_generate=mock_llm)
    engine.set_worker(mock_worker)

    execution = await engine.execute("Build something complex")
    assert len(execution.tasks) >= 1
    assert execution.tasks[0].description == "Do the thing"
    assert execution.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_swarm_engine_heuristic_without_llm():
    """SwarmEngine without llm_generate should use heuristic decomposition."""

    async def mock_worker(description: str) -> str:
        return f"Done: {description}"

    engine = SwarmEngine(llm_generate=None)
    engine.set_worker(mock_worker)

    execution = await engine.execute("1. First\n2. Second\n3. Third")
    assert len(execution.tasks) == 3
    assert execution.status == TaskStatus.COMPLETED
