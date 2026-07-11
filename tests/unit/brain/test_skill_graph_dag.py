import pytest

from hbllm.brain.skills.skill_registry import Skill, SkillGraphExecutor, SkillRegistry


@pytest.mark.asyncio
async def test_linear_fallback_execution(tmp_path):
    registry = SkillRegistry(data_dir=str(tmp_path))

    skill = Skill(
        skill_id="linear_skill",
        name="Linear Test",
        description="Linear steps fallback",
        category="test",
        steps=["echo step1", "echo step2"],
        tools_used=[],
        success_criteria="ok",
    )

    execution_trace = []

    async def mock_runner(node, context, outputs):
        execution_trace.append(node["action"])
        return f"output_for_{node['id']}"

    executor = SkillGraphExecutor(registry, mock_runner)
    result = await executor.execute(skill, {})

    assert len(execution_trace) == 2
    assert execution_trace[0] == "echo step1"
    assert execution_trace[1] == "echo step2"
    assert result["statuses"]["step_0"] == "completed"
    assert result["statuses"]["step_1"] == "completed"


@pytest.mark.asyncio
async def test_parallel_and_conditional_dag(tmp_path):
    registry = SkillRegistry(data_dir=str(tmp_path))

    # DAG:
    #       root (check config)
    #      /    \
    #     a      b (conditional on root.output == 'sqlite')
    #      \    /
    #       join

    nodes = [
        {"id": "root", "type": "command", "action": "determine_db"},
        {"id": "branch_a", "type": "command", "action": "run_a"},
        {"id": "branch_b", "type": "command", "action": "run_b"},
        {"id": "join_node", "type": "command", "action": "finalize"},
    ]

    edges = [
        {"source": "root", "target": "branch_a"},
        {"source": "root", "target": "branch_b", "condition": "root == sqlite"},
        {"source": "branch_a", "target": "join_node"},
        {"source": "branch_b", "target": "join_node"},
    ]

    skill = Skill(
        skill_id="dag_skill",
        name="DAG Test",
        description="DAG conditional check",
        category="test",
        steps=[],
        tools_used=[],
        success_criteria="ok",
        nodes=nodes,
        edges=edges,
    )

    execution_trace = []

    async def mock_runner(node, context, outputs):
        execution_trace.append(node["id"])
        if node["id"] == "root":
            # Return sqlite to satisfy conditional edge for branch_b
            return "sqlite"
        return "ok"

    executor = SkillGraphExecutor(registry, mock_runner)
    result = await executor.execute(skill, {})

    assert "root" in execution_trace
    assert "branch_a" in execution_trace
    assert "branch_b" in execution_trace
    assert "join_node" in execution_trace
    assert result["statuses"]["join_node"] == "completed"


@pytest.mark.asyncio
async def test_conditional_skip(tmp_path):
    registry = SkillRegistry(data_dir=str(tmp_path))

    nodes = [
        {"id": "root", "type": "command", "action": "determine_db"},
        {"id": "branch_b", "type": "command", "action": "run_b"},
    ]

    edges = [
        {"source": "root", "target": "branch_b", "condition": "root == pgsql"},
    ]

    skill = Skill(
        skill_id="skip_skill",
        name="Skip Test",
        description="DAG conditional skip",
        category="test",
        steps=[],
        tools_used=[],
        success_criteria="ok",
        nodes=nodes,
        edges=edges,
    )

    execution_trace = []

    async def mock_runner(node, context, outputs):
        execution_trace.append(node["id"])
        if node["id"] == "root":
            # Return sqlite, which fails the pgsql condition
            return "sqlite"
        return "ok"

    executor = SkillGraphExecutor(registry, mock_runner)
    result = await executor.execute(skill, {})

    assert "root" in execution_trace
    assert "branch_b" not in execution_trace  # Skipped
    assert (
        result["statuses"]["branch_b"] == "completed"
    )  # resolver marks skipped conditional branches completed


@pytest.mark.asyncio
async def test_dependency_failure_propagation(tmp_path):
    registry = SkillRegistry(data_dir=str(tmp_path))

    nodes = [
        {"id": "step1", "type": "command", "action": "succeed"},
        {"id": "step2", "type": "command", "action": "fail"},
        {"id": "step3", "type": "command", "action": "run_after_fail"},
    ]

    edges = [
        {"source": "step1", "target": "step2"},
        {"source": "step2", "target": "step3"},
    ]

    skill = Skill(
        skill_id="failure_skill",
        name="Failure Test",
        description="Dependency failure propagation",
        category="test",
        steps=[],
        tools_used=[],
        success_criteria="ok",
        nodes=nodes,
        edges=edges,
    )

    async def mock_runner(node, context, outputs):
        if node["id"] == "step2":
            raise RuntimeError("Step 2 Failed")
        return "ok"

    executor = SkillGraphExecutor(registry, mock_runner)
    result = await executor.execute(skill, {})

    assert result["statuses"]["step1"] == "completed"
    assert result["statuses"]["step2"] == "failed"
    assert result["statuses"]["step3"] == "failed"  # Blocked and marked failed
