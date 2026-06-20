import pytest_asyncio

"""Tests for Procedural Memory — skill storage and retrieval."""

import pytest

from hbllm.memory.procedural import ProceduralMemory


@pytest_asyncio.fixture
async def proc_mem(tmp_path):
    mem = ProceduralMemory(db_path=tmp_path / "test.db")
    await mem.init_db()
    yield mem
    await mem.close()


@pytest.mark.asyncio
async def test_store_and_find_skill(proc_mem):
    """Store a skill and retrieve it by keyword."""
    await proc_mem.store_skill(
        tenant_id="t1",
        skill_name="deploy_docker",
        trigger_pattern="deploy a docker container",
        steps=[{"action": "build_image"}, {"action": "push_registry"}, {"action": "deploy"}],
    )

    results = await proc_mem.find_skill("t1", "docker")
    assert len(results) == 1
    assert results[0]["skill_name"] == "deploy_docker"
    assert len(results[0]["steps"]) == 3


@pytest.mark.asyncio
async def test_skill_tenant_isolation(proc_mem):
    """Skills are scoped to tenant_id."""
    await proc_mem.store_skill("t1", "skill_a", "pattern_a", [{"step": 1}])
    await proc_mem.store_skill("t2", "skill_b", "pattern_b", [{"step": 2}])

    assert len(await proc_mem.find_skill("t1", "skill")) == 1
    assert (await proc_mem.find_skill("t1", "skill"))[0]["skill_name"] == "skill_a"
    assert (await proc_mem.find_skill("t2", "skill"))[0]["skill_name"] == "skill_b"


@pytest.mark.asyncio
async def test_record_usage_and_success_rate(proc_mem):
    """Usage count increments and success rate updates with EMA."""
    skill_id = await proc_mem.store_skill("t1", "test_skill", "test", [{"do": "it"}])

    await proc_mem.record_usage(skill_id, success=True)
    await proc_mem.record_usage(skill_id, success=True)
    await proc_mem.record_usage(skill_id, success=False)

    results = await proc_mem.find_skill("t1", "test_skill")
    assert results[0]["usage_count"] == 3
    assert results[0]["success_rate"] < 1.0  # Degraded by one failure


@pytest.mark.asyncio
async def test_get_most_used(proc_mem):
    """Most-used skills appear first."""
    await proc_mem.store_skill("t1", "rarely_used", "rare", [{"step": 1}])
    id2 = await proc_mem.store_skill("t1", "commonly_used", "common", [{"step": 2}])

    await proc_mem.record_usage(id2, success=True)
    await proc_mem.record_usage(id2, success=True)
    await proc_mem.record_usage(id2, success=True)

    top = await proc_mem.get_most_used("t1", top_k=2)
    assert top[0]["skill_name"] == "commonly_used"
    assert top[0]["usage_count"] == 3


@pytest.mark.asyncio
async def test_update_existing_skill(proc_mem):
    """Storing a skill with the same name updates it."""
    await proc_mem.store_skill("t1", "my_skill", "old pattern", [{"old": "steps"}])
    await proc_mem.store_skill("t1", "my_skill", "new pattern", [{"new": "steps"}])

    results = await proc_mem.find_skill("t1", "my_skill")
    assert len(results) == 1
    assert results[0]["steps"] == [{"new": "steps"}]
    assert results[0]["trigger_pattern"] == "new pattern"


@pytest.mark.asyncio
async def test_delete_skill(proc_mem):
    """Skills can be deleted."""
    skill_id = await proc_mem.store_skill("t1", "temp_skill", "temp", [{"step": 1}])
    assert await proc_mem.delete_skill(skill_id)
    assert len(await proc_mem.find_skill("t1", "temp_skill")) == 0


@pytest.mark.asyncio
async def test_find_no_results(proc_mem):
    """Finding with no matching skills returns empty list."""
    assert await proc_mem.find_skill("t1", "nonexistent") == []
