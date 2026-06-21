"""Tests for AuditTrail — safety governance logging."""

import pytest
import pytest_asyncio

from hbllm.security.audit_trail import AuditTrail


@pytest_asyncio.fixture
async def audit(tmp_path):
    a = AuditTrail(db_path=tmp_path / "audit.db", max_age_days=30)
    await a.init_db()
    return a


@pytest.mark.asyncio
async def test_log_entry(audit):
    """Entries can be logged and retrieved."""
    entry_id = audit.log(
        "t1", "lock.unlock", "iot", risk_tier=3, source="user", target="front_door"
    )
    assert entry_id > 0


@pytest.mark.asyncio
async def test_query_by_tenant(audit):
    """Entries are filtered by tenant."""
    audit.log("t1", "light.on", "iot", source="autonomy")
    audit.log("t2", "light.off", "iot", source="autonomy")
    entries = audit.query(tenant_id="t1", hours=1)
    assert len(entries) == 1
    assert entries[0].tenant_id == "t1"


@pytest.mark.asyncio
async def test_query_by_category(audit):
    """Entries are filtered by category."""
    audit.log("t1", "lock.unlock", "iot", source="user")
    audit.log("t1", "file.delete", "system", source="user")
    entries = audit.query(tenant_id="t1", category="iot", hours=1)
    assert len(entries) == 1
    assert entries[0].category == "iot"


@pytest.mark.asyncio
async def test_query_by_risk_tier(audit):
    """Entries are filtered by minimum risk tier."""
    audit.log("t1", "light.on", "iot", risk_tier=1, source="autonomy")
    audit.log("t1", "lock.unlock", "iot", risk_tier=3, source="user")
    entries = audit.query(tenant_id="t1", min_risk_tier=3, hours=1)
    assert len(entries) == 1
    assert entries[0].risk_tier == 3


@pytest.mark.asyncio
async def test_query_by_result(audit):
    """Entries are filtered by result."""
    audit.log("t1", "lock.unlock", "iot", result="denied", source="autonomy")
    audit.log("t1", "light.on", "iot", result="success", source="autonomy")
    denied = audit.query(tenant_id="t1", result="denied", hours=1)
    assert len(denied) == 1
    assert denied[0].result == "denied"


@pytest.mark.asyncio
async def test_summary(audit):
    """Summary aggregates correctly."""
    audit.log("t1", "a1", "iot", risk_tier=1, source="user", result="success")
    audit.log("t1", "a2", "system", risk_tier=3, source="autonomy", result="denied")
    audit.log("t1", "a3", "iot", risk_tier=0, source="autonomy", result="success")
    summary = audit.get_summary("t1", hours=1)
    assert summary["total_actions"] == 3
    assert summary["by_category"]["iot"] == 2
    assert summary["by_result"]["success"] == 2
    assert summary["high_risk_actions"] == 1


@pytest.mark.asyncio
async def test_hash_chain_integrity(audit):
    """Hash chain integrity check passes for fresh entries."""
    audit.log("t1", "a1", "iot", source="user")
    audit.log("t1", "a2", "iot", source="user")
    integrity = audit.verify_integrity()
    assert integrity["status"] == "ok"
    assert integrity["entries_checked"] == 2
    assert integrity["missing_hashes"] == 0


@pytest.mark.asyncio
async def test_entry_to_dict(audit):
    """AuditEntry.to_dict produces valid output."""
    audit.log("t1", "test.action", "test", risk_tier=2, source="unit_test", target="target")
    entries = audit.query(tenant_id="t1", hours=1)
    d = entries[0].to_dict()
    assert d["action"] == "test.action"
    assert d["risk_tier"] == 2
    assert d["target"] == "target"


@pytest.mark.asyncio
async def test_prune_old_entries(audit):
    """Pruning removes only old entries."""
    audit.log("t1", "old_action", "iot", source="user")
    # With max_age_days=30, recent entries survive
    pruned = audit.prune_old_entries()
    assert pruned == 0  # Nothing is old enough


@pytest.mark.asyncio
async def test_stats(audit):
    """Stats returns expected fields."""
    audit.log("t1", "a1", "iot", source="user")
    s = audit.stats()
    assert s["total_entries"] == 1
    assert "db_path" in s
