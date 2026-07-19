"""
Unit and integration tests for Multi-Tenant Security Hardening.

Covers:
- TenantContext and SystemContext immutability
- SystemContext capability enforcement
- TenantSQLiteRepository policy check and isolation violations
- BeliefStore partition isolation and schema migrations
- GoalManager dependency and tenant isolation
- EvaluationNode isolation
- Async/concurrent context integrity checks
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from hbllm.brain.emotion.goal_manager import GoalManager
from hbllm.brain.evaluation.evaluation_node import EvaluationNode, EvaluationReport
from hbllm.brain.reasoning.belief_store import BeliefStore
from hbllm.security import (
    SystemContext,
    TenantContext,
    TenantSQLiteRepository,
    require_capability,
)
from hbllm.security.tenant_guard import (
    TenantIsolationError,
    _ctx_system,
    _ctx_system_capabilities,
    get_current_tenant,
)


@pytest.fixture(autouse=True)
def force_strict_guard(monkeypatch):
    monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")


def test_context_immutability():
    """Verify that TenantContext and SystemContext are immutable once entered."""
    ctx = TenantContext(tenant_id="tenant_a")
    assert ctx.tenant_id == "tenant_a"

    with ctx:
        assert get_current_tenant() == "tenant_a"
        with pytest.raises(AttributeError):
            ctx.tenant_id = "tenant_b"

    sys_ctx = SystemContext(capabilities={"maintenance", "db_migration"})
    assert sys_ctx.capabilities == {"maintenance", "db_migration"}

    with sys_ctx:
        assert _ctx_system.get(False) is True
        caps = _ctx_system_capabilities.get()
        assert caps is not None
        assert "maintenance" in caps
        with pytest.raises(AttributeError):
            sys_ctx.capabilities = {"all"}


def test_system_capability_checks():
    """Verify require_capability behavior."""
    with pytest.raises(TenantIsolationError):
        # No SystemContext active
        require_capability("maintenance")

    sys_ctx = SystemContext(capabilities={"maintenance"})
    with sys_ctx:
        # Should succeed
        require_capability("maintenance")
        # Missing capability should fail
        with pytest.raises(TenantIsolationError):
            require_capability("db_migration")


# ─── 2. TenantSQLiteRepository Parameter Validation ──────────────────────────


class DummyTenantRepository(TenantSQLiteRepository):
    """Subclass for testing protected table policies."""

    pass


def test_repository_policy_enforcement():
    """Verify queries to protected tables are blocked without tenant_id or SystemContext."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        repo = DummyTenantRepository(db_path)

        # Create dummy table beliefs
        with sqlite3.connect(db_path) as conn:
            conn.execute("CREATE TABLE beliefs (belief_id TEXT, tenant_id TEXT, concept TEXT)")

        with sqlite3.connect(db_path) as conn:
            # 1. No context active -> fail-secure block
            with pytest.raises(TenantIsolationError) as excinfo:
                repo.execute_tenant(conn, "SELECT * FROM beliefs WHERE concept = ?", ("gravity",))
            assert "attempted to query protected table" in str(excinfo.value).lower()

            # 2. Tenant context active but SQL lacks tenant_id filter -> block
            with TenantContext("tenant_a"):
                with pytest.raises(TenantIsolationError) as excinfo:
                    repo.execute_tenant(
                        conn, "SELECT * FROM beliefs WHERE concept = ?", ("gravity",)
                    )
                assert "missing 'tenant_id' filter clause" in str(excinfo.value).lower()

            # 3. Tenant context active, SQL has tenant_id filter but parameters are missing it -> block
            with TenantContext("tenant_a"):
                with pytest.raises(TenantIsolationError) as excinfo:
                    repo.execute_tenant(
                        conn,
                        "SELECT * FROM beliefs WHERE concept = ? AND tenant_id = ?",
                        ("gravity",),
                    )
                assert "is not present in query parameters" in str(excinfo.value).lower()

            # 4. Tenant context active, correct query and parameter -> succeed
            with TenantContext("tenant_a"):
                cur = repo.execute_tenant(
                    conn,
                    "SELECT * FROM beliefs WHERE concept = ? AND tenant_id = ?",
                    ("gravity", "tenant_a"),
                )
                assert cur.fetchall() == []

            # 5. SystemContext active -> bypass restriction
            with SystemContext(capabilities={"maintenance"}):
                cur = repo.execute_tenant(
                    conn,
                    "SELECT * FROM beliefs WHERE concept = ?",
                    ("gravity",),
                    required_capability="maintenance",
                )
                assert cur.fetchall() == []


# ─── 3. BeliefStore Multi-Tenant Isolation & Migration ────────────────────────


def test_belief_store_tenant_isolation():
    """Verify BeliefStore strictly isolates tenant data and supports SystemContext maintenance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = BeliefStore(tmpdir)

        # Tenant A stores a belief
        with TenantContext("tenant_a"):
            store.store_belief(concept="gravity", claim="gravity is real", source="physicists")
            beliefs = store.get_beliefs_for_concept("gravity")
            assert len(beliefs) == 1
            assert beliefs[0].claim == "gravity is real"

        # Tenant B cannot see Tenant A's belief
        with TenantContext("tenant_b"):
            beliefs = store.get_beliefs_for_concept("gravity")
            assert len(beliefs) == 0

            # Tenant B stores their own belief
            store.store_belief(
                concept="gravity", claim="gravity is artificial", source="conspiracists"
            )
            beliefs = store.get_beliefs_for_concept("gravity")
            assert len(beliefs) == 1
            assert beliefs[0].claim == "gravity is artificial"

        # Verify that get_belief by ID is isolated
        belief_a_id = None
        with TenantContext("tenant_a"):
            belief_a_id = store.get_beliefs_for_concept("gravity")[0].belief_id

        with TenantContext("tenant_b"):
            # Tenant B tries to get Tenant A's belief by ID -> blocked / None
            assert store.get_belief(belief_a_id) is None

        # System maintenance task can view all beliefs without tenant filters
        with SystemContext(capabilities={"belief_maintenance"}):
            with sqlite3.connect(store._db_path) as conn:
                rows = conn.execute("SELECT claim FROM beliefs ORDER BY created_at").fetchall()
                assert len(rows) == 2
                assert rows[0][0] == "gravity is real"
                assert rows[1][0] == "gravity is artificial"


def test_belief_store_v1_v2_migration():
    """Verify BeliefStore correctly upgrades v1 database to v2 with tenant_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "belief_store.db"

        # Step 1: Create a v1 schema database with PRAGMA user_version = 1
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE beliefs (
                    belief_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    belief_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    status TEXT NOT NULL,
                    domain TEXT DEFAULT '',
                    data TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_updated REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE persistent_contradictions (
                    contradiction_id TEXT PRIMARY KEY,
                    concept TEXT NOT NULL,
                    claim_a TEXT NOT NULL,
                    claim_b TEXT NOT NULL,
                    severity REAL NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolution TEXT DEFAULT '',
                    data TEXT NOT NULL,
                    detected_at REAL NOT NULL,
                    resolved_at REAL
                )
            """)
            conn.execute(
                "INSERT INTO beliefs VALUES ('b1', 'physics', 'speed of light is constant', 'factual', 1.0, 'active', 'phys', '{}', 123.0, 123.0)"
            )
            conn.execute("PRAGMA user_version = 1")

        # Step 2: Initialize BeliefStore which should auto-migrate from v1 to v2
        BeliefStore(tmpdir)

        # Verify database is now v2 and contains tenant_id set to '__legacy__'
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute("PRAGMA user_version")
            assert cur.fetchone()[0] == 2

            row = conn.execute("SELECT tenant_id, belief_id FROM beliefs").fetchone()
            assert row[0] == "__legacy__"
            assert row[1] == "b1"


# ─── 4. GoalManager Multi-Tenant Isolation ────────────────────────────────────


def test_goal_manager_tenant_isolation():
    """Verify GoalManager isolates goals, priorities, and dependency resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = GoalManager(tmpdir)

        with TenantContext("tenant_a"):
            g_parent = manager.create_goal(
                name="Improve Vision", description="Tune CNN models", goal_type="learning"
            )
            g_child = manager.create_goal(
                name="Train CNN", description="Epochs 1-10", goal_type="learning"
            )
            manager.subordinate_to(g_child.goal_id, g_parent.goal_id)

            active = manager.get_active_goals()
            assert len(active) == 2

        # Tenant B has zero goals initially
        with TenantContext("tenant_b"):
            assert len(manager.get_active_goals()) == 0
            manager.create_goal(
                name="Optimize Cache", description="Clean sqlite index", goal_type="optimization"
            )
            assert len(manager.get_active_goals()) == 1


# ─── 5. EvaluationNode Tenant Isolation ───────────────────────────────────────


def test_evaluation_node_tenant_isolation():
    """Verify EvaluationNode isolates cognitive metrics per tenant."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "evals.db"

        # Initialize node and database
        node = EvaluationNode(node_id="eval_test", db_path=db_path)
        node._init_db()

        report_a = EvaluationReport(
            correlation_id="corr_a",
            timestamp=time.time(),
            task_success=0.9,
            plan_validity=1.0,
            tool_accuracy=0.85,
            memory_usage=0.7,
            confidence_error=0.1,
            overall_score=0.9,
            flags=[],
        )

        # Persist under Tenant A
        with TenantContext("tenant_a"):
            node._persist_evaluation(report_a)

            # Restore and verify Tenant A sees it
            node._restore_from_db()
            assert len(node._evaluations) == 1
            assert node._evaluations[0].correlation_id == "corr_a"
            assert node._total_evaluated == 1

        # Tenant B should see an empty evaluation list
        with TenantContext("tenant_b"):
            node._restore_from_db()
            assert len(node._evaluations) == 0
            assert node._total_evaluated == 0


# ─── 6. Concurrent Context Propagation & Verification ──────────────────────────


@pytest.mark.asyncio
async def test_concurrent_context_isolation():
    """Verify that multiple concurrent tasks running under different TenantContexts retain context integrity."""

    async def worker(tenant_name: str, delay: float):
        ctx = TenantContext(tenant_name)
        with ctx:
            assert get_current_tenant() == tenant_name
            await asyncio.sleep(delay)
            # Re-verify context integrity after async sleep (interleaved execution)
            assert get_current_tenant() == tenant_name
        # Context should be cleared outside the block
        assert get_current_tenant() is None

    # Run tasks concurrently with staggered execution times
    tasks = [
        worker("tenant_x", 0.05),
        worker("tenant_y", 0.02),
        worker("tenant_z", 0.08),
        worker("tenant_x", 0.01),
    ]
    await asyncio.gather(*tasks)
