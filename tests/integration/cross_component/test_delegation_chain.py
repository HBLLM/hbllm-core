"""Tests for DelegationChain — long-running autonomous task execution."""

import tempfile
from pathlib import Path

from hbllm.brain.planning.delegation_chain import (
    Delegation,
    DelegationManager,
    DelegationStatus,
    DelegationStep,
    StepSensitivity,
    StepStatus,
)


class TestDelegationStep:
    def test_defaults(self):
        step = DelegationStep(description="Run tests")
        assert step.status == StepStatus.PENDING
        assert step.sensitivity == StepSensitivity.SAFE

    def test_serialization(self):
        step = DelegationStep(
            description="Deploy",
            action="shell",
            sensitivity=StepSensitivity.SENSITIVE,
        )
        d = step.to_dict()
        s2 = DelegationStep.from_dict(d)
        assert s2.description == "Deploy"
        assert s2.sensitivity == StepSensitivity.SENSITIVE


class TestDelegation:
    def test_progress_empty(self):
        d = Delegation(tenant_id="t1", objective="Test")
        assert d.progress == 0.0

    def test_progress_partial(self):
        d = Delegation(
            tenant_id="t1",
            objective="Test",
            steps=[
                DelegationStep(description="S1", status=StepStatus.COMPLETED),
                DelegationStep(description="S2", status=StepStatus.PENDING),
            ],
        )
        assert d.progress == 0.5

    def test_progress_complete(self):
        d = Delegation(
            tenant_id="t1",
            objective="Test",
            steps=[
                DelegationStep(description="S1", status=StepStatus.COMPLETED),
                DelegationStep(description="S2", status=StepStatus.COMPLETED),
            ],
        )
        assert d.progress == 1.0

    def test_current_step(self):
        d = Delegation(
            tenant_id="t1",
            objective="Test",
            steps=[
                DelegationStep(description="S1", status=StepStatus.COMPLETED),
                DelegationStep(description="S2", status=StepStatus.PENDING),
            ],
        )
        assert d.current_step.description == "S2"

    def test_serialization(self):
        d = Delegation(
            tenant_id="t1",
            objective="Deploy staging",
            steps=[DelegationStep(description="Run tests")],
        )
        data = d.to_dict()
        d2 = Delegation.from_dict(data)
        assert d2.objective == "Deploy staging"
        assert len(d2.steps) == 1


class TestDelegationManager:
    def test_create(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                "t1",
                objective="Deploy v2",
                steps=[DelegationStep(description="Run tests")],
            )
            assert d.objective == "Deploy v2"
            assert (Path(tmpdir) / "t1" / f"{d.id}.json").exists()

    def test_next_step_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                "t1",
                objective="Test",
                steps=[DelegationStep(description="Step 1")],
            )
            step = mgr.next_step("t1", d.id)
            assert step is not None
            assert step.status == StepStatus.RUNNING

    def test_next_step_sensitive_pauses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                "t1",
                objective="Test",
                steps=[
                    DelegationStep(
                        description="Sensitive step",
                        sensitivity=StepSensitivity.SENSITIVE,
                    )
                ],
            )
            step = mgr.next_step("t1", d.id)
            assert step.status == StepStatus.WAITING_APPROVAL
            d_updated = mgr.get("t1", d.id)
            assert d_updated.status == DelegationStatus.PAUSED

    def test_approve_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                "t1",
                objective="Test",
                steps=[
                    DelegationStep(
                        description="Needs approval",
                        sensitivity=StepSensitivity.SENSITIVE,
                    )
                ],
            )
            step = mgr.next_step("t1", d.id)
            mgr.approve_step("t1", d.id, step.id)
            d_updated = mgr.get("t1", d.id)
            assert d_updated.status == DelegationStatus.ACTIVE
            assert d_updated.steps[0].status == StepStatus.RUNNING

    def test_complete_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                "t1",
                objective="Test",
                steps=[DelegationStep(description="S1")],
            )
            step = mgr.next_step("t1", d.id)
            mgr.complete_step("t1", d.id, step.id, result="Tests passed")
            d_updated = mgr.get("t1", d.id)
            assert d_updated.status == DelegationStatus.COMPLETED

    def test_fail_step_with_retry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                "t1",
                objective="Test",
                steps=[DelegationStep(description="Flaky step")],
            )
            step = mgr.next_step("t1", d.id)
            mgr.fail_step("t1", d.id, step.id, error="timeout")
            d_updated = mgr.get("t1", d.id)
            # Should retry (step reset to PENDING)
            assert d_updated.steps[0].status == StepStatus.PENDING
            assert d_updated.retry_count == 1

    def test_fail_step_exhausted_retries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create(
                "t1",
                objective="Test",
                steps=[DelegationStep(description="Always fails")],
            )
            # Exhaust retries (max_retries=2 by default)
            for _ in range(3):
                step = mgr.next_step("t1", d.id)
                if step and step.status == StepStatus.RUNNING:
                    mgr.fail_step("t1", d.id, step.id, error="fail")
            d_updated = mgr.get("t1", d.id)
            assert d_updated.status == DelegationStatus.FAILED

    def test_cancel(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create("t1", objective="Cancel me", steps=[])
            mgr.cancel("t1", d.id)
            d_updated = mgr.get("t1", d.id)
            assert d_updated.status == DelegationStatus.CANCELLED

    def test_list_active(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            mgr.create("t1", objective="Active 1", steps=[])
            mgr.create("t1", objective="Active 2", steps=[])
            active = mgr.list_active("t1")
            assert len(active) == 2

    def test_persistence_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            d = mgr.create("t1", objective="Persistent", steps=[DelegationStep(description="S1")])

            mgr2 = DelegationManager(storage_dir=tmpdir)
            found = mgr2.get("t1", d.id)
            assert found is not None
            assert found.objective == "Persistent"

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = DelegationManager(storage_dir=tmpdir)
            mgr.create("t1", objective="Test", steps=[])
            stats = mgr.stats("t1")
            assert stats["total"] == 1
            assert stats["active"] == 1
