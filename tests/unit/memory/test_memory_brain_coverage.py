"""
Memory, Persistence & Brain Subsystems — Integration test coverage.

Covers uncovered lines in:
  - hbllm/persistence/state.py (BrainState sync API)
  - hbllm/memory/scope.py
  - hbllm/brain/self_state.py (ToolReliability, EpistemicCalibration, SelfState)
  - hbllm/brain/mesh/delegator.py
  - hbllm/brain/control/guard.py (IntentEnvelope, IntentIntegrityEngine)
  - hbllm/brain/observability/tracer.py (DecisionTrace, DecisionTraceLedger)
  - hbllm/brain/snn/comprehension/calibrator.py
  - hbllm/brain/simulation/risk.py
  - hbllm/brain/causality/causal_memory.py
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ═══════════════════════════════════════════════════════════════════════
# persistence/state.py — BrainState (synchronous SQLite backend)
# ═══════════════════════════════════════════════════════════════════════


class TestBrainState:
    @pytest.fixture
    def state(self, tmp_path):
        from hbllm.persistence.state import BrainState

        return BrainState(path=str(tmp_path / "test_brain.db"))

    def test_save_and_load(self, state):
        state.save("test_key", {"data": "hello"}, tenant_id="t1")
        loaded = state.load("test_key", tenant_id="t1")
        assert loaded["data"] == "hello"

    def test_load_missing_returns_none(self, state):
        assert state.load("nonexistent", tenant_id="t1") is None

    def test_save_overwrite(self, state):
        state.save("key", {"v": 1}, tenant_id="t1")
        state.save("key", {"v": 2}, tenant_id="t1")
        assert state.load("key", tenant_id="t1")["v"] == 2

    def test_delete(self, state):
        state.save("key", {"v": 1}, tenant_id="t1")
        state.delete("key", tenant_id="t1")
        assert state.load("key", tenant_id="t1") is None

    def test_delete_nonexistent(self, state):
        state.delete("nonexistent", tenant_id="t1")  # should not raise

    def test_messages_crud(self, state):
        state.append_message("user", "Hello", tenant_id="t1")
        state.append_message("assistant", "Hi!", tenant_id="t1")
        msgs = state.get_messages(tenant_id="t1")
        assert len(msgs) == 2

    def test_messages_limit(self, state):
        for i in range(10):
            state.append_message("user", f"msg {i}", tenant_id="t1")
        msgs = state.get_messages(tenant_id="t1", limit=3)
        assert len(msgs) == 3

    def test_clear_messages(self, state):
        state.append_message("user", "Hello", tenant_id="t1")
        state.clear_messages(tenant_id="t1")
        msgs = state.get_messages(tenant_id="t1")
        assert len(msgs) == 0

    def test_checkpoint(self, state):
        state.checkpoint({"model": "v1"}, tenant_id="t1")
        latest = state.latest_checkpoint(tenant_id="t1")
        assert latest is not None

    def test_list_checkpoints(self, state):
        state.checkpoint({"v": 1}, tenant_id="t1")
        state.checkpoint({"v": 2}, tenant_id="t1")
        cps = state.list_checkpoints(tenant_id="t1")
        assert len(cps) >= 2

    def test_tool_logging(self, state):
        state.log_tool_call(
            tool_name="python_exec",
            input_data="print(1)",
            output="1",
            duration_ms=42.0,
            tenant_id="t1",
            user_id="",
            device_id="",
        )
        logs = state.get_tool_logs(tenant_id="t1")
        assert len(logs) >= 1
        assert logs[0]["tool"] == "python_exec"

    def test_tool_logs_filter_by_tool(self, state):
        state.log_tool_call("tool_a", "in", "out", 10, "t1")
        state.log_tool_call("tool_b", "in", "out", 20, "t1")
        logs = state.get_tool_logs(tenant_id="t1", tool_name="tool_a")
        assert len(logs) == 1

    def test_close(self, state):
        state.close()
        state.close()  # double close should not raise

    def test_tenant_isolation(self, state):
        state.save("key", {"v": "A"}, tenant_id="t1")
        state.save("key", {"v": "B"}, tenant_id="t2")
        assert state.load("key", tenant_id="t1")["v"] == "A"
        assert state.load("key", tenant_id="t2")["v"] == "B"


# ═══════════════════════════════════════════════════════════════════════
# memory/scope.py
# ═══════════════════════════════════════════════════════════════════════


class TestMemoryScope:
    def test_scope_values(self):
        from hbllm.memory.scope import MemoryScope

        assert MemoryScope.WORKING is not None
        assert MemoryScope.EPISODIC is not None
        assert MemoryScope.SEMANTIC is not None
        assert MemoryScope.SENSITIVE is not None


# ═══════════════════════════════════════════════════════════════════════
# brain/self_state.py
# ═══════════════════════════════════════════════════════════════════════


class TestToolReliabilityTracker:
    def test_default_reliability(self):
        from hbllm.brain.self_state import ToolReliabilityTracker

        tracker = ToolReliabilityTracker()
        r = tracker.get_reliability("unknown_tool")
        assert isinstance(r, float) and r >= 0

    def test_record_success(self):
        from hbllm.brain.self_state import ToolReliabilityTracker

        tracker = ToolReliabilityTracker()
        tracker.record_execution("tool1", True)
        r = tracker.get_reliability("tool1")
        assert isinstance(r, float)

    def test_record_failure_degrades(self):
        from hbllm.brain.self_state import ToolReliabilityTracker

        tracker = ToolReliabilityTracker()
        initial = tracker.get_reliability("tool1")
        tracker.record_execution("tool1", False)
        after = tracker.get_reliability("tool1")
        assert after <= initial


class TestEpistemicCalibrationTracker:
    def test_default_calibration(self):
        from hbllm.brain.self_state import EpistemicCalibrationTracker

        tracker = EpistemicCalibrationTracker()
        cal = tracker.get_calibration("unknown")
        assert isinstance(cal, float)

    def test_record_verification_match(self):
        from hbllm.brain.self_state import EpistemicCalibrationTracker

        tracker = EpistemicCalibrationTracker()
        tracker.record_verification("math", predicted_outcome="A", verified_outcome="A", match=True)
        cal = tracker.get_calibration("math")
        assert isinstance(cal, float)

    def test_record_verification_mismatch(self):
        from hbllm.brain.self_state import EpistemicCalibrationTracker

        tracker = EpistemicCalibrationTracker()
        tracker.record_verification(
            "math", predicted_outcome="A", verified_outcome="B", match=False
        )
        cal = tracker.get_calibration("math")
        assert isinstance(cal, float)


class TestSelfStateEngine:
    def test_cognitive_pressure_no_governance(self):
        from hbllm.brain.self_state import SelfStateEngine

        engine = SelfStateEngine(governance=None)
        pressure = engine.get_cognitive_pressure()
        assert isinstance(pressure, float)


# ═══════════════════════════════════════════════════════════════════════
# brain/control/guard.py
# ═══════════════════════════════════════════════════════════════════════


class TestIntentEnvelope:
    def test_create_and_hash(self):
        from hbllm.brain.control.guard import IntentEnvelope
        from hbllm.brain.control.permissions import ActionClass

        env = IntentEnvelope(
            envelope_id="e1",
            goal_description="Run a test",
            planned_actions=[{"tool_name": "python_exec", "args": {"code": "print(1)"}}],
            risk_level=ActionClass.SENSITIVE,
            execution_window_s=60.0,
            allowed_scopes=["local"],
            explanation="Testing",
        )
        h = env.compute_hash()
        assert isinstance(h, str) and len(h) > 0

    def test_hash_deterministic(self):
        from hbllm.brain.control.guard import IntentEnvelope
        from hbllm.brain.control.permissions import ActionClass

        kwargs = dict(
            envelope_id="e1",
            goal_description="Test",
            planned_actions=[{"tool_name": "act", "args": {}}],
            risk_level=ActionClass.SAFE,
            execution_window_s=30.0,
            allowed_scopes=["local"],
            explanation="Test",
        )
        env1 = IntentEnvelope(**kwargs)
        env2 = IntentEnvelope(**kwargs)
        assert env1.compute_hash() == env2.compute_hash()


class TestIntentIntegrityEngine:
    def test_record_and_verify(self):
        from hbllm.brain.control.guard import IntentEnvelope, IntentIntegrityEngine
        from hbllm.brain.control.permissions import ActionClass

        engine = IntentIntegrityEngine()
        actions = [{"tool_name": "act", "args": {}}]
        scopes = ["local"]
        env = IntentEnvelope(
            envelope_id="e1",
            goal_description="Test",
            planned_actions=actions,
            risk_level=ActionClass.SAFE,
            execution_window_s=30.0,
            allowed_scopes=scopes,
            explanation="Test",
        )
        engine.record_approval(env)
        # verify_integrity takes individual params, not the envelope object
        assert engine.verify_integrity("e1", actions, scopes, 30.0)

    def test_verify_unapproved(self):
        from hbllm.brain.control.guard import IntentIntegrityEngine

        engine = IntentIntegrityEngine()
        # Envelope "e99" was never approved
        assert not engine.verify_integrity(
            "e99", [{"tool_name": "act", "args": {}}], ["local"], 30.0
        )

    def test_verify_tampered(self):
        from hbllm.brain.control.guard import IntentEnvelope, IntentIntegrityEngine
        from hbllm.brain.control.permissions import ActionClass

        engine = IntentIntegrityEngine()
        env = IntentEnvelope(
            envelope_id="e3",
            goal_description="Original",
            planned_actions=[{"tool_name": "act", "args": {}}],
            risk_level=ActionClass.SAFE,
            execution_window_s=30.0,
            allowed_scopes=["local"],
            explanation="Test",
        )
        engine.record_approval(env)
        # Tamper: change the actions after approval
        tampered_actions = [{"tool_name": "HACKED", "args": {}}]
        assert not engine.verify_integrity("e3", tampered_actions, ["local"], 30.0)


# ═══════════════════════════════════════════════════════════════════════
# brain/observability/tracer.py
# ═══════════════════════════════════════════════════════════════════════


class TestDecisionTrace:
    def test_explain_decision(self):
        from hbllm.brain.observability.tracer import DecisionTrace

        trace = DecisionTrace(
            trace_id="t1",
            goal_id="g1",
            selected_scenario_id="s1",
        )
        explanation = trace.explain_decision()
        assert isinstance(explanation, str)


class TestDecisionTraceLedger:
    def test_record_and_retrieve(self, tmp_path):
        from hbllm.brain.observability.tracer import DecisionTrace, DecisionTraceLedger

        ledger = DecisionTraceLedger(data_dir=str(tmp_path), ring_size=10)
        trace = DecisionTrace(
            trace_id="t1",
            goal_id="g1",
            selected_scenario_id="s1",
        )
        ledger.record_decision(trace)
        recent = ledger.get_recent_traces()
        assert len(recent) >= 1

    def test_ring_buffer_eviction(self, tmp_path):
        from hbllm.brain.observability.tracer import DecisionTrace, DecisionTraceLedger

        ledger = DecisionTraceLedger(data_dir=str(tmp_path), ring_size=3)
        for i in range(5):
            trace = DecisionTrace(
                trace_id=f"t{i}",
                goal_id=f"g{i}",
                selected_scenario_id="s",
            )
            ledger.record_decision(trace)
        recent = ledger.get_recent_traces()
        assert len(recent) <= 3

    def test_explain_decision_by_id(self, tmp_path):
        from hbllm.brain.observability.tracer import DecisionTrace, DecisionTraceLedger

        ledger = DecisionTraceLedger(data_dir=str(tmp_path), ring_size=10)
        trace = DecisionTrace(
            trace_id="t1",
            goal_id="g1",
            selected_scenario_id="s1",
        )
        ledger.record_decision(trace)
        explanation = ledger.explain_decision("t1")
        assert isinstance(explanation, str)


# ═══════════════════════════════════════════════════════════════════════
# brain/snn/comprehension/calibrator.py
# ═══════════════════════════════════════════════════════════════════════


class TestSNNCalibrator:
    def test_record_outcome(self, tmp_path):
        from hbllm.brain.snn.comprehension.calibrator import SNNCalibrator

        cal = SNNCalibrator(data_dir=str(tmp_path))
        cal.record_outcome(
            domain="math",
            params_used={"threshold": 0.5},
            num_concepts=5,
            response_quality=0.8,
            memory_relevance=0.7,
        )

    def test_record_multiple_outcomes(self, tmp_path):
        from hbllm.brain.snn.comprehension.calibrator import SNNCalibrator

        cal = SNNCalibrator(data_dir=str(tmp_path))
        for i in range(10):
            cal.record_outcome(
                domain="science",
                params_used={"threshold": 0.3 + i * 0.05},
                num_concepts=i + 1,
                response_quality=0.5 + i * 0.05,
                memory_relevance=0.6,
            )


# ═══════════════════════════════════════════════════════════════════════
# brain/mesh/delegator.py
# ═══════════════════════════════════════════════════════════════════════


class TestContractDelegator:
    def test_propose_contract(self):
        from hbllm.brain.mesh.delegator import ContractDelegator

        delegator = ContractDelegator(local_node_id="node_1")
        capsule = MagicMock()
        capsule.priority = 5
        capsule.estimated_cost = 100
        result = delegator.propose_contract("node_2", capsule)
        assert result is not None

    def test_evaluate_contract(self):
        from hbllm.brain.mesh.delegator import ContractDelegator, ContractOffer

        delegator = ContractDelegator(local_node_id="node_1")
        capsule = MagicMock()
        capsule.is_valid = True
        capsule.priority = MagicMock(value=1)  # enum-like with .value
        capsule.estimated_cost = 50
        capsule.capsule_id = "cap_1"
        offer = ContractOffer(capsule=capsule, offered_by="node_2")
        response = delegator.evaluate_contract(offer, current_memory_pressure=0.3)
        assert response is not None


# ═══════════════════════════════════════════════════════════════════════
# brain/simulation/risk.py
# ═══════════════════════════════════════════════════════════════════════


class TestRiskSimulation:
    def test_risk_engine_import(self):
        from hbllm.brain.simulation.risk import RiskEngine

        assert RiskEngine is not None

    def test_risk_category_enum(self):
        from hbllm.brain.simulation.risk import RiskCategory

        assert RiskCategory is not None

    def test_risk_profile(self):
        from hbllm.brain.simulation.risk import RiskProfile

        assert RiskProfile is not None


# ═══════════════════════════════════════════════════════════════════════
# brain/causality/causal_memory.py
# ═══════════════════════════════════════════════════════════════════════


class TestCausalMemory:
    def test_import(self):
        from hbllm.brain.causality import causal_memory

        assert causal_memory is not None
