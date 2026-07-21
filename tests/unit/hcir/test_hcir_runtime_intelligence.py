"""Unit tests for HCIR Runtime Intelligence: Context, Sandboxing, LLM Compiler, and Learning Loop."""

import pytest

from hbllm.hcir.abi import ExecutionMetrics
from hbllm.hcir.compiler import IntentType
from hbllm.hcir.compiler_llm import LLMCompilerFrontend, StructuredIntentPayload
from hbllm.hcir.context import HCIRExecutionContext
from hbllm.hcir.identity import HCIRObjectID
from hbllm.hcir.kernel.capability_sandboxing import (
    CapabilityPermissions,
    CapabilityResourceLimits,
    CapabilitySandboxManager,
    IsolationMode,
    SandboxedCapabilityPolicy,
    TrustLevel,
)
from hbllm.hcir.learning_loop import LearningLoopEngine, SkillCompiler
from hbllm.hcir.receipt import ExecutionReceipt
from hbllm.hcir.types import Scope
from hbllm.hcir.workspace import HCIRWorkspaceState


# ═══════════════════════════════════════════════════════════════════════════
# HCIRExecutionContext Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestHCIRExecutionContext:
    def test_defaults(self):
        ctx = HCIRExecutionContext()
        assert ctx.attention_budget == 1000
        assert ctx.is_simulation() is False

    def test_fork_for_simulation(self):
        ctx = HCIRExecutionContext(attention_budget=1000)
        sim_ctx = ctx.fork_for_simulation(branch_name="sim_branch_1")

        assert sim_ctx.is_simulation() is True
        assert sim_ctx.simulation_branch == "sim_branch_1"
        assert sim_ctx.attention_budget == 500

    def test_consume_budget(self):
        ctx = HCIRExecutionContext(attention_budget=100)
        assert ctx.consume_budget(40) is True
        assert ctx.attention_budget == 60
        assert ctx.consume_budget(80) is False
        assert ctx.attention_budget == 60


# ═══════════════════════════════════════════════════════════════════════════
# Capability Sandboxing Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCapabilitySandboxing:
    def test_sandbox_policy_permissions(self):
        mgr = CapabilitySandboxManager()
        policy = SandboxedCapabilityPolicy(
            capability_name="python_exec",
            provider_id="local_sandbox",
            trust_level=TrustLevel.VERIFIED,
            permissions=CapabilityPermissions(
                allow_filesystem=False,
                allow_network=True,
            ),
        )
        mgr.register_policy(policy)

        assert mgr.check_permission("python_exec", "local_sandbox", "network") is True
        assert mgr.check_permission("python_exec", "local_sandbox", "filesystem") is False

    def test_default_unregistered_policy_denies_dangerous_ops(self):
        mgr = CapabilitySandboxManager()
        assert mgr.check_permission("unknown", "unknown", "filesystem") is False
        assert mgr.check_permission("unknown", "unknown", "read_only") is True


# ═══════════════════════════════════════════════════════════════════════════
# LLM Compiler Frontend Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLLMCompilerFrontend:
    def test_parse_json_payload(self):
        frontend = LLMCompilerFrontend()
        json_data = """{
            "intent": "plan",
            "subject": "solar dehydrator",
            "action": "build",
            "slots": {"material": "wood"},
            "constraints": ["budget < 100"],
            "priority": 0.9
        }"""
        stream = frontend.compile_json(json_data, author="qwen")

        assert stream.author == "qwen"
        assert stream.length > 0
        assert stream.instructions[0].opcode.value == "ASSERT"

    def test_fallback_intent(self):
        frontend = LLMCompilerFrontend()
        payload_dict = {
            "intent": "unknown_intent_type",
            "subject": "battery",
        }
        ast = frontend.payload_to_ast(frontend.parse_payload(payload_dict))
        assert ast.intent == IntentType.QUERY


# ═══════════════════════════════════════════════════════════════════════════
# Learning Loop Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestLearningLoop:
    def test_evaluate_receipt_successful(self):
        ws = HCIRWorkspaceState()
        engine = LearningLoopEngine(ws)

        receipt = ExecutionReceipt(
            execution_id="rcpt_1",
            author="planner",
            success=True,
            capabilities_used=["planning", "search"],
            metrics=ExecutionMetrics(tokens_consumed=100),
        )

        outcome = engine.evaluate_receipt(receipt)
        assert outcome.utility_score > 0.6
        assert outcome.should_extract_skill is True

    def test_process_and_learn_commits_skill(self):
        from hbllm.hcir.kernel.transaction_manager import TransactionManager

        ws = HCIRWorkspaceState()
        tx_mgr = TransactionManager(ws)
        engine = LearningLoopEngine(ws, transaction_manager=tx_mgr)

        receipt = ExecutionReceipt(
            execution_id="rcpt_success",
            author="acme",
            success=True,
            capabilities_used=["causal_analysis"],
            metrics=ExecutionMetrics(tokens_consumed=100),
        )

        tx = engine.process_and_learn(receipt, user_reward=1.0)
        assert tx is not None
        assert tx.status.value == "committed"

        # Verify SkillNode exists in graph
        nodes = ws.graph.all_nodes()
        skill_nodes = [n for n in nodes if n.node_type.value == "skill"]
        assert len(skill_nodes) == 1
