"""
Skill Intelligence Layer (SIL) Node.

Sits conceptually between the Planner and the Executor.
Responsibilities:
1. Skill Selection (based on confidence and cost_score)
2. Execution delegation
3. Failure routing & automatic skill repair (via Failure Analyzer)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from hbllm.brain.skill_registry import SkillRegistry
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

class SkillIntelligenceNode(Node):
    """
    Skill Intelligence Layer (SIL).
    Governs task execution by choosing and invoking learned skills,
    detecting failures, and triggering automated skill repair.
    """
    def __init__(self, node_id: str, skill_registry: SkillRegistry) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["skill_selection", "skill_lifecycle", "execution_governance"],
        )
        self.skill_registry = skill_registry

    async def on_start(self) -> None:
        logger.info("Starting SkillIntelligenceNode")
        await self.bus.subscribe("action.sil_execute", self.handle_message)

    async def on_stop(self) -> None:
        logger.info("Stopping SkillIntelligenceNode")

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        # Expecting a task description or query
        query = message.payload.get("task", "")
        if not query:
            return message.create_error("No 'task' provided to SIL.")

        # 1. Skill Selection
        # Find highest confidence skill matching the query.
        skills = self.skill_registry.find_skill(query, top_k=3)

        # Filter for acceptable confidence (>0.7) and pick best by cost_score or success_rate
        tenant_id = message.tenant_id or "global"
        viable_skills = [s for s in skills if s.confidence_score >= 0.7 and s.success_rate > 0.5 and (s.tenant_id == tenant_id or s.tenant_id == "global")]

        if not viable_skills:
            # Fallback to direct raw execution, letting SkillCompiler capture it on success later.
            return await self._fallback_raw_execution(message, query)

        # Sort by confidence + success_rate
        best_skill = max(viable_skills, key=lambda s: (s.confidence_score, s.success_rate))

        logger.info("SIL selected skill '%s' (v%d) with confidence %.2f for task: %s",
                    best_skill.name, getattr(best_skill, 'version', 1), getattr(best_skill, 'confidence_score', 0.8), query)

        # 1.5 Execution Dry Run (Simulation) for marginal trust
        if 0.7 <= best_skill.confidence_score < 0.85:
            sim_req = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.simulate",
                payload={"action_type": "simulate_skill", "steps": best_skill.steps}
            )
            sim_resp = await self.request("workspace.simulate", sim_req, timeout=10.0)
            if sim_resp and sim_resp.payload.get("prediction") == "FAILURE":
                logger.warning("SIL Simulation failed for skill '%s'. Failing early to avoid side-effects.", best_skill.name)
                return message.create_error(f"Dry-run simulation failed: {sim_resp.payload.get('content')}")

        # 2. Execution Delegation
        start_time = time.time()
        success, error_msg, trace = await self._execute_skill(best_skill, message)
        latency_ms = (time.time() - start_time) * 1000

        # 3. Handle Lifecycle
        if success:
            logger.info("SIL successfully executed skill '%s'", best_skill.name)
            # Rough estimate of tokens used during skill execution
            tokens_used = len("".join(best_skill.steps)) // 4
            self.skill_registry.record_execution(best_skill.skill_id, True, latency_ms, tokens=tokens_used)
            return message.create_response({
                "status": "SUCCESS",
                "skill": best_skill.name,
                "execution_trace": trace
            })
        else:
            logger.warning("SIL failed executing skill '%s': %s", best_skill.name, error_msg)
            # Route to Failure Analyzer
            repair_req = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                topic="action.analyze_failure",
                payload={
                    "skill_name": best_skill.name,
                    "steps": best_skill.steps,
                    "execution_trace": trace,
                    "error_message": error_msg
                }
            )
            repair_resp = await self.request("action.analyze_failure", repair_req, timeout=120.0)

            failure_type = "Unknown"
            if repair_resp and repair_resp.type != MessageType.ERROR:
                rt_payload = repair_resp.payload
                failure_type = rt_payload.get("failure_type", "Unknown")
                if rt_payload.get("repaired") and rt_payload.get("new_steps"):
                    # Create a new version of the skill
                    new_skill = self.skill_registry.version_skill(
                        skill_id=best_skill.skill_id,
                        new_steps=rt_payload.get("new_steps"),
                        test_latency_ms=latency_ms
                    )
                    if new_skill:
                        logger.info("SIL autogenerated new skill version: %s v%d", new_skill.name, getattr(new_skill, 'version', 1))

            # Record failure against the old skill
            self.skill_registry.record_execution(best_skill.skill_id, False, latency_ms, failure_type=failure_type)

            return message.create_error(f"Execution failed: {error_msg}")

    async def _execute_skill(self, skill: Any, origin_message: Message) -> tuple[bool, str, list[dict[str, Any]]]:
        """Simulate executing skill steps by routing to existing primitives."""
        trace: list[dict[str, Any]] = []
        for step_idx, step in enumerate(skill.steps):

            # Hierarchical execution check
            is_sil = False
            task_query = ""
            if isinstance(step, str) and '"action"' in step and 'sil_execute' in step:
                try:
                    parsed = json.loads(step)
                    if parsed.get("action") == "sil_execute":
                        is_sil = True
                        task_query = parsed.get("task", "")
                except Exception:
                    pass

            if is_sil:
                req = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    topic="action.sil_execute",
                    payload={"task": task_query},
                    tenant_id=origin_message.tenant_id,
                    session_id=origin_message.session_id,
                )
                topic = "action.sil_execute"
            else:
                req = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    topic="action.execute_code",
                    payload={"code": step},
                    tenant_id=origin_message.tenant_id,
                    session_id=origin_message.session_id,
                )
                topic = "action.execute_code"

            try:
                resp = await self.request(topic, req, timeout=30.0)
                if resp.type == MessageType.ERROR:
                    trace.append({"step": step, "status": "failed", "error": str(resp.payload.get("error"))})
                    return False, str(resp.payload.get("error", "Execution error")), trace
                elif resp.payload.get("status") != "SUCCESS":
                    trace.append({"step": step, "status": "failed", "error": str(resp.payload.get("error"))})
                    return False, str(resp.payload.get("error", "Sub-process failed")), trace
                trace.append({"step": step, "status": "success", "output": resp.payload.get("output")})
            except Exception as e:
                trace.append({"step": step, "status": "failed", "error": str(e)})
                return False, str(e), trace
        return True, "", trace

    async def _fallback_raw_execution(self, message: Message, query: str) -> Message | None:
        """Fallback to raw planner logic if we had no skill."""
        logger.info("SIL fallback: No matched skill found. Delegating back to raw reasoning tools.")
        return message.create_response({
            "status": "NO_SKILL",
            "reason": "No high confidence skill available for this task. Fallback requested.",
            "execution_trace": []
        })
