"""
Evaluation Node — closes the intelligence feedback loop.

Runs after DecisionNode to measure cognitive performance per interaction.
Feeds results into CognitiveMetrics and GoalManager to enable
scientific iteration rather than guessing.

Metrics measured:
  1. task_success — did the response satisfy the query intent?
  2. plan_validity — was the selected plan well-formed?
  3. tool_accuracy — did tool calls succeed?
  4. memory_usage — was retrieved context useful?
  5. confidence_error — gap between predicted and actual confidence

Flow:
  DecisionNode publishes → sensory.output + system.experience
  EvaluationNode subscribes → system.experience
  EvaluationNode scores → publishes system.evaluation
  CognitiveMetrics records → GoalManager generates improvement goals
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Per-interaction cognitive evaluation."""

    correlation_id: str
    timestamp: float
    task_success: float  # 0-1: did response address query?
    plan_validity: float  # 0-1: was the plan well-structured?
    tool_accuracy: float  # 0-1: did tools work correctly?
    memory_usage: float  # 0-1: was retrieved context relevant?
    confidence_error: float  # 0-1: |predicted - actual| confidence
    overall_score: float  # weighted composite
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "task_success": round(self.task_success, 3),
            "plan_validity": round(self.plan_validity, 3),
            "tool_accuracy": round(self.tool_accuracy, 3),
            "memory_usage": round(self.memory_usage, 3),
            "confidence_error": round(self.confidence_error, 3),
            "overall_score": round(self.overall_score, 3),
            "flags": self.flags,
        }


class EvaluationNode(Node):
    """
    Post-decision evaluation node. Closes the intelligence feedback loop.

    Subscribes to:
        system.experience — decision outputs from DecisionNode
        sensory.output — final responses sent to users
        workspace.thought — intermediate cognitive events
        system.feedback — explicit user feedback

    Publishes:
        system.evaluation — per-interaction evaluation reports
        system.evaluation.aggregate — periodic aggregate metrics
    """

    def __init__(
        self,
        node_id: str,
        cognitive_metrics: Any = None,
        goal_manager: Any = None,
        self_model: Any = None,
        skill_registry: Any = None,
        evaluation_window: int = 100,
        goal_trigger_interval: float = 300.0,  # 5 minutes
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["evaluation", "feedback_loop"],
        )
        self.cognitive_metrics = cognitive_metrics
        self.goal_manager = goal_manager
        self.self_model = self_model
        self.skill_registry = skill_registry

        self.evaluation_window = evaluation_window
        self.goal_trigger_interval = goal_trigger_interval
        self._last_goal_trigger = 0.0

        self.weights = weights or {
            "task_success": 0.30,
            "plan_validity": 0.20,
            "tool_accuracy": 0.20,
            "memory_usage": 0.15,
            "confidence_error": 0.15,
        }

        # Rolling evaluation history
        self._evaluations: list[EvaluationReport] = []
        self._pending_contexts: dict[str, dict[str, Any]] = {}

        # Counters
        self._total_evaluated = 0
        self._total_flagged = 0

    async def on_start(self) -> None:
        logger.info("Starting EvaluationNode (Intelligence Feedback Loop)")
        await self.bus.subscribe("system.experience", self._handle_experience)
        await self.bus.subscribe("sensory.output", self._handle_output)
        await self.bus.subscribe("system.feedback", self._handle_feedback)
        await self.bus.subscribe("evaluation.query", self._handle_query)

    async def on_stop(self) -> None:
        logger.info(
            "Stopping EvaluationNode — evaluated=%d flagged=%d",
            self._total_evaluated,
            self._total_flagged,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Event Handlers ───────────────────────────────────────────────

    async def _handle_experience(self, message: Message) -> None:
        """Capture decision context from DecisionNode output."""
        payload = message.payload
        corr_id = message.correlation_id or message.id

        self._pending_contexts[corr_id] = {
            "intent": payload.get("intent", "answer"),
            "thought_type": payload.get("thought_type", "intuition"),
            "content": payload.get("text", "")[:500],
            "confidence": payload.get("confidence", 0.5),
            "tools_used": payload.get("tools_used", []),
            "memory_hits": payload.get("memory_hits", 0),
            "plan_steps": payload.get("plan_steps", []),
            "timestamp": time.time(),
        }

        # Evaluate immediately with available context
        report = self._evaluate(corr_id, self._pending_contexts[corr_id])
        await self._publish_evaluation(report, message)

    async def _handle_output(self, message: Message) -> None:
        """Capture final output to enrich pending evaluations."""
        corr_id = message.correlation_id or ""
        if corr_id in self._pending_contexts:
            ctx = self._pending_contexts[corr_id]
            ctx["output"] = message.payload.get("text", "")[:500]
            ctx["output_timestamp"] = time.time()

    async def _handle_feedback(self, message: Message) -> None:
        """
        Incorporate explicit user feedback to correct evaluations.

        This closes the loop: user says "bad" → confidence_error increases →
        GoalManager creates improvement goal → system self-improves.
        """
        payload = message.payload
        rating = payload.get("rating", 0)
        corr_id = message.correlation_id or ""

        if corr_id in self._pending_contexts:
            ctx = self._pending_contexts[corr_id]
            predicted = ctx.get("confidence", 0.5)

            # Map rating (-1, 0, 1) to actual success (0, 0.5, 1)
            actual = (rating + 1) / 2.0
            confidence_error = abs(predicted - actual)

            # Record the corrected confidence error
            if self.cognitive_metrics:
                self.cognitive_metrics.record_confidence(predicted, actual)

            logger.info(
                "[EvaluationNode] Feedback received: rating=%d predicted=%.2f actual=%.2f error=%.2f",
                rating,
                predicted,
                actual,
                confidence_error,
            )

    async def _handle_query(self, message: Message) -> Message | None:
        """Return evaluation stats."""
        return message.create_response(self.stats())

    # ── Core Evaluation Engine ───────────────────────────────────────

    def _evaluate(self, corr_id: str, ctx: dict[str, Any]) -> EvaluationReport:
        """Score a single interaction across 5 cognitive dimensions."""
        flags: list[str] = []

        # 1. Task Success — proxy via content length and intent alignment
        task_success = self._score_task_success(ctx)
        if task_success < 0.4:
            flags.append("low_task_success")

        # 2. Plan Validity — were plan steps well-formed?
        plan_validity = self._score_plan_validity(ctx)
        if plan_validity < 0.4:
            flags.append("weak_plan")

        # 3. Tool Accuracy — did tools execute correctly?
        tool_accuracy = self._score_tool_accuracy(ctx)
        if tool_accuracy < 0.5:
            flags.append("tool_failures")

        # 4. Memory Usage — was context retrieval useful?
        memory_usage = self._score_memory_usage(ctx)
        if memory_usage < 0.3:
            flags.append("poor_memory_utilization")

        # 5. Confidence Error — calibration quality
        confidence_error = self._score_confidence_error(ctx)
        if confidence_error > 0.4:
            flags.append("miscalibrated_confidence")

        # Composite weighted score
        overall = (
            self.weights["task_success"] * task_success
            + self.weights["plan_validity"] * plan_validity
            + self.weights["tool_accuracy"] * tool_accuracy
            + self.weights["memory_usage"] * memory_usage
            + self.weights["confidence_error"] * (1.0 - confidence_error)
        )

        report = EvaluationReport(
            correlation_id=corr_id,
            timestamp=time.time(),
            task_success=task_success,
            plan_validity=plan_validity,
            tool_accuracy=tool_accuracy,
            memory_usage=memory_usage,
            confidence_error=confidence_error,
            overall_score=max(0.0, min(1.0, overall)),
            flags=flags,
        )

        # Store in rolling window
        self._evaluations.append(report)
        if len(self._evaluations) > self.evaluation_window:
            self._evaluations = self._evaluations[-self.evaluation_window :]

        self._total_evaluated += 1
        if flags:
            self._total_flagged += 1

        return report

    # ── Individual Scorers ───────────────────────────────────────────

    @staticmethod
    def _score_task_success(ctx: dict[str, Any]) -> float:
        """Estimate task success from response quality signals."""
        content = ctx.get("content", "")
        intent = ctx.get("intent", "answer")
        score = 0.5  # base

        # Content length heuristic (very short = likely incomplete)
        word_count = len(content.split())
        if word_count > 50:
            score += 0.2
        elif word_count > 20:
            score += 0.1
        elif word_count < 5:
            score -= 0.2

        # Intent alignment (code questions should have code blocks)
        if intent in ("code", "execute") and "```" in content:
            score += 0.15
        elif intent == "answer" and word_count > 10:
            score += 0.1

        # Confidence from thought
        confidence = ctx.get("confidence", 0.5)
        score += confidence * 0.15

        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_plan_validity(ctx: dict[str, Any]) -> float:
        """Score plan quality based on structure."""
        plan_steps = ctx.get("plan_steps", [])

        if not plan_steps:
            return 0.6  # no plan needed = acceptable

        score = 0.5
        # Plans with 2-5 steps are well-scoped
        if 2 <= len(plan_steps) <= 5:
            score += 0.2
        elif len(plan_steps) == 1:
            score += 0.1
        elif len(plan_steps) > 10:
            score -= 0.1  # over-complex

        # Each step should be non-empty
        valid_steps = sum(1 for s in plan_steps if isinstance(s, str) and len(s) > 5)
        step_quality = valid_steps / max(len(plan_steps), 1)
        score += step_quality * 0.3

        return max(0.0, min(1.0, score))

    @staticmethod
    def _score_tool_accuracy(ctx: dict[str, Any]) -> float:
        """Score tool invocation success."""
        tools = ctx.get("tools_used", [])
        if not tools:
            return 0.8  # no tools needed = fine

        # Count successful tool executions
        successful = sum(1 for t in tools if isinstance(t, dict) and t.get("success", True))
        total = len(tools)

        if isinstance(tools[0], str):
            # Simple list of tool names — assume success
            return 0.7

        return successful / max(total, 1)

    @staticmethod
    def _score_memory_usage(ctx: dict[str, Any]) -> float:
        """Score memory context utilization."""
        memory_hits = ctx.get("memory_hits", 0)

        if memory_hits == 0:
            return 0.5  # no memory needed or no context available

        # More hits = better context (diminishing returns)
        if memory_hits >= 5:
            return 0.9
        if memory_hits >= 3:
            return 0.7
        return 0.5 + memory_hits * 0.1

    @staticmethod
    def _score_confidence_error(ctx: dict[str, Any]) -> float:
        """Estimate confidence calibration error."""
        confidence = ctx.get("confidence", 0.5)

        # Without explicit feedback, we can only use heuristics
        content = ctx.get("content", "")
        words = content.lower()

        # High confidence + hedging language = miscalibrated
        hedge_words = ["maybe", "perhaps", "might", "possibly", "not sure", "i think"]
        hedge_count = sum(1 for hw in hedge_words if hw in words)

        if confidence > 0.8 and hedge_count >= 2:
            return 0.6  # high confidence but uncertain language

        if confidence < 0.3 and hedge_count == 0:
            return 0.3  # low confidence but definitive language

        return 0.1  # reasonable calibration

    # ── Feedback Loop Integration ────────────────────────────────────

    async def _publish_evaluation(
        self, report: EvaluationReport, original: Message
    ) -> None:
        """Publish evaluation and trigger downstream loop."""
        # 1. Publish evaluation event
        eval_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=original.tenant_id,
            session_id=original.session_id,
            topic="system.evaluation",
            payload=report.to_dict(),
            correlation_id=original.correlation_id,
        )
        await self.bus.publish("system.evaluation", eval_msg)

        # 2. Record metrics
        if self.cognitive_metrics:
            self.cognitive_metrics.record_reasoning(report.task_success)
            self.cognitive_metrics.record_tool_result(report.tool_accuracy > 0.5)
            self.cognitive_metrics.record_memory_hit(report.memory_usage > 0.5)

        # 3. Update self-model
        if self.self_model:
            domain = self._pending_contexts.get(
                report.correlation_id, {}
            ).get("thought_type", "general")
            self.self_model.record_outcome(
                domain=domain,
                success=report.overall_score > 0.6,
                confidence=report.task_success,
            )

        # 4. Trigger goal generation periodically
        now = time.time()
        if now - self._last_goal_trigger > self.goal_trigger_interval:
            await self._trigger_goal_generation()
            self._last_goal_trigger = now

        # 5. Log quality flags
        if report.flags:
            logger.warning(
                "[EvaluationNode] Quality flags: %s (score=%.2f)",
                report.flags,
                report.overall_score,
            )

        # Clean up pending context
        self._pending_contexts.pop(report.correlation_id, None)

    async def _trigger_goal_generation(self) -> None:
        """Aggregate recent evaluations and feed GoalManager."""
        if not self.goal_manager or not self._evaluations:
            return

        recent = self._evaluations[-50:]

        metrics = {
            "hallucination_rate": 1.0 - sum(e.task_success for e in recent) / len(recent),
            "tool_success_rate": sum(e.tool_accuracy for e in recent) / len(recent),
            "memory_utilization": sum(e.memory_usage for e in recent) / len(recent),
            "avg_latency_ms": 0,  # filled from CognitiveMetrics if available
        }

        if self.cognitive_metrics:
            latency = self.cognitive_metrics.get_metric("latency_ms", hours=1)
            metrics["avg_latency_ms"] = latency.get("avg", 0)

        goals = self.goal_manager.generate_from_performance(metrics)
        if goals:
            logger.info(
                "[EvaluationNode] Generated %d improvement goals from evaluation aggregate",
                len(goals),
            )

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        if not self._evaluations:
            return {
                "total_evaluated": 0,
                "total_flagged": 0,
                "avg_score": 0,
            }

        recent = self._evaluations[-50:]
        return {
            "total_evaluated": self._total_evaluated,
            "total_flagged": self._total_flagged,
            "avg_overall_score": round(
                sum(e.overall_score for e in recent) / len(recent), 3
            ),
            "avg_task_success": round(
                sum(e.task_success for e in recent) / len(recent), 3
            ),
            "avg_plan_validity": round(
                sum(e.plan_validity for e in recent) / len(recent), 3
            ),
            "avg_tool_accuracy": round(
                sum(e.tool_accuracy for e in recent) / len(recent), 3
            ),
            "avg_memory_usage": round(
                sum(e.memory_usage for e in recent) / len(recent), 3
            ),
            "avg_confidence_error": round(
                sum(e.confidence_error for e in recent) / len(recent), 3
            ),
            "flag_rate": round(self._total_flagged / max(self._total_evaluated, 1), 3),
            "window_size": len(recent),
        }
