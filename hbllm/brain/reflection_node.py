"""
Reflection Node — periodic batch analysis of cognitive performance trends.

Unlike ExperienceNode (which reflects on individual events as they happen),
ReflectionNode runs periodically to analyze accumulated patterns and update
operational strategies. This is the "think about how I've been thinking" system.

Triggers:
  - Timer-based (every N minutes during active use)
  - Sleep cycle integration (during idle consolidation)
  - Manual via bus message

Outputs:
  - Strategy updates to SelfModel (capability adjustments)
  - Priority updates to GoalManager (new improvement goals)
  - Failure pattern reports for SkillCompilerNode
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


@dataclass
class ReflectionInsight:
    """A single insight from periodic reflection."""

    category: str  # performance, strategy, capability, failure_pattern
    severity: str  # info, warning, critical
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)
    recommended_actions: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "severity": self.severity,
            "description": self.description,
            "evidence": self.evidence,
            "recommended_actions": self.recommended_actions,
            "timestamp": self.timestamp,
        }


@dataclass
class ReflectionSession:
    """Result of a complete reflection pass."""

    session_id: str
    timestamp: float
    window_hours: int
    insights: list[ReflectionInsight] = field(default_factory=list)
    metrics_snapshot: dict[str, Any] = field(default_factory=dict)
    actions_taken: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "window_hours": self.window_hours,
            "insight_count": len(self.insights),
            "insights": [i.to_dict() for i in self.insights],
            "metrics_snapshot": self.metrics_snapshot,
            "actions_taken": self.actions_taken,
        }


class ReflectionNode(Node):
    """
    Periodic batch reflection engine.

    Analyzes accumulated cognitive metrics, evaluation reports, and
    experience events to detect trends, failure patterns, and
    improvement opportunities.

    Subscribes to:
        system.evaluation — evaluation reports (accumulated)
        system.sleep.reflection_trigger — sleep cycle trigger
        reflection.trigger — manual trigger

    Publishes:
        system.reflection.session — complete reflection session results
        system.strategy_update — operational strategy changes
    """

    def __init__(
        self,
        node_id: str,
        cognitive_metrics: Any = None,
        goal_manager: Any = None,
        self_model: Any = None,
        skill_registry: Any = None,
        reflection_interval: float = 600.0,  # 10 minutes
        analysis_window_hours: int = 24,
        min_evaluations: int = 10,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["reflection", "trend_analysis", "strategy_update"],
        )
        self.cognitive_metrics = cognitive_metrics
        self.goal_manager = goal_manager
        self.self_model = self_model
        self.skill_registry = skill_registry

        self.reflection_interval = reflection_interval
        self.analysis_window_hours = analysis_window_hours
        self.min_evaluations = min_evaluations

        # Accumulated evaluation reports
        self._evaluation_history: list[dict[str, Any]] = []
        self._max_history = 500

        # Session history
        self._sessions: list[ReflectionSession] = []
        self._session_count = 0

        # Timer task
        self._timer_task: asyncio.Task[None] | None = None
        self._running = False

    async def on_start(self) -> None:
        logger.info(
            "Starting ReflectionNode (interval=%.0fs, window=%dh)",
            self.reflection_interval,
            self.analysis_window_hours,
        )
        await self.bus.subscribe("system.evaluation", self._accumulate_evaluation)
        await self.bus.subscribe("system.sleep.reflection_trigger", self._handle_sleep_trigger)
        await self.bus.subscribe("reflection.trigger", self._handle_manual_trigger)
        await self.bus.subscribe("reflection.query", self._handle_query)

        # Start periodic timer
        self._running = True
        self._timer_task = asyncio.create_task(self._reflection_loop())

    async def on_stop(self) -> None:
        self._running = False
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        logger.info(
            "Stopping ReflectionNode — sessions=%d",
            self._session_count,
        )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Event Handlers ───────────────────────────────────────────────

    async def _accumulate_evaluation(self, message: Message) -> None:
        """Accumulate evaluation reports for batch analysis."""
        self._evaluation_history.append(message.payload)
        if len(self._evaluation_history) > self._max_history:
            self._evaluation_history = self._evaluation_history[-self._max_history :]

    async def _handle_sleep_trigger(self, message: Message) -> Message | None:
        """Run deep reflection during sleep cycle."""
        logger.info("[ReflectionNode] Sleep-triggered reflection starting...")
        session = await self._run_reflection(deep=True)
        return message.create_response(session.to_dict()) if session else None

    async def _handle_manual_trigger(self, message: Message) -> Message | None:
        """Manual reflection trigger."""
        deep = message.payload.get("deep", False)
        session = await self._run_reflection(deep=deep)
        return message.create_response(session.to_dict()) if session else None

    async def _handle_query(self, message: Message) -> Message | None:
        """Return reflection stats and recent sessions."""
        return message.create_response(self.stats())

    # ── Timer Loop ───────────────────────────────────────────────────

    async def _reflection_loop(self) -> None:
        """Periodic reflection timer."""
        while self._running:
            await asyncio.sleep(self.reflection_interval)
            if not self._running:
                break
            try:
                await self._run_reflection(deep=False)
            except Exception as e:
                logger.error("[ReflectionNode] Reflection loop error: %s", e)

    # ── Core Reflection Engine ───────────────────────────────────────

    async def _run_reflection(self, deep: bool = False) -> ReflectionSession | None:
        """Run a complete reflection session."""
        if len(self._evaluation_history) < self.min_evaluations:
            logger.debug(
                "[ReflectionNode] Skipping reflection — only %d evaluations (need %d)",
                len(self._evaluation_history),
                self.min_evaluations,
            )
            return None

        self._session_count += 1
        session_id = f"reflection_{self._session_count}_{int(time.time())}"

        session = ReflectionSession(
            session_id=session_id,
            timestamp=time.time(),
            window_hours=self.analysis_window_hours,
        )

        # Gather metrics snapshot
        if self.cognitive_metrics:
            session.metrics_snapshot = self.cognitive_metrics.get_dashboard_metrics()

        # Run analysis passes
        recent = self._evaluation_history[-100:]

        # 1. Performance trend analysis
        session.insights.extend(self._analyze_performance_trends(recent))

        # 2. Failure pattern detection
        session.insights.extend(self._detect_failure_patterns(recent))

        # 3. Capability gap analysis
        if self.self_model:
            session.insights.extend(self._analyze_capability_gaps())

        # 4. Deep analysis (during sleep only)
        if deep:
            session.insights.extend(self._deep_strategy_analysis(recent))

        # Act on insights
        actions = await self._act_on_insights(session.insights)
        session.actions_taken = actions

        # Store session
        self._sessions.append(session)
        if len(self._sessions) > 20:
            self._sessions = self._sessions[-20:]

        # Publish session results
        await self.bus.publish(
            "system.reflection.session",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="system.reflection.session",
                payload=session.to_dict(),
            ),
        )

        logger.info(
            "[ReflectionNode] Session %s: %d insights, %d actions taken",
            session_id,
            len(session.insights),
            len(actions),
        )

        return session

    # ── Analysis Passes ──────────────────────────────────────────────

    def _analyze_performance_trends(
        self, evals: list[dict[str, Any]]
    ) -> list[ReflectionInsight]:
        """Detect performance trends (improving, declining, or plateau)."""
        insights: list[ReflectionInsight] = []

        if len(evals) < 10:
            return insights

        # Split into two halves and compare
        midpoint = len(evals) // 2
        first_half = evals[:midpoint]
        second_half = evals[midpoint:]

        for metric in ["task_success", "plan_validity", "tool_accuracy", "memory_usage"]:
            first_avg = sum(e.get(metric, 0.5) for e in first_half) / len(first_half)
            second_avg = sum(e.get(metric, 0.5) for e in second_half) / len(second_half)
            delta = second_avg - first_avg

            if delta < -0.1:
                insights.append(
                    ReflectionInsight(
                        category="performance",
                        severity="warning",
                        description=f"{metric} is declining: {first_avg:.2f} → {second_avg:.2f}",
                        evidence={
                            "metric": metric,
                            "first_half_avg": round(first_avg, 3),
                            "second_half_avg": round(second_avg, 3),
                            "delta": round(delta, 3),
                        },
                        recommended_actions=[
                            f"Investigate declining {metric}",
                            f"Generate improvement goal for {metric}",
                        ],
                    )
                )
            elif delta > 0.1:
                insights.append(
                    ReflectionInsight(
                        category="performance",
                        severity="info",
                        description=f"{metric} is improving: {first_avg:.2f} → {second_avg:.2f}",
                        evidence={
                            "metric": metric,
                            "delta": round(delta, 3),
                        },
                    )
                )

        return insights

    def _detect_failure_patterns(
        self, evals: list[dict[str, Any]]
    ) -> list[ReflectionInsight]:
        """Detect recurring failure categories."""
        insights: list[ReflectionInsight] = []

        # Count flag occurrences
        flag_counts: Counter[str] = Counter()
        for e in evals:
            for flag in e.get("flags", []):
                flag_counts[flag] += 1

        total = len(evals)
        for flag, count in flag_counts.most_common(5):
            rate = count / total
            if rate > 0.3:
                severity = "critical" if rate > 0.5 else "warning"
                insights.append(
                    ReflectionInsight(
                        category="failure_pattern",
                        severity=severity,
                        description=f"Recurring issue: '{flag}' in {rate:.0%} of interactions",
                        evidence={
                            "flag": flag,
                            "count": count,
                            "total": total,
                            "rate": round(rate, 3),
                        },
                        recommended_actions=[
                            f"Create targeted improvement goal for '{flag}'",
                            f"Review recent failures with '{flag}' flag",
                        ],
                    )
                )

        return insights

    def _analyze_capability_gaps(self) -> list[ReflectionInsight]:
        """Use SelfModel to find capability gaps."""
        insights: list[ReflectionInsight] = []

        if not self.self_model:
            return insights

        try:
            metrics = self.self_model.get_metrics()
            domains = metrics.get("domains", {})

            for domain, data in domains.items():
                if isinstance(data, dict):
                    success_rate = data.get("success_rate", 1.0)
                    if success_rate < 0.5:
                        insights.append(
                            ReflectionInsight(
                                category="capability",
                                severity="warning",
                                description=f"Low capability in domain '{domain}': {success_rate:.0%} success",
                                evidence={
                                    "domain": domain,
                                    "success_rate": round(success_rate, 3),
                                    "data": data,
                                },
                                recommended_actions=[
                                    f"Focus learning on '{domain}' domain",
                                    f"Route '{domain}' queries to specialized module",
                                ],
                            )
                        )
        except Exception as e:
            logger.debug("Could not analyze capability gaps: %s", e)

        return insights

    def _deep_strategy_analysis(
        self, evals: list[dict[str, Any]]
    ) -> list[ReflectionInsight]:
        """Deep analysis during sleep — reviews full history."""
        insights: list[ReflectionInsight] = []

        # Analyze overall cognitive health
        if not evals:
            return insights

        overall_scores = [e.get("overall_score", 0.5) for e in evals]
        avg_score = sum(overall_scores) / len(overall_scores)

        if avg_score < 0.5:
            insights.append(
                ReflectionInsight(
                    category="strategy",
                    severity="critical",
                    description=f"Overall cognitive performance below threshold: {avg_score:.2f}",
                    evidence={
                        "avg_score": round(avg_score, 3),
                        "sample_size": len(evals),
                    },
                    recommended_actions=[
                        "Trigger emergency learning cycle",
                        "Review and update system prompt",
                        "Consider increasing planner depth",
                    ],
                )
            )

        # Confidence calibration analysis
        conf_errors = [e.get("confidence_error", 0.1) for e in evals]
        avg_conf_error = sum(conf_errors) / len(conf_errors)

        if avg_conf_error > 0.3:
            insights.append(
                ReflectionInsight(
                    category="strategy",
                    severity="warning",
                    description=f"Confidence miscalibration detected: avg error {avg_conf_error:.2f}",
                    evidence={
                        "avg_confidence_error": round(avg_conf_error, 3),
                    },
                    recommended_actions=[
                        "Adjust confidence estimator weights",
                        "Increase use of hedging language for uncertain domains",
                    ],
                )
            )

        # Tool utilization analysis
        tool_scores = [e.get("tool_accuracy", 0.5) for e in evals]
        low_tool = sum(1 for s in tool_scores if s < 0.5)
        if low_tool / max(len(evals), 1) > 0.3:
            insights.append(
                ReflectionInsight(
                    category="strategy",
                    severity="warning",
                    description=f"Tool failures in {low_tool}/{len(evals)} interactions",
                    evidence={
                        "low_tool_count": low_tool,
                        "total": len(evals),
                    },
                    recommended_actions=[
                        "Review tool definitions and error handling",
                        "Consider fallback strategies for tool failures",
                    ],
                )
            )

        return insights

    # ── Action Execution ─────────────────────────────────────────────

    async def _act_on_insights(
        self, insights: list[ReflectionInsight]
    ) -> list[str]:
        """Take automated actions based on reflection insights."""
        actions: list[str] = []

        for insight in insights:
            if insight.severity == "critical":
                # Critical insights trigger immediate goal creation
                if self.goal_manager:
                    for rec in insight.recommended_actions[:2]:
                        self.goal_manager.add_goal(
                            title=rec,
                            goal_type="optimization",
                            priority="high",
                            metadata={
                                "source": "reflection",
                                "insight": insight.description,
                            },
                        )
                        actions.append(f"Created goal: {rec}")

                # Update self-model
                if self.self_model and insight.category == "capability":
                    domain = insight.evidence.get("domain", "general")
                    self.self_model.record_outcome(
                        domain=domain,
                        success=False,
                        confidence=0.3,
                    )
                    actions.append(f"Updated self-model for domain: {domain}")

            elif insight.severity == "warning":
                # Warning insights create background goals
                if self.goal_manager and insight.recommended_actions:
                    self.goal_manager.add_goal(
                        title=insight.recommended_actions[0],
                        goal_type="learning",
                        priority="medium",
                        metadata={
                            "source": "reflection",
                            "insight": insight.description,
                        },
                    )
                    actions.append(f"Created goal: {insight.recommended_actions[0]}")

        return actions

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_sessions": self._session_count,
            "evaluation_history_size": len(self._evaluation_history),
        }

        if self._sessions:
            latest = self._sessions[-1]
            result["latest_session"] = {
                "session_id": latest.session_id,
                "timestamp": latest.timestamp,
                "insight_count": len(latest.insights),
                "actions_taken": len(latest.actions_taken),
            }
            result["recent_insights"] = [
                {"category": i.category, "severity": i.severity, "description": i.description}
                for i in latest.insights[:5]
            ]

        return result
