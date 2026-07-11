"""
System Decision Node (The Gatekeeper).

Subscribes to ``decision.evaluate``.
In the Global Workspace model, the LLM and Symbolic Solvers only pose "Thoughts."
This Node operates after the Workspace Blackboard has formed a consensus.
It uses the ActionPlanner to produce a structured ActionPlan, enforces tiered
safety checks, and dispatches the final command to the Agent Execution Layer
(Browser, Code Execution, Audio TTS, IoT, MCP, or directly to the User).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from typing import TYPE_CHECKING, Any

from hbllm.brain.evaluation.utility_calibrator import UtilityCalibrator
from hbllm.brain.evaluation.utility_engine import CognitiveUtilityEngine
from hbllm.brain.planning.action_planner import ActionPlanner
from hbllm.brain.planning.action_schema import ActionPlan, ActionType, RiskLevel
from hbllm.brain.snn.expression.models import ExpressionResult
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.core.provider_adapter import ProviderLLM
    from hbllm.brain.governance.policy_engine import PolicyEngine

logger = logging.getLogger(__name__)


class DecisionNode(Node):
    """
    Service node that separates generation from execution.
    Uses an ActionPlanner for structured intent→action mapping,
    tiered safety classification, and PolicyEngine governance
    before dispatching to the appropriate execution channel.
    """

    def __init__(
        self,
        node_id: str,
        llm: ProviderLLM | None = None,
        policy_engine: PolicyEngine | None = None,
        data_dir: str = "data",
    ) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.CORE)
        self.llm = llm  # LLMInterface instance
        self.policy_engine = policy_engine  # PolicyEngine instance
        self._planner = ActionPlanner()
        self.utility_engine = CognitiveUtilityEngine()
        self.calibrator = UtilityCalibrator(data_dir=data_dir)
        self.last_mode = "high"
        self._user_model: Any | None = None  # Optional UserModelEngine — set by factory

        # Expression-side Cognitive Stream (wired by factory if available)
        self.expression_stream: Any | None = None

        # Barge-in: track active generation for mid-response interruption
        self._active_generation_task: asyncio.Task[Any] | None = None
        self._cancel_event: asyncio.Event = asyncio.Event()

        # ── Multi-Rate Control State ─────────────────────────────────────
        # Smoothed band thresholds (updated every 7±1 decisions via γ-EMA)
        self.smoothed_high: float = 0.7
        self.smoothed_med: float = 0.3
        self.smoothed_low: float = 0.0
        # Decision counter for time-scale separation
        self.decision_count: int = 0
        # Cooling state (Schmitt trigger hysteresis)
        self.in_cooling: bool = False
        self.cooling_stable_ticks: int = 0

        # Stable-regime anchor (frozen when exiting cooling with high confidence)
        self._anchor_percentiles: tuple[float, float, float] = (0.7, 0.3, 0.0)

        # Jittered scheduling (anti-aliasing)
        self._next_smooth_tick: int = 7  # ~7 ± 1
        self._next_observe_tick: int = 13  # ~13 ± 2

        # Behavioral recovery tracking
        self._replan_count: int = 0  # replans in current observation window
        self._decisions_in_window: int = 0  # decisions since last observer tick

        # Stability scores
        self.S_ctrl: float = 0.0  # smoothed version of stability score used for routing stability

        # High confidence ticks accumulated in cooling state
        self.cooling_high_conf_ticks: int = 0

        # Bounded invariance (cooling frequency monitor)
        self._cooling_cycle_history: list[
            int
        ] = []  # tracks last 5 cycles (1 = in cooling, 0 = stable)
        self._stable_lock: bool = False  # forces cooling mode if system cycles too frequently

    async def on_start(self) -> None:
        """Subscribe to the decision evaluations from the Workspace."""
        logger.info("Starting DecisionNode (The Gatekeeper)")
        await self.bus.subscribe("decision.evaluate", self.evaluate_workspace_decision)

    async def on_stop(self) -> None:
        logger.info("Stopping DecisionNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ── Barge-in Support ──────────────────────────────────────────────────

    async def cancel_active_generation(self, reason: str = "new_input") -> bool:
        """Cancel any in-flight expression generation.

        Sets the cancel event (checked by ExpressionStream between fragments)
        and optionally cancels the asyncio task.

        Returns:
            True if a generation was actively cancelled, False if nothing was running.
        """
        if self._active_generation_task is None or self._active_generation_task.done():
            return False

        logger.info("[DecisionNode] Barge-in: cancelling active generation (reason=%s)", reason)

        # Signal the ExpressionStream to stop between fragments
        self._cancel_event.set()

        # Give the task a short grace period to exit cleanly
        try:
            await asyncio.wait_for(
                asyncio.shield(self._active_generation_task),
                timeout=1.0,
            )
        except (TimeoutError, asyncio.TimeoutError):
            # Force cancel if it didn't exit gracefully
            self._active_generation_task.cancel()
            try:
                await self._active_generation_task
            except (asyncio.CancelledError, Exception):
                pass
        except (asyncio.CancelledError, Exception):
            pass

        self._active_generation_task = None
        self._cancel_event.clear()

        # Publish interruption event on the bus
        if self._bus is not None:
            from hbllm.network.messages import Message as BusMessage
            from hbllm.network.messages import MessageType as BusMT

            await self._bus.publish(
                "sensory.output.interrupted",
                BusMessage(
                    type=BusMT.EVENT,
                    source_node_id=self.node_id,
                    topic="sensory.output.interrupted",
                    payload={"reason": reason},
                ),
            )

        return True

    # ── Main Entry Point ──────────────────────────────────────────────────

    async def evaluate_workspace_decision(self, message: Message) -> Message | None:
        """
        Triggered when the Workspace has reached a consensus.
        Flow: plan → safety gate (L1) → policy router (L2) → budget controller (L3) → execute.
        """
        # Barge-in: cancel any in-flight generation before processing new input
        await self.cancel_active_generation(reason="new_query")

        payload = message.payload
        original_query = payload.get("original_query", {})
        thought = payload.get("selected_thought", {})

        user_intent = original_query.get("intent", "answer")
        thought_type = thought.get("type", "intuition")
        confidence = thought.get("confidence", 0.0)
        content = thought.get("content") or ""

        logger.info(
            "[DecisionNode] Evaluating %s thought (Confidence: %s)...", thought_type, confidence
        )

        # ── 1. Plan the action ─────────────────────────────────────────────
        plan = self._planner.plan(
            intent=user_intent,
            thought_type=thought_type,
            content=content,
            confidence=confidence,
            original_query=original_query,
        )
        logger.info(
            "[DecisionNode] ActionPlan: %s (risk=%s)", plan.action_type.value, plan.risk_level.value
        )

        # ── 2. Level 1: Enforce safety (tiered) ─────────────────────────────
        if not await self._enforce_safety(plan, message):
            return None  # blocked by policy or safety classifier

        # ── 3. Level 2: Policy Router (Utility Arbitration) ─────────────────
        routed_successfully = await self._arbitrate_utility(plan, message, original_query)
        if not routed_successfully:
            return None

        # ── 4. Level 3: Budget Controller (Resource Limits) ──────────────────
        if not self._check_hard_limits(plan, message):
            await self._publish_output(
                message, "Execution blocked: Resource budget exceeded.", source="budget_controller"
            )
            return None

        # ── 5. Execute the action ──────────────────────────────────────────
        await self._execute_action(plan, message, original_query)

        return None

    async def _arbitrate_utility(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> bool:
        """
        Level 2 Policy Router (Multi-Rate Control System).

        Arbitrates action routing based on utility, load penalty, bootstrap status,
        hysteresis, and context-aware replanning.

        Time-scale separation:
        - Fast (every decision):  Safety gate, routing, dispatch.
        - Medium (every 5 decisions):  Threshold smoothing via γ-EMA.
        - Slow (every 10 decisions):  Lyapunov invariant V(t) and cooling state.
        """
        from datetime import datetime, timezone

        # ── 0. Increment decision counter (drives time-scale separation) ──
        self.decision_count += 1

        payload = message.payload
        thought = payload.get("selected_thought", {})
        user_intent = original_query.get("intent", "answer")
        thought_type = thought.get("type", "intuition")

        # 1. Z-Score Drift Detection (observation only — no routing mutation)
        if self.calibrator.detect_drift():
            logger.warning("[DecisionNode] Utility distribution drift detected (Z-score > 2.0)!")

        # 2. Extract utility parameters
        progress_score = (
            thought.get("progress_score")
            or thought.get("score")
            or thought.get("confidence")
            or 0.0
        )
        tokens_used = thought.get("tokens_used") or payload.get("tokens_used") or 0
        predicted_latency = (
            thought.get("predicted_latency")
            or thought.get("latency_ms")
            or payload.get("predicted_latency")
            or 100.0
        )
        risk_score = thought.get("risk_score") or payload.get("risk_score") or 0.0
        if not risk_score and plan.risk_level == RiskLevel.HIGH:
            risk_score = 1.0
        elif not risk_score and plan.risk_level == RiskLevel.MEDIUM:
            risk_score = 0.5

        # 3. CPU load penalty & Dual-Latency calculation
        cpu_percent = original_query.get("cpu_percent") or payload.get("cpu_percent")
        if cpu_percent is None:
            try:
                import psutil  # type: ignore[import-not-found]

                cpu_percent = psutil.cpu_percent(interval=None) or 20.0
            except Exception:
                cpu_percent = 20.0

        load_penalty = (1.0 / (1.0 + math.exp(-(cpu_percent - 80.0) / 10.0))) * 0.25
        latency = predicted_latency + load_penalty

        # 4. Calculate utility
        breakdown = self.utility_engine.calculate_utility(
            progress_score=progress_score,
            tokens_used=tokens_used,
            latency_ms=latency,
            risk_score=risk_score,
        )
        utility_score = breakdown.utility

        # 4a. Apply quadratic replanning penalty
        replan_depth = original_query.get("replan_depth", 0) or payload.get("replan_depth", 0)
        if replan_depth > 0:
            penalty = 0.05 * (replan_depth**2)
            utility_score = max(0.0, utility_score - penalty)

        # Record trace
        self.calibrator.record_trace(
            trace_id=message.correlation_id or f"trace_{time.time()}",
            decision_point=f"decision_node:{plan.action_type.value}",
            predicted_utility=utility_score,
            actual_outcome=progress_score,
        )

        # Track decisions in current observation window
        self._decisions_in_window += 1

        # ── Medium-Timescale: Threshold Smoothing (jittered ~7) ────
        if self.decision_count >= self._next_smooth_tick:
            self._next_smooth_tick = self.decision_count + 7 + random.randint(-1, 1)

            live_high, live_med, live_low = self.calibrator.get_utility_percentiles()

            # Adaptive blend: anchor_weight scales with smoothed instability S_ctrl
            # Base anchor weight of 0.3, scaling up to 0.9 under instability
            anchor_weight = max(0.3, min(0.9, 0.3 + self.S_ctrl))
            # If stable lock is active, completely rely on anchor
            if self._stable_lock:
                anchor_weight = 1.0

            live_weight = 1.0 - anchor_weight

            live_high = live_weight * live_high + anchor_weight * self._anchor_percentiles[0]
            live_med = live_weight * live_med + anchor_weight * self._anchor_percentiles[1]
            live_low = live_weight * live_low + anchor_weight * self._anchor_percentiles[2]

            gamma = 0.8
            self.smoothed_high = gamma * self.smoothed_high + (1 - gamma) * live_high
            self.smoothed_med = gamma * self.smoothed_med + (1 - gamma) * live_med
            self.smoothed_low = gamma * self.smoothed_low + (1 - gamma) * live_low

        # Use smoothed thresholds for routing
        high_threshold = self.smoothed_high
        med_threshold = self.smoothed_med
        low_threshold = self.smoothed_low

        # Check bootstrap
        readiness = self.calibrator.get_calibration_readiness()
        bootstrap_active = readiness.get("bootstrap_active", False)
        if bootstrap_active:
            high_threshold -= 0.3
            med_threshold -= 0.3
            low_threshold -= 0.3

        # ── Slow-Timescale: Lyapunov Invariant & Cooling (jittered ~13) ─
        if self.decision_count >= self._next_observe_tick:
            self._next_observe_tick = self.decision_count + 13 + random.randint(-2, 2)

            traces = self.calibrator.get_traces()
            recent = traces[:15] if len(traces) >= 15 else traces
            if len(recent) >= 2:
                signed_errors = [t.predicted_utility - t.actual_outcome for t in recent]
                abs_errors = [abs(e) for e in signed_errors]

                # Variance (noise energy)
                mean_abs = sum(abs_errors) / len(abs_errors)
                variance = sum((e - mean_abs) ** 2 for e in abs_errors) / len(abs_errors)

                # Bias (systematic error)
                bias = sum(signed_errors) / len(signed_errors)

                # Unified instability energy S_diag
                S_diag = variance + 2.0 * (bias**2)
            else:
                S_diag = 0.0

            # Smooth diagnostic score to get control score S_ctrl
            self.S_ctrl = 0.7 * self.S_ctrl + 0.3 * S_diag

            # Compute replan rate in the current window
            replan_rate = self._replan_count / max(1, self._decisions_in_window)
            self._replan_count = 0
            self._decisions_in_window = 0

            # ── Schmitt Trigger Gating ──
            if not self.in_cooling:
                if S_diag > 0.08:
                    self.in_cooling = True
                    self.cooling_stable_ticks = 0
                    self.cooling_high_conf_ticks = 0
                    logger.info(
                        "[DecisionNode] Entering cooling state (S_diag=%.4f > 0.08)", S_diag
                    )

                    # ── Bounded Invariance check ──
                    # Track decision_count of cooling entry
                    self._cooling_cycle_history.append(self.decision_count)
                    self._cooling_cycle_history = self._cooling_cycle_history[-3:]
                    if len(self._cooling_cycle_history) >= 3:
                        # If 3 entries occurred within 60 decisions, engage stable lock
                        if self._cooling_cycle_history[-1] - self._cooling_cycle_history[0] <= 60:
                            self._stable_lock = True
                            logger.warning(
                                "[DecisionNode] Stable-lock engaged: system is cycling too frequently!"
                            )
            else:
                # If stable lock is active, require a tighter threshold and more ticks to clear
                target_score = 0.02 if self._stable_lock else 0.04
                required_ticks = 6 if self._stable_lock else 3

                if S_diag < target_score and replan_rate < 0.3:
                    self.cooling_stable_ticks += 1
                    if S_diag < 0.03:
                        self.cooling_high_conf_ticks += 1
                else:
                    self.cooling_stable_ticks = 0
                    self.cooling_high_conf_ticks = 0

                if self.cooling_stable_ticks >= required_ticks:
                    self.in_cooling = False
                    if self._stable_lock:
                        self._cooling_cycle_history.clear()
                    self._stable_lock = False

                    # High-confidence anchor freeze
                    if self.cooling_high_conf_ticks >= required_ticks:
                        self._anchor_percentiles = (
                            self.smoothed_high,
                            self.smoothed_med,
                            self.smoothed_low,
                        )
                        logger.info(
                            "[DecisionNode] Exited cooling (high confidence) - anchor updated."
                        )
                    else:
                        logger.info(
                            "[DecisionNode] Exited cooling (marginal confidence) - anchor preserved."
                        )

                    self.cooling_stable_ticks = 0
                    self.cooling_high_conf_ticks = 0

        # 6. Apply Hysteresis for band switching
        epsilon = 0.05
        high_req = (
            (high_threshold - epsilon) if self.last_mode == "high" else (high_threshold + epsilon)
        )
        if utility_score >= high_req:
            mode = "high"
        else:
            med_req = (
                (med_threshold - epsilon)
                if self.last_mode in ("high", "medium")
                else (med_threshold + epsilon)
            )
            if utility_score >= med_req:
                mode = "medium"
            else:
                low_req = (
                    (low_threshold - epsilon)
                    if self.last_mode in ("high", "medium", "low")
                    else (low_threshold + epsilon)
                )
                if utility_score >= low_req:
                    mode = "low"
                else:
                    mode = "negative"

        # 7. Exploration Override check (blocked during cooling)
        exploration_override = original_query.get("exploration_override", False) or payload.get(
            "exploration_override", False
        )
        if mode == "negative" and exploration_override and not self.in_cooling:
            mode = "low"
            plan.metadata["exploration_mode"] = True

        self.last_mode = mode

        # ── Cooling state enforcement ────────────────────────────────────
        if self.in_cooling:
            plan.metadata["optimize_resources"] = True

        # Apply behaviors based on mode
        if mode == "high":
            return True
        elif mode == "medium":
            plan.metadata["optimize_resources"] = True
            return True
        elif mode == "low" or plan.metadata.get("exploration_mode"):
            plan.metadata["exploration_mode"] = True

            # Enforce exploration TTL decay
            exploration_ttl = (
                original_query.get("exploration_ttl") or payload.get("exploration_ttl") or 300.0
            )
            elapsed_seconds = (datetime.now(timezone.utc) - message.timestamp).total_seconds()
            if elapsed_seconds > exploration_ttl:
                logger.warning(
                    "[DecisionNode] Exploration task expired: elapsed %s > TTL %s",
                    elapsed_seconds,
                    exploration_ttl,
                )
                await self._publish_output(
                    message,
                    f"Exploration task expired and was discarded (TTL: {exploration_ttl}s).",
                    source="exploration_timeout",
                )
                return False

            # Enforce restricted tools in exploration mode
            if plan.action_type not in (
                ActionType.CODE_EXECUTION,
                ActionType.WEB_SEARCH,
                ActionType.TEXT_RESPONSE,
                ActionType.CLARIFY,
            ):
                logger.warning(
                    "[DecisionNode] Tool %s restricted in exploration mode", plan.action_type
                )
                await self._publish_output(
                    message,
                    f"Tool {plan.action_type.value} is restricted in exploration mode.",
                    source="exploration_restriction",
                )
                return False

            return True

        # Mode is negative — replan_depth already extracted above (step 4a)
        self._replan_count += 1
        max_depth = self._get_max_replan_depth(plan, user_intent, thought_type)

        new_replan_depth = replan_depth + 1
        if new_replan_depth <= max_depth:
            logger.info(
                "[DecisionNode] Utility below threshold, routing back to planner for replanning (replan_depth: %s/%s)",
                new_replan_depth,
                max_depth,
            )
            # Send message to planner.decompose with tighter budget
            replan_payload = {
                "text": plan.content,
                "original_query": {
                    **original_query,
                    "replan_depth": new_replan_depth,
                },
                "thought_budget": {
                    "max_tokens": 4096,
                    "max_time_ms": 7500.0,
                    "max_branches": 10,
                },
            }
            replan_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="planner.decompose",
                payload=replan_payload,
                correlation_id=message.correlation_id,
            )
            if self._bus is not None:
                await self.bus.publish("planner.decompose", replan_msg)
            return False
        else:
            logger.warning(
                "[DecisionNode] Re-plan depth exceeded (%s/%s). Deferring task.",
                new_replan_depth,
                max_depth,
            )
            defer_payload = {
                "original_query": original_query,
                "thought": thought,
                "plan": {
                    "action_type": plan.action_type.value,
                    "content": plan.content,
                    "metadata": plan.metadata,
                },
                "reason": "replan_depth_exceeded",
                "replan_depth": replan_depth,
            }
            defer_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="tasks.deferred",
                payload=defer_payload,
                correlation_id=message.correlation_id,
            )
            if self._bus is not None:
                await self.bus.publish("tasks.deferred", defer_msg)
            await self._publish_output(
                message,
                "Task deferred due to low utility and exceeded replanning depth.",
                source="task_deferral",
            )
            return False

    def _check_hard_limits(self, plan: ActionPlan, message: Message) -> bool:
        """
        Level 3 Budget Controller.
        Enforces hard token and memory limits.
        """
        payload = message.payload
        original_query = payload.get("original_query", {})
        thought = payload.get("selected_thought", {})

        # Get budget config
        thought_budget = payload.get("thought_budget", {}) or original_query.get(
            "thought_budget", {}
        )
        max_tokens = thought_budget.get("max_tokens", 8192)

        # Halve budget if medium mode
        if self.last_mode == "medium":
            max_tokens = max_tokens // 2
        elif self.last_mode == "low" or plan.metadata.get("exploration_mode"):
            max_tokens = 1024

        tokens_used = thought.get("tokens_used") or payload.get("tokens_used") or 0
        if tokens_used > max_tokens:
            logger.warning(
                "[DecisionNode] Budget check failed: tokens_used (%s) > max_tokens (%s)",
                tokens_used,
                max_tokens,
            )
            return False

        # Check system memory percent
        memory_limit = thought_budget.get("max_memory_percent", 90.0)
        try:
            import psutil  # type: ignore[import-not-found]

            mem_percent = psutil.virtual_memory().percent
            if mem_percent > memory_limit:
                logger.warning(
                    "[DecisionNode] Memory limit exceeded: %s%% > %s%%", mem_percent, memory_limit
                )
                return False
        except Exception:
            logger.debug("Memory limit check failed (psutil unavailable)", exc_info=True)

        return True

    def _get_max_replan_depth(self, plan: ActionPlan, intent: str, thought_type: str) -> int:
        category = (plan.action_type.value or "").lower()
        intent_l = intent.lower()
        thought_type_l = thought_type.lower()

        # Check coding/optimization
        if (
            any(x in category for x in ("code", "shell"))
            or any(x in intent_l for x in ("code", "optimize", "coding"))
            or any(x in thought_type_l for x in ("code", "optimize", "coding"))
        ):
            return 4
        # Check search
        if "search" in category or "search" in intent_l or "search" in thought_type_l:
            return 2
        return 3

    # ── Safety Enforcement (Tiered) ───────────────────────────────────────

    async def _enforce_safety(self, plan: ActionPlan, message: Message) -> bool:
        """
        Apply tiered safety checks based on the action's risk level.

        Returns ``True`` if the action is approved, ``False`` if blocked.

        - LOW risk (text, audio, clarify): PolicyEngine only.
        - MEDIUM risk (web search, API): PolicyEngine only.
        - HIGH risk (code, IoT, MCP): PolicyEngine + LLM safety classifier.
        """
        content = plan.content

        # ── PolicyEngine (all risk levels) ─────────────────────────────────
        if self.policy_engine:
            try:
                result = self.policy_engine.evaluate(
                    text=content,
                    tenant_id=message.tenant_id or "default",
                    domain="",
                )
                if not result.passed:
                    violations = "; ".join(result.violations)
                    logger.warning("[DecisionNode] Policy violation: %s", violations)
                    await self._publish_output(
                        message,
                        f"I cannot fulfill this request — policy violation: {violations}",
                    )
                    return False
                # Apply any policy transformations (e.g., appended disclaimers)
                plan.content = result.modified_text
                if result.warnings:
                    logger.info("[DecisionNode] Policy warnings: %s", result.warnings)
            except Exception as e:
                logger.warning("[DecisionNode] PolicyEngine evaluation failed: %s", e)

        # ── LLM Safety Classifier (HIGH risk only) ─────────────────────────
        if plan.requires_safety_llm and self.llm:
            try:
                safety = await self.llm.generate_json(
                    f"You are a safety classifier for an AI system. Evaluate whether the following "
                    f"response content is safe to present to a user. Check for:\n"
                    f"- Harmful or dangerous instructions\n"
                    f"- Personal information exposure\n"
                    f"- Illegal activity guidance\n"
                    f"- Explicit or violent content\n\n"
                    f'Content: "{content[:500]}"\n\n'
                    f'Output JSON: {{"safe": true/false, "reason": "brief explanation"}}'
                )

                if not safety.get("safe", True):
                    reason = safety.get("reason", "Content flagged by safety classifier")
                    logger.warning("[DecisionNode] Thought rejected: %s", reason)
                    await self._publish_output(
                        message,
                        f"I cannot fulfill this request due to safety constraints: {reason}",
                    )
                    return False
            except Exception as e:
                logger.warning(
                    "[DecisionNode] Safety classification failed, proceeding cautiously: %s", e
                )

        return True

    # ── Action Execution (Dispatch) ───────────────────────────────────────

    async def _execute_action(
        self,
        plan: ActionPlan,
        message: Message,
        original_query: dict[str, Any],
    ) -> None:
        """Dispatch the approved ActionPlan to the appropriate execution channel."""

        dispatch = {
            ActionType.AUDIO_OUTPUT: self._exec_audio_output,
            ActionType.CODE_EXECUTION: self._exec_code_execution,
            ActionType.WEB_SEARCH: self._exec_web_search,
            ActionType.API_CALL: self._exec_api_call,
            ActionType.IOT_COMMAND: self._exec_iot_command,
            ActionType.MCP_TOOL: self._exec_mcp_tool,
            ActionType.CLARIFY: self._exec_clarify,
            ActionType.TEXT_RESPONSE: self._exec_text_response,
            ActionType.SHELL_EXECUTION: self._exec_shell_execution,
        }

        handler = dispatch.get(plan.action_type, self._exec_text_response)
        await handler(plan, message, original_query)

        # Record experience for salience detection
        await self.bus.publish(
            "system.experience",
            self._make_msg(
                message,
                "system.experience",
                {
                    "text": plan.content[:500],
                    "intent": original_query.get("intent", "answer"),
                    "thought_type": original_query.get("thought_type", ""),
                    "action_type": plan.action_type.value,
                },
            ),
        )

    # ── Individual Execution Handlers ─────────────────────────────────────

    async def _exec_text_response(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Publish plain text to the user interface.

        If the ExpressionStream is wired and comprehension data exists in
        the payload, runs structured thought-by-thought generation.
        Falls back to direct output if expression is unavailable or fails.
        """
        logger.info("[DecisionNode] Dispatching to User Interface.")
        thought_type = plan.metadata.get("thought_type", "intuition_general")

        # ── Expression-side Cognitive Stream (Layer 5) ────────────────────
        comprehension_data = original_query.get("comprehension")
        if self.expression_stream is not None and comprehension_data is not None:
            try:
                # Reset cancel event for this generation
                self._cancel_event.clear()

                # Run as tracked task for barge-in support
                coro = self._run_expression_stream(
                    content=plan.content,
                    comprehension_data=comprehension_data,
                    original_query=original_query,
                    correlation_id=message.correlation_id,
                    cancel_event=self._cancel_event,
                )
                self._active_generation_task = asyncio.create_task(coro)
                expression_result = await self._active_generation_task
                self._active_generation_task = None

                if expression_result is not None and expression_result.text:
                    plan.metadata["expression_reward"] = expression_result.mean_reward
                    plan.metadata["expression_thoughts"] = expression_result.thought_count
                    plan.metadata["expression_revisions"] = expression_result.revision_count
                    plan.metadata["interrupted"] = expression_result.metadata.get(
                        "interrupted", False
                    )
                    await self._publish_output(message, expression_result.text, source=thought_type)
                    return
            except Exception as e:
                logger.warning(
                    "[DecisionNode] ExpressionStream failed, falling back to direct: %s", e
                )

        await self._publish_output(message, plan.content, source=thought_type)

    async def _exec_audio_output(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch text to the Audio Output Node for TTS."""
        logger.info("[DecisionNode] Dispatching to AudioOutputNode.")
        await self.bus.publish(
            "sensory.audio.out",
            self._make_msg(message, "sensory.audio.out", {"text": plan.content}),
        )
        await self._publish_output(message, plan.content, source="audio_speak")

    async def _exec_code_execution(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Execute Python code in the sandbox and synthesize results."""
        logger.info("[DecisionNode] Dispatching to ExecutionNode Sandbox.")
        code = plan.content

        # Clamp execution parameters if exploration_mode is True
        exec_payload = {"code": code}
        if plan.metadata.get("exploration_mode"):
            exec_payload["max_tokens"] = 1024
            exec_payload["max_branches"] = 2
            exec_payload["sandbox_only"] = True

        try:
            exec_msg = self._make_msg(
                message, "task.execute.python", exec_payload, MessageType.QUERY
            )
            exec_resp = await self.bus.request("task.execute.python", exec_msg, timeout=15.0)
            if exec_resp.type == MessageType.ERROR:
                result_text = f"Execution failed: {exec_resp.payload.get('error')}"
            else:
                stdout = exec_resp.payload.get("output", "")
                stderr = exec_resp.payload.get("error", "")
                result_text = f"STDOUT:\n{stdout}"
                if stderr:
                    result_text += f"\nSTDERR:\n{stderr}"
        except Exception as e:
            result_text = f"Execution timeout or error: {e}"

        # Synthesize or directly present the result
        final_text = await self._synthesize_result(
            result_text,
            original_query,
            synthesis_prompt=(
                "You are a helpful assistant. Present the results of the Python code execution "
                "to the user. If the code printed the answer, explain it. If it failed, explain why.\n\n"
                f"User Query: {original_query.get('text', '')}\n\n"
                f"Code Executed:\n{code}\n\n"
                f"Execution Output:\n{result_text}"
            ),
            fallback_prefix="Executed code. Result:",
        )
        await self._publish_output(message, final_text, source="code_execution")

    async def _exec_web_search(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Execute a web search, optionally resolving vague queries from context."""
        logger.info("[DecisionNode] Dispatching to BrowserNode.")
        query = plan.content

        # Resolve vague queries using conversation history
        if plan.metadata.get("needs_context_resolution", False):
            query = await self._resolve_vague_query(query, message)

        max_results = plan.metadata.get("max_results", 3)
        if plan.metadata.get("exploration_mode"):
            # Clamp results for exploration mode
            max_results = min(max_results, 2)

        try:
            search_msg = self._make_msg(
                message,
                "task.execute.search",
                {"query": query, "max_results": max_results},
                MessageType.QUERY,
            )
            search_resp = await self.bus.request("task.execute.search", search_msg, timeout=15.0)
            if search_resp.type == MessageType.ERROR:
                result_text = f"Search failed: {search_resp.payload.get('error')}"
            else:
                result_text = search_resp.payload.get("text", "No search results found.")
        except Exception as e:
            result_text = f"Search timeout or error: {e}"

        final_text = await self._synthesize_result(
            result_text,
            original_query,
            synthesis_prompt=(
                "You are a helpful assistant. Synthesize a comprehensive, direct response to the user's query "
                "based on the following real-time search results. Cite sources/URLs if available.\n\n"
                f"User Query: {original_query.get('text', query)}\n\n"
                f"Search Results:\n{result_text}"
            ),
            fallback_prefix="Search results:",
        )
        await self._publish_output(message, final_text, source="browser")

    async def _exec_api_call(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch to the API execution node."""
        logger.info("[DecisionNode] Dispatching to ApiNode.")
        intent = plan.metadata.get("intent", "tool_synthesis")

        try:
            api_msg = self._make_msg(
                message,
                "task.execute.api",
                {"schema": plan.content, "intent": intent},
                MessageType.QUERY,
            )
            api_resp = await self.bus.request("task.execute.api", api_msg, timeout=15.0)
            if api_resp.type == MessageType.ERROR:
                result_text = f"API execution failed: {api_resp.payload.get('error')}"
            else:
                result_text = api_resp.payload.get("text", str(api_resp.payload))
        except Exception as e:
            result_text = f"API timeout or error: {e}"

        await self._publish_output(message, result_text, source="api_execution")

    async def _exec_iot_command(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch an IoT command via MQTT."""
        logger.info("[DecisionNode] Dispatching to IoT MQTT Node.")
        iot_topic = plan.metadata.get("iot_topic", "hbllm/command")

        await self.bus.publish(
            "iot.publish",
            self._make_msg(
                message,
                "iot.publish",
                {"topic": iot_topic, "payload": plan.content},
            ),
        )
        await self._publish_output(
            message,
            f"Dispatched IoT command to topic: {iot_topic}",
            source="iot",
        )

    async def _exec_mcp_tool(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Dispatch a tool call (MCP tool or native plugin tool) and synthesize results."""
        logger.info("[DecisionNode] Dispatching to Tool Execution.")
        tool_name = plan.metadata.get("tool_name", "")
        arguments = plan.metadata.get("arguments", {})

        try:
            # Try unified tool topic first: action.tool.{tool_name}
            topic = f"action.tool.{tool_name}"
            tool_msg = self._make_msg(
                message,
                topic,
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                },
                MessageType.QUERY,
            )
            tool_resp = await self.bus.request(topic, tool_msg, timeout=15.0)
            if tool_resp.type == MessageType.ERROR:
                result_text = f"Tool call failed: {tool_resp.payload.get('error')}"
            else:
                if "output" in tool_resp.payload:
                    result_text = str(tool_resp.payload["output"])
                elif "text" in tool_resp.payload:
                    result_text = str(tool_resp.payload["text"])
                else:
                    result_text = str(tool_resp.payload)
        except Exception as e:
            logger.warning("Unified tool execution failed, trying fallback: %s", e)
            try:
                # Fallback to mcp.tool_call
                fallback_msg = self._make_msg(
                    message,
                    "mcp.tool_call",
                    {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "content": plan.content,
                    },
                    MessageType.QUERY,
                )
                fallback_resp = await self.bus.request("mcp.tool_call", fallback_msg, timeout=15.0)
                if fallback_resp.type == MessageType.ERROR:
                    result_text = f"Tool call failed: {fallback_resp.payload.get('error')}"
                else:
                    result_text = fallback_resp.payload.get("text", str(fallback_resp.payload))
            except Exception as ex:
                result_text = f"Tool execution error: {ex}"

        # Synthesize result back to the user via LLM
        final_text = await self._synthesize_result(
            result_text,
            original_query,
            synthesis_prompt=(
                "You are a helpful assistant. Present the results of the tool execution "
                "to the user. Synthesize a clean, direct answer based on the output.\n\n"
                f"User Query: {original_query.get('text', '')}\n\n"
                f"Tool Called: {tool_name}\n"
                f"Arguments: {json.dumps(arguments)}\n\n"
                f"Tool Output:\n{result_text}"
            ),
            fallback_prefix=f"Tool '{tool_name}' executed. Result:",
        )
        await self._publish_output(message, final_text, source="tool_execution")

    async def _exec_clarify(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Ask the user to clarify when confidence is too low."""
        logger.info(
            "[DecisionNode] Confidence too low (%.2f), requesting clarification.",
            plan.metadata.get("confidence", 0.0),
        )
        await self._publish_output(
            message,
            "I'm not confident enough in my understanding of your request. Could you please clarify or provide more details?",
            source="clarify",
        )

    # ── Shared Helpers ────────────────────────────────────────────────────

    async def _resolve_vague_query(self, query: str, message: Message) -> str:
        """Resolve a vague search query using conversation history from episodic memory."""
        try:
            tenant_id = message.tenant_id or "default"
            session_id = message.session_id or "default"

            req_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=tenant_id,
                topic="memory.retrieve_recent",
                payload={"session_id": session_id, "limit": 6, "tenant_id": tenant_id},
            )
            resp = await self.bus.request("memory.retrieve_recent", req_msg, timeout=3.0)
            turns = resp.payload.get("turns", [])

            if turns and self.llm:
                history_str = ""
                for turn in turns:
                    role = turn.get("role", "user")
                    content_str = turn.get("content", "")
                    history_str += f"{role.capitalize()}: {content_str}\n"

                resolver_prompt = (
                    "You are an AI assistant helping to resolve search queries. "
                    "The user has requested to search for something, but their query is vague or context-dependent (e.g. 'search it'). "
                    "Based on the recent conversation history, determine what specific, clear search query they intend. "
                    "Do not output any introductory or concluding remarks. Respond ONLY with the resolved search query.\n\n"
                    f"Conversation History:\n{history_str}\n"
                    f"User Request: {query}\n\n"
                    "Resolved Search Query:"
                )
                resolved_query = await self.llm.generate(resolver_prompt)
                resolved_query = resolved_query.strip().strip("\"'")
                if resolved_query:
                    logger.info(
                        "[DecisionNode] Resolved vague query '%s' to '%s'", query, resolved_query
                    )
                    return resolved_query
        except Exception as e:
            logger.warning("[DecisionNode] Failed to resolve vague query context: %s", e)

        return query

    async def _synthesize_result(
        self,
        result_text: str,
        original_query: dict[str, Any],
        synthesis_prompt: str,
        fallback_prefix: str,
    ) -> str:
        """Use the LLM to synthesize raw results into a user-friendly response."""
        if self.llm:
            try:
                return await self.llm.generate(synthesis_prompt)
            except Exception as e:
                logger.warning("Synthesis failed: %s", e)
        return f"{fallback_prefix}\n{result_text}"

    def _make_msg(
        self,
        original: Message,
        topic: str,
        payload: dict[str, Any],
        msg_type: MessageType = MessageType.EVENT,
    ) -> Message:
        """Create a new Message inheriting identifiers from the original."""
        return Message(
            type=msg_type,
            source_node_id=self.node_id,
            tenant_id=original.tenant_id,
            session_id=original.session_id,
            topic=topic,
            payload=payload,
            correlation_id=original.correlation_id,
        )

    async def _run_expression_stream(
        self,
        content: str,
        comprehension_data: dict[str, Any],
        original_query: dict[str, Any],
        correlation_id: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> ExpressionResult | None:
        """Run the expression-side Cognitive Stream on a text response.

        Reconstructs a lightweight UnderstandingState from the comprehension
        payload (injected by RouterNode) and feeds it through the
        ExpressionStream pipeline.

        Args:
            content: The raw text response from workspace consensus.
            comprehension_data: The ``comprehension`` dict from the payload
                (contains concepts, memories, salience_peak).
            original_query: The original query payload.
            correlation_id: Optional correlation ID for fragment streaming.

        Returns:
            ExpressionResult if successful, None if skipped/failed.
        """
        from hbllm.brain.snn.comprehension.models import (
            ActivatedMemory,
            ComprehensionUnit,
            UnderstandingState,
        )

        concepts_raw = comprehension_data.get("concepts", [])
        if not concepts_raw:
            return None

        # Reconstruct UnderstandingState from the serialized payload
        import numpy as np

        concepts = []
        for c in concepts_raw:
            concepts.append(
                ComprehensionUnit(
                    text=c.get("text", ""),
                    embedding=np.zeros(384),  # placeholder — not needed for planning
                    salience=float(c.get("salience", 1.0)),
                    domain_activation=c.get("domains", {}),
                    channel_metadata=c.get("channels", {}),
                    activated_memories=[],
                )
            )

        memories_raw = comprehension_data.get("memories", [])
        all_memories = [
            ActivatedMemory(
                id=m.get("id", ""),
                content=m.get("content", ""),
                score=0.5,
            )
            for m in memories_raw
        ]

        # Distribute memories to concepts that mention related content
        for mem in all_memories:
            if concepts:
                # Assign to first concept as a simple heuristic
                concepts[0].activated_memories.append(mem)

        understanding = UnderstandingState(
            concepts=concepts,
            domain_activations={d: s for c in concepts for d, s in c.domain_activation.items()},
            all_memories=all_memories,
            salience_map=[c.salience for c in concepts],
        )

        query_text = original_query.get("text", "")

        # Wire fragment streaming callback if bus is available
        saved_on_fragment = self.expression_stream.on_fragment
        if self._bus is not None and correlation_id:
            from hbllm.brain.snn.expression.models import ThoughtFragment
            from hbllm.network.messages import Message as BusMessage
            from hbllm.network.messages import MessageType as BusMT

            async def _emit_fragment(fragment: ThoughtFragment) -> None:
                """Publish each fragment to the bus for real-time SSE streaming."""
                await self._bus.publish(
                    "sensory.output.fragment",
                    BusMessage(
                        type=BusMT.EVENT,
                        source_node_id=self.node_id,
                        topic="sensory.output.fragment",
                        payload={
                            "text": fragment.text,
                            "goal_id": fragment.goal_id,
                            "reward_score": fragment.reward_score,
                            "fragment_index": len(
                                self.expression_stream.on_fragment.__dict__.get("_count", [])
                            )
                            if hasattr(self.expression_stream.on_fragment, "__dict__")
                            else 0,
                        },
                        correlation_id=correlation_id,
                    ),
                )

            self.expression_stream.on_fragment = _emit_fragment

        try:
            result = await self.expression_stream.express(
                understanding=understanding,
                base_thought=content,
                original_query=query_text,
                cancel_event=cancel_event,
            )
        finally:
            # Restore original callback
            self.expression_stream.on_fragment = saved_on_fragment

        logger.info(
            "[DecisionNode] ExpressionStream: %d thoughts, mean_reward=%.2f, revisions=%d",
            result.thought_count,
            result.mean_reward,
            result.revision_count,
        )

        # Trigger batch SNN training if enough examples have accumulated
        prm_trainer = getattr(self, "_prm_trainer", None)
        if prm_trainer is not None:
            try:
                metrics = prm_trainer.maybe_train()
                if metrics is not None:
                    logger.info(
                        "[DecisionNode] PRMTrainer batch: acc=%.1f%%, Δw=%.4f",
                        metrics.accuracy * 100,
                        metrics.mean_weight_delta,
                    )
            except Exception as e:
                logger.debug("[DecisionNode] PRMTrainer batch failed (non-fatal): %s", e)

        return result

    async def _publish_output(self, message: Message, text: str, source: str = "decision") -> None:
        """Publish a response to ``sensory.output``."""
        if self._bus is not None:
            await self.bus.publish(
                "sensory.output",
                self._make_msg(message, "sensory.output", {"text": text, "source": source}),
            )
        else:
            logger.warning(
                "[DecisionNode] Node not started. Sensory output skipped: %s (source: %s)",
                text,
                source,
            )

    async def _exec_shell_execution(
        self, plan: ActionPlan, message: Message, original_query: dict[str, Any]
    ) -> None:
        """Execute a shell command via HostShellNode and synthesize results."""
        logger.info("[DecisionNode] Dispatching to HostShellNode.")
        command = plan.content

        try:
            shell_msg = self._make_msg(
                message, "action.execute_shell", {"command": command}, MessageType.QUERY
            )
            shell_resp = await self.bus.request("action.execute_shell", shell_msg, timeout=15.0)
            if shell_resp.type == MessageType.ERROR:
                result_text = f"Command failed: {shell_resp.payload.get('error')}"
            else:
                status = shell_resp.payload.get("status")
                stdout = shell_resp.payload.get("output", "")
                stderr = shell_resp.payload.get("error", "")
                exit_code = shell_resp.payload.get("exit_code", -1)

                if status == "SUCCESS":
                    result_text = f"STDOUT:\n{stdout}"
                    if stderr:
                        result_text += f"\nSTDERR:\n{stderr}"
                else:
                    result_text = f"Command exited with code {exit_code}.\n"
                    if stdout:
                        result_text += f"STDOUT:\n{stdout}\n"
                    if stderr:
                        result_text += f"STDERR:\n{stderr}"
        except Exception as e:
            result_text = f"Command timeout or error: {e}"

        final_text = await self._synthesize_result(
            result_text,
            original_query,
            synthesis_prompt=(
                "You are a helpful assistant. Present the results of the shell command execution "
                "to the user. If the command printed output, explain it. If it failed, explain why.\n\n"
                f"User Query: {original_query.get('text', '')}\n\n"
                f"Command Executed:\n{command}\n\n"
                f"Execution Output:\n{result_text}"
            ),
            fallback_prefix="Executed command. Result:",
        )
        await self._publish_output(message, final_text, source="shell_execution")
