"""ExecutiveCortex — unified persistent cognitive kernel controller.

Orchestrates the cognitive workspace via the CognitiveState blackboard.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.brain.autonomy.task_graph import Goal, GoalStatus, TaskPriority
from hbllm.brain.cognitive_state import (
    CognitiveBudget as StateCognitiveBudget,
)
from hbllm.brain.cognitive_state import (
    CognitivePolicy,
    CognitiveState,
)
from hbllm.brain.intentional_workspace import IntentionalWorkspace
from hbllm.brain.self_model import SelfModel
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class CognitiveExecutiveController(Node):
    """The persistent prefrontal cortex orchestrator.

    Does not execute reasoning directly. Instantiates CognitiveState, configures
    hierarchical policies, and tracks workspace progression reactively.
    """

    def __init__(
        self,
        node_id: str,
        intentional_workspace: IntentionalWorkspace | None = None,
        self_model: SelfModel | None = None,
    ) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.CORE)
        self.intentional_workspace = intentional_workspace or IntentionalWorkspace()
        self.self_model = self_model or SelfModel()

        # State log for event sourcing / state history
        self.state_history: dict[str, list[CognitiveState]] = {}
        # Active sessions tracking correlation_id -> current CognitiveState
        self.active_states: dict[str, CognitiveState] = {}

        self._loop_task: asyncio.Task[None] | None = None
        self._running = False

    async def on_start(self) -> None:
        """Subscribe to goal and workspace events and start the persistent loop."""
        logger.info("Starting CognitiveExecutiveController (Kernel Layer)")
        self._running = True

        # Subscribe to executive cortex triggers
        await self.bus.subscribe("workspace.cognition.goal", self.handle_new_goal)
        await self.bus.subscribe("workspace.cognition.state_change", self.handle_state_change)

        # Persistent background monitoring loop
        self._loop_task = asyncio.create_task(self._executive_loop())

    async def on_stop(self) -> None:
        """Gracefully stop the persistent executive loop."""
        logger.info("Stopping CognitiveExecutiveController")
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def handle_message(self, message: Message) -> Message | None:
        return None

    # ─── Goals Agenda Handler ────────────────────────────────────────

    async def handle_new_goal(self, message: Message) -> Message | None:
        """Ingests a new goal into the Intentional Workspace and initializes cognition."""
        payload = message.payload
        goal_id = payload.get("goal_id") or f"goal_{int(time.time())}"

        metadata = payload.get("metadata", {})
        if "domain" not in metadata and "domain" in payload:
            metadata["domain"] = payload["domain"]

        goal = Goal(
            goal_id=goal_id,
            tenant_id=message.tenant_id or "default",
            name=payload.get("name", "Unnamed Goal"),
            description=payload.get("description", ""),
            status=GoalStatus.ACTIVE,
            priority=TaskPriority(payload.get("priority", "normal")),
            metadata=metadata,
        )

        # 1. Store in the Intentional Workspace agenda
        self.intentional_workspace.add_goal(goal)

        # 2. Formulate Hierarchical Policy & Budget via Bayesian SelfModel
        domain = payload.get("domain", "general")
        policy = self._formulate_policy(domain)

        # 3. Create initial CognitiveState (Version 1)
        state = CognitiveState(goal=goal, policy=policy)
        correlation_id = message.correlation_id or goal_id

        self.active_states[correlation_id] = state
        self.state_history[correlation_id] = [state]

        logger.info(
            "[Executive] Initialized CognitiveState v1 for goal '%s' (domain: %s, policy: %s)",
            goal.name,
            domain,
            policy.reasoning_strategy,
        )

        # 4. Publish initial state to the Event Bus
        await self.bus.publish(
            "workspace.cognition.state_change",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                topic="workspace.cognition.state_change",
                correlation_id=correlation_id,
                payload={"state": state.to_dict()},
            ),
        )
        return None

    # ─── Cognitive Event Sourcing & Transformation ─────────────────────

    async def handle_state_change(self, message: Message) -> Message | None:
        """Observe state updates from planning/simulation/evaluation services."""
        correlation_id = message.correlation_id
        if not correlation_id or correlation_id not in self.active_states:
            return None

        state_dict = message.payload.get("state")
        if not state_dict:
            return None

        # Reconstruct updated state
        incoming_state = self._reconstruct_state(state_dict)
        current_state = self.active_states[correlation_id]

        # Ignore stale state versions
        if incoming_state.version <= current_state.version:
            return None

        # Append to event sourced state log
        self.active_states[correlation_id] = incoming_state
        self.state_history[correlation_id].append(incoming_state)

        logger.debug(
            "[Executive] State updated to version %d (id: %s, parent: %s)",
            incoming_state.version,
            incoming_state.state_id,
            incoming_state.parent_state_id,
        )

        # Check if the Goal has completed or failed (terminal state)
        if incoming_state.goal.status in (GoalStatus.COMPLETED, GoalStatus.FAILED):
            await self._finalize_cognition(correlation_id, incoming_state)

        return None

    # ─── Persistent Executive Loop ───────────────────────────────────

    async def _executive_loop(self) -> None:
        """Continuous background execution function monitoring active agendas."""
        while self._running:
            try:
                await asyncio.sleep(5.0)  # Keep periodic polling light
                active_goals = self.intentional_workspace.get_active_goals()
                if active_goals:
                    logger.debug(
                        "[Executive Loop] Tracking %d active goals in workspace", len(active_goals)
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("[Executive Loop] Error in background monitoring: %s", e)

    # ─── Policy Overrides & Bayesian Formulation ─────────────────────

    def _formulate_policy(self, domain: str) -> CognitivePolicy:
        """Formulate a hierarchical CognitivePolicy based on SelfModel expertise."""
        # Query Bayesian recommendation parameters from self-model
        try:
            capabilities = self.self_model.get_metrics()
            weaknesses = capabilities.get("weaknesses", [])
        except Exception:
            weaknesses = []

        # Determine strategy & budgets
        if domain in weaknesses:
            # Low competence -> High verification, deep simulation, larger model
            budget = StateCognitiveBudget(
                attention_budget=1.0,
                memory_budget=20,
                simulation_budget=10,
                reasoning_budget=2000,
                verification_budget=5,
                planning_budget=60.0,
                tool_budget=8,
            )
            return CognitivePolicy(
                reasoning_strategy="GoT",  # Graph-of-Thought for complex/weak areas
                simulation_depth=2,
                verification_budget=4,
                retrieval_budget=10,
                planner_type="graph",
                model_selection="large",
                budget=budget,
            )
        else:
            # High competence / default -> standard CoT, lower verification loops
            budget = StateCognitiveBudget(
                attention_budget=0.8,
                memory_budget=10,
                simulation_budget=3,
                reasoning_budget=1000,
                verification_budget=2,
                planning_budget=30.0,
                tool_budget=4,
            )
            return CognitivePolicy(
                reasoning_strategy="CoT",
                simulation_depth=1,
                verification_budget=2,
                retrieval_budget=5,
                planner_type="chain",
                model_selection="default",
                budget=budget,
            )

    # ─── Post-Cognition Finalization & Learning ──────────────────────

    async def _finalize_cognition(self, correlation_id: str, state: CognitiveState) -> None:
        """Logs outcomes, evaluates strategy success, and updates SelfModel."""
        success = state.goal.status == GoalStatus.COMPLETED
        duration = time.time() - state.created_at

        logger.info(
            "[Executive] Finalizing goal '%s' (success: %s, duration: %.2fs)",
            state.goal.name,
            success,
            duration,
        )

        # Update the Intentional Workspace persistence status
        self.intentional_workspace.update_goal_status(state.goal.goal_id, state.goal.status)

        # Record outcomes in the SelfModel
        try:
            # Domain-level logging
            domain = state.goal.metadata.get("domain", "general")
            self.self_model.record_outcome(
                domain=domain,
                success=success,
                confidence=state.confidence,
                latency_ms=duration * 1000.0,
            )
        except Exception as e:
            logger.error("[Executive] Failed to log outcome to SelfModel: %s", e)

        # Clean active state records
        if correlation_id in self.active_states:
            del self.active_states[correlation_id]

    # ─── Helper Reconstruction ───────────────────────────────────────

    def _reconstruct_state(self, d: dict[str, Any]) -> CognitiveState:
        """Utility to deserialize a CognitiveState from event payload."""
        goal_dict = d.get("goal", {})
        goal = Goal(
            goal_id=goal_dict.get("goal_id", ""),
            tenant_id=goal_dict.get("tenant_id", ""),
            name=goal_dict.get("name", ""),
            description=goal_dict.get("description", ""),
            status=GoalStatus(goal_dict.get("status", "pending")),
            priority=TaskPriority(goal_dict.get("priority", "normal")),
            created_at=goal_dict.get("created_at", 0.0),
            started_at=goal_dict.get("started_at", 0.0),
            completed_at=goal_dict.get("completed_at", 0.0),
            metadata=goal_dict.get("metadata", {}),
        )

        policy_dict = d.get("policy", {})
        budget_dict = policy_dict.get("budget", {})
        budget = StateCognitiveBudget(
            attention_budget=budget_dict.get("attention_budget", 1.0),
            memory_budget=budget_dict.get("memory_budget", 10),
            simulation_budget=budget_dict.get("simulation_budget", 5),
            reasoning_budget=budget_dict.get("reasoning_budget", 1000),
            verification_budget=budget_dict.get("verification_budget", 3),
            planning_budget=budget_dict.get("planning_budget", 30.0),
            tool_budget=budget_dict.get("tool_budget", 5),
        )

        policy = CognitivePolicy(
            reasoning_strategy=policy_dict.get("reasoning_strategy", "direct"),
            simulation_depth=policy_dict.get("simulation_depth", 1),
            verification_budget=policy_dict.get("verification_budget", 2),
            retrieval_budget=policy_dict.get("retrieval_budget", 5),
            planner_type=policy_dict.get("planner_type", "graph"),
            memory_budget=policy_dict.get("memory_budget", 10),
            model_selection=policy_dict.get("model_selection", "default"),
            reflection_enabled=policy_dict.get("reflection_enabled", True),
            budget=budget,
        )

        from hbllm.brain.cognitive_state import CandidatePlan, Evidence

        evidence_ledger = {}
        for k, ev_dict in d.get("evidence_ledger", {}).items():
            evidence_ledger[k] = Evidence(
                source=ev_dict.get("source", ""),
                confidence=ev_dict.get("confidence", 0.0),
                timestamp=ev_dict.get("timestamp", 0.0),
                generated_by=ev_dict.get("generated_by", ""),
                reasoning_path=ev_dict.get("reasoning_path", []),
            )

        candidate_plans = []
        for p_dict in d.get("candidate_plans", []):
            candidate_plans.append(
                CandidatePlan(
                    plan_id=p_dict.get("plan_id", ""),
                    graph=p_dict.get("graph", {}),
                    origin=p_dict.get("origin", "planner"),
                    confidence=p_dict.get("confidence", 1.0),
                    predicted_reward=p_dict.get("predicted_reward", 0.0),
                    predicted_cost=p_dict.get("predicted_cost", {}),
                    analogy_used=p_dict.get("analogy_used"),
                    simulation_result=p_dict.get("simulation_result"),
                    execution_trace=p_dict.get("execution_trace", []),
                )
            )

        return CognitiveState(
            goal=goal,
            policy=policy,
            state_id=d.get("state_id", ""),
            version=d.get("version", 1),
            parent_state_id=d.get("parent_state_id"),
            retrieved_memory=d.get("retrieved_memory", []),
            simulations=d.get("simulations", []),
            candidate_plans=candidate_plans,
            active_skills=d.get("active_skills", []),
            reflections=d.get("reflections", []),
            beliefs=d.get("beliefs", []),
            evidence_ledger=evidence_ledger,
            working_memory=d.get("working_memory", {}),
            confidence=d.get("confidence", 1.0),
            created_at=d.get("created_at", 0.0),
        )


# ─── Legacy Backward Compatibility Implementations ──────────────────────

from dataclasses import dataclass, field


@dataclass
class ExecutiveDecision:
    """The output of an executive decision cycle."""

    action: str  # "continue_focus" | "switch_to_goal" | "handle_interrupt" | "idle" | "sleep"
    target_goal: str | None = None
    target_event: dict[str, Any] | None = None
    reasoning: str = ""
    budget: dict[str, float] = field(default_factory=dict)
    suppress: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_goal": self.target_goal,
            "reasoning": self.reasoning,
            "budget": self.budget,
            "suppress": self.suppress,
        }


@dataclass
class CognitiveBudget:
    """How to distribute compute across processing tiers."""

    heavy_llm_pct: float = 0.3  # Expensive deep reasoning
    fast_router_pct: float = 0.5  # Quick classification/routing
    reflex_pct: float = 0.2  # Instant pattern matching

    def to_dict(self) -> dict[str, float]:
        return {
            "heavy_llm": round(self.heavy_llm_pct, 2),
            "fast_router": round(self.fast_router_pct, 2),
            "reflex": round(self.reflex_pct, 2),
        }


class ExecutiveCortex:
    """Unified cognitive control — the prefrontal cortex."""

    def __init__(
        self,
        goal_manager: Any | None = None,
        load_manager: Any | None = None,
        attention_system: Any | None = None,
        attention_manager: Any | None = None,
        state_machine: Any | None = None,
        user_model: Any | None = None,
    ) -> None:
        self._goals = goal_manager
        self._load = load_manager
        self._attention = attention_system
        self._attn_mgr = attention_manager
        self._state = state_machine
        self._user_model = user_model

        self._current_focus: str = ""
        self._current_focus_type: str = ""
        self._focus_started: float = 0.0
        self._focus_depth: float = 0.0
        self._last_switch: float = 0.0
        self._switch_count: int = 0
        self._suppressed_events: list[dict[str, Any]] = []

        self._min_focus_duration: float = 30.0
        self._base_switch_cost: float = 0.3
        self._interrupt_threshold: float = 0.7
        self._deep_focus_threshold: float = 0.7

    def decide_next_action(
        self,
        pending_events: list[dict[str, Any]] | None = None,
        tenant_id: str = "default",
    ) -> ExecutiveDecision:
        events = pending_events or []
        pressure = self._get_pressure()

        if pressure > 0.9:
            return ExecutiveDecision(
                action="idle",
                reasoning=f"Cognitive pressure too high ({pressure:.0%}). Shedding load.",
                budget=CognitiveBudget(0.0, 0.2, 0.8).to_dict(),
                suppress=[e.get("event_id", "") for e in events],
            )

        if events:
            interrupt = self._evaluate_interrupts(events)
            if interrupt:
                return interrupt

        if self._current_focus:
            age = time.time() - self._focus_started
            self._focus_depth = min(1.0, age / 300.0)

            switch_decision = self._evaluate_switch(tenant_id)
            if switch_decision:
                return switch_decision

            return ExecutiveDecision(
                action="continue_focus",
                target_goal=self._current_focus,
                reasoning=f"Maintaining focus on '{self._current_focus}' (depth={self._focus_depth:.0%})",
                budget=self._compute_budget(pressure),
            )

        goal_decision = self._select_goal(tenant_id)
        if goal_decision:
            return goal_decision

        return ExecutiveDecision(
            action="idle",
            reasoning="No active goals or pending events.",
            budget=CognitiveBudget(0.0, 0.1, 0.9).to_dict(),
        )

    def should_interrupt(self, event: dict[str, Any]) -> bool:
        urgency = event.get("urgency", 0.5)
        priority = event.get("priority", 0.5)
        event_score = urgency * 0.6 + priority * 0.4

        threshold = self._interrupt_threshold
        if self._focus_depth > self._deep_focus_threshold:
            threshold += 0.15

        switch_cost = self.get_switching_cost()
        threshold += switch_cost * 0.1

        return event_score > threshold

    def _evaluate_interrupts(self, events: list[dict[str, Any]]) -> ExecutiveDecision | None:
        for event in sorted(
            events,
            key=lambda e: e.get("urgency", 0) * 0.6 + e.get("priority", 0) * 0.4,
            reverse=True,
        ):
            if self.should_interrupt(event):
                old_focus = self._current_focus
                self._record_switch(f"interrupt:{event.get('topic', 'unknown')}", "event")
                return ExecutiveDecision(
                    action="handle_interrupt",
                    target_event=event,
                    reasoning=(
                        f"Interrupting '{old_focus}' for urgent event: "
                        f"{event.get('topic', 'unknown')} "
                        f"(urgency={event.get('urgency', '?')})"
                    ),
                    budget=self._compute_budget(self._get_pressure()),
                )

        suppressed = [e.get("event_id", "") for e in events if not self.should_interrupt(e)]
        if suppressed:
            self._suppressed_events.extend(events)
            self._suppressed_events = self._suppressed_events[-50:]
        return None

    # ── Task Switching ───────────────────────────────────────────────

    def get_switching_cost(self) -> float:
        """Human-inspired task switching penalty.

        Switching is expensive when:
            - Deep in current task (focus_depth > 0.7)
            - Recently switched (< 60s since last switch)
            - Many recent switches (fatigue)
        """
        depth_cost = self._focus_depth * 0.4
        recency_cost = max(0.0, 1.0 - (time.time() - self._last_switch) / 60.0) * 0.3
        fatigue_cost = min(0.3, self._switch_count * 0.02)
        return min(1.0, self._base_switch_cost + depth_cost + recency_cost + fatigue_cost)

    def _evaluate_switch(self, tenant_id: str) -> ExecutiveDecision | None:
        """Decide if we should switch away from current focus."""
        if not self._goals:
            return None

        age = time.time() - self._focus_started
        if age < self._min_focus_duration:
            return None

        switching_cost = self.get_switching_cost()

        try:
            active_goals = self._goals.get_active_goals(tenant_id=tenant_id)
        except Exception:
            return None

        if not active_goals:
            return None

        for goal in active_goals:
            goal_name = goal.name if hasattr(goal, "name") else str(goal.get("name", ""))
            goal_priority = self._goal_priority_score(goal)

            if goal_name != self._current_focus and goal_priority > (0.7 + switching_cost):
                self._record_switch(goal_name, "goal")
                return ExecutiveDecision(
                    action="switch_to_goal",
                    target_goal=goal_name,
                    reasoning=(
                        f"Switching from '{self._current_focus}' to '{goal_name}' "
                        f"(priority={goal_priority:.2f} > threshold={0.7 + switching_cost:.2f})"
                    ),
                    budget=self._compute_budget(self._get_pressure()),
                )

        return None

    def _select_goal(self, tenant_id: str) -> ExecutiveDecision | None:
        """Select the best goal when no focus is active."""
        if not self._goals:
            return None

        try:
            active_goals = self._goals.get_active_goals(tenant_id=tenant_id)
        except Exception:
            return None

        if not active_goals:
            return None

        scored = []
        for goal in active_goals:
            score = self._goal_priority_score(goal)
            scored.append((score, goal))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_goal = scored[0]

        goal_name = best_goal.name if hasattr(best_goal, "name") else str(best_goal.get("name", ""))
        self._record_switch(goal_name, "goal")

        return ExecutiveDecision(
            action="switch_to_goal",
            target_goal=goal_name,
            reasoning=f"Selected goal '{goal_name}' (score={best_score:.2f})",
            budget=self._compute_budget(self._get_pressure()),
        )

    # ── Focus Management ─────────────────────────────────────────────

    def set_focus(self, topic: str, focus_type: str = "goal") -> None:
        """Explicitly set current focus."""
        if topic != self._current_focus:
            self._record_switch(topic, focus_type)

    def clear_focus(self) -> None:
        """Clear current focus (task completed or abandoned)."""
        self._current_focus = ""
        self._current_focus_type = ""
        self._focus_depth = 0.0
        self._focus_started = 0.0

    def _record_switch(self, new_focus: str, focus_type: str) -> None:
        """Record a task switch."""
        self._current_focus = new_focus
        self._current_focus_type = focus_type
        self._focus_started = time.time()
        self._focus_depth = 0.0
        self._last_switch = time.time()
        self._switch_count += 1

    # ── Resource Allocation ──────────────────────────────────────────

    def get_cognitive_budget(self) -> CognitiveBudget:
        """Compute resource allocation based on current state."""
        pressure = self._get_pressure()
        budget = CognitiveBudget()

        if pressure > 0.8:
            budget.heavy_llm_pct = 0.1
            budget.fast_router_pct = 0.3
            budget.reflex_pct = 0.6
        elif pressure > 0.5:
            budget.heavy_llm_pct = 0.2
            budget.fast_router_pct = 0.5
            budget.reflex_pct = 0.3
        elif self._focus_depth > 0.7:
            budget.heavy_llm_pct = 0.5
            budget.fast_router_pct = 0.35
            budget.reflex_pct = 0.15
        else:
            budget.heavy_llm_pct = 0.3
            budget.fast_router_pct = 0.5
            budget.reflex_pct = 0.2

        return budget

    def _compute_budget(self, pressure: float) -> dict[str, float]:
        """Compute budget as dict for decision output."""
        budget = self.get_cognitive_budget()
        return budget.to_dict()

    # ── State Readers ────────────────────────────────────────────────

    def _get_pressure(self) -> float:
        """Read cognitive pressure from LoadManager."""
        if not self._load:
            return 0.3  # Default moderate
        try:
            if hasattr(self._load, "get_pressure"):
                return self._load.get_pressure()
            if hasattr(self._load, "pressure"):
                return self._load.pressure
        except Exception:
            pass
        return 0.3

    def _goal_priority_score(self, goal: Any) -> float:
        """Score a goal by priority, deadline, and user alignment."""
        priority_map = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.3,
            "background": 0.1,
        }

        if hasattr(goal, "priority"):
            priority_str = str(goal.priority)
        elif isinstance(goal, dict):
            priority_str = goal.get("priority", "medium")
        else:
            priority_str = "medium"
        priority = priority_map.get(priority_str, 0.5)

        deadline_boost = 0.0
        deadline = None
        if hasattr(goal, "deadline"):
            deadline = goal.deadline
        elif isinstance(goal, dict):
            deadline = goal.get("deadline")

        if deadline and isinstance(deadline, (int, float)):
            time_until = deadline - time.time()
            if time_until < 3600:
                deadline_boost = 0.3
            elif time_until < 86400:
                deadline_boost = 0.15

        alignment = 0.0
        if self._user_model:
            try:
                model = self._user_model.get_model("default")
                focus_value = str(model.current_focus.value).lower()
                goal_name = ""
                if hasattr(goal, "name"):
                    goal_name = goal.name.lower()
                elif isinstance(goal, dict):
                    goal_name = goal.get("name", "").lower()

                focus_words = set(focus_value.split())
                goal_words = set(goal_name.split())
                if focus_words & goal_words:
                    alignment = 0.1
            except Exception:
                pass

        return priority + deadline_boost + alignment

    # ── Introspection ────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Current executive state for debugging/context."""
        return {
            "current_focus": self._current_focus,
            "current_focus_type": self._current_focus_type,
            "focus_depth": round(self._focus_depth, 3),
            "focus_age_seconds": round(time.time() - self._focus_started, 1)
            if self._focus_started
            else 0,
            "switching_cost": round(self.get_switching_cost(), 3),
            "switch_count": self._switch_count,
            "suppressed_events": len(self._suppressed_events),
            "pressure": round(self._get_pressure(), 3),
            "budget": self.get_cognitive_budget().to_dict(),
        }

    def get_suppressed_events(self) -> list[dict[str, Any]]:
        """Get events that were suppressed during focus.

        Call this when focus ends to process deferred events.
        """
        events = self._suppressed_events.copy()
        self._suppressed_events.clear()
        return events

    def reset(self) -> None:
        """Reset executive state (e.g., on system restart)."""
        self.clear_focus()
        self._switch_count = 0
        self._suppressed_events.clear()
        logger.info("ExecutiveCortex reset")
