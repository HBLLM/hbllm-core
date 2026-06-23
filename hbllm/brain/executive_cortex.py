"""
ExecutiveCortex — unified cognitive control.

The prefrontal cortex of HBLLM. Orchestrates existing cognitive modules
into a single decision surface for executive function:

    1. Goal Arbitration    — Which goal gets attention right now?
    2. Attention Allocation — What deserves focus vs background?
    3. Task Switching       — When to switch, with switching cost model
    4. Interruption Control — Should this interrupt current focus?
    5. Resource Allocation  — How to distribute compute budget

Does NOT replace existing modules — it reads their state and makes
arbitration decisions. It is a facade, not a controller.

Usage:
    cortex = ExecutiveCortex(
        goal_manager=goal_mgr,
        load_manager=load_mgr,
    )

    decision = cortex.decide_next_action(current_events)
    if decision.action == "switch_to_goal":
        # Switch to decision.target_goal
    elif decision.action == "handle_interrupt":
        # Handle the interrupt event
    elif decision.action == "continue_focus":
        # Stay on current task
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Decision Types ───────────────────────────────────────────────────────────


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


# ── Executive Cortex ─────────────────────────────────────────────────────────


class ExecutiveCortex:
    """Unified cognitive control — the prefrontal cortex.

    Reads state from existing modules and produces arbitration decisions.
    All parameters are optional — the cortex gracefully degrades when
    subsystems are unavailable.
    """

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

        # Executive state
        self._current_focus: str = ""
        self._current_focus_type: str = ""  # "goal" | "event" | "idle"
        self._focus_started: float = 0.0
        self._focus_depth: float = 0.0  # 0.0 (just started) to 1.0 (deep)
        self._last_switch: float = 0.0
        self._switch_count: int = 0
        self._suppressed_events: list[dict[str, Any]] = []

        # Tunable parameters
        self._min_focus_duration: float = 30.0  # Minimum seconds before switching
        self._base_switch_cost: float = 0.3  # Base cost of task switching
        self._interrupt_threshold: float = 0.7  # Score needed to interrupt focus
        self._deep_focus_threshold: float = 0.7  # When focus depth is "deep"

    # ── Core Decision Loop ───────────────────────────────────────────

    def decide_next_action(
        self,
        pending_events: list[dict[str, Any]] | None = None,
        tenant_id: str = "default",
    ) -> ExecutiveDecision:
        """The core executive decision.

        Considers:
            - Current focus and depth
            - Pending events and their urgency
            - Goal priorities and deadlines
            - Cognitive pressure (resource availability)
            - User alignment (from UserModel)
            - Switching cost

        Returns an ExecutiveDecision specifying what to do next.
        """
        events = pending_events or []
        pressure = self._get_pressure()

        # If under extreme pressure, shed load
        if pressure > 0.9:
            return ExecutiveDecision(
                action="idle",
                reasoning=f"Cognitive pressure too high ({pressure:.0%}). Shedding load.",
                budget=CognitiveBudget(0.0, 0.2, 0.8).to_dict(),
                suppress=[e.get("event_id", "") for e in events],
            )

        # Check for interrupts
        if events:
            interrupt = self._evaluate_interrupts(events)
            if interrupt:
                return interrupt

        # Check if current focus is still valid
        if self._current_focus:
            age = time.time() - self._focus_started
            # Update focus depth based on time spent
            self._focus_depth = min(1.0, age / 300.0)  # Reaches 1.0 after 5 minutes

            # Should we continue or switch?
            switch_decision = self._evaluate_switch(tenant_id)
            if switch_decision:
                return switch_decision

            # Continue current focus
            return ExecutiveDecision(
                action="continue_focus",
                target_goal=self._current_focus,
                reasoning=f"Maintaining focus on '{self._current_focus}' (depth={self._focus_depth:.0%})",
                budget=self._compute_budget(pressure),
            )

        # No current focus — pick best goal
        goal_decision = self._select_goal(tenant_id)
        if goal_decision:
            return goal_decision

        # Nothing to do
        return ExecutiveDecision(
            action="idle",
            reasoning="No active goals or pending events.",
            budget=CognitiveBudget(0.0, 0.1, 0.9).to_dict(),
        )

    # ── Interruption Control ─────────────────────────────────────────

    def should_interrupt(self, event: dict[str, Any]) -> bool:
        """Decide if an event should interrupt current focus.

        Combines:
            - Event urgency/priority
            - Current focus depth
            - Switching cost
            - Time since last switch
        """
        urgency = event.get("urgency", 0.5)
        priority = event.get("priority", 0.5)
        event_score = urgency * 0.6 + priority * 0.4

        # Higher threshold when deeply focused
        threshold = self._interrupt_threshold
        if self._focus_depth > self._deep_focus_threshold:
            threshold += 0.15  # Harder to interrupt when deep

        # Higher threshold if recently switched (prevent thrashing)
        switch_cost = self.get_switching_cost()
        threshold += switch_cost * 0.1

        return event_score > threshold

    def _evaluate_interrupts(self, events: list[dict[str, Any]]) -> ExecutiveDecision | None:
        """Check if any pending event warrants interruption."""
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

        # Suppress non-interrupting events
        suppressed = [e.get("event_id", "") for e in events if not self.should_interrupt(e)]
        if suppressed:
            self._suppressed_events.extend(events)
            # Cap suppressed list
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

        # Don't switch too quickly
        age = time.time() - self._focus_started
        if age < self._min_focus_duration:
            return None

        switching_cost = self.get_switching_cost()

        # Check if a higher-priority goal exists
        try:
            active_goals = self._goals.get_active_goals(tenant_id=tenant_id)
        except Exception:
            return None

        if not active_goals:
            return None

        for goal in active_goals:
            goal_name = goal.name if hasattr(goal, "name") else str(goal.get("name", ""))
            goal_priority = self._goal_priority_score(goal)

            # Is this goal more important than current focus?
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

        # Score and rank goals
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
            # Under heavy load — minimize expensive operations
            budget.heavy_llm_pct = 0.1
            budget.fast_router_pct = 0.3
            budget.reflex_pct = 0.6
        elif pressure > 0.5:
            # Moderate load — balanced
            budget.heavy_llm_pct = 0.2
            budget.fast_router_pct = 0.5
            budget.reflex_pct = 0.3
        elif self._focus_depth > 0.7:
            # Deep focus — allow expensive reasoning
            budget.heavy_llm_pct = 0.5
            budget.fast_router_pct = 0.35
            budget.reflex_pct = 0.15
        else:
            # Default balanced
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
        # Extract priority
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

        # Deadline urgency
        deadline_boost = 0.0
        deadline = None
        if hasattr(goal, "deadline"):
            deadline = goal.deadline
        elif isinstance(goal, dict):
            deadline = goal.get("deadline")

        if deadline and isinstance(deadline, (int, float)):
            time_until = deadline - time.time()
            if time_until < 3600:  # Less than 1 hour
                deadline_boost = 0.3
            elif time_until < 86400:  # Less than 1 day
                deadline_boost = 0.15

        # User alignment (if UserModel available)
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

                # Simple keyword overlap for alignment
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
