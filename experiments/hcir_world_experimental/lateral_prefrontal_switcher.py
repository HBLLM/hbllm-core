"""Lateral Prefrontal Cortex (LPFC) Dual Task Switching & Cognitive Flexibility.

Biologically modeled on mammalian lateral prefrontal cortex (LPFC) executive
task-set reconfiguration (Monsell, 2003; Koechlin & Summerfield, 2007):
1. Executive Cognitive Task Sets:
   - NAVIGATE: Frontier and spatial traversal toward distal targets.
   - INTERACT: Proximal affordance manipulation (switches, barriers, mechanisms).
   - AVOID: Reactive and proactive threat avoidance (hazard cones, patrollers).
   - COLLECT: Resource and collectible acquisition.
   - EXPLORE: Epistemic foraging driven by high world model uncertainty.
2. Hysteresis Gating & Task-Set Inertia:
   - Prevents rapid, maladaptive task flickering (chattering).
   - Requires competitor task affinity to exceed current task affinity by an
     inertia margin Delta_inertia before a switch is initiated:
     A_competitor > A_current + Delta_inertia.
3. Switch Cost Penalty & Schema Inhibition:
   - Quantifies cognitive reconfiguration cost tau_switch based on cognitive load
     and task unfamiliarity.
   - Computes inhibitory suppression signal to suppress residual motor schemas
     associated with the previous task set.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class CognitiveTaskSet(str, Enum):
    """Primate LPFC executive task configurations."""

    NAVIGATE = "navigate"
    INTERACT = "interact"
    AVOID = "avoid"
    COLLECT = "collect"
    EXPLORE = "explore"


@dataclass
class TaskSwitchEvent:
    """Historical record of an executive task-set reconfiguration."""

    from_task: CognitiveTaskSet
    to_task: CognitiveTaskSet
    switch_cost: float
    residual_inertia: float
    timestamp: float


@dataclass
class TaskArbitrationResult:
    """Result of LPFC executive task arbitration cycle."""

    active_task: CognitiveTaskSet
    previous_task: CognitiveTaskSet
    switched: bool
    switch_cost: float
    task_affinities: dict[CognitiveTaskSet, float]
    inhibitory_suppression: float
    reason: str


class LateralPrefrontalTaskSwitcher:
    """Executive controller managing cognitive task switching and schema gating."""

    def __init__(
        self,
        initial_task: CognitiveTaskSet = CognitiveTaskSet.NAVIGATE,
        inertia_margin: float = 0.25,
        base_switch_cost: float = 0.15,
        max_history: int = 100,
    ) -> None:
        """Initialize the LPFC task switcher.

        Args:
            initial_task: Initial active cognitive task set.
            inertia_margin: Minimum affinity margin required to overcome task inertia.
            base_switch_cost: Baseline latency/energy cost for task reconfiguration.
            max_history: Maximum number of switch events to track in working memory.
        """
        self.current_task = initial_task
        self.inertia_margin = max(0.05, float(inertia_margin))
        self.base_switch_cost = max(0.01, float(base_switch_cost))
        self.max_history = max(10, max_history)

        self.switch_history: list[TaskSwitchEvent] = []
        self._step_counter: int = 0
        self._residual_inertia: float = 0.0

    def evaluate_affinities(
        self,
        working_memory: Any = None,
        hazard_distance: float | None = None,
        goal_distance: float | None = None,
        manipulable_adjacent: bool = False,
        collectible_adjacent: bool = False,
        epistemic_uncertainty: float = 0.0,
    ) -> dict[CognitiveTaskSet, float]:
        """Compute task affinity distribution across all executive task sets.

        Returns normalized scores summing to 1.0.
        """
        raw_scores: dict[CognitiveTaskSet, float] = {
            CognitiveTaskSet.NAVIGATE: 0.2,
            CognitiveTaskSet.INTERACT: 0.1,
            CognitiveTaskSet.AVOID: 0.05,
            CognitiveTaskSet.COLLECT: 0.1,
            CognitiveTaskSet.EXPLORE: 0.1,
        }

        # 1. Threat avoidance takes high priority if hazard is close
        if hazard_distance is not None:
            if hazard_distance <= 1.5:
                raw_scores[CognitiveTaskSet.AVOID] += 0.8
            elif hazard_distance <= 3.0:
                raw_scores[CognitiveTaskSet.AVOID] += 0.45

        # 2. Proximal affordances elevate INTERACT and COLLECT
        if manipulable_adjacent:
            raw_scores[CognitiveTaskSet.INTERACT] += 0.6
        if collectible_adjacent:
            raw_scores[CognitiveTaskSet.COLLECT] += 0.55

        # 3. Goal navigation
        if goal_distance is not None:
            if goal_distance > 0:
                # Closer goals encourage direct navigation/approach
                raw_scores[CognitiveTaskSet.NAVIGATE] += max(0.1, 1.0 / (1.0 + 0.2 * goal_distance))

        # 4. Epistemic uncertainty encourages exploratory foraging
        if epistemic_uncertainty > 0.3:
            raw_scores[CognitiveTaskSet.EXPLORE] += min(0.7, epistemic_uncertainty)

        # 5. Check working memory state if provided
        if working_memory is not None:
            barriers = getattr(working_memory, "barriers", None)
            if barriers and len(barriers) > 0 and not manipulable_adjacent:
                # Unresolved barriers boost exploration/navigation
                raw_scores[CognitiveTaskSet.NAVIGATE] += 0.15

        # Softmax or normalized sum
        total = sum(raw_scores.values())
        if total <= 0:
            return {k: 0.2 for k in CognitiveTaskSet}
        return {k: v / total for k, v in raw_scores.items()}

    def arbitrate(
        self,
        affinities: dict[CognitiveTaskSet, float],
        cognitive_load: float = 0.0,
        timestamp: float = 0.0,
    ) -> TaskArbitrationResult:
        """Arbitrate between competing task sets considering hysteresis and inertia.

        Args:
            affinities: Computed affinity distribution for candidate task sets.
            cognitive_load: Current PFC cognitive load [0, 1], increasing switch cost.
            timestamp: Simulation time or step index.

        Returns:
            TaskArbitrationResult indicating active task set and schema inhibition.
        """
        self._step_counter += 1
        current_affinity = affinities.get(self.current_task, 0.0)

        # Find best competitor
        best_candidate = self.current_task
        best_affinity = current_affinity

        for task, score in affinities.items():
            if score > best_affinity:
                best_affinity = score
                best_candidate = task

        # Decay residual inertia
        self._residual_inertia = max(0.0, self._residual_inertia * 0.8)

        switched = False
        switch_cost = 0.0
        inhibitory_suppression = 0.0
        reason = "retained_active_task_set"

        # Check if competitor beats current task + inertia margin
        if best_candidate != self.current_task:
            required_margin = self.inertia_margin + self._residual_inertia
            # Threat avoidance bypasses inertia if affinity is exceptionally high
            if best_candidate == CognitiveTaskSet.AVOID and best_affinity >= 0.5:
                required_margin = 0.05

            if best_affinity > (current_affinity + required_margin):
                prev_task = self.current_task
                self.current_task = best_candidate
                switched = True

                # Compute reconfiguration switch cost
                clamped_load = max(0.0, min(1.0, cognitive_load))
                switch_cost = self.base_switch_cost * (1.0 + clamped_load)
                self._residual_inertia = self.inertia_margin * 0.7
                inhibitory_suppression = min(1.0, 0.5 + 0.5 * clamped_load)
                reason = f"switched_from_{prev_task.value}_to_{best_candidate.value}"

                event = TaskSwitchEvent(
                    from_task=prev_task,
                    to_task=best_candidate,
                    switch_cost=switch_cost,
                    residual_inertia=self._residual_inertia,
                    timestamp=timestamp if timestamp > 0 else float(self._step_counter),
                )
                self.switch_history.append(event)
                if len(self.switch_history) > self.max_history:
                    self.switch_history.pop(0)
            else:
                reason = f"competitor_{best_candidate.value}_inhibited_by_inertia"

        return TaskArbitrationResult(
            active_task=self.current_task,
            previous_task=(
                self.switch_history[-1].from_task
                if switched and self.switch_history
                else self.current_task
            ),
            switched=switched,
            switch_cost=switch_cost,
            task_affinities=affinities,
            inhibitory_suppression=inhibitory_suppression,
            reason=reason,
        )

    def force_task_set(self, task_set: CognitiveTaskSet) -> None:
        """Directly set active task configuration, resetting inertia."""
        self.current_task = task_set
        self._residual_inertia = 0.0

    def get_active_task_set(self) -> CognitiveTaskSet:
        """Get the current active cognitive task set."""
        return self.current_task

    def reset(self, initial_task: CognitiveTaskSet = CognitiveTaskSet.NAVIGATE) -> None:
        """Reset internal history and state."""
        self.current_task = initial_task
        self.switch_history.clear()
        self._step_counter = 0
        self._residual_inertia = 0.0
