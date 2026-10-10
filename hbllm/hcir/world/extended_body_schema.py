"""Inferior Parietal Lobule (IPL) Extended Body Schema & Tool Affordance System.

Modeled on primate posterior parietal cortex and Iriki-Maravita body schema plasticity:
1. Tool-Effector Incorporation: When an agent contacts or grasps a tool (key, pusher, lever),
   the bimodal somatosensory-visual receptive fields expand to incorporate the tool as an
   extension of the physical body schema.
2. Affordance Permeability Modulation: Dynamic barrier states:
   P(passable | door_feat, tool_held) = 1.0, vs P(passable | door_feat, ~tool_held) = 0.0.
   When the matching tool is held, the forward planner and A* search treat resonant barriers
   as permeable with an interaction cost, unlocking door pathways.
3. Effector Collision Hull Dilation: Computes the compound spatial geometry of avatar + held item.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class HeldTool:
    """An object or tool incorporated into the extended parietal body schema."""

    feature_id: int
    acquired_step: int
    role: str = "tool"
    relative_offset: tuple[int, int] = (0, 0)
    unlocked_barriers: set[int] = field(default_factory=set)


class ExtendedBodySchema:
    """Parietal body schema tracking physical effector extent and tool-gated barrier affordances."""

    def __init__(self) -> None:
        self.held_tools: list[HeldTool] = []
        # tool_feature -> set of barrier_features it unlocks
        self.tool_barrier_affinities: dict[int, set[int]] = {}

    def reset_episode(self, retain_dynamics: bool = True) -> None:
        """Reset currently held tools while optionally preserving cross-trial tool affinities."""
        self.held_tools.clear()
        if not retain_dynamics:
            self.tool_barrier_affinities.clear()

    def acquire_tool(
        self,
        feature_id: int,
        step: int,
        role: str = "tool",
        relative_offset: tuple[int, int] = (0, 0),
    ) -> None:
        """Incorporate an acquired tool into the extended body schema."""
        if not any(t.feature_id == feature_id for t in self.held_tools):
            unlocked = set(self.tool_barrier_affinities.get(feature_id, set()))
            tool = HeldTool(
                feature_id=feature_id,
                acquired_step=step,
                role=role,
                relative_offset=relative_offset,
                unlocked_barriers=unlocked,
            )
            self.held_tools.append(tool)
            logger.info(
                "ExtendedBodySchema: Tool %d incorporated into parietal body schema at step %d.",
                feature_id,
                step,
            )

    def expend_tool(self, feature_id: int) -> bool:
        """Remove a single-use tool upon unlocking a barrier."""
        for i, t in enumerate(self.held_tools):
            if t.feature_id == feature_id:
                self.held_tools.pop(i)
                logger.info("ExtendedBodySchema: Expended tool %d.", feature_id)
                return True
        return False

    def is_holding(self, feature_id: int) -> bool:
        """Check if an item of specific feature is currently incorporated in the body schema."""
        return any(t.feature_id == feature_id for t in self.held_tools)

    def register_resonance(self, tool_feature: int, barrier_feature: int) -> None:
        """Learn causal resonance: tool_feature unlocks barrier_feature."""
        self.tool_barrier_affinities.setdefault(tool_feature, set()).add(barrier_feature)
        for t in self.held_tools:
            if t.feature_id == tool_feature:
                t.unlocked_barriers.add(barrier_feature)
        logger.info(
            "ExtendedBodySchema: Learned tool-barrier resonance: Tool %d unlocks Barrier %d.",
            tool_feature,
            barrier_feature,
        )

    def is_barrier_permeable(self, barrier_feature: int) -> bool:
        """Check if any currently held tool makes this barrier feature permeable."""
        for t in self.held_tools:
            # Direct match (e.g. key color matches door color)
            if t.feature_id == barrier_feature:
                return True
            # Learned affinity match
            if barrier_feature in self.tool_barrier_affinities.get(t.feature_id, set()):
                return True
        return False

    def get_permeable_barrier_features(self) -> set[int]:
        """Return the set of all barrier features currently permeable under held tools."""
        permeable: set[int] = set()
        for t in self.held_tools:
            permeable.add(t.feature_id)
            permeable.update(self.tool_barrier_affinities.get(t.feature_id, set()))
        return permeable

    def get_effector_hull(self, avatar_pos: tuple[int, int]) -> set[tuple[int, int]]:
        """Compute the spatial extent of the avatar plus extended tool reach."""
        hull = {avatar_pos}
        for t in self.held_tools:
            if t.relative_offset != (0, 0):
                hull.add(
                    (avatar_pos[0] + t.relative_offset[0], avatar_pos[1] + t.relative_offset[1])
                )
        return hull
