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
4. Declarative Neuro-Symbolic Integration: Synthesizes formal CausalPredicates and HCIRSymbolicWorldTheory rules.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from hbllm.hcir.world.causal_discovery import CausalPredicate

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ToolRole(StrEnum):
    """Functional role of an incorporated tool or held item."""

    KEY = "key"
    PUSHER = "pusher"
    LEVER = "lever"
    WEAPON = "weapon"
    CONTAINER = "container"
    TOOL = "tool"


@dataclass
class HeldTool:
    """An object or tool incorporated into the extended parietal body schema."""

    feature_id: int
    acquired_step: int
    role: ToolRole | str = ToolRole.TOOL
    relative_offset: tuple[int, int] = (0, 0)
    unlocked_barriers: set[int] = field(default_factory=set)

    def to_causal_predicate(self) -> CausalPredicate:
        """Export tool possession state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="held_tool",
            operator="==",
            value=self.feature_id,
            relation=str(self.role),
            target_entity="avatar",
        )


@dataclass
class ActiveManipulandum:
    """An external object currently focused/bound to directional motor actions."""

    entity_id: str
    feature_id: int
    pos: tuple[int, int]
    cells: list[tuple[int, int]] = field(default_factory=list)
    bound_step: int = 0
    role: str = "manipulandum"

    def to_causal_predicate(self) -> CausalPredicate:
        """Export manipulandum state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="active_manipulandum",
            operator="==",
            value=self.feature_id,
            relation=f"pos:{self.pos[0]},{self.pos[1]}",
            target_entity=self.entity_id,
        )


class ExtendedBodySchema:
    """Parietal body schema tracking physical effector extent and tool-gated barrier affordances."""

    def __init__(self) -> None:
        self.held_tools: list[HeldTool] = []
        # tool_feature -> set of barrier_features it unlocks
        self.tool_barrier_affinities: dict[int, set[int]] = {}
        self.active_manipulandum: ActiveManipulandum | None = None

    def reset_episode(self, retain_dynamics: bool = True) -> None:
        """Reset currently held tools while optionally preserving cross-trial tool affinities."""
        self.held_tools.clear()
        self.active_manipulandum = None
        if not retain_dynamics:
            self.tool_barrier_affinities.clear()

    def bind_manipulandum(
        self,
        entity_id: str,
        feature_id: int,
        pos: tuple[int, int],
        cells: list[tuple[int, int]] | None = None,
        step: int = 0,
        role: str = "manipulandum",
    ) -> None:
        """Bind an external entity as the active motor manipulandum."""
        self.active_manipulandum = ActiveManipulandum(
            entity_id=entity_id,
            feature_id=feature_id,
            pos=pos,
            cells=cells or [pos],
            bound_step=step,
            role=role,
        )
        logger.info(
            "ExtendedBodySchema: Bound entity %s (feature %d) at %s as active manipulandum at step %d.",
            entity_id,
            feature_id,
            pos,
            step,
        )

    def unbind_manipulandum(self) -> None:
        """Release currently bound manipulandum."""
        self.active_manipulandum = None

    def has_manipulandum(self) -> bool:
        """Return True if an external manipulandum is currently bound."""
        return self.active_manipulandum is not None

    def update_manipulandum_pos(self, dr: int, dc: int) -> tuple[int, int] | None:
        """Translate the active manipulandum by displacement (dr, dc)."""
        if self.active_manipulandum is None:
            return None
        r, c = self.active_manipulandum.pos
        new_pos = (r + dr, c + dc)
        new_cells = [(cr + dr, cc + dc) for cr, cc in self.active_manipulandum.cells]
        self.active_manipulandum.pos = new_pos
        self.active_manipulandum.cells = new_cells
        return new_pos

    def acquire_tool(
        self,
        feature_id: int,
        step: int,
        role: ToolRole | str = ToolRole.TOOL,
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

    def get_tools_for_barrier(self, barrier_feature: int) -> set[int]:
        """Return tool features known to unlock this barrier feature."""
        res: set[int] = set()
        for tool_feat, barriers in self.tool_barrier_affinities.items():
            if barrier_feature in barriers:
                res.add(tool_feat)
        return res

    def get_effector_hull(self, avatar_pos: tuple[int, int]) -> set[tuple[int, int]]:
        """Compute the spatial extent of the avatar plus extended tool reach."""
        hull = {avatar_pos}
        for t in self.held_tools:
            if t.relative_offset != (0, 0):
                hull.add(
                    (avatar_pos[0] + t.relative_offset[0], avatar_pos[1] + t.relative_offset[1])
                )
        return hull

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export tool possession and unlocked barrier permeability into formal causal predicates."""
        predicates: list[CausalPredicate] = []
        if self.active_manipulandum is not None:
            predicates.append(self.active_manipulandum.to_causal_predicate())
        for tool in self.held_tools:
            predicates.append(tool.to_causal_predicate())
        for tool_feat, barriers in self.tool_barrier_affinities.items():
            for b_feat in barriers:
                predicates.append(
                    CausalPredicate(
                        variable="tool_barrier_resonance",
                        operator="==",
                        value=b_feat,
                        relation=f"tool_{tool_feat}",
                        target_entity="avatar",
                    )
                )
        return predicates

    def to_world_theory(self) -> Any:
        """Synthesize an empirical HCIRSymbolicWorldTheory capturing tool permeability affordances."""
        from hbllm.hcir.world.cortex_causal import HCIRSymbolicWorldTheory

        theory = HCIRSymbolicWorldTheory()
        theory.walkable_predicates.extend(self.export_causal_predicates())
        return theory
