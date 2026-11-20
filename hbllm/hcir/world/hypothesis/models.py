"""Inductive Relational Hypothesis Data Models & Observation Context."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np

from hbllm.hcir.world.causal_discovery import (
    BeliefTransitionType,
    CausalPredicate,
)
from hbllm.hcir.world.percept import PlaceLocus

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HypothesisType(StrEnum):
    """Categorization of relational causal hypotheses."""

    CONTACT_TRANSFORM = "CONTACT_TRANSFORM"
    ACTION_MUTATION = "ACTION_MUTATION"
    REMOTE_MECHANISM = "REMOTE_MECHANISM"
    HAZARD_CONTACT = "HAZARD_CONTACT"
    COMPOSITIONAL_PROGRAM = "COMPOSITIONAL_PROGRAM"


class HypothesisStatus(StrEnum):
    """Lifecycle state of an inductive hypothesis."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"


@dataclass(frozen=True)
class HypothesisTransitionEvent:
    """Immutable audit event recording hypothesis lifecycle state changes."""

    event_type: BeliefTransitionType
    rule_id: str
    rule_type: HypothesisType
    timestamp: float
    confidence_before: float
    confidence_after: float
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RelationalRuleHypothesis:
    """Symbolic relational causal rule hypothesis."""

    rule_id: str
    rule_type: HypothesisType
    premise_feature: int | None = None
    target_feature: int | None = None
    result_feature: int | None = None
    action_id: Any = None
    positive_support: int = 1
    counterexamples: int = 0
    confidence: float = 0.5
    status: HypothesisStatus = HypothesisStatus.TENTATIVE
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def record_support(
        self, weight: float = 1.0, min_support: int = 2
    ) -> HypothesisTransitionEvent:
        """Record positive empirical evidence supporting this rule."""
        conf_before = self.confidence
        self.positive_support += int(weight)
        self.confidence = min(0.99, self.confidence + 0.15 * weight)
        event_type = BeliefTransitionType.HYPOTHESIS_TESTED
        reason = f"Empirical support observed (+{weight:.2f})"

        if self.positive_support >= min_support and self.counterexamples == 0:
            if self.status != HypothesisStatus.CONFIRMED:
                self.status = HypothesisStatus.CONFIRMED
                event_type = BeliefTransitionType.HYPOTHESIS_CONFIRMED
                reason = f"Confirmed with {self.positive_support} supporting observations"

        return HypothesisTransitionEvent(
            event_type=event_type,
            rule_id=self.rule_id,
            rule_type=self.rule_type,
            timestamp=time.time(),
            confidence_before=conf_before,
            confidence_after=self.confidence,
            reason=reason,
        )

    def record_counterexample(self, weight: float = 1.0) -> HypothesisTransitionEvent:
        """Record a falsifying counterexample (Popperian refutation)."""
        conf_before = self.confidence
        self.counterexamples += int(weight)
        self.confidence = max(0.01, self.confidence - 0.4 * weight)
        event_type = BeliefTransitionType.HYPOTHESIS_TESTED
        reason = f"Counterexample observed (+{weight:.2f})"

        if self.counterexamples >= 1:
            if self.status != HypothesisStatus.REFUTED:
                self.status = HypothesisStatus.REFUTED
                event_type = BeliefTransitionType.HYPOTHESIS_FALSIFIED
                reason = f"Falsified by counterexample (count={self.counterexamples})"

        return HypothesisTransitionEvent(
            event_type=event_type,
            rule_id=self.rule_id,
            rule_type=self.rule_type,
            timestamp=time.time(),
            confidence_before=conf_before,
            confidence_after=self.confidence,
            reason=reason,
        )

    @property
    def is_valid(self) -> bool:
        """True if the rule is confirmed or tentative without refutations."""
        return self.status != HypothesisStatus.REFUTED and self.counterexamples == 0

    def to_causal_predicate(self) -> CausalPredicate:
        """Compile this hypothesis into a formal first-order CausalPredicate."""
        if self.rule_type == HypothesisType.COMPOSITIONAL_PROGRAM:
            ast_obj = self.metadata.get("ast")
            if (
                ast_obj is not None
                and getattr(ast_obj, "__class__", None).__name__ == "SynthesizedRuleAST"
            ):
                return CausalPredicate(
                    variable=f"inductive_program:{self.rule_id}",
                    operator="==",
                    value=ast_obj.effect_operator,
                    relation="synthesized_transition_rule",
                    target_entity=ast_obj.precondition.describe(),
                    logic_op=ast_obj.precondition.logic_op,
                    children=list(ast_obj.precondition.children),
                    causal_status=self.status.value,
                    confidence=self.confidence,
                    metadata={"effect_params": dict(ast_obj.effect_params)},
                )
        if self.rule_type == HypothesisType.REMOTE_MECHANISM:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.premise_feature,
                relation="clears_barrier",
                target_entity=str(self.target_feature)
                if self.target_feature is not None
                else self.rule_id,
            )
        elif self.rule_type == HypothesisType.HAZARD_CONTACT:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.premise_feature,
                relation="lethal_hazard",
                target_entity=str(self.premise_feature)
                if self.premise_feature is not None
                else self.rule_id,
            )
        elif self.rule_type == HypothesisType.ACTION_MUTATION:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.action_id,
                relation="transforms_to",
                target_entity=str(self.result_feature)
                if self.result_feature is not None
                else self.rule_id,
            )
        else:
            return CausalPredicate(
                variable=f"inductive_rule:{self.rule_id}",
                operator="==",
                value=self.premise_feature,
                relation="contact_transform",
                target_entity=str(self.result_feature)
                if self.result_feature is not None
                else self.rule_id,
            )


@dataclass(frozen=True)
class EntityStateDelta:
    """Domain-agnostic summary of a single entity or locus transition."""

    locus: PlaceLocus
    prev_feature: int
    curr_feature: int
    within_prev_avatar: bool = False
    within_curr_avatar: bool = False
    adjacent_to_curr_avatar: bool = False


def _shift_numeric_locus(locus: PlaceLocus, delta: tuple[int | float, ...]) -> PlaceLocus | None:
    """Translate a coordinate PlaceLocus by a numeric displacement vector."""
    if not isinstance(locus, tuple) or len(locus) != len(delta):
        return None
    shifted: list[int | float] = []
    for coord, step in zip(locus, delta):
        if not isinstance(coord, (int, float)) or not isinstance(step, (int, float)):
            return None
        val = coord + step
        shifted.append(int(val) if isinstance(coord, int) and isinstance(step, int) else float(val))
    return tuple(shifted)


@dataclass(frozen=True)
class TransitionObservationContext:
    """Immutable snapshot of an embodied state transition.

    Supports both discrete lattice snapshots (``prev_grid``/``curr_grid``) and
    substrate-agnostic lifted sensory/delta summaries (``entity_deltas``,
    ``contact_feature``, adjacent features, and feature counts) over ``PlaceLocus``.
    """

    prev_grid: np.ndarray | None = None
    action: Any = None
    curr_grid: np.ndarray | None = None
    prev_avatar_pos: PlaceLocus | None = None
    curr_avatar_pos: PlaceLocus | None = None
    avatar_h: int = 1
    avatar_w: int = 1
    is_dead: bool = False
    is_won: bool = False
    background_feature: int = 0
    entity_deltas: tuple[EntityStateDelta, ...] | None = None
    contact_feature: int | None = None
    avatar_feature: int | None = None
    prev_adjacent_features: tuple[int, ...] | None = None
    curr_adjacent_features: tuple[int, ...] | None = None
    curr_static_adjacent_features: tuple[int, ...] | None = None
    paired_features: frozenset[int] | None = None
    prev_feature_counts: dict[int, int] | None = None
    curr_feature_counts: dict[int, int] | None = None

    @property
    def diff_mask(self) -> np.ndarray:
        if (
            self.prev_grid is None
            or self.curr_grid is None
            or self.prev_grid.shape != self.curr_grid.shape
        ):
            return np.zeros((0,), dtype=bool)
        return self.prev_grid != self.curr_grid

    @property
    def changed_count(self) -> int:
        if self.entity_deltas is not None:
            return len(self.entity_deltas)
        if (
            self.prev_grid is not None
            and self.curr_grid is not None
            and self.prev_grid.shape == self.curr_grid.shape
        ):
            return int(np.count_nonzero(self.diff_mask))
        return 0

    def is_valid_curr_locus(self) -> bool:
        """Verify ``curr_avatar_pos`` is present and within lattice bounds when grid-backed."""
        if self.curr_avatar_pos is None:
            return False
        if self.prev_grid is None:
            return True
        if (
            not isinstance(self.curr_avatar_pos, tuple)
            or len(self.curr_avatar_pos) != self.prev_grid.ndim
        ):
            return False
        return all(
            isinstance(c, (int, float)) and 0 <= int(c) < size
            for c, size in zip(self.curr_avatar_pos, self.prev_grid.shape)
        )

    def _within_avatar_radius(self, locus: PlaceLocus, avatar_pos: PlaceLocus | None) -> bool:
        if avatar_pos is None:
            return False
        if locus == avatar_pos:
            return True
        if (
            not isinstance(locus, tuple)
            or not isinstance(avatar_pos, tuple)
            or len(locus) != len(avatar_pos)
        ):
            return False
        for axis, (coord, av_coord) in enumerate(zip(locus, avatar_pos)):
            if not isinstance(coord, (int, float)) or not isinstance(av_coord, (int, float)):
                return False
            radius = (
                max(1, self.avatar_h // 2)
                if axis == 0
                else max(1, self.avatar_w // 2)
                if axis == 1
                else 1
            )
            if abs(coord - av_coord) > radius:
                return False
        return True

    def is_within_avatar_footprint(self, delta: EntityStateDelta) -> bool:
        """Return True if ``delta.locus`` falls within pre- or post-step avatar footprint."""
        if delta.within_prev_avatar or delta.within_curr_avatar:
            return True
        return self._within_avatar_radius(
            delta.locus, self.prev_avatar_pos
        ) or self._within_avatar_radius(delta.locus, self.curr_avatar_pos)

    def _is_cardinally_adjacent(self, locus: PlaceLocus, target: PlaceLocus | None) -> bool:
        if (
            not isinstance(locus, tuple)
            or not isinstance(target, tuple)
            or len(locus) != len(target)
        ):
            return False
        dist = 0.0
        for coord, t_coord in zip(locus, target):
            if not isinstance(coord, (int, float)) or not isinstance(t_coord, (int, float)):
                return False
            dist += abs(float(coord) - float(t_coord))
        return abs(dist - 1.0) < 1e-6

    def _cardinal_neighbors_in_grid(
        self, grid: np.ndarray, pos: PlaceLocus | None
    ) -> list[tuple[tuple[int, ...], int]]:
        if not isinstance(pos, tuple) or len(pos) != grid.ndim:
            return []
        if not all(isinstance(c, (int, float)) for c in pos):
            return []
        base = tuple(int(c) for c in pos)
        shape = grid.shape
        neighbors: list[tuple[tuple[int, ...], int]] = []
        for axis, size in enumerate(shape):
            for step in (-1, 1):
                nxt_val = base[axis] + step
                if 0 <= nxt_val < size:
                    nxt = base[:axis] + (nxt_val,) + base[axis + 1 :]
                    if all(0 <= nxt[i] < shape[i] for i in range(len(shape))):
                        neighbors.append((nxt, int(grid[nxt])))
        return neighbors

    def get_entity_deltas(self) -> tuple[EntityStateDelta, ...]:
        """Return lifted entity deltas, deriving them from lattice diffs when omitted."""
        if self.entity_deltas is not None:
            return self.entity_deltas
        if (
            self.prev_grid is None
            or self.curr_grid is None
            or self.prev_grid.shape != self.curr_grid.shape
        ):
            return ()
        coords = np.argwhere(self.diff_mask)
        deltas: list[EntityStateDelta] = []
        for coord_arr in coords:
            locus = tuple(int(v) for v in coord_arr)
            deltas.append(
                EntityStateDelta(
                    locus=locus,
                    prev_feature=int(self.prev_grid[locus]),
                    curr_feature=int(self.curr_grid[locus]),
                    within_prev_avatar=self._within_avatar_radius(locus, self.prev_avatar_pos),
                    within_curr_avatar=self._within_avatar_radius(locus, self.curr_avatar_pos),
                    adjacent_to_curr_avatar=self._is_cardinally_adjacent(
                        locus, self.curr_avatar_pos
                    ),
                )
            )
        res = tuple(deltas)
        object.__setattr__(self, "entity_deltas", res)
        return res

    def get_diff_feature_sets(self) -> tuple[set[int], set[int]]:
        """Return sets of features before and after at changed loci."""
        if self.entity_deltas is not None:
            return (
                {d.prev_feature for d in self.entity_deltas},
                {d.curr_feature for d in self.entity_deltas},
            )
        if (
            self.prev_grid is not None
            and self.curr_grid is not None
            and self.prev_grid.shape == self.curr_grid.shape
        ):
            dm = self.diff_mask
            return set(self.prev_grid[dm].tolist()), set(self.curr_grid[dm].tolist())
        return set(), set()

    def get_contact_feature(self) -> int | None:
        """Return feature at ``curr_avatar_pos`` in pre-transition state."""
        if self.contact_feature is not None:
            return self.contact_feature
        return self.get_prev_feature_at(self.curr_avatar_pos)

    def get_avatar_feature(self) -> int | None:
        """Return feature at ``prev_avatar_pos`` in pre-transition state."""
        if self.avatar_feature is not None:
            return self.avatar_feature
        return self.get_prev_feature_at(self.prev_avatar_pos)

    def get_prev_feature_at(self, locus: PlaceLocus | None) -> int | None:
        """Look up pre-transition feature at ``locus`` across grid or delta substrate."""
        if locus is None:
            return None
        if (
            self.prev_grid is not None
            and isinstance(locus, tuple)
            and len(locus) == self.prev_grid.ndim
            and all(isinstance(c, (int, float)) for c in locus)
        ):
            idx = tuple(int(c) for c in locus)
            if all(0 <= idx[i] < self.prev_grid.shape[i] for i in range(self.prev_grid.ndim)):
                return int(self.prev_grid[idx])
            return None
        for d in self.get_entity_deltas():
            if d.locus == locus:
                return d.prev_feature
        return None

    def get_curr_feature_at(self, locus: PlaceLocus | None) -> int | None:
        """Look up post-transition feature at ``locus`` across grid or delta substrate."""
        if locus is None:
            return None
        if (
            self.curr_grid is not None
            and isinstance(locus, tuple)
            and len(locus) == self.curr_grid.ndim
            and all(isinstance(c, (int, float)) for c in locus)
        ):
            idx = tuple(int(c) for c in locus)
            if all(0 <= idx[i] < self.curr_grid.shape[i] for i in range(self.curr_grid.ndim)):
                return int(self.curr_grid[idx])
            return None
        for d in self.get_entity_deltas():
            if d.locus == locus:
                return d.curr_feature
        return None

    def get_prev_adjacent_features(self) -> tuple[int, ...]:
        """Return non-background features adjacent to ``prev_avatar_pos`` in pre-state."""
        if self.prev_adjacent_features is not None:
            return self.prev_adjacent_features
        if self.prev_grid is not None:
            seen: list[int] = []
            for _, feat in self._cardinal_neighbors_in_grid(self.prev_grid, self.prev_avatar_pos):
                if feat != self.background_feature and feat not in seen:
                    seen.append(feat)
            return tuple(seen)
        return ()

    def get_prev_contact_and_adjacent_features(self) -> tuple[int, ...]:
        """Return non-background features overlapping or adjacent to ``prev_avatar_pos``."""
        if self.prev_grid is not None and isinstance(self.prev_avatar_pos, tuple):
            feats: list[int] = []
            self_feat = self.get_prev_feature_at(self.prev_avatar_pos)
            if self_feat is not None and self_feat != self.background_feature:
                feats.append(self_feat)
            for _, feat in self._cardinal_neighbors_in_grid(self.prev_grid, self.prev_avatar_pos):
                if feat != self.background_feature:
                    feats.append(feat)
            return tuple(feats)
        lifted: list[int] = []
        if self.avatar_feature is not None and self.avatar_feature != self.background_feature:
            lifted.append(self.avatar_feature)
        if self.prev_adjacent_features is not None:
            for feat in self.prev_adjacent_features:
                if feat != self.background_feature:
                    lifted.append(feat)
        elif self.contact_feature is not None and self.contact_feature != self.background_feature:
            lifted.append(self.contact_feature)
        return tuple(lifted)

    def get_curr_adjacent_features(self) -> tuple[int, ...]:
        """Return pre-transition features adjacent to ``curr_avatar_pos``."""
        if self.curr_adjacent_features is not None:
            return self.curr_adjacent_features
        if self.prev_grid is not None:
            return tuple(
                feat
                for _, feat in self._cardinal_neighbors_in_grid(
                    self.prev_grid, self.curr_avatar_pos
                )
            )
        return ()

    def get_curr_changed_and_static_adjacent(
        self, avatar_feat: int | None
    ) -> tuple[list[int], list[int]]:
        """Partition non-background/non-avatar features adjacent to ``curr_avatar_pos``."""
        bg = self.background_feature
        if self.prev_grid is not None:
            dm = self.diff_mask
            changed_adj: list[int] = []
            static_adj: list[int] = []
            for loc, cand_f in self._cardinal_neighbors_in_grid(
                self.prev_grid, self.curr_avatar_pos
            ):
                if cand_f != bg and cand_f != avatar_feat:
                    if dm.size > 0 and bool(dm[loc]):
                        changed_adj.append(cand_f)
                    else:
                        static_adj.append(cand_f)
            return changed_adj, static_adj
        changed_adj = [
            d.prev_feature
            for d in self.get_entity_deltas()
            if d.adjacent_to_curr_avatar and d.prev_feature != bg and d.prev_feature != avatar_feat
        ]
        if self.curr_static_adjacent_features is not None:
            static_adj = [
                f for f in self.curr_static_adjacent_features if f != bg and f != avatar_feat
            ]
        elif not changed_adj and self.curr_adjacent_features is not None:
            static_adj = [f for f in self.curr_adjacent_features if f != bg and f != avatar_feat]
        else:
            static_adj = []
        return changed_adj, static_adj

    def get_prev_feature_count(self, feature: int) -> int:
        """Return occurrence count of ``feature`` before transition."""
        if self.prev_feature_counts is not None:
            return self.prev_feature_counts.get(feature, 0)
        if self.prev_grid is not None:
            return int(np.count_nonzero(self.prev_grid == feature))
        return 0

    def get_curr_feature_count(self, feature: int) -> int:
        """Return occurrence count of ``feature`` after transition."""
        if self.curr_feature_counts is not None:
            return self.curr_feature_counts.get(feature, 0)
        if self.curr_grid is not None:
            return int(np.count_nonzero(self.curr_grid == feature))
        return 0

    def has_feature_count_summary(self) -> bool:
        """Return True if global feature counts are available via arrays or lifted maps."""
        return (self.prev_grid is not None and self.curr_grid is not None) or (
            self.prev_feature_counts is not None and self.curr_feature_counts is not None
        )
