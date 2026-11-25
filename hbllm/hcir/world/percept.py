from __future__ import annotations

"""HCIR Perceptual Frame & Sensory Lifting Protocol.

Provides universal sensory lifting abstractions decoupling domain-specific
sensory transducers (e.g. 2D ARC-AGI grids, AI2-THOR RGB-D, Crafter tiles,
robotics point-clouds) from core cognitive faculties (path planning, causal
induction, active inference, and executive decision-making).

In accordance with HBLLM Invariant 1.7 (General Intelligence Invariant),
the cognitive core never inspects raw domain pixels, color frequencies,
or benchmark-specific visual heuristics. Domain adapters lift raw observations
into a PerceptionFrame, and the core operates exclusively over these percepts.
"""

import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np

from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity

logger = logging.getLogger(__name__)


class TerminalSignal(StrEnum):
    """Domain-agnostic terminal feedback signal."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class EpistemicPhase(StrEnum):
    """Cognitive lifecycle phases for autonomous exploration and problem solving."""

    MOTOR_GROUNDING = "motor_grounding"  # Self-identification & basic action calibration
    EPISTEMIC_EXPLORATION = (
        "epistemic_exploration"  # Trial-and-error hypothesis testing on unknown objects
    )
    EPISTEMIC_PROBE = (
        "epistemic_probe"  # Active epistemic probing upon conflict / unknown discovery
    )
    MENTAL_SIMULATION = "mental_simulation"  # Forward planning in mind
    EXPLOITATION = "exploitation"  # Direct execution of verified solution
    EXPLOITING = "exploitation"  # Direct execution alias
    REPLANNING = "replanning"  # Recovering from unexpected contradiction


@dataclass(frozen=True)
class ActuatorSchema:
    """Typed actuator description supplied by environment adapters (WP0.8)."""

    actuator_id: str
    kind: Literal["discrete", "point", "none"]
    params: Mapping[str, Any] = field(default_factory=dict)
    valid_targets: Sequence[tuple[int, int]] | None = None


@runtime_checkable
class ObservationProtocol(Protocol):
    """Domain-neutral structural protocol satisfied by lattice and non-lattice observations."""

    @property
    def available_actions(self) -> Sequence[Any]: ...

    @property
    def entities(self) -> Sequence[Any]: ...

    @property
    def context_id(self) -> str: ...

    @property
    def episode_id(self) -> str: ...

    @property
    def metadata(self) -> Mapping[str, Any]: ...

    @property
    def terminal(self) -> TerminalSignal | None: ...


@dataclass(frozen=True)
class EmbodiedObservation:
    """Universal domain-agnostic sensory observation consumed by HCIR cognitive core."""

    lattice: np.ndarray | None = None  # dense sensory field (any dims)
    entities: Sequence[Any] = ()  # lifted percepts (SpatialEntity or adapter specs)
    available_actions: Sequence[Any] = ()
    terminal: TerminalSignal | None = None  # SUCCESS / FAILURE / TIMEOUT
    context_id: str = "default_context"  # context scope ID
    episode_id: str = "0"  # episode index ID
    metadata: Mapping[str, Any] = field(default_factory=dict)
    actuators: Sequence[ActuatorSchema] = field(default_factory=list)


class EpisodeBoundaryKind(StrEnum):
    """Typed episode lifecycle boundaries (WP0.5)."""

    DEATH = "death"  # episode ended in failure; retry same level/context
    GOAL_REACHED = "goal_reached"  # task/goal solved; advance
    LEVEL_COMPLETE = "level_complete"  # backward-compatible alias for GOAL_REACHED
    CONTEXT_RESET = "context_reset"  # lattice/context changed mid-stream


class RetentionPolicy(StrEnum):
    """What state a faculty retains across a boundary kind (WP0.5)."""

    EPISODE_LOCAL = "episode_local"  # cleared on episode transitions (goal/death)
    LEVEL_LOCAL = "episode_local"  # deprecated alias -> EPISODE_LOCAL
    CONTEXT_LOCAL = "context_local"  # retained across episode resets within context
    LIFELONG = "lifelong"  # never cleared by boundaries


def apply_boundary_retention(owner: Any, boundary: Any) -> None:
    """Generic per-field retention for faculties (WP0.5/F1).

    Reads ``owner.FIELD_RETENTION`` — a class-level mapping of field name to
    ``RetentionPolicy`` — and clears fields according to the boundary kind:

    - ``EPISODE_LOCAL``: cleared on every episode/context transition.
    - ``CONTEXT_LOCAL``: retained across episode/seed transitions within a world
      context; cleared only on ``CONTEXT_RESET`` or ``retain_dynamics=False``.
    - ``LIFELONG``: never cleared by boundaries.
    """
    retention: dict[str, RetentionPolicy] = getattr(owner, "FIELD_RETENTION", {}) or {}
    if not retention:
        return
    kind = getattr(boundary, "kind", boundary)
    is_context_reset = kind in (
        EpisodeBoundaryKind.CONTEXT_RESET,
        "context_reset",
        "reset",
    ) or not getattr(boundary, "retain_dynamics", True)
    is_episode_transition = is_context_reset or kind in (
        EpisodeBoundaryKind.GOAL_REACHED,
        EpisodeBoundaryKind.LEVEL_COMPLETE,
        "goal_reached",
        "level_complete",
        EpisodeBoundaryKind.DEATH,
        "death",
    )
    if not is_episode_transition:
        return
    for field_name, policy in retention.items():
        should_clear = policy == RetentionPolicy.EPISODE_LOCAL or (
            policy == RetentionPolicy.CONTEXT_LOCAL and is_context_reset
        )
        if not should_clear or not hasattr(owner, field_name):
            continue
        cur = getattr(owner, field_name)
        if isinstance(cur, (set, dict, list)):
            cur.clear()
        elif isinstance(cur, deque):
            cur.clear()
        elif isinstance(cur, (int, float)):
            setattr(owner, field_name, type(cur)())
        else:
            setattr(owner, field_name, None)


@dataclass(frozen=True)
class EpisodeBoundary:
    """Explicit episode lifecycle boundary marker."""

    kind: EpisodeBoundaryKind | Literal["reset", "context_change", "retry"] = (
        EpisodeBoundaryKind.CONTEXT_RESET
    )
    retain_dynamics: bool = True
    same_level: bool | None = None
    same_context: bool | None = None

    def __post_init__(self) -> None:
        if self.same_context is not None:
            resolved = bool(self.same_context)
        elif self.same_level is not None:
            resolved = bool(self.same_level)
        else:
            resolved = bool(
                self.retain_dynamics and self.kind in (EpisodeBoundaryKind.DEATH, "death", "retry")
            )
        object.__setattr__(self, "same_level", resolved)
        object.__setattr__(self, "same_context", resolved)

    @property
    def is_same_context(self) -> bool:
        """True when episode resets within the exact same spatial/situational context."""
        return bool(self.same_context if self.same_context is not None else self.same_level)


def resolve_boundary_flags(boundary: Any) -> tuple[Any, bool, bool]:
    """Return ``(kind, retain_dynamics, is_same_context)`` for a boundary marker or kind."""
    if isinstance(boundary, EpisodeBoundary):
        retain_dyn = bool(boundary.retain_dynamics)
        return boundary.kind, retain_dyn, bool(boundary.is_same_context and retain_dyn)
    kind = boundary
    is_same_ctx = kind in (EpisodeBoundaryKind.DEATH, "death", "retry")
    return kind, True, is_same_ctx


@dataclass
class PerceptionFrame:
    """Universal Perceptual Frame consumed by HCIR Cognitive Core faculties.

    Attributes:
        entities: Segmented spatial entities with bounding boxes, centroids, and roles.
        passable_topology: 2D boolean array (True=walkable/traversable, False=blocked)
            or an abstract topological navigation graph.
        avatar_pos: Detected current coordinates of the controllable agent / avatar.
        avatar_features: Set of feature/color IDs belonging to the avatar.
        candidate_goals: Known or structural candidate goal / exit coordinates.
        static_barriers: Coordinates of impassable static obstacles.
        lethal_cells: Coordinates of known hazardous / lethal positions.
        shape: Dimensions of the spatial observation space (e.g., (H, W)).
        diff: Optional observation difference from the previous frame.
        metadata: Domain-specific auxiliary metadata (e.g., tags, step limit).
    """

    entities: list[SpatialEntity] = field(default_factory=list)
    passable_topology: np.ndarray | None = None
    avatar_pos: tuple[int, int] | None = None
    avatar_features: set[int] = field(default_factory=set)
    candidate_goals: list[tuple[int, int]] = field(default_factory=list)
    static_barriers: set[tuple[int, int]] = field(default_factory=set)
    lethal_cells: set[tuple[int, int]] = field(default_factory=set)
    shape: tuple[int, int] = (0, 0)
    diff: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def bg(self) -> int:
        """Background feature ID from metadata."""
        return int(self.metadata.get("bg", 0))

    def is_passable(self, r: int, c: int) -> bool:
        """Check whether a given grid coordinate is walkable and safe."""
        if self.passable_topology is not None:
            H, W = self.passable_topology.shape
            if 0 <= r < H and 0 <= c < W:
                return bool(self.passable_topology[r, c])
            return False
        if (r, c) in self.static_barriers or (r, c) in self.lethal_cells:
            return False
        H, W = self.shape
        return 0 <= r < H and 0 <= c < W


class BasePerceptionAdapter(ABC):
    """Abstract Base Perception Adapter interface.

    Domain plugins implement this interface to lift raw sensory modalities
    into standardized PerceptionFrames for the cognitive blackbox.
    """

    @abstractmethod
    def lift_percept(
        self,
        raw_observation: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> PerceptionFrame:
        """Lift raw sensory input into a standardized PerceptionFrame."""
        ...


class DefaultPerceptionAdapter(BasePerceptionAdapter):
    """Domain-agnostic default perception adapter for spatial 2D grid observations.

    Uses connected-component analysis and context roles (goals, barriers, avatar)
    to lift raw arrays into typed SpatialEntity components and a passable topology.
    """

    def lift_percept(
        self,
        raw_observation: Any,
        context: Any = None,
        **kwargs: Any,
    ) -> PerceptionFrame:
        """Lift raw 2D grid observation into a typed PerceptionFrame."""
        if isinstance(raw_observation, PerceptionFrame):
            return raw_observation
        if raw_observation is None:
            return PerceptionFrame()

        grid = np.asarray(raw_observation)
        if grid.ndim != 2:
            return PerceptionFrame()

        H, W = grid.shape
        bg = (
            int(kwargs.get("bg", 0))
            if "bg" in kwargs
            else int(np.bincount(grid.flatten().astype(int)).argmax())
        )

        from scipy import ndimage

        lethal_features: set[int] = set()
        lethal_positions: set[tuple[int, int]] = set()
        if context and hasattr(context, "hazard_tracker"):
            lethal_features = set(context.hazard_tracker.known_lethal_features)
            lethal_positions = set(context.hazard_tracker.static_lethal_positions)

        av_feats: set[int] = set()
        ref_av_pos: tuple[int, int] | None = None
        if context is not None:
            try:
                if context.avatar_features:
                    av_feats.update(context.avatar_features)
            except AttributeError:
                pass
            try:
                if context.avatar_feature is not None:
                    av_feats.add(context.avatar_feature)
            except AttributeError:
                pass
            try:
                ref_av_pos = context.avatar_pos
            except AttributeError:
                ref_av_pos = None

        entities: list[SpatialEntity] = []
        unique_feats = [int(x) for x in np.unique(grid) if int(x) != bg]
        for feat in unique_feats:
            mask = grid == feat
            labeled, num_features = ndimage.label(mask)
            for lab in range(1, num_features + 1):
                comp_mask = labeled == lab
                coords = np.argwhere(comp_mask)
                cells = [tuple(int(x) for x in c) for c in coords]
                r_min, c_min = coords.min(axis=0)
                r_max, c_max = coords.max(axis=0)
                centroid = (float(coords[:, 0].mean()), float(coords[:, 1].mean()))

                role = EntityRole.UNKNOWN
                if feat in lethal_features:
                    role = EntityRole.DYNAMIC_HAZARD
                elif context and hasattr(context, "symbolic_theory"):
                    if context.symbolic_theory.is_goal(feat):
                        role = EntityRole.GOAL
                    elif context.symbolic_theory.is_barrier(feat):
                        role = EntityRole.OBSTACLE
                if feat in av_feats:
                    role = EntityRole.AGENT

                ent = SpatialEntity(
                    id=f"entity_{feat}_{r_min}_{c_min}",
                    role=role,
                    feature_id=feat,
                    area=len(cells),
                    centroid=centroid,
                    bounding_box=(int(r_min), int(r_max), int(c_min), int(c_max)),
                    grid_pos=(int(round(centroid[0])), int(round(centroid[1]))),
                    properties={"cells": cells},
                )
                entities.append(ent)

        agent_candidates = [e for e in entities if e.role == EntityRole.AGENT]
        if len(agent_candidates) > 1 and ref_av_pos is not None:
            exact_ag = [
                e
                for e in agent_candidates
                if ref_av_pos in e.properties.get("cells", [e.grid_pos]) or e.grid_pos == ref_av_pos
            ]
            best_ag = (
                exact_ag[0]
                if exact_ag
                else min(
                    agent_candidates,
                    key=lambda e: (
                        abs(e.grid_pos[0] - ref_av_pos[0]) + abs(e.grid_pos[1] - ref_av_pos[1])
                    ),
                )
            )

            def _boxes_touch(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
                r_gap = max(0, b1[0] - b2[1] - 1, b2[0] - b1[1] - 1)
                c_gap = max(0, b1[2] - b2[3] - 1, b2[2] - b1[3] - 1)
                return (b1[0] <= b2[1] and b2[0] <= b1[1] and c_gap == 0) or (
                    b1[2] <= b2[3] and b2[2] <= b1[3] and r_gap == 0
                )

            avatar_body = [best_ag]
            body_ids = {id(best_ag)}
            grew = True
            while grew:
                grew = False
                for e in agent_candidates:
                    if id(e) in body_ids:
                        continue
                    if any(_boxes_touch(e.bounding_box, b.bounding_box) for b in avatar_body):
                        avatar_body.append(e)
                        body_ids.add(id(e))
                        grew = True
            learned_gp_set: set[tuple[int, int]] = set()
            if context is not None:
                try:
                    learned_gp_set = set(context.learned_goal_positions or ())
                except AttributeError:
                    learned_gp_set = set()
            for e in agent_candidates:
                if id(e) not in body_ids:
                    is_sym_goal = (
                        context is not None
                        and hasattr(context, "symbolic_theory")
                        and context.symbolic_theory.is_goal(e.feature_id)
                    )
                    e.role = (
                        EntityRole.GOAL
                        if e.grid_pos in learned_gp_set or is_sym_goal
                        else EntityRole.UNKNOWN
                    )

        barriers: set[tuple[int, int]] = set()
        goals: list[tuple[int, int]] = []
        avatar_pos: tuple[int, int] | None = None
        final_agents: list[SpatialEntity] = []
        lethal_cells: set[tuple[int, int]] = set(lethal_positions)
        for e in entities:
            if e.role == EntityRole.DYNAMIC_HAZARD or (
                e.feature_id is not None and e.feature_id in lethal_features
            ):
                lethal_cells.update(e.properties.get("cells", []))
            elif e.role == EntityRole.OBSTACLE:
                barriers.update(e.properties.get("cells", []))
            elif e.role == EntityRole.GOAL:
                goals.append(e.grid_pos)
            elif e.role == EntityRole.AGENT:
                final_agents.append(e)
        if final_agents:
            tot_area = sum(max(1, e.area) for e in final_agents)
            avatar_pos = (
                int(round(sum(e.centroid[0] * max(1, e.area) for e in final_agents) / tot_area)),
                int(round(sum(e.centroid[1] * max(1, e.area) for e in final_agents) / tot_area)),
            )

        if context and hasattr(context, "learned_barriers"):
            barriers.update(context.learned_barriers)

        candidate_goals: set[tuple[int, int]] = set(goals)
        if context and hasattr(context, "learned_goal_positions"):
            candidate_goals.update(context.learned_goal_positions)

        # Detect structural exit markers, gateway notches, and target zones
        detect_goals_fn = getattr(context, "detect_structural_goals", None)
        if detect_goals_fn is None:
            try:
                from hbllm.hcir.world.cortex_perception import PerceptionEngine

                detect_goals_fn = PerceptionEngine.detect_structural_goals
            except Exception:
                detect_goals_fn = None
        if detect_goals_fn is not None:
            try:
                structural_goals = detect_goals_fn(grid=grid, bg=bg)
                for sg in structural_goals:
                    if sg.get("type") in ("exit_marker", "gateway_notch", "target_zone"):
                        for cp in sg.get("cells", []):
                            candidate_goals.add(cp)
                        pos = sg.get("position")
                        if pos:
                            candidate_goals.add(pos)
            except Exception as exc:
                logger.debug("Structural goal detector fallback skipped: %s", exc)

        if candidate_goals:
            has_known_goal_features = (
                context is not None
                and hasattr(context, "symbolic_theory")
                and bool(getattr(context.symbolic_theory, "goal_features", None))
            )
            for e in entities:
                if e.grid_pos in candidate_goals and e.role not in (
                    EntityRole.AGENT,
                    EntityRole.DYNAMIC_HAZARD,
                ):
                    if not has_known_goal_features or (
                        e.feature_id is not None
                        and e.feature_id in context.symbolic_theory.goal_features
                    ):
                        e.role = EntityRole.GOAL

        passable = np.ones((H, W), dtype=bool)
        for r, c in barriers | lethal_cells:
            if 0 <= r < H and 0 <= c < W:
                passable[r, c] = False

        return PerceptionFrame(
            entities=entities,
            passable_topology=passable,
            avatar_pos=avatar_pos,
            candidate_goals=list(candidate_goals),
            static_barriers=barriers,
            lethal_cells=lethal_cells,
            shape=(H, W),
            metadata={"grid": grid, "bg": bg},
        )


class PerceptionRegistry:
    """Universal registry for sensory perception adapters.

    Allows external environment plugins (ARC-AGI, AI2-THOR, Robotics) to plug
    into the cognitive core without the core having any reverse dependencies on plugins.
    """

    _default_adapter: BasePerceptionAdapter | None = DefaultPerceptionAdapter()
    _adapters: dict[str, BasePerceptionAdapter] = {"default": DefaultPerceptionAdapter()}

    @classmethod
    def register(cls, name: str, adapter: BasePerceptionAdapter, set_default: bool = True) -> None:
        """Register an adapter by name, optionally setting it as the default."""
        cls._adapters[name] = adapter
        if set_default or cls._default_adapter is None:
            cls._default_adapter = adapter

    @classmethod
    def get(cls, name: str) -> BasePerceptionAdapter | None:
        """Get a registered adapter by name."""
        return cls._adapters.get(name)

    @classmethod
    def get_default(cls) -> BasePerceptionAdapter | None:
        """Get the active default perception adapter."""
        return cls._default_adapter

    @classmethod
    def clear(cls) -> None:
        """Reset the registry to default."""
        cls._adapters.clear()
        cls._default_adapter = DefaultPerceptionAdapter()


PlaceLocus = tuple[int | float, ...] | str


def format_locus(locus: PlaceLocus | None) -> str:
    """Render any coordinate tuple or symbolic PlaceLocus as a canonical display string."""
    if locus is None:
        return "none"
    if isinstance(locus, tuple):
        return "(" + ",".join(str(x) for x in locus) + ")"
    return str(locus)


def locus_token(locus: PlaceLocus | None) -> str:
    """Render any coordinate tuple or symbolic PlaceLocus as an identifier-safe token."""
    if locus is None:
        return "none"
    if isinstance(locus, tuple):
        return "_".join(str(x) for x in locus)
    return str(locus)


@dataclass
class ExogenousSignalTracker:
    """Empirically isolates exogenous / uncontrollable signals from controllable dynamics.

    Biological motivation (reafference cancellation / agency attribution):
    - Signals whose mutations are statistically independent of executed actions
      (mutating across multiple distinct non-local actions, I(Delta S; A) ~ 0) are
      promoted into ``exogenous_loci``.
    - External sensory adapters (e.g. plugin perception filters) may also supply explicit
      exogenous masks for non-interactive status readouts or disconnected partitions.
    """

    FIELD_RETENTION = {
        "exogenous_loci": RetentionPolicy.EPISODE_LOCAL,
        "_untargeted_actions": RetentionPolicy.EPISODE_LOCAL,
        "_untargeted_counts": RetentionPolicy.EPISODE_LOCAL,
        "_colinear_tracks": RetentionPolicy.EPISODE_LOCAL,
    }

    exogenous_loci: set[PlaceLocus] = field(default_factory=set)
    _untargeted_actions: dict[PlaceLocus, set[Any]] = field(default_factory=dict, repr=False)
    _untargeted_counts: dict[PlaceLocus, int] = field(default_factory=dict, repr=False)
    _colinear_tracks: dict[tuple[int, int], tuple[int, int, set[Any], list[PlaceLocus]]] = field(
        default_factory=dict, init=False, repr=False
    )
    _cached_partition_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False)
    _cached_partition_rows: Any | None = field(default=None, init=False, repr=False)
    _PARTITION_CACHE: ClassVar[dict[tuple[Any, ...], Any]] = {}

    def reset_episode(self, *, retain_long_term: bool = True) -> None:
        """Reset transient empirical exogeneity statistics on episode boundary."""
        self._cached_partition_key = None
        self._cached_partition_rows = None
        if not retain_long_term:
            self.exogenous_loci.clear()
            self._untargeted_actions.clear()
            self._untargeted_counts.clear()
            self._colinear_tracks.clear()

    def mark_exogenous(self, loci: Any) -> None:
        """Register externally supplied exogenous / non-interactive loci."""
        for loc in loci:
            self.exogenous_loci.add(loc)

    def observe_transition(
        self,
        mutated_loci: list[PlaceLocus] | set[PlaceLocus],
        action: Any,
        *,
        avatar_locus: PlaceLocus | None = None,
        target_locus: PlaceLocus | None = None,
        extra_local_loci: tuple[PlaceLocus | None, ...] = (),
        local_radius: float = 2.5,
    ) -> None:
        """Record non-local mutations to empirically discover exogenous ticking signals."""
        if not mutated_loci:
            return

        def _is_near(a: PlaceLocus, b: PlaceLocus | None) -> bool:
            if b is None or not isinstance(a, tuple) or not isinstance(b, tuple):
                return a == b
            if len(a) != len(b):
                return False
            return all(abs(float(x) - float(y)) <= local_radius for x, y in zip(a, b, strict=True))

        local_refs = (avatar_locus, target_locus, *extra_local_loci)
        nonlocal_2d: list[tuple[int, int]] = []
        for loc in mutated_loci:
            if any(_is_near(loc, ref) for ref in local_refs if ref is not None):
                # Re-grounded by local motor or effector proximity
                continue
            cnt = self._untargeted_counts.get(loc, 0) + 1
            self._untargeted_counts[loc] = cnt
            acts = self._untargeted_actions.setdefault(loc, set())
            if action is not None:
                acts.add(action)
            if len(acts) >= 2 or cnt >= 3:
                self.exogenous_loci.add(loc)
            if isinstance(loc, tuple) and len(loc) == 2:
                nonlocal_2d.append((int(loc[0]), int(loc[1])))

        # 1D colinear sequential gauge aggregation (e.g. timer/fuel bars shifting 1-2px/step):
        for axis in (0, 1):
            groups: dict[int, list[int]] = {}
            for r_i, c_i in nonlocal_2d:
                key_coord = r_i if axis == 0 else c_i
                val_coord = c_i if axis == 0 else r_i
                groups.setdefault(key_coord, []).append(val_coord)
            for key_coord, coords in groups.items():
                if not (1 <= len(coords) <= 4):
                    continue
                line_locs: list[PlaceLocus] = [
                    (key_coord, vc) if axis == 0 else (vc, key_coord) for vc in coords
                ]
                tkey = (axis, key_coord)
                prev = self._colinear_tracks.get(tkey)
                if prev is not None and min(abs(vc - prev[0]) for vc in coords) <= 2:
                    last_vc, tick_cnt, acts_set, hist = prev
                    if any(vc != last_vc for vc in coords):
                        tick_cnt += 1
                    next_vc = min(coords) if min(coords) < last_vc else max(coords)
                    if action is not None:
                        acts_set.add(action)
                    hist.extend(line_locs)
                    if len(hist) > 64:
                        del hist[:-64]
                    self._colinear_tracks[tkey] = (next_vc, tick_cnt, acts_set, hist)
                    if tick_cnt >= 3 or (tick_cnt >= 2 and len(acts_set) >= 2):
                        self.exogenous_loci.update(hist)
                else:
                    init_acts = {action} if action is not None else set()
                    self._colinear_tracks[tkey] = (coords[0], 1, init_acts, list(line_locs))

    def _compute_partitioned_rows(self, lattice: Any, bg: Any = 0) -> Any:
        """Identify rows residing in topologically separated peripheral divider bands."""
        import numpy as np

        arr = np.asarray(lattice)
        if arr.ndim != 2 or arr.shape[0] < 24 or arr.shape[1] == 0:
            return None
        H, W = arr.shape
        try:
            bg_val = int(bg)
        except (TypeError, ValueError):
            bg_val = 0

        use_shared_cache = arr.size <= 16384
        cache_sig = (
            H,
            W,
            bg_val,
            arr.tobytes() if use_shared_cache else id(lattice),
        )
        if self._cached_partition_key == cache_sig and self._cached_partition_rows is not None:
            return self._cached_partition_rows
        if use_shared_cache:
            shared_cached = self._PARTITION_CACHE.get(cache_sig)
            if shared_cached is not None:
                self._cached_partition_key = cache_sig
                self._cached_partition_rows = shared_cached
                return shared_cached

        flags = np.zeros(H, dtype=bool)
        max_val = int(arr.max())
        fast_hist = int(arr.min()) >= 0 and max_val < 256
        if fast_hist:
            n_vals = max_val + 1
            counts = np.stack([np.bincount(row.astype(np.int64), minlength=n_vals) for row in arr])
            dom_vals = np.argmax(counts, axis=1)
            dom_counts = counts[np.arange(H), dom_vals]
            div_rows = (dom_vals != bg_val) & (dom_counts >= int(W * 0.85))
            cum_counts = np.cumsum(counts, axis=0)
            total_counts = cum_counts[-1]
        elif int(arr.min()) >= 0:
            n_vals = max_val + 1
            counts = np.stack([np.bincount(row.astype(np.int64), minlength=n_vals) for row in arr])
            dom_vals = np.argmax(counts, axis=1)
            dom_counts = counts[np.arange(H), dom_vals]
            div_rows = (dom_vals != bg_val) & (dom_counts >= int(W * 0.85))
        else:
            div_rows = np.zeros(H, dtype=bool)
            for rr in range(H):
                vals, cnts = np.unique(arr[rr, :], return_counts=True)
                idx = int(np.argmax(cnts))
                if int(vals[idx]) != bg_val and int(cnts[idx]) >= int(W * 0.85):
                    div_rows[rr] = True

        band_depth = H // 4
        if div_rows[: band_depth + 1].any():
            for rr in range(min(band_depth + 1, H)):
                if div_rows[rr + 1 : min(band_depth + 1, H)].any():
                    flags[rr] = True
        thresh_85 = int(W * 0.85)
        thresh_50 = int(W * 0.50)
        for div_r in range(band_depth, 1, -1):
            if fast_hist:
                line_idx = int(dom_vals[div_r])
                line_cnt = int(dom_counts[div_r])
                if line_cnt >= thresh_85:
                    banner_c = cum_counts[div_r - 1]
                    field_c = total_counts - cum_counts[div_r]
                    b_max_cnt = int(banner_c.max())
                    banner_dom = int(np.argmax(banner_c))
                    field_dom = int(np.argmax(field_c))
                    has_distinct_bg = (banner_dom != field_dom) and (
                        (b_max_cnt / (div_r * W)) >= 0.85
                    )
                    adj_bordered = (
                        line_cnt == W
                        and div_r <= 5
                        and (b_max_cnt < div_r * W or banner_dom != field_dom)
                        and int(dom_vals[div_r + 1]) != field_dom
                        and int(dom_counts[div_r + 1]) >= thresh_50
                    )
                    has_divider_line = (
                        (line_idx != field_dom or line_idx != bg_val) and (line_cnt >= thresh_85)
                    ) or adj_bordered
                    if has_distinct_bg or has_divider_line:
                        flags[: div_r + 1] = True
                        break
                continue
            line = arr[div_r, :]
            vals, cnts = np.unique(line, return_counts=True)
            if len(cnts) > 0:
                idx = int(np.argmax(cnts))
                if int(cnts[idx]) >= thresh_85:
                    line_color = int(vals[idx])
                    banner_region = arr[:div_r, :]
                    field_region = arr[div_r + 1 :, :]
                    b_vals, b_cnts = np.unique(banner_region, return_counts=True)
                    f_vals, f_cnts = np.unique(field_region, return_counts=True)
                    if len(b_cnts) > 0 and len(f_cnts) > 0:
                        b_max = int(np.argmax(b_cnts))
                        banner_dom = int(b_vals[b_max])
                        field_dom = int(f_vals[int(np.argmax(f_cnts))])
                        has_distinct_bg = (banner_dom != field_dom) and (
                            (b_cnts[b_max] / banner_region.size) >= 0.85
                        )
                        has_divider_line = (line_color != field_dom or line_color != bg_val) and (
                            int(cnts[idx]) >= thresh_85
                        )
                        if has_distinct_bg or has_divider_line:
                            flags[: div_r + 1] = True
                            break

        bot_start = max(0, H - 1 - band_depth)
        if div_rows[bot_start:].any():
            for rr in range(bot_start, H):
                if not flags[rr] and div_rows[bot_start:rr].any():
                    flags[rr] = True
        for div_r in range(H - 3, max(H - band_depth - 2, H - H // 3), -1):
            if not (0 <= div_r < H):
                continue
            if fast_hist:
                line_idx = int(dom_vals[div_r])
                line_cnt = int(dom_counts[div_r])
                if line_cnt >= thresh_85:
                    banner_c = total_counts - cum_counts[div_r]
                    field_c = cum_counts[div_r - 1]
                    b_max_cnt = int(banner_c.max())
                    banner_dom = int(np.argmax(banner_c))
                    field_dom = int(np.argmax(field_c))
                    has_distinct_bg = (banner_dom != field_dom) and (
                        (b_max_cnt / ((H - div_r - 1) * W)) >= 0.85
                    )
                    adj_bordered = (
                        line_cnt == W
                        and div_r >= H - 6
                        and (b_max_cnt < (H - div_r - 1) * W or banner_dom != field_dom)
                        and int(dom_vals[div_r - 1]) != field_dom
                        and int(dom_counts[div_r - 1]) >= thresh_50
                    )
                    has_divider_line = (
                        (line_idx != field_dom or line_idx != bg_val) and (line_cnt >= thresh_85)
                    ) or adj_bordered
                    if has_distinct_bg or has_divider_line:
                        flags[div_r:] = True
                        break
                continue
            line = arr[div_r, :]
            vals, cnts = np.unique(line, return_counts=True)
            if len(cnts) > 0:
                idx = int(np.argmax(cnts))
                if int(cnts[idx]) >= thresh_85:
                    line_color = int(vals[idx])
                    banner_region = arr[div_r + 1 :, :]
                    field_region = arr[:div_r, :]
                    b_vals, b_cnts = np.unique(banner_region, return_counts=True)
                    f_vals, f_cnts = np.unique(field_region, return_counts=True)
                    if len(b_cnts) > 0 and len(f_cnts) > 0:
                        b_max = int(np.argmax(b_cnts))
                        banner_dom = int(b_vals[b_max])
                        field_dom = int(f_vals[int(np.argmax(f_cnts))])
                        has_distinct_bg = (banner_dom != field_dom) and (
                            (b_cnts[b_max] / banner_region.size) >= 0.85
                        )
                        has_divider_line = (line_color != field_dom or line_color != bg_val) and (
                            int(cnts[idx]) >= thresh_85
                        )
                        if has_distinct_bg or has_divider_line:
                            flags[div_r:] = True
                            break

        flags.setflags(write=False)
        if use_shared_cache:
            if len(self._PARTITION_CACHE) >= 8:
                self._PARTITION_CACHE.pop(next(iter(self._PARTITION_CACHE)))
            self._PARTITION_CACHE[cache_sig] = flags
        self._cached_partition_key = cache_sig
        self._cached_partition_rows = flags
        return flags

    @classmethod
    def is_partitioned_or_enclosing(
        cls,
        locus: PlaceLocus,
        lattice: Any = None,
        bg: Any = 0,
        *,
        include_outer_edge: bool = False,
        tracker: ExogenousSignalTracker | None = None,
    ) -> bool:
        """True when ``locus`` lies in an enclosing structural boundary or partitioned sideband."""
        if not isinstance(locus, tuple) or len(locus) < 2 or lattice is None:
            return False
        import numpy as np

        arr = np.asarray(lattice)
        if arr.ndim != 2:
            return False
        H, W = arr.shape
        r, c = int(locus[0]), int(locus[1])
        if include_outer_edge and (r == 0 or r == H - 1 or c == 0 or c == W - 1):
            return True
        if not (0 <= r < H and 0 <= c < W):
            return include_outer_edge
        if H < 24 or (not include_outer_edge and W < 24):
            return False

        t = tracker if tracker is not None else cls()
        flags = t._compute_partitioned_rows(arr, bg=bg)
        return bool(flags[r]) if flags is not None else False

    def is_exogenous(
        self,
        locus: PlaceLocus,
        *,
        lattice: Any = None,
        bg: Any = 0,
        include_outer_edge: bool = False,
    ) -> bool:
        """Return True if ``locus`` is empirically exogenous or topologically partitioned."""
        if locus in self.exogenous_loci:
            return True
        return self.is_partitioned_or_enclosing(
            locus,
            lattice=lattice,
            bg=bg,
            include_outer_edge=include_outer_edge,
            tracker=self,
        )
