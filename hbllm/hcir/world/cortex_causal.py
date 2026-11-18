"""Ventromedial Prefrontal Cortex (vmPFC) & Inferior Parietal Causal Induction Faculty.

Biologically modeled on mammalian vmPFC distal causal credit assignment (Pearl's do(a) intervention)
and parietal tool affordance resonance (Iriki-Maravita body schema plasticity):
1. Interventional Action-Effect Binding:
   - Evaluates do(a) interventions: correlates intentional actions at origin/target locations
     with local and distal state mutations (diffs).
   - Generates and verifies causal hypotheses for switches, levers, plates, and keys.
2. Neuro-Symbolic World Theory:
   - Grounds first-order causal predicates for goals, barriers, walkable areas, and pushable cargo.
   - Enforces the HCIR Attractor Invariant across diverse environments without heuristics.
3. Parietal Tool-Barrier Resonance:
   - Tracks held objects in the extended body schema and evaluates barrier permeability.
4. Remote Mechanism Reverse Affordance Lookup:
   - When an objective is blocked by a barrier, queries vmPFC for governing distal triggers.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any, Literal

import numpy as np

from hbllm.hcir.graph import ConceptNode
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity
from hbllm.hcir.transactions import TransactionOp, TransactionOperation
from hbllm.hcir.world.causal_discovery import CausalPredicate
from hbllm.hcir.world.percept import PlaceLocus, TerminalSignal

logger = logging.getLogger(__name__)


class OntologyEvidenceRank(IntEnum):
    """Hierarchy of empirical evidence strength for ontology induction."""

    UNVERIFIED = 0
    HEURISTIC = 10
    CANDIDATE = 20
    HYDRATION_DEFAULT = 40
    TRAVERSAL = 60
    COLLISION = 80
    INTERACTION = 85
    LETHAL = 95
    TERMINAL_REWARD = 100


def parse_evidence(evidence: Any) -> tuple[int, str]:
    """Parse raw evidence into (rank, descriptive_str)."""
    if evidence is None:
        return OntologyEvidenceRank.UNVERIFIED, "unverified"
    if (
        evidence == TerminalSignal.SUCCESS
        or evidence == "terminal_reward"
        or evidence == "terminal_signal"
    ):
        return OntologyEvidenceRank.TERMINAL_REWARD, "terminal_reward:success"
    if isinstance(evidence, TerminalSignal):
        return (
            OntologyEvidenceRank.TERMINAL_REWARD
            if evidence == TerminalSignal.SUCCESS
            else OntologyEvidenceRank.UNVERIFIED,
            f"terminal_signal:{evidence.value}",
        )
    if isinstance(evidence, (int, OntologyEvidenceRank)):
        return int(evidence), f"rank:{int(evidence)}"
    if isinstance(evidence, str):
        ev_lower = evidence.lower()
        if any(tok in ev_lower for tok in ("lethal", "death", "fatal", "hazard")):
            return OntologyEvidenceRank.LETHAL, evidence
        if "terminal" in ev_lower or "reward" in ev_lower or "success" in ev_lower:
            return OntologyEvidenceRank.TERMINAL_REWARD, evidence
        if (
            "interaction" in ev_lower
            or "contact" in ev_lower
            or "displacement" in ev_lower
            or "push" in ev_lower
        ):
            return OntologyEvidenceRank.INTERACTION, evidence
        if "collision" in ev_lower or "bump" in ev_lower or "wall" in ev_lower:
            return OntologyEvidenceRank.COLLISION, evidence
        if "hydration" in ev_lower or "prior" in ev_lower or "background" in ev_lower:
            return OntologyEvidenceRank.HYDRATION_DEFAULT, evidence
        if ev_lower.startswith("topology:"):
            return OntologyEvidenceRank.HYDRATION_DEFAULT + 10, evidence
        if (
            "traversal" in ev_lower
            or "step" in ev_lower
            or "walk" in ev_lower
            or "free_space" in ev_lower
        ):
            return OntologyEvidenceRank.TRAVERSAL, evidence
        if "candidate" in ev_lower or "hypothesis" in ev_lower or "heuristic" in ev_lower:
            return OntologyEvidenceRank.CANDIDATE, evidence
        return OntologyEvidenceRank.HEURISTIC, evidence
    if isinstance(evidence, dict):
        if (
            evidence.get("has_terminal_reward") is True
            or evidence.get("terminal") == TerminalSignal.SUCCESS
            or evidence.get("provenance") in ("terminal_reward", "terminal_reward:success")
        ):
            return OntologyEvidenceRank.TERMINAL_REWARD, str(
                evidence.get("provenance", "terminal_reward")
            )
        if "rank" in evidence:
            return int(evidence["rank"]), str(
                evidence.get("provenance", f"rank:{evidence['rank']}")
            )
    return OntologyEvidenceRank.HEURISTIC, str(evidence)


class GovernedSetView(set):
    """Set view that enforces ontology provenance by routing mutations through governed callbacks."""

    def __init__(
        self,
        items: Any,
        on_add: Any | None = None,
        on_discard: Any | None = None,
        on_clear: Any | None = None,
    ) -> None:
        super().__init__(items)
        self._on_add = on_add
        self._on_discard = on_discard
        self._on_clear = on_clear

    def add(self, item: Any) -> None:
        super().add(item)
        if self._on_add is not None:
            self._on_add(item)

    def discard(self, item: Any) -> None:
        super().discard(item)
        if self._on_discard is not None:
            self._on_discard(item)

    def remove(self, item: Any) -> None:
        super().remove(item)
        if self._on_discard is not None:
            self._on_discard(item)

    def clear(self) -> None:
        super().clear()
        if self._on_clear is not None:
            self._on_clear()

    def update(self, *s: Any) -> None:
        for items in s:
            for item in items:
                self.add(item)


class SpelkePhysicalAxiom(StrEnum):
    """Elizabeth Spelke's infant/primate Core Knowledge physical principles."""

    FREE_SPACE = "free_space"
    COHESION = "cohesion"
    SOLIDITY_CONTINUITY = "solidity_continuity"
    CONTACT_CAUSALITY = "contact_causality"
    DIRECTIONAL_INERTIA = "directional_inertia"


@dataclass
class ActionAffordance:
    """Declared or empirically discovered motor affordance for an embodied action.

    Decouples cognitive reasoning from specific game environments, hardware drivers,
    cameras, depth sensors, or LiDAR peripherals.
    """

    action_id: Any
    name: str = ""
    requires_spatial_target: bool = False
    target_param_keys: tuple[str, ...] = ("x", "y")
    is_displacement: bool = False
    delta: tuple[int, ...] | None = None
    is_state_transform: bool = False
    is_focus_switch: bool = False
    is_undo: bool = False
    is_value_mutator: bool = False
    is_index_selector: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionalAffordance:
    """Expresses that feature is traversable / operable iff condition predicate holds.

    General form of key->door, switch->gate, cargo->receptacle causal dependencies.
    """

    feature: Any
    condition: CausalPredicate
    evidence: str = ""
    target_action: str = "traverse"
    prerequisite_feature: Any | None = None


@dataclass
class HCIRSymbolicWorldTheory:
    """Neuro-symbolic domain-agnostic world theory grounded in HCIR CausalPredicates.

    Maintains intensional causal rules and first-order predicates across context transitions,
    eliminating fragile procedural heuristics.

    Semantic Role Abstraction: Knowledge is stored at two levels:
    1. Feature-level sets (_goal_features, _walkable_features, etc.) — layout-specific bindings
       that map raw perceptual feature IDs to ontological roles.
    2. Role-level sets (_semantic_roles) — abstract role descriptions that survive across
       context transitions where feature IDs may change. When transitioning to a new context,
       feature-level bindings are cleared but role knowledge is retained as priors for
       re-grounding in the new layout.
    """

    goal_predicates: list[CausalPredicate] = field(default_factory=list)
    barrier_predicates: list[CausalPredicate] = field(default_factory=list)
    _context_epoch: int = 0
    _has_prior_context_transition: bool = False
    _awaiting_context_revalidation: bool = False
    _labels_non_stationary: bool = False
    _current_context_verified: set[tuple[str, Any]] = field(default_factory=set)
    _current_entity_distribution_stats: dict[Any, tuple[int, float, int]] = field(
        default_factory=dict
    )
    _prior_entity_distribution_stats: dict[Any, tuple[int, float, int]] = field(
        default_factory=dict
    )
    walkable_predicates: list[CausalPredicate] = field(default_factory=list)
    cargo_predicates: list[CausalPredicate] = field(default_factory=list)
    receptacle_predicates: list[CausalPredicate] = field(default_factory=list)
    actuator_predicates: list[CausalPredicate] = field(default_factory=list)
    domain_predicates: list[CausalPredicate] = field(default_factory=list)
    _domain_predicate_keys: set[tuple[Any, ...]] = field(default_factory=set, repr=False)

    # Fast-lookup cached sets (private, accessed via read-only properties)
    _goal_features: set[Any] = field(default_factory=set)
    _candidate_goal_features: set[Any] = field(default_factory=set)
    _explored_features: set[Any] = field(default_factory=set)
    _explored_loci: set[PlaceLocus] = field(default_factory=set)
    _probed_actuator_loci: set[PlaceLocus] = field(default_factory=set)
    _barrier_features: set[Any] = field(default_factory=set)
    _walkable_features: set[Any] = field(default_factory=set)
    _cargo_features: set[Any] = field(default_factory=set)
    _receptacle_features: set[Any] = field(default_factory=set)
    _actuator_features: set[Any] = field(default_factory=set)
    _avatar_features: set[Any] = field(default_factory=set)
    _conditional_affordances: dict[Any, ConditionalAffordance] = field(default_factory=dict)
    _feature_evidence: dict[tuple[str, Any], tuple[int, str]] = field(default_factory=dict)
    _barrier_eval_cache: dict[Any, bool] = field(default_factory=dict, repr=False)
    _walkable_eval_cache: dict[Any, bool] = field(default_factory=dict, repr=False)

    # Functional action categorization
    undo_actions: set[Any] = field(default_factory=set)
    focus_switch_actions: set[Any] = field(default_factory=set)
    index_selector_actions: set[Any] = field(default_factory=set)
    value_mutator_actions: set[Any] = field(default_factory=set)
    active_manipulandum_id: str | None = None
    last_effector_target_coord: tuple[int, int] | None = None
    last_effector_target_entity_id: str | None = None

    # Telic global board goal induction
    telic_goal_type: str = ""  # e.g. "deplete", "navigate", "cargo"
    telic_target_feature: Any | None = None

    # Elizabeth Spelke's Core Knowledge physical axioms
    spelke_axioms: dict[SpelkePhysicalAxiom, CausalPredicate] = field(default_factory=dict)

    # Tool-barrier empirical affinities and distal actuation ledger
    tool_barrier_affinities: dict[Any, set[Any]] = field(default_factory=dict)
    tool_barrier_incompatibilities: dict[Any, set[Any]] = field(default_factory=dict)
    distal_actuations: list[dict[str, Any]] = field(default_factory=list)
    relational_confidence_map: dict[str, float] = field(default_factory=dict)

    # ── Semantic Role Abstraction (Feature→Role Transfer Layer) ──────────
    # Bidirectional mapping between raw perceptual feature IDs and abstract semantic roles.
    # Roles (e.g. "goal", "key", "door", "hazard", "floor") survive across context transitions
    # where feature IDs may be relabeled; feature bindings are layout-specific.
    _feature_to_role: dict[Any, str] = field(default_factory=dict)
    _role_to_features: dict[str, set[Any]] = field(default_factory=dict)
    # Structural role priors: count how many times each role was observed across episodes.
    # In a new context, these priors guide re-grounding of unknown features.
    _role_priors: dict[str, int] = field(default_factory=dict)
    # Conditional affordance templates (role-level, survive relabeling)
    _role_conditional_templates: list[dict[str, Any]] = field(default_factory=list)

    @property
    def goal_features(self) -> frozenset[Any]:
        """Empirically verified confirmed goal features (read-only)."""
        return frozenset(self._goal_features)

    @property
    def candidate_goal_features(self) -> frozenset[Any]:
        """Candidate goal hypotheses pending empirical verification (read-only)."""
        return frozenset(self._candidate_goal_features)

    @property
    def explored_features(self) -> frozenset[Any]:
        """Features that have been collected, visited, or epistemically satisfied (read-only)."""
        return frozenset(self._explored_features)

    @property
    def explored_loci(self) -> frozenset[PlaceLocus]:
        """Loci that have been visited or epistemically satisfied (read-only)."""
        return frozenset(self._explored_loci)

    @property
    def explored_positions(self) -> frozenset[PlaceLocus]:
        """Backward-compatible alias for explored_loci."""
        return frozenset(self._explored_loci)

    # ── Semantic Role Abstraction API ────────────────────────────────────

    def bind_feature_to_role(self, feature_id: Any, role: str) -> None:
        """Bind a raw perceptual feature ID to an abstract semantic role.

        If the feature was already bound to a different role, the old binding is replaced.
        Roles: 'goal', 'barrier', 'walkable', 'key', 'door', 'hazard', 'floor', 'cargo',
               'receptacle', 'actuator', 'avatar'.

        When role-level conditional affordance templates exist (from a prior context's
        ``snapshot_conditional_affordance_templates``), this method lazily re-instantiates
        any template whose *both* sides (feature role + prerequisite role) are now bound,
        providing automatic cross-context transfer of key→door style affordances.
        """
        old_role = self._feature_to_role.get(feature_id)
        if old_role == role:
            return
        if old_role is not None:
            self._role_to_features.get(old_role, set()).discard(feature_id)
        self._feature_to_role[feature_id] = role
        self._role_to_features.setdefault(role, set()).add(feature_id)
        self._role_priors[role] = self._role_priors.get(role, 0) + 1

        # Lazy conditional affordance re-instantiation from role-level templates.
        # Only fires when templates were snapshotted at a prior context boundary.
        if self._role_conditional_templates:
            self._try_instantiate_templates_for_role(role, feature_id)

    def _try_instantiate_templates_for_role(self, role: str, feature_id: Any) -> None:
        """Instantiate any conditional affordance templates whose *both* sides are now bound.

        Called internally by ``bind_feature_to_role`` to lazily re-create conditional
        affordances (e.g. key→door) when a new context assigns different feature IDs to
        the same semantic roles.
        """
        for tmpl in self._role_conditional_templates:
            if tmpl["feature_role"] == role:
                # This feature matches the template's target role — look for prerequisite
                prereq_role = tmpl.get("prerequisite_role")
                if prereq_role is None:
                    continue
                prereq_feats = self._role_to_features.get(prereq_role, set())
                for pf in prereq_feats:
                    if feature_id not in self._conditional_affordances:
                        self.induce_conditional_affordance(
                            feature=feature_id,
                            condition=CausalPredicate(
                                variable="inventory",
                                operator="contains",
                                value=pf,
                            ),
                            evidence=tmpl.get("evidence", "role_template_transfer"),
                            prerequisite_feature=pf,
                        )
            elif tmpl.get("prerequisite_role") == role:
                # This feature matches the template's prerequisite role — look for target
                target_role = tmpl["feature_role"]
                target_feats = self._role_to_features.get(target_role, set())
                for tf in target_feats:
                    if tf not in self._conditional_affordances:
                        self.induce_conditional_affordance(
                            feature=tf,
                            condition=CausalPredicate(
                                variable="inventory",
                                operator="contains",
                                value=feature_id,
                            ),
                            evidence=tmpl.get("evidence", "role_template_transfer"),
                            prerequisite_feature=feature_id,
                        )

    def get_role(self, feature_id: Any) -> str | None:
        """Return the semantic role bound to a feature ID, or None if unbound."""
        return self._feature_to_role.get(feature_id)

    def get_features_for_role(self, role: str) -> frozenset[Any]:
        """Return all feature IDs currently bound to the given role."""
        return frozenset(self._role_to_features.get(role, set()))

    @property
    def role_priors(self) -> dict[str, int]:
        """Read-only view of accumulated role observation counts."""
        return dict(self._role_priors)

    @property
    def known_role_count(self) -> int:
        """Number of distinct semantic roles observed across all episodes."""
        return len(self._role_priors)

    def has_role_prior(self, role: str) -> bool:
        """Whether the given role has ever been observed in any prior episode."""
        return self._role_priors.get(role, 0) > 0

    def snapshot_conditional_affordance_templates(self) -> None:
        """Snapshot current conditional affordances as role-level templates for cross-context transfer.

        Called before a context transition so that feature-specific affordances (e.g., feature 3
        unlocks feature 2) are lifted to role-level templates (e.g., 'key' unlocks 'door').
        """
        self._role_conditional_templates.clear()
        for feat, aff in self._conditional_affordances.items():
            feat_role = self._feature_to_role.get(feat)
            prereq_role = (
                self._feature_to_role.get(aff.prerequisite_feature)
                if aff.prerequisite_feature is not None
                else None
            )
            if feat_role is not None:
                self._role_conditional_templates.append(
                    {
                        "feature_role": feat_role,
                        "prerequisite_role": prereq_role,
                        "target_action": aff.target_action,
                        "evidence": f"role_template:{aff.evidence}",
                    }
                )

    def apply_role_priors_to_new_features(self, feature_id: Any, bg_feature: Any = 0) -> str | None:
        """Attempt to assign a role to an unknown feature using structural role priors.

        Called during perception when a feature is first encountered in a new context.
        Returns the assigned role if a prior match is found, else None.
        """
        if feature_id in self._feature_to_role or feature_id == bg_feature:
            return self._feature_to_role.get(feature_id)
        # Unknown feature — cannot assign role from prior without evidence
        return None

    def rebind_role_from_evidence(self, feature_id: Any, role: str, evidence: str = "") -> None:
        """Re-bind a feature to a role when empirical evidence confirms the mapping.

        This is the primary mechanism for cross-context transfer: when the engine discovers
        that feature 7 behaves like a 'key' (e.g., it unlocks a barrier), it binds
        feature 7 to role 'key'. Any conditional affordance templates involving role 'key'
        are then instantiated with feature 7.
        """
        self.bind_feature_to_role(feature_id, role)
        # Re-instantiate conditional affordance templates that reference this role
        for tmpl in self._role_conditional_templates:
            if tmpl["feature_role"] == role:
                # This feature matches the template's target role — look for prerequisite
                prereq_role = tmpl.get("prerequisite_role")
                prereq_feats = (
                    self._role_to_features.get(prereq_role, set()) if prereq_role else set()
                )
                for pf in prereq_feats:
                    if feature_id not in self._conditional_affordances:
                        self.induce_conditional_affordance(
                            feature=feature_id,
                            condition=CausalPredicate(
                                variable="inventory",
                                operator="contains",
                                value=pf,
                            ),
                            evidence=tmpl.get("evidence", "role_template"),
                            prerequisite_feature=pf,
                        )
            elif tmpl.get("prerequisite_role") == role:
                # This feature matches the template's prerequisite role — look for target
                target_role = tmpl["feature_role"]
                target_feats = self._role_to_features.get(target_role, set())
                for tf in target_feats:
                    if tf not in self._conditional_affordances:
                        self.induce_conditional_affordance(
                            feature=tf,
                            condition=CausalPredicate(
                                variable="inventory",
                                operator="contains",
                                value=feature_id,
                            ),
                            evidence=tmpl.get("evidence", "role_template"),
                            prerequisite_feature=feature_id,
                        )

    def clear_feature_bindings(self) -> None:
        """Clear all feature→role bindings without clearing role priors or templates.

        Called on context transition (is_new_context=True) to reset layout-specific
        feature IDs while preserving abstract role knowledge for re-grounding.
        """
        self._feature_to_role.clear()
        self._role_to_features.clear()

    @property
    def labels_non_stationary(self) -> bool:
        """True once empirical evidence proves raw feature labels permute across levels."""
        return self._labels_non_stationary

    @property
    def has_prior_context_transition(self) -> bool:
        """True once at least one context transition has occurred."""
        return self._has_prior_context_transition

    def mark_verified_in_context(self, kind: str, feature_id: Any) -> None:
        """Record that ``(kind, feature_id)`` was empirically observed in the current context."""
        self._current_context_verified.add((kind, feature_id))

    def is_verified_in_context(self, kind: str, feature_id: Any) -> bool:
        """Return True if ``(kind, feature_id)`` was observed in the current context."""
        return (kind, feature_id) in self._current_context_verified

    def context_verified_features(self, kind: str) -> set[Any]:
        """Return feature IDs verified for ``kind`` in the current context."""
        return {feat for k, feat in self._current_context_verified if k == kind}

    def mark_cross_context_label_shift(self, reason: str = "") -> None:
        """Record empirical falsification of cross-context raw label invariance."""
        if not self._labels_non_stationary:
            logger.warning(
                "HCIRSymbolicWorldTheory: Cross-context raw label stationarity falsified (%s)",
                reason or "contradiction",
            )
        self._labels_non_stationary = True

    def begin_context_epoch(self) -> None:
        """Advance context epoch on a context boundary and mark prior labels unverified."""
        self._context_epoch += 1
        self._has_prior_context_transition = True
        self._awaiting_context_revalidation = True
        self._prior_entity_distribution_stats = dict(self._current_entity_distribution_stats)
        self._current_entity_distribution_stats.clear()
        self._current_context_verified.clear()

    @staticmethod
    def _compute_grid_structural_stats(
        grid: np.ndarray,
        bg_feature: Any = 0,
        avatar_features: set[Any] | None = None,
    ) -> dict[Any, tuple[int, float, int]]:
        """Summarize per-feature count and boundary concentration for invariance checks."""
        if not isinstance(grid, np.ndarray) or grid.ndim != 2 or grid.size == 0:
            return {}
        H, W = grid.shape
        total_cells = H * W
        av = avatar_features or set()
        if np.issubdtype(grid.dtype, np.integer):
            amin, amax = int(grid.min()), int(grid.max())
            if 0 <= amin and amax < 256:
                bc = np.bincount(grid.ravel(), minlength=amax + 1)
                vals_arr = np.flatnonzero(bc)
                counts_arr = bc[vals_arr]
                if H >= 2 and W >= 2:
                    border_1d = np.concatenate(
                        [grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1]]
                    )
                else:
                    border_1d = grid.ravel()
                b_bc = np.bincount(border_1d, minlength=amax + 1)
                stats: dict[Any, tuple[int, float, int]] = {}
                for v_int, cnt_int in zip(vals_arr.tolist(), counts_arr.tolist()):
                    if v_int == bg_feature or v_int in av:
                        continue
                    stats[v_int] = (cnt_int, int(b_bc[v_int]) / max(1, cnt_int), total_cells)
                return stats
        border_mask = np.zeros((H, W), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        vals, counts = np.unique(grid, return_counts=True)
        stats = {}
        for v_np, cnt_np in zip(vals.tolist(), counts.tolist(), strict=False):
            if v_np == bg_feature or v_np in av:
                continue
            border_cnt = int(np.count_nonzero((grid == v_np) & border_mask))
            stats[int(v_np)] = (int(cnt_np), border_cnt / max(1, int(cnt_np)), total_cells)
        return stats

    def observe_entity_distribution(
        self,
        stats: dict[Any, tuple[int, float, int]],
    ) -> bool:
        """Record domain-agnostic entity distribution stats; return True on role inversion.

        Expects ``stats`` mapping ``feature_id -> (count, boundary_fraction, total_sites)``.
        Checks only features actively present (count >= 1) so disjoint out-of-vocabulary
        feature sets (e.g. visual transfer) do not false-trigger.
        """
        self._current_entity_distribution_stats = dict(stats)
        if not self._awaiting_context_revalidation:
            return False
        self._awaiting_context_revalidation = False
        if not self._prior_entity_distribution_stats or not stats:
            return False

        def _is_structural_boundary(cnt: int, bfrac: float, tot: int) -> bool:
            return cnt >= max(8, tot // 10) and bfrac >= 0.25

        def _is_sparse_interior(cnt: int, bfrac: float, _tot: int) -> bool:
            return 1 <= cnt <= 4 and bfrac < 0.25

        for feat, (cur_cnt, cur_bfrac, cur_tot) in stats.items():
            prior = self._prior_entity_distribution_stats.get(feat)
            if prior is None:
                continue
            p_cnt, p_bfrac, p_tot = prior
            # Sparse entity becoming perimeter/structural substrate (or vice-versa)
            if _is_sparse_interior(p_cnt, p_bfrac, p_tot) and _is_structural_boundary(
                cur_cnt, cur_bfrac, cur_tot
            ):
                return True
            if _is_structural_boundary(p_cnt, p_bfrac, p_tot) and _is_sparse_interior(
                cur_cnt, cur_bfrac, cur_tot
            ):
                return True
        return False

    def observe_grid_structure(
        self,
        grid: np.ndarray,
        bg_feature: Any = 0,
        avatar_features: set[Any] | None = None,
    ) -> bool:
        """Lattice convenience wrapper delegating to ``observe_entity_distribution``."""
        cur_stats = self._compute_grid_structural_stats(
            grid, bg_feature=bg_feature, avatar_features=avatar_features
        )
        return self.observe_entity_distribution(cur_stats)

    def clear_context_feature_ontology(
        self, bg_feature: Any = 0, preserve_current_context: bool = False
    ) -> None:
        """Reset layout/label-bound feature ontology while preserving abstract role knowledge."""
        keep_goal = self.context_verified_features("goal") if preserve_current_context else set()
        keep_bar = self.context_verified_features("barrier") if preserve_current_context else set()
        keep_walk = (
            self.context_verified_features("walkable") if preserve_current_context else set()
        )
        keep_walk.add(bg_feature)
        keep_cargo = self.context_verified_features("cargo") if preserve_current_context else set()
        keep_rec = (
            self.context_verified_features("receptacle") if preserve_current_context else set()
        )
        keep_act = self.context_verified_features("actuator") if preserve_current_context else set()
        keep_cond = (
            self.context_verified_features("conditional") if preserve_current_context else set()
        )

        def _keep_preds(
            preds: list[CausalPredicate], keep_feats: set[Any]
        ) -> list[CausalPredicate]:
            if not keep_feats:
                return []
            return [p for p in preds if p.variable == "feature_id" and p.value in keep_feats]

        self._goal_features = set(keep_goal)
        self._candidate_goal_features = set()
        self.goal_predicates = _keep_preds(self.goal_predicates, keep_goal)

        self._barrier_features = set(keep_bar)
        self.barrier_predicates = _keep_preds(self.barrier_predicates, keep_bar)

        self._walkable_features = set(keep_walk)
        self.walkable_predicates = _keep_preds(self.walkable_predicates, keep_walk)

        self._cargo_features = set(keep_cargo)
        self.cargo_predicates = _keep_preds(self.cargo_predicates, keep_cargo)

        self._receptacle_features = set(keep_rec)
        self.receptacle_predicates = _keep_preds(self.receptacle_predicates, keep_rec)

        self._actuator_features = set(keep_act)
        self.actuator_predicates = _keep_preds(self.actuator_predicates, keep_act)

        self._conditional_affordances = {
            k: v for k, v in self._conditional_affordances.items() if k in keep_cond
        }
        self._explored_features.clear()
        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()

        keep_ev_map = {
            "goal": keep_goal,
            "barrier": keep_bar,
            "walkable": keep_walk,
            "cargo": keep_cargo,
            "receptacle": keep_rec,
            "actuator": keep_act,
            "avatar": self._avatar_features,
        }
        self._feature_evidence = {
            (k, f): v
            for (k, f), v in self._feature_evidence.items()
            if k in keep_ev_map and f in keep_ev_map[k]
        }

        if preserve_current_context:
            verified_feats = {f for _, f in self._current_context_verified}
            stale_feats = [f for f in self._feature_to_role if f not in verified_feats]
            for f in stale_feats:
                old_r = self._feature_to_role.pop(f, None)
                if old_r is not None and old_r in self._role_to_features:
                    self._role_to_features[old_r].discard(f)
        else:
            self.clear_feature_bindings()

    def mark_feature_explored(self, feature: Any, reason: str = "") -> None:
        """Mark a feature as epistemically explored so curiosity does not re-target it."""
        self._explored_features.add(feature)
        logger.info(
            "HCIRSymbolicWorldTheory: Marked feature %s explored (reason=%s)", feature, reason
        )

    def is_feature_explored(self, feature: Any) -> bool:
        return feature in self._explored_features

    def mark_locus_explored(self, locus: PlaceLocus, reason: str = "") -> None:
        """Mark a spatial or symbolic locus as epistemically explored."""
        self._explored_loci.add(locus)
        if reason == "actuator_probed":
            self._probed_actuator_loci.add(locus)
        logger.info("HCIRSymbolicWorldTheory: Marked locus %s explored (reason=%s)", locus, reason)

    def mark_position_explored(self, pos: PlaceLocus, reason: str = "") -> None:
        """Backward-compatible alias for mark_locus_explored."""
        self.mark_locus_explored(pos, reason=reason)

    def unmark_locus_explored(self, locus: PlaceLocus) -> None:
        """Unmark a locus from explored markers when world state unlocks it."""
        self._explored_loci.discard(locus)
        self._probed_actuator_loci.discard(locus)

    def unmark_position_explored(self, pos: PlaceLocus) -> None:
        """Backward-compatible alias for unmark_locus_explored."""
        self.unmark_locus_explored(pos)

    def induce_relational_cue_pair(self, feature_id: Any, locations: Sequence[PlaceLocus]) -> None:
        """Record a relational co-occurrence pair linking an indicator cue and an actuator (WP-C2)."""
        self.record_causal_predicate(
            CausalPredicate(
                variable=f"feature_{feature_id}_relational_pair",
                operator="==",
                value=True,
                relation="cue_actuator_resonance",
                target_entity=str(list(locations)),
                confidence=0.85,
            )
        )

    def is_locus_explored(self, locus: PlaceLocus) -> bool:
        """Return True if a locus has been marked as explored."""
        return locus in self._explored_loci

    def is_position_explored(self, pos: PlaceLocus) -> bool:
        """Backward-compatible alias for is_locus_explored."""
        return self.is_locus_explored(pos)

    def is_locus_actuator_probed(self, locus: PlaceLocus) -> bool:
        """Return True when an adjacent interact/actuator action failed at locus."""
        return locus in self._probed_actuator_loci

    def is_position_actuator_probed(self, pos: PlaceLocus) -> bool:
        """Backward-compatible alias for is_locus_actuator_probed."""
        return self.is_locus_actuator_probed(pos)

    @property
    def barrier_features(self) -> frozenset[Any]:
        """Empirically verified barrier / obstacle features (read-only)."""
        return frozenset(self._barrier_features)

    @property
    def walkable_features(self) -> frozenset[Any]:
        """Empirically verified safe / walkable features (read-only)."""
        return frozenset(self._walkable_features)

    @property
    def cargo_features(self) -> frozenset[Any]:
        """Empirically verified pushable cargo features (read-only)."""
        return frozenset(self._cargo_features)

    @property
    def receptacle_features(self) -> frozenset[Any]:
        """Empirically verified receptacle / dock features (read-only)."""
        return frozenset(self._receptacle_features)

    @property
    def actuator_features(self) -> frozenset[Any]:
        """Empirically verified distal actuator / switch features (read-only)."""
        return frozenset(self._actuator_features)

    @property
    def avatar_features(self) -> frozenset[Any]:
        """Empirically verified self-avatar features (read-only)."""
        return frozenset(self._avatar_features)

    @property
    def target_features(self) -> frozenset[Any]:
        """Alias for goal_features (read-only)."""
        return frozenset(self._goal_features)

    @property
    def conditional_affordances(self) -> dict[Any, ConditionalAffordance]:
        """Learned conditional affordances (read-only mapping)."""
        return dict(self._conditional_affordances)

    def induce_conditional_affordance(
        self,
        feature: Any,
        condition: CausalPredicate,
        evidence: str = "",
        prerequisite_feature: Any | None = None,
        target_action: str = "traverse",
    ) -> None:
        """Induce conditional affordance for feature governed by condition predicate."""
        bg = getattr(self, "_bg_feature", None)
        if (bg is not None and feature == bg) or feature in self._walkable_features:
            return
        prereq = prerequisite_feature if prerequisite_feature is not None else condition.value
        aff = ConditionalAffordance(
            feature=feature,
            condition=condition,
            evidence=evidence,
            target_action=target_action,
            prerequisite_feature=prereq,
        )
        self._conditional_affordances[feature] = aff
        self.record_causal_predicate(
            CausalPredicate(
                variable=f"affordance_gated:{feature}",
                operator="==",
                value=prereq,
                relation="requires_prerequisite",
                target_entity=str(prereq),
                metadata={"evidence": evidence, "feature": feature},
            )
        )
        if prereq is not None and prereq not in self._avatar_features:
            if self.get_role(prereq) in (None, "walkable"):
                self.bind_feature_to_role(prereq, "actuator")
        if feature not in self._avatar_features and feature not in self._goal_features:
            if self.get_role(feature) in (None, "walkable"):
                self.bind_feature_to_role(feature, "barrier")
        ev_str = str(evidence).lower()
        if "template" not in ev_str and "hydration" not in ev_str:
            self.mark_verified_in_context("conditional", feature)
        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()

    def retract_conditional_affordance(self, feature: Any, reason: str = "") -> None:
        """Retract conditional affordance for feature."""
        if feature in self._conditional_affordances:
            del self._conditional_affordances[feature]
            self._barrier_eval_cache.clear()
            self._walkable_eval_cache.clear()

    def get_conditional_affordance(self, feature: Any) -> ConditionalAffordance | None:
        """Retrieve conditional affordance for feature if defined."""
        return self._conditional_affordances.get(feature)

    def is_conditionally_traversable(
        self, feature: Any, agent_state: dict[str, Any] | None = None
    ) -> bool:
        """Returns True if feature is traversable under current agent state."""
        aff = self._conditional_affordances.get(feature)
        if aff is None:
            return self.is_walkable(feature, agent_state=agent_state)
        if agent_state is None:
            return False
        return aff.condition.evaluate(agent_state)

    @property
    def causal_predicates(self) -> list[CausalPredicate]:
        """Declarative causal predicates recorded in the symbolic world theory."""
        return self.domain_predicates

    @staticmethod
    def _predicate_key(predicate: CausalPredicate) -> tuple[Any, ...]:
        return (
            predicate.variable,
            predicate.operator,
            repr(predicate.value),
            predicate.relation,
            predicate.target_entity,
        )

    def record_causal_predicate(self, predicate: CausalPredicate) -> None:
        """Record an empirical causal predicate into the symbolic world theory."""
        if len(self._domain_predicate_keys) != len(self.domain_predicates):
            self._domain_predicate_keys = {self._predicate_key(p) for p in self.domain_predicates}
        key = self._predicate_key(predicate)
        if key not in self._domain_predicate_keys:
            self._domain_predicate_keys.add(key)
            self.domain_predicates.append(predicate)

    def record_tool_barrier_affinity(self, tool_feat: Any, barrier_feat: Any) -> None:
        """Record empirical causal resonance where tool_feat cleared barrier_feat."""
        self.tool_barrier_affinities.setdefault(tool_feat, set()).add(barrier_feat)

    def record_tool_barrier_incompatibility(self, tool_feat: Any, barrier_feat: Any) -> None:
        """Record empirical causal failure where tool_feat failed to clear barrier_feat."""
        self.tool_barrier_incompatibilities.setdefault(tool_feat, set()).add(barrier_feat)

    def has_tool_barrier_affinity(
        self, tool_feat: Any | None, barrier_feat: Any | None = None
    ) -> bool:
        """Check if tool_feat has empirical resonance with barrier_feat (or any barrier if barrier_feat is None)."""
        if tool_feat is None:
            return False
        affinities = self.tool_barrier_affinities.get(tool_feat)
        if not affinities:
            return False
        if barrier_feat is None:
            return len(affinities) > 0
        return barrier_feat in affinities

    def is_tool_barrier_incompatible(
        self, tool_feat: Any | None, barrier_feat: Any | None = None
    ) -> bool:
        """Check if tool_feat has confirmed incompatibility with barrier_feat."""
        if tool_feat is None:
            return False
        incompat = self.tool_barrier_incompatibilities.get(tool_feat)
        if not incompat:
            return False
        if barrier_feat is None:
            return len(incompat) > 0
        return barrier_feat in incompat

    def record_distal_actuation(
        self,
        trigger_feature: int | tuple[int, int] | None = None,
        target_feature: int | tuple[int, int] | None = None,
        affected_pos: tuple[int, int] | None = None,
        trigger_coord: tuple[int, int] | None = None,
        target_coord: tuple[int, int] | None = None,
        trigger_feat: int | None = None,
        target_feat: int | None = None,
    ) -> None:
        """Record distal causal coupling between actuator switch and target gate."""
        self.distal_actuations.append(
            {
                "trigger_coord": trigger_coord
                or (trigger_feature if isinstance(trigger_feature, tuple) else None),
                "target_coord": target_coord
                or affected_pos
                or (target_feature if isinstance(target_feature, tuple) else None),
                "trigger_feat": trigger_feat
                or (trigger_feature if isinstance(trigger_feature, int) else None),
                "target_feat": target_feat
                or (target_feature if isinstance(target_feature, int) else None),
            }
        )

    def is_avatar_feature(self, feature_id: Any) -> bool:
        """True if feature_id matches any registered avatar feature."""
        return feature_id in self._avatar_features

    def is_goal(self, entity: SpatialEntity | int | str | Any) -> bool:
        """True if entity or feature satisfies any induced goal predicate or confirmed goal feature."""
        if isinstance(entity, (int, np.integer, str)):
            feat = int(entity) if isinstance(entity, (int, np.integer)) else entity
            if feat in self._avatar_features:
                return False
            if feat in self._goal_features:
                return True
            if not self._goal_features and feat in self._candidate_goal_features:
                return True
            props = {"feature_id": feat, "visual_id": feat, "id": feat, "name": str(feat)}
            return any(pred.evaluate(props) for pred in self.goal_predicates)

        feat = getattr(entity, "feature_id", getattr(entity, "id", getattr(entity, "name", None)))
        role = getattr(entity, "role", None)
        if feat in self._avatar_features:
            return False
        if role == EntityRole.GOAL or str(role).lower() in ("goal", "entityrole.goal", "target"):
            return True
        if feat in self._goal_features:
            return True
        if not self._goal_features and feat in self._candidate_goal_features:
            return True
        props = {
            "feature_id": feat,
            "visual_id": feat,
            "id": feat,
            "name": str(getattr(entity, "name", feat)),
            "area": getattr(entity, "area", 1),
            "role": str(role),
            **getattr(entity, "properties", {}),
        }
        return any(pred.evaluate(props) for pred in self.goal_predicates)

    def is_goal_feature(self, feature_id: Any) -> bool:
        """True if feature satisfies any induced goal predicate or confirmed goal feature."""
        return self.is_goal(feature_id)

    def is_barrier(self, feature_id: Any, agent_state: dict[str, Any] | None = None) -> bool:
        """True if feature satisfies any induced barrier predicate (enforcing HCIR Attractor Invariant)."""
        if feature_id in self._avatar_features:
            return False
        if feature_id in self._goal_features:
            return False
        if not self._goal_features and feature_id in self._candidate_goal_features:
            return False
        if feature_id in self._conditional_affordances:
            if agent_state is not None:
                return not self.is_conditionally_traversable(feature_id, agent_state)
            return True
        if feature_id in self._barrier_features:
            return True
        cached = self._barrier_eval_cache.get(feature_id)
        if cached is not None:
            return cached
        props = {"feature_id": feature_id, "visual_id": feature_id, "id": feature_id}
        res = any(pred.evaluate(props) for pred in self.barrier_predicates)
        self._barrier_eval_cache[feature_id] = res
        return res

    def is_walkable(self, feature_id: Any, agent_state: dict[str, Any] | None = None) -> bool:
        """True if feature satisfies any induced walkable space predicate."""
        if feature_id in self._conditional_affordances:
            if agent_state is not None:
                return self.is_conditionally_traversable(feature_id, agent_state)
            return False
        if feature_id in self._walkable_features:
            return True
        cached = self._walkable_eval_cache.get(feature_id)
        if cached is not None:
            return cached
        props = {"feature_id": feature_id, "visual_id": feature_id, "id": feature_id}
        res = any(pred.evaluate(props) for pred in self.walkable_predicates)
        self._walkable_eval_cache[feature_id] = res
        return res

    def is_cargo(self, entity: SpatialEntity | int | str | Any) -> bool:
        """True if entity or feature satisfies any pushable cargo predicate."""
        if isinstance(entity, (int, np.integer, str)):
            feat = int(entity) if isinstance(entity, (int, np.integer)) else entity
            if feat in self._cargo_features:
                return True
            props = {"feature_id": feat, "visual_id": feat, "id": feat, "name": str(feat)}
            return any(pred.evaluate(props) for pred in self.cargo_predicates)

        feat = getattr(entity, "feature_id", getattr(entity, "id", None))
        if feat in self._cargo_features:
            return True
        role = getattr(entity, "role", None)
        if role in (EntityRole.MANIPULABLE, getattr(EntityRole, "CARGO", None)) or str(
            role
        ).lower() in ("cargo", "manipulable", "pushable"):
            return True
        props = {
            "feature_id": feat,
            "visual_id": feat,
            "id": feat,
            "area": getattr(entity, "area", 1),
            "role": str(role),
            **getattr(entity, "properties", {}),
        }
        return any(pred.evaluate(props) for pred in self.cargo_predicates)

    def is_receptacle(self, entity: SpatialEntity | int | str | Any) -> bool:
        """True if entity or feature satisfies any receptacle predicate."""
        if isinstance(entity, (int, np.integer, str)):
            feat = int(entity) if isinstance(entity, (int, np.integer)) else entity
            if feat in self._receptacle_features:
                return True
            props = {"feature_id": feat, "visual_id": feat, "id": feat, "name": str(feat)}
            return any(pred.evaluate(props) for pred in self.receptacle_predicates)

        feat = getattr(entity, "feature_id", getattr(entity, "id", None))
        if feat in self._receptacle_features:
            return True
        role = getattr(entity, "role", None)
        if role == EntityRole.RECEPTACLE or str(role).lower() in (
            "receptacle",
            "entityrole.receptacle",
            "dock",
        ):
            return True
        props = {
            "feature_id": feat,
            "visual_id": feat,
            "id": feat,
            "area": getattr(entity, "area", 1),
            "role": str(role),
            **getattr(entity, "properties", {}),
        }
        return any(pred.evaluate(props) for pred in self.receptacle_predicates)

    def induce_goal(self, feature_id: Any, evidence: Any = None) -> bool:
        """Induce universal goal predicate with minimum description length and evidence gating.

        A confirmed goal strictly requires terminal-reward provenance (rank >= TERMINAL_REWARD).
        If evidence is insufficient, it is registered as a candidate goal only.
        A confirmed goal will not delete an existing barrier unless its evidence rank
        strictly outranks the barrier's evidence.
        """
        if feature_id in self._avatar_features:
            return False

        rank, ev_str = parse_evidence(evidence or TerminalSignal.SUCCESS)
        if rank < OntologyEvidenceRank.TERMINAL_REWARD:
            logger.info(
                "HCIRSymbolicWorldTheory: Cannot induce confirmed goal for feature %s without "
                "terminal-reward evidence (rank=%d, ev=%s); registering as candidate goal only",
                feature_id,
                rank,
                ev_str,
            )
            self.induce_candidate_goal(feature_id, evidence=evidence)
            return False

        if feature_id in self._barrier_features:
            barrier_rank, bar_ev = self._feature_evidence.get(
                ("barrier", feature_id), (OntologyEvidenceRank.COLLISION, "collision")
            )
            if rank <= barrier_rank:
                logger.info(
                    "HCIRSymbolicWorldTheory: Cannot delete barrier %s for goal: goal evidence rank %d "
                    "does not outrank barrier rank %d (%s)",
                    feature_id,
                    rank,
                    barrier_rank,
                    bar_ev,
                )
                return False
            self._barrier_features.discard(feature_id)
            self.barrier_predicates = [
                p
                for p in self.barrier_predicates
                if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
            ]
            self._barrier_eval_cache.clear()
            self._walkable_eval_cache.clear()

        if feature_id not in self._goal_features:
            self._goal_features.add(feature_id)
            self._candidate_goal_features.add(feature_id)
            self._feature_evidence[("goal", feature_id)] = (rank, ev_str)
            self.bind_feature_to_role(feature_id, "goal")
            pred = CausalPredicate(
                variable="feature_id",
                operator="==",
                value=feature_id,
                metadata={"evidence": ev_str, "rank": rank, "role": "goal"},
            )
            self.goal_predicates.append(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Confirmed Goal Rule: %s (provenance=%s, rank=%d)",
                pred.describe(),
                ev_str,
                rank,
            )
        self._walkable_features.add(feature_id)
        self._feature_evidence[("walkable", feature_id)] = (rank, ev_str)
        # Note: do NOT bind_feature_to_role("walkable") here — the primary role is "goal".
        # The feature is added to _walkable_features for functional walkability queries,
        # but its semantic role for cross-context transfer remains "goal".
        return True

    def induce_candidate_goal(self, feature_id: Any, evidence: Any = None) -> bool:
        """Induce candidate goal hypothesis pending terminal verification."""
        if (
            feature_id in self._avatar_features
            or feature_id in self._cargo_features
            or feature_id in self._barrier_features
        ):
            return False
        rank, ev_str = parse_evidence(evidence)
        self._candidate_goal_features.add(feature_id)
        self._feature_evidence[("candidate_goal", feature_id)] = (rank, ev_str)
        return True

    def induce_barrier(self, feature_id: Any, evidence: Any = None) -> bool:
        """Induce universal barrier predicate (enforcing HCIR Attractor Invariant)."""
        if feature_id in self._avatar_features:
            return False

        rank, ev_str = parse_evidence(evidence or "collision")
        if rank == OntologyEvidenceRank.UNVERIFIED:
            rank = OntologyEvidenceRank.COLLISION
            ev_str = "collision"

        if feature_id in self._goal_features:
            goal_rank, g_ev = self._feature_evidence.get(
                ("goal", feature_id), (OntologyEvidenceRank.TERMINAL_REWARD, "terminal_reward")
            )
            if goal_rank >= rank and rank < OntologyEvidenceRank.LETHAL:
                logger.info(
                    "HCIRSymbolicWorldTheory: Cannot induce barrier on confirmed goal %s (goal rank %d >= barrier rank %d)",
                    feature_id,
                    goal_rank,
                    rank,
                )
                return False
            self.retract_goal(feature_id, reason=ev_str)

        if feature_id in self._cargo_features:
            cargo_rank, c_ev = self._feature_evidence.get(
                ("cargo", feature_id), (OntologyEvidenceRank.INTERACTION, "cargo")
            )
            if cargo_rank >= rank:
                logger.info(
                    "HCIRSymbolicWorldTheory: Cannot induce barrier on verified cargo %s (cargo rank %d >= barrier rank %d)",
                    feature_id,
                    cargo_rank,
                    rank,
                )
                return False
            self._cargo_features.discard(feature_id)
            self.cargo_predicates = [
                p
                for p in self.cargo_predicates
                if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
            ]

        if feature_id in self._walkable_features:
            walk_rank, w_ev = self._feature_evidence.get(
                ("walkable", feature_id), (OntologyEvidenceRank.TRAVERSAL, "traversal")
            )
            if walk_rank > rank:
                logger.info(
                    "HCIRSymbolicWorldTheory: Cannot induce barrier on walkable feature %s (walk rank %d > barrier rank %d)",
                    feature_id,
                    walk_rank,
                    rank,
                )
                return False
            self._walkable_features.discard(feature_id)

        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()
        if "hydration" not in ev_str.lower() and "template" not in ev_str.lower():
            self.mark_verified_in_context("barrier", feature_id)
        existing_rank, _ = self._feature_evidence.get(("barrier", feature_id), (0, ""))
        if feature_id not in self._barrier_features:
            self._barrier_features.add(feature_id)
            self._feature_evidence[("barrier", feature_id)] = (rank, ev_str)
            self.bind_feature_to_role(feature_id, "barrier")
            pred = CausalPredicate(
                variable="feature_id",
                operator="==",
                value=feature_id,
                metadata={"evidence": ev_str, "rank": rank, "role": "barrier"},
            )
            self.barrier_predicates.append(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Barrier Rule: %s (provenance=%s, rank=%d)",
                pred.describe(),
                ev_str,
                rank,
            )
        elif rank > existing_rank:
            self._feature_evidence[("barrier", feature_id)] = (rank, ev_str)
        return True

    def induce_walkable(self, feature_id: Any, evidence: Any = None) -> bool:
        """Induce universal walkable space predicate with self-correcting belief revision."""
        rank, ev_str = parse_evidence(evidence or "traversal")
        if rank == OntologyEvidenceRank.UNVERIFIED:
            rank = OntologyEvidenceRank.TRAVERSAL
            ev_str = "traversal"

        if feature_id in self._barrier_features:
            barrier_rank, bar_ev = self._feature_evidence.get(
                ("barrier", feature_id), (OntologyEvidenceRank.COLLISION, "collision")
            )
            if rank <= barrier_rank:
                logger.info(
                    "HCIRSymbolicWorldTheory: Walkable evidence rank %d does not outrank barrier rank %d for feature %s",
                    rank,
                    barrier_rank,
                    feature_id,
                )
                return False
            self._barrier_features.discard(feature_id)
            self.barrier_predicates = [
                p
                for p in self.barrier_predicates
                if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
            ]

        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()
        ev_low = ev_str.lower()
        if "hydration" not in ev_low and "template" not in ev_low and "prior" not in ev_low:
            self.mark_verified_in_context("walkable", feature_id)
        existing_rank, _ = self._feature_evidence.get(("walkable", feature_id), (0, ""))
        if feature_id not in self._walkable_features:
            self._walkable_features.add(feature_id)
            self._feature_evidence[("walkable", feature_id)] = (rank, ev_str)
            # Only bind to "walkable" role if no more specific role is already assigned
            # (goal, cargo, receptacle, actuator take priority over walkable).
            if self.get_role(feature_id) is None:
                self.bind_feature_to_role(feature_id, "walkable")
            pred = CausalPredicate(
                variable="feature_id",
                operator="==",
                value=feature_id,
                metadata={"evidence": ev_str, "rank": rank, "role": "walkable"},
            )
            self.walkable_predicates.append(pred)
        elif rank > existing_rank:
            self._feature_evidence[("walkable", feature_id)] = (rank, ev_str)
        return True

    def induce_cargo(self, feature_id: Any, evidence: Any = None) -> bool:
        """Induce pushable cargo predicate with governed belief revision."""
        rank, ev_str = parse_evidence(evidence or "cargo")
        if rank == OntologyEvidenceRank.UNVERIFIED:
            rank = OntologyEvidenceRank.INTERACTION
            ev_str = "contact_displacement"

        if feature_id in self._barrier_features:
            barrier_rank, bar_ev = self._feature_evidence.get(
                ("barrier", feature_id), (OntologyEvidenceRank.COLLISION, "collision")
            )
            if barrier_rank > rank:
                return False
            self._barrier_features.discard(feature_id)
            self.barrier_predicates = [
                p
                for p in self.barrier_predicates
                if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
            ]

        self._walkable_features.discard(feature_id)
        self._goal_features.discard(feature_id)
        self._candidate_goal_features.discard(feature_id)
        if "hydration" not in ev_str.lower() and "template" not in ev_str.lower():
            self.mark_verified_in_context("cargo", feature_id)
        if feature_id not in self._cargo_features:
            self._cargo_features.add(feature_id)
            self._feature_evidence[("cargo", feature_id)] = (rank, ev_str)
            self.bind_feature_to_role(feature_id, "cargo")
            pred = CausalPredicate(
                variable="feature_id",
                operator="==",
                value=feature_id,
                metadata={"evidence": ev_str, "rank": rank, "role": "cargo"},
            )
            self.cargo_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Cargo Rule: %s", pred.describe())
        return True

    def induce_receptacle(self, feature_id: Any, evidence: Any = None) -> bool:
        """Induce universal receptacle predicate."""
        rank, ev_str = parse_evidence(evidence or "receptacle")
        if "hydration" not in ev_str.lower() and "template" not in ev_str.lower():
            self.mark_verified_in_context("receptacle", feature_id)
        if feature_id not in self._receptacle_features:
            self._receptacle_features.add(feature_id)
            self._feature_evidence[("receptacle", feature_id)] = (rank, ev_str)
            self.bind_feature_to_role(feature_id, "receptacle")
            pred = CausalPredicate(
                variable="feature_id",
                operator="==",
                value=feature_id,
                metadata={"evidence": ev_str, "rank": rank, "role": "receptacle"},
            )
            self.receptacle_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Receptacle Rule: %s", pred.describe())
        return True

    def induce_actuator(self, feature_id: Any, evidence: Any = None) -> bool:
        """Induce distal actuator / switch affordance."""
        rank, ev_str = parse_evidence(evidence or "actuator")
        if "hydration" not in ev_str.lower() and "template" not in ev_str.lower():
            self.mark_verified_in_context("actuator", feature_id)
        if feature_id not in self._actuator_features:
            self._actuator_features.add(feature_id)
            self._feature_evidence[("actuator", feature_id)] = (rank, ev_str)
            self.bind_feature_to_role(feature_id, "actuator")
            pred = CausalPredicate(
                variable="feature_id",
                operator="==",
                value=feature_id,
                metadata={"evidence": ev_str, "rank": rank, "role": "actuator"},
            )
            self.actuator_predicates.append(pred)
            logger.info("HCIRSymbolicWorldTheory: Induced Actuator Rule: %s", pred.describe())
        return True

    def induce_avatar(self, feature_id: Any, evidence: Any = None) -> bool:
        """Register avatar feature and purge from candidate/goal/barrier sets."""
        rank, ev_str = parse_evidence(evidence or "motor_grounding")
        self._avatar_features.add(feature_id)
        self._feature_evidence[("avatar", feature_id)] = (rank, ev_str)
        self.bind_feature_to_role(feature_id, "avatar")
        self._goal_features.discard(feature_id)
        self._candidate_goal_features.discard(feature_id)
        self._barrier_features.discard(feature_id)
        return True

    def retract_goal(self, feature_id: Any, reason: str = "") -> None:
        """Retract a goal feature and its associated predicates."""
        self._goal_features.discard(feature_id)
        self._candidate_goal_features.discard(feature_id)
        self.goal_predicates = [
            p
            for p in self.goal_predicates
            if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
        ]
        self._feature_evidence.pop(("goal", feature_id), None)
        self._feature_evidence.pop(("candidate_goal", feature_id), None)

    def retract_candidate_goal(self, feature_id: Any, reason: str = "") -> None:
        """Retract a candidate goal hypothesis."""
        self._candidate_goal_features.discard(feature_id)
        self._feature_evidence.pop(("candidate_goal", feature_id), None)

    def retract_barrier(self, feature_id: Any, reason: str = "") -> None:
        """Retract a barrier feature and clear evaluation caches."""
        self._barrier_features.discard(feature_id)
        self.barrier_predicates = [
            p
            for p in self.barrier_predicates
            if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
        ]
        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()
        self._feature_evidence.pop(("barrier", feature_id), None)

    def retract_walkable(self, feature_id: Any, reason: str = "") -> None:
        """Retract a walkable feature and clear evaluation caches."""
        self._walkable_features.discard(feature_id)
        self.walkable_predicates = [
            p
            for p in self.walkable_predicates
            if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
        ]
        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()
        self._feature_evidence.pop(("walkable", feature_id), None)

    def retract_cargo(self, feature_id: Any, reason: str = "") -> None:
        """Retract a cargo feature."""
        self._cargo_features.discard(feature_id)
        self.cargo_predicates = [
            p
            for p in self.cargo_predicates
            if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
        ]
        self._feature_evidence.pop(("cargo", feature_id), None)

    def retract_receptacle(self, feature_id: Any, reason: str = "") -> None:
        """Retract a receptacle feature."""
        self._receptacle_features.discard(feature_id)
        self.receptacle_predicates = [
            p
            for p in self.receptacle_predicates
            if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
        ]
        self._feature_evidence.pop(("receptacle", feature_id), None)

    def retract_actuator(self, feature_id: Any, reason: str = "") -> None:
        """Retract an actuator feature."""
        self._actuator_features.discard(feature_id)
        self.actuator_predicates = [
            p
            for p in self.actuator_predicates
            if not (p.variable == "feature_id" and p.operator == "==" and p.value == feature_id)
        ]
        self._feature_evidence.pop(("actuator", feature_id), None)

    def clear_goals(self) -> None:
        """Clear all goal rules and cached features."""
        self._goal_features.clear()
        self._candidate_goal_features.clear()
        self.goal_predicates.clear()
        to_pop = [k for k in self._feature_evidence if k[0] in ("goal", "candidate_goal")]
        for k in to_pop:
            self._feature_evidence.pop(k, None)

    def clear_candidate_goals(self) -> None:
        """Clear all candidate goal hypotheses."""
        self._candidate_goal_features.clear()
        to_pop = [k for k in self._feature_evidence if k[0] == "candidate_goal"]
        for k in to_pop:
            self._feature_evidence.pop(k, None)

    def clear_explored_loci(self, clear_features: bool = False) -> None:
        """Clear episode-local explored locus markers (and optionally explored features on context change)."""
        self._explored_loci.clear()
        self._probed_actuator_loci.clear()
        if clear_features:
            self._explored_features.clear()

    def clear_explored_positions(self, clear_features: bool = False) -> None:
        """Backward-compatible alias for clear_explored_loci."""
        self.clear_explored_loci(clear_features=clear_features)

    def clear_barriers(self) -> None:
        """Clear all barrier rules and cached features."""
        self._barrier_features.clear()
        self.barrier_predicates.clear()
        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()
        to_pop = [k for k in self._feature_evidence if k[0] == "barrier"]
        for k in to_pop:
            self._feature_evidence.pop(k, None)

    def induce_depletion_goal(self, feature_id: Any, evidence: Any = None) -> None:
        """Induce a global board depletion goal (clear board of this feature)."""
        self.telic_goal_type = "deplete"
        self.telic_target_feature = feature_id
        self.induce_candidate_goal(feature_id, evidence=evidence or "depletion_hypothesis")
        logger.info(
            "HCIRSymbolicWorldTheory: Induced Telic Depletion Candidate Goal for feature %s (evidence=%s)",
            feature_id,
            evidence,
        )

    def induce_undo_action(self, action: Any) -> None:
        """Induce universal empirical state-rollback / undo affordance."""
        if action not in self.undo_actions:
            self.undo_actions.add(action)
            pred = CausalPredicate(
                variable="action_role",
                operator="==",
                value="undo",
                target_entity=str(action),
            )
            self.register_predicate(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Empirical UNDO Operator: Action %s", action
            )

    def induce_index_selector(self, action: Any) -> None:
        """Induce dial / cursor index navigation affordance."""
        if action not in self.index_selector_actions:
            self.index_selector_actions.add(action)
            pred = CausalPredicate(
                variable="action_role",
                operator="==",
                value="index_selector",
                target_entity=str(action),
            )
            self.register_predicate(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Empirical Index Selector: Action %s", action
            )

    def induce_value_mutator(self, action: Any) -> None:
        """Induce dial / state value mutation affordance."""
        if action not in self.value_mutator_actions:
            self.value_mutator_actions.add(action)
            pred = CausalPredicate(
                variable="action_role",
                operator="==",
                value="value_mutator",
                target_entity=str(action),
            )
            self.register_predicate(pred)
            logger.info(
                "HCIRSymbolicWorldTheory: Induced Empirical Value Mutator: Action %s", action
            )

    def register_predicate(self, predicate: CausalPredicate) -> None:
        """Register a generic or learned causal predicate into the symbolic world theory."""
        if len(self._domain_predicate_keys) != len(self.domain_predicates):
            self._domain_predicate_keys = {self._predicate_key(p) for p in self.domain_predicates}
        key = self._predicate_key(predicate)
        if key not in self._domain_predicate_keys:
            self._domain_predicate_keys.add(key)
            self.domain_predicates.append(predicate)

    def export_causal_predicates(self) -> list[CausalPredicate]:
        """Export all registered, domain, goal, barrier, and walkable causal predicates."""
        all_preds = list(self.domain_predicates)
        all_preds.extend(self.goal_predicates)
        all_preds.extend(self.barrier_predicates)
        all_preds.extend(self.walkable_predicates)
        all_preds.extend(self.cargo_predicates)
        all_preds.extend(self.receptacle_predicates)
        all_preds.extend(self.actuator_predicates)
        all_preds.extend(self.spelke_axioms.values())
        return all_preds

    def get_feature_evidence(self, kind: str, feature_id: Any) -> tuple[int, str]:
        """Return (rank, provenance) recorded for `(kind, feature_id)`."""
        return self._feature_evidence.get(
            (kind, feature_id), (OntologyEvidenceRank.UNVERIFIED, "unverified")
        )

    def has_verified_walkable(self, feature_id: Any) -> bool:
        """Return True only when `feature_id` has empirical traversal/higher walkable evidence."""
        if feature_id not in self._walkable_features:
            return False
        rank, ev_str = self.get_feature_evidence("walkable", feature_id)
        if rank >= OntologyEvidenceRank.TRAVERSAL:
            return True
        return False

    def seed_core_knowledge(self, bg_feature: Any = 0) -> None:
        """Seed domain-general Spelke core cognitive knowledge (Physics & Free Space axioms).

        Grounds infant/primate Core Knowledge systems (Spelke 1990, Carey 2009):
        1. FREE_SPACE: Environmental background affords uninhibited locomotion.
        2. SOLIDITY_CONTINUITY: Solid entity boundaries resist traversal unless actuated;
           spatiotemporal paths cannot pass through solid obstacles.
        3. CONTACT_CAUSALITY: State changes and displacements require direct spatial contact
           or grounded distal interventions.
        4. COHESION: Physical entities maintain cohesive bounded spatial extents.
        5. DIRECTIONAL_INERTIA: Displacements maintain directional continuity unless
           interrupted by collisions or external forces.
        """
        self.induce_walkable(bg_feature, evidence="hydration:free_space_prior")

        # 1. Free Space
        self.spelke_axioms[SpelkePhysicalAxiom.FREE_SPACE] = CausalPredicate(
            variable="feature_id",
            operator="==",
            value=bg_feature,
        )

        # 2. Solidity & Spatiotemporal Continuity
        self.spelke_axioms[SpelkePhysicalAxiom.SOLIDITY_CONTINUITY] = CausalPredicate(
            variable="is_solid",
            operator="==",
            value=True,
        )

        # 3. Contact Causality
        self.spelke_axioms[SpelkePhysicalAxiom.CONTACT_CAUSALITY] = CausalPredicate(
            variable="requires_contact",
            operator="==",
            value=True,
        )

        # 4. Cohesion / Boundedness
        self.spelke_axioms[SpelkePhysicalAxiom.COHESION] = CausalPredicate(
            variable="is_cohesive",
            operator="==",
            value=True,
        )

        # 5. Directional Inertia
        self.spelke_axioms[SpelkePhysicalAxiom.DIRECTIONAL_INERTIA] = CausalPredicate(
            variable="maintains_inertia",
            operator="==",
            value=True,
        )

    def evaluates_spelke_continuity(
        self,
        origin: tuple[int, int],
        destination: tuple[int, int],
        barrier_set: set[tuple[int, int]],
        grid_bounds: tuple[int, int] | None = None,
    ) -> bool:
        """Evaluate Spelke spatiotemporal continuity.

        True if there exists a barrier-free topological conduit from origin to destination.
        Returns False if solid boundaries block continuous motion without actuation.
        """
        if origin == destination:
            return True
        if origin in barrier_set or destination in barrier_set:
            return False

        r_min = min(origin[0], destination[0]) - 2
        r_max = max(origin[0], destination[0]) + 2
        c_min = min(origin[1], destination[1]) - 2
        c_max = max(origin[1], destination[1]) + 2
        if grid_bounds:
            r_min = max(0, r_min)
            c_min = max(0, c_min)
            r_max = min(grid_bounds[0] - 1, r_max)
            c_max = min(grid_bounds[1] - 1, c_max)

        from collections import deque

        queue: deque[tuple[int, int]] = deque([origin])
        visited: set[tuple[int, int]] = {origin}

        while queue:
            curr = queue.popleft()
            if curr == destination:
                return True
            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if r_min <= nr <= r_max and c_min <= nc <= c_max:
                    if nxt not in visited and nxt not in barrier_set:
                        visited.add(nxt)
                        queue.append(nxt)
        return False

    def evaluates_spelke_contact(
        self,
        agent_pos: tuple[int, int],
        target_pos: tuple[int, int],
        adjacency: Literal["4-way", "8-way"] | bool = "4-way",
    ) -> bool:
        """Evaluate Spelke mechanical contact causality.

        True if agent is adjacent to the target entity affording direct mechanical actuation.
        """
        dr = abs(agent_pos[0] - target_pos[0])
        dc = abs(agent_pos[1] - target_pos[1])
        if adjacency == "8-way":
            return max(dr, dc) == 1
        return (dr + dc) == 1

    def evaluates_spelke_cohesion(
        self,
        entity_cells: set[tuple[int, int]],
        displacement: tuple[int, int],
        barrier_set: set[tuple[int, int]],
        grid_bounds: tuple[int, int] | None = None,
    ) -> bool:
        """Evaluate Spelke Cohesion & Rigid Body Motion.

        Objects move as cohesive wholes with rigid connectivity.
        True if all constituent cells under displacement (dr, dc):
        1. Maintain mutual relative connectivity without tearing.
        2. Remain strictly inside grid_bounds.
        3. Do not intersect any static barriers (excluding cells currently vacated by the entity).
        """
        if not entity_cells:
            return True
        dr, dc = displacement
        displaced_cells = {(r + dr, c + dc) for r, c in entity_cells}

        if grid_bounds:
            H, W = grid_bounds
            for r, c in displaced_cells:
                if not (0 <= r < H and 0 <= c < W):
                    return False

        # Collision check with external barriers
        external_barriers = barrier_set - entity_cells
        if any(cell in external_barriers for cell in displaced_cells):
            return False

        return True

    def evaluates_spelke_inertia(
        self,
        position: tuple[int, int],
        velocity: tuple[int, int],
        steps: int = 1,
        obstacles: set[tuple[int, int]] | None = None,
        bounce: bool = False,
        grid_bounds: tuple[int, int] | None = None,
    ) -> list[tuple[int, int]]:
        """Evaluate Spelke Directional Inertia & Momentum Conservation.

        Objects maintain directional trajectory and velocity across time steps
        unless deflected or halted by external boundaries or obstacles.

        Returns projected sequence of coordinates [(r_1, c_1), ..., (r_n, c_n)].
        """
        trajectory: list[tuple[int, int]] = []
        curr_r, curr_c = position
        vr, vc = velocity
        obs = obstacles or set()

        for _ in range(steps):
            next_r, next_c = curr_r + vr, curr_c + vc
            hit = False

            if grid_bounds:
                H, W = grid_bounds
                if not (0 <= next_r < H and 0 <= next_c < W):
                    hit = True

            if (next_r, next_c) in obs:
                hit = True

            if hit:
                if bounce:
                    vr, vc = -vr, -vc
                    next_r, next_c = curr_r + vr, curr_c + vc
                    if grid_bounds:
                        H, W = grid_bounds
                        next_r = max(0, min(H - 1, next_r))
                        next_c = max(0, min(W - 1, next_c))
                else:
                    break

            curr_r, curr_c = next_r, next_c
            trajectory.append((curr_r, curr_c))

        return trajectory

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Convert all symbolic world predicates and ontology into atomic HCIR graph transactions."""
        ops: list[TransactionOperation] = []

        # Ground Elizabeth Spelke physical axioms
        for axiom, pred in self.spelke_axioms.items():
            node_id = f"spelke_axiom_{axiom.value}"
            node = ConceptNode(
                id=node_id,
                label=f"spelke_{axiom.value}",
                definition=pred.describe(),
                domain="spelke_core_knowledge",
                properties={
                    "axiom": axiom.value,
                    "predicate": pred.describe(),
                    "variable": pred.variable,
                    "operator": pred.operator,
                    "value": str(pred.value),
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )

        for role, feats in [
            ("goal", self._goal_features),
            ("barrier", self._barrier_features),
            ("walkable", self._walkable_features),
            ("cargo", self._cargo_features),
            ("receptacle", self._receptacle_features),
        ]:
            for feat in feats:
                node_id = f"ontology_{role}_{feat}"
                rank, ev_str = self._feature_evidence.get((role, feat), (0, "unverified"))
                node = ConceptNode(
                    id=node_id,
                    label=f"{role}_{feat}",
                    definition=f"Learned intensional {role} entity with feature {feat}",
                    domain="spatial_ontology",
                    properties={
                        "role": role,
                        "feature_id": feat,
                        "is_intensional": True,
                        "provenance": ev_str,
                        "evidence_rank": rank,
                        "has_terminal_reward": rank >= OntologyEvidenceRank.TERMINAL_REWARD,
                    },
                )
                ops.append(
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=node_id,
                        node_data=node.model_dump(),
                    )
                )
        if self.telic_goal_type:
            node_id = "ontology_telic_goal"
            node = ConceptNode(
                id=node_id,
                label=f"telic_goal_{self.telic_goal_type}",
                definition=f"Global telic goal: {self.telic_goal_type} targeting {self.telic_target_feature}",
                domain="spatial_ontology",
                properties={
                    "telic_goal_type": self.telic_goal_type,
                    "telic_target_feature": self.telic_target_feature,
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )

        for feat in self._actuator_features:
            node_id = f"ontology_actuator_{feat}"
            rank, ev_str = self._feature_evidence.get(("actuator", feat), (0, "unverified"))
            node = ConceptNode(
                id=node_id,
                label=f"actuator_{feat}",
                definition=f"Learned actuator feature {feat}",
                domain="spatial_ontology",
                properties={
                    "role": "actuator",
                    "feature_id": feat,
                    "is_intensional": True,
                    "provenance": ev_str,
                    "evidence_rank": rank,
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )

        for i, pred in enumerate(self.domain_predicates):
            node_id = f"causal_predicate_{i}_{pred.variable}"
            node = ConceptNode(
                id=node_id,
                label=f"pred_{pred.variable}",
                definition=pred.describe(),
                domain="causal_predicate",
                properties={
                    "variable": pred.variable,
                    "operator": pred.operator,
                    "value": str(pred.value) if pred.value is not None else "",
                    "relation": pred.relation,
                    "target_entity": pred.target_entity,
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node_id,
                    node_data=node.model_dump(),
                )
            )

        return ops

    def hydrate_from_nodes(self, nodes: Sequence[Any]) -> None:
        """Reconstruct symbolic world theory from stored HCIR ConceptNodes."""
        for node in nodes:
            if not isinstance(node, ConceptNode):
                continue
            props = node.properties or {}
            domain = node.domain

            if domain == "spatial_ontology":
                role = props.get("role")
                feat = props.get("feature_id")
                if role and feat is not None:
                    try:
                        feat_val = int(feat)
                    except (ValueError, TypeError):
                        feat_val = feat

                    if role == "goal":
                        has_reward = (
                            props.get("has_terminal_reward") is True
                            or props.get("provenance")
                            in (
                                "terminal_reward",
                                "terminal_reward:success",
                                str(TerminalSignal.SUCCESS),
                            )
                            or props.get("evidence_rank", 0) >= OntologyEvidenceRank.TERMINAL_REWARD
                            or (
                                props.get("has_terminal_reward") is not False
                                and props.get("evidence_rank") is None
                                and not any(
                                    w in str(props.get("provenance", "")).lower()
                                    for w in ("unverified", "candidate", "heuristic")
                                )
                                and (
                                    "goal" in node.id.lower()
                                    or "winning" in str(getattr(node, "definition", "")).lower()
                                    or "win" in str(props.get("description", "")).lower()
                                )
                            )
                        )
                        if has_reward:
                            self.induce_goal(feat_val, evidence="hydration:terminal_reward")
                        else:
                            logger.info(
                                "HCIRSymbolicWorldTheory: Hydration skipped unverified goal feature %s (no terminal reward provenance)",
                                feat_val,
                            )
                            self.induce_candidate_goal(feat_val, evidence="hydration:unverified")
                    elif role == "barrier":
                        self.induce_barrier(feat_val, evidence="hydration")
                    elif role == "walkable":
                        self.induce_walkable(feat_val, evidence="hydration")
                    elif role == "cargo":
                        self.induce_cargo(feat_val, evidence="hydration")
                    elif role == "receptacle":
                        self.induce_receptacle(feat_val, evidence="hydration")
                    elif role == "actuator":
                        self.induce_actuator(feat_val, evidence="hydration")

                if "telic_goal_type" in props:
                    self.telic_goal_type = str(props["telic_goal_type"])
                    self.telic_target_feature = props.get("telic_target_feature")

            elif domain == "spelke_core_knowledge":
                axiom_str = props.get("axiom")
                if axiom_str:
                    try:
                        axiom_enum = SpelkePhysicalAxiom(axiom_str)
                        self.spelke_axioms[axiom_enum] = CausalPredicate(
                            variable=str(props.get("variable", "")),
                            operator=str(props.get("operator", "==")),
                            value=props.get("value", True),
                        )
                    except ValueError:
                        logger.debug("Unknown SpelkePhysicalAxiom skipped: %r", axiom_str)

            elif domain == "causal_predicate":
                pred = CausalPredicate(
                    variable=str(props.get("variable", "")),
                    operator=str(props.get("operator", "==")),
                    value=props.get("value"),
                    relation=str(props.get("relation", "")),
                    target_entity=str(props.get("target_entity", "")),
                )
                self.record_causal_predicate(pred)

    def prune_stale_hypotheses(self) -> int:
        """Prune invalid/stale hypotheses and clear evaluation caches during idle consolidation."""
        pruned = len(self._barrier_eval_cache) + len(self._walkable_eval_cache)
        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()

        # Remove conflicting candidate goals that are confirmed walkable or barriers
        conflicts = (self._candidate_goal_features & self._walkable_features) | (
            self._candidate_goal_features & self._barrier_features
        )
        for c in conflicts:
            if c not in self._goal_features:
                self._candidate_goal_features.discard(c)
                pruned += 1

        return pruned

    def clear(self) -> None:
        """Clear all intensional predicates when resetting across distinct games."""
        self.goal_predicates.clear()
        self.barrier_predicates.clear()
        self.walkable_predicates.clear()
        self._barrier_eval_cache.clear()
        self._walkable_eval_cache.clear()
        self.cargo_predicates.clear()
        self.receptacle_predicates.clear()
        self.actuator_predicates.clear()
        self._goal_features.clear()
        self._candidate_goal_features.clear()
        self._explored_features.clear()
        self._explored_loci.clear()
        self._probed_actuator_loci.clear()
        self._barrier_features.clear()
        self._walkable_features.clear()
        self._cargo_features.clear()
        self._receptacle_features.clear()
        self._actuator_features.clear()
        self._avatar_features.clear()
        self._feature_evidence.clear()
        self.spelke_axioms.clear()
        self.telic_goal_type = ""
        self.telic_target_feature = None
        # Full clear: reset all role abstraction state (including priors and templates)
        self._conditional_affordances.clear()
        self._feature_to_role.clear()
        self._role_to_features.clear()
        self._role_priors.clear()
        self._role_conditional_templates.clear()
        self._context_epoch = 0
        self._has_prior_context_transition = False
        self._awaiting_context_revalidation = False
        self._labels_non_stationary = False
        self._current_context_verified.clear()
        self._current_entity_distribution_stats.clear()
        self._prior_entity_distribution_stats.clear()


def invalidate_stale_context_ontology(
    theory: HCIRSymbolicWorldTheory,
    *,
    hazard_tracker: Any = None,
    episodic_cortex: Any = None,
    working_memory: Any = None,
    engine_or_view: Any = None,
    bg_feature: Any = 0,
    preserve_current_context: bool = False,
    reason: str = "",
) -> None:
    """Purge stale cross-context raw-feature bindings upon label-invariance falsification.

    Preserves domain-general role priors, conditional affordance role templates,
    avatar/background identity, and motor dynamics while clearing feature-ID-keyed
    caches that were not confirmed in the current context.
    """
    theory.mark_cross_context_label_shift(reason=reason)
    theory.clear_context_feature_ontology(
        bg_feature=bg_feature, preserve_current_context=preserve_current_context
    )
    theory.tool_barrier_affinities.clear()

    keep_lethal = theory.context_verified_features("lethal") if preserve_current_context else set()
    if hazard_tracker is not None:
        known_lethal = getattr(hazard_tracker, "known_lethal_features", None)
        if isinstance(known_lethal, set):
            known_lethal.intersection_update(
                keep_lethal
            ) if preserve_current_context else known_lethal.clear()
        if not preserve_current_context:
            static_pos = getattr(hazard_tracker, "static_lethal_positions", None)
            if isinstance(static_pos, set):
                static_pos.clear()

    if episodic_cortex is not None:
        disc_lethal = getattr(episodic_cortex, "discovered_lethal_features", None)
        if isinstance(disc_lethal, set):
            disc_lethal.intersection_update(
                keep_lethal
            ) if preserve_current_context else disc_lethal.clear()

    if working_memory is not None:
        hyp_engine = getattr(working_memory, "hypothesis_engine", None)
        if hyp_engine is not None and hasattr(hyp_engine, "reset_episode"):
            hyp_engine.reset_episode(retain_confirmed=False)
        remote_causal = getattr(working_memory, "remote_causal", None)
        if remote_causal is not None and hasattr(remote_causal, "reset_episode"):
            remote_causal.reset_episode(retain_long_term=False)
        for wm_attr in (
            "concept_role_priors",
            "tool_barrier_affinities",
            "mental_plan",
            "subgoal_stack",
            "topological_subgoals",
            "candidate_goals",
            "exhausted_candidate_goals",
            "unlocked_barriers",
            "activated_triggers",
            "quiescent_features",
            "effective_features",
            "feature_quiescent_counts",
        ):
            coll = getattr(working_memory, wm_attr, None)
            if hasattr(coll, "clear"):
                coll.clear()
        if not preserve_current_context:
            v_wm = getattr(working_memory, "visuospatial", None)
            if v_wm is not None:
                for v_attr in ("learned_barriers", "learned_goal_positions"):
                    v_coll = getattr(v_wm, v_attr, None)
                    if hasattr(v_coll, "clear"):
                        v_coll.clear()

    eng = (
        getattr(engine_or_view, "_wrapped_engine", engine_or_view)
        if engine_or_view is not None
        else None
    )
    if eng is not None:
        mobile = getattr(eng, "mobile_threat_features", None)
        if isinstance(mobile, set):
            mobile.clear()
        m_plan = getattr(eng, "mental_plan", None)
        if hasattr(m_plan, "clear"):
            m_plan.clear()
        if hasattr(eng, "active_arbitrated_goal"):
            eng.active_arbitrated_goal = None
        eng_state = getattr(eng, "state", None)
        if eng_state is not None:
            if hasattr(eng_state, "active_arbitrated_goal"):
                eng_state.active_arbitrated_goal = None
            if hasattr(eng_state, "mental_plan") and hasattr(eng_state.mental_plan, "clear"):
                eng_state.mental_plan.clear()
        if not preserve_current_context:
            for attr in ("learned_barriers", "learned_goal_positions", "learned_cargo_features"):
                coll = getattr(eng, attr, None)
                if isinstance(coll, set):
                    coll.clear()
        else:
            keep_goal = theory.context_verified_features("goal")
            if not keep_goal:
                l_goals = getattr(eng, "learned_goal_positions", None)
                if isinstance(l_goals, set):
                    l_goals.clear()
