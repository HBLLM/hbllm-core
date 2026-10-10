from __future__ import annotations

"""Prefrontal Cortex Inductive Logic & Relational Hypothesis Testing Faculty.

Biologically modeled on mammalian prefrontal cortex inductive hypothesis generation,
Popperian counterexample refutation, and causal rule abstraction:
1. Relational Rule Generation:
   - Observes state transitions and abstracts candidate symbolic rules:
     - Contact transformations (e.g. avatar stepping on switch clears barrier).
     - Action mutations (e.g. effector action toggles feature X -> Y).
     - Remote mechanism bindings (key of feature K unlocks door D).
2. Popperian Counterexample Refutation:
   - Tracks positive empirical support vs falsifications.
   - Immediately refutes hypotheses upon counterexample observation.
3. Invariant Retention:
   - Retains verified causal invariants across retries and episodes.
4. Frontopolar Subgoal Synergy:
   - Informs subgoal planning of confirmed keys, switches, and triggers.
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class HypothesisType(StrEnum):
    """Categorization of relational causal hypotheses."""

    CONTACT_TRANSFORM = "CONTACT_TRANSFORM"
    ACTION_MUTATION = "ACTION_MUTATION"
    REMOTE_MECHANISM = "REMOTE_MECHANISM"
    HAZARD_CONTACT = "HAZARD_CONTACT"


class HypothesisStatus(StrEnum):
    """Lifecycle state of an inductive hypothesis."""

    TENTATIVE = "TENTATIVE"
    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"


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

    def record_support(self, weight: float = 1.0) -> None:
        """Record positive empirical evidence supporting this rule."""
        self.positive_support += int(weight)
        self.confidence = min(0.99, self.confidence + 0.15 * weight)
        if self.positive_support >= 2 and self.counterexamples == 0:
            self.status = HypothesisStatus.CONFIRMED

    def record_counterexample(self, weight: float = 1.0) -> None:
        """Record a falsifying counterexample (Popperian refutation)."""
        self.counterexamples += int(weight)
        self.confidence = max(0.01, self.confidence - 0.4 * weight)
        if self.counterexamples >= 1:
            self.status = HypothesisStatus.REFUTED

    @property
    def is_valid(self) -> bool:
        """True if the rule is confirmed or tentative without refutations."""
        return self.status != HypothesisStatus.REFUTED and self.counterexamples == 0


class InductiveHypothesisEngine:
    """Prefrontal Cortex Inductive Logic and Relational Rule Discovery Engine."""

    def __init__(self, min_support_to_confirm: int = 2) -> None:
        self.min_support_to_confirm = min_support_to_confirm
        self.hypotheses: dict[str, RelationalRuleHypothesis] = {}

    def observe_transition(
        self,
        prev_grid: np.ndarray,
        action: Any,
        curr_grid: np.ndarray,
        prev_avatar_pos: tuple[int, int] | None = None,
        curr_avatar_pos: tuple[int, int] | None = None,
        is_dead: bool = False,
        is_won: bool = False,
        background_feature: int = 0,
    ) -> list[RelationalRuleHypothesis]:
        """Observe state transition, formulate new hypotheses, and test active ones.

        Returns:
            List of hypotheses updated or created during this transition.
        """
        if prev_grid.shape != curr_grid.shape:
            return []

        H, W = curr_grid.shape
        diff_mask = prev_grid != curr_grid
        changed_count = int(np.count_nonzero(diff_mask))

        updated_rules: list[RelationalRuleHypothesis] = []

        # 1. Hazard contact hypothesis
        if is_dead and prev_avatar_pos is not None:
            # Check what feature was at or adjacent to avatar's position
            pr, pc = prev_avatar_pos
            for dr, dc in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    feat = int(prev_grid[nr, nc])
                    if feat != background_feature:
                        hid = f"hazard_contact_{feat}"
                        if hid in self.hypotheses:
                            hyp = self.hypotheses[hid]
                            hyp.record_support()
                            updated_rules.append(hyp)
                        else:
                            hyp = RelationalRuleHypothesis(
                                rule_id=hid,
                                rule_type=HypothesisType.HAZARD_CONTACT,
                                premise_feature=feat,
                                positive_support=1,
                                confidence=0.7,
                                description=f"Feature {feat} contact causes death",
                            )
                            self.hypotheses[hid] = hyp
                            updated_rules.append(hyp)

        # 2. Remote Mechanism & Contact Transformation
        if curr_avatar_pos is not None and prev_avatar_pos is not None:
            cr, cc = curr_avatar_pos
            contact_feat = int(prev_grid[cr, cc])
            if contact_feat != background_feature:
                # Find features that disappeared distally
                old_diff_feats = set(prev_grid[diff_mask].tolist())
                new_diff_feats = set(curr_grid[diff_mask].tolist())
                cleared_feats = old_diff_feats - new_diff_feats

                for cleared in cleared_feats:
                    if cleared != contact_feat and cleared != background_feature:
                        hid = f"remote_mech_{contact_feat}_clears_{cleared}"
                        if hid in self.hypotheses:
                            hyp = self.hypotheses[hid]
                            hyp.record_support()
                            updated_rules.append(hyp)
                        else:
                            hyp = RelationalRuleHypothesis(
                                rule_id=hid,
                                rule_type=HypothesisType.REMOTE_MECHANISM,
                                premise_feature=contact_feat,
                                target_feature=cleared,
                                positive_support=1,
                                confidence=0.6,
                                description=f"Contact with feature {contact_feat} clears barrier {cleared}",
                            )
                            self.hypotheses[hid] = hyp
                            updated_rules.append(hyp)

        # 3. Action Mutation: In-place or effector click mutations
        if changed_count in (1, 2) and action is not None:
            changed_coords = np.argwhere(diff_mask)
            for r, c in changed_coords:
                old_f = int(prev_grid[r, c])
                new_f = int(curr_grid[r, c])
                if old_f != new_f:
                    hid = f"action_{action}_mutates_{old_f}_to_{new_f}"
                    if hid in self.hypotheses:
                        hyp = self.hypotheses[hid]
                        hyp.record_support()
                        updated_rules.append(hyp)
                    else:
                        hyp = RelationalRuleHypothesis(
                            rule_id=hid,
                            rule_type=HypothesisType.ACTION_MUTATION,
                            premise_feature=old_f,
                            result_feature=new_f,
                            action_id=action,
                            positive_support=1,
                            confidence=0.55,
                            description=f"Action {action} transforms feature {old_f} into {new_f}",
                        )
                        self.hypotheses[hid] = hyp
                        updated_rules.append(hyp)

        # 4. Popperian Counterexample Check on Active Hypotheses
        # If a remote mechanism premise was visited but no barrier cleared, record counterexample
        if curr_avatar_pos is not None and not is_dead:
            cr, cc = curr_avatar_pos
            c_feat = int(prev_grid[cr, cc])
            for hyp in list(self.hypotheses.values()):
                if hyp.rule_type == HypothesisType.REMOTE_MECHANISM:
                    if hyp.premise_feature == c_feat:
                        # Check if target feature actually cleared in this step
                        target_f = hyp.target_feature
                        if target_f is not None and target_f in curr_grid:
                            # Target barrier is still present on the board
                            prev_target_count = int(np.count_nonzero(prev_grid == target_f))
                            curr_target_count = int(np.count_nonzero(curr_grid == target_f))
                            if curr_target_count >= prev_target_count:
                                hyp.record_counterexample()
                                updated_rules.append(hyp)

        return updated_rules

    def get_confirmed_rules(self) -> list[RelationalRuleHypothesis]:
        """Return all empirically confirmed relational rules."""
        return [
            h
            for h in self.hypotheses.values()
            if h.status == HypothesisStatus.CONFIRMED and h.is_valid
        ]

    def get_rule_for_barrier(self, barrier_feat: int) -> RelationalRuleHypothesis | None:
        """Find confirmed or highest-confidence trigger rule for opening barrier."""
        candidates = [
            h for h in self.hypotheses.values() if h.target_feature == barrier_feat and h.is_valid
        ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda h: (h.status == HypothesisStatus.CONFIRMED, h.confidence), reverse=True
        )
        return candidates[0]

    def reset_episode(self, retain_confirmed: bool = True) -> None:
        """Reset episodic hypotheses between levels or attempts."""
        if retain_confirmed:
            self.hypotheses = {
                k: h
                for k, h in self.hypotheses.items()
                if h.status == HypothesisStatus.CONFIRMED and h.is_valid
            }
        else:
            self.hypotheses.clear()
