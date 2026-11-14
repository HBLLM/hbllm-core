"""Library Learning & DreamCoder Wake/Sleep Skill Abstraction Engine — HCIR §6 / AGI Gap 2.

Biologically grounded in procedural memory consolidation and Josh Tenenbaum's
DreamCoder program induction paradigm (Ellis et al., 2021; Tenenbaum et al., 2011):
1. Subgoal Sequence Telemetry:
   - Tracks invocation frequencies, success rates, and cross-context diversity
     of action intents and subgoal sequences.
2. Wake Phase (Online Experience Ingestion):
   - Ingests successful episode execution traces and updates heuristic search
     proposal distributions conditioned on environmental context features.
3. Sleep Phase (Offline Library Consolidation):
   - Identifies high-utility recurrent sub-programs that appear across multiple
     distinct environments (cross-context reuse >= 2).
   - Automatically abstracts them into first-class composite macro skills and
     SpatialActionIntent primitives.
   - Commits newly inducted macros into TieredWorkspace.persistent for zero-shot transfer.
4. Learned Search Proposal Distribution:
   - Maintains an empirical Dirichlet-multinomial proposal distribution over
     candidate skills and macro intents conditioned on context features, replacing
     brute-force combinatorial enumeration.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hbllm.hcir.nodes import SkillNode
from hbllm.hcir.skills.declarative import (
    ActionAffordancePredicate,
    DeclarativeNeuroSymbolicSkill,
    SubgoalSequence,
    SymbolicSubgoal,
)
from hbllm.hcir.spatial_planner import SpatialActionIntent
from hbllm.hcir.transactions import TransactionOp, TransactionOperation

if TYPE_CHECKING:
    from hbllm.hcir.workspace_tiers import TieredWorkspace

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Subgoal Pattern Telemetry & Records
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SubgoalPatternRecord:
    """Historical telemetry for a discovered sub-sequence of action intents."""

    pattern: tuple[str, ...]
    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    contexts_seen: set[str] = field(default_factory=set)
    is_abstracted_macro: bool = False
    macro_name: str | None = None
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    @property
    def length(self) -> int:
        return len(self.pattern)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / max(1, total) if total > 0 else 0.5

    @property
    def context_diversity(self) -> int:
        return len(self.contexts_seen)

    @property
    def compression_gain(self) -> float:
        """Raw algorithmic compression gain: (length - 1) * invocation_count."""
        return max(0.0, float((self.length - 1) * self.invocation_count))

    @property
    def utility_score(self) -> float:
        """DreamCoder library utility score.

        Balances sequence length, invocation frequency, empirical success rate,
        and cross-context generalization diversity.
        """
        if self.length < 2 or self.invocation_count == 0:
            return 0.0
        diversity_multiplier = math.log2(1.0 + self.context_diversity)
        return (self.length - 1) * self.invocation_count * self.success_rate * diversity_multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern": list(self.pattern),
            "invocation_count": self.invocation_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "contexts_seen": sorted(list(self.contexts_seen)),
            "is_abstracted_macro": self.is_abstracted_macro,
            "macro_name": self.macro_name,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "utility_score": self.utility_score,
            "compression_gain": self.compression_gain,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubgoalPatternRecord:
        return cls(
            pattern=tuple(data.get("pattern", ())),
            invocation_count=int(data.get("invocation_count", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            contexts_seen=set(data.get("contexts_seen", [])),
            is_abstracted_macro=bool(data.get("is_abstracted_macro", False)),
            macro_name=data.get("macro_name"),
            created_at=float(data.get("created_at", time.time())),
            last_used_at=float(data.get("last_used_at", time.time())),
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Subgoal Pattern Tracker
# ═══════════════════════════════════════════════════════════════════════════


class SubgoalPatternTracker:
    """Tracks recurrence, success, and context diversity of intent sub-sequences."""

    def __init__(
        self,
        min_pattern_length: int = 2,
        max_pattern_length: int = 5,
    ) -> None:
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.patterns: dict[tuple[str, ...], SubgoalPatternRecord] = {}

    def _normalize_intent(self, intent: Any) -> str:
        if hasattr(intent, "value"):
            return str(intent.value)
        return str(intent)

    def record_subgoal_trace(
        self,
        intents: Sequence[Any],
        context_id: str = "default",
        success: bool = True,
    ) -> list[tuple[str, ...]]:
        """Extract all k-grams from trace and update pattern records."""
        normalized = [self._normalize_intent(i) for i in intents]
        n = len(normalized)
        now = time.time()
        updated_patterns: list[tuple[str, ...]] = []

        for length in range(self.min_pattern_length, min(self.max_pattern_length + 1, n + 1)):
            for start in range(n - length + 1):
                pat = tuple(normalized[start : start + length])
                rec = self.patterns.get(pat)
                if rec is None:
                    rec = SubgoalPatternRecord(pattern=pat, created_at=now)
                    self.patterns[pat] = rec

                rec.invocation_count += 1
                rec.last_used_at = now
                rec.contexts_seen.add(str(context_id))
                if success:
                    rec.success_count += 1
                else:
                    rec.failure_count += 1
                updated_patterns.append(pat)

        return updated_patterns

    def get_qualifying_patterns(
        self,
        min_invocations: int = 2,
        min_contexts: int = 2,
        min_success_rate: float = 0.5,
        min_utility: float = 1.0,
    ) -> list[SubgoalPatternRecord]:
        """Find non-abstracted patterns meeting DreamCoder consolidation criteria."""
        candidates: list[SubgoalPatternRecord] = []
        for rec in self.patterns.values():
            if rec.is_abstracted_macro:
                continue
            if rec.invocation_count < min_invocations:
                continue
            if rec.context_diversity < min_contexts:
                continue
            if rec.success_rate < min_success_rate:
                continue
            if rec.utility_score < min_utility:
                continue
            candidates.append(rec)

        candidates.sort(
            key=lambda r: (r.utility_score, r.context_diversity, r.length), reverse=True
        )
        return candidates

    def mark_as_abstracted(self, pattern: tuple[str, ...], macro_name: str) -> None:
        rec = self.patterns.get(pattern)
        if rec is not None:
            rec.is_abstracted_macro = True
            rec.macro_name = macro_name

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_pattern_length": self.min_pattern_length,
            "max_pattern_length": self.max_pattern_length,
            "patterns": {"__sep__".join(k): v.to_dict() for k, v in self.patterns.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SubgoalPatternTracker:
        inst = cls(
            min_pattern_length=int(data.get("min_pattern_length", 2)),
            max_pattern_length=int(data.get("max_pattern_length", 5)),
        )
        patterns_data = data.get("patterns", {})
        for k_str, v_dict in patterns_data.items():
            k_tuple = tuple(k_str.split("__sep__"))
            inst.patterns[k_tuple] = SubgoalPatternRecord.from_dict(v_dict)
        return inst


# ═══════════════════════════════════════════════════════════════════════════
# 3. Learned Heuristic Search Proposal Distribution
# ═══════════════════════════════════════════════════════════════════════════


class LearnedProposalDistribution:
    """Empirical Dirichlet-multinomial proposal distribution over candidate schemas.

    Guides heuristic search during MCTS / counterfactual simulation by prioritizing
    actions and subgoals that have historically succeeded in similar contexts.
    """

    def __init__(self, prior_pseudocount: float = 0.5) -> None:
        self.prior_pseudocount = prior_pseudocount
        # feature_key -> intent_name -> weight
        self.feature_intent_weights: dict[str, dict[str, float]] = {}
        # global intent prior counts
        self.global_intent_counts: dict[str, float] = {}

    def extract_context_keys(self, context_features: dict[str, Any]) -> set[str]:
        """Convert arbitrary environment context into canonical symbolic feature tokens."""
        keys: set[str] = set()
        for k, v in context_features.items():
            if isinstance(v, bool) and v:
                keys.add(f"flag:{k}")
            elif isinstance(v, (int, float)) and v > 0:
                keys.add(f"present:{k}")
                if v >= 2:
                    keys.add(f"multi:{k}")
            elif isinstance(v, (set, list, tuple)) and len(v) > 0:
                keys.add(f"has_{k}")
                if len(v) >= 2:
                    keys.add(f"multi_{k}")
            elif isinstance(v, str) and v:
                keys.add(f"{k}:{v}")
        if not keys:
            keys.add("default_context")
        return keys

    def update(
        self,
        context_features: dict[str, Any],
        intents: Sequence[Any],
        success: bool = True,
    ) -> None:
        """Update proposal distribution weights based on episode outcome."""
        keys = self.extract_context_keys(context_features)
        delta = 1.0 if success else -0.2

        for raw_intent in intents:
            intent_name = raw_intent.value if hasattr(raw_intent, "value") else str(raw_intent)
            # Update global counts
            cur_global = self.global_intent_counts.get(intent_name, 0.0)
            self.global_intent_counts[intent_name] = max(
                0.0, cur_global + (1.0 if success else 0.0)
            )

            # Update feature conditioned weights
            for k in keys:
                if k not in self.feature_intent_weights:
                    self.feature_intent_weights[k] = {}
                cur_w = self.feature_intent_weights[k].get(intent_name, 0.0)
                self.feature_intent_weights[k][intent_name] = max(0.0, cur_w + delta)

    def compute_distribution(
        self,
        context_features: dict[str, Any],
        candidate_pool: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute normalized proposal probability for candidate intents/schemas."""
        keys = self.extract_context_keys(context_features)
        pool = set(candidate_pool) if candidate_pool else set(self.global_intent_counts.keys())
        if not pool:
            # Baseline default primitives
            pool = {"NAVIGATE", "INTERACT", "ACTIVATE", "PICKUP", "DROP"}

        raw_scores: dict[str, float] = {}
        for candidate in pool:
            # Base prior with smoothing
            base = self.global_intent_counts.get(candidate, 0.0) + self.prior_pseudocount
            score = base
            for k in keys:
                feat_map = self.feature_intent_weights.get(k)
                if feat_map and candidate in feat_map:
                    score += feat_map[candidate] * 2.0
            raw_scores[candidate] = score

        total = sum(raw_scores.values())
        if total <= 0:
            return {c: 1.0 / len(pool) for c in pool}
        return {c: s / total for c, s in raw_scores.items()}

    def sample_candidates(
        self,
        context_features: dict[str, Any],
        top_k: int = 5,
        candidate_pool: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return top_k candidate intents/schemas sorted by proposal probability."""
        dist = self.compute_distribution(context_features, candidate_pool=candidate_pool)
        sorted_candidates = sorted(dist.items(), key=lambda item: item[1], reverse=True)
        return sorted_candidates[:top_k]

    def propose_open_ended_intents(
        self,
        context_key: tuple[int, ...] | dict[str, Any],
        macro_skills: dict[str, Any] | None = None,
        hypotheses: dict[str, Any] | Sequence[Any] | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rank primitives, synthesized macro-skills, and epistemic hypothesis probes via MDL+Dirichlet posterior."""
        if isinstance(context_key, dict):
            ctx_features = dict(context_key)
        else:
            ctx_features = {f"ctx_{idx}": int(val) for idx, val in enumerate(context_key)}

        pool: list[str] = ["NAVIGATE_TO", "INTERACT", "PICKUP", "PUSH_OBJECT", "EXPLORE"]
        for k in self.global_intent_counts:
            if k not in pool:
                pool.append(k)

        dl_lengths: dict[str, float] = {name: 1.0 for name in pool}
        probe_bonuses: dict[str, float] = {}

        if macro_skills:
            for skill_name, skill_obj in macro_skills.items():
                label = (
                    skill_name if str(skill_name).startswith("macro:") else f"macro:{skill_name}"
                )
                if label not in pool:
                    pool.append(label)
                prog = getattr(skill_obj, "program", None)
                subgoals = getattr(prog, "subgoals", ()) if prog is not None else ()
                dl_lengths[label] = max(1.0, float(len(subgoals)) * 0.5)

        if hypotheses is not None:
            hyp_iter = hypotheses.values() if isinstance(hypotheses, dict) else hypotheses
            for hyp in hyp_iter:
                rid = getattr(hyp, "rule_id", None)
                status = str(getattr(hyp, "status", ""))
                conf = float(getattr(hyp, "confidence", 0.5))
                if rid and ("TENTATIVE" in status or conf < 0.85):
                    label = f"probe_hypothesis:{rid}"
                    if label not in pool:
                        pool.append(label)
                    ast_obj = (getattr(hyp, "metadata", None) or {}).get("ast")
                    dl_lengths[label] = float(getattr(ast_obj, "ast_depth", 2))
                    probe_bonuses[label] = max(0.05, 1.0 - conf) * 0.35

        base_dist = self.compute_distribution(ctx_features, candidate_pool=pool)
        weighted: dict[str, float] = {}
        for cand, p in base_dist.items():
            dl = dl_lengths.get(cand, 1.0)
            mdl_prior = math.exp(-0.08 * max(0.0, dl - 1.0))
            weighted[cand] = max(1e-9, p * mdl_prior + probe_bonuses.get(cand, 0.0))

        total = sum(weighted.values())
        if total > 0:
            normalized = [(c, round(v / total, 6)) for c, v in weighted.items()]
        else:
            normalized = [(c, 1.0 / max(1, len(weighted))) for c in weighted]
        normalized.sort(key=lambda item: item[1], reverse=True)
        return normalized[:top_k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_pseudocount": self.prior_pseudocount,
            "feature_intent_weights": self.feature_intent_weights,
            "global_intent_counts": self.global_intent_counts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearnedProposalDistribution:
        inst = cls(prior_pseudocount=float(data.get("prior_pseudocount", 0.5)))
        inst.feature_intent_weights = data.get("feature_intent_weights", {})
        inst.global_intent_counts = data.get("global_intent_counts", {})
        return inst


# ═══════════════════════════════════════════════════════════════════════════
# 4. Master Library Learner (DreamCoder Wake/Sleep)
# ═══════════════════════════════════════════════════════════════════════════


class LibraryLearner:
    """Master DreamCoder Wake/Sleep Library Learning & Skill Abstraction Engine.

    Coordinates wake-phase trace recording, offline sleep-phase subroutine abstraction,
    proposal distribution updates, and persistent workspace consolidation.
    """

    def __init__(
        self,
        pattern_tracker: SubgoalPatternTracker | None = None,
        proposal_distribution: LearnedProposalDistribution | None = None,
    ) -> None:
        self.pattern_tracker = pattern_tracker or SubgoalPatternTracker()
        self.proposal_distribution = proposal_distribution or LearnedProposalDistribution()
        self.macro_skills: dict[str, DeclarativeNeuroSymbolicSkill] = {}
        self.abstracted_macros: dict[str, SubgoalPatternRecord] = {}

    def wake_record_episode(
        self,
        intents_or_steps: Sequence[Any],
        context_id: str = "default",
        context_features: dict[str, Any] | None = None,
        success: bool = True,
    ) -> list[tuple[str, ...]]:
        """Wake Phase: Ingest episode experience trace and update proposal distribution."""
        intents: list[str] = []
        for item in intents_or_steps:
            if hasattr(item, "intent"):
                intent_val = getattr(item.intent, "value", str(item.intent))
            elif hasattr(item, "value"):
                intent_val = str(item.value)
            else:
                intent_val = str(item)
            intents.append(intent_val)

        if not intents:
            return []

        # 1. Update pattern tracker
        updated = self.pattern_tracker.record_subgoal_trace(
            intents=intents,
            context_id=context_id,
            success=success,
        )

        # 2. Update learned proposal distribution
        ctx_feats = dict(context_features or {})
        self.proposal_distribution.update(
            context_features=ctx_feats,
            intents=intents,
            success=success,
        )

        logger.debug(
            "LibraryLearner Wake: Recorded trace of %d steps across context '%s' (success=%s)",
            len(intents),
            context_id,
            success,
        )
        return updated

    def sleep_consolidation(
        self,
        tiered_workspace: TieredWorkspace | None = None,
        tenant_id: str = "default",
        min_invocations: int = 2,
        min_contexts: int = 2,
        min_success_rate: float = 0.5,
    ) -> list[str]:
        """Sleep Phase: Offline abstraction of high-utility recurring subroutines into macros."""
        qualifying = self.pattern_tracker.get_qualifying_patterns(
            min_invocations=min_invocations,
            min_contexts=min_contexts,
            min_success_rate=min_success_rate,
        )

        newly_abstracted: list[str] = []

        for rec in qualifying:
            macro_name = f"macro_{'_'.join(s.lower() for s in rec.pattern)}"
            if macro_name in self.macro_skills:
                continue

            # Build declarative SubgoalSequence
            subgoals: list[SymbolicSubgoal] = []
            for intent_str in rec.pattern:
                try:
                    intent = SpatialActionIntent(intent_str)
                except Exception:
                    intent = SpatialActionIntent.NAVIGATE_TO
                subgoals.append(
                    SymbolicSubgoal(
                        intent=intent,
                        target_query={"role": "abstracted_target"},
                    )
                )

            subgoal_seq = SubgoalSequence(*subgoals)

            # Build DeclarativeNeuroSymbolicSkill subclass
            class AbstractedMacroSkill(DeclarativeNeuroSymbolicSkill):
                skill_name = macro_name
                description = (
                    f"Abstracted DreamCoder macro routine: {' -> '.join(rec.pattern)} "
                    f"(utility: {rec.utility_score:.2f}, contexts: {rec.context_diversity})"
                )
                signature = ActionAffordancePredicate(required=set())
                program = subgoal_seq

            skill_inst = AbstractedMacroSkill()
            self.macro_skills[macro_name] = skill_inst
            self.abstracted_macros[macro_name] = rec
            self.pattern_tracker.mark_as_abstracted(rec.pattern, macro_name)

            # Register macro in proposal distribution
            self.proposal_distribution.global_intent_counts[macro_name] = rec.invocation_count

            newly_abstracted.append(macro_name)
            logger.info(
                "LibraryLearner Sleep: Abstracted macro '%s' from pattern %s (contexts=%d, utility=%.2f)",
                macro_name,
                rec.pattern,
                rec.context_diversity,
                rec.utility_score,
            )

        # Commit to persistent tier if available
        if newly_abstracted and tiered_workspace is not None:
            self.persist_to_workspace(tiered_workspace, tenant_id=tenant_id)

        return newly_abstracted

    def persist_to_workspace(
        self,
        workspace: TieredWorkspace,
        tenant_id: str = "default",
    ) -> list[str]:
        """Commit all abstracted macro skills into workspace.persistent as SkillNodes."""
        persisted_ids: list[str] = []
        ops: list[TransactionOperation] = []

        for name, skill in self.macro_skills.items():
            rec = self.abstracted_macros.get(name)
            node_id = f"macro_skill_{name}"
            skill_node = SkillNode(
                id=node_id,
                name=name,
                skill_name=name,
                description=getattr(skill, "description", f"Abstracted macro {name}"),
                preconditions=[str(skill.signature)],
                postconditions=[],
                success_rate=float(rec.success_rate) if rec else 0.8,
                invocation_count=int(rec.invocation_count) if rec else 1,
                properties={
                    "is_macro": True,
                    "macro_pattern": list(rec.pattern) if rec else [],
                    "utility_score": float(rec.utility_score) if rec else 1.0,
                    "contexts_seen": sorted(list(rec.contexts_seen)) if rec else [],
                    "tenant_id": tenant_id,
                    "timestamp": time.time(),
                },
            )
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=skill_node.id,
                    node_data=skill_node.model_dump(),
                    node=skill_node,
                )
            )
            persisted_ids.append(node_id)

        if ops:
            grant = getattr(workspace, "_system_grant", None)
            workspace.persistent.commit_operations(
                ops,
                author="LibraryLearnerSleepConsolidation",
                grant=grant,
            )
            logger.info(
                "LibraryLearner: Persisted %d macro skills to workspace.persistent", len(ops)
            )

        return persisted_ids

    def load_from_workspace(
        self,
        workspace: TieredWorkspace,
        tenant_id: str = "default",
    ) -> int:
        """Hydrate macro skills from workspace.persistent."""
        loaded_count = 0
        for node in workspace.persistent.graph.nodes.values():
            props = getattr(node, "properties", {})
            if props.get("is_macro"):
                name = getattr(node, "name", "") or getattr(node, "skill_name", "")
                pattern = tuple(props.get("macro_pattern", []))
                if name and pattern and name not in self.macro_skills:
                    subgoals = [
                        SymbolicSubgoal(
                            intent=SpatialActionIntent(s)
                            if s in SpatialActionIntent.__members__
                            else SpatialActionIntent.NAVIGATE_TO,
                            target_query={"role": "abstracted_target"},
                        )
                        for s in pattern
                    ]
                    subgoal_seq = SubgoalSequence(*subgoals)

                    class HydratedMacroSkill(DeclarativeNeuroSymbolicSkill):
                        skill_name = name
                        signature = ActionAffordancePredicate(required=set())
                        program = subgoal_seq

                    self.macro_skills[name] = HydratedMacroSkill()
                    rec = SubgoalPatternRecord(
                        pattern=pattern,
                        invocation_count=int(getattr(node, "invocation_count", 1)),
                        contexts_seen=set(props.get("contexts_seen", [])),
                        is_abstracted_macro=True,
                        macro_name=name,
                    )
                    self.abstracted_macros[name] = rec
                    self.pattern_tracker.patterns[pattern] = rec
                    self.proposal_distribution.global_intent_counts[name] = rec.invocation_count
                    loaded_count += 1

        if loaded_count > 0:
            logger.info(
                "LibraryLearner: Hydrated %d macro skills from persistent tier", loaded_count
            )
        return loaded_count

    def sample_proposals(
        self,
        context_features: dict[str, Any],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Query top candidate actions or macro skills ranked by proposal distribution."""
        candidate_pool = list(self.proposal_distribution.global_intent_counts.keys())
        for m in self.macro_skills:
            if m not in candidate_pool:
                candidate_pool.append(m)
        return self.proposal_distribution.sample_candidates(
            context_features=context_features,
            top_k=top_k,
            candidate_pool=candidate_pool,
        )

    def sample_open_ended_proposals(
        self,
        context_key: tuple[int, ...] | dict[str, Any],
        hypotheses: dict[str, Any] | Sequence[Any] | None = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rank primitives, consolidated macro skills, and active hypothesis-probing intents."""
        return self.proposal_distribution.propose_open_ended_intents(
            context_key=context_key,
            macro_skills=self.macro_skills,
            hypotheses=hypotheses,
            top_k=top_k,
        )
