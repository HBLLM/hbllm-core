"""Unified Transition-Outcome Knowledge Base backed by HCIR Graph Beliefs.

Replaces brittle hardcoded feature-role buckets with a single, context-indexed
empirical transition model storing Dirichlet/Beta outcome sufficient statistics
and learned state-transition causal rules as canonical HCIR ``BeliefNode`` and
``LearnedRuleNode`` entities.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from hbllm.hcir.graph import BeliefNode, LearnedRuleNode
from hbllm.hcir.transactions import TransactionOp, TransactionOperation


class TransitionOutcome(StrEnum):
    """Empirical outcomes observed after executing an action in a local context."""

    TRAVERSED = "traversed"
    BLOCKED = "blocked"
    TERMINAL_WIN = "terminal_win"
    TERMINAL_FAIL = "terminal_fail"
    DISPLACED_OTHER = "displaced_other"
    RECEIVED_ENTITY = "received_entity"
    MUTATED_STATE = "mutated_state"


_OUT_TRAVERSED: str = TransitionOutcome.TRAVERSED.value
_OUT_BLOCKED: str = TransitionOutcome.BLOCKED.value
_OUT_DISPLACED_OTHER: str = TransitionOutcome.DISPLACED_OTHER.value
_OUT_RECEIVED_ENTITY: str = TransitionOutcome.RECEIVED_ENTITY.value
_OUT_MUTATED_STATE: str = TransitionOutcome.MUTATED_STATE.value
_OUT_TERMINAL_WIN: str = TransitionOutcome.TERMINAL_WIN.value
_OUT_TERMINAL_FAIL: str = TransitionOutcome.TERMINAL_FAIL.value


@dataclass(frozen=True)
class TransitionEdgeEvaluation:
    """Universal search edge evaluation returned by the transition knowledge base."""

    traversable: bool
    step_cost: float
    risk_cost: float
    goal_value: float
    info_gain: float
    mutates_state: bool
    confidence: float


def _encode_context_key(key: Any) -> tuple[Any, str, str]:
    """Encode an arbitrary context key into (json_value, key_kind, safe_id_token)."""
    if isinstance(key, tuple):
        vals = [int(x) if isinstance(x, int) and not isinstance(x, bool) else x for x in key]
        token = "tuple_" + "_".join(str(v) for v in vals)
        return vals, "tuple", token
    if isinstance(key, list):
        token = "tuple_" + "_".join(str(v) for v in key)
        return list(key), "tuple", token
    if isinstance(key, int) and not isinstance(key, bool):
        return int(key), "int", f"int_{key}"
    return str(key), "str", f"str_{key}"


def _belief_node_id(scope: str, context_key: Any) -> tuple[Any, str, str]:
    """Return (serialized_key, key_kind, node_id) for a transition belief."""
    serialized_key, key_kind, token = _encode_context_key(context_key)
    return serialized_key, key_kind, f"kb_transition_{scope}_{token}"


def _rule_node_id(trigger_key: Any, effect_signature: str) -> tuple[Any, str, str]:
    """Return (serialized_trigger, key_kind, node_id) for a causal rule."""
    serialized_trig, key_kind, token = _encode_context_key(trigger_key)
    safe_eff = effect_signature.replace(":", "_").replace(" ", "_")
    return serialized_trig, key_kind, f"kb_rule_{token}_{safe_eff}"


def _decode_context_key(raw: Any, key_kind: str) -> Any:
    """Reconstruct a typed context key from serialized HCIR node properties."""
    if key_kind == "tuple" and isinstance(raw, (list, tuple)):
        return tuple(int(x) if isinstance(x, (int, float)) else x for x in raw)
    if key_kind == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    return raw


@dataclass
class TransitionBelief:
    """Empirical Dirichlet/Beta transition-outcome statistics for a context or locus."""

    scope: str
    context_key: Any
    counts: dict[str, float] = field(default_factory=dict)
    observed_effects: dict[str, int] = field(default_factory=dict)
    evidence_log: list[str] = field(default_factory=list)
    verified_epoch: int | None = None
    outcome_epochs: dict[str, int] = field(default_factory=dict)
    _total_obs: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._total_obs = float(sum(self.counts.values()))

    def record(
        self,
        outcome: TransitionOutcome,
        *,
        weight: float = 1.0,
        effect_signature: str = "",
        evidence: str = "",
        epoch: int | None = None,
    ) -> None:
        """Record an observed transition outcome and optional effect signature."""
        inc = max(0.0, float(weight))
        if inc > 0.0:
            key = outcome.value
            self.counts[key] = self.counts.get(key, 0.0) + inc
            self._total_obs += inc
            if epoch is not None:
                self.outcome_epochs[key] = epoch
        if effect_signature:
            self.observed_effects[effect_signature] = (
                self.observed_effects.get(effect_signature, 0) + 1
            )
        if evidence:
            if evidence in self.evidence_log:
                self.evidence_log.remove(evidence)
            self.evidence_log.append(evidence)
            if len(self.evidence_log) > 8:
                del self.evidence_log[: len(self.evidence_log) - 8]
        if epoch is not None:
            self.verified_epoch = epoch

    def decay_or_clear_outcome(self, outcome: TransitionOutcome) -> None:
        """Remove accumulated counts for an invalidated outcome."""
        key = outcome.value
        removed = self.counts.pop(key, 0.0)
        if removed:
            self._total_obs = max(0.0, self._total_obs - removed)
        self.outcome_epochs.pop(key, None)

    def total_observations(self) -> float:
        """Return total weighted outcome observations."""
        return self._total_obs

    def traversability_posterior(self, prior: float = 0.5) -> float:
        """Return Beta posterior mean P(traversable | observations)."""
        pos = self.counts.get(_OUT_TRAVERSED, 0.0) + self.counts.get(_OUT_TERMINAL_WIN, 0.0)
        neg = self.counts.get(_OUT_BLOCKED, 0.0) + self.counts.get(_OUT_TERMINAL_FAIL, 0.0)
        denom = pos + neg + 2.0 * prior
        if denom <= 0.0:
            return 0.5
        return float((pos + prior) / denom)

    def _outcome_prob_key(self, key: str, prior: float = 0.0) -> float:
        total = self._total_obs
        if total <= 0.0:
            return float(prior / (1.0 + prior)) if prior > 0.0 else 0.0
        return float((self.counts.get(key, 0.0) + prior) / (total + prior))

    def outcome_prob(self, outcome: TransitionOutcome, prior: float = 0.0) -> float:
        """Return empirical probability of ``outcome`` across all recorded observations."""
        key = outcome.value if isinstance(outcome, TransitionOutcome) else str(outcome)
        return self._outcome_prob_key(key, prior=prior)

    def epistemic_uncertainty(self) -> float:
        """Return epistemic uncertainty shrinking smoothly with experience."""
        return float(1.0 / (1.0 + self._total_obs))

    def confidence(self) -> float:
        """Return epistemic confidence in [0, 1)."""
        return float(1.0 - self.epistemic_uncertainty())

    def to_belief_node(self) -> BeliefNode:
        """Serialize this transition belief as a canonical HCIR ``BeliefNode``."""
        serialized_key, key_kind, node_id = _belief_node_id(self.scope, self.context_key)
        claim = (
            f"Transition outcome distribution for {self.scope}={serialized_key}: "
            f"P(trav)={self.traversability_posterior():.2f}"
        )
        return BeliefNode(
            id=node_id,
            claim=claim,
            statement=claim,
            epistemic_confidence=self.confidence(),
            belief_type="causal",
            evidence_sources=list(self.evidence_log),
            properties={
                "domain": "transition_kb",
                "scope": self.scope,
                "context_key": serialized_key,
                "key_kind": key_kind,
                "counts": dict(self.counts),
                "observed_effects": dict(self.observed_effects),
                "verified_epoch": self.verified_epoch,
                "outcome_epochs": dict(self.outcome_epochs),
            },
        )

    @classmethod
    def from_belief_node(cls, node: BeliefNode) -> TransitionBelief | None:
        """Reconstruct a ``TransitionBelief`` from a stored HCIR ``BeliefNode``."""
        props = node.properties or {}
        if props.get("domain") != "transition_kb":
            return None
        scope = str(props.get("scope", "feature"))
        key_kind = str(props.get("key_kind", "str"))
        context_key = _decode_context_key(props.get("context_key"), key_kind)
        raw_counts = props.get("counts") or {}
        counts = {str(k): float(v) for k, v in raw_counts.items()}
        raw_effects = props.get("observed_effects") or {}
        effects = {str(k): int(v) for k, v in raw_effects.items()}
        raw_outcome_epochs = props.get("outcome_epochs") or {}
        outcome_epochs = {str(k): int(v) for k, v in raw_outcome_epochs.items()}
        verified_epoch = props.get("verified_epoch")
        return cls(
            scope=scope,
            context_key=context_key,
            counts=counts,
            observed_effects=effects,
            evidence_log=list(node.evidence_sources or ()),
            verified_epoch=int(verified_epoch) if verified_epoch is not None else None,
            outcome_epochs=outcome_epochs,
        )


@dataclass
class CausalEffectRule:
    """Learned state-transition operator mapping an intervention to an effect."""

    trigger_key: Any
    effect_signature: str
    precondition_tokens: tuple[str, ...] = ()
    occurrences: int = 1
    confidence: float = 0.8
    evidence: str = ""

    def to_learned_rule_node(self) -> LearnedRuleNode:
        """Serialize this causal operator as a canonical HCIR ``LearnedRuleNode``."""
        serialized_trig, key_kind, node_id = _rule_node_id(self.trigger_key, self.effect_signature)
        return LearnedRuleNode(
            id=node_id,
            condition=f"trigger={serialized_trig}",
            prediction=self.effect_signature,
            confidence=max(0.0, min(1.0, float(self.confidence))),
            source_error_count=max(0, int(self.occurrences)),
            domain="transition_kb",
            properties={
                "trigger_key": serialized_trig,
                "key_kind": key_kind,
                "effect_signature": self.effect_signature,
                "precondition_tokens": list(self.precondition_tokens),
                "occurrences": int(self.occurrences),
                "evidence": self.evidence,
            },
        )

    @classmethod
    def from_learned_rule_node(cls, node: LearnedRuleNode) -> CausalEffectRule | None:
        """Reconstruct a ``CausalEffectRule`` from a stored HCIR ``LearnedRuleNode``."""
        if node.domain != "transition_kb":
            return None
        props = node.properties or {}
        key_kind = str(props.get("key_kind", "str"))
        trigger_key = _decode_context_key(props.get("trigger_key"), key_kind)
        eff = str(props.get("effect_signature") or node.prediction)
        preconds = tuple(str(t) for t in (props.get("precondition_tokens") or ()))
        return cls(
            trigger_key=trigger_key,
            effect_signature=eff,
            precondition_tokens=preconds,
            occurrences=int(props.get("occurrences", node.source_error_count or 1)),
            confidence=float(node.confidence),
            evidence=str(props.get("evidence", "")),
        )


@dataclass
class TransitionKnowledgeBase:
    """Unified empirical transition & causal-rule knowledge base backed by HCIR nodes."""

    _feature_beliefs: dict[Any, TransitionBelief] = field(default_factory=dict)
    _locus_beliefs: dict[tuple[int, int], TransitionBelief] = field(default_factory=dict)
    _causal_rules: dict[tuple[str, str], CausalEffectRule] = field(default_factory=dict)
    _context_epoch: int = 0
    _version: int = field(default=0, init=False, repr=False)
    _feature_eval_cache: dict[tuple[Any, bool, float, float], TransitionEdgeEvaluation] = field(
        default_factory=dict, init=False, repr=False
    )
    _outcome_features_cache: dict[tuple[str, float], frozenset[Any]] = field(
        default_factory=dict, init=False, repr=False
    )
    _impassable_cache: frozenset[Any] | None = field(default=None, init=False, repr=False)
    _traversable_cache: frozenset[Any] | None = field(default=None, init=False, repr=False)
    _cached_tx_version: int = field(default=-1, init=False, repr=False)
    _cached_tx_ops: list[TransactionOperation] | None = field(default=None, init=False, repr=False)

    def _invalidate_caches(self) -> None:
        self._version += 1
        self._feature_eval_cache.clear()
        self._outcome_features_cache.clear()
        self._impassable_cache = None
        self._traversable_cache = None
        self._cached_tx_ops = None

    def record_outcome(
        self,
        context_key: Any,
        outcome: TransitionOutcome,
        *,
        scope: str = "feature",
        weight: float = 1.0,
        effect_signature: str = "",
        evidence: str = "",
        mark_verified: bool = True,
    ) -> TransitionBelief:
        """Record an empirical transition outcome under a feature or locus key."""
        epoch = self._context_epoch if mark_verified else None
        if scope == "locus":
            locus = (int(context_key[0]), int(context_key[1]))
            belief = self._locus_beliefs.get(locus)
            if belief is None:
                belief = TransitionBelief(scope="locus", context_key=locus)
                self._locus_beliefs[locus] = belief
        else:
            belief = self._feature_beliefs.get(context_key)
            if belief is None:
                belief = TransitionBelief(scope="feature", context_key=context_key)
                self._feature_beliefs[context_key] = belief
        belief.record(
            outcome,
            weight=weight,
            effect_signature=effect_signature,
            evidence=evidence,
            epoch=epoch,
        )
        self._invalidate_caches()
        return belief

    def retract_outcome(
        self,
        context_key: Any,
        outcome: TransitionOutcome,
        *,
        scope: str = "feature",
    ) -> None:
        """Retract accumulated counts for a specific outcome under a context key."""
        if scope == "locus":
            locus = (int(context_key[0]), int(context_key[1]))
            belief = self._locus_beliefs.get(locus)
        else:
            belief = self._feature_beliefs.get(context_key)
        if belief is not None:
            belief.decay_or_clear_outcome(outcome)
            self._invalidate_caches()

    def retract_outcome_all(self, outcome: TransitionOutcome, *, scope: str = "feature") -> None:
        """Retract an outcome across all recorded beliefs within ``scope``."""
        if scope in ("all", "both"):
            stores = [self._feature_beliefs, self._locus_beliefs]
        else:
            stores = [self._locus_beliefs if scope == "locus" else self._feature_beliefs]
        for store in stores:
            for belief in store.values():
                belief.decay_or_clear_outcome(outcome)
        self._invalidate_caches()

    def record_causal_rule(
        self,
        trigger_key: Any,
        effect_signature: str,
        *,
        precondition_tokens: Sequence[str] = (),
        evidence: str = "",
        confidence: float = 0.8,
    ) -> CausalEffectRule:
        """Register or reinforce a learned state-transition causal rule."""
        _, _, token = _encode_context_key(trigger_key)
        rule_key = (token, effect_signature)
        existing = self._causal_rules.get(rule_key)
        if existing is not None:
            existing.occurrences += 1
            existing.confidence = max(existing.confidence, float(confidence))
            if precondition_tokens:
                existing.precondition_tokens = tuple(precondition_tokens)
            if evidence:
                existing.evidence = evidence
            self._invalidate_caches()
            return existing
        rule = CausalEffectRule(
            trigger_key=trigger_key,
            effect_signature=effect_signature,
            precondition_tokens=tuple(precondition_tokens),
            occurrences=1,
            confidence=float(confidence),
            evidence=evidence,
        )
        self._causal_rules[rule_key] = rule
        self._invalidate_caches()
        return rule

    def get_belief(
        self,
        context_key: Any,
        *,
        scope: str = "feature",
    ) -> TransitionBelief | None:
        """Return stored belief for ``context_key``, if present."""
        if scope == "locus":
            locus = (int(context_key[0]), int(context_key[1]))
            return self._locus_beliefs.get(locus)
        return self._feature_beliefs.get(context_key)

    def query_causal_rules(self, trigger_key: Any | None = None) -> list[CausalEffectRule]:
        """Return all causal effect rules, optionally filtered by trigger key."""
        if trigger_key is None:
            return list(self._causal_rules.values())
        _, _, token = _encode_context_key(trigger_key)
        return [rule for (r_tok, _), rule in self._causal_rules.items() if r_tok == token]

    def evaluate_transition(
        self,
        feature_id: Any,
        *,
        locus: tuple[int, int] | None = None,
        active_state_tokens: frozenset[str] | set[str] | None = None,
        unknown_traversable: bool = True,
        unknown_cost_penalty: float = 0.0,
        risk_weight: float = 50.0,
    ) -> TransitionEdgeEvaluation:
        """Evaluate edge traversability, cost, risk, goal value, and uncertainty."""
        belief: TransitionBelief | None = None
        if locus is not None and self._locus_beliefs:
            locus_belief = self._locus_beliefs.get((int(locus[0]), int(locus[1])))
            if locus_belief is not None and locus_belief._total_obs > 0.0:
                belief = locus_belief
        use_cache = belief is None and not active_state_tokens
        cache_key = (
            (feature_id, unknown_traversable, unknown_cost_penalty, risk_weight)
            if use_cache
            else None
        )
        if cache_key is not None:
            cached_ev = self._feature_eval_cache.get(cache_key)
            if cached_ev is not None:
                return cached_ev

        if belief is None:
            feat_belief = self._feature_beliefs.get(feature_id)
            if feat_belief is not None and feat_belief._total_obs > 0.0:
                belief = feat_belief

        if belief is None:
            res = TransitionEdgeEvaluation(
                traversable=unknown_traversable,
                step_cost=(1.0 + max(0.0, unknown_cost_penalty))
                if unknown_traversable
                else math.inf,
                risk_cost=0.0,
                goal_value=0.0,
                info_gain=1.0,
                mutates_state=False,
                confidence=0.0,
            )
            if cache_key is not None:
                self._feature_eval_cache[cache_key] = res
            return res

        p_trav = belief.traversability_posterior()
        p_fatal = belief._outcome_prob_key(_OUT_TERMINAL_FAIL)
        p_win = belief._outcome_prob_key(_OUT_TERMINAL_WIN)
        p_mut = belief._outcome_prob_key(_OUT_MUTATED_STATE) + belief._outcome_prob_key(
            _OUT_DISPLACED_OTHER
        )

        if active_state_tokens:
            unlock_tokens = {f"unlocks:{feature_id}"}
            if locus is not None:
                unlock_tokens.add(f"unlocks:{(int(locus[0]), int(locus[1]))}")
            if not unlock_tokens.isdisjoint(active_state_tokens):
                p_trav = max(p_trav, 0.9)

        traversable = (p_win > 0.0 or p_trav >= 0.5) and p_fatal < 0.5
        step_cost = (1.0 + max(0.0, 1.0 - p_trav)) if traversable else math.inf
        info_gain = belief.epistemic_uncertainty()
        res = TransitionEdgeEvaluation(
            traversable=traversable,
            step_cost=float(step_cost),
            risk_cost=float(risk_weight * p_fatal),
            goal_value=float(p_win),
            info_gain=info_gain,
            mutates_state=p_mut > 0.0,
            confidence=float(1.0 - info_gain),
        )
        if cache_key is not None:
            self._feature_eval_cache[cache_key] = res
        return res

    def has_feature_outcome(self, feature_id: Any, outcome: TransitionOutcome) -> bool:
        """Return True if ``feature_id`` has recorded observations of ``outcome``."""
        belief = self._feature_beliefs.get(feature_id)
        return belief is not None and belief.counts.get(outcome.value, 0.0) > 0.0

    def features_with_outcome(
        self, outcome: TransitionOutcome, *, min_count: float = 0.0
    ) -> frozenset[Any]:
        """Return feature keys having observed ``outcome`` above ``min_count``."""
        key = outcome.value
        ck = (key, min_count)
        cached = self._outcome_features_cache.get(ck)
        if cached is not None:
            return cached
        res = frozenset(
            k for k, b in self._feature_beliefs.items() if b.counts.get(key, 0.0) > min_count
        )
        self._outcome_features_cache[ck] = res
        return res

    def impassable_features(self) -> frozenset[Any]:
        """Return feature keys empirically evaluated as non-traversable."""
        if self._impassable_cache is not None:
            return self._impassable_cache
        res = frozenset(
            k
            for k, b in self._feature_beliefs.items()
            if (
                b.counts.get(_OUT_BLOCKED, 0.0) > 0.0 or b.counts.get(_OUT_TERMINAL_FAIL, 0.0) > 0.0
            )
            and not self.evaluate_transition(k).traversable
        )
        self._impassable_cache = res
        return res

    def traversable_features(self) -> frozenset[Any]:
        """Return feature keys empirically evaluated as safely traversable."""
        if self._traversable_cache is not None:
            return self._traversable_cache
        res = frozenset(
            k
            for k, b in self._feature_beliefs.items()
            if (
                b.counts.get(_OUT_TRAVERSED, 0.0) > 0.0
                or b.counts.get(_OUT_TERMINAL_WIN, 0.0) > 0.0
            )
            and self.evaluate_transition(k).traversable
        )
        self._traversable_cache = res
        return res

    def loci_with_outcome(
        self, outcome: TransitionOutcome, *, min_count: float = 0.0
    ) -> frozenset[tuple[int, int]]:
        """Return locus coordinates having observed ``outcome`` above ``min_count``."""
        key = outcome.value
        return frozenset(
            (int(k[0]), int(k[1]))
            for k, b in self._locus_beliefs.items()
            if b.counts.get(key, 0.0) > min_count
        )

    def impassable_loci(self) -> frozenset[tuple[int, int]]:
        """Return coordinate loci empirically evaluated as non-traversable."""
        return frozenset(
            (int(k[0]), int(k[1]))
            for k, b in self._locus_beliefs.items()
            if (
                b.counts.get(_OUT_BLOCKED, 0.0) > 0.0 or b.counts.get(_OUT_TERMINAL_FAIL, 0.0) > 0.0
            )
            and not b.traversability_posterior() >= 0.5
        )

    def traversable_loci(self) -> frozenset[tuple[int, int]]:
        """Return coordinate loci empirically evaluated as safely traversable."""
        return frozenset(
            (int(k[0]), int(k[1]))
            for k, b in self._locus_beliefs.items()
            if (
                b.counts.get(_OUT_TRAVERSED, 0.0) > 0.0
                or b.counts.get(_OUT_TERMINAL_WIN, 0.0) > 0.0
            )
            and b.traversability_posterior() >= 0.5
        )

    def begin_context_epoch(self, epoch: int | None = None) -> None:
        """Advance context epoch and drop episode-local coordinate beliefs."""
        self._context_epoch = (self._context_epoch + 1) if epoch is None else int(epoch)
        self._locus_beliefs.clear()
        self._invalidate_caches()

    def clear_context_beliefs(self, *, preserve_current_epoch: bool = False) -> None:
        """Drop locus beliefs and unverified feature beliefs across context shifts."""
        self._locus_beliefs.clear()
        if not preserve_current_epoch:
            self._feature_beliefs.clear()
            self._invalidate_caches()
            return
        retained: dict[Any, TransitionBelief] = {}
        for k, b in self._feature_beliefs.items():
            stale_keys = [
                ok
                for ok in b.counts
                if b.outcome_epochs.get(ok, b.verified_epoch) != self._context_epoch
            ]
            for ok in stale_keys:
                b.counts.pop(ok, None)
                b.outcome_epochs.pop(ok, None)
            if stale_keys:
                b._total_obs = float(sum(b.counts.values()))
            if b.total_observations() > 0.0:
                retained[k] = b
        self._feature_beliefs = retained
        self._invalidate_caches()

    def clear_locus_beliefs(self) -> None:
        """Clear coordinate-scoped locus transition beliefs without clearing feature beliefs."""
        self._locus_beliefs.clear()
        self._invalidate_caches()

    def clear(self) -> None:
        """Reset all stored beliefs, causal rules, and context epoch counters."""
        self._feature_beliefs.clear()
        self._locus_beliefs.clear()
        self._causal_rules.clear()
        self._context_epoch = 0
        self._invalidate_caches()

    def exported_node_ids(self) -> list[str]:
        """Return deterministic HCIR node IDs for all non-empty beliefs and causal rules."""
        node_ids: list[str] = []
        for belief in self._feature_beliefs.values():
            if belief._total_obs > 0.0:
                _, _, nid = _belief_node_id(belief.scope, belief.context_key)
                node_ids.append(nid)
        for belief in self._locus_beliefs.values():
            if belief._total_obs > 0.0:
                _, _, nid = _belief_node_id(belief.scope, belief.context_key)
                node_ids.append(nid)
        for rule in self._causal_rules.values():
            _, _, nid = _rule_node_id(rule.trigger_key, rule.effect_signature)
            node_ids.append(nid)
        return node_ids

    def to_transaction_operations(self) -> list[TransactionOperation]:
        """Export all transition beliefs and causal rules as HCIR graph transactions."""
        if self._cached_tx_ops is not None and self._cached_tx_version == self._version:
            return list(self._cached_tx_ops)
        ops: list[TransactionOperation] = []
        for belief in self._feature_beliefs.values():
            if belief._total_obs <= 0.0:
                continue
            node = belief.to_belief_node()
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node.id,
                    node_data=node.model_dump(),
                )
            )
        for belief in self._locus_beliefs.values():
            if belief._total_obs <= 0.0:
                continue
            node = belief.to_belief_node()
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node.id,
                    node_data=node.model_dump(),
                )
            )
        for rule in self._causal_rules.values():
            node = rule.to_learned_rule_node()
            ops.append(
                TransactionOperation(
                    op=TransactionOp.UPSERT_NODE,
                    node_id=node.id,
                    node_data=node.model_dump(),
                )
            )
        self._cached_tx_ops = ops
        self._cached_tx_version = self._version
        return list(ops)

    def hydrate_from_nodes_with_keys(self, nodes: Sequence[Any]) -> tuple[int, set[Any]]:
        """Hydrate beliefs and causal rules from HCIR nodes, returning (count, feature_keys)."""
        hydrated = 0
        hydrated_feature_keys: set[Any] = set()
        for node in nodes:
            if isinstance(node, BeliefNode):
                belief = TransitionBelief.from_belief_node(node)
                if belief is None:
                    continue
                if belief.scope == "locus" and isinstance(belief.context_key, tuple):
                    self._locus_beliefs[
                        (int(belief.context_key[0]), int(belief.context_key[1]))
                    ] = belief
                else:
                    self._feature_beliefs[belief.context_key] = belief
                    hydrated_feature_keys.add(belief.context_key)
                hydrated += 1
            elif isinstance(node, LearnedRuleNode):
                rule = CausalEffectRule.from_learned_rule_node(node)
                if rule is None:
                    continue
                _, _, token = _encode_context_key(rule.trigger_key)
                self._causal_rules[(token, rule.effect_signature)] = rule
                hydrated += 1
        if hydrated > 0:
            self._invalidate_caches()
        return hydrated, hydrated_feature_keys

    def hydrate_from_nodes(self, nodes: Sequence[Any]) -> int:
        """Hydrate beliefs and causal rules from stored HCIR graph nodes."""
        hydrated, _ = self.hydrate_from_nodes_with_keys(nodes)
        return hydrated


__all__ = [
    "CausalEffectRule",
    "TransitionBelief",
    "TransitionEdgeEvaluation",
    "TransitionKnowledgeBase",
    "TransitionOutcome",
]
