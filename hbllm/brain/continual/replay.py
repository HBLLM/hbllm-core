"""Tri-Modal Sleep Replay Engine for A22.

Implements salience-guided offline replay across three modalities:
1. ERROR_REPLAY: Replays high-prediction-error failures to identify boundary constraints.
2. SUCCESS_REPLAY: Replays novel successful trajectories to consolidate reusable schemas.
3. CONTRASTIVE_REPLAY: Replays success vs near-identical failure pairs to isolate critical structural deltas.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hbllm.brain.continual.store import EpisodicTrace

logger = logging.getLogger(__name__)


class ReplayKind(str, Enum):
    """The three modalities of offline sleep replay."""

    ERROR_REPLAY = "error_replay"
    SUCCESS_REPLAY = "success_replay"
    CONTRASTIVE_REPLAY = "contrastive_replay"


@dataclass
class ContrastivePair:
    """A pair of near-identical episodes that produced opposing physical outcomes."""

    pair_id: str = field(default_factory=lambda: f"cp_{uuid.uuid4().hex[:8]}")
    domain: str = ""
    success_trace: EpisodicTrace | None = None
    failure_trace: EpisodicTrace | None = None
    isolated_delta: dict[str, tuple[Any, Any]] = field(
        default_factory=dict
    )  # prop -> (success_val, failure_val)
    salience_score: float = 0.90


@dataclass
class ReplayCandidate:
    """A prioritized candidate sequence selected for mental replay and consolidation."""

    candidate_id: str = field(default_factory=lambda: f"rc_{uuid.uuid4().hex[:8]}")
    kind: ReplayKind = ReplayKind.SUCCESS_REPLAY
    domain: str = ""
    traces: list[EpisodicTrace] = field(default_factory=list)
    salience_score: float = 0.50
    contrastive_pair: ContrastivePair | None = None


class SleepReplayEngine:
    """Identifies and constructs salience-weighted replay candidates from episodic memory."""

    def filter_and_prioritize_replays(
        self,
        traces: list[EpisodicTrace],
        domain_uncertainties: dict[str, float] | None = None,
    ) -> list[ReplayCandidate]:
        """Select top replay candidates based on prediction errors, novelty, and metacognitive uncertainty."""
        uncertainties = domain_uncertainties or {}
        candidates: list[ReplayCandidate] = []

        # 1. Separate successes and failures
        successes_by_domain: dict[str, list[EpisodicTrace]] = {}
        failures_by_domain: dict[str, list[EpisodicTrace]] = {}

        for t in traces:
            u_weight = uncertainties.get(t.domain, 0.30)
            salience = (t.prediction_error * 0.6) + (u_weight * 0.4)
            t.salience_score = round(salience, 4)

            if t.is_success:
                successes_by_domain.setdefault(t.domain, []).append(t)
            else:
                failures_by_domain.setdefault(t.domain, []).append(t)

        # 2. Formulate Contrastive Replay Pairs
        for domain, succ_list in successes_by_domain.items():
            fail_list = failures_by_domain.get(domain, [])
            for s in succ_list:
                for f in fail_list:
                    # Check if action sequences are identical/equivalent
                    if len(s.actions) == len(f.actions) and [a[0] for a in s.actions] == [
                        a[0] for a in f.actions
                    ]:
                        delta = self._isolate_context_delta(s.context_props, f.context_props)
                        if delta:
                            cp = ContrastivePair(
                                domain=domain,
                                success_trace=s,
                                failure_trace=f,
                                isolated_delta=delta,
                                salience_score=0.95,
                            )
                            candidates.append(
                                ReplayCandidate(
                                    kind=ReplayKind.CONTRASTIVE_REPLAY,
                                    domain=domain,
                                    traces=[s, f],
                                    salience_score=0.95,
                                    contrastive_pair=cp,
                                )
                            )

        # 3. Formulate Error Replays (High prediction error)
        for domain, fail_list in failures_by_domain.items():
            for f in fail_list:
                if f.prediction_error >= 0.50:
                    candidates.append(
                        ReplayCandidate(
                            kind=ReplayKind.ERROR_REPLAY,
                            domain=domain,
                            traces=[f],
                            salience_score=min(1.0, f.salience_score + 0.20),
                        )
                    )

        # 4. Formulate Success Replays (Novel successful strategies)
        for domain, succ_list in successes_by_domain.items():
            for s in succ_list:
                candidates.append(
                    ReplayCandidate(
                        kind=ReplayKind.SUCCESS_REPLAY,
                        domain=domain,
                        traces=[s],
                        salience_score=s.salience_score,
                    )
                )

        # Sort descending by salience
        candidates.sort(key=lambda c: c.salience_score, reverse=True)
        return candidates

    def _isolate_context_delta(
        self,
        succ_props: dict[str, Any],
        fail_props: dict[str, Any],
    ) -> dict[str, tuple[Any, Any]]:
        """Isolate differing context properties between two episodes."""
        delta: dict[str, tuple[Any, Any]] = {}
        all_keys = set(succ_props.keys()) | set(fail_props.keys())
        for k in all_keys:
            v_succ = succ_props.get(k)
            v_fail = fail_props.get(k)
            if v_succ != v_fail:
                delta[k] = (v_succ, v_fail)
        return delta
