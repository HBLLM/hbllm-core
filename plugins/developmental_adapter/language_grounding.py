"""Grounded Lexical Acquisition Engine (Stage D10).

Acquires an embodied vocabulary by fast-mapping linguistic tokens
onto pre-existing cognitive graph entities, properties, actions, and spatial relations
via cross-situational exposure without a pre-compiled lexicon.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .blank_brain import BlankBrainSubstrate
from .environment import BabyWorldEnvironment
from .types import (
    BabyActionType,
    BabyObjectType,
    BabyRelationType,
    BeliefTransitionEvent,
    BeliefTransitionType,
    LexicalCategory,
    LexicalEntry,
)

logger = logging.getLogger(__name__)


class LanguageGroundingEngine:
    """Statistical cross-situational learner for vocabulary acquisition."""

    def __init__(self, substrate: BlankBrainSubstrate, env: BabyWorldEnvironment) -> None:
        self.substrate = substrate
        self.env = env
        self.lexicon: dict[str, LexicalEntry] = {}
        self.co_occurrence_matrix: dict[str, dict[str, int]] = {}

    def tokenize(self, utterance: str) -> list[str]:
        """Split teacher utterance into normalized linguistic tokens."""
        clean = re.sub(r"[^\w\s]", "", utterance.lower())
        tokens = [t.strip() for t in clean.split() if t.strip()]
        # Filter purely functional filler words
        stop_words = {"the", "a", "an", "at", "look", "is"}
        return [t for t in tokens if t not in stop_words]

    def observe_paired_demonstration(self, utterance: str, scene_context: dict[str, Any]) -> None:
        """Process one paired utterance-scene demonstration to update co-occurrences."""
        tokens = self.tokenize(utterance)

        # Extract referents present in context
        grounded_candidates: list[tuple[str, LexicalCategory]] = []

        if "action" in scene_context:
            act = scene_context["action"]
            act_val = act.value if isinstance(act, BabyActionType) else str(act).upper()
            grounded_candidates.append((act_val, LexicalCategory.VERB))

        if "relation" in scene_context:
            rel = scene_context["relation"]
            rel_val = rel.value if isinstance(rel, BabyRelationType) else str(rel).upper()
            grounded_candidates.append((rel_val, LexicalCategory.PREPOSITION))

        if "entity_type" in scene_context:
            etype = scene_context["entity_type"]
            etype_val = etype.value if isinstance(etype, BabyObjectType) else str(etype).lower()
            grounded_candidates.append((etype_val, LexicalCategory.NOUN))

        if "color" in scene_context:
            grounded_candidates.append(
                (str(scene_context["color"]).lower(), LexicalCategory.ADJECTIVE)
            )

        if "property" in scene_context:
            grounded_candidates.append(
                (str(scene_context["property"]).lower(), LexicalCategory.ADJECTIVE)
            )

        if "instrument" in scene_context:
            grounded_candidates.append(("INSTRUMENT_WITH", LexicalCategory.PREPOSITION))

        # Cross-situational associative updates
        for token in tokens:
            if token not in self.co_occurrence_matrix:
                self.co_occurrence_matrix[token] = {}

            for symbol, cat in grounded_candidates:
                key = f"{cat.value}:{symbol}"
                self.co_occurrence_matrix[token][key] = (
                    self.co_occurrence_matrix[token].get(key, 0) + 1
                )

        # Update best grounded entry for each token
        for token, associations in self.co_occurrence_matrix.items():
            if not associations:
                continue
            best_target, max_count = max(associations.items(), key=lambda item: item[1])
            cat_str, sym = best_target.split(":", 1)
            cat = LexicalCategory(cat_str)

            conf = min(0.99, 0.4 + 0.2 * max_count)
            entry = LexicalEntry(
                token=token,
                category=cat,
                grounded_symbol=sym,
                co_occurrence_count=max_count,
                confidence=conf,
            )
            self.lexicon[token] = entry
            self.substrate.lexical_mapping[token] = sym

            if hasattr(self.substrate, "profile") and hasattr(
                self.substrate.profile, "belief_transitions"
            ):
                self.substrate.profile.belief_transitions.append(
                    BeliefTransitionEvent(
                        event_type=BeliefTransitionType.LEXICON_GROUNDED,
                        variable="lexicon",
                        condition=f"{token} -> {sym} ({cat.value})",
                        posterior_confidence=conf,
                        evidence={"count": max_count},
                    )
                )

    def ground_utterance(self, utterance: str) -> list[LexicalEntry]:
        """Translate an incoming sentence into its grounded cognitive entries."""
        tokens = self.tokenize(utterance)
        return [self.lexicon[t] for t in tokens if t in self.lexicon]
