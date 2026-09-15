"""Cross-Lingual Grounding & Multilingual Ingestion Engine (Milestone A24).

Binds natural language tokens across multiple languages (English, French, Spanish, German)
directly into invariant physical simulation primitives (BabyObjectType, BabyActionType,
BabyRelationType) and causal schemas. Enables the developmental agent to read foreign literature,
ground concepts into language-agnostic physical invariants, and answer cross-lingual Socratic queries.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from .types import (
    BabyActionType,
    BabyObjectType,
    BabyRelationType,
)

logger = logging.getLogger(__name__)


class SupportedLanguage(str, Enum):
    """Officially supported natural languages for cross-lingual grounding."""

    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"
    GERMAN = "de"


class MultilingualLexiconRegistry:
    """Translation and semantic grounding registry across supported languages."""

    # Stopwords used for fast language detection
    STOPWORDS: dict[SupportedLanguage, set[str]] = {
        SupportedLanguage.ENGLISH: {
            "the",
            "and",
            "is",
            "in",
            "it",
            "of",
            "to",
            "with",
            "that",
            "this",
        },
        SupportedLanguage.FRENCH: {
            "le",
            "la",
            "les",
            "et",
            "est",
            "dans",
            "un",
            "une",
            "des",
            "du",
            "pour",
        },
        SupportedLanguage.SPANISH: {
            "el",
            "la",
            "los",
            "las",
            "y",
            "es",
            "en",
            "un",
            "una",
            "de",
            "para",
            "con",
        },
        SupportedLanguage.GERMAN: {
            "der",
            "die",
            "das",
            "und",
            "ist",
            "in",
            "ein",
            "eine",
            "mit",
            "für",
            "den",
        },
    }

    # Copula patterns for definition extraction across languages
    DEFINITION_PATTERNS: dict[SupportedLanguage, list[str]] = {
        SupportedLanguage.ENGLISH: [
            r"\b([A-Za-z\-]{3,20})\s+(?:is|are|means|denotes|refers to)\s+([^.\n]{15,120})\.",
        ],
        SupportedLanguage.FRENCH: [
            r"\b([A-Za-z\-]{3,20})\s+(?:est|sont|désigne|signifie)\s+([^.\n]{15,120})\.",
        ],
        SupportedLanguage.SPANISH: [
            r"\b([A-Za-z\-]{3,20})\s+(?:es|son|significa|denota)\s+([^.\n]{15,120})\.",
        ],
        SupportedLanguage.GERMAN: [
            r"\b([A-Za-z\-]{3,20})\s+(?:ist|sind|bedeutet|bezeichnet)\s+([^.\n]{15,120})\.",
        ],
    }

    # Multilingual token mappings into core physical/relational invariants (dynamically loaded)
    CONTAINER_TOKENS: dict[str, SupportedLanguage] = {}
    TOOL_TOKENS: dict[str, SupportedLanguage] = {}
    BALL_TOKENS: dict[str, SupportedLanguage] = {}
    BLOCK_TOKENS: dict[str, SupportedLanguage] = {}
    ACTION_TOKENS: dict[str, tuple[BabyActionType, SupportedLanguage]] = {}
    RELATION_TOKENS: dict[str, tuple[BabyRelationType, SupportedLanguage]] = {}
    COLOR_TOKENS: dict[str, tuple[str, SupportedLanguage]] = {}

    @classmethod
    def load_translation_file(cls, tsv_path: Path | str | None = None) -> int:
        """Load multilingual translation tokens dynamically from a curriculum data file."""
        if tsv_path is None:
            tsv_path = Path(__file__).parent / "curriculum_data" / "multilingual_lexicon.tsv"
        else:
            tsv_path = Path(tsv_path)

        if not tsv_path.exists():
            logger.warning(f"Translation file not found at: {tsv_path}")
            return 0

        count = 0
        try:
            with open(tsv_path, encoding="utf-8") as f:
                lines = f.read().strip().splitlines()
            if not lines:
                return 0

            for line in lines[1:]:
                if not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) < 5:
                    continue
                token = parts[0].strip().lower()
                lang_code = parts[1].strip().lower()
                category = parts[2].strip().upper()
                role = parts[3].strip().upper()
                canonical = parts[4].strip().lower()

                try:
                    lang = SupportedLanguage(lang_code)
                except ValueError:
                    continue

                if role == "CONTAINER":
                    cls.CONTAINER_TOKENS[token] = lang
                elif role == "TOOL":
                    cls.TOOL_TOKENS[token] = lang
                elif role == "BALL":
                    cls.BALL_TOKENS[token] = lang
                elif role == "BLOCK":
                    cls.BLOCK_TOKENS[token] = lang
                elif category == "VERB":
                    try:
                        act_type = BabyActionType(canonical)
                    except ValueError:
                        act_type = getattr(BabyActionType, role, BabyActionType.PUSH)
                    cls.ACTION_TOKENS[token] = (act_type, lang)
                elif category == "PREPOSITION":
                    try:
                        rel_type = BabyRelationType(canonical)
                    except ValueError:
                        rel_type = getattr(BabyRelationType, role, BabyRelationType.INSIDE)
                    cls.RELATION_TOKENS[token] = (rel_type, lang)
                elif category == "ADJECTIVE":
                    cls.COLOR_TOKENS[token] = (canonical, lang)

                count += 1
        except Exception as e:
            logger.warning(f"Error loading multilingual lexicon from {tsv_path}: {e}")

        return count

    @classmethod
    def detect_language(cls, text: str) -> SupportedLanguage:
        """Detect language of input text via header metadata or stopword frequencies."""
        text_lower = text[:5000].lower()

        # Check explicit Project Gutenberg header
        if "language: french" in text_lower or "langue: français" in text_lower:
            return SupportedLanguage.FRENCH
        if "language: spanish" in text_lower or "idioma: español" in text_lower:
            return SupportedLanguage.SPANISH
        if "language: german" in text_lower or "sprache: deutsch" in text_lower:
            return SupportedLanguage.GERMAN
        if "language: english" in text_lower:
            return SupportedLanguage.ENGLISH

        # Frequency scoring via token set intersection
        words = set(re.findall(r"\b[a-zàâéèêëîïôùûüÿçñäöß]{2,15}\b", text_lower))
        best_lang = SupportedLanguage.ENGLISH
        best_score = 0

        for lang, stopwords in cls.STOPWORDS.items():
            common = len(words & stopwords)
            if common > best_score:
                best_score = common
                best_lang = lang

        return best_lang

    @classmethod
    def ground_multilingual_term(cls, term: str) -> dict[str, Any] | None:
        """Map any supported multilingual surface word into canonical physical simulation types."""
        t_clean = term.strip().lower()

        # Check container
        if t_clean in cls.CONTAINER_TOKENS:
            return {
                "canonical_name": "box",
                "entity_type": BabyObjectType.BOX,
                "is_container": True,
                "language": cls.CONTAINER_TOKENS[t_clean],
            }

        # Check tool
        if t_clean in cls.TOOL_TOKENS:
            return {
                "canonical_name": "stick",
                "entity_type": BabyObjectType.TOOL,
                "is_tool": True,
                "language": cls.TOOL_TOKENS[t_clean],
            }

        # Check ball
        if t_clean in cls.BALL_TOKENS:
            return {
                "canonical_name": "ball",
                "entity_type": BabyObjectType.BALL,
                "rollable": True,
                "language": cls.BALL_TOKENS[t_clean],
            }

        # Check block
        if t_clean in cls.BLOCK_TOKENS:
            return {
                "canonical_name": "block",
                "entity_type": BabyObjectType.BLOCK,
                "rollable": False,
                "language": cls.BLOCK_TOKENS[t_clean],
            }

        # Check action
        if t_clean in cls.ACTION_TOKENS:
            action, lang = cls.ACTION_TOKENS[t_clean]
            return {"action": action, "language": lang}

        # Check relation
        if t_clean in cls.RELATION_TOKENS:
            relation, lang = cls.RELATION_TOKENS[t_clean]
            return {"relation": relation, "language": lang}

        # Check color
        if t_clean in cls.COLOR_TOKENS:
            color, lang = cls.COLOR_TOKENS[t_clean]
            return {"color": color, "language": lang}

        return None

    @classmethod
    def parse_multilingual_worked_problem(cls, instruction: str) -> dict[str, Any]:
        """Parse natural language instruction in EN, FR, ES, or DE into a simulation configuration."""
        inst_low = instruction.lower()
        words = re.findall(r"\b[a-zàâéèêëîïôùûüÿçñäöß\-]{2,20}\b", inst_low)

        target_shape = BabyObjectType.BALL
        target_color = "red"
        target_action = BabyActionType.PUSH
        target_relation = BabyRelationType.INSIDE
        needs_tool = False

        for w in words:
            grounded = cls.ground_multilingual_term(w)
            if not grounded:
                continue

            if "entity_type" in grounded:
                if grounded["entity_type"] in (BabyObjectType.BALL, BabyObjectType.BLOCK):
                    target_shape = grounded["entity_type"]
                elif grounded["entity_type"] == BabyObjectType.TOOL:
                    needs_tool = True
            elif "action" in grounded:
                target_action = grounded["action"]
            elif "relation" in grounded:
                target_relation = grounded["relation"]
            elif "color" in grounded:
                target_color = grounded["color"]

        if target_action == BabyActionType.PULL:
            needs_tool = True

        return {
            "instruction": instruction,
            "target_shape": target_shape,
            "target_color": target_color,
            "target_action": target_action,
            "target_relation": target_relation,
            "needs_tool": needs_tool,
        }


# Bootstrap multilingual token tables dynamically from curriculum data file
MultilingualLexiconRegistry.load_translation_file()
