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

    # Multilingual token mappings into core physical/relational invariants
    CONTAINER_TOKENS: dict[str, SupportedLanguage] = {
        # English
        "box": SupportedLanguage.ENGLISH,
        "container": SupportedLanguage.ENGLISH,
        "vessel": SupportedLanguage.ENGLISH,
        "crate": SupportedLanguage.ENGLISH,
        "bin": SupportedLanguage.ENGLISH,
        "chest": SupportedLanguage.ENGLISH,
        "crucible": SupportedLanguage.ENGLISH,
        # French
        "boîte": SupportedLanguage.FRENCH,
        "boite": SupportedLanguage.FRENCH,
        "récipient": SupportedLanguage.FRENCH,
        "recipient": SupportedLanguage.FRENCH,
        "vaisseau": SupportedLanguage.FRENCH,
        "caisse": SupportedLanguage.FRENCH,
        "coffre": SupportedLanguage.FRENCH,
        "creuset": SupportedLanguage.FRENCH,
        # Spanish
        "caja": SupportedLanguage.SPANISH,
        "recipiente": SupportedLanguage.SPANISH,
        "vaso": SupportedLanguage.SPANISH,
        "arca": SupportedLanguage.SPANISH,
        "cofre": SupportedLanguage.SPANISH,
        "crisol": SupportedLanguage.SPANISH,
        # German
        "kasten": SupportedLanguage.GERMAN,
        "schachtel": SupportedLanguage.GERMAN,
        "behälter": SupportedLanguage.GERMAN,
        "behalter": SupportedLanguage.GERMAN,
        "gefäß": SupportedLanguage.GERMAN,
        "gefaess": SupportedLanguage.GERMAN,
        "kiste": SupportedLanguage.GERMAN,
        "tiegel": SupportedLanguage.GERMAN,
    }

    TOOL_TOKENS: dict[str, SupportedLanguage] = {
        # English
        "stick": SupportedLanguage.ENGLISH,
        "tool": SupportedLanguage.ENGLISH,
        "lever": SupportedLanguage.ENGLISH,
        "rod": SupportedLanguage.ENGLISH,
        "bar": SupportedLanguage.ENGLISH,
        "prybar": SupportedLanguage.ENGLISH,
        # French
        "bâton": SupportedLanguage.FRENCH,
        "baton": SupportedLanguage.FRENCH,
        "outil": SupportedLanguage.FRENCH,
        "levier": SupportedLanguage.FRENCH,
        "tige": SupportedLanguage.FRENCH,
        "barre": SupportedLanguage.FRENCH,
        "pince": SupportedLanguage.FRENCH,
        # Spanish
        "palo": SupportedLanguage.SPANISH,
        "herramienta": SupportedLanguage.SPANISH,
        "palanca": SupportedLanguage.SPANISH,
        "vara": SupportedLanguage.SPANISH,
        "barra": SupportedLanguage.SPANISH,
        "pinzas": SupportedLanguage.SPANISH,
        # German
        "stock": SupportedLanguage.GERMAN,
        "werkzeug": SupportedLanguage.GERMAN,
        "hebel": SupportedLanguage.GERMAN,
        "stab": SupportedLanguage.GERMAN,
        "stange": SupportedLanguage.GERMAN,
        "zange": SupportedLanguage.GERMAN,
    }

    BALL_TOKENS: dict[str, SupportedLanguage] = {
        # English
        "ball": SupportedLanguage.ENGLISH,
        "sphere": SupportedLanguage.ENGLISH,
        "marble": SupportedLanguage.ENGLISH,
        "globe": SupportedLanguage.ENGLISH,
        # French
        "balle": SupportedLanguage.FRENCH,
        "sphère": SupportedLanguage.FRENCH,
        "bille": SupportedLanguage.FRENCH,
        # Spanish
        "pelota": SupportedLanguage.SPANISH,
        "esfera": SupportedLanguage.SPANISH,
        "bola": SupportedLanguage.SPANISH,
        "canica": SupportedLanguage.SPANISH,
        # German
        "kugel": SupportedLanguage.GERMAN,
        "sphäre": SupportedLanguage.GERMAN,
        "murmel": SupportedLanguage.GERMAN,
    }

    BLOCK_TOKENS: dict[str, SupportedLanguage] = {
        # English
        "block": SupportedLanguage.ENGLISH,
        "cube": SupportedLanguage.ENGLISH,
        "brick": SupportedLanguage.ENGLISH,
        "stone": SupportedLanguage.ENGLISH,
        # French
        "bloc": SupportedLanguage.FRENCH,
        "brique": SupportedLanguage.FRENCH,
        "pierre": SupportedLanguage.FRENCH,
        # Spanish
        "bloque": SupportedLanguage.SPANISH,
        "cubo": SupportedLanguage.SPANISH,
        "ladrillo": SupportedLanguage.SPANISH,
        "piedra": SupportedLanguage.SPANISH,
        # German
        "klotz": SupportedLanguage.GERMAN,
        "würfel": SupportedLanguage.GERMAN,
        "wuerfel": SupportedLanguage.GERMAN,
        "stein": SupportedLanguage.GERMAN,
    }

    ACTION_TOKENS: dict[str, tuple[BabyActionType, SupportedLanguage]] = {
        # PULL
        "pull": (BabyActionType.PULL, SupportedLanguage.ENGLISH),
        "drag": (BabyActionType.PULL, SupportedLanguage.ENGLISH),
        "haul": (BabyActionType.PULL, SupportedLanguage.ENGLISH),
        "tirer": (BabyActionType.PULL, SupportedLanguage.FRENCH),
        "traîner": (BabyActionType.PULL, SupportedLanguage.FRENCH),
        "trainer": (BabyActionType.PULL, SupportedLanguage.FRENCH),
        "tirar": (BabyActionType.PULL, SupportedLanguage.SPANISH),
        "jalar": (BabyActionType.PULL, SupportedLanguage.SPANISH),
        "arrastrar": (BabyActionType.PULL, SupportedLanguage.SPANISH),
        "ziehen": (BabyActionType.PULL, SupportedLanguage.GERMAN),
        "schleppen": (BabyActionType.PULL, SupportedLanguage.GERMAN),
        # PUSH
        "push": (BabyActionType.PUSH, SupportedLanguage.ENGLISH),
        "shove": (BabyActionType.PUSH, SupportedLanguage.ENGLISH),
        "press": (BabyActionType.PUSH, SupportedLanguage.ENGLISH),
        "pousser": (BabyActionType.PUSH, SupportedLanguage.FRENCH),
        "presser": (BabyActionType.PUSH, SupportedLanguage.FRENCH),
        "empujar": (BabyActionType.PUSH, SupportedLanguage.SPANISH),
        "presionar": (BabyActionType.PUSH, SupportedLanguage.SPANISH),
        "drücken": (BabyActionType.PUSH, SupportedLanguage.GERMAN),
        "druecken": (BabyActionType.PUSH, SupportedLanguage.GERMAN),
        "schieben": (BabyActionType.PUSH, SupportedLanguage.GERMAN),
        # ROLL
        "roll": (BabyActionType.ROLL, SupportedLanguage.ENGLISH),
        "rouler": (BabyActionType.ROLL, SupportedLanguage.FRENCH),
        "rodar": (BabyActionType.ROLL, SupportedLanguage.SPANISH),
        "rollen": (BabyActionType.ROLL, SupportedLanguage.GERMAN),
        # GRASP
        "grasp": (BabyActionType.GRASP, SupportedLanguage.ENGLISH),
        "hold": (BabyActionType.GRASP, SupportedLanguage.ENGLISH),
        "saisir": (BabyActionType.GRASP, SupportedLanguage.FRENCH),
        "tenir": (BabyActionType.GRASP, SupportedLanguage.FRENCH),
        "agarrar": (BabyActionType.GRASP, SupportedLanguage.SPANISH),
        "sujetar": (BabyActionType.GRASP, SupportedLanguage.SPANISH),
        "greifen": (BabyActionType.GRASP, SupportedLanguage.GERMAN),
        "halten": (BabyActionType.GRASP, SupportedLanguage.GERMAN),
    }

    RELATION_TOKENS: dict[str, tuple[BabyRelationType, SupportedLanguage]] = {
        # INSIDE
        "inside": (BabyRelationType.INSIDE, SupportedLanguage.ENGLISH),
        "in": (BabyRelationType.INSIDE, SupportedLanguage.ENGLISH),
        "within": (BabyRelationType.INSIDE, SupportedLanguage.ENGLISH),
        "dans": (BabyRelationType.INSIDE, SupportedLanguage.FRENCH),
        "dedans": (BabyRelationType.INSIDE, SupportedLanguage.FRENCH),
        "en": (BabyRelationType.INSIDE, SupportedLanguage.SPANISH),
        "dentro": (BabyRelationType.INSIDE, SupportedLanguage.SPANISH),
        "drinnen": (BabyRelationType.INSIDE, SupportedLanguage.GERMAN),
        "innerhalb": (BabyRelationType.INSIDE, SupportedLanguage.GERMAN),
        # NEAR
        "near": (BabyRelationType.NEAR, SupportedLanguage.ENGLISH),
        "beside": (BabyRelationType.NEAR, SupportedLanguage.ENGLISH),
        "près": (BabyRelationType.NEAR, SupportedLanguage.FRENCH),
        "pres": (BabyRelationType.NEAR, SupportedLanguage.FRENCH),
        "cerca": (BabyRelationType.NEAR, SupportedLanguage.SPANISH),
        "junto": (BabyRelationType.NEAR, SupportedLanguage.SPANISH),
        "nah": (BabyRelationType.NEAR, SupportedLanguage.GERMAN),
        "nahe": (BabyRelationType.NEAR, SupportedLanguage.GERMAN),
        "neben": (BabyRelationType.NEAR, SupportedLanguage.GERMAN),
    }

    COLOR_TOKENS: dict[str, tuple[str, SupportedLanguage]] = {
        # Red
        "red": ("red", SupportedLanguage.ENGLISH),
        "rouge": ("red", SupportedLanguage.FRENCH),
        "rouges": ("red", SupportedLanguage.FRENCH),
        "rojo": ("red", SupportedLanguage.SPANISH),
        "roja": ("red", SupportedLanguage.SPANISH),
        "rojos": ("red", SupportedLanguage.SPANISH),
        "rojas": ("red", SupportedLanguage.SPANISH),
        "rot": ("red", SupportedLanguage.GERMAN),
        "rote": ("red", SupportedLanguage.GERMAN),
        "roten": ("red", SupportedLanguage.GERMAN),
        "rotes": ("red", SupportedLanguage.GERMAN),
        # Green
        "green": ("green", SupportedLanguage.ENGLISH),
        "vert": ("green", SupportedLanguage.FRENCH),
        "verte": ("green", SupportedLanguage.FRENCH),
        "verts": ("green", SupportedLanguage.FRENCH),
        "vertes": ("green", SupportedLanguage.FRENCH),
        "verde": ("green", SupportedLanguage.SPANISH),
        "verdes": ("green", SupportedLanguage.SPANISH),
        "grün": ("green", SupportedLanguage.GERMAN),
        "grüne": ("green", SupportedLanguage.GERMAN),
        "grünen": ("green", SupportedLanguage.GERMAN),
        "grünes": ("green", SupportedLanguage.GERMAN),
        "gruen": ("green", SupportedLanguage.GERMAN),
        "gruene": ("green", SupportedLanguage.GERMAN),
        "gruenen": ("green", SupportedLanguage.GERMAN),
        # Blue
        "blue": ("blue", SupportedLanguage.ENGLISH),
        "bleu": ("blue", SupportedLanguage.FRENCH),
        "bleue": ("blue", SupportedLanguage.FRENCH),
        "bleus": ("blue", SupportedLanguage.FRENCH),
        "bleues": ("blue", SupportedLanguage.FRENCH),
        "azul": ("blue", SupportedLanguage.SPANISH),
        "azules": ("blue", SupportedLanguage.SPANISH),
        "blau": ("blue", SupportedLanguage.GERMAN),
        "blaue": ("blue", SupportedLanguage.GERMAN),
        "blauen": ("blue", SupportedLanguage.GERMAN),
        "blaues": ("blue", SupportedLanguage.GERMAN),
        # Yellow
        "yellow": ("yellow", SupportedLanguage.ENGLISH),
        "jaune": ("yellow", SupportedLanguage.FRENCH),
        "jaunes": ("yellow", SupportedLanguage.FRENCH),
        "amarillo": ("yellow", SupportedLanguage.SPANISH),
        "amarilla": ("yellow", SupportedLanguage.SPANISH),
        "amarillos": ("yellow", SupportedLanguage.SPANISH),
        "amarillas": ("yellow", SupportedLanguage.SPANISH),
        "gelb": ("yellow", SupportedLanguage.GERMAN),
        "gelbe": ("yellow", SupportedLanguage.GERMAN),
        "gelben": ("yellow", SupportedLanguage.GERMAN),
        "gelbes": ("yellow", SupportedLanguage.GERMAN),
    }

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
