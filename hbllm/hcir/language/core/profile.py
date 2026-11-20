"""Declarative Language Profile Schema for Universal Cognitive Grammar.

Defines the typological, lexical, morphological, and realization properties of
any human language as a declarative data structure, allowing new languages to
be seeded and trained without changing cognitive core code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class TypologyConfig(BaseModel):
    """Typological parameters governing clausal syntax and word order."""

    word_order: str = "SVO"  # "SVO", "SOV", "VSO"
    adposition_type: str = "preposition"  # "preposition" | "postposition"
    wh_movement: str = "front"  # "front" | "in_situ"
    yes_no_strategy: str = "inversion"  # "inversion" | "particle" | "tag"
    copula_style: str = (
        "explicit_verb"  # "explicit_verb" | "zero_copula" | "postpositional_particle"
    )
    has_articles: bool = True
    case_marking: bool = False  # e.g., accusative/dative suffixes or postpositions


class ClosedClassLexicon(BaseModel):
    """Closed-class functional vocabulary mapping surface tokens to HCIR constructs."""

    specifiers: dict[str, str] = Field(
        default_factory=dict
    )  # surface token -> "definite" | "indefinite" | "proximal" | "distal" | "generic"
    adpositions: dict[str, str] = Field(
        default_factory=dict
    )  # surface adposition -> HCIR relation predicate (e.g., "sur" -> "located_on")
    question_markers: dict[str, dict[str, str]] = Field(
        default_factory=dict
    )  # question word -> {"target": "location"|"cause"|"procedure"|"definition", "predicate": ...}
    question_particles: list[str] = Field(
        default_factory=list
    )  # particle markers for yes/no queries (e.g. "ද", "ஆ", "か")
    copula_forms: list[str] = Field(
        default_factory=list
    )  # copular verbs/particles (e.g. ["is", "are"], ["est", "sont"])
    command_verbs: dict[str, str] = Field(
        default_factory=dict
    )  # surface/lemma verb -> capability or action (e.g. "lister" -> "fs.list")
    polite_wrappers: list[str] = Field(
        default_factory=list
    )  # conversational prefixes to strip (e.g. ["s'il vous plaît", "please"])
    expository_markers: dict[str, list[str]] = Field(
        default_factory=dict
    )  # relation cues: "exemplification", "cause", "require", "enable", "because"


class MorphologyConfig(BaseModel):
    """Rule-based inflection and lemmatization parameters."""

    strip_suffixes: list[str] = Field(default_factory=list)
    strip_prefixes: list[str] = Field(default_factory=list)
    plural_endings: list[str] = Field(default_factory=list)
    stemming_rules: list[list[str]] = Field(
        default_factory=list
    )  # list of [target_suffix, replacement] pairs


class RealizationTemplates(BaseModel):
    """Surface realization templates for verbalizing cognitive states."""

    location_known: str = "{theme} is {adposition} {location}."
    location_unknown: str = "I do not know where {theme} is."
    verification_true: str = "Yes, {theme} is {adposition} {location}."
    verification_false: str = "No, {theme} is not {adposition} {location}."
    definition_template: str = "{subject} is {definition}."
    cause_template: str = "{effect} occurs because {cause} occurs."
    procedure_template: str = "{target} requires {step}."
    command_receipt: str = "Goal accepted."
    understood_assertion: str = "Understood."


class LanguageProfile(BaseModel):
    """Complete declarative definition of a natural language frontend for HCIR."""

    iso_code: str
    name: str
    typology: TypologyConfig = Field(default_factory=TypologyConfig)
    closed_class: ClosedClassLexicon = Field(default_factory=ClosedClassLexicon)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)
    templates: RealizationTemplates = Field(default_factory=RealizationTemplates)

    @classmethod
    def from_file(cls, path: Path | str) -> LanguageProfile:
        """Load a declarative language profile from a JSON file."""
        text = Path(path).read_text(encoding="utf-8")
        return cls.model_validate_json(text)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LanguageProfile:
        """Instantiate a profile from a Python dictionary."""
        return cls.model_validate(data)

    def to_file(self, path: Path | str) -> None:
        """Save profile to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
