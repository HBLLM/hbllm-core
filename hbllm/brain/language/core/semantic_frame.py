"""Semantic Frame — language-neutral cognitive interface objects for A16.

Core Invariants:
1. Language is an interface to cognition, not cognition itself.
2. Different languages parse into the EXACT SAME language-neutral SemanticFrame.
3. Thematic role-based arguments (agent, patient, recipient, destination, etc.)
   prevent Anglo-centric grammatical bias.
4. Language metadata (tense, politeness, evidentiality) is decoupled from core meaning.
5. Errors in interpretation are explicit epistemic states (UNRESOLVED_LANGUAGE,
   AMBIGUOUS_REFERENCE, etc.), never silent guesses.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class FrameType(StrEnum):
    """Primary intent/illocutionary category of the utterance."""

    ASSERTION = "assertion"  # Stating a fact/observation ("The ball is on the table")
    QUERY = "query"          # Inquiring about state ("Where is the ball?", "Is the cup red?")
    COMMAND = "command"      # Action directive ("Move the box to the table")
    RESPONSE = "response"    # Cognitive answer to a query
    ERROR = "error"          # Unresolvable/ambiguous language state


class ThematicRole(StrEnum):
    """Language-independent thematic roles (Fillmore case grammar)."""

    AGENT = "agent"              # Initiator of action ("The robot pushed the box")
    PATIENT = "patient"          # Entity undergoing change ("The robot pushed the box")
    THEME = "theme"              # Entity whose state/location is described ("The ball is on the table")
    RECIPIENT = "recipient"      # Target receiver ("Give the ball to John")
    SOURCE = "source"            # Origin location ("The ball fell from the table")
    DESTINATION = "destination"  # Target location ("Put the cup on the shelf")
    LOCATION = "location"        # Spatial setting ("The ball is on the table")
    INSTRUMENT = "instrument"    # Tool used ("Pushed with a stick")


class LanguageErrorType(StrEnum):
    """Explicit epistemic error states when parsing/grounding cannot be completed."""

    UNRESOLVED_LANGUAGE = "unresolved_language"            # Sentence cannot be syntactically parsed
    GROUNDING_FAILED = "grounding_failed"                  # Specified entity does not exist in HCIR
    AMBIGUOUS_REFERENCE = "ambiguous_reference"            # Multiple candidate entities match
    UNSUPPORTED_CONSTRUCTION = "unsupported_construction"  # Grammatical pattern not supported
    UNSUPPORTED_LANGUAGE = "unsupported_language"          # Language code not recognized


@dataclass(frozen=True)
class EntityReference:
    """A linguistic reference to an entity or concept."""

    concept_name: str | None = None  # e.g., "ball", "table", "box"
    properties: dict[str, Any] = field(default_factory=dict)  # e.g., {"color": "red"}
    specifier: str = "definite"  # "definite" ('the'), "indefinite" ('a'), "demonstrative" ('this'/'that'), "anaphoric" ('it'/'them')
    raw_text: str = ""
    discourse_id: str | None = None  # Resolved during anaphora resolution


@dataclass
class LanguageMetadata:
    """Language-specific surface metadata decoupled from core semantic meaning."""

    language: str = "en"  # "en", "si", "ja", etc.
    raw_text: str = ""
    tense: str = "present"  # "past", "present", "future"
    aspect: str = "simple"  # "simple", "progressive", "perfect"
    polarity: bool = True  # True = positive, False = negated ("not on the table")
    politeness: str = "neutral"  # "informal", "neutral", "formal", "honorific"
    evidentiality: str = "direct"  # "direct", "reported", "inferred"
    utterance_id: str = field(default_factory=lambda: f"utt_{uuid.uuid4().hex[:8]}")


@dataclass
class SemanticFrame:
    """Language-neutral semantic frame representing structured utterance meaning."""

    frame_type: FrameType
    predicate: str = ""  # e.g., "located_on", "push", "color_of", "supports"
    arguments: dict[ThematicRole, EntityReference] = field(default_factory=dict)
    query_target: str | None = None  # "location", "property", "verification", "existence"
    metadata: LanguageMetadata = field(default_factory=LanguageMetadata)
    error_type: LanguageErrorType | None = None
    error_detail: str = ""
    provenance: str = ""

    @property
    def is_error(self) -> bool:
        return self.frame_type == FrameType.ERROR or self.error_type is not None

    def get_role(self, role: ThematicRole) -> EntityReference | None:
        return self.arguments.get(role)

    def set_role(self, role: ThematicRole, ref: EntityReference) -> None:
        self.arguments[role] = ref


@dataclass
class GroundedSemanticFrame:
    """A SemanticFrame whose entity references have been resolved to HCIR node IDs."""

    frame: SemanticFrame
    grounded_entities: dict[ThematicRole, str] = field(default_factory=dict)  # role -> HCIR node_id
    grounded_concepts: dict[ThematicRole, str] = field(default_factory=dict)  # role -> concept_id
    candidate_entities: dict[ThematicRole, list[str]] = field(default_factory=dict)
    grounding_success: bool = True
    grounding_error: LanguageErrorType | None = None
    grounding_error_detail: str = ""
