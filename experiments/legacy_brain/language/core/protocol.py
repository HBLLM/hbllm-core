"""Protocol definitions for A16 language engines.

Decouples the core cognitive runtime from language-specific implementations.
Any language frontend (English, Sinhala, Japanese, etc.) implements these protocols.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from hbllm.brain.language.core.epistemic_policy import CognitiveEpistemicState
from hbllm.brain.language.core.semantic_frame import SemanticFrame


@runtime_checkable
class LanguageParser(Protocol):
    """Protocol for parsing a natural language string into a SemanticFrame."""

    def parse(self, text: str) -> SemanticFrame:
        """Parse raw utterance text into a language-neutral SemanticFrame."""
        ...


@runtime_checkable
class LanguageRealizer(Protocol):
    """Protocol for verbalizing cognitive epistemic states into natural text."""

    def realize(
        self,
        epistemic_state: CognitiveEpistemicState,
        original_frame: SemanticFrame | None = None,
    ) -> str:
        """Generate natural language utterance from EpistemicState."""
        ...

    def realize_frame(self, frame: SemanticFrame) -> str:
        """Generate natural language directly from a SemanticFrame (e.g. for translation)."""
        ...
