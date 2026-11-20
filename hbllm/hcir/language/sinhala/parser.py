"""Sinhala Semantic Parser for A16.

Parses Sinhala utterances (SOV word order + postpositions) into
language-neutral SemanticFrames.
Implements the LanguageParser protocol via the Universal Declarative Engine.
"""

from __future__ import annotations

from pathlib import Path

from hbllm.hcir.language.core.profile import LanguageProfile
from hbllm.hcir.language.core.universal_engine import UniversalLanguageParser
from hbllm.hcir.language.sinhala.lexicon import SinhalaLexicon

_SI_PROFILE_PATH = Path(__file__).resolve().parent.parent / "profiles" / "si.json"


class SinhalaParser(UniversalLanguageParser):
    """Parses Sinhala utterances into language-neutral SemanticFrames using Universal Engine."""

    def __init__(self, lexicon: SinhalaLexicon | None = None) -> None:
        profile = LanguageProfile.from_file(_SI_PROFILE_PATH)
        lex = lexicon or SinhalaLexicon()
        super().__init__(profile=profile, lexicon=lex)
        self._lexicon = lex
