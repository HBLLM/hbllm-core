"""Tamil language engine for A16 non-LLM multilingual cognition."""

from hbllm.hcir.language.tamil.lexicon import TamilLexicalEntry, TamilLexicon, TamilPOS
from hbllm.hcir.language.tamil.parser import TamilParser
from hbllm.hcir.language.tamil.realizer import TamilRealizer

__all__ = [
    "TamilPOS",
    "TamilLexicalEntry",
    "TamilLexicon",
    "TamilParser",
    "TamilRealizer",
]
