"""Tamil language engine for A16 non-LLM multilingual cognition."""

from hbllm.brain.language.tamil.lexicon import TamilLexicalEntry, TamilLexicon, TamilPOS
from hbllm.brain.language.tamil.parser import TamilParser
from hbllm.brain.language.tamil.realizer import TamilRealizer

__all__ = [
    "TamilPOS",
    "TamilLexicalEntry",
    "TamilLexicon",
    "TamilParser",
    "TamilRealizer",
]
