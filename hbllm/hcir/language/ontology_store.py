"""Backward-compatible re-export — canonical location is ``hbllm.hcir.stores.ontology_store``."""

from hbllm.hcir.stores.ontology_store import (  # noqa: F401
    DEFAULT_ONTOLOGY_DB,
    LexicalOntologyStore,
    StoredLexicalEntry,
)

__all__ = ["DEFAULT_ONTOLOGY_DB", "LexicalOntologyStore", "StoredLexicalEntry"]
