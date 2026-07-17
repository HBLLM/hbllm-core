"""
Knowledge Base — file/folder ingestion, chunking, and semantic search.

Core knowledge management backed by HBLLM SemanticMemory.
"""

from hbllm.core.constants import SKIP_DIRS
from hbllm.knowledge.knowledge_base import (
    MARKITDOWN_EXTENSIONS,
    MAX_FILE_SIZE,
    SUPPORTED_EXTENSIONS,
    KnowledgeBase,
    Source,
    get_all_supported_extensions,
)

__all__ = [
    "KnowledgeBase",
    "Source",
    "SUPPORTED_EXTENSIONS",
    "MARKITDOWN_EXTENSIONS",
    "SKIP_DIRS",
    "MAX_FILE_SIZE",
    "get_all_supported_extensions",
]
