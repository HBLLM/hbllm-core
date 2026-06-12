"""
Distinct Causal Graph Memory System.

Physically separates causal graphs from semantic memory vector space to prevent
contamination, allowing directed causal relations to be stored and queried.
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.causality.causal_graph import CausalGraph, CausalLink

logger = logging.getLogger(__name__)


class CausalMemorySystem:
    """
    Interface for logging and querying directed causal relations.
    """

    def __init__(self, data_dir: str = "data") -> None:
        self.graph = CausalGraph(data_dir=data_dir)

    def log_causal_relation(
        self,
        cause: str,
        effect: str,
        probability: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> CausalLink:
        """
        Log a directed causal relation: cause -> effect.
        """
        link = CausalLink(
            source_id=cause,
            target_id=effect,
            probability=probability,
            metadata=metadata or {},
        )
        self.graph._insert(link)
        logger.info(
            "[CausalMemory] Logged causal relation: '%s' -> caused -> '%s' (p=%.2f)",
            cause,
            effect,
            probability,
        )
        return link

    def get_effects_for_cause(self, cause: str) -> list[CausalLink]:
        """Get all effects caused by a specific cause."""
        return self.graph.get_effects(cause)

    def get_causes_for_effect(self, effect: str) -> list[CausalLink]:
        """Get all causes for a specific effect."""
        return self.graph.get_causes(effect)
