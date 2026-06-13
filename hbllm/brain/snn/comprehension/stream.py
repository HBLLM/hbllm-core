"""
Comprehension Stream — Full pipeline from input to structured understanding.

Orchestrates the layered comprehension process:
    Layer 0: LexicalBuffer absorbs subword noise
    Layer 1: LexicalSignals computes cheap features (0.01ms/word)
    Layer 2: ComprehensionEnsemble detects concept boundaries via SNN
    Layer 3: On spike — ONNX embedding + domain classification + memory (4ms/concept)
    Layer 4: Aggregation into UnderstandingState

Cost model:
    Per word:    lexical signals (0.01ms) + SNN step (0.05ms) = 0.06ms
    Per concept: ONNX embedding (2ms) + memory search (2ms) = 4ms
    20-word input, 3 concepts: 1.2ms + 12ms = 13.2ms
    Previous design (all embedding): 2ms × 20 = 40ms → 3× faster
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Awaitable

import numpy as np

from hbllm.brain.snn.comprehension.ensemble import ComprehensionEnsemble
from hbllm.brain.snn.comprehension.lexical import LexicalBuffer, LexicalSignals
from hbllm.brain.snn.comprehension.models import (
    ActivatedMemory,
    ComprehensionUnit,
    UnderstandingState,
)

logger = logging.getLogger(__name__)

# Type alias for the memory search function injected by factory
MemorySearchFn = Callable[[str], Awaitable[list[ActivatedMemory]]]


class ComprehensionStream:
    """Full comprehension pipeline.

    Processes input text into a structured UnderstandingState by:
    1. Computing cheap lexical signals per word (~0.01ms)
    2. Feeding signals through a 5-channel SNN ensemble (~0.05ms)
    3. On clause spike: computing ONNX embedding + memory retrieval (~4ms)
    4. Aggregating all concepts into a unified understanding

    The expensive operations (embedding + memory) fire only on SNN spikes,
    making cost proportional to *concepts* rather than *tokens*.
    """

    def __init__(
        self,
        ensemble: ComprehensionEnsemble,
        lexical_buffer: LexicalBuffer,
        encoder: Callable[[str], Any],
        domain_centroids: dict[str, Any],
        memory_search_fn: MemorySearchFn | None = None,
    ) -> None:
        """Initialize the comprehension stream.

        Args:
            ensemble: The 5-channel SNN neuron ensemble.
            lexical_buffer: Subword noise absorber.
            encoder: Callable that encodes text to a numpy embedding vector.
                     Typically RouterNode._encode_text.
            domain_centroids: Dict mapping domain names to centroid vectors.
                              Shared reference from RouterNode.
            memory_search_fn: Async callable (text) -> list[ActivatedMemory].
                              If None, memory retrieval is skipped.
        """
        self.ensemble = ensemble
        self.lexical_buffer = lexical_buffer
        self.encoder = encoder
        self.domain_centroids = domain_centroids
        self.memory_search_fn = memory_search_fn

        # Per-query state
        self._word_buffer: list[str] = []
        self._prev_concept_words: list[str] | None = None
        self._concepts: list[ComprehensionUnit] = []

    async def comprehend(self, input_text: str) -> UnderstandingState:
        """Process input into structured understanding.

        This is the main entry point.  For a 20-word input with 3 concept
        boundaries, total cost is ~13ms (vs ~40ms with per-token embedding).

        Args:
            input_text: The raw user query text.

        Returns:
            An UnderstandingState with concepts, domain activations,
            memories, and salience map.
        """
        # Reset state for new query
        self._word_buffer = []
        self._concepts = []
        # Keep prev_concept_words across calls for inter-concept novelty

        # Tokenize at word level
        words = input_text.split()

        for word in words:
            timestamp = time.time()

            # Step 1: Compute cheap lexical signals (~0.01ms)
            signals = LexicalSignals.compute(
                word=word,
                buffer=self._word_buffer,
                prev_concept_words=self._prev_concept_words,
            )

            # Step 2: Feed signals to ensemble (~0.05ms for 5 neurons)
            fired_channels = self.ensemble.step(signals, timestamp)

            self._word_buffer.append(word)

            # Step 3: Check if clause neuron fired (primary boundary)
            clause_fired = any(ch == "clause" for ch, _ in fired_channels)

            if clause_fired:
                # NOW compute the expensive stuff — only on spike
                concept_text = " ".join(self._word_buffer).strip()

                # Collect metadata from other channels that fired
                metadata_channels = {
                    ch: spike.strength
                    for ch, spike in fired_channels
                    if ch != "clause"
                }

                unit = await self._process_concept(
                    concept_text,
                    metadata_channels,
                    timestamp,
                )
                self._concepts.append(unit)

                # Reset buffer
                self._prev_concept_words = list(self._word_buffer)
                self._word_buffer = []

        # Flush remaining buffer
        if self._word_buffer:
            concept_text = " ".join(self._word_buffer).strip()
            if concept_text:
                unit = await self._process_concept(
                    concept_text, {}, time.time()
                )
                self._concepts.append(unit)

        return self._build_understanding()

    async def _process_concept(
        self,
        text: str,
        channel_metadata: dict[str, float],
        timestamp: float,
    ) -> ComprehensionUnit:
        """Process a completed concept.

        THIS is where embeddings happen — called only on spike, not per token.
        """
        # Expensive: ONNX embedding (~2ms) — but only per concept
        try:
            embedding = self.encoder(text)
        except Exception:
            logger.warning("Embedding failed for concept: '%s...'", text[:30])
            embedding = np.zeros(384)

        # Domain classification against centroids (~0.1ms)
        domain_scores: dict[str, float] = {}
        for domain, centroid in self.domain_centroids.items():
            try:
                emb_norm = float(np.linalg.norm(embedding))
                cen_norm = float(np.linalg.norm(centroid))
                if emb_norm > 0 and cen_norm > 0:
                    sim = float(
                        np.dot(embedding, centroid) / (emb_norm * cen_norm)
                    )
                    if sim > 0.3:
                        domain_scores[domain] = sim
            except (ValueError, TypeError):
                continue

        # Targeted memory retrieval (~2ms)
        memories: list[ActivatedMemory] = []
        if self.memory_search_fn is not None:
            try:
                memories = await self.memory_search_fn(text)
            except Exception:
                logger.debug(
                    "Memory search failed for concept: '%s...'", text[:30]
                )

        # Compute salience from channel metadata
        # Constraint and surprise channels boost salience
        base_salience = 1.0
        if "constraint" in channel_metadata:
            base_salience += channel_metadata["constraint"] * 0.5
        if "surprise" in channel_metadata:
            base_salience += channel_metadata["surprise"] * 0.3

        return ComprehensionUnit(
            text=text,
            embedding=embedding,
            activated_memories=memories,
            domain_activation=domain_scores,
            salience=base_salience,
            channel_metadata=channel_metadata,
            timestamp=timestamp,
        )

    def _build_understanding(self) -> UnderstandingState:
        """Aggregate concepts into final understanding."""
        state = UnderstandingState(
            concepts=list(self._concepts),
            domain_activations={},
            all_memories=[],
            salience_map=[c.salience for c in self._concepts],
        )

        # Accumulate domain activations (salience-weighted)
        for concept in self._concepts:
            for domain, score in concept.domain_activation.items():
                weighted = score * concept.salience
                current = state.domain_activations.get(domain, 0.0)
                state.domain_activations[domain] = current + weighted

        # Deduplicate memories across concepts
        seen: set[str] = set()
        for concept in self._concepts:
            for m in concept.activated_memories:
                if m.id not in seen:
                    seen.add(m.id)
                    state.all_memories.append(m)

        # Reset per-query state
        self._concepts = []
        self._word_buffer = []

        return state

    def reset(self) -> None:
        """Reset all internal state for a fresh query."""
        self._word_buffer = []
        self._prev_concept_words = None
        self._concepts = []
        self.ensemble.reset()
        self.lexical_buffer.reset()
