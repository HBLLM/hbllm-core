"""Memory Importance Scorer — Ebbinghaus-inspired forgetting curve.

Computes memory importance using:
    1. Base importance (set at creation — user-explicit > inferred)
    2. Reinforcement bonus (increases with each access)
    3. Decay factor (exponential: 0.5^(age_days / half_life))
    4. Emotional weight (emotionally significant memories decay slower)

Configurable half-lives per memory type:
    - Episodic: 30 days
    - Procedural: 90 days
    - Value/preference: 180 days

Memories below a configurable threshold are flagged for archival
(not deletion — they can be recovered).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Seconds per day
_DAY_S = 86400.0


@dataclass
class ImportanceConfig:
    """Configuration for importance scoring."""

    # Half-life in days (how long until importance decays to 50%)
    episodic_half_life: float = 30.0
    procedural_half_life: float = 90.0
    value_half_life: float = 180.0
    semantic_half_life: float = 60.0

    # How much each access reinforces the memory
    access_reinforcement: float = 0.1

    # Max reinforcement bonus (cap)
    max_reinforcement: float = 2.0

    # Emotional weight multiplier (emotionally tagged memories decay slower)
    emotional_weight_factor: float = 1.5

    # Below this threshold, memories are flagged for archival
    archival_threshold: float = 0.1

    # Minimum importance (never drops below this)
    floor: float = 0.01


@dataclass
class ScoredMemory:
    """A memory with its computed importance score."""

    memory_id: str
    content: str = ""
    raw_importance: float = 1.0  # Base importance at creation
    access_count: int = 0
    created_at: float = 0.0
    last_accessed: float = 0.0
    emotional_weight: float = 0.0  # 0.0 = neutral, 1.0 = highly emotional
    memory_type: str = "episodic"  # episodic, procedural, value, semantic
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed
    computed_importance: float = 0.0
    should_archive: bool = False


class ImportanceScorer:
    """Computes memory importance with Ebbinghaus-inspired decay.

    Usage::

        scorer = ImportanceScorer()

        # Score a memory
        scored = scorer.score(ScoredMemory(
            memory_id="abc",
            raw_importance=1.0,
            access_count=5,
            created_at=time.time() - 86400 * 15,  # 15 days old
            last_accessed=time.time() - 86400 * 2,  # 2 days since last access
            memory_type="episodic",
        ))
        print(scored.computed_importance)  # e.g., 0.72

        # Batch score and rank
        ranked = scorer.rank_memories(memories)
    """

    def __init__(self, config: ImportanceConfig | None = None) -> None:
        self.config = config or ImportanceConfig()

    def score(self, memory: ScoredMemory, now: float | None = None) -> ScoredMemory:
        """Compute the importance score for a single memory.

        The formula is:
            importance = (base + reinforcement) × decay × emotional_boost

        Where:
            base = raw_importance (set at creation)
            reinforcement = min(access_count × rate, max_reinforcement)
            decay = 0.5^(age_days / half_life)
            emotional_boost = 1.0 + emotional_weight × factor
        """
        now = now or time.time()

        # 1. Base importance
        base = max(0.0, memory.raw_importance)

        # 2. Reinforcement from access
        reinforcement = min(
            memory.access_count * self.config.access_reinforcement,
            self.config.max_reinforcement,
        )

        # 3. Decay based on age
        age_days = max(0, (now - memory.created_at) / _DAY_S)
        half_life = self._get_half_life(memory.memory_type)

        # Exponential decay: 0.5^(age / half_life)
        if half_life > 0:
            decay = math.pow(0.5, age_days / half_life)
        else:
            decay = 1.0  # No decay

        # 4. Recency boost (recently accessed memories get a small boost)
        if memory.last_accessed > 0:
            recency_days = max(0, (now - memory.last_accessed) / _DAY_S)
            recency_boost = math.exp(-recency_days / 7.0)  # Decays with ~1 week half-life
        else:
            recency_boost = 0.0

        # 5. Emotional weight (slows decay)
        emotional_boost = 1.0 + memory.emotional_weight * self.config.emotional_weight_factor

        # Final score
        importance = (base + reinforcement + recency_boost * 0.3) * decay * emotional_boost
        importance = max(self.config.floor, importance)

        # Update memory
        memory.computed_importance = round(importance, 4)
        memory.should_archive = importance < self.config.archival_threshold

        return memory

    def rank_memories(
        self,
        memories: list[ScoredMemory],
        now: float | None = None,
    ) -> list[ScoredMemory]:
        """Score and rank a list of memories by importance (highest first)."""
        now = now or time.time()
        scored = [self.score(m, now) for m in memories]
        scored.sort(key=lambda m: m.computed_importance, reverse=True)
        return scored

    def get_archivable(
        self,
        memories: list[ScoredMemory],
        now: float | None = None,
    ) -> list[ScoredMemory]:
        """Get memories that should be archived (below threshold)."""
        now = now or time.time()
        return [m for m in self.rank_memories(memories, now) if m.should_archive]

    def consolidate(
        self,
        memories: list[ScoredMemory],
        now: float | None = None,
    ) -> tuple[list[ScoredMemory], list[ScoredMemory]]:
        """Split memories into keep and archive lists.

        This is the main method called by SleepNode during idle periods.

        Returns:
            (keep, archive) — two lists of scored memories.
        """
        now = now or time.time()
        scored = self.rank_memories(memories, now)

        keep = [m for m in scored if not m.should_archive]
        archive = [m for m in scored if m.should_archive]

        if archive:
            logger.info(
                "Memory consolidation: keeping %d, archiving %d (threshold=%.2f)",
                len(keep),
                len(archive),
                self.config.archival_threshold,
            )

        return keep, archive

    def _get_half_life(self, memory_type: str) -> float:
        """Get the decay half-life for a memory type."""
        half_lives = {
            "episodic": self.config.episodic_half_life,
            "procedural": self.config.procedural_half_life,
            "value": self.config.value_half_life,
            "semantic": self.config.semantic_half_life,
        }
        return half_lives.get(memory_type, self.config.episodic_half_life)

    def stats(self) -> dict[str, Any]:
        """Scorer configuration summary."""
        return {
            "half_lives": {
                "episodic": self.config.episodic_half_life,
                "procedural": self.config.procedural_half_life,
                "value": self.config.value_half_life,
                "semantic": self.config.semantic_half_life,
            },
            "archival_threshold": self.config.archival_threshold,
            "access_reinforcement": self.config.access_reinforcement,
        }
