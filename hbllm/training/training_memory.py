"""
Training Memory — loss-indexed document memory for curriculum learning.

Records per-document training metrics during pre-training, enabling:
- Identify hard documents (high loss) for focused re-training
- Identify mastered domains (low loss) to skip or reduce sampling
- Provide replay candidates for continual learning
- Track learning curves per domain/topic
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """A training document's learning record."""

    doc_hash: int
    loss: float
    step: int
    domain: str = "general"
    concepts: list[str] = field(default_factory=list)
    text_preview: str = ""


@dataclass
class DomainStats:
    """Aggregated statistics for a domain/topic."""

    domain: str
    total_docs: int = 0
    total_loss: float = 0.0
    min_loss: float = float("inf")
    max_loss: float = 0.0
    first_seen_step: int = 0
    last_seen_step: int = 0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.total_docs)

    @property
    def mastery_score(self) -> float:
        """0-1 score, higher = better mastered. Based on average loss."""
        # Typical loss ranges: 10+ (random) → 3-4 (decent) → 1-2 (good)
        return max(0.0, min(1.0, 1.0 - (self.avg_loss - 1.0) / 9.0))


# Domain detection keywords
DOMAIN_KEYWORDS = {
    "code": [
        "def ",
        "class ",
        "function ",
        "import ",
        "const ",
        "var ",
        "return ",
        "if (",
        "for (",
        "while ",
        "print(",
        "console.log",
        "async ",
        "await ",
    ],
    "math": [
        "equation",
        "theorem",
        "proof",
        "integral",
        "derivative",
        "matrix",
        "probability",
        "statistical",
        "variance",
        "polynomial",
        "algebra",
    ],
    "science": [
        "experiment",
        "hypothesis",
        "molecule",
        "electron",
        "genome",
        "neuron",
        "quantum",
        "relativity",
        "thermodynamic",
        "ecosystem",
    ],
    "reasoning": [
        "therefore",
        "consequently",
        "because",
        "although",
        "however",
        "furthermore",
        "in conclusion",
        "on the other hand",
        "implies",
    ],
    "factual": [
        "born in",
        "founded in",
        "located in",
        "capital of",
        "invented",
        "discovered",
        "according to",
        "published in",
        "established",
    ],
    "creative": [
        "once upon",
        "she said",
        "he whispered",
        "the sky",
        "beautiful",
        "imagine",
        "story",
        "poem",
        "character",
        "narrative",
    ],
}


def detect_domain(text: str) -> str:
    """Detect the primary domain/topic of a text."""
    text_lower = text.lower()
    scores: dict[str, int] = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[domain] = score

    if not scores:
        return "general"

    return max(scores, key=lambda k: scores[k])


class TrainingMemory:
    """
    Training-time memory that indexes documents by loss and domain.

    Enables curriculum learning and continual learning strategies:
    - Hard documents (high loss) can be replayed
    - Mastered domains can have reduced sampling
    - New unseen domains get priority
    """

    def __init__(self, max_records: int = 50000):
        self.records: list[DocumentRecord] = []
        self.domain_stats: dict[str, DomainStats] = {}
        self._max_records = max_records
        self._step_losses: list[tuple[int, float]] = []  # (step, avg_loss)

    def record(
        self,
        text: str,
        loss: float,
        step: int,
        concepts: list[str] | None = None,
    ) -> DocumentRecord:
        """Record a training document and its loss."""
        domain = detect_domain(text)
        doc_hash = hash(text[:500])

        rec = DocumentRecord(
            doc_hash=doc_hash,
            loss=loss,
            step=step,
            domain=domain,
            concepts=concepts or [],
            text_preview=text[:100],
        )

        # Store record (with eviction if full)
        if len(self.records) >= self._max_records:
            # Keep high-loss records (more interesting) and evict low-loss ones
            self.records.sort(key=lambda r: r.loss)
            self.records = self.records[len(self.records) // 4 :]  # Remove bottom 25%

        self.records.append(rec)

        # Update domain statistics
        if domain not in self.domain_stats:
            self.domain_stats[domain] = DomainStats(
                domain=domain,
                first_seen_step=step,
            )
        ds = self.domain_stats[domain]
        ds.total_docs += 1
        ds.total_loss += loss
        ds.min_loss = min(ds.min_loss, loss)
        ds.max_loss = max(ds.max_loss, loss)
        ds.last_seen_step = step

        return rec

    def record_batch(
        self,
        texts: list[str],
        loss: float,
        step: int,
        concepts_list: list[list[str]] | None = None,
    ) -> None:
        """Record a batch of documents with the same batch loss."""
        for i, text in enumerate(texts):
            concepts = concepts_list[i] if concepts_list else None
            self.record(text, loss, step, concepts)

    def get_hard_documents(self, top_k: int = 100) -> list[DocumentRecord]:
        """Get the hardest (highest loss) documents for replay."""
        sorted_recs = sorted(self.records, key=lambda r: r.loss, reverse=True)
        return sorted_recs[:top_k]

    def get_mastered_domains(self, threshold: float = 0.7) -> list[str]:
        """Get domains with mastery score above threshold."""
        return [
            d.domain
            for d in self.domain_stats.values()
            if d.mastery_score >= threshold and d.total_docs >= 10
        ]

    def get_weak_domains(self, threshold: float = 0.3) -> list[str]:
        """Get domains with mastery score below threshold."""
        return [
            d.domain
            for d in self.domain_stats.values()
            if d.mastery_score < threshold and d.total_docs >= 10
        ]

    def record_step_loss(self, step: int, avg_loss: float) -> None:
        """Track overall loss trajectory."""
        self._step_losses.append((step, avg_loss))

    def stats(self) -> dict[str, Any]:
        """Summary statistics."""
        return {
            "total_records": len(self.records),
            "domains": {
                name: {
                    "docs": ds.total_docs,
                    "avg_loss": round(ds.avg_loss, 4),
                    "mastery": round(ds.mastery_score, 3),
                    "min_loss": round(ds.min_loss, 4),
                    "max_loss": round(ds.max_loss, 4),
                }
                for name, ds in sorted(
                    self.domain_stats.items(),
                    key=lambda x: x[1].total_docs,
                    reverse=True,
                )
            },
            "mastered_domains": self.get_mastered_domains(),
            "weak_domains": self.get_weak_domains(),
        }

    def save(self, path: str | Path) -> None:
        """Save training memory to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "saved_at": time.time(),
                "total_records": len(self.records),
                "total_domains": len(self.domain_stats),
            },
            "domain_stats": {
                name: {
                    "total_docs": ds.total_docs,
                    "avg_loss": round(ds.avg_loss, 4),
                    "min_loss": round(ds.min_loss, 4) if ds.min_loss != float("inf") else None,
                    "max_loss": round(ds.max_loss, 4),
                    "mastery_score": round(ds.mastery_score, 3),
                    "first_seen_step": ds.first_seen_step,
                    "last_seen_step": ds.last_seen_step,
                }
                for name, ds in self.domain_stats.items()
            },
            "hard_documents": [
                {
                    "preview": r.text_preview,
                    "loss": round(r.loss, 4),
                    "domain": r.domain,
                    "step": r.step,
                    "concepts": r.concepts[:5],
                }
                for r in self.get_hard_documents(50)
            ],
            "loss_trajectory": [{"step": s, "loss": round(l, 4)} for s, l in self._step_losses],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            "Training memory saved: %d records, %d domains → %s",
            len(self.records),
            len(self.domain_stats),
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> TrainingMemory:
        """Load training memory from JSON."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        mem = cls()
        for name, ds_data in data.get("domain_stats", {}).items():
            mem.domain_stats[name] = DomainStats(
                domain=name,
                total_docs=ds_data["total_docs"],
                total_loss=ds_data["avg_loss"] * ds_data["total_docs"],
                min_loss=ds_data.get("min_loss") or float("inf"),
                max_loss=ds_data.get("max_loss", 0),
                first_seen_step=ds_data.get("first_seen_step", 0),
                last_seen_step=ds_data.get("last_seen_step", 0),
            )

        mem._step_losses = [
            (entry["step"], entry["loss"]) for entry in data.get("loss_trajectory", [])
        ]

        logger.info("Loaded training memory: %d domains", len(mem.domain_stats))
        return mem
