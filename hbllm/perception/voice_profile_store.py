"""
Voice Profile Store — Biometric voice embedding storage for speaker identification.

Stores per-tenant speaker embeddings (256-dim Resemblyzer GE2E vectors) in SQLite
with optional PostgreSQL backend. Embeddings are not reversible to audio.

Schema:
    voice_profiles (
        tenant_id, speaker_id, speaker_name,
        embedding (BLOB), enrollment_samples, similarity_threshold,
        created_at, updated_at
    )
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import struct
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default cosine similarity threshold for speaker identification
DEFAULT_SIMILARITY_THRESHOLD = 0.75


def _embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding.astype(np.float32))


def _bytes_to_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes back to a float32 numpy array."""
    count = len(data) // 4  # 4 bytes per float32
    return np.array(struct.unpack(f"{count}f", data), dtype=np.float32)


class VoiceProfileStore:
    """
    SQLite-backed voice profile storage.

    Each tenant has isolated voice profiles. Embeddings are 256-dim float32
    vectors extracted by Resemblyzer's GE2E encoder.

    Usage:
        store = VoiceProfileStore("./data/voice_profiles.db")
        store.save_profile("tenant1", "dumith", "Dumith", embedding, samples=5)
        speaker_id, confidence = store.identify("tenant1", query_embedding)
    """

    def __init__(self, path: str = "./data/voice_profiles.db") -> None:
        self.path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.path, timeout=5.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_tables()

    def _init_tables(self) -> None:
        """Create voice profile tables."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS voice_profiles (
                tenant_id TEXT NOT NULL,
                speaker_id TEXT NOT NULL,
                speaker_name TEXT NOT NULL DEFAULT '',
                embedding BLOB NOT NULL,
                enrollment_samples INTEGER DEFAULT 1,
                similarity_threshold REAL DEFAULT 0.75,
                metadata TEXT DEFAULT '{}',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (tenant_id, speaker_id)
            );

            CREATE INDEX IF NOT EXISTS idx_voice_profiles_tenant
                ON voice_profiles(tenant_id);
        """)
        self._conn.commit()

    def save_profile(
        self,
        tenant_id: str,
        speaker_id: str,
        speaker_name: str,
        embedding: np.ndarray,
        enrollment_samples: int = 1,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save or update a voice profile."""
        now = time.time()
        self._conn.execute(
            """INSERT INTO voice_profiles
               (tenant_id, speaker_id, speaker_name, embedding,
                enrollment_samples, similarity_threshold, metadata,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(tenant_id, speaker_id) DO UPDATE SET
                   speaker_name = excluded.speaker_name,
                   embedding = excluded.embedding,
                   enrollment_samples = excluded.enrollment_samples,
                   similarity_threshold = excluded.similarity_threshold,
                   metadata = excluded.metadata,
                   updated_at = excluded.updated_at
            """,
            (
                tenant_id,
                speaker_id,
                speaker_name,
                _embedding_to_bytes(embedding),
                enrollment_samples,
                similarity_threshold,
                json.dumps(metadata or {}),
                now,
                now,
            ),
        )
        self._conn.commit()
        logger.info(
            "Saved voice profile: tenant=%s speaker=%s name='%s' samples=%d",
            tenant_id,
            speaker_id,
            speaker_name,
            enrollment_samples,
        )

    def get_profile(self, tenant_id: str, speaker_id: str) -> dict[str, Any] | None:
        """Get a single voice profile."""
        row = self._conn.execute(
            "SELECT * FROM voice_profiles WHERE tenant_id = ? AND speaker_id = ?",
            (tenant_id, speaker_id),
        ).fetchone()
        if row:
            return self._row_to_dict(row)
        return None

    def list_profiles(self, tenant_id: str) -> list[dict[str, Any]]:
        """List all voice profiles for a tenant."""
        rows = self._conn.execute(
            "SELECT * FROM voice_profiles WHERE tenant_id = ? ORDER BY speaker_name",
            (tenant_id,),
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete_profile(self, tenant_id: str, speaker_id: str) -> bool:
        """Delete a voice profile. Returns True if deleted."""
        cursor = self._conn.execute(
            "DELETE FROM voice_profiles WHERE tenant_id = ? AND speaker_id = ?",
            (tenant_id, speaker_id),
        )
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted voice profile: tenant=%s speaker=%s", tenant_id, speaker_id)
        return deleted

    def identify(
        self,
        tenant_id: str,
        query_embedding: np.ndarray,
        threshold: float | None = None,
    ) -> tuple[str, str, float]:
        """
        Identify a speaker from their voice embedding.

        Compares the query embedding against all enrolled profiles for the tenant
        using cosine similarity.

        Args:
            tenant_id: Tenant to search within
            query_embedding: 256-dim float32 voice embedding
            threshold: Override minimum similarity (default: per-profile or 0.75)

        Returns:
            (speaker_id, speaker_name, confidence) — ("unknown", "", 0.0) if no match
        """
        rows = self._conn.execute(
            "SELECT speaker_id, speaker_name, embedding, similarity_threshold "
            "FROM voice_profiles WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchall()

        if not rows:
            return ("unknown", "", 0.0)

        best_id = "unknown"
        best_name = ""
        best_score = 0.0

        # Normalize query embedding
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        for row in rows:
            stored_emb = _bytes_to_embedding(row["embedding"])
            stored_norm = stored_emb / (np.linalg.norm(stored_emb) + 1e-8)

            # Cosine similarity
            similarity = float(np.dot(query_norm, stored_norm))
            profile_threshold = threshold or row["similarity_threshold"]

            if similarity > best_score and similarity >= profile_threshold:
                best_score = similarity
                best_id = row["speaker_id"]
                best_name = row["speaker_name"]

        return (best_id, best_name, best_score)

    def update_embedding(
        self,
        tenant_id: str,
        speaker_id: str,
        new_embedding: np.ndarray,
        weight: float = 0.1,
    ) -> bool:
        """
        Incrementally update a speaker's embedding with a new sample.

        Uses exponential moving average: emb = (1-w) * old + w * new
        This allows the profile to adapt to voice changes over time.

        Args:
            tenant_id: Tenant ID
            speaker_id: Speaker ID
            new_embedding: New 256-dim embedding
            weight: Blending weight for new sample (0.0-1.0)

        Returns:
            True if profile was updated, False if not found.
        """
        row = self._conn.execute(
            "SELECT embedding, enrollment_samples FROM voice_profiles "
            "WHERE tenant_id = ? AND speaker_id = ?",
            (tenant_id, speaker_id),
        ).fetchone()

        if not row:
            return False

        old_emb = _bytes_to_embedding(row["embedding"])
        samples = row["enrollment_samples"]

        # EMA blend
        blended = (1.0 - weight) * old_emb + weight * new_embedding
        # Re-normalize
        blended = blended / (np.linalg.norm(blended) + 1e-8)

        self._conn.execute(
            "UPDATE voice_profiles SET embedding = ?, enrollment_samples = ?, updated_at = ? "
            "WHERE tenant_id = ? AND speaker_id = ?",
            (
                _embedding_to_bytes(blended),
                samples + 1,
                time.time(),
                tenant_id,
                speaker_id,
            ),
        )
        self._conn.commit()
        logger.debug(
            "Updated voice profile: tenant=%s speaker=%s (samples=%d)",
            tenant_id,
            speaker_id,
            samples + 1,
        )
        return True

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a dict (without raw embedding bytes)."""
        return {
            "tenant_id": row["tenant_id"],
            "speaker_id": row["speaker_id"],
            "speaker_name": row["speaker_name"],
            "enrollment_samples": row["enrollment_samples"],
            "similarity_threshold": row["similarity_threshold"],
            "metadata": json.loads(row["metadata"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
