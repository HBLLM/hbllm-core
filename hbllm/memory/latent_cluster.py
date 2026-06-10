"""
Latent Memory Cluster Manager.

Handles online centroid-based vector clustering, cluster statistics,
stability-guarded split and merge consolidation maintenance, and
TF-IDF cluster auto-labeling.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class LatentClusterManager:
    """
    Manages latent memory clusters.

    Assigns documents to clusters based on vector embedding cosine similarity,
    tracks stats, handles merge/split maintenance with cluster maturity gates,
    and generates descriptive labels using TF-IDF.
    """

    def __init__(self, max_clusters: int = 32) -> None:
        self.max_clusters = max_clusters
        # cluster_id -> normalized centroid vector
        self.centroids: dict[int, np.ndarray] = {}
        # doc_id -> cluster_id
        self.cluster_assignments: dict[str, int] = {}
        # cluster_id -> stats dict
        self.cluster_stats: dict[int, dict[str, Any]] = {}
        # Counter for generating unique cluster IDs
        self._next_cluster_id = 0

    def get_cluster_label(self, cluster_id: int, documents: dict[str, dict[str, Any]]) -> str:
        """Generate a label for a cluster based on TF-IDF keywords of its documents."""
        doc_ids = [d_id for d_id, c_id in self.cluster_assignments.items() if c_id == cluster_id]
        if not doc_ids:
            return f"Cluster {cluster_id} (empty)"

        # 1. Tokenize all documents in all clusters to build corpus statistics
        all_words_per_doc: dict[str, list[str]] = {}
        for d_id in self.cluster_assignments.keys():
            doc = documents.get(d_id)
            if doc and "content" in doc:
                text = str(doc["content"]).lower()
                words = re.findall(r"\b[a-zA-Z]{3,15}\b", text)
                # Filter out extremely common stop words
                stop_words = {
                    "the",
                    "and",
                    "for",
                    "that",
                    "this",
                    "with",
                    "from",
                    "you",
                    "your",
                    "have",
                    "are",
                    "was",
                    "were",
                    "but",
                    "not",
                    "they",
                    "their",
                    "there",
                }
                words = [w for w in words if w not in stop_words]
                all_words_per_doc[d_id] = words

        # 2. Compute IDF across all active docs
        total_docs = len(all_words_per_doc)
        doc_freq: Counter[str] = Counter()
        for words in all_words_per_doc.values():
            for w in set(words):
                doc_freq[w] += 1

        idf: dict[str, float] = {}
        for w, df in doc_freq.items():
            idf[w] = np.log((total_docs + 1) / (df + 1)) + 1.0

        # 3. Compute TF-IDF specifically for this cluster's docs
        cluster_words: list[str] = []
        for d_id in doc_ids:
            if d_id in all_words_per_doc:
                cluster_words.extend(all_words_per_doc[d_id])

        if not cluster_words:
            return f"Cluster {cluster_id} (unlabeled)"

        tf = Counter(cluster_words)
        tf_idf: dict[str, float] = {}
        for w, count in tf.items():
            tf_idf[w] = count * idf.get(w, 1.0)

        # Get top 3 terms
        top_terms = [
            item[0] for item in sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:3]
        ]
        if top_terms:
            return f"Cluster {cluster_id} ({', '.join(top_terms)})"
        return f"Cluster {cluster_id}"

    def assign_to_cluster(self, doc_id: str, vector: np.ndarray, doc_content: str = "") -> int:
        """
        Online assignment of a document vector to a cluster.

        Assigns to the closest centroid if cosine similarity >= 0.70.
        Otherwise, spawns a new cluster unless max_clusters limit is reached.
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        # Ensure vector is L2-normalized
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            vector = np.zeros_like(vector)

        # Align shapes of vector and centroids if they don't match (for TF-IDF dynamic vocabulary growth)
        for c_id in list(self.centroids.keys()):
            centroid = self.centroids[c_id]
            if len(centroid) < len(vector):
                padded = np.zeros(len(vector))
                padded[: len(centroid)] = centroid
                self.centroids[c_id] = padded
            elif len(vector) < len(centroid):
                padded = np.zeros(len(centroid))
                padded[: len(vector)] = vector
                vector = padded

        if not self.centroids:
            cluster_id = self._create_cluster(vector)
            self.cluster_assignments[doc_id] = cluster_id
            self._update_stats(cluster_id, [vector])
            return cluster_id

        # Calculate cosine similarities to all existing centroids
        best_sim = -1.0
        best_cluster = -1
        for c_id, centroid in self.centroids.items():
            c_norm = np.linalg.norm(centroid)
            if c_norm == 0:
                continue
            sim = float(np.dot(vector, centroid) / (c_norm + 1e-9))
            if sim > best_sim:
                best_sim = sim
                best_cluster = c_id

        if best_sim >= 0.70:
            # Assign to existing
            self.cluster_assignments[doc_id] = best_cluster
            # Recalculate centroid based on all members in the cluster
            # Gather other members' vectors
            # Note: This method assumes we can fetch all vectors or update online.
            # We'll update the centroid online here, but in consolidation we recompute fully.
            current_centroid = self.centroids[best_cluster]
            current_size = self.cluster_stats.get(best_cluster, {}).get("size", 1)
            new_centroid = current_centroid * current_size + vector
            new_centroid_norm = np.linalg.norm(new_centroid)
            if new_centroid_norm > 0:
                self.centroids[best_cluster] = new_centroid / new_centroid_norm

            # Simple online stats update (proper variance calculation done during consolidation)
            stats = self.cluster_stats.setdefault(best_cluster, self._default_stats())
            stats["size"] += 1
            stats["last_used"] = time.time()
            return best_cluster
        else:
            # Spawn new cluster or fall back if max reached
            if len(self.centroids) >= self.max_clusters:
                self.cluster_assignments[doc_id] = best_cluster
                stats = self.cluster_stats.setdefault(best_cluster, self._default_stats())
                stats["size"] += 1
                stats["last_used"] = time.time()
                return best_cluster
            else:
                cluster_id = self._create_cluster(vector)
                self.cluster_assignments[doc_id] = cluster_id
                self._update_stats(cluster_id, [vector])
                return cluster_id

    def _create_cluster(self, vector: np.ndarray) -> int:
        cluster_id = self._next_cluster_id
        self._next_cluster_id += 1
        self.centroids[cluster_id] = vector.copy()
        self.cluster_stats[cluster_id] = self._default_stats()
        return cluster_id

    def _default_stats(self) -> dict[str, Any]:
        return {
            "size": 1,
            "variance": 0.0,
            "activation_count": 0,
            "positive_feedback_count": 0,
            "success_rate": 1.0,
            "age_cycles": 0,
            "last_used": time.time(),
        }

    def _update_stats(self, cluster_id: int, member_vectors: list[np.ndarray]) -> None:
        stats = self.cluster_stats.setdefault(cluster_id, self._default_stats())
        stats["size"] = len(member_vectors)
        centroid = self.centroids[cluster_id]
        if len(member_vectors) > 0:
            similarities = []
            for v in member_vectors:
                # Align shapes if mismatched
                if len(centroid) < len(v):
                    padded = np.zeros(len(v))
                    padded[: len(centroid)] = centroid
                    centroid = padded
                    self.centroids[cluster_id] = centroid
                elif len(v) < len(centroid):
                    padded = np.zeros(len(centroid))
                    padded[: len(v)] = v
                    v = padded
                v_norm = np.linalg.norm(v)
                c_norm = np.linalg.norm(centroid)
                if v_norm == 0 or c_norm == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(np.dot(v, centroid) / (v_norm * c_norm + 1e-9))
            # Variance defined as mean cosine distance (1.0 - similarity)
            stats["variance"] = float(np.mean([1.0 - s for s in similarities]))
        else:
            stats["variance"] = 0.0

        if stats["activation_count"] > 0:
            stats["success_rate"] = (
                float(stats["positive_feedback_count"]) / stats["activation_count"]
            )
        else:
            stats["success_rate"] = 1.0

    def maintain_clusters(
        self,
        doc_vectors: dict[str, np.ndarray],
        synaptic_weights: dict[str, dict[str, float]],
    ) -> None:
        """
        Consolidation maintenance:
        1. Recompute centroids, sizes, variances from current assignments.
        2. Merge centroids with cosine similarity > 0.85.
        3. Split clusters with variance > 0.25 under maturity conditions (size >= 8 and age >= 3).
        """
        if not self.centroids:
            return

        # 1. Clean assignments and recompute
        active_assignments = {
            d_id: c_id for d_id, c_id in self.cluster_assignments.items() if d_id in doc_vectors
        }
        self.cluster_assignments = active_assignments

        # Map cluster -> doc vectors
        cluster_docs = defaultdict(list)
        for d_id, c_id in self.cluster_assignments.items():
            cluster_docs[c_id].append(doc_vectors[d_id])

        # Remove empty clusters from tracking
        for c_id in list(self.centroids.keys()):
            if c_id not in cluster_docs:
                self.centroids.pop(c_id, None)
                self.cluster_stats.pop(c_id, None)

        # Recompute centroid and variance for all active clusters, increment age
        for c_id, vectors in cluster_docs.items():
            # Align shapes within vectors of this cluster
            max_len = max(len(v) for v in vectors)
            aligned_vectors = []
            for v in vectors:
                if len(v) < max_len:
                    padded = np.zeros(max_len)
                    padded[: len(v)] = v
                    aligned_vectors.append(padded)
                else:
                    aligned_vectors.append(v)
            vectors = aligned_vectors
            cluster_docs[c_id] = vectors

            mean_vector = np.mean(vectors, axis=0)
            norm = np.linalg.norm(mean_vector)
            if norm > 0:
                self.centroids[c_id] = mean_vector / norm
            else:
                self.centroids[c_id] = mean_vector

            self._update_stats(c_id, vectors)
            self.cluster_stats[c_id]["age_cycles"] += 1

        # 2. Merge highly similar clusters
        merged_any = True
        while merged_any and len(self.centroids) > 1:
            merged_any = False
            best_merge_sim = -1.0
            best_pair: tuple[int, int] | None = None

            c_ids = list(self.centroids.keys())
            for i in range(len(c_ids)):
                for j in range(i + 1, len(c_ids)):
                    c1, c2 = c_ids[i], c_ids[j]
                    centroid1 = self.centroids[c1]
                    centroid2 = self.centroids[c2]
                    # Align shapes of centroids
                    if len(centroid1) < len(centroid2):
                        padded = np.zeros(len(centroid2))
                        padded[: len(centroid1)] = centroid1
                        centroid1 = padded
                        self.centroids[c1] = centroid1
                    elif len(centroid2) < len(centroid1):
                        padded = np.zeros(len(centroid1))
                        padded[: len(centroid2)] = centroid2
                        centroid2 = padded
                        self.centroids[c2] = centroid2
                    sim = float(
                        np.dot(centroid1, centroid2)
                        / (np.linalg.norm(centroid1) * np.linalg.norm(centroid2) + 1e-9)
                    )
                    if sim > best_merge_sim:
                        best_merge_sim = sim
                        best_pair = (c1, c2)

            if best_pair and best_merge_sim > 0.85:
                c_keep, c_del = best_pair
                # Keep the one with larger size (or c_keep if sizes are equal)
                size_keep = self.cluster_stats[c_keep]["size"]
                size_del = self.cluster_stats[c_del]["size"]
                if size_del > size_keep:
                    c_keep, c_del = c_del, c_keep

                logger.info(
                    "Merging cluster %d into %d (similarity %.3f)", c_del, c_keep, best_merge_sim
                )

                # Reassign docs
                for d_id, c_id in list(self.cluster_assignments.items()):
                    if c_id == c_del:
                        self.cluster_assignments[d_id] = c_keep

                # Combine statistics
                stats_keep = self.cluster_stats[c_keep]
                stats_del = self.cluster_stats[c_del]

                total_activations = stats_keep["activation_count"] + stats_del["activation_count"]
                total_positive = (
                    stats_keep["positive_feedback_count"] + stats_del["positive_feedback_count"]
                )

                stats_keep["activation_count"] = total_activations
                stats_keep["positive_feedback_count"] = total_positive
                stats_keep["last_used"] = max(stats_keep["last_used"], stats_del["last_used"])
                stats_keep["age_cycles"] = 0  # Reset age to allow stabilization

                # Recompute centroid and stats
                vectors_keep = cluster_docs[c_keep] + cluster_docs[c_del]
                cluster_docs[c_keep] = vectors_keep
                cluster_docs.pop(c_del, None)

                mean_vector = np.mean(vectors_keep, axis=0)
                norm = np.linalg.norm(mean_vector)
                if norm > 0:
                    self.centroids[c_keep] = mean_vector / norm
                else:
                    self.centroids[c_keep] = mean_vector

                self._update_stats(c_keep, vectors_keep)

                # Remove c_del
                self.centroids.pop(c_del, None)
                self.cluster_stats.pop(c_del, None)

                # Blending Hebbian synaptic weights for the merged clusters
                self._blend_weights_on_merge(c_keep, c_del, size_keep, size_del, synaptic_weights)

                merged_any = True

        # 3. Split high-variance clusters under maturity check (size >= 8 and age >= 3)
        for c_id in list(self.centroids.keys()):
            stats = self.cluster_stats[c_id]
            # Split Stability Condition: min_cluster_age = 3 consolidations AND min_cluster_size = 8
            if stats["variance"] > 0.25 and stats["size"] >= 8 and stats["age_cycles"] >= 3:
                self._split_cluster(c_id, cluster_docs[c_id], doc_vectors, synaptic_weights)

    def _blend_weights_on_merge(
        self,
        c_keep: int,
        c_del: int,
        size_keep: int,
        size_del: int,
        synaptic_weights: dict[str, dict[str, float]],
    ) -> None:
        """Blend Hebbian synaptic weight rows and columns during a merge."""
        k_key = f"cluster_{c_keep}"
        d_key = f"cluster_{c_del}"

        # If they aren't in synaptic_weights yet, skip
        if k_key not in synaptic_weights and d_key not in synaptic_weights:
            return

        total_size = size_keep + size_del
        if total_size <= 0:
            return

        w_keep = synaptic_weights.setdefault(k_key, {k_key: 1.0})
        w_del = synaptic_weights.get(d_key, {d_key: 1.0})

        # Blend row
        all_keys = set(w_keep.keys()) | set(w_del.keys())
        blended_row = {}
        for other in all_keys:
            val_keep = w_keep.get(other, 1.0 if other == k_key else 0.0)
            val_del = w_del.get(other, 1.0 if other == d_key else 0.0)
            blended_row[other] = (val_keep * size_keep + val_del * size_del) / total_size

        synaptic_weights[k_key] = blended_row

        # Blend columns in other rows
        for other_row_key, other_row in synaptic_weights.items():
            if other_row_key == d_key:
                continue
            val_keep = other_row.get(k_key, 1.0 if other_row_key == k_key else 0.0)
            val_del = other_row.get(d_key, 1.0 if other_row_key == d_key else 0.0)
            other_row[k_key] = (val_keep * size_keep + val_del * size_del) / total_size
            other_row.pop(d_key, None)

        # Remove deleted cluster from synaptic matrix
        synaptic_weights.pop(d_key, None)

    def _split_cluster(
        self,
        c_id: int,
        vectors: list[np.ndarray],
        doc_vectors: dict[str, np.ndarray],
        synaptic_weights: dict[str, dict[str, float]],
    ) -> None:
        """Split a cluster into two using 2-means clustering."""
        if len(vectors) < 2:
            return

        # Find documents assigned to this cluster
        member_doc_ids = [
            d_id
            for d_id, assigned_c_id in self.cluster_assignments.items()
            if assigned_c_id == c_id
        ]
        if len(member_doc_ids) != len(vectors):
            return

        # 2-means initialization: Find the two vectors that are most dissimilar (cosine similarity)
        best_dissim = 2.0
        idx1, idx2 = 0, 1
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1, v2 = vectors[i], vectors[j]
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
                if sim < best_dissim:
                    best_dissim = sim
                    idx1, idx2 = i, j

        center1 = vectors[idx1].copy()
        center2 = vectors[idx2].copy()

        # Run 2-means iterations
        assignments = np.zeros(len(vectors), dtype=int)
        for _ in range(10):
            # Assignment step
            changed = False
            for idx, v in enumerate(vectors):
                sim1 = np.dot(v, center1) / (np.linalg.norm(v) * np.linalg.norm(center1) + 1e-9)
                sim2 = np.dot(v, center2) / (np.linalg.norm(v) * np.linalg.norm(center2) + 1e-9)
                new_assign = 0 if sim1 >= sim2 else 1
                if assignments[idx] != new_assign:
                    assignments[idx] = new_assign
                    changed = True

            if not changed:
                break

            # Update step
            group1 = [vectors[i] for i in range(len(vectors)) if assignments[i] == 0]
            group2 = [vectors[i] for i in range(len(vectors)) if assignments[i] == 1]

            if not group1 or not group2:
                # Fallback if one group is empty
                break

            mean1 = np.mean(group1, axis=0)
            norm1 = np.linalg.norm(mean1)
            center1 = mean1 / norm1 if norm1 > 0 else mean1

            mean2 = np.mean(group2, axis=0)
            norm2 = np.linalg.norm(mean2)
            center2 = mean2 / norm2 if norm2 > 0 else mean2

        # Final split assignments
        group1 = [vectors[i] for i in range(len(vectors)) if assignments[i] == 0]
        group2 = [vectors[i] for i in range(len(vectors)) if assignments[i] == 1]

        # Prevent splitting if one group is tiny
        if len(group1) < 2 or len(group2) < 2:
            return

        c_new = self._create_cluster(center2)
        logger.info(
            "Splitting cluster %d (size %d) into cluster %d (size %d) and cluster %d (size %d)",
            c_id,
            len(vectors),
            c_id,
            len(group1),
            c_new,
            len(group2),
        )

        # Reassign documents
        for idx, doc_id in enumerate(member_doc_ids):
            if assignments[idx] == 1:
                self.cluster_assignments[doc_id] = c_new

        # Update centroid for c_id (c_keep)
        self.centroids[c_id] = center1.copy()

        # Reset stats and recalculate
        self._update_stats(c_id, group1)
        self._update_stats(c_new, group2)
        self.cluster_stats[c_id]["age_cycles"] = 0
        self.cluster_stats[c_new]["age_cycles"] = 0

        # Duplicate row/col in synaptic_weights for the split cluster
        self._split_weights(c_id, c_new, len(group1), len(group2), synaptic_weights)

    def _split_weights(
        self,
        c_orig: int,
        c_new: int,
        size_orig: int,
        size_new: int,
        synaptic_weights: dict[str, dict[str, float]],
    ) -> None:
        """Split Hebbian synaptic weights for a split cluster."""
        orig_key = f"cluster_{c_orig}"
        new_key = f"cluster_{c_new}"

        if orig_key not in synaptic_weights:
            return

        orig_row = synaptic_weights[orig_key]

        # Create new row copying orig row
        new_row = orig_row.copy()
        new_row[new_key] = 1.0  # self-connection
        orig_row[new_key] = 0.5  # connection to the other split half
        new_row[orig_key] = 0.5

        synaptic_weights[new_key] = new_row

        # Update columns in other rows
        for other_key, other_row in synaptic_weights.items():
            if other_key in (orig_key, new_key):
                continue
            other_row[new_key] = other_row.get(orig_key, 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize latent cluster state to a JSON-compatible dictionary."""
        return {
            "centroids": {str(k): v.tolist() for k, v in self.centroids.items()},
            "cluster_assignments": self.cluster_assignments,
            "cluster_stats": self.cluster_stats,
            "next_cluster_id": self._next_cluster_id,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Deserialize latent cluster state from a dictionary."""
        self._next_cluster_id = data.get("next_cluster_id", 0)
        self.cluster_assignments = data.get("cluster_assignments", {})
        self.cluster_stats = data.get("cluster_stats", {})
        centroids_raw = data.get("centroids", {})
        self.centroids = {int(k): np.array(v) for k, v in centroids_raw.items()}
