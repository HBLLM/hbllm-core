"""
Unit and integration tests for Latent Memory Clusters and Self-Organizing Associative Retrieval.
"""

from __future__ import annotations

import shutil
import tempfile

import numpy as np

from hbllm.memory.latent_cluster import LatentClusterManager
from hbllm.memory.priming import WorkingMemoryPrimer
from hbllm.memory.semantic import SemanticMemory


class TestLatentClusterManager:
    """Verifies LatentClusterManager online clustering and maintenance mechanics."""

    def test_online_clustering_and_max_cap(self):
        manager = LatentClusterManager(max_clusters=3)

        # 1. Create first document/cluster
        v0 = np.array([1.0, 0.0, 0.0])
        c0 = manager.assign_to_cluster("doc0", v0, "quantum gravity physics")
        assert c0 == 0
        assert manager.cluster_stats[0]["size"] == 1
        assert np.array_equal(manager.centroids[0], v0)

        # 2. Add similar document (should join cluster 0)
        v1 = np.array([0.9, 0.1, 0.0])  # cos similarity ~0.99
        c1 = manager.assign_to_cluster("doc1", v1, "quantum mechanics string theory")
        assert c1 == 0
        assert manager.cluster_stats[0]["size"] == 2

        # 3. Add dissimilar document (should spawn cluster 1)
        v2 = np.array([0.0, 1.0, 0.0])  # cos similarity 0.0
        c2 = manager.assign_to_cluster("doc2", v2, "laravel routing controllers")
        assert c2 == 1

        # 4. Add third dissimilar document (should spawn cluster 2)
        v3 = np.array([0.0, 0.0, 1.0])
        c3 = manager.assign_to_cluster("doc3", v3, "personal dog address")
        assert c3 == 2

        # 5. Add fourth dissimilar document (should hit max_clusters cap of 3 and join closest)
        v4 = np.array([0.1, -0.1, 0.9])
        c4 = manager.assign_to_cluster("doc4", v4, "budget finance taxes invoice")
        # should join cluster 2 as it's closest to [0, 0, 1]
        assert c4 == 2

    def test_get_cluster_label(self):
        manager = LatentClusterManager()

        # Populate documents
        docs = {
            "d1": {"content": "quantum physics relativity gravity"},
            "d2": {"content": "quantum relativity space mechanics"},
            "d3": {"content": "laravel PHP eloquent routing controller"},
        }

        manager.assign_to_cluster("d1", np.array([1.0, 0.0]), docs["d1"]["content"])
        manager.assign_to_cluster("d2", np.array([0.9, 0.1]), docs["d2"]["content"])
        manager.assign_to_cluster("d3", np.array([0.0, 1.0]), docs["d3"]["content"])

        # Run maintain_clusters to settle assignments and update sizes
        vectors = {
            "d1": np.array([1.0, 0.0]),
            "d2": np.array([0.9, 0.1]),
            "d3": np.array([0.0, 1.0]),
        }
        synWeights = {}
        manager.maintain_clusters(vectors, synWeights)

        label0 = manager.get_cluster_label(0, docs)
        label1 = manager.get_cluster_label(1, docs)

        assert "quantum" in label0 or "relativity" in label0
        assert "laravel" in label1 or "php" in label1

    def test_merge_maintenance(self):
        manager = LatentClusterManager()
        vectors = {
            "d1": np.array([1.0, 0.0]),
            "d2": np.array([0.99, 0.01]),
        }
        manager.assign_to_cluster("d1", vectors["d1"])
        manager.assign_to_cluster("d2", vectors["d2"])

        # Force them into different clusters initially
        manager.cluster_assignments["d1"] = 0
        manager.cluster_assignments["d2"] = 1

        # Set up synaptic weights matrix
        synWeights = {
            "cluster_0": {"cluster_0": 1.0, "cluster_1": 0.1},
            "cluster_1": {"cluster_0": 0.1, "cluster_1": 1.0},
        }

        # Run maintain clusters - highly similar centroids should trigger merge
        manager.maintain_clusters(vectors, synWeights)
        assert len(manager.centroids) == 1
        assert 0 in manager.centroids  # Cluster 1 merged into 0
        assert "cluster_1" not in synWeights
        assert "cluster_0" in synWeights

    def test_split_maintenance_with_maturity_gate(self):
        manager = LatentClusterManager()

        # Create a cluster with 8 documents that are split into two groups
        vectors = {}
        synWeights = {"cluster_0": {"cluster_0": 1.0}}

        # Group A
        for i in range(4):
            doc_id = f"docA_{i}"
            v = np.array([1.0, 0.01 * i])
            vectors[doc_id] = v
            manager.assign_to_cluster(doc_id, v)

        # Group B
        for i in range(4):
            doc_id = f"docB_{i}"
            v = np.array([0.0, 1.0 + 0.01 * i])
            vectors[doc_id] = v
            manager.assign_to_cluster(doc_id, v)

        # Force all documents to belong to cluster 0 initially
        for doc_id in vectors.keys():
            manager.cluster_assignments[doc_id] = 0

        # Run maintain_clusters initially. Variance will be high (~0.5), size is 8.
        # But age_cycles is only 1 after the first consolidation. It should NOT split yet.
        manager.maintain_clusters(vectors, synWeights)
        assert len(manager.centroids) == 1
        assert manager.cluster_stats[0]["size"] == 8
        assert manager.cluster_stats[0]["variance"] > 0.25
        assert manager.cluster_stats[0]["age_cycles"] == 1

        # Run a second consolidation cycle (age becomes 2). Still should NOT split.
        manager.maintain_clusters(vectors, synWeights)
        assert len(manager.centroids) == 1
        assert manager.cluster_stats[0]["age_cycles"] == 2

        # Run a third consolidation cycle (age becomes 3). NOW it meets maturity and splits!
        manager.maintain_clusters(vectors, synWeights)
        assert len(manager.centroids) == 2
        assert 0 in manager.centroids
        other_key = [k for k in manager.centroids.keys() if k != 0][0]
        assert manager.cluster_stats[0]["size"] == 4
        assert manager.cluster_stats[other_key]["size"] == 4
        assert manager.cluster_stats[0]["age_cycles"] == 0
        assert manager.cluster_stats[other_key]["age_cycles"] == 0


class TestSNNVectorProjection:
    """Verifies dense vector-to-centroid projection SNN stimulation."""

    def test_stimulate_by_vector_quadratic(self):
        primer = WorkingMemoryPrimer()

        # Setup mock cluster centroids
        centroids = {
            0: np.array([1.0, 0.0, 0.0]),  # Laravel/Coding
            1: np.array([0.0, 1.0, 0.0]),  # Physics
        }

        # Query vector close to Laravel
        query_vec = np.array([0.9, 0.05, 0.0])  # Similarity to Laravel centroid ~0.99
        primer.stimulate_by_vector(query_vec, centroids)

        boosts = primer.get_boosts()

        # similarity = 0.99
        # expected stimulus = (0.99 - 0.35)^2 * 3.0 = 0.64^2 * 3.0 = 0.4096 * 3.0 = 1.2288
        # Spiking accumulator will cap potential or fire if threshold met
        assert boosts["cluster_0"] > 0.5
        assert "cluster_1" not in boosts or boosts["cluster_1"] == 0.0


class TestSemanticMemoryLifecycle:
    """Verifies that SemanticMemory integrates clustering, feedback, and persistence."""

    def test_e2e_cluster_storage_and_feedback(self):
        mem = SemanticMemory()

        # Store some documents
        d1 = mem.store("quantum loop gravity physics", metadata={"domain": "physics"})
        mem.store("einstein general relativity space", metadata={"domain": "physics"})
        d3 = mem.store("laravel routing controller php", metadata={"domain": "coding"})

        # Check that they were assigned to clusters
        doc1 = mem.documents[d1]
        doc3 = mem.documents[d3]

        assert doc1["metadata"]["domain"] == "physics"
        assert doc3["metadata"]["domain"] == "coding"

        c1 = mem.cluster_manager.cluster_assignments[d1]
        c3 = mem.cluster_manager.cluster_assignments[d3]
        assert c1 is not None
        assert c3 is not None
        assert c1 != c3

        # Run search with primer
        primer = WorkingMemoryPrimer()
        results = mem.search("laravel controller", top_k=3, primer=primer)
        assert results[0]["id"] == d3

        # Check that activation_count was incremented for doc3's cluster
        c_id = mem.cluster_manager.cluster_assignments[d3]
        assert mem.cluster_manager.cluster_stats[c_id]["activation_count"] == 1

        # Provide positive feedback
        mem.feedback(d3, useful=True)
        assert mem.cluster_manager.cluster_stats[c_id]["positive_feedback_count"] == 1
        assert mem.cluster_manager.cluster_stats[c_id]["success_rate"] == 1.0

    def test_save_and_load_disk_persistence(self):
        tmpdir = tempfile.mkdtemp()
        try:
            mem = SemanticMemory()
            d1 = mem.store("quantum physics relativity")
            d2 = mem.store("laravel php routing")

            # Save to disk
            mem.save_to_disk(tmpdir)

            # Load from disk
            loaded_mem = SemanticMemory.load_from_disk(tmpdir)

            assert loaded_mem.count == 2
            assert d1 in loaded_mem.documents
            assert d2 in loaded_mem.documents
            assert (
                loaded_mem.documents[d1]["metadata"]["domain"]
                == mem.documents[d1]["metadata"]["domain"]
            )

            # Centroids should be restored
            assert len(loaded_mem.cluster_manager.centroids) == len(mem.cluster_manager.centroids)
            c_id = mem.cluster_manager.cluster_assignments[d1]
            assert np.allclose(
                loaded_mem.cluster_manager.centroids[c_id], mem.cluster_manager.centroids[c_id]
            )

        finally:
            shutil.rmtree(tmpdir)
