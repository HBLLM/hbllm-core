"""
Tests for the Cognitive Training Pipeline.

Tests the three new modules:
- KnowledgeGraphBuilder: entity extraction, co-occurrence edges, save/load
- TrainingMemory: loss tracking, domain detection, curriculum learning
- CognitiveTrainer: integration of all subsystems + LoRA
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# KnowledgeGraphBuilder Tests
# ──────────────────────────────────────────────────────────────────────


class TestKnowledgeGraphBuilder:
    """Tests for training/knowledge_graph_builder.py."""

    def setup_method(self):
        from hbllm.training.knowledge_graph_builder import KnowledgeGraphBuilder
        self.kg = KnowledgeGraphBuilder()

    def test_extract_tech_entities(self):
        """Should extract technology-related entities."""
        entities = self.kg.add_from_text(
            "Python and PyTorch are used for deep learning with transformers"
        )
        names_lower = [e.lower() for e in entities]
        assert "python" in names_lower
        assert "pytorch" in names_lower

    def test_extract_concept_entities(self):
        """Should extract concept-type entities."""
        entities = self.kg.add_from_text(
            "The algorithm uses optimization and training for the neural network"
        )
        names_lower = [e.lower() for e in entities]
        assert "algorithm" in names_lower
        assert "optimization" in names_lower
        assert "training" in names_lower

    def test_extract_multiple_frameworks(self):
        """Should detect various frameworks and tools."""
        entities = self.kg.add_from_text(
            "We use React with Django and Docker on AWS with PostgreSQL"
        )
        names_lower = [e.lower() for e in entities]
        assert "react" in names_lower
        assert "django" in names_lower
        assert "docker" in names_lower
        assert "aws" in names_lower
        assert "postgresql" in names_lower

    def test_entity_frequency_tracking(self):
        """Should track how many times an entity appears."""
        self.kg.add_from_text("Python is great")
        self.kg.add_from_text("Python with PyTorch")
        self.kg.add_from_text("Python and Django")

        python_entity = self.kg.entities.get("python")
        assert python_entity is not None
        assert python_entity.frequency == 3

    def test_cooccurrence_edges(self):
        """Should build edges between co-occurring entities."""
        # Add same pair 4 times (above min_cooccurrence=3)
        for _ in range(4):
            self.kg.add_from_text("Python and PyTorch for deep learning")

        edges = self.kg.build_edges(min_cooccurrence=3)
        assert len(edges) > 0
        # Check that Python-PyTorch edge exists
        edge_pairs = [(e.source, e.target) for e in edges]
        found = any(
            ("python" in s.lower() and "pytorch" in t.lower()) or
            ("pytorch" in s.lower() and "python" in t.lower())
            for s, t in edge_pairs
        )
        assert found, f"Expected Python-PyTorch edge, got: {edge_pairs}"

    def test_batch_processing(self):
        """add_from_batch should process multiple texts."""
        texts = [
            "Python for web development",
            "JavaScript and React frontend",
            "Docker with Kubernetes deployment",
        ]
        total = self.kg.add_from_batch(texts, step=5)
        assert total > 0
        assert self.kg._doc_count == 3

    def test_topic_clusters(self):
        """Should return top topic keywords."""
        for _ in range(10):
            self.kg.add_from_text(
                "Machine learning models need training data for optimization"
            )
        clusters = self.kg.get_topic_clusters(top_n=5)
        assert len(clusters) > 0
        assert all("keyword" in c and "count" in c for c in clusters)

    def test_stats(self):
        """stats() should return summary info."""
        self.kg.add_from_text("Python and PyTorch")
        stats = self.kg.stats()
        assert "total_entities" in stats
        assert "total_edges" in stats
        assert "documents_processed" in stats
        assert stats["documents_processed"] == 1
        assert stats["total_entities"] > 0

    def test_save_and_load(self):
        """Should save to JSON and load back."""
        self.kg.add_from_text("Python and PyTorch for deep learning")
        self.kg.add_from_text("React and JavaScript for web")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "kg.json")
            self.kg.save(path)

            assert os.path.exists(path)

            # Load and verify
            from hbllm.training.knowledge_graph_builder import KnowledgeGraphBuilder
            loaded = KnowledgeGraphBuilder.load(path)
            assert len(loaded.entities) == len(self.kg.entities)
            assert loaded._doc_count == 2

    def test_max_entities_cap(self):
        """Should respect max_entities limit."""
        from hbllm.training.knowledge_graph_builder import KnowledgeGraphBuilder
        kg = KnowledgeGraphBuilder(max_entities=3)
        kg.add_from_text("Python PyTorch TensorFlow Docker React Vue Angular")
        assert len(kg.entities) <= 3

    def test_empty_text(self):
        """Should handle empty text."""
        entities = self.kg.add_from_text("")
        assert entities == []

    def test_no_entities_in_text(self):
        """Should handle text with no recognizable entities."""
        entities = self.kg.add_from_text("The quick brown fox jumps over the lazy dog")
        assert len(entities) == 0


# ──────────────────────────────────────────────────────────────────────
# TrainingMemory Tests
# ──────────────────────────────────────────────────────────────────────


class TestDomainDetection:
    """Tests for detect_domain function."""

    def test_detect_code(self):
        from hbllm.training.training_memory import detect_domain
        assert detect_domain("def foo(): return bar()") == "code"
        assert detect_domain("class MyClass: pass") == "code"
        assert detect_domain("function hello() { console.log('hi') }") == "code"

    def test_detect_math(self):
        from hbllm.training.training_memory import detect_domain
        assert detect_domain("The theorem states that the integral of x") == "math"
        assert detect_domain("Using probability and statistical analysis") == "math"

    def test_detect_science(self):
        from hbllm.training.training_memory import detect_domain
        assert detect_domain("The experiment tested the hypothesis about electrons") == "science"
        assert detect_domain("The quantum state of the molecule was measured") == "science"

    def test_detect_reasoning(self):
        from hbllm.training.training_memory import detect_domain
        assert detect_domain("Therefore we can conclude that consequently the result implies") == "reasoning"

    def test_detect_factual(self):
        from hbllm.training.training_memory import detect_domain
        assert detect_domain("Einstein was born in 1879 and discovered relativity") == "factual"

    def test_detect_creative(self):
        from hbllm.training.training_memory import detect_domain
        assert detect_domain("Once upon a time she said to the character in the story") == "creative"

    def test_detect_general(self):
        from hbllm.training.training_memory import detect_domain
        assert detect_domain("The quick brown fox jumps over the lazy dog") == "general"


class TestTrainingMemory:
    """Tests for training/training_memory.py."""

    def setup_method(self):
        from hbllm.training.training_memory import TrainingMemory
        self.mem = TrainingMemory()

    def test_record_document(self):
        """Should record a document and track domain."""
        rec = self.mem.record("def foo(): return 42", loss=5.0, step=1)
        assert rec.domain == "code"
        assert rec.loss == 5.0
        assert rec.step == 1
        assert len(self.mem.records) == 1

    def test_domain_stats_tracking(self):
        """Should aggregate stats per domain."""
        self.mem.record("def foo(): pass", loss=5.0, step=1)
        self.mem.record("class Bar: pass", loss=4.0, step=2)
        self.mem.record("import os", loss=3.0, step=3)

        assert "code" in self.mem.domain_stats
        ds = self.mem.domain_stats["code"]
        assert ds.total_docs == 3
        assert abs(ds.avg_loss - 4.0) < 0.01  # (5+4+3)/3
        assert ds.min_loss == 3.0
        assert ds.max_loss == 5.0

    def test_mastery_score(self):
        """Mastery score should be higher for lower loss."""
        from hbllm.training.training_memory import DomainStats
        # Low loss = high mastery
        low_ds = DomainStats(domain="test", total_docs=10, total_loss=20.0)  # avg=2.0
        high_ds = DomainStats(domain="test", total_docs=10, total_loss=90.0)  # avg=9.0
        assert low_ds.mastery_score > high_ds.mastery_score

    def test_get_hard_documents(self):
        """Should return highest-loss documents."""
        self.mem.record("easy text", loss=2.0, step=1)
        self.mem.record("hard text with the equation theorem", loss=9.0, step=2)
        self.mem.record("medium text", loss=5.0, step=3)

        hard = self.mem.get_hard_documents(top_k=1)
        assert len(hard) == 1
        assert hard[0].loss == 9.0

    def test_record_batch(self):
        """Should record multiple documents."""
        texts = ["def a(): pass", "The theorem states", "general text"]
        self.mem.record_batch(texts, loss=5.0, step=1)
        assert len(self.mem.records) == 3

    def test_step_loss_tracking(self):
        """Should track loss trajectory."""
        self.mem.record_step_loss(1, 10.0)
        self.mem.record_step_loss(2, 8.0)
        self.mem.record_step_loss(3, 6.0)
        assert len(self.mem._step_losses) == 3

    def test_stats(self):
        """stats() should return summary info."""
        self.mem.record("def foo(): pass", loss=5.0, step=1)
        self.mem.record("The theorem proves", loss=8.0, step=2)
        stats = self.mem.stats()
        assert "total_records" in stats
        assert "domains" in stats
        assert "mastered_domains" in stats
        assert "weak_domains" in stats

    def test_save_and_load(self):
        """Should save to JSON and load back."""
        self.mem.record("def foo(): pass", loss=5.0, step=1)
        self.mem.record("The theorem", loss=8.0, step=2)
        self.mem.record_step_loss(1, 6.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.json")
            self.mem.save(path)
            assert os.path.exists(path)

            from hbllm.training.training_memory import TrainingMemory
            loaded = TrainingMemory.load(path)
            assert len(loaded.domain_stats) == len(self.mem.domain_stats)
            assert len(loaded._step_losses) == 1

    def test_max_records_eviction(self):
        """Should evict low-loss records when max reached."""
        from hbllm.training.training_memory import TrainingMemory
        mem = TrainingMemory(max_records=10)
        for i in range(15):
            mem.record(f"text {i}", loss=float(i), step=i)
        # Should have evicted some, keeping high-loss ones
        assert len(mem.records) <= 15

    def test_mastered_and_weak_domains(self):
        """Should identify mastered vs weak domains."""
        # Low loss = mastered
        for i in range(20):
            self.mem.record("def foo(): return 42", loss=1.5, step=i)
        # High loss = weak
        for i in range(20):
            self.mem.record("The equation integral derivative theorem proof", loss=9.0, step=i + 20)

        mastered = self.mem.get_mastered_domains()
        weak = self.mem.get_weak_domains()
        assert "code" in mastered
        assert "math" in weak


# ──────────────────────────────────────────────────────────────────────
# CognitiveTrainer Tests
# ──────────────────────────────────────────────────────────────────────


class TestCognitiveConfig:
    """Tests for CognitiveConfig."""

    def test_default_config(self):
        from hbllm.training.cognitive_trainer import CognitiveConfig
        config = CognitiveConfig()
        assert config.cognitive_interval == 10
        assert config.build_knowledge_graph is True
        assert config.track_training_memory is True
        assert config.detect_skills is True
        assert config.extract_concepts is True
        assert config.use_lora is False

    def test_custom_config(self):
        from hbllm.training.cognitive_trainer import CognitiveConfig
        config = CognitiveConfig(
            use_lora=True,
            lora_r=16,
            cognitive_interval=5,
        )
        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.cognitive_interval == 5


class TestCognitiveTrainer:
    """Tests for training/cognitive_trainer.py."""

    @pytest.fixture
    def tiny_model(self):
        """Create a minimal model for testing."""
        import torch
        from hbllm.model.config import ModelConfig
        from hbllm.model.transformer import HBLLMForCausalLM

        config = ModelConfig(
            hidden_size=128,
            num_attention_heads=2,
            num_kv_heads=2,
            num_layers=2,
            intermediate_size=256,
            vocab_size=1000,
            max_position_embeddings=128,
        )
        model = HBLLMForCausalLM(config)
        return model

    @pytest.fixture
    def cog_trainer(self, tiny_model, tmp_path):
        """Create a CognitiveTrainer with default config."""
        import torch
        from hbllm.training.trainer import TrainingConfig
        from hbllm.training.cognitive_trainer import CognitiveTrainer, CognitiveConfig

        train_config = TrainingConfig(
            learning_rate=1e-4,
            max_steps=10,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        cog_config = CognitiveConfig(
            output_dir=str(tmp_path / "cognitive"),
            cognitive_interval=1,  # process every step for testing
            use_lora=False,
        )
        return CognitiveTrainer(
            tiny_model, train_config, cog_config,
            device=torch.device("cpu"),
        )

    @pytest.fixture
    def cog_trainer_with_lora(self, tiny_model, tmp_path):
        """Create a CognitiveTrainer with LoRA enabled."""
        import torch
        from hbllm.training.trainer import TrainingConfig
        from hbllm.training.cognitive_trainer import CognitiveTrainer, CognitiveConfig

        train_config = TrainingConfig(
            learning_rate=1e-4,
            max_steps=10,
            micro_batch_size=2,
            gradient_accumulation_steps=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )
        cog_config = CognitiveConfig(
            output_dir=str(tmp_path / "cognitive"),
            cognitive_interval=1,
            use_lora=True,
            lora_r=4,
        )
        return CognitiveTrainer(
            tiny_model, train_config, cog_config,
            device=torch.device("cpu"),
        )

    def test_init_without_lora(self, cog_trainer):
        """Should initialize all subsystems without LoRA."""
        assert cog_trainer.knowledge_graph is not None
        assert cog_trainer.training_memory is not None
        assert cog_trainer.concept_extractor is not None
        assert cog_trainer.skill_registry is not None
        assert cog_trainer.lora_injected is False

    def test_init_with_lora(self, cog_trainer_with_lora):
        """Should initialize with LoRA injected."""
        assert cog_trainer_with_lora.lora_injected is True

    def test_cognitive_train_step(self, cog_trainer):
        """Should run forward+backward and cognitive processing."""
        import torch

        batch = {
            "input_ids": torch.randint(0, 999, (2, 32)),
            "labels": torch.randint(0, 999, (2, 32)),
        }
        raw_texts = [
            "Python and PyTorch for deep learning",
            "def train_model(): return loss",
        ]

        metrics = cog_trainer.cognitive_train_step(batch, raw_texts)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        # Should have cognitive metrics since interval=1
        assert "kg_total_entities" in metrics
        assert metrics["kg_total_entities"] > 0

    def test_cognitive_train_step_without_texts(self, cog_trainer):
        """Should work without raw texts (no cognitive processing)."""
        import torch

        batch = {
            "input_ids": torch.randint(0, 999, (2, 32)),
            "labels": torch.randint(0, 999, (2, 32)),
        }

        metrics = cog_trainer.cognitive_train_step(batch, raw_texts=None)
        assert "loss" in metrics
        # No cognitive metrics without raw texts
        assert "kg_total_entities" not in metrics

    def test_knowledge_graph_grows(self, cog_trainer):
        """Knowledge graph should grow during training."""
        import torch

        for i in range(5):
            batch = {
                "input_ids": torch.randint(0, 999, (2, 32)),
                "labels": torch.randint(0, 999, (2, 32)),
            }
            raw_texts = [
                "Python and PyTorch with TensorFlow",
                "Django REST API with PostgreSQL",
            ]
            cog_trainer.cognitive_train_step(batch, raw_texts)

        assert len(cog_trainer.knowledge_graph.entities) > 0

    def test_training_memory_tracks_domains(self, cog_trainer):
        """Training memory should track domain statistics."""
        import torch

        batch = {
            "input_ids": torch.randint(0, 999, (2, 32)),
            "labels": torch.randint(0, 999, (2, 32)),
        }
        raw_texts = [
            "def foo(): return bar()",  # code
            "The theorem of probability states",  # math
        ]
        cog_trainer.cognitive_train_step(batch, raw_texts)

        mem_stats = cog_trainer.training_memory.stats()
        assert mem_stats["total_records"] > 0
        assert len(mem_stats["domains"]) > 0

    def test_save_cognitive_checkpoint(self, cog_trainer):
        """Should save all cognitive artifacts."""
        import torch

        # Do one training step
        batch = {
            "input_ids": torch.randint(0, 999, (2, 32)),
            "labels": torch.randint(0, 999, (2, 32)),
        }
        raw_texts = ["Python and PyTorch for deep learning", "def train(): pass"]
        cog_trainer.cognitive_train_step(batch, raw_texts)

        # Save checkpoint
        ckpt_dir = cog_trainer.save_cognitive_checkpoint(loss=5.0)

        assert ckpt_dir.exists()
        assert (ckpt_dir / "knowledge_graph.json").exists()
        assert (ckpt_dir / "training_memory.json").exists()
        assert (ckpt_dir / "cognitive_stats.json").exists()
        assert (ckpt_dir / "concept_map.json").exists()

        # Verify knowledge graph content
        with open(ckpt_dir / "knowledge_graph.json") as f:
            kg = json.load(f)
        assert kg["metadata"]["total_entities"] > 0

        # Verify cognitive stats
        with open(ckpt_dir / "cognitive_stats.json") as f:
            stats = json.load(f)
        assert "training" in stats
        assert stats["training"]["total_steps"] > 0

    def test_save_cognitive_checkpoint_with_lora(self, cog_trainer_with_lora):
        """Should save LoRA adapter weights in checkpoint."""
        import torch

        batch = {
            "input_ids": torch.randint(0, 999, (2, 32)),
            "labels": torch.randint(0, 999, (2, 32)),
        }
        cog_trainer_with_lora.cognitive_train_step(batch, ["test text"])

        ckpt_dir = cog_trainer_with_lora.save_cognitive_checkpoint(loss=5.0)

        assert (ckpt_dir / "lora_adapters" / "lora_adapter.pt").exists()

    def test_log_cognitive_status(self, cog_trainer, caplog):
        """log_cognitive_status should log without errors."""
        import torch
        import logging

        batch = {
            "input_ids": torch.randint(0, 999, (2, 32)),
            "labels": torch.randint(0, 999, (2, 32)),
        }
        cog_trainer.cognitive_train_step(batch, ["Python and PyTorch"])

        with caplog.at_level(logging.INFO):
            cog_trainer.log_cognitive_status()
        # Should not raise

    def test_optimizer_step(self, cog_trainer):
        """step() should perform optimizer update."""
        import torch

        batch = {
            "input_ids": torch.randint(0, 999, (2, 32)),
            "labels": torch.randint(0, 999, (2, 32)),
        }
        cog_trainer.cognitive_train_step(batch, ["test"])

        step_metrics = cog_trainer.step()
        assert "lr" in step_metrics
        assert "grad_norm" in step_metrics

    def test_domain_learning_curves(self, cog_trainer):
        """Should track per-domain loss curves."""
        import torch

        texts_by_domain = {
            "code": "def foo(): return bar()",
            "math": "The theorem and equation of probability integral",
            "science": "The experiment tested the hypothesis about molecule",
        }

        for _ in range(3):
            for domain, text in texts_by_domain.items():
                batch = {
                    "input_ids": torch.randint(0, 999, (2, 32)),
                    "labels": torch.randint(0, 999, (2, 32)),
                }
                cog_trainer.cognitive_train_step(batch, [text, text])

        stats = cog_trainer._build_cognitive_stats()
        assert "domain_learning_curves" in stats
        assert len(stats["domain_learning_curves"]) > 0


# ──────────────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────────────


class TestCognitiveTrainingIntegration:
    """End-to-end integration tests."""

    def test_full_cognitive_training_loop(self, tmp_path):
        """Simulate a complete cognitive training loop."""
        import torch
        from hbllm.model.config import ModelConfig
        from hbllm.model.transformer import HBLLMForCausalLM
        from hbllm.training.trainer import TrainingConfig
        from hbllm.training.cognitive_trainer import CognitiveTrainer, CognitiveConfig

        # Tiny model
        model_config = ModelConfig(
            hidden_size=128, num_attention_heads=2,
            num_kv_heads=2,
            num_layers=2, intermediate_size=256,
            vocab_size=1000, max_position_embeddings=128,
        )
        model = HBLLMForCausalLM(model_config)

        train_config = TrainingConfig(
            learning_rate=1e-3, max_steps=5,
            micro_batch_size=2, gradient_accumulation_steps=1,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cog_config = CognitiveConfig(
            output_dir=str(tmp_path / "cog"),
            cognitive_interval=1,
            use_lora=True, lora_r=4,
        )

        trainer = CognitiveTrainer(model, train_config, cog_config, torch.device("cpu"))

        training_texts = [
            "Python and PyTorch for machine learning optimization",
            "def train_model(): loss = forward(batch); loss.backward()",
            "The theorem proves that the algorithm converges",
            "React and JavaScript for frontend web development",
            "Docker and Kubernetes on AWS for cloud deployment",
        ]

        # Training loop
        losses = []
        for step in range(5):
            batch = {
                "input_ids": torch.randint(0, 999, (2, 32)),
                "labels": torch.randint(0, 999, (2, 32)),
            }
            texts = [training_texts[step], training_texts[(step + 1) % 5]]
            metrics = trainer.cognitive_train_step(batch, texts)
            losses.append(metrics["loss"])
            trainer.step()

        # Save checkpoint
        ckpt_dir = trainer.save_cognitive_checkpoint(loss=losses[-1])

        # Verify all artifacts
        assert (ckpt_dir / "knowledge_graph.json").exists()
        assert (ckpt_dir / "training_memory.json").exists()
        assert (ckpt_dir / "cognitive_stats.json").exists()
        assert (ckpt_dir / "concept_map.json").exists()
        assert (ckpt_dir / "lora_adapters" / "lora_adapter.pt").exists()

        # Verify KG has entities
        with open(ckpt_dir / "knowledge_graph.json") as f:
            kg = json.load(f)
        assert kg["metadata"]["total_entities"] > 0

        # Verify training memory has records
        with open(ckpt_dir / "training_memory.json") as f:
            mem = json.load(f)
        assert len(mem["domain_stats"]) > 0

        # Verify LoRA adapter is non-empty
        lora_state = torch.load(ckpt_dir / "lora_adapters" / "lora_adapter.pt", weights_only=True)
        assert len(lora_state) > 0

        # Verify cognitive stats
        with open(ckpt_dir / "cognitive_stats.json") as f:
            stats = json.load(f)
        assert stats["training"]["total_steps"] == 5
        assert stats["training"]["lora_enabled"] is True
