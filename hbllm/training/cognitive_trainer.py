"""
Cognitive Trainer — trains the model AND builds cognitive structures.

Wraps the standard Trainer to also:
- Extract entities and build a knowledge graph during training
- Detect skills/domains and track mastery per topic
- Index documents by loss for curriculum learning
- Optionally train LoRA domain adapters alongside the base model

The output is a "cognitive checkpoint" containing:
  model.pt + knowledge_graph.json + skill_registry.db +
  training_memory.json + lora_adapters/ + cognitive_stats.json
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from hbllm.training.trainer import Trainer, TrainingConfig
from hbllm.training.knowledge_graph_builder import KnowledgeGraphBuilder
from hbllm.training.training_memory import TrainingMemory, detect_domain
from hbllm.memory.concept_extractor import ConceptExtractor
from hbllm.brain.skill_registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass
class CognitiveConfig:
    """Configuration for cognitive training."""
    output_dir: str = "./cognitive_checkpoints"

    # How often to run cognitive processing (every N steps)
    cognitive_interval: int = 10

    # Knowledge graph
    build_knowledge_graph: bool = True
    max_entities: int = 10000

    # Training memory
    track_training_memory: bool = True
    max_memory_records: int = 50000

    # Skill detection
    detect_skills: bool = True

    # Concept extraction
    extract_concepts: bool = True
    min_concept_frequency: int = 3

    # LoRA adapters
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05


class CognitiveTrainer:
    """
    Cognitive-aware trainer that builds brain structures during training.

    Usage:
        model = HBLLMForCausalLM(config)
        train_config = TrainingConfig(...)
        cog_config = CognitiveConfig(use_lora=True)
        cog_trainer = CognitiveTrainer(model, train_config, cog_config, device)

        for batch, raw_texts in dataloader:
            metrics = cog_trainer.cognitive_train_step(batch, raw_texts)

        cog_trainer.save_cognitive_checkpoint()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_config: TrainingConfig,
        cognitive_config: CognitiveConfig,
        device: torch.device,
    ):
        self.cognitive_config = cognitive_config
        self.output_dir = Path(cognitive_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ─── LoRA injection (before trainer init) ─────────────────────
        self.lora_injected = False
        if cognitive_config.use_lora:
            from hbllm.modules.lora import LoRAManager
            self.lora_manager = LoRAManager()
            injected = self.lora_manager.inject(
                model,
                r=cognitive_config.lora_r,
                lora_alpha=cognitive_config.lora_alpha,
                lora_dropout=cognitive_config.lora_dropout,
            )
            self.lora_injected = True
            logger.info("LoRA injected into %d layers (r=%d)", len(injected), cognitive_config.lora_r)

            # Log trainable vs total params
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info("Parameters: %s total, %s trainable (%.1f%%)",
                        f"{total:,}", f"{trainable:,}", trainable / total * 100)

        # ─── Standard trainer ─────────────────────────────────────────
        self.trainer = Trainer(model, train_config, device=device)
        self.model = model
        self.device = device

        # ─── Cognitive subsystems ─────────────────────────────────────
        if cognitive_config.build_knowledge_graph:
            self.knowledge_graph = KnowledgeGraphBuilder(
                max_entities=cognitive_config.max_entities,
            )
        else:
            self.knowledge_graph = None

        if cognitive_config.track_training_memory:
            self.training_memory = TrainingMemory(
                max_records=cognitive_config.max_memory_records,
            )
        else:
            self.training_memory = None

        if cognitive_config.extract_concepts:
            self.concept_extractor = ConceptExtractor(
                min_frequency=cognitive_config.min_concept_frequency,
            )
        else:
            self.concept_extractor = None

        if cognitive_config.detect_skills:
            self.skill_registry = SkillRegistry(
                data_dir=str(self.output_dir),
            )
        else:
            self.skill_registry = None

        # ─── Tracking ────────────────────────────────────────────────
        self._step_count = 0
        self._cognitive_step_count = 0
        self._raw_text_buffer: list[str] = []
        self._domain_losses: dict[str, list[float]] = {}
        self._start_time = time.time()

    @property
    def global_step(self) -> int:
        return self.trainer.global_step

    @property
    def config(self) -> TrainingConfig:
        return self.trainer.config

    def cognitive_train_step(
        self,
        batch: dict[str, torch.Tensor],
        raw_texts: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Perform one training step with cognitive processing.

        Args:
            batch: Tokenized batch with input_ids and labels
            raw_texts: Original text for cognitive processing (optional)

        Returns:
            Metrics dict with loss + cognitive stats
        """
        self._step_count += 1

        # 1. Standard forward + backward
        metrics = self.trainer.train_step(batch)

        # 2. Cognitive processing (every N steps, with raw text)
        if raw_texts and self._step_count % self.cognitive_config.cognitive_interval == 0:
            self._cognitive_step_count += 1
            cognitive_metrics = self._cognitive_process(raw_texts, metrics["loss"])
            metrics.update(cognitive_metrics)

        # 3. Buffer raw texts for batch-level cognitive processing
        if raw_texts:
            self._raw_text_buffer.extend(raw_texts)
            # Prevent unbounded growth
            if len(self._raw_text_buffer) > 10000:
                self._raw_text_buffer = self._raw_text_buffer[-5000:]

        return metrics

    def step(self) -> dict[str, Any]:
        """Optimizer step (after gradient accumulation). Delegates to trainer."""
        return self.trainer.step()

    def _cognitive_process(self, texts: list[str], loss: float) -> dict[str, Any]:
        """Run cognitive subsystems on raw texts."""
        cog_metrics: dict[str, Any] = {}

        # Knowledge graph: extract entities
        if self.knowledge_graph:
            entity_count = self.knowledge_graph.add_from_batch(texts, self._step_count)
            cog_metrics["kg_entities_found"] = entity_count
            cog_metrics["kg_total_entities"] = len(self.knowledge_graph.entities)

        # Training memory: record loss per domain
        if self.training_memory:
            for text in texts:
                self.training_memory.record(text, loss, self._step_count)
            self.training_memory.record_step_loss(self._step_count, loss)
            cog_metrics["memory_records"] = len(self.training_memory.records)
            cog_metrics["domains_seen"] = len(self.training_memory.domain_stats)

            # Track per-domain loss
            for text in texts:
                domain = detect_domain(text)
                self._domain_losses.setdefault(domain, []).append(loss)

        # Skill detection: check for domain patterns
        if self.skill_registry and self._cognitive_step_count % 10 == 0:
            for text in texts:
                domain = detect_domain(text)
                if domain != "general":
                    self.skill_registry.extract_and_store(
                        task_description=f"Learn {domain} patterns from training data",
                        execution_trace=[{"action": f"Processed {domain} document at step {self._step_count}"}],
                        tools_used=["pre-training"],
                        success=loss < 8.0,  # reasonable threshold
                        category=domain,
                    )

        return cog_metrics

    def save_cognitive_checkpoint(self, loss: float = 0.0) -> Path:
        """
        Save a full cognitive checkpoint:
        - Model weights
        - LoRA adapters
        - Knowledge graph
        - Training memory
        - Skill registry (already saved in SQLite)
        - Cognitive stats
        """
        ckpt_dir = self.output_dir / f"step_{self._step_count}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Saving cognitive checkpoint to %s", ckpt_dir)

        # 1. Model checkpoint
        self.trainer.save_checkpoint(loss=loss)
        logger.info("  [1/5] Model checkpoint saved")

        # 2. LoRA adapters
        if self.lora_injected:
            from hbllm.modules.lora import LoRAManager
            lora_dir = ckpt_dir / "lora_adapters"
            lora_dir.mkdir(exist_ok=True)
            lora_state = LoRAManager.get_lora_state_dict(self.model)
            torch.save(lora_state, lora_dir / "lora_adapter.pt")
            logger.info("  [2/5] LoRA adapters saved (%d params)", sum(v.numel() for v in lora_state.values()))
        else:
            logger.info("  [2/5] LoRA adapters — skipped (not enabled)")

        # 3. Knowledge graph
        if self.knowledge_graph:
            kg_path = ckpt_dir / "knowledge_graph.json"
            self.knowledge_graph.save(kg_path)
            kg_stats = self.knowledge_graph.stats()
            logger.info("  [3/5] Knowledge graph: %d entities, %d edges",
                        kg_stats["total_entities"], kg_stats.get("total_edges", 0))
        else:
            logger.info("  [3/5] Knowledge graph — skipped")

        # 4. Training memory
        if self.training_memory:
            mem_path = ckpt_dir / "training_memory.json"
            self.training_memory.save(mem_path)
            mem_stats = self.training_memory.stats()
            logger.info("  [4/5] Training memory: %d records, %d domains",
                        mem_stats["total_records"], len(mem_stats["domains"]))
            if mem_stats["mastered_domains"]:
                logger.info("         Mastered: %s", ", ".join(mem_stats["mastered_domains"]))
            if mem_stats["weak_domains"]:
                logger.info("         Weak: %s", ", ".join(mem_stats["weak_domains"]))
        else:
            logger.info("  [4/5] Training memory — skipped")

        # 5. Cognitive stats summary
        stats = self._build_cognitive_stats()
        stats_path = ckpt_dir / "cognitive_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("  [5/5] Cognitive stats saved")

        # Concept map (from accumulated texts)
        if self.concept_extractor and self._raw_text_buffer:
            concepts = self.concept_extractor.extract_from_queries(
                self._raw_text_buffer[-2000:]  # last 2000 texts
            )
            concept_map = [
                {
                    "name": c.name,
                    "keywords": c.keywords,
                    "frequency": c.frequency,
                    "rules": c.rules,
                    "confidence": c.confidence,
                }
                for c in concepts[:100]
            ]
            concept_path = ckpt_dir / "concept_map.json"
            with open(concept_path, "w") as f:
                json.dump(concept_map, f, indent=2)
            logger.info("  [+] Concept map: %d concepts extracted", len(concept_map))

        logger.info("=" * 60)
        logger.info("Cognitive checkpoint complete: %s", ckpt_dir)
        return ckpt_dir

    def _build_cognitive_stats(self) -> dict:
        """Build summary statistics for the training run."""
        elapsed = time.time() - self._start_time

        stats = {
            "training": {
                "total_steps": self._step_count,
                "cognitive_steps": self._cognitive_step_count,
                "elapsed_seconds": round(elapsed, 1),
                "lora_enabled": self.lora_injected,
            },
        }

        if self.knowledge_graph:
            stats["knowledge_graph"] = self.knowledge_graph.stats()

        if self.training_memory:
            stats["training_memory"] = self.training_memory.stats()

        if self.skill_registry:
            stats["skills"] = self.skill_registry.stats()

        # Per-domain loss curves
        if self._domain_losses:
            stats["domain_learning_curves"] = {
                domain: {
                    "samples": len(losses),
                    "avg_loss": round(sum(losses) / len(losses), 4),
                    "recent_loss": round(sum(losses[-10:]) / len(losses[-10:]), 4) if len(losses) >= 10 else None,
                    "improving": (
                        sum(losses[-10:]) / len(losses[-10:]) <
                        sum(losses[:10]) / max(1, len(losses[:10]))
                    ) if len(losses) >= 20 else None,
                }
                for domain, losses in self._domain_losses.items()
            }

        return stats

    def log_cognitive_status(self) -> None:
        """Log a summary of cognitive processing state."""
        if self.knowledge_graph:
            kg = self.knowledge_graph.stats()
            logger.info("  [KG] %d entities | top: %s",
                        kg["total_entities"],
                        ", ".join(e["name"] for e in kg["top_entities"][:5]))

        if self.training_memory:
            mem = self.training_memory.stats()
            domains_str = " | ".join(
                f"{name}:{info['mastery']:.0%}"
                for name, info in list(mem["domains"].items())[:5]
            )
            logger.info("  [Memory] %d records | %s", mem["total_records"], domains_str)

        if self.skill_registry:
            skills = self.skill_registry.stats()
            logger.info("  [Skills] %d skills across %d categories",
                        skills["total_skills"], skills["categories"])
