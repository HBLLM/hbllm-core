"""
Model evaluation benchmarks for HBLLM.

Provides:
  - Perplexity evaluation on held-out data
  - HellaSwag accuracy (common sense reasoning)
  - Sample generation quality review
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates model quality with standard benchmarks.

    Usage:
        evaluator = ModelEvaluator(model, tokenizer, device)
        results = evaluator.evaluate_all(eval_dataloader)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model.eval()

    # ─── Perplexity ──────────────────────────────────────────────────

    @torch.no_grad()
    def compute_perplexity(
        self,
        dataloader: Any,
        max_batches: int = 100,
    ) -> dict[str, float]:
        """
        Compute perplexity on evaluation data.

        Args:
            dataloader: PyTorch DataLoader yielding {input_ids, labels}
            max_batches: Max evaluation batches

        Returns:
            Dict with perplexity, avg_loss, total_tokens
        """
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for batch in dataloader:
            if num_batches >= max_batches:
                break

            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]

            batch_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            num_batches += 1

        avg_loss = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(avg_loss, 100))  # Clamp to avoid overflow

        return {
            "perplexity": round(perplexity, 2),
            "avg_loss": round(avg_loss, 4),
            "total_tokens": total_tokens,
            "num_batches": num_batches,
        }

    # ─── HellaSwag ───────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate_hellaswag(
        self,
        examples: list[dict] | None = None,
        max_examples: int = 200,
    ) -> dict[str, float]:
        """
        HellaSwag common-sense reasoning benchmark.

        Each example has a context and 4 continuations; model picks
        the one with lowest perplexity.

        Args:
            examples: Pre-loaded examples or None to load from HuggingFace
            max_examples: Max examples to evaluate

        Returns:
            Dict with accuracy, correct, total
        """
        if examples is None:
            examples = self._load_hellaswag(max_examples)

        correct = 0
        total = 0

        for ex in examples[:max_examples]:
            context = ex.get("ctx", ex.get("context", ""))
            endings = ex.get("endings", [])
            label = int(ex.get("label", 0))

            if not endings or not context:
                continue

            # Score each continuation
            scores = []
            for ending in endings:
                full_text = context + " " + ending
                ids = self.tokenizer.encode(full_text)
                input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs if isinstance(outputs, torch.Tensor) else outputs["loss"]
                scores.append(loss.item())

            # Lowest loss = best continuation
            prediction = scores.index(min(scores))
            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        return {
            "accuracy": round(accuracy, 4),
            "correct": correct,
            "total": total,
        }

    def _load_hellaswag(self, max_examples: int) -> list[dict]:
        """Load HellaSwag from HuggingFace."""
        try:
            from datasets import load_dataset
            ds = load_dataset("Rowan/hellaswag", split="validation", streaming=True)
            examples = []
            for item in ds:
                examples.append(item)
                if len(examples) >= max_examples:
                    break
            return examples
        except Exception as e:
            logger.warning("Could not load HellaSwag: %s", e)
            return []

    # ─── Sample Generation ───────────────────────────────────────────

    @torch.no_grad()
    def generate_samples(
        self,
        prompts: list[str] | None = None,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
    ) -> list[dict[str, str]]:
        """
        Generate text samples for manual quality review.

        Returns:
            List of {prompt, generated} dicts
        """
        if prompts is None:
            prompts = [
                "The meaning of life is",
                "In the year 2050, technology will",
                "The most important scientific discovery was",
                "Once upon a time in a distant land,",
            ]

        results = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

            try:
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.9,
                )
                generated = self.tokenizer.decode(output[0].tolist())
            except Exception as e:
                generated = f"[Error: {e}]"

            results.append({"prompt": prompt, "generated": generated})

        return results

    # ─── Combined Evaluation ─────────────────────────────────────────

    def evaluate_all(
        self,
        eval_dataloader: Any = None,
        hellaswag: bool = True,
        generate: bool = True,
        max_eval_batches: int = 100,
        max_hellaswag: int = 200,
    ) -> dict[str, Any]:
        """Run all evaluations and return combined results."""
        results: dict[str, Any] = {}

        if eval_dataloader is not None:
            logger.info("Computing perplexity...")
            t0 = time.time()
            results["perplexity"] = self.compute_perplexity(
                eval_dataloader, max_batches=max_eval_batches
            )
            logger.info(
                "Perplexity: %.2f (%.1fs)",
                results["perplexity"]["perplexity"],
                time.time() - t0,
            )

        if hellaswag:
            logger.info("Evaluating HellaSwag...")
            t0 = time.time()
            results["hellaswag"] = self.evaluate_hellaswag(max_examples=max_hellaswag)
            logger.info(
                "HellaSwag accuracy: %.1f%% (%.1fs)",
                results["hellaswag"]["accuracy"] * 100,
                time.time() - t0,
            )

        if generate:
            logger.info("Generating samples...")
            results["samples"] = self.generate_samples()
            for s in results["samples"]:
                logger.info("  Prompt: %s...", s["prompt"][:50])
                logger.info("  Output: %s...", s["generated"][:100])

        return results
