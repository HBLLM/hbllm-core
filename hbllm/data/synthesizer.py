"""
Data Synthesizer.

Uses the local HBLLM model (via LLMInterface) to generate synthetic
training data for new domains. Falls back to template-based generation
when no model is available.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ─── Templates (fallback when no model is loaded) ───────────────────────────

_FALLBACK_TEMPLATES = [
    ("What is {topic}?", "{topic} is a field of study that encompasses..."),
    ("How does {topic} work?", "The core mechanism of {topic} is based on..."),
    ("Can you explain the main principles of {topic}?", "Key principles of {topic} include..."),
    ("What are the applications of {topic}?", "Major applications of {topic} are..."),
    ("Give an example related to {topic}.", "An example of a concept in {topic} is..."),
    ("Why is {topic} important?", "{topic} is important because it provides..."),
    ("What are the challenges in {topic}?", "The main challenges in {topic} include..."),
    ("How has {topic} evolved over time?", "The evolution of {topic} has progressed through..."),
]


class DataSynthesizer:
    """
    Generates synthetic Q&A pairs for a given topic.

    When a model + tokenizer are provided, uses the LLM to produce
    diverse, high-quality training data. Otherwise, falls back to
    template-based generation.
    """

    def __init__(self, model: Any = None, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._llm: Any = None

    def _get_llm(self) -> Any:
        """Lazily construct LLMInterface from model + tokenizer."""
        if self._llm is not None:
            return self._llm

        if self.model is None or self.tokenizer is None:
            return None

        try:
            from hbllm.brain.llm_interface import LLMInterface

            self._llm = LLMInterface(self.model, self.tokenizer)
            return self._llm
        except Exception as e:
            logger.warning("Could not initialize LLMInterface: %s", e)
            return None

    def generate_dataset(
        self,
        topic: str,
        num_samples: int = 5,
        output_dir: str = "workspace/synthetic",
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Generate a synthetic dataset for the given topic.

        Args:
            topic: The domain to generate data for.
            num_samples: Number of Q&A pairs to generate.
            output_dir: Directory to save the dataset.
            few_shot_examples: Optional list of example Q&A dicts for few-shot prompting.

        Returns:
            The path to the generated JSONL dataset.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"synthetic_{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.jsonl"
        filepath = os.path.join(output_dir, filename)

        logger.info("Generating %d synthetic samples for topic '%s'...", num_samples, topic)

        llm = self._get_llm()

        if llm is not None:
            dataset = self._generate_with_llm(llm, topic, num_samples, few_shot_examples)
        else:
            logger.info("No model loaded — using template fallback for data synthesis")
            dataset = self._generate_with_templates(topic, num_samples)

        with open(filepath, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")

        logger.info("Synthetic dataset saved to %s (%d samples)", filepath, len(dataset))
        return filepath

    def _generate_with_llm(
        self,
        llm: Any,
        topic: str,
        num_samples: int,
        few_shot_examples: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Generate data using the actual LLM model."""
        import asyncio

        dataset = []

        # Build few-shot context
        few_shot = ""
        if few_shot_examples:
            for ex in few_shot_examples[:3]:  # Max 3 examples
                few_shot += (
                    f"Q: {ex.get('instruction', ex.get('question', ''))}\n"
                    f"A: {ex.get('response', ex.get('answer', ''))}\n\n"
                )

        for i in range(num_samples):
            prompt = self._build_generation_prompt(topic, i, few_shot)

            try:
                # Run async generate in sync context
                loop = asyncio.new_event_loop()
                try:
                    raw_output = loop.run_until_complete(
                        llm.generate(prompt, max_tokens=256, temperature=0.8)
                    )
                finally:
                    loop.close()

                question, answer = self._parse_qa_output(raw_output, topic, i)
            except Exception as e:
                logger.warning("LLM generation failed for sample %d: %s — using template", i, e)
                question, answer = self._template_pair(topic, i)

            dataset.append(
                {
                    "instruction": question,
                    "context": "",
                    "response": answer,
                    "topic": topic,
                    "source": "llm_generated",
                }
            )

        return dataset

    def _build_generation_prompt(self, topic: str, index: int, few_shot: str) -> str:
        """Build a prompt that instructs the model to generate a Q&A pair."""
        aspects = [
            "definition and overview",
            "how it works",
            "key principles",
            "real-world applications",
            "common challenges",
            "historical evolution",
            "comparison with alternatives",
            "best practices",
        ]
        aspect = aspects[index % len(aspects)]

        prompt = (
            f"You are a domain expert. Generate a training question and detailed answer "
            f"about {topic}, focusing on {aspect}.\n\n"
        )
        if few_shot:
            prompt += f"Here are some examples of the format:\n{few_shot}\n"

        prompt += (
            f"Now generate a NEW question and answer about {topic} ({aspect}).\n"
            f"Format:\nQ: <question>\nA: <detailed answer>\n"
        )
        return prompt

    def _parse_qa_output(self, raw_output: str, topic: str, index: int) -> tuple[str, str]:
        """Parse a Q&A pair from LLM output."""
        lines = raw_output.strip().split("\n")

        question = ""
        answer_parts = []
        in_answer = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Q:") and not question:
                question = stripped[2:].strip()
            elif stripped.startswith("A:"):
                in_answer = True
                answer_parts.append(stripped[2:].strip())
            elif in_answer:
                answer_parts.append(stripped)

        answer = " ".join(answer_parts).strip()

        # If parsing failed, use raw output as answer with a generated question
        if not question or not answer:
            if raw_output.strip():
                question = f"Explain {topic} in detail."
                answer = raw_output.strip()
            else:
                return self._template_pair(topic, index)

        return question, answer

    def _generate_with_templates(self, topic: str, num_samples: int) -> list[dict[str, Any]]:
        """Fallback: generate data from templates."""
        dataset = []
        for i in range(num_samples):
            question, answer = self._template_pair(topic, i)
            dataset.append(
                {
                    "instruction": question,
                    "context": "",
                    "response": answer,
                    "topic": topic,
                    "source": "template",
                }
            )
        return dataset

    @staticmethod
    def _template_pair(topic: str, index: int) -> tuple[str, str]:
        """Get a template-based Q&A pair."""
        q_tpl, a_tpl = _FALLBACK_TEMPLATES[index % len(_FALLBACK_TEMPLATES)]
        return q_tpl.format(topic=topic), a_tpl.format(topic=topic)
