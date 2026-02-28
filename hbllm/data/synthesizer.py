"""
Data Synthesizer.

Provides utilities for using the base Language Model to generate 
synthetic training data for new domains dynamically. This is used
during the Self-Expansion phase.
"""

from __future__ import annotations

import logging
import torch
import json
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

class DataSynthesizer:
    """
    Generates synthetic Q&A pairs for a given topic using the base model.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate_dataset(self, topic: str, num_samples: int = 5, output_dir: str = "workspace/synthetic") -> str:
        """
        Generate a synthetic dataset for the given topic.
        
        Args:
            topic: The domain to generate data for.
            num_samples: Number of Q&A pairs to generate.
            output_dir: Directory to save the dataset.
            
        Returns:
            The path to the generated JSONL dataset.
        """
        os.makedirs(output_dir, exist_ok=True)
        filename = f"synthetic_{topic.replace(' ', '_')}_{uuid.uuid4().hex[:8]}.jsonl"
        filepath = os.path.join(output_dir, filename)
        
        logger.info("Generating %d synthetic samples for topic '%s'...", num_samples, topic)
        
        dataset = []
        
        # In a real deployed self-expansion system, you would use few-shot 
        # prompting on a highly capable base model to generate these.
        # For our prototype, since our 125M model might just output gibberish for
        # complex prompts, we will simulate the generation with high quality curated templates
        # or highly constrained prompting.
        
        # We simulate the LM generating domain-specific data:
        templates = [
            ("What is {topic}?", "The study of {topic} involves..."),
            ("How does {topic} work?", "The core mechanism of {topic} is based on..."),
            ("Can you explain the main principles of {topic}?", "Key principles of {topic} include..."),
            ("What are the applications of {topic}?", "Major applications of {topic} are..."),
            ("Give an example related to {topic}.", "An example of a concept in {topic} is..."),
        ]
        
        for i in range(num_samples):
            # Select a template cyclically
            q_template, a_template = templates[i % len(templates)]
            
            # Formulate the prompt as if the LLM was asked to generate it
            sys_prompt = f"Generate a training question and answer about {topic}.\n"
            
            # Simulate the base model "answering" this meta-prompt to generate synthetic data
            # (In reality, `self.model.generate` would be called here. We use simulation
            # to keep the prototype fast and avoid low-quality base model outputs poisoning the LoRA).
            
            question = q_template.format(topic=topic)
            answer = a_template.format(topic=topic) + f" [Synthetic Data Fragment {i}]"
            
            # Format as standard instruction tuning data
            sample = {
                "instruction": question,
                "context": "",
                "response": answer,
                "topic": topic
            }
            dataset.append(sample)
            
        with open(filepath, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
                
        logger.info("Synthetic dataset saved to %s", filepath)
        return filepath
