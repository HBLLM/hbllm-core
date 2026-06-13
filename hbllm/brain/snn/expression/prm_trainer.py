"""
PRM Trainer — batch STDP training pipeline.

Runs periodic training sweeps over the ``TrainingCollector``'s
accumulated data, reinforcing weight patterns that correlate with
accepted fragments and weakening patterns that correlate with revisions.

This upgrades the v3 online-only learning (1 example at a time) with
batch training that produces measurable accuracy improvements.

Usage::

    trainer = PRMTrainer(trained_prm)
    if trainer.should_train():
        metrics = trainer.train()
        print(f"Accuracy: {metrics.pre_accuracy:.0%} → {metrics.post_accuracy:.0%}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Outcome of a batch training sweep.

    Attributes:
        examples_trained: Total examples used in this sweep.
        epochs_completed: Number of full passes over the data.
        pre_accuracy: Accept/revise prediction accuracy before training.
        post_accuracy: Accuracy after training.
        weight_delta: Total absolute weight change across all projections.
        training_duration_ms: Wall-clock time for the sweep.
        timestamp: When the training completed.
    """

    examples_trained: int = 0
    epochs_completed: int = 0
    pre_accuracy: float = 0.0
    post_accuracy: float = 0.0
    weight_delta: float = 0.0
    training_duration_ms: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples_trained": self.examples_trained,
            "epochs_completed": self.epochs_completed,
            "pre_accuracy": round(self.pre_accuracy, 4),
            "post_accuracy": round(self.post_accuracy, 4),
            "weight_delta": round(self.weight_delta, 6),
            "training_duration_ms": round(self.training_duration_ms, 1),
        }


class PRMTrainer:
    """Batch STDP training for the TrainedPRM's RewardNetwork.

    Periodically sweeps over accumulated training data, running
    multiple epochs of supervised STDP updates.

    Training flow:
    1. ``should_train()`` — check if enough new data accumulated
    2. ``train()`` — run batch STDP sweep
    3. ``get_metrics()`` — inspect training history

    Args:
        trained_prm: The TrainedPRM to train (accesses its
            RewardNetwork and TrainingCollector).
        epochs: Number of full passes over the data per sweep.
        batch_size: Number of examples per STDP batch.
        min_new_examples: Minimum new examples before training
            is triggered.
        settle_steps: SNN simulation steps per training example.
    """

    def __init__(
        self,
        trained_prm: Any,
        epochs: int = 3,
        batch_size: int = 20,
        min_new_examples: int = 20,
        settle_steps: int = 3,
    ) -> None:
        self._prm = trained_prm
        self._epochs = epochs
        self._batch_size = batch_size
        self._min_new_examples = min_new_examples
        self._settle_steps = settle_steps

        self._last_trained_count = 0
        self._history: list[TrainingMetrics] = []

    def should_train(self) -> bool:
        """Check if enough new data has accumulated for training.

        Returns:
            True if at least ``min_new_examples`` new examples
            have been recorded since the last training sweep.
        """
        current_count = self._prm.collector.count
        new_examples = current_count - self._last_trained_count
        return new_examples >= self._min_new_examples

    def train(self) -> TrainingMetrics:
        """Run a full batch STDP training sweep.

        Performs multiple epochs over the accumulated training data,
        measuring accuracy before and after.

        Returns:
            TrainingMetrics with accuracy and weight change data.
        """
        start = time.monotonic()

        collector = self._prm.collector
        network = self._prm.reward_network
        examples = collector.get_all()

        if not examples:
            return TrainingMetrics(timestamp=time.time())

        # Measure pre-training accuracy
        pre_accuracy = self._measure_accuracy(network, examples)

        # Snapshot weights before training
        pre_weights = self._snapshot_weights(network)

        # Run training epochs
        for epoch in range(self._epochs):
            # Process in batches
            for batch_start in range(0, len(examples), self._batch_size):
                batch = examples[batch_start:batch_start + self._batch_size]

                for example in batch:
                    self._train_example(network, example)

        # Measure post-training accuracy
        post_accuracy = self._measure_accuracy(network, examples)

        # Compute weight delta
        post_weights = self._snapshot_weights(network)
        weight_delta = self._compute_weight_delta(pre_weights, post_weights)

        self._last_trained_count = collector.count

        elapsed_ms = (time.monotonic() - start) * 1000

        metrics = TrainingMetrics(
            examples_trained=len(examples),
            epochs_completed=self._epochs,
            pre_accuracy=pre_accuracy,
            post_accuracy=post_accuracy,
            weight_delta=weight_delta,
            training_duration_ms=elapsed_ms,
            timestamp=time.time(),
        )

        self._history.append(metrics)

        logger.info(
            "PRMTrainer: %d examples × %d epochs, "
            "accuracy %.1f%% → %.1f%%, Δw=%.4f, %.1fms",
            len(examples),
            self._epochs,
            pre_accuracy * 100,
            post_accuracy * 100,
            weight_delta,
            elapsed_ms,
        )

        return metrics

    def get_metrics(self) -> list[TrainingMetrics]:
        """Get full training history."""
        return list(self._history)

    @property
    def last_metrics(self) -> TrainingMetrics | None:
        """Get the most recent training metrics, or None."""
        return self._history[-1] if self._history else None

    @property
    def total_sweeps(self) -> int:
        """Total number of training sweeps completed."""
        return len(self._history)

    def _measure_accuracy(self, network: Any, examples: list) -> float:
        """Measure prediction accuracy on examples.

        For each example, run the network and check if the accept/revise
        prediction matches the actual outcome.
        """
        if not examples:
            return 0.0

        correct = 0
        for example in examples:
            result = network.score(example.features)
            predicted_accept = result["accept_score"] > result["revise_score"]
            if predicted_accept == example.accepted:
                correct += 1

        return correct / len(examples)

    def _train_example(self, network: Any, example: Any) -> None:
        """Run one STDP training step on a single example."""
        network.reset()
        t = time.time()

        input_currents = [
            example.features.get("heuristic_relevance", 0.0),
            example.features.get("heuristic_coherence", 0.0),
            example.features.get("heuristic_completeness", 0.0),
            example.features.get("heuristic_conciseness", 0.0),
            example.features.get("goal_salience", 0.0),
            example.features.get("text_length_ratio", 0.0),
        ]

        # Supervised signal
        if example.accepted:
            output_bias = [0.5, 0.0]
        else:
            output_bias = [0.0, 0.5]

        for step in range(self._settle_steps):
            network.network.step(
                {
                    "input": input_currents,
                    "output": output_bias,
                },
                t + step * 0.001,
                learn=True,
            )

    def _snapshot_weights(self, network: Any) -> list[float]:
        """Capture all projection weights as a flat list."""
        weights = []
        for proj in network.network._projections:
            w_matrix = proj.get_weight_matrix()
            for row in w_matrix:
                weights.extend(row)
        return weights

    def _compute_weight_delta(
        self, before: list[float], after: list[float]
    ) -> float:
        """Compute total absolute weight change."""
        if len(before) != len(after):
            return 0.0
        return sum(abs(a - b) for a, b in zip(before, after))
