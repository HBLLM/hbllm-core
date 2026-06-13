"""
Trained Process Reward Model (PRM) — learnable fragment scoring.

Replaces the hardcoded ``RewardEvaluator`` heuristics with a trainable
SNN-based model that learns from experience which fragments are good.

Architecture:
    1. The existing ``RewardEvaluator`` heuristic scores become *input
       features* for the SNN (not replaced — augmented).
    2. A ``RewardNetwork`` (4-layer SpikingNetwork) learns to weight
       these signals based on downstream outcomes.
    3. A ``TrainingCollector`` accumulates (features, outcome) pairs
       for online STDP training.

Components:
    RewardNetwork      — 4-layer SNN for fragment quality prediction
    TrainedPRM         — wraps RewardEvaluator with learnable scoring
    TrainingCollector  — circular buffer for training examples

Warm-start:
    The SNN is initialized with weights that replicate the heuristic
    blend, so it starts performing identically to RewardEvaluator.
    STDP then nudges weights from experience.

Usage::

    from hbllm.brain.snn.expression.trained_prm import TrainedPRM
    from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator

    evaluator = RewardEvaluator(encoder=onnx_encode)
    prm = TrainedPRM(reward_evaluator=evaluator)

    fragment = prm.evaluate("generated text", goal)
    # ... downstream: user accepts or we revise
    prm.record_outcome(fragment, accepted=True)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hbllm.brain.snn.expression.models import ThoughtFragment, ThoughtGoal
from hbllm.brain.snn.expression.reward_evaluator import RewardEvaluator
from hbllm.brain.snn.lif import LIFConfig
from hbllm.brain.snn.network import NeuronLayer, SpikingNetwork

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# RewardNetwork — SNN for fragment quality prediction
# ═══════════════════════════════════════════════════════════════════════════


class RewardNetwork:
    """4-layer SNN for evaluating fragment quality.

    Layers:
        input (6 neurons):
            - heuristic_relevance, heuristic_coherence,
              heuristic_completeness, heuristic_conciseness
            - goal_salience, text_length_ratio

        hidden (8 neurons):
            Learns nonlinear feature combinations.

        quality (4 neurons):
            - neurons 0-1: high quality detectors
            - neurons 2-3: low quality detectors

        output (2 neurons):
            - neuron 0: accept (fragment is good)
            - neuron 1: revise (fragment needs revision)

    Args:
        stdp_rule: Optional STDP rule for all projections.
        settle_steps: Steps to let network settle per evaluation.
    """

    def __init__(
        self,
        stdp_rule: Any | None = None,
        settle_steps: int = 3,
    ) -> None:
        self._settle_steps = settle_steps
        self._network = SpikingNetwork(name="reward")

        # Input layer
        self._network.add_layer(
            NeuronLayer(
                name="input",
                neuron_count=6,
                config=LIFConfig(
                    threshold=0.25,
                    decay_half_life=0.3,
                    reset_potential=0.0,
                    refractory_period=0.01,
                ),
            )
        )

        # Hidden layer
        self._network.add_layer(
            NeuronLayer(
                name="hidden",
                neuron_count=8,
                config=LIFConfig(
                    threshold=0.4,
                    decay_half_life=0.5,
                    reset_potential=0.0,
                    refractory_period=0.02,
                ),
            )
        )

        # Quality layer
        self._network.add_layer(
            NeuronLayer(
                name="quality",
                neuron_count=4,
                config=LIFConfig(
                    threshold=0.45,
                    decay_half_life=0.4,
                    reset_potential=0.0,
                    refractory_period=0.02,
                ),
            )
        )

        # Output layer
        self._network.add_layer(
            NeuronLayer(
                name="output",
                neuron_count=2,
                config=LIFConfig(
                    threshold=0.35,
                    decay_half_life=0.4,
                    reset_potential=0.0,
                    refractory_period=0.01,
                ),
            )
        )

        # Input → Hidden projection
        # Warm-start: weight the heuristic features according to
        # RewardEvaluator's default weights (0.4, 0.2, 0.25, 0.15)
        # Features: [relevance, coherence, completeness, conciseness,
        #            goal_salience, text_length_ratio]
        input_to_hidden = [
            # relevance (weight 0.4) → strong to all hidden neurons
            [0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.10],
            # coherence (weight 0.2) → moderate
            [0.20, 0.25, 0.20, 0.15, 0.10, 0.10, 0.15, 0.10],
            # completeness (weight 0.25) → moderate-strong
            [0.25, 0.20, 0.30, 0.25, 0.15, 0.10, 0.10, 0.15],
            # conciseness (weight 0.15) → lighter
            [0.15, 0.10, 0.10, 0.15, 0.20, 0.10, 0.10, 0.10],
            # goal_salience → auxiliary signal
            [0.10, 0.15, 0.10, 0.10, 0.15, 0.20, 0.10, 0.10],
            # text_length_ratio → auxiliary signal
            [0.05, 0.10, 0.10, 0.05, 0.10, 0.10, 0.15, 0.20],
        ]

        self._network.connect(
            "input",
            "hidden",
            initial_weights=input_to_hidden,
            stdp_rule=stdp_rule,
        )

        # Hidden → Quality projection
        hidden_to_quality = [
            # Hidden neurons 0-3 → high quality detectors
            [0.5, 0.4, 0.1, 0.1],
            [0.4, 0.5, 0.1, 0.1],
            [0.3, 0.4, 0.2, 0.1],
            [0.3, 0.3, 0.2, 0.2],
            # Hidden neurons 4-7 → mixed
            [0.2, 0.2, 0.3, 0.3],
            [0.1, 0.1, 0.4, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.1, 0.4, 0.5],
        ]

        self._network.connect(
            "hidden",
            "quality",
            initial_weights=hidden_to_quality,
            stdp_rule=stdp_rule,
        )

        # Quality → Output projection
        quality_to_output = [
            # high quality → accept
            [0.6, 0.1],
            [0.5, 0.1],
            # low quality → revise
            [0.1, 0.5],
            [0.1, 0.6],
        ]

        self._network.connect(
            "quality",
            "output",
            initial_weights=quality_to_output,
            stdp_rule=stdp_rule,
        )

    def score(self, features: dict[str, float]) -> dict[str, float]:
        """Score a fragment's features through the network.

        Args:
            features: Dict with keys matching input neuron roles.

        Returns:
            Dict with 'accept_score', 'revise_score', 'reward' keys.
        """
        self._network.reset()

        input_currents = [
            features.get("heuristic_relevance", 0.0),
            features.get("heuristic_coherence", 0.0),
            features.get("heuristic_completeness", 0.0),
            features.get("heuristic_conciseness", 0.0),
            features.get("goal_salience", 0.0),
            features.get("text_length_ratio", 0.0),
        ]

        t = time.time()
        last_result: dict[str, list] = {}

        for step in range(self._settle_steps):
            last_result = self._network.step(
                {"input": input_currents},
                t + step * 0.001,
                learn=True,
            )

        # Extract output
        if "output" not in last_result:
            return {"accept_score": 0.5, "revise_score": 0.5, "reward": 0.5}

        out_spikes = last_result["output"]
        accept = out_spikes[0].strength if out_spikes[0].fired else 0.0
        revise = out_spikes[1].strength if out_spikes[1].fired else 0.0

        # Soft fallback from membrane potentials
        if accept == 0.0 and revise == 0.0:
            potentials = self._network.get_layer("output").get_potential_vector()
            threshold = self._network.get_layer("output").config.threshold
            accept = min(1.0, potentials[0] / max(0.01, threshold))
            revise = min(1.0, potentials[1] / max(0.01, threshold))

        total = accept + revise
        reward = accept / total if total > 0 else 0.5

        return {
            "accept_score": accept,
            "revise_score": revise,
            "reward": reward,
        }

    @property
    def network(self) -> SpikingNetwork:
        """Access the underlying SpikingNetwork."""
        return self._network

    def reset(self) -> None:
        """Reset network state."""
        self._network.reset()


# ═══════════════════════════════════════════════════════════════════════════
# TrainingCollector — accumulates training examples
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrainingExample:
    """A single training data point for the PRM."""

    features: dict[str, float] = field(default_factory=dict)
    accepted: bool = True
    reward_score: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "accepted": self.accepted,
            "reward_score": self.reward_score,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingExample:
        return cls(
            features=data.get("features", {}),
            accepted=data.get("accepted", True),
            reward_score=data.get("reward_score", 0.0),
            timestamp=data.get("timestamp", 0.0),
        )


class TrainingCollector:
    """Circular buffer for PRM training examples.

    Stores recent (features, outcome) pairs and provides batch
    retrieval for periodic STDP training sweeps.

    Args:
        max_size: Maximum number of examples to retain.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._max_size = max_size
        self._examples: list[TrainingExample] = []

    def record(
        self,
        features: dict[str, float],
        accepted: bool,
        reward_score: float = 0.0,
    ) -> None:
        """Record a training example.

        Args:
            features: Input features used for scoring.
            accepted: Whether the fragment was accepted or revised.
            reward_score: The reward score assigned by the evaluator.
        """
        example = TrainingExample(
            features=features,
            accepted=accepted,
            reward_score=reward_score,
            timestamp=time.time(),
        )
        self._examples.append(example)

        # Circular buffer: drop oldest when full
        if len(self._examples) > self._max_size:
            self._examples = self._examples[-self._max_size :]

    def get_recent(self, n: int = 50) -> list[TrainingExample]:
        """Get the N most recent examples."""
        return self._examples[-n:]

    def get_all(self) -> list[TrainingExample]:
        """Get all stored examples."""
        return list(self._examples)

    @property
    def count(self) -> int:
        """Number of stored examples."""
        return len(self._examples)

    @property
    def accept_rate(self) -> float:
        """Fraction of examples that were accepted."""
        if not self._examples:
            return 0.0
        accepted = sum(1 for e in self._examples if e.accepted)
        return accepted / len(self._examples)

    def save(self, path: str | Path) -> None:
        """Persist examples to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                [e.to_dict() for e in self._examples],
                f,
                indent=2,
            )
        logger.info("TrainingCollector saved %d examples to %s", len(self._examples), path)

    def load(self, path: str | Path) -> None:
        """Load examples from JSON."""
        path = Path(path)
        if not path.exists():
            logger.info("No training data at %s", path)
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._examples = [TrainingExample.from_dict(d) for d in data]
            logger.info("TrainingCollector loaded %d examples", len(self._examples))
        except Exception as e:
            logger.warning("Failed to load training data: %s", e)


# ═══════════════════════════════════════════════════════════════════════════
# TrainedPRM — wraps RewardEvaluator with learnable scoring
# ═══════════════════════════════════════════════════════════════════════════


class TrainedPRM:
    """SNN-trained Process Reward Model.

    Wraps the existing ``RewardEvaluator`` heuristics with a learnable
    ``RewardNetwork``.  The heuristic scores become input features for
    the SNN, which learns to predict fragment quality from experience.

    **Fallback behavior**: Until ``fallback_threshold`` training examples
    have been collected, the heuristic reward is blended with (or used
    instead of) the SNN reward.  This ensures the system works from day
    one while the SNN learns.

    Args:
        reward_evaluator: The base heuristic evaluator.
        reward_network: Optional pre-configured RewardNetwork.
            If None, a default is created.
        training_collector: Optional pre-loaded TrainingCollector.
        stdp_rule: Optional STDP rule for the RewardNetwork.
        fallback_threshold: Number of training examples needed before
            the SNN is trusted. Default 50.
        snn_blend_weight: Weight of SNN score in the blend.
            0.0 = pure heuristic, 1.0 = pure SNN.
            Ramps up linearly from 0.0 to this value as training
            data accumulates.
    """

    def __init__(
        self,
        reward_evaluator: RewardEvaluator,
        reward_network: RewardNetwork | None = None,
        training_collector: TrainingCollector | None = None,
        stdp_rule: Any | None = None,
        fallback_threshold: int = 50,
        snn_blend_weight: float = 0.6,
    ) -> None:
        self._heuristic = reward_evaluator
        self._network = reward_network or RewardNetwork(stdp_rule=stdp_rule)
        self._collector = training_collector or TrainingCollector()
        self._fallback_threshold = fallback_threshold
        self._snn_blend_weight = snn_blend_weight

        # Cache last features for outcome recording
        self._last_features: dict[str, float] = {}

    def evaluate(
        self,
        fragment_text: str,
        goal: ThoughtGoal,
        prev_fragment_text: str | None = None,
    ) -> ThoughtFragment:
        """Evaluate a fragment using heuristic + SNN hybrid.

        Args:
            fragment_text: The LLM-generated text.
            goal: The thought goal this fragment addresses.
            prev_fragment_text: Previous fragment for coherence.

        Returns:
            ThoughtFragment with blended reward score.
        """
        # Step 1: Get heuristic scores
        heuristic_fragment = self._heuristic.evaluate(fragment_text, goal, prev_fragment_text)

        # Step 2: Extract features for SNN
        features = self._extract_features(heuristic_fragment, goal, fragment_text)
        self._last_features = features

        # Step 3: Get SNN score
        snn_result = self._network.score(features)

        # Step 4: Blend based on training data availability
        blend = self._compute_blend_weight()
        heuristic_reward = heuristic_fragment.reward_score
        snn_reward = snn_result["reward"]
        blended_reward = (1.0 - blend) * heuristic_reward + blend * snn_reward

        # Build result fragment
        result = ThoughtFragment(
            goal_id=heuristic_fragment.goal_id,
            text=heuristic_fragment.text,
            reward_score=blended_reward,
            coherence_score=heuristic_fragment.coherence_score,
            relevance_score=heuristic_fragment.relevance_score,
            revision_count=heuristic_fragment.revision_count,
            metadata={
                **heuristic_fragment.metadata,
                "prm_heuristic_reward": heuristic_reward,
                "prm_snn_reward": snn_reward,
                "prm_blend_weight": blend,
                "prm_accept_score": snn_result["accept_score"],
                "prm_revise_score": snn_result["revise_score"],
                "prm_training_count": self._collector.count,
            },
        )

        return result

    def record_outcome(
        self,
        fragment: ThoughtFragment,
        accepted: bool,
    ) -> None:
        """Record whether a fragment was accepted or revised.

        This drives online STDP learning: the RewardNetwork sees
        what features correlate with accepted/revised outcomes.

        Args:
            fragment: The evaluated fragment.
            accepted: True if the fragment was accepted, False if revised.
        """
        features = self._last_features or self._extract_features_from_meta(fragment)

        self._collector.record(
            features=features,
            accepted=accepted,
            reward_score=fragment.reward_score,
        )

        # Run a training step: feed features through network with
        # the correct output emphasis
        self._train_step(features, accepted)

    def should_revise(self, fragment: ThoughtFragment) -> bool:
        """SNN-informed revision decision.

        Uses the blended reward and SNN accept/revise signals.
        """
        # Use the heuristic threshold as baseline
        if fragment.reward_score < self._heuristic.min_acceptable_reward:
            return True

        # If SNN has strong revise signal, trust it
        revise_score = fragment.metadata.get("prm_revise_score", 0.0)
        accept_score = fragment.metadata.get("prm_accept_score", 0.0)
        if revise_score > accept_score * 1.5:
            return True

        return False

    def _extract_features(
        self,
        fragment: ThoughtFragment,
        goal: ThoughtGoal,
        text: str,
    ) -> dict[str, float]:
        """Extract SNN input features from a fragment + goal."""
        estimated_tokens = max(1, len(text) // 4)
        budget = max(1, goal.max_tokens)

        return {
            "heuristic_relevance": fragment.relevance_score,
            "heuristic_coherence": fragment.coherence_score,
            "heuristic_completeness": fragment.metadata.get("completeness", 0.5),
            "heuristic_conciseness": fragment.metadata.get("conciseness", 0.5),
            "goal_salience": min(1.0, goal.salience),
            "text_length_ratio": min(2.0, estimated_tokens / budget) / 2.0,
        }

    def _extract_features_from_meta(self, fragment: ThoughtFragment) -> dict[str, float]:
        """Fallback feature extraction from fragment metadata."""
        return {
            "heuristic_relevance": fragment.relevance_score,
            "heuristic_coherence": fragment.coherence_score,
            "heuristic_completeness": fragment.metadata.get("completeness", 0.5),
            "heuristic_conciseness": fragment.metadata.get("conciseness", 0.5),
            "goal_salience": fragment.metadata.get("goal_salience", 0.5),
            "text_length_ratio": fragment.metadata.get("text_length_ratio", 0.5),
        }

    def _compute_blend_weight(self) -> float:
        """Compute the SNN blend weight based on training data.

        Ramps linearly from 0.0 to snn_blend_weight as training
        examples accumulate up to fallback_threshold.
        """
        if self._collector.count >= self._fallback_threshold:
            return self._snn_blend_weight

        # Linear ramp
        progress = self._collector.count / max(1, self._fallback_threshold)
        return self._snn_blend_weight * progress

    def _train_step(
        self,
        features: dict[str, float],
        accepted: bool,
    ) -> None:
        """Single STDP training step on the RewardNetwork.

        Feeds features with a biased output signal: if accepted,
        the accept neuron gets extra current; if revised, the
        revise neuron does.
        """
        self._network.reset()
        t = time.time()

        input_currents = [
            features.get("heuristic_relevance", 0.0),
            features.get("heuristic_coherence", 0.0),
            features.get("heuristic_completeness", 0.0),
            features.get("heuristic_conciseness", 0.0),
            features.get("goal_salience", 0.0),
            features.get("text_length_ratio", 0.0),
        ]

        # Bias the output layer based on the true outcome
        # This creates a supervised signal for STDP
        if accepted:
            output_bias = [0.5, 0.0]  # push accept neuron
        else:
            output_bias = [0.0, 0.5]  # push revise neuron

        for step in range(3):
            self._network.network.step(
                {
                    "input": input_currents,
                    "output": output_bias,
                },
                t + step * 0.001,
                learn=True,
            )

    @property
    def reward_network(self) -> RewardNetwork:
        """Access the RewardNetwork."""
        return self._network

    @property
    def collector(self) -> TrainingCollector:
        """Access the TrainingCollector."""
        return self._collector

    @property
    def heuristic(self) -> RewardEvaluator:
        """Access the base heuristic evaluator."""
        return self._heuristic
