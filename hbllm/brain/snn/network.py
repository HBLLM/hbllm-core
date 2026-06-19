"""
Multi-Layer Spiking Neural Network Framework.

Provides generic building blocks for constructing deep spiking networks
with learnable inter-layer connections.  This module is the foundation
for the v3 roadmap: enabling the SNN to perform reasoning across
multiple layers rather than single-layer segmentation/gating.

Core components:
    NeuronLayer      — a named group of LIF neurons with a shared role
    LayerProjection  — a weighted, STDP-capable connection between layers
    SpikingNetwork   — orchestrates multi-layer spike propagation

Design principles:
    1. **Generic**: Not tied to comprehension or expression — any layer
       topology can be built (feedforward, recurrent, lateral).
    2. **STDP-ready**: Every LayerProjection can optionally learn via
       the existing STDPRule from ``plasticity.py``.
    3. **Topological propagation**: Layers are stepped in dependency order
       so spikes flow correctly from input → intermediate → output.
    4. **Serializable**: Full network state (weights, potentials) can be
       persisted and restored.

Usage::

    from hbllm.brain.snn.network import NeuronLayer, LayerProjection, SpikingNetwork
    from hbllm.brain.snn.lif import LIFConfig

    net = SpikingNetwork("reasoning")
    net.add_layer(NeuronLayer("input", 4, LIFConfig(threshold=0.5)))
    net.add_layer(NeuronLayer("hidden", 8, LIFConfig(threshold=0.7)))
    net.add_layer(NeuronLayer("output", 3, LIFConfig(threshold=0.6)))
    net.connect("input", "hidden")
    net.connect("hidden", "output")

    results = net.step({"input": [0.3, 0.8, 0.1, 0.9]}, timestamp)
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hbllm.brain.snn.lif import LIFConfig, LIFNeuron, SpikeEvent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# NeuronLayer — a group of LIF neurons
# ═══════════════════════════════════════════════════════════════════════════


class NeuronLayer:
    """A named group of LIF neurons with a shared semantic role.

    Each neuron in the layer has an independent membrane potential
    but shares the same LIF configuration (threshold, decay, etc.).

    Args:
        name: Unique identifier for this layer (e.g. ``"association"``).
        neuron_count: Number of neurons in this layer.
        config: Shared LIF configuration for all neurons.
    """

    def __init__(
        self,
        name: str,
        neuron_count: int,
        config: LIFConfig,
    ) -> None:
        self.name = name
        self.neuron_count = neuron_count
        self.config = config
        self.neurons: list[LIFNeuron] = [
            LIFNeuron(
                config=LIFConfig(
                    threshold=config.threshold,
                    decay_half_life=config.decay_half_life,
                    reset_potential=config.reset_potential,
                    refractory_period=config.refractory_period,
                ),
                neuron_id=f"{name}.{i}",
            )
            for i in range(neuron_count)
        ]
        self._last_spikes: list[SpikeEvent] = []

    def step(self, currents: list[float], timestamp: float) -> list[SpikeEvent]:
        """Step all neurons with their individual input currents.

        Args:
            currents: Input current for each neuron. Must have length
                ``neuron_count``. Use 0.0 for neurons with no input.
            timestamp: Current timestamp in seconds.

        Returns:
            List of SpikeEvent for each neuron.

        Raises:
            ValueError: If ``len(currents) != neuron_count``.
        """
        if len(currents) != self.neuron_count:
            raise ValueError(
                f"Layer '{self.name}' expects {self.neuron_count} currents, got {len(currents)}"
            )

        spikes = [
            neuron.step(current, timestamp) for neuron, current in zip(self.neurons, currents)
        ]
        self._last_spikes = spikes
        return spikes

    def get_spike_vector(self) -> list[bool]:
        """Binary spike vector from the last step.

        Returns:
            List of bools, True where the neuron fired.
        """
        return [s.fired for s in self._last_spikes]

    def get_strength_vector(self) -> list[float]:
        """Spike strength vector from the last step.

        Returns:
            List of floats; 0.0 where the neuron didn't fire.
        """
        return [s.strength if s.fired else 0.0 for s in self._last_spikes]

    def get_potential_vector(self) -> list[float]:
        """Current membrane potentials of all neurons."""
        return [n.v for n in self.neurons]

    def fired_count(self) -> int:
        """Number of neurons that fired in the last step."""
        return sum(1 for s in self._last_spikes if s.fired)

    def reset(self) -> None:
        """Reset all neuron states."""
        for neuron in self.neurons:
            neuron.reset_state()
        self._last_spikes = []


# ═══════════════════════════════════════════════════════════════════════════
# LayerProjection — weighted connection between layers
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _ProjectionWeight:
    """Internal weight tracking for a single source→target connection."""

    weight: float = 0.0
    base_weight: float = 0.0
    last_pre_time: float = 0.0
    last_post_time: float = 0.0
    update_count: int = 0


class LayerProjection:
    """Weighted projection from a source layer to a target layer.

    Maps each source neuron's spike output to each target neuron's
    input current via a weight matrix.  Supports optional STDP
    plasticity for learnable connections.

    The weight matrix has shape ``[source_size × target_size]``:
    ``weights[i][j]`` is the connection from source neuron *i*
    to target neuron *j*.

    Args:
        source_name: Name of the source layer.
        target_name: Name of the target layer.
        source_size: Number of neurons in the source layer.
        target_size: Number of neurons in the target layer.
        initial_weights: Optional weight matrix. If None, initialized
            with uniform weights ``1.0 / source_size``.
        stdp_rule: Optional STDP rule for learning. If None, weights
            are fixed.
    """

    def __init__(
        self,
        source_name: str,
        target_name: str,
        source_size: int,
        target_size: int,
        initial_weights: list[list[float]] | None = None,
        stdp_rule: Any | None = None,
    ) -> None:
        self.source_name = source_name
        self.target_name = target_name
        self.source_size = source_size
        self.target_size = target_size
        self._stdp_rule = stdp_rule
        self._global_step = 0

        # Initialize weight matrix
        if initial_weights is not None:
            if len(initial_weights) != source_size:
                raise ValueError(
                    f"Weight matrix rows ({len(initial_weights)}) != source_size ({source_size})"
                )
            self._weights: list[list[_ProjectionWeight]] = []
            for i, row in enumerate(initial_weights):
                if len(row) != target_size:
                    raise ValueError(
                        f"Weight matrix row {i} cols ({len(row)}) != target_size ({target_size})"
                    )
                self._weights.append([_ProjectionWeight(weight=w, base_weight=w) for w in row])
        else:
            # Uniform initialization
            default_w = 1.0 / max(1, source_size)
            self._weights = [
                [
                    _ProjectionWeight(weight=default_w, base_weight=default_w)
                    for _ in range(target_size)
                ]
                for _ in range(source_size)
            ]

    def project(self, source_spikes: list[SpikeEvent], timestamp: float) -> list[float]:
        """Convert source spikes to target input currents.

        For each target neuron *j*, computes::

            current_j = Σ_i (fired_i × strength_i × weight_i_j)

        Args:
            source_spikes: Spike events from the source layer.
            timestamp: Current timestamp.

        Returns:
            List of input currents for each target neuron.
        """
        self._global_step += 1
        target_currents = [0.0] * self.target_size

        for i, spike in enumerate(source_spikes):
            if not spike.fired:
                continue
            for j in range(self.target_size):
                pw = self._weights[i][j]
                target_currents[j] += spike.strength * pw.weight
                pw.last_pre_time = timestamp

        return target_currents

    def apply_stdp(
        self,
        source_spikes: list[SpikeEvent],
        target_spikes: list[SpikeEvent],
        timestamp: float,
    ) -> int:
        """Apply STDP learning between source and target spikes.

        Args:
            source_spikes: Spike events from the source layer.
            target_spikes: Spike events from the target layer.
            timestamp: Current timestamp.

        Returns:
            Number of weight updates applied.
        """
        if self._stdp_rule is None:
            return 0

        updates = 0
        for i, src in enumerate(source_spikes):
            for j, tgt in enumerate(target_spikes):
                pw = self._weights[i][j]

                if src.fired:
                    pw.last_pre_time = timestamp
                if tgt.fired:
                    pw.last_post_time = timestamp

                if not (src.fired or tgt.fired):
                    continue
                if pw.last_pre_time == 0.0 or pw.last_post_time == 0.0:
                    continue

                dt = pw.last_post_time - pw.last_pre_time
                if abs(dt) > self._stdp_rule.time_constant * 5:
                    continue

                if dt > 0:
                    delta = self._stdp_rule.learning_rate * math.exp(
                        -abs(dt) / self._stdp_rule.time_constant
                    )
                elif dt < 0:
                    delta = -self._stdp_rule.learning_rate * math.exp(
                        -abs(dt) / self._stdp_rule.time_constant
                    )
                else:
                    delta = self._stdp_rule.learning_rate * 0.5

                old = pw.weight
                pw.weight = max(
                    self._stdp_rule.w_min,
                    min(self._stdp_rule.w_max, pw.weight + delta),
                )
                if pw.weight != old:
                    pw.update_count += 1
                    updates += 1

        return updates

    def get_weight_matrix(self) -> list[list[float]]:
        """Get the current weight matrix as plain floats."""
        return [[pw.weight for pw in row] for row in self._weights]

    def get_total_updates(self) -> int:
        """Total STDP updates across all connections."""
        return sum(pw.update_count for row in self._weights for pw in row)

    def to_dict(self) -> dict[str, Any]:
        """Serialize projection for persistence."""
        return {
            "source_name": self.source_name,
            "target_name": self.target_name,
            "source_size": self.source_size,
            "target_size": self.target_size,
            "weights": [
                [
                    {
                        "weight": pw.weight,
                        "base_weight": pw.base_weight,
                        "update_count": pw.update_count,
                    }
                    for pw in row
                ]
                for row in self._weights
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], stdp_rule: Any | None = None) -> LayerProjection:
        """Restore from persisted state."""
        proj = cls(
            source_name=data["source_name"],
            target_name=data["target_name"],
            source_size=data["source_size"],
            target_size=data["target_size"],
            stdp_rule=stdp_rule,
        )
        for i, row_data in enumerate(data.get("weights", [])):
            for j, w_data in enumerate(row_data):
                if i < proj.source_size and j < proj.target_size:
                    pw = proj._weights[i][j]
                    pw.weight = w_data.get("weight", pw.weight)
                    pw.base_weight = w_data.get("base_weight", pw.base_weight)
                    pw.update_count = w_data.get("update_count", 0)
        return proj


# ═══════════════════════════════════════════════════════════════════════════
# SpikingNetwork — multi-layer orchestrator
# ═══════════════════════════════════════════════════════════════════════════


class SpikingNetwork:
    """A multi-layer spiking neural network.

    Manages layers and projections, propagating spikes from input
    layers through intermediate layers to output layers.  Layers are
    stepped in topological order based on their connections.

    Supports:
    - **Feedforward**: input → hidden → output
    - **Lateral**: connections within the same layer
    - **STDP learning**: on any projection

    Args:
        name: Human-readable name for this network.
    """

    def __init__(self, name: str = "spiking_network") -> None:
        self.name = name
        self._layers: dict[str, NeuronLayer] = {}
        self._projections: list[LayerProjection] = []
        self._layer_order: list[str] | None = None  # cached topo sort
        self._step_count = 0

    def add_layer(self, layer: NeuronLayer) -> None:
        """Add a neuron layer to the network.

        Args:
            layer: The NeuronLayer to add.

        Raises:
            ValueError: If a layer with the same name already exists.
        """
        if layer.name in self._layers:
            raise ValueError(f"Layer '{layer.name}' already exists")
        self._layers[layer.name] = layer
        self._layer_order = None  # invalidate cache

    def connect(
        self,
        source_name: str,
        target_name: str,
        initial_weights: list[list[float]] | None = None,
        stdp_rule: Any | None = None,
    ) -> LayerProjection:
        """Create a projection between two layers.

        Args:
            source_name: Name of the source layer.
            target_name: Name of the target layer.
            initial_weights: Optional weight matrix.
            stdp_rule: Optional STDP rule for learning.

        Returns:
            The created LayerProjection.

        Raises:
            ValueError: If either layer doesn't exist.
        """
        if source_name not in self._layers:
            raise ValueError(f"Source layer '{source_name}' not found")
        if target_name not in self._layers:
            raise ValueError(f"Target layer '{target_name}' not found")

        source = self._layers[source_name]
        target = self._layers[target_name]

        proj = LayerProjection(
            source_name=source_name,
            target_name=target_name,
            source_size=source.neuron_count,
            target_size=target.neuron_count,
            initial_weights=initial_weights,
            stdp_rule=stdp_rule,
        )
        self._projections.append(proj)
        self._layer_order = None  # invalidate cache
        return proj

    def _get_layer_order(self) -> list[str]:
        """Compute topological order of layers.

        Layers with no incoming connections are processed first.
        Handles lateral (self) connections gracefully.
        """
        if self._layer_order is not None:
            return self._layer_order

        # Build adjacency: which layers feed into which
        incoming: dict[str, set[str]] = {name: set() for name in self._layers}
        for proj in self._projections:
            # Skip self-connections for ordering purposes
            if proj.source_name != proj.target_name:
                incoming[proj.target_name].add(proj.source_name)

        # Kahn's algorithm
        order: list[str] = []
        ready = [n for n, deps in incoming.items() if len(deps) == 0]

        while ready:
            node = ready.pop(0)
            order.append(node)
            for name, deps in incoming.items():
                if node in deps:
                    deps.remove(node)
                    if len(deps) == 0 and name not in order:
                        ready.append(name)

        # Add any remaining layers (cycles — shouldn't happen in feedforward)
        for name in self._layers:
            if name not in order:
                order.append(name)

        self._layer_order = order
        return order

    def step(
        self,
        input_currents: dict[str, list[float]],
        timestamp: float,
        learn: bool = True,
    ) -> dict[str, list[SpikeEvent]]:
        """Propagate spikes through all layers.

        Layers are stepped in topological order.  For each layer:
        1. Collect input currents (external + from projections)
        2. Step all neurons
        3. Record spikes for downstream projections
        4. Apply STDP if learning is enabled

        Args:
            input_currents: External input for specific layers.
                Format: ``{layer_name: [current_per_neuron]}``.
                Layers not in this dict receive only projected input.
            timestamp: Current timestamp in seconds.
            learn: Whether to apply STDP updates.

        Returns:
            Dict mapping layer names to their spike events.
        """
        self._step_count += 1
        layer_order = self._get_layer_order()
        all_spikes: dict[str, list[SpikeEvent]] = {}

        for layer_name in layer_order:
            layer = self._layers[layer_name]

            # Start with external input or zeros
            if layer_name in input_currents:
                currents = list(input_currents[layer_name])
                # Pad if needed
                while len(currents) < layer.neuron_count:
                    currents.append(0.0)
                currents = currents[: layer.neuron_count]
            else:
                currents = [0.0] * layer.neuron_count

            # Add projected input from upstream layers
            for proj in self._projections:
                if proj.target_name != layer_name:
                    continue
                if proj.source_name not in all_spikes:
                    continue  # source hasn't been stepped yet
                projected = proj.project(all_spikes[proj.source_name], timestamp)
                for j in range(layer.neuron_count):
                    currents[j] += projected[j]

            # Step the layer
            spikes = layer.step(currents, timestamp)
            all_spikes[layer_name] = spikes

        # Apply STDP learning on all projections
        if learn:
            for proj in self._projections:
                if proj.source_name in all_spikes and proj.target_name in all_spikes:
                    proj.apply_stdp(
                        all_spikes[proj.source_name],
                        all_spikes[proj.target_name],
                        timestamp,
                    )

        return all_spikes

    def get_layer(self, name: str) -> NeuronLayer:
        """Get a layer by name.

        Raises:
            KeyError: If the layer doesn't exist.
        """
        return self._layers[name]

    def get_all_spikes(self) -> dict[str, list[bool]]:
        """Get binary spike vectors for all layers from last step."""
        return {name: layer.get_spike_vector() for name, layer in self._layers.items()}

    @property
    def step_count(self) -> int:
        """Total steps executed."""
        return self._step_count

    @property
    def layer_names(self) -> list[str]:
        """Names of all layers in topological order."""
        return self._get_layer_order()

    def get_total_updates(self) -> int:
        """Total STDP updates across all projections."""
        return sum(p.get_total_updates() for p in self._projections)

    def reset(self) -> None:
        """Reset all layers and step counter."""
        for layer in self._layers.values():
            layer.reset()
        self._step_count = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full network for persistence."""
        return {
            "name": self.name,
            "version": 1,
            "step_count": self._step_count,
            "layers": [
                {
                    "name": layer.name,
                    "neuron_count": layer.neuron_count,
                    "config": {
                        "threshold": layer.config.threshold,
                        "decay_half_life": layer.config.decay_half_life,
                        "reset_potential": layer.config.reset_potential,
                        "refractory_period": layer.config.refractory_period,
                    },
                }
                for layer in self._layers.values()
            ],
            "projections": [p.to_dict() for p in self._projections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], stdp_rule: Any | None = None) -> SpikingNetwork:
        """Restore from persisted state."""
        net = cls(name=data.get("name", "spiking_network"))
        net._step_count = data.get("step_count", 0)

        for layer_data in data.get("layers", []):
            config_d = layer_data.get("config", {})
            config = LIFConfig(
                threshold=config_d.get("threshold", 1.0),
                decay_half_life=config_d.get("decay_half_life", 1.0),
                reset_potential=config_d.get("reset_potential", 0.0),
                refractory_period=config_d.get("refractory_period", 0.0),
            )
            layer = NeuronLayer(
                name=layer_data["name"],
                neuron_count=layer_data["neuron_count"],
                config=config,
            )
            net.add_layer(layer)

        for proj_data in data.get("projections", []):
            proj = LayerProjection.from_dict(proj_data, stdp_rule)
            net._projections.append(proj)

        return net

    def save(self, path: str | Path) -> None:
        """Save network to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(
            "SpikingNetwork '%s' saved: %d layers, %d projections, step %d",
            self.name,
            len(self._layers),
            len(self._projections),
            self._step_count,
        )

    @classmethod
    def load(cls, path: str | Path, stdp_rule: Any | None = None) -> SpikingNetwork:
        """Load network from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.info("No persisted network at %s", path)
            return cls()
        try:
            with open(path) as f:
                data = json.load(f)
            return cls.from_dict(data, stdp_rule)
        except Exception as e:
            logger.warning("Failed to load network: %s", e)
            return cls()
