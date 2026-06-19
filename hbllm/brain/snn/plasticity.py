"""
Synaptic Plasticity — STDP for learnable SNN connections.

Implements Spike-Timing Dependent Plasticity (STDP) as the learning rule
for synaptic connections between signal sources and LIF neurons.

Core components:
    SynapticConnection — a single learnable weight (source → target)
    STDPRule           — the timing-based weight update rule
    PlasticWeightMatrix — manages all connections for a neuron ensemble

This module lives at the ``snn/`` level because it is shared by both
the comprehension ensemble and the expression controller.

Design principles:
    1. STDP is biologically plausible: causal firing strengthens,
       anti-causal firing weakens.
    2. Weight bounds prevent runaway potentiation (excitatory only in v1).
    3. Unused connections decay slowly to prevent stale weights.
    4. Fully serializable for persistence across restarts.
    5. Optional — ensembles work without plasticity (backward compatible).

Usage::

    from hbllm.brain.snn.plasticity import STDPRule, PlasticWeightMatrix

    static_weights = {
        "entity": {"semantic_weight": 0.5, "topic_shift": 0.3},
        "clause": {"punctuation": 0.3, "buffer_pressure": 0.3},
    }
    rule = STDPRule(learning_rate=0.01)
    matrix = PlasticWeightMatrix(static_weights, rule)

    # During ensemble step:
    weights = matrix.get_weights("entity")  # returns learned weights
    matrix.record_signals(signals, timestamp)
    matrix.record_spikes(["entity"], timestamp)  # triggers STDP update
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SynapticConnection — a single learnable weight
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SynapticConnection:
    """A single learnable synaptic weight between a source signal and target neuron.

    Tracks spike timing for STDP updates.

    Attributes:
        source: Signal name (e.g. ``"semantic_weight"``).
        target: Neuron/channel name (e.g. ``"clause"``).
        weight: Current synaptic weight.
        base_weight: Original static weight (for reference/reset).
        last_pre_time: Last time the source signal was strongly active.
        last_post_time: Last time the target neuron spiked.
        update_count: Total STDP updates applied to this connection.
        last_reinforced_step: Global step count when last reinforced.
    """

    source: str = ""
    target: str = ""
    weight: float = 0.0
    base_weight: float = 0.0
    last_pre_time: float = 0.0
    last_post_time: float = 0.0
    update_count: int = 0
    last_reinforced_step: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for persistence."""
        return {
            "source": self.source,
            "target": self.target,
            "weight": self.weight,
            "base_weight": self.base_weight,
            "last_pre_time": self.last_pre_time,
            "last_post_time": self.last_post_time,
            "update_count": self.update_count,
            "last_reinforced_step": self.last_reinforced_step,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SynapticConnection:
        """Restore from persisted dict."""
        return cls(
            source=d.get("source", ""),
            target=d.get("target", ""),
            weight=d.get("weight", 0.0),
            base_weight=d.get("base_weight", 0.0),
            last_pre_time=d.get("last_pre_time", 0.0),
            last_post_time=d.get("last_post_time", 0.0),
            update_count=d.get("update_count", 0),
            last_reinforced_step=d.get("last_reinforced_step", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════
# STDPRule — the learning rule
# ═══════════════════════════════════════════════════════════════════════════


class STDPRule:
    """Spike-Timing Dependent Plasticity learning rule.

    Updates synaptic weights based on the relative timing of pre-synaptic
    (source signal) and post-synaptic (target neuron spike) activity.

    - **Causal** (pre fires before post): *strengthen* (Δw > 0)
      The signal helped cause the spike → reward the connection.

    - **Anti-causal** (post fires before pre): *weaken* (Δw < 0)
      The signal wasn't needed for the spike → punish the connection.

    The update magnitude decays exponentially with the time difference::

        Δw = η × exp(-|Δt| / τ) × sign(t_post - t_pre)

    Args:
        learning_rate: Step size for weight updates (η). Default 0.01.
        time_constant: STDP time window in seconds (τ). Default 0.5.
        w_min: Minimum allowed weight. Default 0.0 (no inhibition in v1).
        w_max: Maximum allowed weight. Default 2.0.
        pre_threshold: Minimum signal strength to count as "active". Default 0.1.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        time_constant: float = 0.5,
        w_min: float = 0.0,
        w_max: float = 2.0,
        pre_threshold: float = 0.1,
    ) -> None:
        self.learning_rate = learning_rate
        self.time_constant = time_constant
        self.w_min = w_min
        self.w_max = w_max
        self.pre_threshold = pre_threshold

    def update(
        self,
        conn: SynapticConnection,
        pre_active: bool,
        post_fired: bool,
        timestamp: float,
        global_step: int = 0,
    ) -> float:
        """Apply STDP update to a connection.

        Should be called after each ensemble step with the current
        pre/post activity state.

        Args:
            conn: The synaptic connection to update.
            pre_active: Whether the source signal was active (above threshold).
            post_fired: Whether the target neuron spiked.
            timestamp: Current timestamp in seconds.
            global_step: Global step counter for decay tracking.

        Returns:
            The weight delta applied (can be 0.0 if no update).
        """
        # Update timing records
        if pre_active:
            conn.last_pre_time = timestamp
        if post_fired:
            conn.last_post_time = timestamp

        # STDP only triggers when we have both a recent pre and post event
        if not (pre_active or post_fired):
            return 0.0

        # Need both pre and post times to compute Δt
        if conn.last_pre_time == 0.0 or conn.last_post_time == 0.0:
            return 0.0

        # Compute timing difference
        dt = conn.last_post_time - conn.last_pre_time

        # Only update if the events are within the STDP window
        if abs(dt) > self.time_constant * 5:
            # Too far apart — no meaningful correlation
            return 0.0

        # STDP kernel: exponential decay with sign from timing
        if dt > 0:
            # Causal: pre before post → potentiate
            delta = self.learning_rate * math.exp(-abs(dt) / self.time_constant)
        elif dt < 0:
            # Anti-causal: post before pre → depress
            delta = -self.learning_rate * math.exp(-abs(dt) / self.time_constant)
        else:
            # Simultaneous — small potentiation
            delta = self.learning_rate * 0.5

        # Apply update
        old_weight = conn.weight
        conn.weight = max(self.w_min, min(self.w_max, conn.weight + delta))
        actual_delta = conn.weight - old_weight

        if actual_delta != 0.0:
            conn.update_count += 1
            conn.last_reinforced_step = global_step

        return actual_delta

    def to_dict(self) -> dict[str, Any]:
        """Serialize rule parameters."""
        return {
            "learning_rate": self.learning_rate,
            "time_constant": self.time_constant,
            "w_min": self.w_min,
            "w_max": self.w_max,
            "pre_threshold": self.pre_threshold,
        }


# ═══════════════════════════════════════════════════════════════════════════
# PlasticWeightMatrix — ensemble-level weight management
# ═══════════════════════════════════════════════════════════════════════════


class PlasticWeightMatrix:
    """Manages learnable weights for a set of signal→neuron connections.

    Wraps the static ``_signal_weights`` dict from ``ComprehensionEnsemble``
    (or ``ThoughtController``) with STDP-capable connections.  Falls back to
    static weights for any connections not yet tracked.

    Args:
        static_weights: The original hardcoded weight dict.
            Format: ``{channel: {signal: weight, ...}, ...}``.
        stdp_rule: The STDP learning rule to apply.
        decay_interval: Apply unused weight decay every N steps.
        decay_rate: Fraction to decay per interval (default 0.01 = 1%).
    """

    def __init__(
        self,
        static_weights: dict[str, dict[str, float]],
        stdp_rule: STDPRule,
        decay_interval: int = 100,
        decay_rate: float = 0.01,
    ) -> None:
        self._static_weights = static_weights
        self._stdp_rule = stdp_rule
        self._decay_interval = decay_interval
        self._decay_rate = decay_rate
        self._global_step = 0

        # Initialize SynapticConnections from static weights
        self._connections: dict[str, dict[str, SynapticConnection]] = {}
        for channel, signals in static_weights.items():
            self._connections[channel] = {}
            for signal, weight in signals.items():
                self._connections[channel][signal] = SynapticConnection(
                    source=signal,
                    target=channel,
                    weight=weight,
                    base_weight=weight,
                )

        # Track recent signal activity for STDP timing
        self._last_signals: dict[str, float] = {}

    @property
    def global_step(self) -> int:
        """Current global step count."""
        return self._global_step

    def get_weights(self, channel: str) -> dict[str, float]:
        """Get current weights for a channel.

        Returns learned weights if available, static weights as fallback.

        Args:
            channel: The neuron channel name (e.g. ``"clause"``).

        Returns:
            Dict mapping signal names to their current weights.
        """
        if channel in self._connections:
            return {signal: conn.weight for signal, conn in self._connections[channel].items()}

        # Fallback to static weights
        return dict(self._static_weights.get(channel, {}))

    def record_signals(self, signals: dict[str, float], timestamp: float) -> None:
        """Record which signals were active (pre-synaptic activity).

        Call this *before* ``record_spikes()`` in each ensemble step.

        Args:
            signals: The signal dict from ``LexicalSignals.compute()``.
            timestamp: Current timestamp.
        """
        self._last_signals = dict(signals)
        self._global_step += 1

        # Update pre-synaptic timing for all connections
        for channel_conns in self._connections.values():
            for signal, conn in channel_conns.items():
                signal_val = signals.get(signal, 0.0)
                if signal_val >= self._stdp_rule.pre_threshold:
                    conn.last_pre_time = timestamp

    def record_spikes(
        self,
        fired_channels: list[str],
        timestamp: float,
    ) -> dict[str, float]:
        """Record which neurons fired and trigger STDP updates.

        Call this *after* ``record_signals()`` and the ensemble step.

        Args:
            fired_channels: List of channel names that spiked.
            timestamp: Current timestamp.

        Returns:
            Dict of ``{channel.signal: delta}`` for all weight updates.
        """
        all_deltas: dict[str, float] = {}

        for channel in fired_channels:
            if channel not in self._connections:
                continue

            for signal, conn in self._connections[channel].items():
                signal_val = self._last_signals.get(signal, 0.0)
                pre_active = signal_val >= self._stdp_rule.pre_threshold

                delta = self._stdp_rule.update(
                    conn=conn,
                    pre_active=pre_active,
                    post_fired=True,
                    timestamp=timestamp,
                    global_step=self._global_step,
                )

                if delta != 0.0:
                    key = f"{channel}.{signal}"
                    all_deltas[key] = delta

        # Also update connections for channels that did NOT fire
        # (anti-causal depression if signal was active but neuron didn't fire)
        non_fired = set(self._connections.keys()) - set(fired_channels)
        for channel in non_fired:
            for signal, conn in self._connections[channel].items():
                signal_val = self._last_signals.get(signal, 0.0)
                pre_active = signal_val >= self._stdp_rule.pre_threshold

                if pre_active:
                    # Signal was active but neuron didn't fire — mild depression
                    delta = self._stdp_rule.update(
                        conn=conn,
                        pre_active=True,
                        post_fired=False,
                        timestamp=timestamp,
                        global_step=self._global_step,
                    )
                    if delta != 0.0:
                        key = f"{channel}.{signal}"
                        all_deltas[key] = delta

        # Periodic decay of unused connections
        if self._global_step % self._decay_interval == 0:
            self._decay_unused()

        return all_deltas

    def _decay_unused(self) -> None:
        """Decay weights that haven't been reinforced recently.

        Weights drift toward their base (static) values if not actively
        reinforced by STDP updates.  This prevents stale learned weights
        from dominating when input patterns change.
        """
        for channel_conns in self._connections.values():
            for conn in channel_conns.values():
                steps_since = self._global_step - conn.last_reinforced_step
                if steps_since > self._decay_interval:
                    # Nudge toward base weight
                    diff = conn.base_weight - conn.weight
                    conn.weight += diff * self._decay_rate

    def get_weight_drift(self) -> dict[str, float]:
        """Get the total drift from static weights for each connection.

        Useful for monitoring how much the SNN has learned.

        Returns:
            Dict of ``{channel.signal: drift}`` where drift = learned - static.
        """
        drift: dict[str, float] = {}
        for channel, conns in self._connections.items():
            for signal, conn in conns.items():
                d = conn.weight - conn.base_weight
                if abs(d) > 1e-6:
                    drift[f"{channel}.{signal}"] = round(d, 6)
        return drift

    def get_total_updates(self) -> int:
        """Get total STDP update count across all connections."""
        return sum(
            conn.update_count for conns in self._connections.values() for conn in conns.values()
        )

    def reset_to_static(self) -> None:
        """Reset all weights to their original static values."""
        for channel_conns in self._connections.values():
            for conn in channel_conns.values():
                conn.weight = conn.base_weight
                conn.update_count = 0
                conn.last_reinforced_step = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full weight matrix for persistence.

        Returns a JSON-compatible dict containing all connections,
        the STDP rule parameters, and global state.
        """
        connections: dict[str, dict[str, Any]] = {}
        for channel, conns in self._connections.items():
            connections[channel] = {signal: conn.to_dict() for signal, conn in conns.items()}

        return {
            "version": 1,
            "global_step": self._global_step,
            "stdp_rule": self._stdp_rule.to_dict(),
            "connections": connections,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        static_weights: dict[str, dict[str, float]],
        stdp_rule: STDPRule | None = None,
    ) -> PlasticWeightMatrix:
        """Restore from persisted state.

        Args:
            data: Dict from ``to_dict()``.
            static_weights: The current static weight defaults
                (in case new connections were added since persistence).
            stdp_rule: Override STDP rule (if None, uses the persisted one).

        Returns:
            Restored PlasticWeightMatrix with learned weights.
        """
        if stdp_rule is None:
            rule_data = data.get("stdp_rule", {})
            stdp_rule = STDPRule(
                learning_rate=rule_data.get("learning_rate", 0.01),
                time_constant=rule_data.get("time_constant", 0.5),
                w_min=rule_data.get("w_min", 0.0),
                w_max=rule_data.get("w_max", 2.0),
                pre_threshold=rule_data.get("pre_threshold", 0.1),
            )

        matrix = cls(static_weights, stdp_rule)
        matrix._global_step = data.get("global_step", 0)

        # Restore learned connection weights
        persisted_conns = data.get("connections", {})
        for channel, conns in persisted_conns.items():
            if channel in matrix._connections:
                for signal, conn_data in conns.items():
                    if signal in matrix._connections[channel]:
                        restored = SynapticConnection.from_dict(conn_data)
                        # Preserve the current base_weight (may have changed)
                        restored.base_weight = matrix._connections[channel][signal].base_weight
                        matrix._connections[channel][signal] = restored

        return matrix

    def save(self, path: str | Path) -> None:
        """Save learned weights to a JSON file.

        Args:
            path: File path to save to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(
            "PlasticWeightMatrix saved: %d connections, %d total updates, step %d",
            sum(len(c) for c in self._connections.values()),
            self.get_total_updates(),
            self._global_step,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        static_weights: dict[str, dict[str, float]],
        stdp_rule: STDPRule | None = None,
    ) -> PlasticWeightMatrix:
        """Load learned weights from a JSON file.

        Falls back to a fresh matrix if the file doesn't exist or is corrupt.

        Args:
            path: File path to load from.
            static_weights: The current static weight defaults.
            stdp_rule: Override STDP rule.

        Returns:
            Restored PlasticWeightMatrix.
        """
        path = Path(path)
        if not path.exists():
            logger.info("No persisted weights at %s, starting fresh", path)
            return cls(static_weights, stdp_rule or STDPRule())

        try:
            with open(path) as f:
                data = json.load(f)
            matrix = cls.from_dict(data, static_weights, stdp_rule)
            logger.info(
                "PlasticWeightMatrix loaded: step %d, %d total updates",
                matrix._global_step,
                matrix.get_total_updates(),
            )
            return matrix
        except Exception as e:
            logger.warning("Failed to load weights from %s: %s. Starting fresh.", path, e)
            return cls(static_weights, stdp_rule or STDPRule())
