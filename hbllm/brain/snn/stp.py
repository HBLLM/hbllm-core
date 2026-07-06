"""
Short-Term Plasticity (STP) — transient synaptic modulation.

STP is **activity state**, not learning.  It models the temporary
facilitation or depression of synaptic transmission based on recent
firing history.

Architecture::

    Synapse = permanent_weight (modified only by STDP)
                    ×
              stp_factor (transient modulation from STP)
                    =
              effective_weight (used for current computation)

Key distinction from STDP:

    - **STDP** modifies ``permanent_weight`` — long-term learning
    - **STP** produces a transient ``stp_factor`` — activity state

STP lives entirely outside ``PlasticWeightMatrix``.  The learning
rule never sees STP modulation.

Tsodyks-Markram model (1997):

    Short-Term Facilitation (STF):
        - Repeated firing temporarily increases release probability
        - Models attention bursts, priming, expectation
        - Recovery: exponential return to baseline

    Short-Term Depression (STD):
        - Repeated firing temporarily depletes synaptic resources
        - Models habituation, saturation, fatigue
        - Recovery: exponential replenishment

    Combined:
        - Both effects with independent time constants
        - Effective modulation = release_probability × available_resources

Usage::

    from hbllm.brain.snn.stp import STPConfig, STPManager

    stp = STPManager(STPConfig(mode="combined"))

    # During each timestep in LayerProjection.project():
    effective = stp.get_effective_weight(
        permanent_weight=0.5,
        synapse_id=(source_idx, target_idx),
        timestamp=time.time(),
        spiked=True,
    )
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# STPConfig — configuration for short-term plasticity
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class STPConfig:
    """Configuration for the Tsodyks-Markram STP model.

    Parameters:
        U: Baseline release probability [0, 1].  Higher U means each
           spike releases a larger fraction of available resources.
        tau_d: Depression recovery time constant (seconds).  How fast
           depleted resources replenish.  Shorter = faster recovery.
        tau_f: Facilitation recovery time constant (seconds).  How fast
           elevated release probability decays back to U.  Longer = more
           sustained facilitation.
        mode: STP behavior mode.
           ``"facilitation"`` — only facilitation (STF)
           ``"depression"`` — only depression (STD)
           ``"combined"`` — both effects (default, most realistic)
    """

    U: float = 0.2
    tau_d: float = 0.2
    tau_f: float = 1.5
    mode: str = "combined"

    # ── Preset factory methods ──

    @classmethod
    def facilitation_only(cls, **overrides: Any) -> STPConfig:
        """Pure facilitation. For attention bursts and priming."""
        return cls(U=0.15, tau_d=0.1, tau_f=2.0, mode="facilitation", **overrides)

    @classmethod
    def depression_only(cls, **overrides: Any) -> STPConfig:
        """Pure depression. For habituation and saturation."""
        return cls(U=0.5, tau_d=0.5, tau_f=0.1, mode="depression", **overrides)

    @classmethod
    def combined_default(cls, **overrides: Any) -> STPConfig:
        """Standard combined STP. Both effects with balanced dynamics."""
        return cls(U=0.2, tau_d=0.2, tau_f=1.5, mode="combined", **overrides)

    @classmethod
    def fast_attention(cls, **overrides: Any) -> STPConfig:
        """Fast facilitation with minimal depression.

        For attention-like rapid enhancement followed by quick recovery.
        """
        return cls(U=0.1, tau_d=0.05, tau_f=0.5, mode="combined", **overrides)


# ═══════════════════════════════════════════════════════════════════════════
# SynapticActivity — per-synapse transient state
# ═══════════════════════════════════════════════════════════════════════════


class SynapticActivity:
    """Per-synapse STP state using the Tsodyks-Markram model.

    NOT a learning rule — purely transient modulation that decays
    back to baseline between spikes.

    State variables:
        u: Current release probability.  Increases on each spike
           (facilitation) and decays back to ``U`` between spikes.
        x: Available synaptic resources [0, 1].  Decreases on each
           spike (depression) and recovers toward 1.0 between spikes.
        last_spike_time: Timestamp of the last pre-synaptic spike.

    The effective modulation factor is ``u × x``, which scales the
    permanent weight to produce the effective weight.
    """

    def __init__(self, config: STPConfig) -> None:
        self.config = config
        self.u: float = config.U  # Release probability
        self.x: float = 1.0  # Available resources
        self.last_spike_time: float = 0.0

    def modulate(self, timestamp: float, spiked: bool) -> float:
        """Compute the transient STP modulation factor.

        Called on every timestep.  When a pre-synaptic spike occurs,
        the release probability and available resources are updated.
        Between spikes, both variables recover toward baseline.

        Args:
            timestamp: Current time in seconds.
            spiked: Whether the pre-synaptic neuron fired on this step.

        Returns:
            The modulation factor.  Multiply by permanent_weight to
            get the effective weight.  Range roughly [0.0, ~3.0]
            depending on facilitation dynamics.
        """
        cfg = self.config
        dt = max(0.0, timestamp - self.last_spike_time) if self.last_spike_time > 0 else 0.0

        if spiked and dt > 0:
            # Recovery between spikes (before processing this spike)
            if cfg.mode in ("depression", "combined"):
                # Resources recover: x → 1.0
                self.x = 1.0 - (1.0 - self.x) * math.exp(-dt / cfg.tau_d)

            if cfg.mode in ("facilitation", "combined"):
                # Release probability decays: u → U
                self.u = cfg.U + (self.u - cfg.U) * math.exp(-dt / cfg.tau_f)

            # Spike event: update state
            # Facilitation: u increases
            u_pre = self.u
            if cfg.mode in ("facilitation", "combined"):
                self.u = self.u + cfg.U * (1.0 - self.u)

            # Depression: resources are consumed
            if cfg.mode in ("depression", "combined"):
                self.x = self.x - u_pre * self.x

            self.last_spike_time = timestamp

        elif not spiked and self.last_spike_time > 0 and dt > 0:
            # No spike: just recover toward baseline
            if cfg.mode in ("depression", "combined"):
                self.x = 1.0 - (1.0 - self.x) * math.exp(-dt / cfg.tau_d)
            if cfg.mode in ("facilitation", "combined"):
                self.u = cfg.U + (self.u - cfg.U) * math.exp(-dt / cfg.tau_f)

        # Modulation factor = u × x
        return self.u * self.x

    def reset(self) -> None:
        """Reset to baseline state."""
        self.u = self.config.U
        self.x = 1.0
        self.last_spike_time = 0.0

    def to_dict(self) -> dict[str, float]:
        """Serialize state."""
        return {
            "u": self.u,
            "x": self.x,
            "last_spike_time": self.last_spike_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: STPConfig) -> SynapticActivity:
        """Restore from persisted state."""
        activity = cls(config)
        activity.u = data.get("u", config.U)
        activity.x = data.get("x", 1.0)
        activity.last_spike_time = data.get("last_spike_time", 0.0)
        return activity


# ═══════════════════════════════════════════════════════════════════════════
# STPManager — manages STP state for all synapses in a projection
# ═══════════════════════════════════════════════════════════════════════════


class STPManager:
    """Manages Short-Term Plasticity state for all synapses in a projection.

    Each synapse (identified by ``(source_idx, target_idx)``) has its own
    ``SynapticActivity`` instance with independent facilitation/depression
    dynamics.

    Usage in ``LayerProjection.project()``::

        effective_weight = stp_manager.get_effective_weight(
            permanent_weight=weight_matrix[i][j],
            synapse_id=(i, j),
            timestamp=timestamp,
            spiked=source_spikes[i].fired,
        )

    Args:
        config: STP configuration shared by all synapses in this manager.
        source_size: Number of source neurons.
        target_size: Number of target neurons.
    """

    def __init__(
        self,
        config: STPConfig,
        source_size: int,
        target_size: int,
    ) -> None:
        self.config = config
        self.source_size = source_size
        self.target_size = target_size

        # Per-synapse activity state
        self._activities: dict[tuple[int, int], SynapticActivity] = {}

    def _get_activity(self, synapse_id: tuple[int, int]) -> SynapticActivity:
        """Get or lazily create the SynapticActivity for a synapse."""
        if synapse_id not in self._activities:
            self._activities[synapse_id] = SynapticActivity(self.config)
        return self._activities[synapse_id]

    def get_effective_weight(
        self,
        permanent_weight: float,
        synapse_id: tuple[int, int],
        timestamp: float,
        spiked: bool,
    ) -> float:
        """Compute the effective synaptic weight after STP modulation.

        Args:
            permanent_weight: The STDP-modifiable weight.
            synapse_id: ``(source_idx, target_idx)`` pair.
            timestamp: Current time in seconds.
            spiked: Whether the pre-synaptic neuron fired.

        Returns:
            ``permanent_weight × stp_factor``
        """
        activity = self._get_activity(synapse_id)
        stp_factor = activity.modulate(timestamp, spiked)
        return permanent_weight * stp_factor

    def get_modulation_factor(
        self,
        synapse_id: tuple[int, int],
    ) -> float:
        """Get the current STP modulation factor for a synapse.

        Returns 1.0 (no modulation) if the synapse hasn't been seen.
        """
        if synapse_id not in self._activities:
            return 1.0
        return self._activities[synapse_id].u * self._activities[synapse_id].x

    def reset(self) -> None:
        """Reset all synaptic activities to baseline."""
        for activity in self._activities.values():
            activity.reset()

    @property
    def active_synapse_count(self) -> int:
        """Number of synapses with STP state."""
        return len(self._activities)

    def get_stats(self) -> dict[str, Any]:
        """Return summary statistics of STP modulation across all synapses."""
        if not self._activities:
            return {"active_synapses": 0, "mean_modulation": 1.0}

        factors = [a.u * a.x for a in self._activities.values()]
        return {
            "active_synapses": len(factors),
            "mean_modulation": sum(factors) / len(factors),
            "min_modulation": min(factors),
            "max_modulation": max(factors),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize all STP state."""
        return {
            "config": {
                "U": self.config.U,
                "tau_d": self.config.tau_d,
                "tau_f": self.config.tau_f,
                "mode": self.config.mode,
            },
            "activities": {f"{k[0]}_{k[1]}": v.to_dict() for k, v in self._activities.items()},
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        source_size: int,
        target_size: int,
    ) -> STPManager:
        """Restore from persisted state."""
        cfg_data = data.get("config", {})
        config = STPConfig(
            U=cfg_data.get("U", 0.2),
            tau_d=cfg_data.get("tau_d", 0.2),
            tau_f=cfg_data.get("tau_f", 1.5),
            mode=cfg_data.get("mode", "combined"),
        )
        manager = cls(config, source_size, target_size)

        for key_str, act_data in data.get("activities", {}).items():
            parts = key_str.split("_")
            if len(parts) == 2:
                synapse_id = (int(parts[0]), int(parts[1]))
                manager._activities[synapse_id] = SynapticActivity.from_dict(act_data, config)

        return manager
