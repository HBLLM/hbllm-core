"""
Winner-Take-All (WTA) — lateral inhibition for competitive attention.

Implements lateral inhibition where the strongest-firing neurons suppress
weaker competitors.  This is used everywhere in cortex for:

    - Selective attention (one thought wins focus)
    - Saliency selection (most important event gets processing)
    - Working memory stabilization (k items maintained)
    - Domain classification (one domain wins routing)

Designed as **composable** from the start — local WTA competitions
at each processing level feed into an executive WTA:

    Memory candidates → local WTA → survivors
    Reasoning candidates → local WTA → survivors
    Action candidates → local WTA → survivors
                              ↓
                      Executive WTA → final selection

Two modes:
    - **Hard WTA**: Only the neuron with max strength survives.
      Others are completely suppressed.
    - **Soft WTA** (default): Winner's strength amplified, losers'
      strength attenuated proportionally.  Allows graded competition.

Usage::

    from hbllm.brain.snn.wta import WinnerTakeAll, HierarchicalWTA, WTAConfig

    # Single-level competition
    wta = WinnerTakeAll(WTAConfig(k_winners=1, soft_wta=False))
    winners = wta.compete(spike_events)

    # Hierarchical: local WTAs → executive
    hwta = HierarchicalWTA(
        local_configs={"memory": WTAConfig(k_winners=3), "reasoning": WTAConfig(k_winners=2)},
        executive_config=WTAConfig(k_winners=2),
    )
    local = hwta.compete_local("memory", memory_spikes)
    final = hwta.compete_executive({"memory": local, "reasoning": reasoning_local})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from hbllm.brain.snn.lif import SpikeEvent

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# WTAConfig — configuration for winner-take-all competition
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WTAConfig:
    """Configuration for Winner-Take-All competition.

    Attributes:
        inhibition_strength: How strongly the winner suppresses losers.
            Higher values = sharper competition.  Only affects soft WTA.
        soft_wta: If True, use graded competition (losers attenuated).
            If False, use hard competition (losers completely suppressed).
        k_winners: Number of surviving winners.  k=1 gives classic WTA.
            k>1 gives k-WTA, useful for working memory buffers.
        min_strength: Minimum spike strength to participate in competition.
            Spikes below this are not considered (noise floor).
    """

    inhibition_strength: float = 2.0
    soft_wta: bool = True
    k_winners: int = 1
    min_strength: float = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# WinnerTakeAll — single-level lateral inhibition
# ═══════════════════════════════════════════════════════════════════════════


class WinnerTakeAll:
    """Single-level lateral inhibition.  Composable building block.

    Takes a list of spike events and returns a modified list where
    losers are suppressed according to the competition mode.

    Args:
        config: WTA configuration.
    """

    def __init__(self, config: WTAConfig | None = None) -> None:
        self.config = config or WTAConfig()
        self._competition_count: int = 0
        self._total_events_processed: int = 0

    def compete(self, spike_events: list[SpikeEvent]) -> list[SpikeEvent]:
        """Apply lateral inhibition to a set of spike events.

        Only considers spikes where ``fired=True`` and
        ``strength >= min_strength``.  Non-fired spikes are passed
        through unchanged (they're already silent).

        Args:
            spike_events: Spike events from a neuron layer or
                aggregated from multiple sources.

        Returns:
            Modified list of SpikeEvent with losers suppressed.
            Winners retain their original strength (hard WTA) or
            get amplified (soft WTA).
        """
        self._competition_count += 1
        self._total_events_processed += len(spike_events)

        cfg = self.config

        # Find active (fired) spikes above noise floor
        active_indices = [
            i for i, s in enumerate(spike_events) if s.fired and s.strength >= cfg.min_strength
        ]

        # No competition needed if <= k winners
        if len(active_indices) <= cfg.k_winners:
            return list(spike_events)

        # Sort by strength (descending) to find top-k
        active_indices.sort(key=lambda i: spike_events[i].strength, reverse=True)
        winner_indices = set(active_indices[: cfg.k_winners])
        loser_indices = set(active_indices[cfg.k_winners :])

        # Build result
        # Pre-compute the max winner strength (invariant across iterations)
        winner_max = spike_events[active_indices[0]].strength if cfg.soft_wta else 0.0

        result: list[SpikeEvent] = []
        for i, spike in enumerate(spike_events):
            if i in winner_indices:
                if cfg.soft_wta:
                    # Amplify winner proportionally to inhibition strength
                    amplified_strength = spike.strength * (1.0 + cfg.inhibition_strength * 0.1)
                    result.append(
                        SpikeEvent(
                            fired=True,
                            strength=amplified_strength,
                            timestamp=spike.timestamp,
                        )
                    )
                else:
                    # Hard WTA: winner keeps original strength
                    result.append(spike)
            elif i in loser_indices:
                if cfg.soft_wta:
                    # Attenuate loser proportionally
                    ratio = spike.strength / max(winner_max, 1e-10)
                    attenuated = spike.strength * (ratio / cfg.inhibition_strength)
                    result.append(
                        SpikeEvent(
                            fired=attenuated >= cfg.min_strength,
                            strength=attenuated,
                            timestamp=spike.timestamp,
                        )
                    )
                else:
                    # Hard WTA: losers completely suppressed
                    result.append(
                        SpikeEvent(
                            fired=False,
                            strength=0.0,
                            timestamp=spike.timestamp,
                        )
                    )
            else:
                # Non-active spike — pass through
                result.append(spike)

        return result

    def get_winners(self, spike_events: list[SpikeEvent]) -> list[int]:
        """Return indices of winning neurons (top-k by strength).

        Args:
            spike_events: Spike events to evaluate.

        Returns:
            List of indices of the k strongest fired neurons.
        """
        cfg = self.config
        active = [
            (i, s.strength)
            for i, s in enumerate(spike_events)
            if s.fired and s.strength >= cfg.min_strength
        ]
        active.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in active[: cfg.k_winners]]

    def get_stats(self) -> dict[str, Any]:
        """Return competition statistics."""
        return {
            "competitions": self._competition_count,
            "total_events": self._total_events_processed,
            "k_winners": self.config.k_winners,
            "mode": "soft" if self.config.soft_wta else "hard",
        }

    def reset_stats(self) -> None:
        """Reset competition counters."""
        self._competition_count = 0
        self._total_events_processed = 0


# ═══════════════════════════════════════════════════════════════════════════
# HierarchicalWTA — multi-level competition
# ═══════════════════════════════════════════════════════════════════════════


class HierarchicalWTA:
    """Multi-level WTA: local competitions feed into executive competition.

    The cortex is composed of many local competitions rather than one
    global one.  This class provides that architecture:

    1. **Local WTA** per domain: Each processing stream (memory,
       reasoning, action, etc.) runs its own WTA with independent
       configuration.

    2. **Executive WTA**: Winners from all local competitions compete
       in a final executive-level competition to determine which
       domain's output gets priority.

    Args:
        local_configs: Dict mapping domain names to their WTA configs.
        executive_config: Config for the executive-level competition.
    """

    def __init__(
        self,
        local_configs: dict[str, WTAConfig] | None = None,
        executive_config: WTAConfig | None = None,
    ) -> None:
        self._local: dict[str, WinnerTakeAll] = {}
        if local_configs:
            for domain, cfg in local_configs.items():
                self._local[domain] = WinnerTakeAll(cfg)

        self._executive = WinnerTakeAll(executive_config or WTAConfig(k_winners=2, soft_wta=True))

    def add_domain(self, domain: str, config: WTAConfig | None = None) -> None:
        """Add or replace a local WTA for a domain.

        Args:
            domain: Domain name (e.g., ``"memory"``, ``"reasoning"``).
            config: WTA configuration for this domain.
        """
        self._local[domain] = WinnerTakeAll(config or WTAConfig())

    def compete_local(
        self,
        domain: str,
        spike_events: list[SpikeEvent],
    ) -> list[SpikeEvent]:
        """Run local competition for a specific domain.

        If the domain doesn't have a registered WTA, one is created
        with default config.

        Args:
            domain: Which processing stream these spikes belong to.
            spike_events: Spike events from this domain's layer.

        Returns:
            Surviving spikes after local competition.
        """
        if domain not in self._local:
            self._local[domain] = WinnerTakeAll(WTAConfig())

        return self._local[domain].compete(spike_events)

    def compete_executive(
        self,
        local_winners: dict[str, list[SpikeEvent]],
    ) -> list[SpikeEvent]:
        """Run executive competition across all domain winners.

        Flattens all local winners into a single list, competes them,
        and returns the global winners.

        Args:
            local_winners: Dict mapping domain names to their local
                WTA survivors.

        Returns:
            Global winners after executive competition.
        """
        # Flatten all local winners
        all_winners: list[SpikeEvent] = []
        domain_map: list[str] = []  # Track which domain each spike came from

        for domain, spikes in local_winners.items():
            for spike in spikes:
                if spike.fired:
                    all_winners.append(spike)
                    domain_map.append(domain)

        if not all_winners:
            return []

        return self._executive.compete(all_winners)

    def get_local_winners(
        self,
        domain: str,
        spike_events: list[SpikeEvent],
    ) -> list[int]:
        """Get indices of local winners for a domain."""
        if domain not in self._local:
            return []
        return self._local[domain].get_winners(spike_events)

    @property
    def domains(self) -> list[str]:
        """List of registered domains."""
        return list(self._local.keys())

    def get_stats(self) -> dict[str, Any]:
        """Return statistics for all WTA levels."""
        return {
            "local": {domain: wta.get_stats() for domain, wta in self._local.items()},
            "executive": self._executive.get_stats(),
        }
