"""Thalamic Reticular Nucleus (TRN) Selective Inhibitory Gating & Sensory Channel Shielding.

Biologically grounded in TRN searchlight and selective sensory gating
(Crick, 1984; Halassa et al., 2014; Wimmer et al., 2015; Crabtree, 2018):
1. Inhibitory Thalamic Shell Architecture:
   - TRN forms a GABAergic shell enveloping thalamic sensory relay nuclei,
     modulating signal transmission from periphery to cortex.
2. Cross-Modal Distractor Suppression:
   - When prefrontal attention focuses on a primary task modality (e.g., fine visual target tracking),
     TRN elevates inhibitory drive onto competing sensory channels (peripheral vision, ambient noise, tactile drag).
3. Conflict-Driven Sensory Shielding:
   - Under elevated ACC conflict or high cognitive deliberation load, TRN selectively shields
     foveal / goal-relevant streams to prevent working memory distractor overload.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hbllm.hcir.world.causal_discovery import CausalPredicate

logger = logging.getLogger(__name__)


@dataclass
class TRNGateChannel:
    """A sensory relay communication channel gated by TRN GABAergic inhibition."""

    channel_name: str
    modality: (
        str  # "visual_foveal", "visual_peripheral", "kinesthetic", "tactile_haptic", "auditory"
    )
    pass_ratio: (
        float  # [0.0, 1.0] transmission throughput (1.0 = fully open, 0.0 = completely blocked)
    )
    inhibitory_strength: float  # [0.0, 1.0] level of active TRN hyperpolarization
    is_shielded: bool = False


@dataclass
class SensoryShieldingState:
    """Active TRN sensory gating configuration snapshot."""

    focused_modality: str
    conflict_level: float
    channels: dict[str, TRNGateChannel]
    suppressed_channels: list[str]
    effective_throughput: float

    def to_causal_predicate(self) -> CausalPredicate:
        """Export sensory shielding state as a declarative HCIR CausalPredicate."""
        return CausalPredicate(
            variable="sensory_shielding_state",
            operator="==",
            value=f"focused:{self.focused_modality}:suppressed:{len(self.suppressed_channels)}:throughput:{self.effective_throughput:.2f}",
            relation="trn_inhibitory_gating",
            target_entity="thalamic_reticular_nucleus",
        )


class ThalamicReticularGatingEngine:
    """TRN inhibitory gating engine selectively filtering sensory streams during focused cognition."""

    def __init__(self) -> None:
        """Initialize TRN gating engine with canonical sensory relay pathways."""
        self.channels: dict[str, TRNGateChannel] = {}
        self._init_canonical_channels()
        self.last_state: SensoryShieldingState | None = None
        self.total_evaluations: int = 0

    def _init_canonical_channels(self) -> None:
        """Register the 5 primary thalamic relay pathways."""
        self.channels = {
            "visual_foveal": TRNGateChannel(
                channel_name="visual_foveal",
                modality="visual",
                pass_ratio=1.0,
                inhibitory_strength=0.0,
                is_shielded=True,
            ),
            "visual_peripheral": TRNGateChannel(
                channel_name="visual_peripheral",
                modality="visual",
                pass_ratio=0.80,
                inhibitory_strength=0.20,
                is_shielded=False,
            ),
            "kinesthetic_proprioceptive": TRNGateChannel(
                channel_name="kinesthetic_proprioceptive",
                modality="kinesthetic",
                pass_ratio=1.0,
                inhibitory_strength=0.0,
                is_shielded=True,
            ),
            "tactile_haptic": TRNGateChannel(
                channel_name="tactile_haptic",
                modality="tactile",
                pass_ratio=0.75,
                inhibitory_strength=0.25,
                is_shielded=False,
            ),
            "auditory_ambient": TRNGateChannel(
                channel_name="auditory_ambient",
                modality="auditory",
                pass_ratio=0.60,
                inhibitory_strength=0.40,
                is_shielded=False,
            ),
        }

    def modulate_sensory_shielding(
        self,
        attended_modality: str,
        conflict_level: float = 0.0,
    ) -> SensoryShieldingState:
        """Calibrate TRN inhibitory gating based on attentional focus and cognitive conflict.

        Args:
            attended_modality: Primary sensory stream ("visual", "kinesthetic", "tactile", etc.).
            conflict_level: Prefrontal / ACC conflict score in [0.0, 1.0].

        Returns:
            SensoryShieldingState with updated throughput and suppressed channel list.
        """
        self.total_evaluations += 1
        clamped_conflict = max(0.0, min(1.0, float(conflict_level)))
        suppressed: list[str] = []

        total_pass = 0.0
        for name, ch in self.channels.items():
            if ch.modality == attended_modality:
                # Disinhibit attended channel: full transmission, maximum shielding
                ch.inhibitory_strength = 0.0
                ch.pass_ratio = 1.0
                ch.is_shielded = True
            else:
                # Suppress unattended channel: inhibition scales with conflict level
                inhibition = 0.30 + 0.60 * clamped_conflict
                ch.inhibitory_strength = min(1.0, inhibition)
                ch.pass_ratio = max(0.05, 1.0 - ch.inhibitory_strength)
                ch.is_shielded = False
                if ch.pass_ratio < 0.40:
                    suppressed.append(name)

            total_pass += ch.pass_ratio

        mean_throughput = total_pass / len(self.channels) if self.channels else 1.0

        state = SensoryShieldingState(
            focused_modality=attended_modality,
            conflict_level=clamped_conflict,
            channels=self.channels,
            suppressed_channels=suppressed,
            effective_throughput=float(mean_throughput),
        )
        self.last_state = state
        return state

    def filter_signal(
        self,
        channel_name: str,
        raw_amplitude: float,
    ) -> float:
        """Apply TRN transmission gate to a sensory input signal."""
        if channel_name not in self.channels:
            return float(raw_amplitude)

        ch = self.channels[channel_name]
        return float(raw_amplitude * ch.pass_ratio)

    def reset(self) -> None:
        """Reset gating state to default equilibrium."""
        self._init_canonical_channels()
        self.last_state = None
        self.total_evaluations = 0
