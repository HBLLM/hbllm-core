"""Perception — multimodal input nodes (vision, audio, sensor fusion)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

# Extend package search path to include experiments/legacy_perception
_exp_path = Path(__file__).resolve().parents[2] / "experiments" / "legacy_perception"
if _exp_path.exists() and str(_exp_path) not in __path__:
    __path__.append(str(_exp_path))

# PEP 562 Lazy Loading: Map public exports to (module_path, attribute_name).
# Eliminates monolithic dependency loading when importing submodules (e.g. edge/robotics plugins).
_EXPORTS: dict[str, tuple[str, str]] = {
    "AudioInputNode": ("hbllm.perception.audio_in_node", "AudioInputNode"),
    "AudioOutputNode": ("hbllm.perception.audio_out_node", "AudioOutputNode"),
    "EpistemicSpatialGrid": ("hbllm.perception.spatial_grid", "EpistemicSpatialGrid"),
    "EventLog": ("hbllm.perception.event_log", "EventLog"),
    "EventNormalizer": ("hbllm.perception.normalizer", "EventNormalizer"),
    "EventOrigin": ("hbllm.perception.reality_bus", "EventOrigin"),
    "FovealFixation": ("hbllm.perception.saccadic_attention", "FovealFixation"),
    "PerceptionEvent": ("hbllm.perception.reality_bus", "PerceptionEvent"),
    "PerceptionModality": ("hbllm.perception.reality_bus", "PerceptionModality"),
    "PhysicalObservationEvent": (
        "hbllm.perception.physical_perception_stream",
        "PhysicalObservationEvent",
    ),
    "PhysicalPerceptionStream": (
        "hbllm.perception.physical_perception_stream",
        "PhysicalPerceptionStream",
    ),
    "RealityEventBus": ("hbllm.perception.reality_bus", "RealityEventBus"),
    "SaccadicAttentionSystem": ("hbllm.perception.saccadic_attention", "SaccadicAttentionSystem"),
    "SpatialCellPartition": ("hbllm.perception.spatial_grid", "SpatialCellPartition"),
    "VisionNode": ("hbllm.perception.vision_node", "VisionNode"),
    "MultimodalProjector": ("hbllm.perception.vector_projector", "MultimodalProjector"),
    "ReflexArc": ("hbllm.perception.reflex_arc", "ReflexArc"),
    "ReflexRule": ("hbllm.perception.reflex_arc", "ReflexRule"),
}

__all__ = sorted(list(_EXPORTS.keys()))


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_EXPORTS.keys()))
