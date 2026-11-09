from __future__ import annotations

"""Composite Observation Adapter — bridges domain-specific observations to SGE.

Converts various domain-specific observation formats (numpy grids, robotics
joint states, text/NER payloads) into standardised ``ObservationEnvelope`` +
``PerceptualFeatureVector`` streams.  The ``CompositeObservationAdapter``
dispatches to the correct sub-adapter based on the detected modality.

All operations are deterministic and LLM-free.
"""

import hashlib
import time
import uuid
from typing import Any

import numpy as np

from hbllm.hcir.grounding.contracts.enums import DialogueAct, ModalitySource, TrafficClass
from hbllm.hcir.grounding.contracts.observation import ObservationEnvelope
from hbllm.hcir.grounding.features.feature_extractor import (
    FeatureExtractorRegistry,
    PerceptualFeatureVector,
)


class GridObservationAdapter:
    """Converts numpy grid arrays (e.g. ARC-AGI, Sokoban) into observation
    envelopes and perceptual feature vectors via spatial grid extraction."""

    def __init__(
        self,
        channel_id: str = "grid_perception",
        feature_registry: FeatureExtractorRegistry | None = None,
    ) -> None:
        self._channel_id = channel_id
        self._registry = feature_registry or FeatureExtractorRegistry.default()
        self._seq = 0

    def adapt(
        self,
        grid: np.ndarray,
        *,
        entities: list[dict[str, Any]] | None = None,
        step: int | None = None,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        """Converts a grid observation into an envelope + feature vectors."""
        self._seq += 1
        now = time.time()
        obs_id = f"obs_grid_s{step or self._seq}_{uuid.uuid4().hex[:6]}"

        # Build payload hash from grid content
        grid_bytes = grid.tobytes() if hasattr(grid, "tobytes") else bytes(str(grid), "utf-8")
        payload_hash = hashlib.sha256(grid_bytes).hexdigest()[:16]

        # Extract unique non-zero entity features from grid
        entity_colors: set[int] = set()
        if grid.ndim == 2:
            unique_vals = set(int(v) for v in np.unique(grid) if v != 0)
            entity_colors = unique_vals

        envelope = ObservationEnvelope(
            observation_id=obs_id,
            source_modality=ModalitySource.VISION,
            source_channel_id=self._channel_id,
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=self._seq,
            wall_timestamp=now,
            monotonic_timestamp=time.monotonic(),
            payload_ref=f"hcir://grid/{obs_id}",
            clock_domain="grid_clock",
            payload_hash=payload_hash,
            metadata={
                "grid_shape": list(grid.shape),
                "entity_count": len(entities or []),
                "unique_colors": sorted(entity_colors),
            },
        )

        # Build grid-shaped payload for feature extraction
        envelope.metadata["grid_data"] = grid.tolist() if hasattr(grid, "tolist") else grid
        if entities:
            envelope.metadata["entities"] = entities

        features = self._registry.extract_all(envelope)
        return envelope, features


class RoboticsObservationAdapter:
    """Converts robotics state (position, joint angles, velocities, forces)
    into observation envelopes and kinematic feature vectors."""

    def __init__(
        self,
        channel_id: str = "robotics_perception",
        feature_registry: FeatureExtractorRegistry | None = None,
    ) -> None:
        self._channel_id = channel_id
        self._registry = feature_registry or FeatureExtractorRegistry.default()
        self._seq = 0

    def adapt(
        self,
        position: tuple[float, ...],
        *,
        joint_angles: tuple[float, ...] | None = None,
        velocities: tuple[float, ...] | None = None,
        forces: tuple[float, ...] | None = None,
        gripper_state: float = 0.0,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        """Converts robotics state into envelope + kinematic feature vectors."""
        self._seq += 1
        now = time.time()
        obs_id = f"obs_robot_s{self._seq}_{uuid.uuid4().hex[:6]}"

        envelope = ObservationEnvelope(
            observation_id=obs_id,
            source_modality=ModalitySource.PROPRIOCEPTIVE,
            source_channel_id=self._channel_id,
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=self._seq,
            wall_timestamp=now,
            monotonic_timestamp=time.monotonic(),
            payload_ref=f"hcir://robot/{obs_id}",
            clock_domain="robot_clock",
            metadata={
                "position": list(position),
                "joint_angles": list(joint_angles or []),
                "velocities": list(velocities or []),
                "forces": list(forces or []),
                "gripper_state": gripper_state,
            },
        )

        features = self._registry.extract_all(envelope)
        return envelope, features


class TextObservationAdapter:
    """Converts text/NER payloads into observation envelopes and
    semantic text feature vectors."""

    def __init__(
        self,
        channel_id: str = "text_perception",
        feature_registry: FeatureExtractorRegistry | None = None,
    ) -> None:
        self._channel_id = channel_id
        self._registry = feature_registry or FeatureExtractorRegistry.default()
        self._seq = 0

    def adapt(
        self,
        text: str,
        *,
        named_entities: dict[str, list[str]] | None = None,
        embeddings: list[float] | None = None,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        """Converts text input into envelope + semantic feature vectors."""
        self._seq += 1
        now = time.time()
        obs_id = f"obs_text_s{self._seq}_{uuid.uuid4().hex[:6]}"

        payload_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        # Convert dict-style NER {"OBJECT": ["red cube"]} → list-of-dicts
        # [{"text": "red cube", "type": "OBJECT"}] expected by SemanticTextFeatureExtractor
        ne_list: list[dict[str, Any]] = []
        for label, texts in (named_entities or {}).items():
            for t in texts:
                ne_list.append({"text": t, "type": label, "confidence": 1.0})

        envelope = ObservationEnvelope(
            observation_id=obs_id,
            source_modality=ModalitySource.TEXT,
            source_channel_id=self._channel_id,
            traffic_class=TrafficClass.CLASS_DISCRETE,
            sequence_number=self._seq,
            wall_timestamp=now,
            monotonic_timestamp=time.monotonic(),
            payload_ref=f"hcir://text/{obs_id}",
            clock_domain="text_clock",
            payload_hash=payload_hash,
            metadata={
                "text": text,
                "text_length": len(text),
                "named_entities": ne_list,
                "pre_extracted_embeddings": embeddings or [],
            },
        )

        features = self._registry.extract_all(envelope)
        return envelope, features


class SemanticLLMObservationAdapter:
    """Converts natural language utterances and semantic LLM dialogue payloads into
    standardized ObservationEnvelope(source_modality=ModalitySource.SEMANTIC_LLM_ADAPTER)
    packets and semantic feature vectors.
    """

    def __init__(
        self,
        channel_id: str = "dialogue_channel",
        feature_registry: FeatureExtractorRegistry | None = None,
    ) -> None:
        self._channel_id = channel_id
        self._registry = feature_registry or FeatureExtractorRegistry.default()
        self._seq = 0

    def adapt(
        self,
        text: str,
        *,
        speaker_id: str = "user",
        dialogue_act: DialogueAct | str = DialogueAct.COMMAND,
        named_entities: dict[str, list[str]] | list[dict[str, Any]] | None = None,
        embeddings: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        step: int | None = None,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        """Converts dialogue input into an envelope + semantic feature vectors."""
        self._seq += 1
        now = time.time()
        obs_id = f"obs_dialogue_s{step or self._seq}_{uuid.uuid4().hex[:6]}"
        payload_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

        # Normalize named_entities to list[dict[str, Any]]
        ne_list: list[dict[str, Any]] = []
        if isinstance(named_entities, dict):
            for label, items in named_entities.items():
                for item in items:
                    ne_list.append({"text": str(item), "type": label, "confidence": 1.0})
        elif isinstance(named_entities, list):
            ne_list = list(named_entities)

        tokens = [
            w.strip(".,!?:;\"'()[]{}") for w in text.lower().split() if w.strip(".,!?:;\"'()[]{}")
        ]

        meta: dict[str, Any] = {
            "text": text,
            "speaker_id": speaker_id,
            "dialogue_act": (
                dialogue_act.value if isinstance(dialogue_act, DialogueAct) else str(dialogue_act)
            ),
            "tokens": tokens,
            "named_entities": ne_list,
        }
        if embeddings:
            meta["sentence_embedding"] = embeddings
        if metadata:
            meta.update(metadata)

        envelope = ObservationEnvelope(
            observation_id=obs_id,
            source_modality=ModalitySource.SEMANTIC_LLM_ADAPTER,
            source_channel_id=self._channel_id,
            traffic_class=TrafficClass.CLASS_DISCRETE,
            sequence_number=self._seq,
            wall_timestamp=now,
            monotonic_timestamp=time.monotonic(),
            payload_ref=f"hcir://dialogue/{obs_id}",
            clock_domain="dialogue_clock",
            payload_hash=payload_hash,
            metadata=meta,
        )

        features = self._registry.extract_all(envelope)
        return envelope, features


class CompositeObservationAdapter:
    """Dispatches to the correct domain adapter based on input type.

    Usage::

        adapter = CompositeObservationAdapter()
        envelope, features = adapter.adapt_grid(grid_array)
        envelope, features = adapter.adapt_robotics(position=(1.0, 2.0, 3.0))
        envelope, features = adapter.adapt_text("The red cube is on the table.")
        envelope, features = adapter.adapt_dialogue("push the red block")
    """

    def __init__(
        self,
        feature_registry: FeatureExtractorRegistry | None = None,
    ) -> None:
        registry = feature_registry or FeatureExtractorRegistry.default()
        self.grid_adapter = GridObservationAdapter(feature_registry=registry)
        self.robotics_adapter = RoboticsObservationAdapter(feature_registry=registry)
        self.text_adapter = TextObservationAdapter(feature_registry=registry)
        self.semantic_adapter = SemanticLLMObservationAdapter(feature_registry=registry)
        self._registry = registry

    def adapt_grid(
        self,
        grid: np.ndarray,
        **kwargs: Any,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        return self.grid_adapter.adapt(grid, **kwargs)

    def adapt_robotics(
        self,
        position: tuple[float, ...],
        **kwargs: Any,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        return self.robotics_adapter.adapt(position, **kwargs)

    def adapt_text(
        self,
        text: str,
        **kwargs: Any,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        return self.text_adapter.adapt(text, **kwargs)

    def adapt_dialogue(
        self,
        text: str,
        **kwargs: Any,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        return self.semantic_adapter.adapt(text, **kwargs)

    def adapt_driver_input(
        self,
        driver_input: Any,
        **kwargs: Any,
    ) -> tuple[ObservationEnvelope, list[PerceptualFeatureVector]]:
        """Lifts a DriverInput (from any connected device/environment) into an
        ObservationEnvelope + PerceptualFeatureVector stream.
        """
        mod = getattr(driver_input, "modality", None)
        raw = getattr(driver_input, "raw_data", None)
        meta = getattr(driver_input, "metadata", {}) or {}

        # 1. 2D Grid / Spatial
        if (
            str(mod) in ("grid_2d", "DriverModality.GRID_2D")
            or isinstance(raw, np.ndarray)
            and raw.ndim == 2
        ):
            grid = raw if isinstance(raw, np.ndarray) else np.array(raw)
            return self.adapt_grid(
                grid,
                entities=meta.get("entities"),
                step=meta.get("step") or meta.get("StepCounter"),
                **kwargs,
            )

        # 2. Proprioception / Kinematic / Robotics joint states
        if (
            str(mod) in ("proprioception", "DriverModality.PROPRIOCEPTION")
            or isinstance(raw, (tuple, list))
            and len(raw) in (2, 3)
            and all(isinstance(x, (int, float)) for x in raw)
        ):
            pos = tuple(float(x) for x in raw)
            return self.adapt_robotics(
                position=pos,
                joint_angles=meta.get("joint_angles"),
                velocities=meta.get("velocities"),
                forces=meta.get("forces"),
                gripper_state=float(meta.get("gripper_state", 0.0)),
                **kwargs,
            )

        # 3. Dialogue / Semantic LLM Adapter
        if str(mod) in (
            "semantic_llm_adapter",
            "dialogue",
            "DriverModality.SEMANTIC_LLM_ADAPTER",
            "DriverModality.DIALOGUE",
        ):
            return self.adapt_dialogue(
                str(raw),
                speaker_id=meta.get("speaker_id", "user"),
                dialogue_act=meta.get("dialogue_act", DialogueAct.COMMAND),
                named_entities=meta.get("named_entities"),
                embeddings=meta.get("embeddings"),
                metadata=meta,
                **kwargs,
            )

        # 4. Text / Language
        if str(mod) in ("text", "DriverModality.TEXT") or isinstance(raw, str):
            return self.adapt_text(
                str(raw),
                named_entities=meta.get("named_entities"),
                embeddings=meta.get("embeddings"),
                **kwargs,
            )

        # 5. Fallback: structured observation
        return self.adapt_text(
            str(raw) if raw is not None else "",
            named_entities=meta.get("named_entities"),
            **kwargs,
        )
