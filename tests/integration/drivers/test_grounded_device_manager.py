from __future__ import annotations

"""Integration tests for Grounded Device Manager & Sensory Bridging.

Verifies that DriverInput from connected devices adapts into SGE ObservationEnvelope
and PerceptualFeatureVectors, that MIMO multi-driver inputs fuse across modalities,
and that DeviceBridge infers modality capabilities.
"""

import numpy as np

from hbllm.drivers.base import (
    BaseDriver,
    DriverAction,
    DriverCapability,
    DriverFeedback,
    DriverInput,
    DriverModality,
)
from hbllm.drivers.cognitive_blackbox import CognitiveBlackbox
from hbllm.drivers.manager import DriverManager
from hbllm.hcir.grounding.contracts.enums import ModalitySource
from hbllm.hcir.grounding.features.composite_observation_adapter import (
    CompositeObservationAdapter,
)
from hbllm.serving.device_bridge import DeviceBridge, DeviceInfo


class MockSpatialDriver(BaseDriver):
    def __init__(self, name: str = "mock_camera") -> None:
        super().__init__(
            name=name,
            capabilities={DriverCapability.SPATIAL_2D, DriverCapability.STEP_BASED_EXECUTION},
        )

    def connect(self, target: object) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_inputs(self) -> DriverInput:
        grid = np.zeros((4, 4), dtype=int)
        grid[1, 2] = 5
        return DriverInput(
            raw_data=grid,
            modality=DriverModality.GRID_2D,
            source_id=self.name,
            metadata={"entities": [{"id": "obj1", "position": (1.0, 2.0), "color": 5.0}]},
        )

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        return [DriverAction(action_id=1)]

    def send_output(self, action: DriverAction) -> object:
        return {"status": "ok"}

    def process_feedback(self, raw_result: object) -> DriverFeedback:
        return DriverFeedback(success=True)


class MockKinematicDriver(BaseDriver):
    def __init__(self, name: str = "mock_arm") -> None:
        super().__init__(
            name=name,
            capabilities={DriverCapability.DISCRETE_ACTIONS},
        )

    def connect(self, target: object) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_inputs(self) -> DriverInput:
        return DriverInput(
            raw_data=(1.5, 2.5, 0.0),
            modality=DriverModality.PROPRIOCEPTION,
            source_id=self.name,
            metadata={"velocities": (0.1, 0.0, 0.0), "gripper_state": 0.8},
        )

    def get_action_list(self, inputs: DriverInput) -> list[DriverAction]:
        return [DriverAction(action_id=2)]

    def send_output(self, action: DriverAction) -> object:
        return {"status": "ok"}

    def process_feedback(self, raw_result: object) -> DriverFeedback:
        return DriverFeedback(success=True)


class TestGroundedObservationAdapters:
    """Tests lifting of DriverInputs into SGE ObservationEnvelopes."""

    def test_adapt_grid_driver_input(self) -> None:
        adapter = CompositeObservationAdapter()
        grid = np.zeros((5, 5), dtype=int)
        grid[2, 3] = 4
        inp = DriverInput(
            raw_data=grid,
            modality=DriverModality.GRID_2D,
            source_id="camera_1",
            metadata={"step": 1, "entities": [{"id": "e1", "position": (2.0, 3.0), "color": 4.0}]},
        )
        envelope, features = adapter.adapt_driver_input(inp)

        assert envelope.source_modality == ModalitySource.VISION
        assert len(features) >= 1
        assert features[0].feature_metadata.get("entity_id") == "e1"

    def test_adapt_kinematic_driver_input(self) -> None:
        adapter = CompositeObservationAdapter()
        inp = DriverInput(
            raw_data=(1.0, 2.0, 3.0),
            modality=DriverModality.PROPRIOCEPTION,
            source_id="robot_joint_1",
            metadata={"gripper_state": 0.5},
        )
        envelope, _ = adapter.adapt_driver_input(inp)

        assert envelope.source_modality == ModalitySource.PROPRIOCEPTIVE
        assert envelope.metadata["position"] == [1.0, 2.0, 3.0]
        assert envelope.metadata["gripper_state"] == 0.5

    def test_adapt_text_driver_input(self) -> None:
        adapter = CompositeObservationAdapter()
        inp = DriverInput(
            raw_data="Pick up the red cube",
            modality=DriverModality.TEXT,
            source_id="chat_ui",
            metadata={"named_entities": {"OBJECT": ["red cube"]}},
        )
        envelope, features = adapter.adapt_driver_input(inp)

        assert envelope.source_modality == ModalitySource.TEXT
        assert len(features) >= 1


class TestDriverManagerGrounding:
    """Tests DriverManager with SGE grounding and multi-driver fusion."""

    def test_driver_manager_adapt_to_grounding(self) -> None:
        dm = DriverManager()
        grid = np.zeros((3, 3), dtype=int)
        inp = DriverInput(raw_data=grid, modality=DriverModality.GRID_2D)
        envelope, features = dm.adapt_to_grounded_observation(inp)

        assert envelope.source_modality == ModalitySource.VISION

    def test_fuse_mimo_observations(self) -> None:
        dm = DriverManager()
        cam = MockSpatialDriver()
        arm = MockKinematicDriver()

        dm.register(cam)
        dm.register(arm)
        dm.bind_concurrent(cam.name, target="sim")
        dm.bind_concurrent(arm.name, target="sim")

        fused = dm.fuse_mimo_observations(entity_id="robot_cell")
        assert fused.entity_id == "robot_cell"
        assert len(fused.fused_embedding) > 0
        assert len(fused.contributing_feature_ids) >= 1


class TestCognitiveBlackboxGrounding:
    """Tests that CognitiveBlackbox initializes epistemic engines with SGE grounding."""

    def test_epistemic_engine_has_grounding_bridge(self) -> None:
        cb = CognitiveBlackbox()
        engine = cb.get_epistemic_engine("test_domain")

        assert engine is not None
        assert engine.grounding_bridge is not None
        assert engine.cortical_pipeline.grounding_bridge is engine.grounding_bridge

    def test_observe_grounds_driver_input_into_sge(self) -> None:
        cb = CognitiveBlackbox()
        grid = np.zeros((4, 4), dtype=int)
        grid[1, 2] = 3
        inp = DriverInput(
            raw_data=grid,
            modality=DriverModality.GRID_2D,
            source_id="camera_sensor",
            metadata={"entities": [{"id": "target_box", "position": (1.0, 2.0), "color": 3.0}]},
        )
        cb.observe(inp, perception_data={"grid": grid})

        engine = cb.get_epistemic_engine("camera_sensor")
        assert engine.grounding_bridge is not None
        # Verify SGE entity tracker ingested the observation
        candidates = engine.grounding_bridge.tracker.get_active_candidates()
        assert len(candidates) >= 1

    def test_decide_and_update_through_sge_motor_gateway(self) -> None:
        from hbllm.hcir.grounding.contracts.actions import ActionLifecycleState

        cb = CognitiveBlackbox()
        grid = np.zeros((4, 4), dtype=int)
        inp = DriverInput(
            raw_data=grid,
            modality=DriverModality.GRID_2D,
            source_id="robot_source",
            metadata={"entities": [{"id": "obj", "position": (2.0, 2.0), "color": 1.0}]},
        )
        cb.observe(inp, perception_data={"grid": grid})

        avail = [DriverAction(action_id=1, parameters={"speed": 0.5})]
        chosen = cb.decide(avail, source_id="robot_source")
        assert chosen.action_id == 1

        bridge = cb.get_epistemic_engine("robot_source").grounding_bridge
        assert bridge is not None
        assert bridge.state.last_execution_record is not None
        assert bridge.state.last_execution_record.state == ActionLifecycleState.DISPATCHED

        # Update with feedback
        cb.update(chosen, DriverFeedback(success=True), source_id="robot_source")
        assert bridge.state.last_execution_record.state == ActionLifecycleState.EXECUTION_REPORTED

    def test_driver_manager_cognitive_step_full_sge_grounding_cycle(self) -> None:
        from hbllm.hcir.grounding.contracts.actions import ActionLifecycleState

        dm = DriverManager()
        cb = CognitiveBlackbox()
        dm.set_cognitive_engine(cb)

        cam = MockSpatialDriver(name="cam_primary")
        dm.register(cam)
        dm.bind(cam.name, target="world")

        # Execute single cognitive step
        feedback = dm.execute_cognitive_step()
        assert feedback.success is True

        engine = cb.get_epistemic_engine("cam_primary")
        assert engine.grounding_bridge is not None
        candidates = engine.grounding_bridge.tracker.get_active_candidates()
        assert len(candidates) >= 1
        assert engine.grounding_bridge.state.last_execution_record is not None
        assert (
            engine.grounding_bridge.state.last_execution_record.state
            == ActionLifecycleState.EXECUTION_REPORTED
        )


class TestDeviceBridgeModalities:
    """Tests DeviceBridge modality inference for SGE."""

    def test_device_info_inferred_modalities(self) -> None:
        phone = DeviceInfo(
            device_id="phone-1",
            tenant_id="t1",
            device_type="mobile",
            capabilities=["audio", "display", "camera"],
        )
        mods = phone.modalities
        assert "audio" in mods
        assert "vision" in mods
        assert "text" in mods

    def test_device_bridge_get_device_modalities(self) -> None:
        bridge = DeviceBridge()
        bridge.register_device(
            DeviceInfo(
                device_id="speaker-1",
                tenant_id="t1",
                device_type="speaker",
                capabilities=["audio"],
            )
        )
        mods = bridge.get_device_modalities("speaker-1")
        assert mods == ["audio"]
