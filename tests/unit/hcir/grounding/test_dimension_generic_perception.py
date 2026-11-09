from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from hbllm.hcir.grounding.contracts.enums import ModalitySource, TrafficClass
from hbllm.hcir.grounding.contracts.observation import (
    CoordinateType,
    ObservationEnvelope,
    ReferenceFrame,
)
from hbllm.hcir.grounding.features.composite_observation_adapter import (
    CompositeObservationAdapter,
    RoboticsObservationAdapter,
)
from hbllm.hcir.spatial_planner import EntityRole, SpatialEntity


class TestReferenceFrameAndDimensionGenericPerception:
    """Validates ReferenceFrame contracts, dimension-generic coordinate lifting,
    and continuous N-dimensional spatial entities across grounding and perception.
    """

    def test_reference_frame_creation_and_validation(self) -> None:
        """Verify ReferenceFrame creation, axis validation, and coordinate types."""
        # 2D discrete grid frame
        grid_frame = ReferenceFrame(
            frame_id="custom_grid",
            dimensions=2,
            coordinate_type=CoordinateType.DISCRETE,
            axes=("row", "col"),
            scale=(1.0, 1.0),
            bounds=((0.0, 19.0), (0.0, 19.0)),
            unit="cell",
        )
        assert grid_frame.frame_id == "custom_grid"
        assert grid_frame.dimensions == 2
        assert grid_frame.coordinate_type == CoordinateType.DISCRETE
        assert grid_frame.axes == ("row", "col")
        assert grid_frame.scale == (1.0, 1.0)
        assert grid_frame.unit == "cell"

        # Validation errors
        with pytest.raises(ValueError, match="frame_id must be non-empty"):
            ReferenceFrame(
                frame_id="",
                dimensions=2,
                coordinate_type=CoordinateType.DISCRETE,
                axes=("row", "col"),
            )

        with pytest.raises(ValueError, match="dimensions must be positive"):
            ReferenceFrame(
                frame_id="invalid",
                dimensions=0,
                coordinate_type=CoordinateType.CONTINUOUS,
                axes=(),
            )

        with pytest.raises(ValueError, match="axes length"):
            ReferenceFrame(
                frame_id="mismatch",
                dimensions=3,
                coordinate_type=CoordinateType.CONTINUOUS,
                axes=("x", "y"),
            )

        with pytest.raises(ValueError, match="scale length"):
            ReferenceFrame(
                frame_id="scale_mismatch",
                dimensions=2,
                coordinate_type=CoordinateType.DISCRETE,
                axes=("r", "c"),
                scale=(1.0, 1.0, 1.0),
            )

        with pytest.raises(ValueError, match="All scale components must be positive"):
            ReferenceFrame(
                frame_id="bad_scale",
                dimensions=2,
                coordinate_type=CoordinateType.DISCRETE,
                axes=("r", "c"),
                scale=(1.0, -0.5),
            )

        with pytest.raises(ValueError, match="Invalid bound at dim 0"):
            ReferenceFrame(
                frame_id="bad_bounds",
                dimensions=2,
                coordinate_type=CoordinateType.DISCRETE,
                axes=("r", "c"),
                bounds=((10.0, 2.0), (0.0, 5.0)),
            )

    def test_reference_frame_factories(self) -> None:
        """Verify standard ReferenceFrame factory constructors."""
        # 2D Grid
        rf_grid = ReferenceFrame.grid_2d(shape=(25, 30))
        assert rf_grid.frame_id == "grid_2d"
        assert rf_grid.dimensions == 2
        assert rf_grid.coordinate_type == CoordinateType.DISCRETE
        assert rf_grid.axes == ("row", "col")
        assert rf_grid.bounds == ((0.0, 24.0), (0.0, 29.0))
        assert rf_grid.unit == "cell"

        # 3D Cartesian
        rf_cart = ReferenceFrame.cartesian_3d(
            bounds=((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)),
            unit="meter",
        )
        assert rf_cart.frame_id == "cartesian_3d"
        assert rf_cart.dimensions == 3
        assert rf_cart.coordinate_type == CoordinateType.CONTINUOUS
        assert rf_cart.axes == ("x", "y", "z")
        assert rf_cart.bounds == ((-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0))
        assert rf_cart.unit == "meter"

        # 7-DOF Joint Space
        rf_joint = ReferenceFrame.joint_space(dof=7)
        assert rf_joint.frame_id == "joint_space"
        assert rf_joint.dimensions == 7
        assert rf_joint.coordinate_type == CoordinateType.CONTINUOUS
        assert rf_joint.axes == (
            "joint_0",
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        )
        assert rf_joint.unit == "radian"

        # Generic N-D
        rf_nd = ReferenceFrame.generic_nd(dimensions=5, coordinate_type=CoordinateType.CONTINUOUS)
        assert rf_nd.dimensions == 5
        assert len(rf_nd.axes) == 5
        assert rf_nd.scale == (1.0, 1.0, 1.0, 1.0, 1.0)

    def test_reference_frame_point_transform_and_bounds(self) -> None:
        """Verify scale-space point transformations and bounds checking."""
        src_frame = ReferenceFrame.cartesian_3d(scale=(1.0, 1.0, 1.0), unit="meter")
        tgt_frame = ReferenceFrame.cartesian_3d(scale=(1000.0, 1000.0, 1000.0), unit="millimeter")

        point_m = (1.5, 0.25, -0.75)
        point_mm = src_frame.transform_point(point_m, tgt_frame)
        assert point_mm == (1500.0, 250.0, -750.0)

        # Invert transform
        roundtrip = tgt_frame.transform_point(point_mm, src_frame)
        assert math.isclose(roundtrip[0], point_m[0])
        assert math.isclose(roundtrip[1], point_m[1])
        assert math.isclose(roundtrip[2], point_m[2])

        # Bounds check
        bounded_frame = ReferenceFrame.cartesian_3d(
            bounds=((0.0, 2.0), (-1.0, 1.0), (0.0, 1.0)),
        )
        assert bounded_frame.is_within_bounds((1.0, 0.0, 0.5)) is True
        assert bounded_frame.is_within_bounds((3.0, 0.0, 0.5)) is False
        assert bounded_frame.is_within_bounds((1.0, -2.0, 0.5)) is False

        # Dimension mismatch
        with pytest.raises(ValueError, match="Cannot transform point between mismatched"):
            src_frame.transform_point(point_m, ReferenceFrame.grid_2d())

    def test_observation_envelope_reference_frame_binding_and_inference(self) -> None:
        """Verify ObservationEnvelope retains explicit or automatically infers ReferenceFrame."""
        # Explicit reference frame
        custom_frame = ReferenceFrame.joint_space(dof=6)
        env_explicit = ObservationEnvelope.create(
            observation_id="obs_exp_1",
            source_modality=ModalitySource.PROPRIOCEPTIVE,
            source_channel_id="robotics_channel",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=1,
            payload_ref="hcir://robot/obs_1",
            reference_frame=custom_frame,
        )
        assert env_explicit.reference_frame == custom_frame

        # Inferred vision reference frame from grid_shape
        env_vis = ObservationEnvelope.create(
            observation_id="obs_vis_1",
            source_modality=ModalitySource.VISION,
            source_channel_id="camera_channel",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=2,
            payload_ref="hcir://grid/obs_2",
            metadata={"grid_shape": [12, 16]},
        )
        assert env_vis.reference_frame is not None
        assert env_vis.reference_frame.dimensions == 2
        assert env_vis.reference_frame.coordinate_type == CoordinateType.DISCRETE
        assert env_vis.reference_frame.bounds == ((0.0, 11.0), (0.0, 15.0))

        # Inferred 3D proprioception reference frame from position
        env_robot = ObservationEnvelope.create(
            observation_id="obs_robot_1",
            source_modality=ModalitySource.PROPRIOCEPTIVE,
            source_channel_id="arm_channel",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=3,
            payload_ref="hcir://robot/obs_3",
            metadata={"position": [0.45, -0.12, 0.88]},
        )
        assert env_robot.reference_frame is not None
        assert env_robot.reference_frame.dimensions == 3
        assert env_robot.reference_frame.coordinate_type == CoordinateType.CONTINUOUS
        assert env_robot.reference_frame.axes == ("x", "y", "z")

        # Inferred generic N-D continuous frame for 7-DOF positions
        env_7d = ObservationEnvelope.create(
            observation_id="obs_joint_1",
            source_modality=ModalitySource.PROPRIOCEPTIVE,
            source_channel_id="arm_channel",
            traffic_class=TrafficClass.CLASS_HIGH_FREQ,
            sequence_number=4,
            payload_ref="hcir://robot/obs_4",
            metadata={"position": [0.0, 0.1, -0.2, 0.5, 0.0, -0.1, 0.3]},
        )
        assert env_7d.reference_frame is not None
        assert env_7d.reference_frame.dimensions == 7
        assert env_7d.reference_frame.coordinate_type == CoordinateType.CONTINUOUS

    def test_grid_and_robotics_adapters_populate_reference_frames(self) -> None:
        """Verify GridObservationAdapter, RoboticsObservationAdapter, and CompositeObservationAdapter."""
        composite = CompositeObservationAdapter()

        # Grid adaptation
        grid_2d = np.zeros((10, 15), dtype=np.int32)
        grid_2d[2, 3] = 1
        env_grid, _ = composite.adapt_grid(grid_2d)
        assert env_grid.reference_frame is not None
        assert env_grid.reference_frame.dimensions == 2
        assert env_grid.reference_frame.coordinate_type == CoordinateType.DISCRETE
        assert env_grid.reference_frame.bounds == ((0.0, 9.0), (0.0, 14.0))

        # Robotics adaptation (3D continuous)
        pos_3d = (0.5, 0.2, 0.1)
        env_robot, _ = composite.adapt_robotics(
            position=pos_3d,
            joint_angles=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            gripper_state=1.0,
        )
        assert env_robot.reference_frame is not None
        assert env_robot.reference_frame.dimensions == 3
        assert env_robot.reference_frame.coordinate_type == CoordinateType.CONTINUOUS

        # Composite driver input dispatch with continuous joint states
        class MockDriverInput:
            def __init__(self, raw: Any, mod: str) -> None:
                self.raw_data = raw
                self.modality = mod
                self.metadata: dict[str, Any] = {}

        driver_input = MockDriverInput(raw=(0.0, 0.2, -0.4, 0.8, -0.1, 0.5, 0.3), mod="robotics")
        env_driver, _ = composite.adapt_driver_input(driver_input)
        assert env_driver.reference_frame is not None
        assert env_driver.reference_frame.dimensions == 7
        assert env_driver.reference_frame.coordinate_type == CoordinateType.CONTINUOUS

    def test_spatial_entity_nd_generalization(self) -> None:
        """Verify SpatialEntity generalizes from 2D discrete grids to N-dimensional continuous spaces."""
        # 1. Backward-compatible 2D grid entity
        ent_2d = SpatialEntity(
            id="e_grid_1",
            role=EntityRole.AGENT,
            grid_pos=(4, 7),
            centroid=(4.0, 7.0),
        )
        assert ent_2d.dimensionality == 2
        assert ent_2d.centroid == (4.0, 7.0)
        assert ent_2d.grid_pos == (4, 7)
        assert ent_2d.position == (4.0, 7.0)
        assert ent_2d.coordinates == (4.0, 7.0)
        assert ent_2d.bounding_box == (4, 4, 7, 7)

        # 2. 3D continuous metric entity
        rf_3d = ReferenceFrame.cartesian_3d()
        ent_3d = SpatialEntity(
            id="e_robot_block",
            role=EntityRole.MANIPULABLE,
            position=(0.35, -0.20, 0.82),
            reference_frame=rf_3d,
            bounds_nd=((0.30, 0.40), (-0.25, -0.15), (0.80, 0.84)),
        )
        assert ent_3d.dimensionality == 3
        assert ent_3d.centroid == (0.35, -0.20, 0.82)
        assert ent_3d.position == (0.35, -0.20, 0.82)
        assert ent_3d.coordinates == (0.35, -0.20, 0.82)
        assert ent_3d.grid_pos == (0, 0, 1)  # Rounded ints
        assert ent_3d.reference_frame == rf_3d
        assert ent_3d.bounds_nd == ((0.30, 0.40), (-0.25, -0.15), (0.80, 0.84))

        # Coordinate setter update
        ent_3d.position = (0.50, -0.10, 0.85)
        assert ent_3d.centroid == (0.50, -0.10, 0.85)

        # 3. 7D robot joint configuration entity
        rf_7d = ReferenceFrame.joint_space(dof=7)
        joint_state = (0.0, -0.785, 0.0, 1.571, 0.0, 0.785, 0.0)
        ent_7d = SpatialEntity(
            id="e_arm_pose",
            role=EntityRole.AGENT,
            position=joint_state,
            reference_frame=rf_7d,
        )
        assert ent_7d.dimensionality == 7
        assert ent_7d.centroid == joint_state
        assert ent_7d.reference_frame == rf_7d

    def test_spatial_entity_nd_distance_calculation(self) -> None:
        """Verify Euclidean distance computation across arbitrary N dimensions."""
        # 2D distance
        e1_2d = SpatialEntity(centroid=(0.0, 0.0))
        e2_2d = SpatialEntity(centroid=(3.0, 4.0))
        assert math.isclose(e1_2d.distance_to(e2_2d), 5.0)
        assert math.isclose(e1_2d.distance_to((3.0, 4.0)), 5.0)

        # 3D distance
        e1_3d = SpatialEntity(position=(1.0, 2.0, 3.0))
        e2_3d = SpatialEntity(position=(4.0, 6.0, 3.0))
        assert math.isclose(e1_3d.distance_to(e2_3d), 5.0)
        assert math.isclose(e1_3d.distance_to((4.0, 6.0, 3.0)), 5.0)

        # 4D distance
        e1_4d = SpatialEntity(position=(0.0, 0.0, 0.0, 0.0))
        e2_4d = SpatialEntity(position=(1.0, 1.0, 1.0, 1.0))
        assert math.isclose(e1_4d.distance_to(e2_4d), 2.0)

        # Dimension mismatch
        with pytest.raises(ValueError, match="Dimension mismatch in distance_to"):
            e1_2d.distance_to(e1_3d)

    def test_continuous_robotics_lifting_scenario(self) -> None:
        """Verify lifting continuous robotics state and perceived workspace objects
        into typed SpatialEntity components with continuous metric grounding.
        """
        adapter = RoboticsObservationAdapter()

        end_effector_pos = (0.40, 0.15, 0.35)
        joint_angles = (0.1, -0.4, 0.3, 1.2, -0.2, 0.5)
        velocities = (0.01, 0.0, 0.02, 0.0, 0.0, 0.0)

        detected_objects = [
            {
                "id": "target_cube",
                "role": EntityRole.MANIPULABLE,
                "position": (0.42, 0.16, 0.36),
                "properties": {"color": "red", "weight_kg": 0.25},
            },
            {
                "id": "obstacle_strut",
                "role": EntityRole.OBSTACLE,
                "position": (0.80, 0.50, 0.35),
                "properties": {"material": "steel"},
            },
            {
                "id": "receptacle_bin",
                "role": EntityRole.RECEPTACLE,
                "position": (0.10, 0.60, 0.20),
                "properties": {"capacity": 5},
            },
        ]

        # Lift into typed SpatialEntity components
        entities = adapter.lift_spatial_entities(
            position=end_effector_pos,
            joint_angles=joint_angles,
            velocities=velocities,
            gripper_state=0.0,
            objects=detected_objects,
        )

        assert len(entities) == 4
        avatar = entities[0]
        assert avatar.id == "robot_avatar"
        assert avatar.role == EntityRole.AGENT
        assert avatar.centroid == end_effector_pos
        assert avatar.dimensionality == 3
        assert avatar.properties["joint_angles"] == list(joint_angles)

        target = entities[1]
        assert target.id == "target_cube"
        assert target.role == EntityRole.MANIPULABLE
        assert target.properties["color"] == "red"

        obstacle = entities[2]
        assert obstacle.id == "obstacle_strut"
        assert obstacle.role == EntityRole.OBSTACLE

        receptacle = entities[3]
        assert receptacle.id == "receptacle_bin"
        assert receptacle.role == EntityRole.RECEPTACLE

        # Spatial metric reasoning: target is within grasping proximity (0.025m)
        dist_to_target = avatar.distance_to(target)
        assert dist_to_target < 0.03

        # Obstacle is distant (0.53m)
        dist_to_obstacle = avatar.distance_to(obstacle)
        assert dist_to_obstacle > 0.50
