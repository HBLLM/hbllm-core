"""Tests for Epistemic Routing and QoS Constraints."""

from hbllm.brain.mesh.registry import (
    NodeCapabilities,
    NodeProfile,
    NodeRegistry,
    NodeType,
    TaskPriorityClass,
)
from hbllm.brain.mesh.router import EpistemicRouter


def test_epistemic_routing():
    registry = NodeRegistry("phone_1", NodeType.PHONE)

    car = NodeProfile("car_1", NodeType.CAR)
    desktop = NodeProfile("desktop_1", NodeType.DESKTOP, NodeCapabilities(has_gpu=True))
    server = NodeProfile("server_1", NodeType.CLOUD_SERVER, NodeCapabilities(has_npu=True))

    registry.register_node(car)
    registry.register_node(desktop)
    registry.register_node(server)

    router = EpistemicRouter(registry)

    # Route by Domain Authority
    assert router.route_task("vehicle", TaskPriorityClass.INTERACTIVE) == "car_1"
    assert router.route_task("long_term_memory", TaskPriorityClass.BACKGROUND) == "server_1"
    assert router.route_task("filesystem", TaskPriorityClass.BACKGROUND) == "desktop_1"

    # QoS constraints: Route heavy compute task
    # Desktop has GPU, server has NPU.
    # If domain is unknown, it falls back to compute capable. Both capable, sorts by battery (1.0 default)
    # Desktop will be first or server based on sort stability.
    routed = router.route_task(
        "unknown_heavy", TaskPriorityClass.BACKGROUND, required_compute="high"
    )
    assert routed in ["desktop_1", "server_1"]

    # Check QoS drop: Background task on dying battery
    dying_phone = NodeProfile(
        "phone_2", NodeType.PHONE, NodeCapabilities(battery_level=0.1, is_charging=False)
    )
    registry.register_node(dying_phone)
    # Only dying phone available
    registry.nodes = {"phone_2": dying_phone}
    assert router.route_task("unknown", TaskPriorityClass.BACKGROUND) is None
    # But interactive task should still route
    assert router.route_task("unknown", TaskPriorityClass.INTERACTIVE) == "phone_2"
