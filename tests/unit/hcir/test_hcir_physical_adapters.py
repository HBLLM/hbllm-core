"""Unit tests for Physical & Sensor Capability Adapters."""

from hbllm.hcir.adapters.actuator_adapter import ActuatorCapabilityAdapter, ActuatorCommand
from hbllm.hcir.adapters.sensor_adapter import SensorCapabilityAdapter, SensorReading
from hbllm.hcir.adapters.telemetry_adapter import TelemetryStreamAdapter
from hbllm.hcir.graph import WorldVariableNode
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.types import BranchMode
from hbllm.hcir.workspace import HCIRWorkspaceState

# ═══════════════════════════════════════════════════════════════════════════
# Sensor Capability Adapter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSensorCapabilityAdapter:
    def test_ingest_sensor_reading(self):
        ws = HCIRWorkspaceState()
        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )

        adapter = SensorCapabilityAdapter(services)
        reading = SensorReading(
            sensor_id="temp_01",
            sensor_type="thermocouple",
            variable_name="temperature",
            value=25.8,
            unit="Celsius",
        )

        tx = adapter.ingest_reading(reading)
        assert tx is not None
        assert tx.status.value == "committed"

        wv = ws.get_node("wv_temperature")
        assert wv is not None
        assert isinstance(wv, WorldVariableNode)
        assert wv.value == 25.8


# ═══════════════════════════════════════════════════════════════════════════
# Actuator Capability Adapter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestActuatorCapabilityAdapter:
    def test_simulation_mode_suppresses_hardware_dispatch(self):
        ws = HCIRWorkspaceState(branch_mode=BranchMode.SIMULATION)
        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )

        adapter = ActuatorCapabilityAdapter(services)

        called = []
        adapter.register_hardware_handler("relay_01", lambda p: called.append(p) or True)

        cmd = ActuatorCommand(actuator_id="relay_01", command_name="turn_on")
        success = adapter.dispatch_command_sync(cmd, ws)

        assert success is True
        # Hardware handler should NOT be called in SIMULATION mode!
        assert len(called) == 0

    def test_live_mode_executes_hardware_handler(self):
        ws = HCIRWorkspaceState(branch_mode=BranchMode.LIVE)
        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )

        adapter = ActuatorCapabilityAdapter(services)

        called = []

        def handle_relay(params):
            called.append(params)
            return True

        adapter.register_hardware_handler("relay_01", handle_relay)

        cmd = ActuatorCommand(
            actuator_id="relay_01", command_name="turn_on", parameters={"voltage": 12}
        )
        success = adapter.dispatch_command_sync(cmd, ws)

        assert success is True
        assert len(called) == 1
        assert called[0] == {"voltage": 12}


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry Stream Adapter Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTelemetryStreamAdapter:
    def test_sliding_window_aggregation(self):
        ws = HCIRWorkspaceState()
        services = KernelServices(
            workspace=ws,
            transaction_manager=TransactionManager(ws),
            capability_resolver=CapabilityResolver(),
            scheduler=CognitiveScheduler(),
        )

        adapter = TelemetryStreamAdapter(
            services, environment_name="Greenhouse_Alpha", window_size=3
        )

        for val in [20.0, 22.0, 24.0]:
            adapter.push_reading(
                SensorReading(
                    sensor_id="s1",
                    sensor_type="temp",
                    variable_name="temperature",
                    value=val,
                )
            )

        # Window automatically flushes when size 3 is reached
        env_node = ws.get_node("env_state_Greenhouse_Alpha")
        assert env_node is not None
        assert "temperature" in env_node.active_variables
