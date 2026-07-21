import logging

import pytest

logger = logging.getLogger(__name__)

from hbllm.hcir.adapters.sensor_adapter import SensorCapabilityAdapter, SensorReading
from hbllm.hcir.adapters.snn_adapter import SNNCapabilityExecutor, SNNPopulationConfig
from hbllm.hcir.compiler_llm import LLMCompilerFrontend
from hbllm.hcir.counterfactual_planner import CounterfactualPlanner
from hbllm.hcir.delta_transport import DeltaTransportProtocol
from hbllm.hcir.graph import ActionNode, GoalNode
from hbllm.hcir.interpreter import HCIRInterpreter
from hbllm.hcir.kernel.capability_resolver import CapabilityResolver
from hbllm.hcir.kernel.scheduler import CognitiveScheduler
from hbllm.hcir.kernel.services import KernelServices
from hbllm.hcir.kernel.transaction_manager import TransactionManager
from hbllm.hcir.learning_loop import LearningLoopEngine
from hbllm.hcir.optimizer import HCIROptimizer
from hbllm.hcir.workspace import HCIRWorkspaceState


class TestHCIREndToEndSwarm:
    @pytest.mark.asyncio
    async def test_full_cognitive_loop_and_swarm_sync(self):
        # 1. Initialize local Node A workspace & kernel services
        ws_a = HCIRWorkspaceState()
        tx_mgr_a = TransactionManager(ws_a)
        resolver_a = CapabilityResolver()
        scheduler_a = CognitiveScheduler()
        services_a = KernelServices(
            workspace=ws_a,
            transaction_manager=tx_mgr_a,
            capability_resolver=resolver_a,
            scheduler=scheduler_a,
        )

        # 2. Ingest sensor reading via SensorCapabilityAdapter
        sensor_adapter = SensorCapabilityAdapter(services_a)
        reading = SensorReading(
            sensor_id="temp_01",
            sensor_type="thermocouple",
            variable_name="temperature",
            value=28.5,
            unit="Celsius",
        )
        sensor_tx = sensor_adapter.ingest_reading(reading)
        assert sensor_tx is not None
        assert ws_a.get_node("wv_temperature") is not None

        # 3. LLM Compiler Frontend: Structured Intent -> Bytecode
        llm_compiler = LLMCompilerFrontend()
        json_intent = """{
            "intent": "plan",
            "subject": "greenhouse_cooling",
            "action": "trigger_cooling_fan",
            "slots": {"target_temp": 24.0},
            "priority": 0.9
        }"""
        stream = llm_compiler.compile_json(json_intent, author="qwen")
        assert stream.length > 0

        # 4. Optimize bytecode via HCIROptimizer
        optimizer = HCIROptimizer()
        opt_stream = optimizer.optimize(stream)
        assert opt_stream.length > 0

        # 5. Execute stream with ExecutionReceipt
        interpreter = HCIRInterpreter(ws_a, services_a)
        res, receipt = await interpreter.execute_with_receipt(
            opt_stream, process_id="proc_e2e", thread_id="thr_e2e"
        )
        assert res.success is True
        assert receipt.execution_id.startswith("rcpt_")

        # 6. Counterfactual Predictive Planning (FORK -> SIMULATE -> MERGE)
        cf_planner = CounterfactualPlanner(ws_a, services_a)
        goal = GoalNode(id="g_cooling", description="Maintain optimal temperature")
        candidates = [
            ActionNode(id="act_fan", intent="turn_on_fan", estimated_cost=10),
            ActionNode(id="act_mist", intent="trigger_misting", estimated_cost=80),
        ]
        best_candidate = await cf_planner.evaluate_and_select(goal, candidates)
        assert best_candidate.action.id == "act_fan"

        # 7. SNN Cognitive Accelerator Execution
        snn_executor = SNNCapabilityExecutor(SNNPopulationConfig(num_neurons=8))
        snn_res = await snn_executor.execute(None, ws_a, services_a)
        assert snn_res.success is True
        assert len(snn_res.delta.add_nodes) == 1

        # 8. Learning Loop: Experience -> Skill Node Induction
        learning_engine = LearningLoopEngine(ws_a, transaction_manager=tx_mgr_a)
        skill_tx = learning_engine.process_and_learn(receipt, user_reward=1.0)
        assert skill_tx is not None

        # 9. Distributed Swarm Transport (Node A -> Node B sync)
        transport_a = DeltaTransportProtocol(device_id="Node_A", secret_key="swarm_key")
        transport_b = DeltaTransportProtocol(device_id="Node_B", secret_key="swarm_key")

        delta = snn_res.delta
        packet = transport_a.create_packet(delta, target_device_id="Node_B")

        ws_b = HCIRWorkspaceState()
        sync_success = transport_b.verify_and_apply(packet.to_dict(), ws_b)
        assert sync_success is True

        logger.info("End-to-end cognitive loop benchmark completed successfully.")
