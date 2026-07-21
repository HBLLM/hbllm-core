"""
SNN Cognitive Accelerator Adapter — wires Spiking Neural Networks as HCIR capability executors.

Provides hardware/neuromorphic spike-train acceleration for cognitive subgraphs:

    HCIR Subgraph (ObservationNodes / WorldVariables)
                        │
             SNNCapabilityExecutor
                        ├── Spike Encoder (Payload floats → Spike Trains)
                        ├── LIF Neuron Population Simulation (Spike Propagation)
                        └── Spike Decoder (Spike Firing Frequencies → HCIRDelta)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from hbllm.hcir.abi import ExecutionMetrics, ExecutionResult, ICognitiveNodeABI
from hbllm.hcir.graph import BeliefNode, NodeLifecycle
from hbllm.hcir.transactions import HCIRDelta
from hbllm.hcir.types import Provenance, Scope, UncertaintyVector

logger = logging.getLogger(__name__)


@dataclass
class SpikeTrain:
    """A temporal sequence of neuronal spikes."""

    neuron_id: int
    spike_times_ms: list[float] = field(default_factory=list)
    membrane_potential: float = 0.0


@dataclass
class SNNPopulationConfig:
    """Configuration for a Leaky Integrate-and-Fire (LIF) SNN population."""

    num_neurons: int = 16
    v_thresh: float = 1.0
    v_reset: float = 0.0
    tau_m: float = 10.0  # Membrane time constant (ms)
    leak_rate: float = 0.1


class SNNCapabilityExecutor(ICognitiveNodeABI):
    """Executes SNN spike-train reasoning on HCIR graph subgraphs.

    Usage::

        snn_adapter = SNNCapabilityExecutor(config=SNNPopulationConfig())
        res = await snn_adapter.execute(transaction, workspace, services)
    """

    supported_hcir_versions = ["1.0.0"]
    required_kernel_services = ["CapabilityResolver", "TransactionManager"]
    declared_capabilities = ["snn_reasoning", "spiking_accelerator"]

    def __init__(self, config: SNNPopulationConfig | None = None) -> None:
        self._config = config or SNNPopulationConfig()

    def encode_nodes_to_spikes(self, nodes: Any) -> list[SpikeTrain]:
        """Encode graph node values into temporal spike trains using rate coding."""
        node_list = list(nodes)
        spike_trains: list[SpikeTrain] = []
        for idx, node in enumerate(node_list[: self._config.num_neurons]):
            # Extract numeric value from node payload or uncertainty
            val = 0.5
            if hasattr(node, "value") and isinstance(node.value, (int, float)):
                val = float(node.value)
            elif hasattr(node, "uncertainty"):
                val = node.uncertainty.confidence

            # Rate coding: higher value = more frequent spikes
            num_spikes = int(val * 10)
            spike_times = [t * 2.0 for t in range(num_spikes)]
            spike_trains.append(
                SpikeTrain(
                    neuron_id=idx,
                    spike_times_ms=spike_times,
                    membrane_potential=val,
                )
            )

        return spike_trains

    def simulate_lif_population(self, inputs: list[SpikeTrain]) -> list[float]:
        """Simulate Leaky Integrate-and-Fire (LIF) neuron population step."""
        firing_rates: list[float] = []

        for st in inputs:
            v = 0.0
            spikes = 0
            for t in range(20):  # 20ms simulation window
                # Integration with leak
                v = v * (1.0 - self._config.leak_rate) + (st.membrane_potential * 0.2)
                if v >= self._config.v_thresh:
                    spikes += 1
                    v = self._config.v_reset

            firing_rate = spikes / 20.0
            firing_rates.append(firing_rate)

        return firing_rates

    async def execute(
        self,
        transaction: Any,
        workspace: Any,
        services: Any,
    ) -> ExecutionResult:
        """ABI execution contract for SNN reasoning."""
        start_time = time.monotonic()

        # Extract target nodes from workspace graph
        nodes = workspace.graph.all_nodes()
        spike_trains = self.encode_nodes_to_spikes(nodes)
        firing_rates = self.simulate_lif_population(spike_trains)

        avg_firing_rate = sum(firing_rates) / len(firing_rates) if firing_rates else 0.5

        delta = HCIRDelta()
        # Decode firing rate into an SNN-derived BeliefNode
        snn_belief = BeliefNode(
            id=f"belief_snn_{int(time.time())}",
            claim=f"SNN population firing equilibrium rate: {avg_firing_rate:.3f}",
            belief_type="neural_equilibrium",
            lifecycle=NodeLifecycle.ACTIVE,
            uncertainty=UncertaintyVector(confidence=min(1.0, avg_firing_rate * 2.0)),
            provenance=Provenance(created_by="snn_capability_executor", source_type="inferred"),
            scope=Scope(tenant_id="default"),
            tags=["snn_reasoning", "spiking_neural_network"],
        )
        delta.add_nodes.append(snn_belief.model_dump())

        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        logger.info(
            "SNN capability execution completed in %d ms (avg rate=%.3f)",
            elapsed_ms,
            avg_firing_rate,
        )

        return ExecutionResult(
            delta=delta,
            metrics=ExecutionMetrics(elapsed_ms=elapsed_ms, tokens_consumed=5),
            success=True,
        )
