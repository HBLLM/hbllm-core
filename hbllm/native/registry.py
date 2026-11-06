"""Native Acceleration Registry & Adaptive Cost-Model Dispatcher for HBLLM.

Provides:
1. Centralized discovery and access to native Rust-accelerated execution kernels.
2. Dynamic WorkloadProfile and AdaptiveDispatcher comparing expected execution costs.
3. Graceful pure-Python fallback when native binaries are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NativeCapabilityInfo:
    name: str
    available: bool
    description: str
    native_module: str | None = None


@dataclass(frozen=True)
class WorkloadProfile:
    """Quantitative characterization of an HCIR workload for cost-model routing."""

    batch_size: int = 1
    state_node_count: int = 0
    has_geometric_collision: bool = False
    action_complexity: int = 1
    estimated_prefix_sharing: float = 0.0


class AdaptiveDispatcher:
    """Workload-aware cost-model dispatcher determining whether to execute in Python or Native Rust."""

    def __init__(self) -> None:
        # Calibrated baseline cost constants (microseconds)
        self.py_base_cost_us = 1.5
        self.py_step_cost_us = 0.8
        self.py_geom_step_cost_us = 60.0  # Emulated Python AABB spatial collision & center-of-mass

        self.rust_ffi_base_cost_us = 1200.0  # PyDict / PyList argument extraction & FFI crossing
        self.rust_step_cost_us = 0.05
        self.rust_geom_step_cost_us = 0.5

    def estimate_python_cost_us(self, operation: str, profile: WorkloadProfile) -> float:
        """Estimate Python execution time in microseconds."""
        if operation == "simulation":
            step_cost = (
                self.py_geom_step_cost_us
                if profile.has_geometric_collision
                else self.py_step_cost_us
            )
            return self.py_base_cost_us + profile.batch_size * profile.action_complexity * step_cost
        elif operation in ("snapshot", "hcir_graph"):
            return max(1.0, profile.state_node_count * 0.045)  # deepcopy cost
        elif operation == "canonical_hash":
            return profile.state_node_count * 0.012
        elif operation == "structure_matcher":
            return 25.0 * (profile.action_complexity**2)
        return 1.0

    def estimate_native_cost_us(self, operation: str, profile: WorkloadProfile) -> float:
        """Estimate Native Rust execution time in microseconds."""
        if operation == "simulation":
            step_cost = (
                self.rust_geom_step_cost_us
                if profile.has_geometric_collision
                else self.rust_step_cost_us
            )
            effective_branches = max(
                1.0, profile.batch_size * (1.0 - profile.estimated_prefix_sharing)
            )
            parallel_factor = max(1.0, min(8.0, profile.batch_size / 2.0))
            return (
                self.rust_ffi_base_cost_us
                + (effective_branches * profile.action_complexity * step_cost) / parallel_factor
            )
        elif operation in ("snapshot", "hcir_graph"):
            return 0.005  # O(1) root pointer clone
        elif operation == "canonical_hash":
            return 0.5 + profile.state_node_count * 0.002
        elif operation == "structure_matcher":
            return 2.0 + 0.8 * profile.action_complexity
        return 1.0

    def should_native_execute(
        self, operation: str, profile: WorkloadProfile, is_native_available: bool
    ) -> bool:
        """Determines if the native path is computationally cheaper than pure Python."""
        if not is_native_available:
            return False
        py_cost = self.estimate_python_cost_us(operation, profile)
        rust_cost = self.estimate_native_cost_us(operation, profile)
        return rust_cost < py_cost


class NativeAccelerationRegistry:
    """Centralized registry for high-performance native HCIR substrate capabilities."""

    _instance: NativeAccelerationRegistry | None = None

    def __init__(self) -> None:
        self._capabilities: dict[str, NativeCapabilityInfo] = {}
        self._dispatcher = AdaptiveDispatcher()
        self._discover_capabilities()

    @classmethod
    def get_instance(cls) -> NativeAccelerationRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _discover_capabilities(self) -> None:
        crates = [
            (
                "hcir_graph",
                "hbllm_hcir_graph",
                "Persistent HCIR Graph with chunk-granular structural sharing and canonical BLAKE3 hashing",
            ),
            (
                "simulation",
                "hbllm_simulation_engine",
                "Rayon multi-threaded counterfactual mental simulation sandbox with geometric stability",
            ),
            (
                "structure_matcher",
                "hbllm_structure_matcher",
                "Bitset-accelerated analogical subgraph isomorphism and systematicity alignment",
            ),
            (
                "tokenizer",
                "hbllm_tokenizer_rs",
                "SIMD-accelerated Byte-Pair Encoding (BPE) vocab tokenizer",
            ),
            (
                "compute_kernel",
                "hbllm_compute_kernel",
                "Vectorized tensor and matrix arithmetic compute kernels",
            ),
            (
                "semantic_search",
                "hbllm_semantic_search",
                "Accelerated vector search and cosine similarity ranking",
            ),
            (
                "perception",
                "hbllm_perception",
                "Fast 2D/3D visual and spatial perception rasterization kernels",
            ),
            (
                "confidence",
                "hbllm_confidence",
                "Bayesian confidence calibration and distribution estimation",
            ),
            (
                "data_tools",
                "hbllm_data_tools",
                "Zero-copy streaming JSON/Arrow serialization and data ingest",
            ),
            (
                "policy_eval",
                "hbllm_policy_eval",
                "High-throughput rule matching and safety policy evaluation",
            ),
            (
                "concept_extract",
                "hbllm_concept_extract",
                "Fast keyword and n-gram lexical concept extraction",
            ),
            (
                "knowledge_graph",
                "hbllm_knowledge_graph",
                "High-performance graph traversal and entity relation queries",
            ),
            (
                "network_utils",
                "hbllm_network_utils",
                "Optimized packet serialization and protocol encoding",
            ),
        ]

        import importlib

        for cap_name, mod_name, desc in crates:
            available = False
            try:
                importlib.import_module(mod_name)
                available = True
            except ImportError:
                pass

            self._capabilities[cap_name] = NativeCapabilityInfo(
                name=cap_name,
                available=available,
                description=desc,
                native_module=mod_name if available else None,
            )

    def available(self, capability: str) -> bool:
        """Check if a native acceleration capability is available."""
        cap = self._capabilities.get(capability)
        return cap.available if cap is not None else False

    def should_execute_native(
        self,
        capability: str,
        profile: WorkloadProfile | None = None,
    ) -> bool:
        """Cost-model routing query determining whether to use Python or Native Rust path."""
        if not self.available(capability):
            return False
        if profile is None:
            return True
        return self._dispatcher.should_native_execute(capability, profile, True)

    def get_info(self, capability: str) -> NativeCapabilityInfo | None:
        """Get capability metadata."""
        return self._capabilities.get(capability)

    def list_capabilities(self) -> dict[str, NativeCapabilityInfo]:
        """List all registered capabilities and their availability status."""
        return dict(self._capabilities)

    def status_summary(self) -> str:
        """Format human-readable capability status table."""
        lines = [
            "Native Acceleration Capabilities:",
            f"{'Capability':<20} {'Status':<12} {'Module':<25} {'Description'}",
            "-" * 85,
        ]
        for name, info in sorted(self._capabilities.items()):
            status = "AVAILABLE" if info.available else "UNAVAILABLE"
            module = info.native_module or "(pure Python fallback)"
            lines.append(f"{name:<20} {status:<12} {module:<25} {info.description}")
        return "\n".join(lines)


# Singleton instance for quick access
native = NativeAccelerationRegistry.get_instance()
