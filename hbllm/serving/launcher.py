"""
Multi-Server Launcher — boots HBLLM nodes from cluster.yaml.

Supports two modes:
  1. Single-server (--server all): boots everything in one process with InProcessBus
  2. Multi-server (--server <role>): boots only that role's nodes with RedisBus

Usage:
  # Single-server (local dev / testing)
  python -m hbllm.serving.launcher --server all

  # Multi-server (production — run on each machine)
  python -m hbllm.serving.launcher --server gateway --config config/cluster.yaml
  python -m hbllm.serving.launcher --server domain_math --config config/cluster.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from hbllm.network.bus import InProcessBus
from hbllm.network.circuit_breaker import CircuitBreakerRegistry
from hbllm.network.cluster_config import ClusterConfig, load_cluster_config
from hbllm.network.health import HealthMonitor
from hbllm.network.load_balancer import LoadBalancer
from hbllm.network.node import Node
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


# ─── Node Factory ─────────────────────────────────────────────────────────────
# Maps node ID strings from cluster.yaml to actual Node instances.

def _create_node(node_id: str, config: ClusterConfig) -> Node | None:
    """
    Create a Node instance from its ID string.
    
    Supports:
      - Standard node IDs (router, workspace, planner, etc.)
      - Domain modules:  domain:<name> (e.g. domain:math)
    """
    # Domain module: domain:<name>
    if node_id.startswith("domain:"):
        domain_name = node_id.split(":", 1)[1]
        from hbllm.modules.base_module import DomainModuleNode
        return DomainModuleNode(
            node_id=f"domain_{domain_name}",
            domain=domain_name,
        )

    # Standard nodes
    node_map: dict[str, tuple[str, str]] = {
        # node_id → (module_path, class_name)
        "router":       ("hbllm.brain.router_node", "RouterNode"),
        "workspace":    ("hbllm.brain.workspace_node", "WorkspaceNode"),
        "planner":      ("hbllm.brain.planner_node", "PlannerNode"),
        "critic":       ("hbllm.brain.critic_node", "CriticNode"),
        "decision":     ("hbllm.brain.decision_node", "DecisionNode"),
        "meta":         ("hbllm.brain.meta_node", "MetaReasoningNode"),
        "identity":     ("hbllm.brain.identity_node", "IdentityNode"),
        "curiosity":    ("hbllm.brain.curiosity_node", "CuriosityNode"),
        "collective":   ("hbllm.brain.collective_node", "CollectiveNode"),
        "learner":      ("hbllm.brain.learner_node", "LearnerNode"),
        "sleep_cycle":  ("hbllm.brain.sleep_node", "SleepCycleNode"),
        "world_model":  ("hbllm.brain.world_model_node", "WorldModelNode"),
        "memory":       ("hbllm.memory.memory_node", "MemoryNode"),
    }

    if node_id not in node_map:
        logger.warning("Unknown node ID '%s' — skipping", node_id)
        return None

    module_path, class_name = node_map[node_id]

    try:
        import importlib
        module = importlib.import_module(module_path)
        node_class = getattr(module, class_name)
        return node_class(node_id=node_id)
    except Exception as e:
        logger.error("Failed to create node '%s': %s", node_id, e)
        return None


# ─── Server Bootstrap ─────────────────────────────────────────────────────────

class ServerInstance:
    """
    A running HBLLM server instance — manages bus, registry, nodes, and lifecycle.
    """

    def __init__(
        self,
        server_name: str,
        config: ClusterConfig,
    ):
        self.server_name = server_name
        self.config = config
        self.bus = None
        self.registry: ServiceRegistry | None = None
        self.circuit_breakers = CircuitBreakerRegistry(
            failure_threshold=config.node_defaults.circuit_breaker_threshold,
            recovery_timeout=config.node_defaults.circuit_breaker_recovery,
        )
        self.load_balancer: LoadBalancer | None = None
        self.health_monitor: HealthMonitor | None = None
        self.nodes: list[Node] = []
        self._running = False

    @property
    def is_single_server(self) -> bool:
        return self.server_name == "all"

    async def start(self) -> None:
        """Boot the server: create bus, registry, nodes, and start everything."""
        logger.info(
            "Starting HBLLM server '%s' (cluster=%s, mode=%s)",
            self.server_name,
            self.config.cluster.name,
            "single-server" if self.is_single_server else "multi-server",
        )

        # ── 1. Create bus ──
        if self.is_single_server:
            self.bus = InProcessBus()
        else:
            from hbllm.network.redis_bus import RedisBus
            self.bus = RedisBus(redis_url=self.config.cluster.redis_url)

        await self.bus.start()

        # ── 2. Create registry ──
        if self.is_single_server:
            self.registry = ServiceRegistry(
                health_check_interval=self.config.node_defaults.heartbeat_interval,
            )
        else:
            from hbllm.network.redis_registry import RedisRegistry
            self.registry = RedisRegistry(
                redis_url=self.config.cluster.redis_url,
                health_check_interval=self.config.node_defaults.heartbeat_interval,
            )

        await self.registry.start(self.bus)

        # ── 3. Create load balancer ──
        self.load_balancer = LoadBalancer(
            registry=self.registry,
            circuit_breakers=self.circuit_breakers,
            strategy=self.config.load_balancing.strategy,
            health_weight=self.config.load_balancing.health_weight,
            latency_weight=self.config.load_balancing.latency_weight,
        )

        # ── 4. Determine which nodes to boot ──
        if self.is_single_server:
            node_ids = self.config.get_all_nodes()
            if not node_ids:
                # No config — boot core nodes for local dev
                node_ids = [
                    "router", "workspace", "planner", "critic",
                    "decision", "meta", "memory", "identity",
                    "curiosity", "collective", "learner",
                ]
        else:
            server_config = self.config.get_server(self.server_name)
            node_ids = server_config.nodes

        # ── 5. Create and start nodes ──
        for node_id in node_ids:
            node = _create_node(node_id, self.config)
            if node is not None:
                try:
                    await node.start(self.bus)
                    await self.registry.register(node.get_info())
                    self.nodes.append(node)
                    logger.info("Started node: %s", node.node_id)
                except Exception as e:
                    logger.error("Failed to start node '%s': %s", node_id, e)

        # ── 6. Start health monitor ──
        self.health_monitor = HealthMonitor(
            registry=self.registry,
            circuit_breakers=self.circuit_breakers,
            bus=self.bus,
            check_interval=self.config.node_defaults.heartbeat_interval,
        )
        await self.health_monitor.start()

        self._running = True
        logger.info(
            "Server '%s' started with %d nodes",
            self.server_name,
            len(self.nodes),
        )

    async def stop(self) -> None:
        """Gracefully shut down all nodes and infrastructure."""
        logger.info("Stopping server '%s'...", self.server_name)
        self._running = False

        # Stop health monitor
        if self.health_monitor:
            await self.health_monitor.stop()

        # Stop nodes in reverse order
        for node in reversed(self.nodes):
            try:
                if self.registry:
                    await self.registry.deregister(node.node_id)
                await node.stop()
                logger.info("Stopped node: %s", node.node_id)
            except Exception as e:
                logger.error("Error stopping node '%s': %s", node.node_id, e)

        # Stop registry
        if self.registry:
            await self.registry.stop()

        # Stop bus
        if self.bus:
            await self.bus.stop()

        logger.info("Server '%s' stopped", self.server_name)

    def summary(self) -> str:
        """Return a human-readable summary of this server instance."""
        lines = [
            f"╔══ HBLLM Server: {self.server_name} ══",
            f"║ Cluster: {self.config.cluster.name}",
            f"║ Mode:    {'single-server' if self.is_single_server else 'multi-server'}",
            f"║ Bus:     {'InProcessBus' if self.is_single_server else 'RedisBus'}",
            f"║ Nodes:   {len(self.nodes)}",
        ]
        for node in self.nodes:
            lines.append(f"║   • {node.node_id} ({node.node_type.value})")
        lines.append("╚" + "═" * 40)
        return "\n".join(lines)


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

async def _run_server(server_name: str, config_path: str | None) -> None:
    config = load_cluster_config(config_path)
    server = ServerInstance(server_name, config)

    # Graceful shutdown on SIGTERM/SIGINT
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await server.start()
    print(server.summary())

    # Wait until shutdown signal
    await shutdown_event.wait()
    await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HBLLM Multi-Server Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m hbllm.serving.launcher --server all
  python -m hbllm.serving.launcher --server gateway --config config/cluster.yaml
  python -m hbllm.serving.launcher --validate --config config/cluster.yaml
  python -m hbllm.serving.launcher --list-nodes
        """,
    )
    parser.add_argument("--server", "-s", default=None,
                        help="Server role to boot, or 'all' for single-server")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to cluster.yaml")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--validate", action="store_true",
                        help="Validate cluster config and exit")
    parser.add_argument("--list-nodes", action="store_true",
                        help="List available node types and exit")
    parser.add_argument("--plugins", default=None,
                        help="Path to plugins directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_nodes:
        _handle_list_nodes()
        return

    if args.validate:
        _handle_validate(args.config)
        return

    if not args.server:
        parser.error("--server is required (use 'all' for single-server mode)")

    asyncio.run(_run_server(args.server, args.config))


def _handle_list_nodes() -> None:
    """Print all available node types."""
    node_map = {
        "router":        "RouterNode — LLM intent classification + routing",
        "workspace":     "WorkspaceNode — Global Workspace (blackboard)",
        "planner":       "PlannerNode — Graph-of-Thoughts DAG planner",
        "critic":        "CriticNode — Real-time thought evaluation",
        "decision":      "DecisionNode — Action gatekeeper + output",
        "meta":          "MetaReasoningNode — Strategy selection",
        "identity":      "IdentityNode — Persona + goals per tenant",
        "curiosity":     "CuriosityNode — Knowledge gap detection",
        "collective":    "CollectiveNode — Cross-instance sharing",
        "learner":       "LearnerNode — RLHF / DPO training loop",
        "sleep_cycle":   "SleepCycleNode — Memory consolidation",
        "world_model":   "WorldModelNode — Environment simulation",
        "memory":        "MemoryNode — Episodic + semantic storage",
        "domain:<name>": "DomainModuleNode — Domain expert (e.g. domain:math)",
    }
    print("╔══ Available HBLLM Node Types ══")
    for node_id, desc in node_map.items():
        print(f"║  {node_id:<16}  {desc}")
    print("╚" + "═" * 50)


def _handle_validate(config_path: str | None) -> None:
    """Validate cluster config and print report."""
    try:
        config = load_cluster_config(config_path)
    except FileNotFoundError as e:
        print(f"✗ Config file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Config parse error: {e}")
        sys.exit(1)

    errors = config.validate_topology()

    print(f"╔══ Cluster Validation: {config.cluster.name} ══")
    print(f"║ Redis:   {config.cluster.redis_url}")
    print(f"║ Servers: {len(config.servers)}")
    print(f"║ Nodes:   {len(config.get_all_nodes())}")

    all_nodes = config.get_all_nodes()
    import_ok = 0
    for node_id in all_nodes:
        node = _create_node(node_id, config)
        if node is not None:
            import_ok += 1
        else:
            errors.append(f"Cannot import node: {node_id}")

    print(f"║ Imports: {import_ok} ok, {len(all_nodes) - import_ok} failed")

    if errors:
        print("║ ✗ ERRORS:")
        for err in errors:
            print(f"║   • {err}")
        print("╚" + "═" * 50)
        sys.exit(1)
    else:
        print("║ ✓ Config is valid!")
        print("╚" + "═" * 50)


if __name__ == "__main__":
    main()

