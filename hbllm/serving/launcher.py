"""
Multi-Server Launcher — boots HBLLM HCIR Cognitive OS and memory/domain nodes.

Supports single-server (--server all, InProcessBus) and multi-server (--server <role>, RedisBus).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.circuit_breaker import CircuitBreakerRegistry
from hbllm.network.cluster_config import ClusterConfig, load_cluster_config
from hbllm.network.health import HealthMonitor
from hbllm.network.load_balancer import LoadBalancer
from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType
from hbllm.network.registry import ServiceRegistry

logger = logging.getLogger(__name__)


class HCIRCognitiveNode(Node):
    """Distributed bus node hosting a canonical Pure-HCIR ``CognitiveSystem``."""

    def __init__(self, node_id: str = "hcir") -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.CORE,
            capabilities=["hcir", "reasoning", "dialogue", "planning"],
        )
        self._system: Any = None
        self._sub_query: Any = None
        self._sub_workspace: Any = None

    async def on_start(self) -> None:
        from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

        self._system = build_cognitive_system(
            CognitiveSystemConfig(world_id=f"node_{self.node_id}", memory_dir=None)
        )
        if self._system.runtime is not None:
            await self._system.runtime.start()
        self._sub_query = await self.bus.subscribe("router.query", self.handle_message)
        self._sub_workspace = await self.bus.subscribe("workspace.update", self.handle_message)

    async def on_stop(self) -> None:
        if self._sub_query is not None:
            await self.bus.unsubscribe(self._sub_query)
        if self._sub_workspace is not None:
            await self.bus.unsubscribe(self._sub_workspace)
        if self._system is not None and getattr(self._system, "runtime", None) is not None:
            await self._system.runtime.stop()

    async def handle_message(self, message: Message) -> Message | None:
        if self._system is None:
            return None
        text = str(message.payload.get("text", "") or "")
        if not text:
            return None
        result = await asyncio.to_thread(
            self._system.process_dialogue,
            text,
            speaker_id=message.source_node_id or "human",
        )
        resp_text = str(getattr(result, "response_text", "") or "")
        await self.bus.publish(
            "sensory.output",
            Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                correlation_id=message.correlation_id,
                topic="sensory.output",
                payload={"text": resp_text, "source": f"hcir:{self.node_id}"},
            ),
        )
        return message.create_response({"text": resp_text, "source": f"hcir:{self.node_id}"})


def _create_node(node_id: str, config: ClusterConfig) -> Node | None:
    """Create a Node instance from its ID string."""
    if node_id.startswith("domain:"):
        domain_name = node_id.split(":", 1)[1]
        from hbllm.modules.base_module import DomainModuleNode

        return DomainModuleNode(
            node_id=f"domain_{domain_name}",
            domain_name=domain_name,
            model=None,  # type: ignore[arg-type]
            tokenizer=None,
        )

    if node_id == "memory":
        from hbllm.memory.memory_node import MemoryNode

        return MemoryNode(node_id=node_id)

    hcir_roles = {
        "hcir",
        "router",
        "workspace",
        "planner",
        "critic",
        "decision",
        "meta",
        "identity",
        "curiosity",
        "collective",
        "learner",
        "sleep_cycle",
        "world_model",
    }
    if node_id in hcir_roles:
        return HCIRCognitiveNode(node_id=node_id)

    logger.warning("Unknown node ID '%s' — skipping", node_id)
    return None


class ServerInstance:
    """A running HBLLM server instance — manages bus, registry, nodes, and lifecycle."""

    def __init__(self, server_name: str, config: ClusterConfig) -> None:
        self.server_name = server_name
        self.config = config
        self.bus: MessageBus | None = None
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
        if self.is_single_server:
            self.bus = InProcessBus()
        else:
            from hbllm.network.redis_bus import RedisBus

            self.bus = RedisBus(redis_url=self.config.cluster.redis_url)

        await self.bus.start()

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

        self.load_balancer = LoadBalancer(
            registry=self.registry,
            circuit_breakers=self.circuit_breakers,
            strategy=self.config.load_balancing.strategy,
            health_weight=self.config.load_balancing.health_weight,
            latency_weight=self.config.load_balancing.latency_weight,
        )

        if self.is_single_server:
            node_ids = self.config.get_all_nodes() or ["hcir", "memory"]
        else:
            server_config = self.config.get_server(self.server_name)
            node_ids = server_config.nodes

        if self.bus is None:
            raise RuntimeError("MessageBus failed to initialize.")

        for node_id in node_ids:
            node = _create_node(node_id, self.config)
            if node is not None:
                try:
                    await node.start(self.bus)
                    await self.registry.register(node.get_info())
                    self.nodes.append(node)
                except Exception as e:
                    logger.error("Failed to start node '%s': %s", node_id, e)

        self.health_monitor = HealthMonitor(
            registry=self.registry,
            circuit_breakers=self.circuit_breakers,
            bus=self.bus,
            check_interval=self.config.node_defaults.heartbeat_interval,
        )
        await self.health_monitor.start()
        self._running = True

    async def stop(self) -> None:
        """Gracefully shut down all nodes and infrastructure."""
        self._running = False
        if self.health_monitor:
            await self.health_monitor.stop()
        for node in reversed(self.nodes):
            try:
                if self.registry:
                    await self.registry.deregister(node.node_id)
                await node.stop()
            except Exception as e:
                logger.error("Error stopping node '%s': %s", node.node_id, e)
        if self.registry:
            await self.registry.stop()
        if self.bus:
            await self.bus.stop()

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


async def _run_server(server_name: str, config_path: str | None) -> None:
    config = load_cluster_config(config_path)
    server = ServerInstance(server_name, config)

    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await server.start()
    print(server.summary())
    await shutdown_event.wait()
    await server.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="HBLLM Multi-Server HCIR Launcher")
    parser.add_argument("--server", "-s", default=None)
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--list-nodes", action="store_true")
    parser.add_argument("--plugins", default=None)

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
        "hcir": "HCIRCognitiveNode — Canonical Pure-HCIR Cognitive OS runtime",
        "memory": "MemoryNode — Episodic + semantic storage",
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
    except Exception as e:
        print(f"✗ Config error: {e}")
        sys.exit(1)

    errors = config.validate_topology()
    all_nodes = config.get_all_nodes()
    import_ok = 0
    for node_id in all_nodes:
        if _create_node(node_id, config) is not None:
            import_ok += 1
        else:
            errors.append(f"Cannot import node: {node_id}")

    print(f"Imports: {import_ok} ok, {len(all_nodes) - import_ok} failed")
    if errors:
        sys.exit(1)
    print("Config is valid!")


if __name__ == "__main__":
    main()
