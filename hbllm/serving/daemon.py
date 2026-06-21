"""
Cognitive Daemon — always-on HBLLM process.

Boots the full Brain with AutonomyCore running, keeps it alive as a
long-running daemon. Serves HTTP (FastAPI) while the cognitive heartbeat
ticks in the background.

Modes:
    foreground  — runs in current terminal (default)
    systemd     — journal-compatible logging, no interactive I/O

Usage::

    # Foreground (development)
    python -m hbllm.serving.daemon

    # With specific provider
    python -m hbllm.serving.daemon --provider openai/gpt-4o-mini

    # Local model
    python -m hbllm.serving.daemon --local --model-size 1.5b

    # As a systemd service
    python -m hbllm.serving.daemon --mode systemd
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
from typing import Any

logger = logging.getLogger(__name__)


class CognitiveDaemon:
    """Long-running cognitive process with autonomy heartbeat.

    Manages the full lifecycle:
        1. Boot Brain via BrainFactory
        2. Start AutonomyCore (cognitive heartbeat)
        3. Start ProactiveProcessor (output bridge)
        4. Optionally start HTTP server (FastAPI)
        5. Run until shutdown signal
        6. Graceful drain and persist state
    """

    def __init__(
        self,
        provider: str = "openai/gpt-4o-mini",
        *,
        local: bool = False,
        model_size: str = "125m",
        checkpoint_path: str | None = None,
        data_dir: str = "data",
        host: str = "0.0.0.0",
        port: int = 8000,
        serve_http: bool = True,
        enable_perception: bool = True,
        provider_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.provider = provider
        self.local = local
        self.model_size = model_size
        self.checkpoint_path = checkpoint_path
        self.data_dir = data_dir
        self.host = host
        self.port = port
        self.serve_http = serve_http
        self.enable_perception = enable_perception
        self.provider_kwargs = provider_kwargs or {}

        self._brain: Any = None
        self._autonomy: Any = None
        self._proactive: Any = None
        self._server_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._boot_time = 0.0

    async def start(self) -> None:
        """Boot the full cognitive system."""
        self._boot_time = time.monotonic()

        logger.info("═" * 60)
        logger.info("  HBLLM Cognitive Daemon starting...")
        logger.info("═" * 60)

        # ── 1. Create Brain ──────────────────────────────────────
        from hbllm.brain.factory import BrainConfig, BrainFactory

        config = BrainConfig(
            data_dir=self.data_dir,
            inject_awareness=True,
            inject_perception=self.enable_perception,
        )

        if self.enable_perception:
            logger.info("Perception ENABLED — audio/vision nodes will be injected")
        else:
            logger.info("Perception DISABLED — use --no-perception to re-enable")

        if self.local:
            self._brain = await BrainFactory.create_local(
                checkpoint_path=self.checkpoint_path,
                model_size=self.model_size,
                config=config,
            )
        else:
            self._brain = await BrainFactory.create(
                provider=self.provider,
                config=config,
                **self.provider_kwargs,
            )

        logger.info("Brain created with %d nodes", len(self._brain.nodes))

        # ── 2. Start AutonomyCore ────────────────────────────────
        from hbllm.brain.autonomy.loop import AutonomyCore

        self._autonomy = AutonomyCore()

        # Register default reflex rules
        self._register_default_reflexes()

        # Register default proactive handlers
        await self._register_default_handlers()

        # Start the heartbeat
        await self._autonomy.start(self._brain.bus)

        # Store reference on brain
        self._brain.autonomy_core = self._autonomy

        logger.info("AutonomyCore started — cognitive heartbeat active")

        # ── 2b. Wire core safety subsystems into AutonomyCore ─────────────
        if self._brain.restraint_engine:
            self._autonomy.set_restraint_engine(self._brain.restraint_engine)
        if self._brain.audit_trail:
            self._autonomy.set_audit_trail(self._brain.audit_trail)

        # Wire NotificationSuppressor if available
        try:
            from hbllm.brain.autonomy.notification_suppressor import (
                NotificationSuppressor,
            )

            suppressor = NotificationSuppressor()
            self._autonomy.set_notification_suppressor(suppressor)
            self._brain._notification_suppressor = suppressor
            logger.info("NotificationSuppressor wired into AutonomyCore")
        except Exception as e:
            logger.debug("NotificationSuppressor not wired: %s", e)

        # Wire PII redactor into memory nodes
        if self._brain.pii_redactor:
            for node in self._brain.nodes:
                if hasattr(node, "_pii_redactor"):
                    node._pii_redactor = self._brain.pii_redactor
                    logger.debug("PII redactor injected into %s", node.node_id)

        # ── 3. Start ProactiveProcessor ──────────────────────────
        from hbllm.serving.notifications import NotificationGateway
        from hbllm.serving.proactive import ProactiveProcessor, SSEChannel

        gateway = NotificationGateway()
        sse = SSEChannel()

        self._proactive = ProactiveProcessor(
            gateway=gateway,
            pipeline=self._brain.pipeline,
            sse_channel=sse,
        )
        await self._proactive.start(self._brain.bus)

        # Wire autonomy → proactive processor
        self._autonomy.set_action_handler(self._route_autonomy_action)

        # Store references on brain
        self._brain.notification_gateway = gateway
        self._brain.proactive_processor = self._proactive
        self._brain.sse_channel = sse

        logger.info("ProactiveProcessor started — notifications active")

        # ── 4. Optionally start HTTP server ──────────────────────
        if self.serve_http:
            self._server_task = asyncio.create_task(self._start_http_server())

        elapsed = time.monotonic() - self._boot_time
        logger.info("═" * 60)
        logger.info("  Cognitive Daemon ready (%.1fs boot)", elapsed)
        logger.info("  Autonomy: ACTIVE | Pipeline: READY")
        if self.serve_http:
            logger.info("  HTTP: http://%s:%d", self.host, self.port)
        logger.info("═" * 60)

    async def _route_autonomy_action(self, msg: Any) -> None:
        """Route autonomy actions through the bus for ProactiveProcessor."""
        if self._brain and self._brain.bus:
            await self._brain.bus.publish(msg.topic, msg)

    async def stop(self) -> None:
        """Graceful shutdown with state persistence."""
        logger.info("Cognitive Daemon shutting down...")

        # Stop HTTP server
        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        # Stop ProactiveProcessor
        if self._proactive:
            await self._proactive.stop()

        # Stop AutonomyCore
        if self._autonomy:
            await self._autonomy.stop()

        # Shutdown Brain (persists state, drains requests)
        if self._brain:
            # Stop OfflineManager
            if getattr(self._brain, "offline_manager", None):
                await self._brain.offline_manager.stop()

            # Prune old audit entries
            if getattr(self._brain, "audit_trail", None):
                pruned = self._brain.audit_trail.prune_old_entries()
                if pruned:
                    logger.info("Pruned %d old audit entries", pruned)

            await self._brain.shutdown()

        elapsed = time.monotonic() - self._boot_time
        logger.info("Cognitive Daemon stopped (uptime: %.0fs)", elapsed)

    async def run_forever(self) -> None:
        """Run until shutdown signal (SIGTERM/SIGINT)."""
        loop = asyncio.get_running_loop()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received")
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

        await self.start()

        # Wait for shutdown
        await self._shutdown_event.wait()
        await self.stop()

    async def _start_http_server(self) -> None:
        """Start FastAPI/uvicorn in the background."""
        try:
            import uvicorn

            from hbllm.serving.api import create_app

            app = create_app(brain=self._brain)

            config = uvicorn.Config(
                app=app,
                host=self.host,
                port=self.port,
                log_level="info",
                access_log=False,
            )
            server = uvicorn.Server(config)
            await server.serve()
        except ImportError:
            logger.warning(
                "uvicorn or fastapi not installed — HTTP server disabled. "
                "Install with: pip install uvicorn fastapi"
            )
        except Exception:
            logger.exception("HTTP server failed")

    def _register_default_reflexes(self) -> None:
        """Register built-in reflex rules from the reflex library."""
        from hbllm.brain.autonomy.reflexes import get_all_reflexes

        # Load all reflexes from the library (system, environment, routine, security)
        all_reflexes = get_all_reflexes()
        for name, rule in all_reflexes.items():
            self._autonomy.add_reflex(name, rule)

        logger.info("Registered %d reflexes from library", len(all_reflexes))

    async def _register_default_handlers(self) -> None:
        """Register built-in proactive handlers (slow path)."""
        from hbllm.network.messages import Message, MessageType

        async def _goal_progress_check() -> list[Message] | None:
            """Check active goals and report progress."""
            if not self._brain or not self._brain.goal_manager:
                return None

            goals = self._brain.goal_manager.get_active_goals()
            messages = []
            for goal in goals[:3]:  # Max 3 goal updates per tick
                progress = getattr(goal, "progress", 0.0)
                if progress > 0 and progress < 1.0:
                    messages.append(
                        Message(
                            type=MessageType.EVENT,
                            source_node_id="autonomy_proactive",
                            topic="proactive.push",
                            payload={
                                "title": f"Goal: {goal.name}",
                                "body": f"Progress: {progress:.0%}",
                                "priority": "suggestion",
                                "category": "goal",
                                "metadata": {"goal_id": goal.goal_id},
                            },
                        )
                    )
            return messages if messages else None

        async def _awareness_digest() -> list[Message] | None:
            """Periodic awareness summary from CognitiveAwareness."""
            if not self._brain or not self._brain.awareness:
                return None

            awareness = self._brain.awareness
            if hasattr(awareness, "get_recent_patterns"):
                patterns = awareness.get_recent_patterns()
                if patterns:
                    summary = "; ".join(p.get("description", str(p)) for p in patterns[:3])
                    return [
                        Message(
                            type=MessageType.EVENT,
                            source_node_id="autonomy_proactive",
                            topic="proactive.push",
                            payload={
                                "title": "Activity Insight",
                                "body": summary,
                                "priority": "suggestion",
                                "category": "insight",
                            },
                        )
                    ]
            return None

        self._autonomy.add_proactive_handler("goal_progress", _goal_progress_check)
        self._autonomy.add_proactive_handler("awareness_digest", _awareness_digest)

        logger.debug("Registered %d default proactive handlers", 2)

    # ── Introspection ─────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Full daemon telemetry snapshot."""
        result: dict[str, Any] = {
            "uptime_s": round(time.monotonic() - self._boot_time, 2),
            "brain_nodes": len(self._brain.nodes) if self._brain else 0,
        }
        if self._autonomy:
            result["autonomy"] = self._autonomy.snapshot()
        if self._proactive:
            result["proactive"] = self._proactive.snapshot()
        if self._brain and self._brain.cognitive_metrics:
            result["metrics"] = self._brain.cognitive_metrics.get_dashboard_metrics()

        # Core safety subsystem stats
        if self._brain:
            safety: dict[str, Any] = {}
            if getattr(self._brain, "pii_redactor", None):
                safety["pii_redactor"] = self._brain.pii_redactor.stats()
            if getattr(self._brain, "restraint_engine", None):
                safety["restraint"] = self._brain.restraint_engine.stats()
            if getattr(self._brain, "audit_trail", None):
                safety["audit_trail"] = self._brain.audit_trail.stats()
            if getattr(self._brain, "offline_manager", None):
                safety["offline"] = self._brain.offline_manager.stats()
            if getattr(self._brain, "voice_auth", None):
                safety["voice_auth"] = self._brain.voice_auth.stats()
            if getattr(self._brain, "rollback_registry", None):
                safety["rollback"] = self._brain.rollback_registry.stats()
            if safety:
                result["safety"] = safety

        return result


# ── CLI Entry Point ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HBLLM Cognitive Daemon — always-on cognitive process",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("HBLLM_PROVIDER", "openai/gpt-4o-mini"),
        help="LLM provider (e.g., openai/gpt-4o-mini, anthropic/claude-sonnet-4-20250514)",
    )
    parser.add_argument("--local", action="store_true", help="Use local model")
    parser.add_argument("--model-size", default="125m", help="Local model size")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path")
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("HBLLM_DATA_DIR", "data"),
        help="Data directory",
    )
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port")
    parser.add_argument("--no-http", action="store_true", help="Disable HTTP server")
    parser.add_argument(
        "--no-perception",
        action="store_true",
        help="Disable perception nodes (audio/vision) — useful on headless servers",
    )
    parser.add_argument(
        "--mode",
        choices=["foreground", "systemd"],
        default="foreground",
        help="Run mode",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    args = parser.parse_args()

    # Configure logging
    log_format = (
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if args.mode == "foreground"
        else "%(levelname)s %(name)s: %(message)s"  # systemd journal
    )
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format=log_format,
    )

    daemon = CognitiveDaemon(
        provider=args.provider,
        local=args.local,
        model_size=args.model_size,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        serve_http=not args.no_http,
        enable_perception=not args.no_perception,
    )

    asyncio.run(daemon.run_forever())


if __name__ == "__main__":
    main()
