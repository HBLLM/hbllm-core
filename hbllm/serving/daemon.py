"""
Cognitive Daemon — always-on HBLLM HCIR Cognitive OS process.

Boots the canonical HCIR CognitiveSystem via ``build_cognitive_system()``,
starts ``ExecutiveRuntime`` and ``ProactiveProcessor``, and optionally serves HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CognitiveDaemon:
    """Long-running HCIR cognitive process with executive rhythm and proactive notifications."""

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
        self._cog_sys: Any = None
        self._bus: Any = None
        self._proactive: Any = None
        self._server_task: asyncio.Task[None] | None = None
        self._shutdown_event = asyncio.Event()
        self._boot_time = 0.0

    async def start(self) -> None:
        """Boot the HCIR CognitiveSystem, event bus, and proactive processor."""
        self._boot_time = time.monotonic()

        logger.info("═" * 60)
        logger.info("  HBLLM HCIR Cognitive Daemon starting...")
        logger.info("═" * 60)

        from hbllm.network.bus import InProcessBus
        from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system
        from hbllm.serving.notifications import NotificationGateway
        from hbllm.serving.proactive import ProactiveProcessor, SSEChannel

        cog_sys = build_cognitive_system(
            CognitiveSystemConfig(
                world_id="daemon",
                memory_dir=Path(self.data_dir) / "worlds" / "daemon",
                enable_grounding=self.enable_perception,
                enable_autonomous_curiosity=True,
            )
        )
        self._cog_sys = cog_sys
        self._brain = cog_sys
        if cog_sys.runtime is not None:
            await cog_sys.runtime.start()

        self._bus = InProcessBus()
        await self._bus.start()

        gateway = NotificationGateway()
        sse = SSEChannel()
        self._proactive = ProactiveProcessor(
            gateway=gateway,
            pipeline=None,
            sse_channel=sse,
        )
        await self._proactive.start(self._bus)

        if self.serve_http:
            self._server_task = asyncio.create_task(self._start_http_server())

        elapsed = time.monotonic() - self._boot_time
        logger.info("  HCIR Cognitive Daemon ready (%.2fs boot)", elapsed)

    async def stop(self) -> None:
        """Graceful shutdown of daemon subsystems."""
        logger.info("Cognitive Daemon shutting down...")

        if self._server_task and not self._server_task.done():
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass

        if self._proactive:
            await self._proactive.stop()

        if self._bus:
            await self._bus.stop()

        if self._cog_sys and getattr(self._cog_sys, "runtime", None) is not None:
            await self._cog_sys.runtime.stop()

        logger.info("Cognitive Daemon stopped")

    async def run_forever(self) -> None:
        """Run until shutdown signal (SIGTERM/SIGINT)."""
        loop = asyncio.get_running_loop()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received")
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

        await self.start()
        await self._shutdown_event.wait()
        await self.stop()

    async def _start_http_server(self) -> None:
        """Start FastAPI/uvicorn in the background."""
        try:
            import uvicorn

            from hbllm.serving.api import app

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

    def snapshot(self) -> dict[str, Any]:
        """Full daemon telemetry snapshot."""
        node_count = 0
        if self._cog_sys is not None and hasattr(self._cog_sys, "workspace"):
            node_count = int(self._cog_sys.workspace.graph.node_count)
        elif self._brain is not None and hasattr(self._brain, "nodes"):
            node_count = len(self._brain.nodes)

        result: dict[str, Any] = {
            "uptime_s": round(time.monotonic() - self._boot_time, 2),
            "brain_nodes": node_count,
        }
        if self._proactive:
            result["proactive"] = self._proactive.snapshot()
        if self._cog_sys is not None and hasattr(self._cog_sys, "tiered_workspace"):
            tw = self._cog_sys.tiered_workspace
            result["hcir"] = {
                "working_nodes": tw.working_state.graph.node_count,
                "brain_nodes": tw.brain.graph.node_count,
                "persistent_nodes": tw.persistent.graph.node_count,
            }
        return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HBLLM HCIR Cognitive Daemon — always-on cognitive process",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        default=os.environ.get("HBLLM_PROVIDER", "openai/gpt-4o-mini"),
        help="Optional fallback LLM provider",
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
        help="Disable grounding perception — useful on headless servers",
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
    log_format = (
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        if args.mode == "foreground"
        else "%(levelname)s %(name)s: %(message)s"
    )
    logging.basicConfig(level=getattr(logging, args.log_level), format=log_format)

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
