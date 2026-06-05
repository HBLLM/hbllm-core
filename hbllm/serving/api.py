"""
FastAPI HTTP Server for the HBLLM Cognitive Architecture.

Exposes the full brain pipeline (Router → Workspace → Domain Modules →
Critic → Decision) as REST endpoints with multi-tenant session isolation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pathlib
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hbllm.config import HBLLMCoreConfig
from hbllm.network.messages import Message, MessageType, QueryPayload
from hbllm.security.audit_log import AuditLog
from hbllm.serving.auth import JWTAuthMiddleware
from hbllm.serving.security import BodySizeLimitMiddleware, sanitize_input

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Request / Response Schemas ───────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat message from a tenant."""

    tenant_id: str = Field(default="", description="Unique tenant identifier (overridden by JWT)")
    user_id: str = Field(default="", description="User identifier (overridden by JWT)")
    device_id: str = Field(default="", description="Device identifier (overridden by JWT)")
    session_id: str = Field(default="default_session", description="Session identifier")
    text: str = Field(..., min_length=1, description="User message text")
    model_size: str = Field(default="125M", description="Model size to use")
    provider: str | None = Field(
        default=None, description="LLM provider override (openai, anthropic, local)"
    )
    system_prompt: str | None = Field(default=None, description="Optional system prompt")


class FederatedEnvelopeRequest(BaseModel):
    """Encapsulated cryptographic intent envelope for P2P Federation."""

    envelope: dict[str, Any] = Field(
        ...,
        description="The signed envelope containing the sender ID, timestamp, and target intent payload.",
    )
    signature: str = Field(
        ..., description="Hex-encoded Ed25519 signature of the serialized envelope."
    )


class ChatResponse(BaseModel):
    """Response from the cognitive pipeline."""

    tenant_id: str
    user_id: str = "default"
    device_id: str = "default"
    session_id: str
    correlation_id: str
    response_text: str
    source_node: str = "decision"
    confidence: float = 0.0
    provider_used: str | None = None
    usage: dict[str, int] | None = None


class OpenAIMessage(BaseModel):
    role: str
    content: str
    name: str | None = None


class OpenAICompletionRequest(BaseModel):
    model: str = "hbllm-125m"
    messages: list[OpenAIMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class HealthResponse(BaseModel):
    """Server health check response."""

    status: str = "healthy"
    nodes_registered: int = 0
    bus_type: str = "in_process"
    provider_mode: str = "full"  # 'full' (brain) or 'provider' (external LLM)


class FeedbackRequest(BaseModel):
    """User feedback on a response (thumbs up/down for RLHF loop)."""

    tenant_id: str = Field(..., description="Tenant who sent the feedback")
    message_id: str = Field(..., description="Correlation ID of the response being rated")
    rating: int = Field(..., ge=-1, le=1, description="-1 (bad), 0 (neutral), 1 (good)")
    prompt: str | None = Field(default=None, description="Original prompt text")
    response: str | None = Field(default=None, description="Response text that was rated")
    comment: str | None = Field(default=None, description="Optional user comment")


class SyncEpisodicRequest(BaseModel):
    """Batch of episodic memories to sync upstream."""

    memories: list[dict[str, Any]] = Field(
        ..., description="List of episodic memory dictionaries to append"
    )


class SyncSemanticRequest(BaseModel):
    """Batch of semantic knowledge items to sync upstream."""

    knowledge_items: list[dict[str, Any]] = Field(
        ..., description="List of knowledge items to append"
    )


class WebRTCOfferRequest(BaseModel):
    """SDP offer from an edge device for a high-bandwidth data channel."""

    sdp: str = Field(..., description="Session Description Protocol payload")
    type: str = Field(..., description="Should be 'offer'")


# ─── Global State ─────────────────────────────────────────────────────────────

from hbllm.serving.state import _get_node_map, _state


async def _boot_brain(
    app: Any = None,
    model_size: str = "125m",
    bus_type: str = "inprocess",
    redis_url: str | None = None,
) -> None:
    """Initialize the full brain pipeline."""
    # 1. Bus
    import os

    redis_url = redis_url or os.getenv("HBLLM_REDIS_URL", "redis://localhost:6379")
    from hbllm.network.bus import MessageBus

    bus: MessageBus
    if bus_type == "redis":
        from hbllm.network.redis_bus import RedisBus

        bus = RedisBus(redis_url=redis_url)
    else:
        from hbllm.network.bus import InProcessBus

        bus = InProcessBus()

    from hbllm.network.rate_limiter import RateLimitInterceptor

    limiter = RateLimitInterceptor(target_rpm=60.0)
    bus.add_interceptor(limiter.intercept)
    await bus.start()

    # 2. Delegate cognitive loading to unified factory
    from hbllm.brain.factory import BrainConfig, BrainFactory, _is_slow_cpu

    is_slow = _is_slow_cpu()
    if is_slow:
        logger.info(
            "Slow CPU-only system detected. Dynamically disabling perception, fuzzy logic, and symbolic logic nodes to save RAM/CPU."
        )

    # We use create_local for overarching OSS usage by default
    config = BrainConfig(
        inject_memory=True,
        inject_identity=True,
        inject_curiosity=True,
        inject_perception=not is_slow,
        inject_fuzzy_logic=not is_slow,
        inject_symbolic_logic=not is_slow,
    )

    provider_name = os.getenv("HBLLM_PROVIDER")
    provider_model = os.getenv("HBLLM_PROVIDER_MODEL")

    # Known external provider prefixes — anything else with "/" is a HuggingFace model ID
    _KNOWN_PROVIDERS = {"openai", "anthropic", "ollama", "local", "groq"}

    is_provider = False
    if provider_name:
        is_provider = True
    elif "/" in model_size and not os.path.exists(model_size):
        # Check if the prefix is a known provider (e.g. "openai/gpt-4o")
        # vs a HuggingFace model ID (e.g. "Qwen/Qwen2.5-0.5B")
        prefix = model_size.split("/")[0].lower()
        if prefix in _KNOWN_PROVIDERS:
            is_provider = True
        else:
            # Treat as a HuggingFace model ID → load locally
            is_provider = False
            logger.info("Detected HuggingFace model ID: %s — routing to local loading", model_size)

    if is_provider:
        if "/" in model_size and not provider_name:
            prov = model_size
        else:
            prov = (
                f"{provider_name}/{provider_model}"
                if provider_model
                else (provider_name or model_size)
            )

        logger.info("Initializing Brain factory with external provider: %s", prov)

        provider_kwargs = {}
        base_url = (
            os.getenv("HBLLM_PROVIDER_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OLLAMA_BASE_URL")
            or os.getenv("GROQ_BASE_URL")
        )
        if base_url:
            provider_kwargs["base_url"] = base_url

        api_key = (
            os.getenv("HBLLM_PROVIDER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("GROQ_API_KEY")
        )
        if api_key:
            provider_kwargs["api_key"] = api_key

        brain = await BrainFactory.create(provider=prov, config=config, bus=bus, **provider_kwargs)
    else:
        logger.info("Initializing Brain factory with local model: %s", model_size)
        brain = await BrainFactory.create_local(model_size=model_size, config=config, bus=bus)

    # 3. Load External Plugins
    import pathlib

    from hbllm.network.plugin_manager import PluginManager

    plugin_dir = pathlib.Path(__file__).resolve().parent.parent.parent / "plugins"
    pm = PluginManager(
        plugin_dirs=[plugin_dir], bus=brain.bus, registry=brain.registry, app=app, brain=brain
    )
    pm.discover()
    await pm.load_all()

    # 4. Start Synapse Gateway
    from hbllm.serving.synapse_gateway import SynapseGateway

    audit_log = _state.get("audit_log")
    gateway = SynapseGateway(bus=brain.bus, audit_log=audit_log)
    await gateway.start()
    _state["synapse_gateway"] = gateway

    # 5. Start WebRTC Gateway (optional high-bandwidth plane)
    try:
        from hbllm.network.webrtc_gateway import WebRTCGateway

        webrtc_gateway = WebRTCGateway(bus=brain.bus)
        _state["webrtc_gateway"] = webrtc_gateway
        logger.info("WebRTC Gateway initialized for high-bandwidth perception")
    except ImportError:
        logger.warning("aiortc not installed. WebRTC perception plane disabled.")

    # Initialize FederatedMailbox
    try:
        from hbllm.network.federation.mailbox import FederatedMailbox

        mailbox = FederatedMailbox(bus=brain.bus, embedder=getattr(brain, "semantic_memory", None))
        _state["federated_mailbox"] = mailbox
        logger.info("🔒 Zero-Trust Federated Mailbox initialized successfully.")
    except Exception as e:
        logger.warning("Failed to initialize Federated Mailbox: %s", e)

    _state["brain"] = brain
    _state["config"] = config
    _state["bus"] = bus
    _state["plugin_manager"] = pm
    _state["bus_type"] = bus_type
    _state["mode"] = "full"

    logger.info("Brain pipeline booted via factory.")


async def _shutdown_brain() -> None:
    """Gracefully shutdown all nodes and the bus."""
    gateway = _state.get("synapse_gateway")
    if gateway:
        await gateway.stop()

    webrtc_gateway = _state.get("webrtc_gateway")
    if webrtc_gateway:
        await webrtc_gateway.stop()

    brain = _state.get("brain")
    if brain:
        await brain.shutdown()

    logger.info("Brain pipeline shutdown complete.")


# ─── FastAPI Application ──────────────────────────────────────────────────────


async def _boot_provider_mode() -> None:
    """Lightweight mode: use external LLM providers without the full brain."""
    import os

    from hbllm.serving.provider import get_provider

    provider_name = os.getenv("HBLLM_PROVIDER", "openai")
    provider_model = os.getenv("HBLLM_PROVIDER_MODEL", None)

    kwargs = {}
    if provider_model:
        kwargs["model"] = provider_model

    provider = get_provider(provider_name, **kwargs)
    _state["provider"] = provider
    _state["mode"] = "provider"
    _state["bus"] = None

    logger.info("Provider mode: %s (no full brain)", provider.name)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Boot the brain on startup, fall back to provider mode if it fails."""
    import os

    workers = int(os.getenv("WEB_CONCURRENCY", os.getenv("WORKERS", "1")))
    if workers > 1:
        logger.error(
            "HBLLM cannot launch with multiple workers in full brain mode. Found %d workers.",
            workers,
        )
        raise RuntimeError(
            "HBLLM memory architecture is stateful. You must run with exactly 1 worker "
            "(e.g., uvicorn --workers 1 or WEB_CONCURRENCY=1)."
        )

    model_size = os.getenv("HBLLM_MODEL_SIZE", "125m")

    # Initialize Core Config & Security components
    config = HBLLMCoreConfig.load()
    if config.security.audit_enabled:
        _state["audit_log"] = AuditLog(db_path=config.security.audit_db_path)

    try:
        await _boot_brain(app=app, model_size=model_size)
        logger.info("Full brain pipeline active")
    except Exception as e:
        logger.warning("Full brain boot failed (%s). Falling back to provider mode.", e)
        await _boot_provider_mode()

    # ── Cloud features (SaaS layer) — graceful fallback for OSS mode ──
    try:
        from fastapi.staticfiles import StaticFiles
        from hbllm_cloud.admin_api import create_admin_router  # type: ignore[import-not-found]
        from hbllm_cloud.dashboard.routes import (  # type: ignore[import-not-found]
            create_dashboard_router,
        )
        from hbllm_cloud.tenant_manager import TenantManager  # type: ignore[import-not-found]

        from hbllm.brain.policy_engine import PolicyEngine
        from hbllm.serving.security import ApiKeyManager

        tm = TenantManager(db_path="data/tenants.db")
        pe = PolicyEngine()
        pe.load_from_yaml("config/policies.yaml")
        akm = ApiKeyManager()
        akm.load_from_env()

        _state["tenant_manager"] = tm
        _state["policy_engine"] = pe
        _state["api_key_manager"] = akm

        # REST API
        admin_router = create_admin_router(tm, pe, akm)
        app.include_router(admin_router, prefix="/v1/admin")

        # Dashboard UI
        dashboard_router = create_dashboard_router(tm, pe, akm)
        app.include_router(dashboard_router, prefix="/admin")

        # Static files
        from pathlib import Path

        import hbllm_cloud.dashboard as _dash_pkg  # type: ignore[import-not-found]

        static_dir = Path(_dash_pkg.__file__).parent / "static"
        if static_dir.exists():
            app.mount("/admin/static", StaticFiles(directory=str(static_dir)), name="admin-static")

        # Auth middleware (protects /admin/* except /admin/login and /admin/static)
        from hbllm_cloud.dashboard.auth import (  # type: ignore[import-not-found]
            DashboardAuthMiddleware,
        )

        app.add_middleware(DashboardAuthMiddleware)

        # API security middleware (protects /v1/* with API key auth + rate limiting)
        from hbllm_cloud.api_middleware import (  # type: ignore[import-not-found]
            ApiSecurityMiddleware,
        )

        from hbllm.serving.security import RateLimiter

        rate_limiter = RateLimiter(requests_per_minute=60.0, burst_size=10.0)
        _state["rate_limiter"] = rate_limiter
        app.add_middleware(
            ApiSecurityMiddleware,
            api_key_manager=akm,
            rate_limiter=rate_limiter,
            usage_tracker=_state.get("usage_tracker"),
            billing=_state.get("billing"),
        )

        logger.info("☁️  Cloud features enabled (admin API, dashboard, auth, rate limiting)")
    except ImportError:
        logger.info("🔓 Running in open-source mode (no cloud features)")

    # ── Knowledge Base + Usage + Billing ──
    try:
        from hbllm_cloud.billing import BillingManager  # type: ignore[import-not-found]
        from hbllm_cloud.knowledge.api import (  # type: ignore[import-not-found]
            create_knowledge_router,
        )
        from hbllm_cloud.knowledge.embeddings import (  # type: ignore[import-not-found]
            EmbeddingsService,
        )
        from hbllm_cloud.knowledge.processor import (  # type: ignore[import-not-found]
            DocumentProcessor,
        )
        from hbllm_cloud.knowledge.vector_store import VectorStore  # type: ignore[import-not-found]
        from hbllm_cloud.usage import UsageTracker  # type: ignore[import-not-found]

        embeddings_svc = EmbeddingsService(provider=os.getenv("HBLLM_EMBEDDING_PROVIDER", "openai"))
        vector_store = VectorStore(db_path="data/vectors.db")
        processor = DocumentProcessor()
        usage_tracker = UsageTracker(db_path="data/usage.db")
        billing = BillingManager()

        _state["embeddings"] = embeddings_svc
        _state["vector_store"] = vector_store
        _state["processor"] = processor
        _state["usage_tracker"] = usage_tracker
        _state["billing"] = billing

        # Knowledge Base API
        kb_router = create_knowledge_router(vector_store, embeddings_svc, processor, usage_tracker)
        app.include_router(kb_router)

        # Usage & Billing endpoints
        @app.get("/v1/usage")
        async def get_usage(api_req: Request, days: int = 30) -> Any:
            tenant_id = getattr(api_req.state, "tenant_id", "default")
            return usage_tracker.get_tenant_usage(tenant_id, days)

        @app.get("/v1/usage/daily")
        async def get_daily_usage(api_req: Request, days: int = 30) -> Any:
            tenant_id = getattr(api_req.state, "tenant_id", "default")
            return {"daily": usage_tracker.get_daily_usage(tenant_id, days)}

        @app.get("/v1/billing/plans")
        async def list_plans() -> Any:
            return {"plans": billing.list_plans()}

        @app.get("/v1/billing")
        async def get_billing(api_req: Request, days: int = 30) -> Any:
            tenant_id = getattr(api_req.state, "tenant_id", "default")
            usage = usage_tracker.get_tenant_usage(tenant_id, days)
            cost = usage_tracker.estimate_cost(tenant_id, days)
            return {"usage": usage, "cost": cost}

        logger.info("📊 Knowledge base, usage tracking, and billing enabled")
    except ImportError as e:
        logger.info("Knowledge/billing modules not available: %s", e)

    # ── Tenant Portal (self-service UI) ──
    try:
        from fastapi.staticfiles import StaticFiles as SF2
        from hbllm_cloud.portal.routes import create_portal_router  # type: ignore[import-not-found]

        portal_router = create_portal_router(
            tenant_manager=_state.get("tenant_manager"),
            api_key_manager=_state.get("api_key_manager"),
            vector_store=_state.get("vector_store"),
            usage_tracker=_state.get("usage_tracker"),
            billing=_state.get("billing"),
            embeddings_service=_state.get("embeddings"),
            processor=_state.get("processor"),
        )
        app.include_router(portal_router, prefix="/portal")

        import hbllm_cloud.portal as _portal_pkg  # type: ignore[import-not-found]

        portal_static = Path(_portal_pkg.__file__).parent / "static"
        if portal_static.exists():
            app.mount("/portal/static", SF2(directory=str(portal_static)), name="portal-static")

        logger.info("🌐 Tenant portal enabled at /portal/")
    except ImportError as e:
        logger.info("Tenant portal not available: %s", e)

    # ── High-Value Platform Features ──
    try:
        from hbllm_cloud.agents import AgentRegistry  # type: ignore[import-not-found]
        from hbllm_cloud.extraction import DataExtractor  # type: ignore[import-not-found]
        from hbllm_cloud.multilang import MultiLanguageService  # type: ignore[import-not-found]
        from hbllm_cloud.workflows import WorkflowEngine  # type: ignore[import-not-found]

        provider = _state.get("provider")
        workflow_engine = WorkflowEngine(provider=provider)
        extractor = DataExtractor(provider=provider)
        agent_registry = AgentRegistry()
        multilang = MultiLanguageService(provider=provider)

        _state["workflow_engine"] = workflow_engine
        _state["extractor"] = extractor
        _state["agent_registry"] = agent_registry
        _state["multilang"] = multilang

        # ── Workflow Endpoints ──
        @app.get("/v1/workflows/templates")
        async def list_workflow_templates() -> Any:
            return {"templates": workflow_engine.list_templates()}

        @app.post("/v1/workflows/run")
        async def run_workflow(request: Request) -> Any:
            body = await request.json()
            result = await workflow_engine.run(
                template_id=body.get("template_id"),
                steps=body.get("steps"),
                variables=body.get("variables", {}),
            )
            return {
                "workflow_id": result.workflow_id,
                "steps_completed": result.steps_completed,
                "outputs": result.outputs,
                "tokens_used": result.tokens_used,
                "duration_ms": result.duration_ms,
            }

        # ── Data Extraction Endpoints ──
        @app.get("/v1/extract/schemas")
        async def list_extraction_schemas() -> Any:
            return {"schemas": extractor.list_schemas()}

        @app.post("/v1/extract")
        async def extract_data(request: Request) -> Any:
            body = await request.json()
            result = await extractor.extract(
                text=body["text"],
                schema_id=body.get("schema_id"),
                schema=body.get("schema"),
                tenant_id=body.get("tenant_id", ""),
            )
            return {
                "schema_id": result.schema_id,
                "data": result.data,
                "confidence": result.confidence,
                "tokens_used": result.tokens_used,
            }

        @app.post("/v1/classify")
        async def classify_text(request: Request) -> Any:
            body = await request.json()
            return await extractor.classify(
                text=body["text"],
                categories=body["categories"],
                multi_label=body.get("multi_label", False),
            )

        # ── Agent Endpoints ──
        @app.post("/v1/agents")
        async def create_agent(request: Request) -> Any:
            body = await request.json()
            tenant_id = getattr(request.state, "tenant_id", None) or body.get(
                "tenant_id", "default"
            )
            agent = agent_registry.create_agent(
                tenant_id=tenant_id,
                name=body["name"],
                system_prompt=body["system_prompt"],
                tools=body.get("tools", []),
            )
            return {"agent_id": agent.agent_id, "name": agent.name}

        @app.get("/v1/agents")
        async def list_agents(request: Request) -> Any:
            tenant_id = getattr(request.state, "tenant_id", "default")
            agents = agent_registry.list_agents(tenant_id)
            return {"agents": [{"agent_id": a.agent_id, "name": a.name} for a in agents]}

        @app.get("/v1/agents/tools")
        async def list_agent_tools(request: Request) -> Any:
            tenant_id = getattr(request.state, "tenant_id", "default")
            return {"tools": agent_registry.list_tools(tenant_id)}

        @app.post("/v1/agents/{agent_id}/run")
        async def run_agent(agent_id: str, request: Request) -> Any:
            body = await request.json()
            tenant_id = getattr(request.state, "tenant_id", "default")
            context = {
                "vector_store": _state.get("vector_store"),
                "embeddings": _state.get("embeddings"),
                "tenant_id": tenant_id,
            }
            result = await agent_registry.run_agent(
                agent_id=agent_id,
                user_message=body["message"],
                provider=provider,
                context=context,
            )
            return {
                "response": result.response,
                "tool_calls": result.tool_calls,
                "iterations": result.iterations,
                "tokens_used": result.tokens_used,
            }

        @app.post("/v1/agents/tools")
        async def register_tool(request: Request) -> Any:
            body = await request.json()
            tenant_id = getattr(request.state, "tenant_id", "default")
            tool = agent_registry.register_webhook_tool(
                tenant_id=tenant_id,
                name=body["name"],
                description=body["description"],
                webhook_url=body["webhook_url"],
                parameters=body.get("parameters"),
            )
            return {"name": tool.name, "registered": True}

        # ── Multi-language Endpoints ──
        @app.post("/v1/detect-language")
        async def detect_language(request: Request) -> Any:
            body = await request.json()
            lang = await multilang.detect_language(body["text"])
            return {"language": lang, "name": multilang.SUPPORTED_LANGUAGES.get(lang, lang)}

        @app.post("/v1/translate")
        async def translate_text(request: Request) -> Any:
            body = await request.json()
            result = await multilang.translate(
                text=body["text"],
                target_lang=body["target_lang"],
                source_lang=body.get("source_lang"),
            )
            return {
                "translated": result.translated,
                "source_lang": result.source_lang,
                "target_lang": result.target_lang,
                "tokens_used": result.tokens_used,
            }

        @app.get("/v1/languages")
        async def list_languages() -> Any:
            return {"languages": multilang.list_languages()}

        # ── Embeddable Widget ──
        import hbllm_cloud.widget as _widget_pkg  # type: ignore[import-not-found]
        from fastapi.responses import FileResponse

        widget_path = Path(_widget_pkg.__file__).parent / "hbllm-widget.js"

        @app.get("/widget/hbllm-widget.js")
        async def serve_widget() -> Any:
            return FileResponse(
                str(widget_path),
                media_type="application/javascript",
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Access-Control-Allow-Origin": "*",
                },
            )

        logger.info(
            "🚀 Platform features enabled (workflows, extraction, agents, multilang, widget)"
        )
    except ImportError as e:
        logger.info("Platform features not available: %s", e)

    # ── AGI Differentiators: Multi-Perspective Analysis + XAI ──
    try:
        from hbllm_cloud.multi_perspective import (  # type: ignore[import-not-found]
            MultiPerspectiveAnalyzer,
        )
        from hbllm_cloud.xai import ExplainabilityEngine  # type: ignore[import-not-found]

        provider = _state.get("provider")
        mpa = MultiPerspectiveAnalyzer(provider=provider)
        xai = ExplainabilityEngine(db_path="data/xai.db")

        _state["multi_perspective"] = mpa
        _state["xai"] = xai

        # ── Multi-Perspective Analysis ──
        @app.get("/v1/analyze/lenses")
        async def list_lenses() -> Any:
            return {"lenses": mpa.list_lenses()}

        @app.post("/v1/analyze")
        async def multi_perspective_analyze(request: Request) -> Any:
            body = await request.json()
            result = await mpa.analyze(
                query=body["query"],
                lenses=body.get("lenses"),
                context=body.get("context", ""),
                tenant_id=body.get("tenant_id", ""),
            )
            return {
                "request_id": result.request_id,
                "synthesis": result.synthesis,
                "consensus_confidence": result.consensus_confidence,
                "perspectives": [
                    {
                        "lens": p.lens_id,
                        "name": p.lens_name,
                        "icon": p.icon,
                        "node_analog": p.node_analog,
                        "analysis": p.analysis,
                        "confidence": p.confidence,
                        "processing_time_ms": p.processing_time_ms,
                    }
                    for p in result.perspectives
                ],
                "lenses_used": result.lenses_used,
                "total_tokens": result.total_tokens,
                "total_duration_ms": result.total_duration_ms,
            }

        @app.post("/v1/analyze/quick")
        async def quick_analyze(request: Request) -> Any:
            body = await request.json()
            return await mpa.quick_analyze(
                query=body["query"],
                context=body.get("context", ""),
            )

        # ── Explainable AI (XAI) ──
        @app.get("/v1/xai/traces")
        async def get_traces(request: Request, limit: int = 50) -> Any:
            tenant_id = getattr(request.state, "tenant_id", "default")
            return {"traces": xai.get_tenant_traces(tenant_id, limit)}

        @app.get("/v1/xai/trace/{trace_id}")
        async def get_trace(trace_id: str) -> Any:
            trace = xai.get_trace(trace_id)
            if not trace:
                raise HTTPException(status_code=404, detail="Trace not found")
            return trace

        @app.get("/v1/xai/audit")
        async def get_audit_log(request: Request, limit: int = 100) -> Any:
            tenant_id = getattr(request.state, "tenant_id", "default")
            return {"audit_log": xai.get_audit_log(tenant_id, limit)}

        @app.get("/v1/xai/compliance")
        async def compliance_report(request: Request, days: int = 30) -> Any:
            tenant_id = getattr(request.state, "tenant_id", "default")
            return xai.get_compliance_report(tenant_id, days)

        logger.info("🧠 AGI features enabled (multi-perspective analysis, explainable AI)")
    except ImportError as e:
        logger.info("AGI features not available: %s", e)

    yield

    # Cleanup
    if "vector_store" in _state:
        _state["vector_store"].close()
    if "usage_tracker" in _state:
        _state["usage_tracker"].close()
    if "tenant_manager" in _state:
        _state["tenant_manager"].close()
    if "xai" in _state:
        _state["xai"].close()
    await _shutdown_brain()


app = FastAPI(
    title="HBLLM Cognitive API",
    description="REST API for the Human Brain LLM — a modular cognitive architecture with multi-tenant isolation.",
    version="1.0.0",
    lifespan=lifespan,
)

from fastapi.responses import JSONResponse

from hbllm.security.tenant_guard import TenantIsolationError


@app.exception_handler(TenantIsolationError)
async def tenant_isolation_exception_handler(request: Request, exc: TenantIsolationError):
    logger.warning("Tenant isolation violation blocked: %s", exc)
    return JSONResponse(
        status_code=403,
        content={"detail": f"Access denied: {str(exc)}"},
    )


_cors_origins = os.environ.get(
    "HBLLM_CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:8080,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(JWTAuthMiddleware)


# ─── Studio Router Registration ───────────────────────────────────────────────
from hbllm.serving.studio import router as studio_router

app.include_router(studio_router)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check server health and node count."""
    brain = _state.get("brain")
    node_count = len(brain.nodes) if brain else 0
    mode = _state.get("mode", "unknown")
    _state.get("provider")
    return HealthResponse(
        status="healthy",
        nodes_registered=node_count,
        bus_type=_state.get("bus_type", "unknown"),
        provider_mode=mode,
    )


@app.get("/metrics")
async def metrics() -> Any:
    """Return real-time MessageBus performance metrics."""
    brain = _state.get("brain")
    if not brain or not hasattr(brain.bus, "metrics"):
        return {"error": "Bus not initialized or metrics unavailable"}
    return brain.bus.metrics.snapshot()


# ─── Provider-based Chat (lightweight) ────────────────────────────────────────


async def _chat_via_provider(request: ChatRequest) -> ChatResponse:
    """Handle chat using the LLM provider abstraction (no brain pipeline)."""
    from hbllm.serving.provider import get_provider

    correlation_id = str(uuid.uuid4())

    # Resolve provider: request override → state default
    if request.provider:
        if (
            request.provider == "local"
            and _state.get("brain")
            and hasattr(_state["brain"], "provider")
        ):
            provider = _state["brain"].provider
        else:
            provider = get_provider(request.provider)
    elif "provider" in _state:
        provider = _state["provider"]
    elif _state.get("brain") and hasattr(_state["brain"], "provider"):
        provider = _state["brain"].provider
    else:
        raise HTTPException(status_code=503, detail="No LLM provider configured")

    # Build messages
    messages = []

    bus = _state.get("bus")
    system_content = request.system_prompt
    if not system_content:
        if bus:
            try:
                from hbllm.brain.prompt_helper import get_dynamic_system_prompt

                system_content = await get_dynamic_system_prompt(
                    bus, request.tenant_id, "api_server"
                )
            except Exception as e:
                logger.warning("Failed to generate dynamic system prompt: %s", e)
        if not system_content:
            system_content = (
                "You are Sentra, an advanced cognitive AI assistant powered by the HBLLM modular architecture. "
                "You have access to various cognitive and tool modules, including a BrowserNode (which allows "
                "you to browse the web and search for real-time information), an ExecutionNode (for running "
                "Python code in a secure sandbox), a LogicNode (powered by Z3 for symbolic reasoning), and a "
                "persistent memory node. Be helpful, precise, and accurate."
            )

    # ── RAG: inject relevant knowledge base context ──
    rag_context = ""
    vector_store = _state.get("vector_store")
    embeddings_svc = _state.get("embeddings")
    if vector_store and embeddings_svc:
        try:
            query_embedding = await embeddings_svc.embed_single(request.text)
            results = vector_store.search(
                tenant_id=request.tenant_id,
                query_embedding=query_embedding,
                top_k=3,
                min_score=0.4,
            )
            if results:
                context_parts = []
                for r in results:
                    context_parts.append(f"[{r['filename']}]: {r['content']}")
                rag_context = "\n\n".join(context_parts)
                system_content += (
                    "\n\n--- Relevant Knowledge Base Documents ---\n"
                    + rag_context
                    + "\n--- End of Documents ---\n"
                    "Use the above documents to inform your answer when relevant."
                )
        except Exception as e:
            logger.warning("RAG retrieval failed: %s", e)

    # ── Retrieve and format conversation history ──
    history = []
    if bus:
        try:
            hist_msg = Message(
                type=MessageType.QUERY,
                source_node_id="api_server",
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                topic="memory.retrieve_recent",
                payload={"session_id": request.session_id, "limit": 30},
                correlation_id=correlation_id,
            )
            hist_resp = await bus.request("memory.retrieve_recent", hist_msg, timeout=3.0)
            history = hist_resp.payload.get("turns", [])
        except Exception as e:
            logger.warning("Failed to retrieve conversation history: %s", e)

    # Exclude the current query if it has already been saved to the DB
    if history and history[-1].get("role") == "user" and history[-1].get("content") == request.text:
        history = history[:-1]

    # Filter to only user/assistant turns
    filtered_history = [t for t in history if t.get("role") in ("user", "assistant")]

    messages.append({"role": "system", "content": system_content})

    # Append history turns in chronological order
    for turn in filtered_history:
        messages.append({"role": turn.get("role"), "content": turn.get("content", "")})

    messages.append({"role": "user", "content": sanitize_input(request.text)})

    # Apply policy engine if available
    policy_engine = _state.get("policy_engine")
    if policy_engine:
        violation = policy_engine.check(request.text, tenant_id=request.tenant_id)
        if violation and violation.get("action") == "block":
            return ChatResponse(
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                correlation_id=correlation_id,
                response_text=f"Request blocked by policy: {violation.get('reason', 'policy violation')}",
                source_node="policy_engine",
                provider_used=provider.name,
            )

    # Call provider
    try:
        response = await provider.generate(messages)
    except Exception as e:
        logger.error("Provider %s failed: %s", provider.name, e)
        raise HTTPException(status_code=502, detail=f"LLM provider error: {e}")

    # ── Track usage ──
    usage_tracker = _state.get("usage_tracker")
    if usage_tracker:
        usage_tracker.record_chat(
            tenant_id=request.tenant_id,
            prompt_tokens=response.usage.get("prompt_tokens", 0),
            completion_tokens=response.usage.get("completion_tokens", 0),
            provider=provider.name,
        )

    return ChatResponse(
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        correlation_id=correlation_id,
        response_text=response.content,
        source_node="provider",
        provider_used=provider.name,
        usage=response.usage,
    )


# ─── Brain-based Chat (full pipeline) ────────────────────────────────────────


async def _chat_via_brain(request: ChatRequest) -> ChatResponse:
    """Handle chat using the full brain pipeline."""
    from hbllm.brain.factory import BrainConfig

    config = _state.get("config") or BrainConfig()
    timeout = config.api_timeout

    bus = _state["bus"]
    correlation_id = str(uuid.uuid4())

    response_future: asyncio.Future[Message] = asyncio.get_running_loop().create_future()

    async def output_handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    sub = await bus.subscribe("sensory.output", output_handler)

    # Store user message in memory
    memory_msg = Message(
        type=MessageType.EVENT,
        source_node_id="api_server",
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        topic="memory.store",
        payload={
            "session_id": request.session_id,
            "tenant_id": request.tenant_id,
            "role": "user",
            "content": request.text,
            "user_id": request.user_id,
            "device_id": request.device_id,
        },
    )
    await bus.publish("memory.store", memory_msg)

    # ── Retrieve and format conversation history ──
    history = []
    try:
        hist_msg = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            topic="memory.retrieve_recent",
            payload={"session_id": request.session_id, "limit": 30},
            correlation_id=correlation_id,
        )
        hist_resp = await bus.request("memory.retrieve_recent", hist_msg, timeout=3.0)
        history = hist_resp.payload.get("turns", [])
    except Exception as e:
        logger.warning("Failed to retrieve conversation history: %s", e)

    # Exclude the current query if it has already been saved to the DB
    if history and history[-1].get("role") == "user" and history[-1].get("content") == request.text:
        history = history[:-1]

    # Filter to only user/assistant turns
    filtered_history = [t for t in history if t.get("role") in ("user", "assistant")]

    prompt_text = request.text
    if filtered_history:
        flattened_history = ""
        for turn in filtered_history:
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            flattened_history += f"{role}: {content}\n\n"
        prompt_text = flattened_history + f"User: {request.text}\n\nAssistant:"

    # Send to router
    query_msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        device_id=request.device_id,
        session_id=request.session_id,
        topic="router.query",
        payload=QueryPayload(text=sanitize_input(prompt_text)).model_dump(),
        correlation_id=correlation_id,
    )
    await bus.publish("router.query", query_msg)

    try:
        result_msg = await asyncio.wait_for(response_future, timeout=timeout)
        response_text = result_msg.payload.get("text", "")
        source = result_msg.payload.get("source", "decision")

        # Store assistant response
        store_msg = Message(
            type=MessageType.EVENT,
            source_node_id="api_server",
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            topic="memory.store",
            payload={
                "session_id": request.session_id,
                "tenant_id": request.tenant_id,
                "role": "assistant",
                "content": response_text,
                "user_id": request.user_id,
                "device_id": request.device_id,
            },
        )
        await bus.publish("memory.store", store_msg)

        return ChatResponse(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
            device_id=request.device_id,
            session_id=request.session_id,
            correlation_id=correlation_id,
            response_text=response_text,
            source_node=source,
        )
    except (TimeoutError, asyncio.TimeoutError):
        raise HTTPException(status_code=504, detail=f"Pipeline timed out ({int(timeout)}s)")
    finally:
        # Always clean up subscription to prevent memory leak
        await bus.unsubscribe(sub)


@app.post("/v1/federation/mailbox")
async def receive_federated_envelope(request: FederatedEnvelopeRequest) -> dict[str, Any]:
    """
    Exposed zero-trust endpoint to receive, decrypt, and process incoming P2P federation envelopes.
    """
    mailbox = _state.get("federated_mailbox")
    if not mailbox:
        raise HTTPException(
            status_code=503, detail="Federation mailbox is not active or enabled on this node."
        )

    envelope_package = {"envelope": request.envelope, "signature": request.signature}

    result = await mailbox.receive_envelope(envelope_package)
    if result.get("status") == "error":
        raise HTTPException(
            status_code=400, detail=result.get("reason", "Malformed envelope payload.")
        )
    elif result.get("status") == "blocked":
        raise HTTPException(
            status_code=403, detail=result.get("reason", "Security threat blocked.")
        )

    return result


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(api_req: Request, request: ChatRequest) -> ChatResponse | Any:
    """
    Send a message through the HBLLM system.

    Automatically routes to the full brain pipeline or external LLM provider
    based on system mode. Use 'provider' field to force a specific provider.
    """
    request.tenant_id = getattr(api_req.state, "tenant_id", "default")
    request.user_id = getattr(api_req.state, "user_id", "default")
    request.device_id = getattr(api_req.state, "device_id", "default")
    mode = _state.get("mode", "provider")

    # Explicit provider override always uses provider path
    if request.provider:
        return await _chat_via_provider(request)

    # Route based on mode
    if mode == "full" and _state.get("brain"):
        return await _chat_via_brain(request)
    else:
        return await _chat_via_provider(request)


from fastapi import WebSocket, WebSocketDisconnect


@app.websocket("/v1/synapse/ws")
async def synapse_websocket(websocket: WebSocket) -> Any:
    """
    WebSocket endpoint for Synapse Edge Devices.

    Authentication: Pass a JWT token as a query parameter:
        ws://host/v1/synapse/ws?token=<jwt>

    The token must contain `tenant_id`, `user_id`, and `device_id` claims.
    Identity is extracted from the verified JWT — not from untrusted params.
    """
    import os

    import jwt as pyjwt

    # ── 1. Extract and verify JWT ──
    token = websocket.query_params.get("token", "")
    if not token:
        await websocket.close(code=1008, reason="Missing authentication token")
        return

    secret_key = os.environ.get("HBLLM_JWT_SECRET", "")
    if not secret_key:
        # In dev mode, accept the token from _state if the auth middleware generated one
        auth_mw = next(
            (
                m
                for m in getattr(app, "user_middleware", [])
                if hasattr(m, "kwargs") and "secret_key" in m.kwargs
            ),
            None,
        )
        if auth_mw:
            secret_key = auth_mw.kwargs.get("secret_key", "")

    if not secret_key:
        logger.warning("Synapse WS: No JWT secret configured — rejecting connection")
        await websocket.close(code=1011, reason="Server auth not configured")
        return

    try:
        payload = pyjwt.decode(token, secret_key, algorithms=["HS256"])
    except pyjwt.ExpiredSignatureError:
        await websocket.close(code=1008, reason="Token expired")
        return
    except pyjwt.InvalidTokenError:
        await websocket.close(code=1008, reason="Invalid token")
        return

    tenant_id = payload.get("tenant_id")
    if not tenant_id:
        await websocket.close(code=1008, reason="Token missing tenant_id")
        return

    user_id = payload.get("user_id", "default")
    device_id = payload.get("device_id", "default")

    # ── 2. Connect to SynapseGateway ──
    gateway = _state.get("synapse_gateway")
    if not gateway:
        await websocket.close(code=1011, reason="Synapse Gateway not initialized")
        return

    await gateway.connect(websocket, tenant_id, user_id, device_id)

    from hbllm.security.tenant_guard import TenantContext

    try:
        async with TenantContext(tenant_id, user_id=user_id, device_id=device_id):
            while True:
                data = await websocket.receive_text()
                await gateway.handle_inbound_message(tenant_id, user_id, device_id, data)
    except WebSocketDisconnect:
        gateway.disconnect(tenant_id, user_id, device_id)
    except Exception as e:
        logger.error(f"Synapse WebSocket error: {e}")
        gateway.disconnect(tenant_id, user_id, device_id)


@app.post("/v1/synapse/webrtc/offer")
async def webrtc_offer(api_req: Request, request: WebRTCOfferRequest) -> Any:
    """Negotiate a WebRTC P2P connection for high-bandwidth perception streams."""
    tenant_id = getattr(api_req.state, "tenant_id", "default")
    user_id = getattr(api_req.state, "user_id", "default")
    device_id = getattr(api_req.state, "device_id", "default")

    webrtc_gateway = _state.get("webrtc_gateway")
    if not webrtc_gateway:
        raise HTTPException(
            status_code=501, detail="WebRTC perception plane is disabled on this Core"
        )

    try:
        answer = await webrtc_gateway.handle_offer(
            tenant_id, user_id, device_id, request.sdp, request.type
        )
        return answer
    except Exception as e:
        logger.error(f"WebRTC negotiation failed for {device_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/stream")
async def chat_stream(api_req: Request, request: ChatRequest) -> StreamingResponse:
    """
    Stream a response from the cognitive pipeline as Server-Sent Events (SSE).

    Each SSE event has the format:
        data: {"token": "...", "done": false}
        data: {"token": "", "done": true, "correlation_id": "..."}
    """
    request.tenant_id = getattr(api_req.state, "tenant_id", "default")
    request.user_id = getattr(api_req.state, "user_id", "default")
    request.device_id = getattr(api_req.state, "device_id", "default")
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    correlation_id = str(uuid.uuid4())

    token_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def output_handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id:
            text = msg.payload.get("text", "")
            await token_queue.put(text)
            await token_queue.put(None)  # Signal completion

    await bus.subscribe("sensory.output", output_handler)

    # Store user message in memory
    memory_msg = Message(
        type=MessageType.EVENT,
        source_node_id="api_server",
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        device_id=request.device_id,
        session_id=request.session_id,
        topic="memory.store",
        payload={
            "session_id": request.session_id,
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "device_id": request.device_id,
            "role": "user",
            "content": request.text,
        },
    )
    await bus.publish("memory.store", memory_msg)

    # Send to router
    query_msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=request.tenant_id,
        user_id=request.user_id,
        device_id=request.device_id,
        session_id=request.session_id,
        topic="router.query",
        payload=QueryPayload(text=request.text).model_dump(),
        correlation_id=correlation_id,
    )
    await bus.publish("router.query", query_msg)

    from hbllm.brain.factory import BrainConfig

    config = _state.get("config") or BrainConfig()
    timeout = config.stream_timeout

    async def event_generator() -> Any:
        import json as _json

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(token_queue.get(), timeout=timeout)
                except (TimeoutError, asyncio.TimeoutError):
                    yield f"data: {_json.dumps({'token': '', 'done': True, 'error': 'timeout'})}\n\n"
                    break

                if chunk is None:
                    yield f"data: {_json.dumps({'token': '', 'done': True, 'correlation_id': correlation_id})}\n\n"
                    break
                else:
                    yield f"data: {_json.dumps({'token': chunk, 'done': False})}\n\n"
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── OpenAI Compatible API ───────────────────────────────────────────────────

import time


@app.post("/v1/chat/completions")
async def chat_completions(api_req: Request, request: OpenAICompletionRequest) -> Any:
    """
    OpenAI-compatible chat completions endpoint for AI coding assistants like Cursor, Claude Code, etc.
    """
    from hbllm.brain.factory import BrainConfig

    config = _state.get("config") or BrainConfig()
    api_timeout = config.api_timeout
    stream_timeout = config.stream_timeout
    tenant_id = getattr(api_req.state, "tenant_id", "default")
    session_id = str(uuid.uuid4())
    correlation_id = str(uuid.uuid4())

    # Flatten the messages list into a single prompt for HBLLM
    text_prompt = ""
    for msg in request.messages:
        text_prompt += f"{msg.role.capitalize()}: {msg.content}\n\n"

    text_prompt += "Assistant:"

    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    if not request.stream:
        response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

        async def output_handler(msg: Message) -> None:
            if msg.correlation_id == correlation_id and not response_future.done():
                response_future.set_result(msg)

        sub = await bus.subscribe("sensory.output", output_handler)

        query_msg = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id=tenant_id,
            session_id=session_id,
            topic="router.query",
            payload=QueryPayload(text=text_prompt).model_dump(),
            correlation_id=correlation_id,
        )
        await bus.publish("router.query", query_msg)

        try:
            result_msg = await asyncio.wait_for(response_future, timeout=api_timeout)
            response_text = result_msg.payload.get("text", "")

            return {
                "id": f"chatcmpl-{correlation_id}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(text_prompt) // 4,
                    "completion_tokens": len(response_text) // 4,
                    "total_tokens": (len(text_prompt) + len(response_text)) // 4,
                },
            }
        except (TimeoutError, asyncio.TimeoutError):
            raise HTTPException(status_code=504, detail=f"Pipeline timed out ({int(api_timeout)}s)")
        finally:
            await bus.unsubscribe(sub)

    else:
        # Stream=True handling
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()

        async def stream_handler(msg: Message) -> None:
            if msg.correlation_id == correlation_id:
                text = msg.payload.get("text", "")
                await token_queue.put(text)
                await token_queue.put(None)  # Signal completion

        await bus.subscribe("sensory.output", stream_handler)

        query_msg = Message(
            type=MessageType.QUERY,
            source_node_id="api_server",
            tenant_id=tenant_id,
            session_id=session_id,
            topic="router.query",
            payload=QueryPayload(text=text_prompt).model_dump(),
            correlation_id=correlation_id,
        )
        await bus.publish("router.query", query_msg)

        async def event_generator() -> Any:
            import json as _json

            try:
                # Issue initial chunk
                yield f"data: {_json.dumps({'id': f'chatcmpl-{correlation_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

                while True:
                    try:
                        chunk = await asyncio.wait_for(token_queue.get(), timeout=stream_timeout)
                    except (TimeoutError, asyncio.TimeoutError):
                        yield f"data: {_json.dumps({'id': f'chatcmpl-{correlation_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'timeout'}]})}\n\n"
                        break

                    if chunk is None:
                        yield f"data: {_json.dumps({'id': f'chatcmpl-{correlation_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    else:
                        # Stream tokens naturally
                        tokens = chunk.split(" ")
                        for idx, tok in enumerate(tokens):
                            piece = tok if idx == 0 else " " + tok
                            yield f"data: {_json.dumps({'id': f'chatcmpl-{correlation_id}', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': request.model, 'choices': [{'index': 0, 'delta': {'content': piece}, 'finish_reason': None}]})}\n\n"

            except asyncio.CancelledError:
                pass

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )


@app.get("/v1/memory/{session_id}")
async def get_memory(request: Request, session_id: str, limit: int = 20) -> Any:
    """Retrieve recent conversation history for a tenant's session."""
    tenant_id = getattr(request.state, "tenant_id", "default")
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def memory_handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe("memory.retrieve_recent.response", memory_handler)

    query = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=tenant_id,
        session_id=session_id,
        topic="memory.retrieve_recent",
        payload={"session_id": session_id, "tenant_id": tenant_id, "limit": limit},
        correlation_id=correlation_id,
    )
    await bus.publish("memory.retrieve_recent", query)

    try:
        result = await asyncio.wait_for(response_future, timeout=5.0)
        return result.payload
    except (TimeoutError, asyncio.TimeoutError):
        return {"session_id": session_id, "turns": []}


_sync_dedup_cache: set[str] = set()


@app.post("/v1/sync/episodic")
async def sync_episodic(api_req: Request, request: SyncEpisodicRequest) -> Any:
    """Sync a batch of episodic memories from an edge device (append strategy)."""
    tenant_id = getattr(api_req.state, "tenant_id", "default")
    user_id = getattr(api_req.state, "user_id", "default")
    device_id = getattr(api_req.state, "device_id", "default")

    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    import hashlib
    import json

    synced_count = 0
    for mem in request.memories:
        payload = dict(mem)

        # Deduplication hash based on content
        content_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if content_hash in _sync_dedup_cache:
            continue

        # Add to cache and restrict size to prevent memory leaks
        _sync_dedup_cache.add(content_hash)
        if len(_sync_dedup_cache) > 10000:
            # Pop an arbitrary item if cache gets too large
            _sync_dedup_cache.pop()

        payload["tenant_id"] = tenant_id
        payload["user_id"] = user_id
        payload["device_id"] = device_id

        msg = Message(
            type=MessageType.EVENT,
            source_node_id=f"edge_sync_{device_id}",
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            session_id=mem.get("session_id", "sync_session"),
            topic="memory.store",
            payload=payload,
        )
        await bus.publish("memory.store", msg)
        synced_count += 1

    return {
        "status": "success",
        "synced": synced_count,
        "skipped": len(request.memories) - synced_count,
    }


@app.post("/v1/sync/semantic")
async def sync_semantic(api_req: Request, request: SyncSemanticRequest) -> Any:
    """Sync a batch of semantic knowledge items from an edge device (append strategy)."""
    tenant_id = getattr(api_req.state, "tenant_id", "default")
    user_id = getattr(api_req.state, "user_id", "default")
    device_id = getattr(api_req.state, "device_id", "default")

    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    import hashlib
    import json

    synced_count = 0
    for item in request.knowledge_items:
        payload = dict(item)

        # Deduplication hash based on content
        content_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if content_hash in _sync_dedup_cache:
            continue

        _sync_dedup_cache.add(content_hash)
        if len(_sync_dedup_cache) > 10000:
            _sync_dedup_cache.pop()

        payload["tenant_id"] = tenant_id
        payload["user_id"] = user_id
        payload["device_id"] = device_id

        msg = Message(
            type=MessageType.EVENT,
            source_node_id=f"edge_sync_{device_id}",
            tenant_id=tenant_id,
            user_id=user_id,
            device_id=device_id,
            session_id="sync_session",
            topic="knowledge.store",
            payload=payload,
        )
        await bus.publish("knowledge.store", msg)
        synced_count += 1

    return {
        "status": "success",
        "synced": synced_count,
        "skipped": len(request.knowledge_items) - synced_count,
    }


@app.post("/v1/feedback")
async def submit_feedback(request: FeedbackRequest) -> Any:
    """
    Submit user feedback on a response for RLHF / DPO continuous learning.

    Feedback is published to the LearnerNode which accumulates samples
    and triggers DPO training once a batch threshold is reached.
    """
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    feedback_msg = Message(
        type=MessageType.FEEDBACK,
        source_node_id="api_server",
        tenant_id=request.tenant_id,
        topic="system.feedback",
        payload={
            "message_id": request.message_id,
            "rating": request.rating,
            "prompt": request.prompt,
            "response": request.response,
            "comment": request.comment,
        },
    )
    await bus.publish("system.feedback", feedback_msg)

    return {
        "status": "accepted",
        "message_id": request.message_id,
        "rating": request.rating,
    }


# ─── Knowledge Graph Endpoints ────────────────────────────────────────────────


@app.get("/v1/knowledge/{entity}")
async def knowledge_neighbors(
    entity: str, direction: str = "both", relation_type: str | None = None
) -> Any:
    """Query KnowledgeGraph neighbors for an entity."""
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe("knowledge.response", handler)

    query = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        topic="knowledge.query",
        payload={
            "action": "neighbors",
            "entity": entity,
            "direction": direction,
            "relation_type": relation_type,
        },
        correlation_id=correlation_id,
    )
    await bus.publish("knowledge.query", query)

    try:
        result = await asyncio.wait_for(response_future, timeout=5.0)
        return result.payload
    except (TimeoutError, asyncio.TimeoutError):
        return {"neighbors": [], "entity": entity}


@app.get("/v1/knowledge/path")
async def knowledge_path(from_entity: str, to_entity: str, max_depth: int = 5) -> Any:
    """Find shortest path between two entities in the KnowledgeGraph."""
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe("knowledge.response", handler)

    query = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        topic="knowledge.query",
        payload={"action": "path", "from": from_entity, "to": to_entity, "max_depth": max_depth},
        correlation_id=correlation_id,
    )
    await bus.publish("knowledge.query", query)

    try:
        result = await asyncio.wait_for(response_future, timeout=5.0)
        return result.payload
    except (TimeoutError, asyncio.TimeoutError):
        return {"path": None, "from": from_entity, "to": to_entity}


@app.get("/v1/knowledge/subgraph/{entity}")
async def knowledge_subgraph(entity: str, depth: int = 2) -> Any:
    """Extract a subgraph around an entity."""
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe("knowledge.response", handler)

    query = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        topic="knowledge.query",
        payload={"action": "subgraph", "entity": entity, "depth": depth},
        correlation_id=correlation_id,
    )
    await bus.publish("knowledge.query", query)

    try:
        result = await asyncio.wait_for(response_future, timeout=5.0)
        return result.payload
    except (TimeoutError, asyncio.TimeoutError):
        return {"subgraph": {"entities": [], "relations": []}, "entity": entity}


@app.get("/v1/knowledge/stats")
async def knowledge_stats() -> Any:
    """Get KnowledgeGraph statistics."""
    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")

    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

    async def handler(msg: Message) -> None:
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe("knowledge.response", handler)

    query = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        topic="knowledge.query",
        payload={"action": "stats"},
        correlation_id=correlation_id,
    )
    await bus.publish("knowledge.query", query)

    try:
        result = await asyncio.wait_for(response_future, timeout=5.0)
        return result.payload
    except (TimeoutError, asyncio.TimeoutError):
        return {"entity_count": 0, "relation_count": 0}


# ─── Rules Endpoint ───────────────────────────────────────────────────────────


@app.get("/v1/rules")
async def list_rules() -> Any:
    """List extracted if→then rules from the RuleExtractorNode."""
    # Find the rule extractor node in the running nodes
    brain = _state.get("brain")
    node_map = _get_node_map(brain)
    for node in node_map.values():
        if hasattr(node, "rules"):
            return {
                "rules": [r.to_dict() for r in node.rules],
                "total": len(node.rules),
            }
    return {"rules": [], "total": 0}


# ─── WebSocket Streaming ─────────────────────────────────────────────────────


@app.websocket("/v1/chat/ws")
async def chat_websocket(ws: WebSocket) -> None:
    """
    Bidirectional WebSocket for real-time chat streaming.

    Authentication: Pass a JWT token as a query parameter:
        ws://host/v1/chat/ws?token=<jwt>
    """
    import os

    import jwt as pyjwt

    from hbllm.security.tenant_guard import TenantContext

    # ── 1. Extract and verify JWT ──
    token = ws.query_params.get("token", "")
    if not token and os.environ.get("HBLLM_ENV", "").lower() != "production":
        from hbllm.security.identity_resolver import resolve_sovereign_identity

        tenant_id, device_id = resolve_sovereign_identity()
        user_id = "default"
    else:
        if not token:
            await ws.close(code=1008, reason="Missing authentication token")
            return

        secret_key = os.environ.get("HBLLM_JWT_SECRET", "")
        if not secret_key:
            auth_mw = next(
                (
                    m
                    for m in getattr(app, "user_middleware", [])
                    if hasattr(m, "kwargs") and "secret_key" in m.kwargs
                ),
                None,
            )
            if auth_mw:
                secret_key = auth_mw.kwargs.get("secret_key", "")

        if not secret_key:
            logger.warning("Chat WS: No JWT secret configured — rejecting connection")
            await ws.close(code=1011, reason="Server auth not configured")
            return

        try:
            payload = pyjwt.decode(token, secret_key, algorithms=["HS256"])
            tenant_id = payload.get("tenant_id")
            if not tenant_id:
                await ws.close(code=1008, reason="Token missing tenant_id")
                return
            user_id = payload.get("user_id", "default")
            device_id = payload.get("device_id", "default")
        except pyjwt.ExpiredSignatureError:
            await ws.close(code=1008, reason="Token expired")
            return
        except pyjwt.InvalidTokenError:
            await ws.close(code=1008, reason="Invalid token")
            return

    await ws.accept()
    import json as _json

    brain = _state.get("brain")
    bus = getattr(brain, "bus", None)
    if not bus:
        await ws.send_json({"error": "Brain pipeline not initialized"})
        await ws.close(code=1011)
        return

    try:
        async with TenantContext(tenant_id, user_id=user_id, device_id=device_id):
            while True:
                # Receive user message
                raw = await ws.receive_text()
                try:
                    data = _json.loads(raw)
                except _json.JSONDecodeError:
                    await ws.send_json({"error": "Invalid JSON"})
                    continue

                session_id = data.get("session_id", str(uuid.uuid4()))
                text = data.get("text", "")

                if not text:
                    await ws.send_json({"error": "Missing 'text' field"})
                    continue

                correlation_id = str(uuid.uuid4())
                response_future: asyncio.Future[Message] = asyncio.get_event_loop().create_future()

                async def output_handler(msg: Message) -> None:
                    if msg.correlation_id == correlation_id and not response_future.done():
                        response_future.set_result(msg)

                await bus.subscribe("sensory.output", output_handler)

                # Store user message
                memory_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id="api_server",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    device_id=device_id,
                    session_id=session_id,
                    topic="memory.store",
                    payload={
                        "session_id": session_id,
                        "tenant_id": tenant_id,
                        "role": "user",
                        "content": text,
                        "user_id": user_id,
                        "device_id": device_id,
                    },
                )
                await bus.publish("memory.store", memory_msg)

                # Route the query
                query_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id="api_server",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    device_id=device_id,
                    session_id=session_id,
                    topic="router.query",
                    payload=QueryPayload(text=text).model_dump(),
                    correlation_id=correlation_id,
                )
                await bus.publish("router.query", query_msg)

                from hbllm.brain.factory import BrainConfig

                config = _state.get("config") or BrainConfig()
                api_timeout = config.api_timeout

                # Wait for response and stream it
                try:
                    result_msg = await asyncio.wait_for(response_future, timeout=api_timeout)
                    response_text = result_msg.payload.get("text", "")

                    # Stream token-by-token for a natural feel
                    tokens = response_text.split(" ")
                    for i, token in enumerate(tokens):
                        piece = token if i == 0 else " " + token
                        await ws.send_json({"token": piece, "done": False})

                    await ws.send_json(
                        {"token": "", "done": True, "correlation_id": correlation_id}
                    )

                    # Store assistant response
                    store_msg = Message(
                        type=MessageType.EVENT,
                        source_node_id="api_server",
                        tenant_id=tenant_id,
                        user_id=user_id,
                        device_id=device_id,
                        session_id=session_id,
                        topic="memory.store",
                        payload={
                            "session_id": session_id,
                            "tenant_id": tenant_id,
                            "role": "assistant",
                            "content": response_text,
                            "user_id": user_id,
                            "device_id": device_id,
                        },
                    )
                    await bus.publish("memory.store", store_msg)

                except (TimeoutError, asyncio.TimeoutError):
                    await ws.send_json({"token": "", "done": True, "error": "timeout"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error("WebSocket error: %s", e)
        try:
            await ws.close(code=1011)
        except Exception:
            pass


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def main() -> None:
    """Run the FastAPI server with uvicorn."""
    import uvicorn

    parser = argparse.ArgumentParser(description="HBLLM Cognitive API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "hbllm.serving.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
