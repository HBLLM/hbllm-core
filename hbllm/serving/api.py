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
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType, QueryPayload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Request / Response Schemas ───────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message from a tenant."""
    tenant_id: str = Field(..., description="Unique tenant identifier")
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Session identifier")
    text: str = Field(..., min_length=1, description="User message text")
    model_size: str = Field(default="125M", description="Model size to use")
    provider: str | None = Field(default=None, description="LLM provider override (openai, anthropic, local)")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")


class ChatResponse(BaseModel):
    """Response from the cognitive pipeline."""
    tenant_id: str
    session_id: str
    correlation_id: str
    response_text: str
    source_node: str = "decision"
    confidence: float = 0.0
    provider_used: str | None = None
    usage: dict[str, int] | None = None


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


# ─── Global State ─────────────────────────────────────────────────────────────

_state: dict[str, Any] = {}


async def _boot_brain(model_size: str = "125M", bus_type: str = "inprocess", redis_url: str = "redis://localhost:6379"):
    """Initialize the full brain pipeline."""
    # Lazy imports — keeps module importable without the full ML stack
    import torch
    from hbllm.brain.planner_node import PlannerNode
    from hbllm.brain.router_node import RouterNode
    from hbllm.brain.llm_interface import LLMInterface
    from hbllm.brain.workspace_node import WorkspaceNode
    from hbllm.brain.world_model_node import WorldModelNode
    from hbllm.brain.sleep_node import SleepCycleNode
    from hbllm.brain.critic_node import CriticNode
    from hbllm.brain.decision_node import DecisionNode
    from hbllm.brain.learner_node import LearnerNode
    from hbllm.brain.spawner_node import SpawnerNode
    from hbllm.brain.meta_node import MetaReasoningNode
    from hbllm.memory.memory_node import MemoryNode
    from hbllm.modules.base_module import DomainModuleNode
    from hbllm.perception.vision_node import VisionNode
    from hbllm.perception.audio_in_node import AudioInputNode
    from hbllm.perception.audio_out_node import AudioOutputNode
    from hbllm.actions.execution_node import ExecutionNode
    from hbllm.actions.browser_node import BrowserNode
    from hbllm.actions.logic_node import LogicNode
    from hbllm.actions.fuzzy_node import FuzzyNode
    from hbllm.actions.api_node import ApiNode
    from hbllm.network.redis_bus import RedisBus
    from hbllm.network.registry import ServiceRegistry
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForCausalLM
    from hbllm_tokenizer_rs import Vocab
    
    # 1. Bus
    if bus_type == "redis":
        bus = RedisBus(redis_url=redis_url)
    else:
        bus = InProcessBus()
    
    registry = ServiceRegistry()
    await bus.start()

    # 2. Model
    logger.info("Loading base transformer model (%s)...", model_size)
    config = get_config(model_size)
    model = HBLLMForCausalLM(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # 3. Tokenizer
    logger.info("Loading tokenizer...")
    vocab = Vocab.from_file("test_workspace/vocab.json")

    # 4. LLM Interface
    llm_interface = LLMInterface(model=model, tokenizer=vocab, device=device)

    # 5. Nodes
    nodes = [
        MemoryNode(node_id="memory_01", db_path="chat_memory.db"),
        RouterNode(node_id="router_01", llm=llm_interface),
        PlannerNode(node_id="planner_01"),
        LearnerNode(node_id="learner_01"),
        SpawnerNode(node_id="spawner_01", model=model, tokenizer=vocab),
        MetaReasoningNode(node_id="meta_01"),
        VisionNode(node_id="vision_01"),
        AudioInputNode(node_id="audio_in_01", model_size="tiny"),
        AudioOutputNode(node_id="audio_out_01"),
        ExecutionNode(node_id="exec_01"),
        BrowserNode(node_id="browser_01"),
        LogicNode(node_id="logic_01", llm=llm_interface),
        FuzzyNode(node_id="fuzzy_01", llm=llm_interface),
        WorkspaceNode(node_id="workspace_01"),
        WorldModelNode(node_id="world_model_01"),
        SleepCycleNode(node_id="sleep_01", idle_timeout_seconds=60.0),
        CriticNode(node_id="critic_01", llm=llm_interface),
        DecisionNode(node_id="decision_01", llm=llm_interface),
        ApiNode(node_id="api_01", llm=llm_interface),
        DomainModuleNode(node_id="domain_general", domain_name="general", model=model, tokenizer=vocab),
        DomainModuleNode(node_id="domain_coding", domain_name="coding", model=model, tokenizer=vocab),
        DomainModuleNode(node_id="domain_math", domain_name="math", model=model, tokenizer=vocab),
    ]

    for node in nodes:
        await registry.register(node.get_info())
        await node.start(bus)
    
    _state["bus"] = bus
    _state["registry"] = registry
    _state["nodes"] = nodes
    _state["bus_type"] = bus_type
    
    _state["mode"] = "full"
    logger.info("Brain pipeline booted with %d nodes.", len(nodes))


async def _shutdown_brain():
    """Gracefully shutdown all nodes and the bus."""
    nodes = _state.get("nodes", [])
    for node in reversed(nodes):
        await node.stop()
    
    bus = _state.get("bus")
    if bus:
        await bus.stop()
    
    logger.info("Brain pipeline shutdown complete.")


# ─── FastAPI Application ──────────────────────────────────────────────────────

async def _boot_provider_mode():
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
async def lifespan(app: FastAPI):
    """Boot the brain on startup, fall back to provider mode if it fails."""
    try:
        await _boot_brain()
        logger.info("Full brain pipeline active")
    except Exception as e:
        logger.warning("Full brain boot failed (%s). Falling back to provider mode.", e)
        await _boot_provider_mode()

    # ── Cloud features (SaaS layer) — graceful fallback for OSS mode ──
    try:
        from hbllm_cloud.tenant_manager import TenantManager
        from hbllm_cloud.admin_api import create_admin_router
        from hbllm_cloud.dashboard.routes import create_dashboard_router
        from hbllm.brain.policy_engine import PolicyEngine
        from hbllm.serving.security import ApiKeyManager
        from fastapi.staticfiles import StaticFiles

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
        import hbllm_cloud.dashboard as _dash_pkg
        static_dir = Path(_dash_pkg.__file__).parent / "static"
        if static_dir.exists():
            app.mount("/admin/static", StaticFiles(directory=str(static_dir)), name="admin-static")

        # Auth middleware (protects /admin/* except /admin/login and /admin/static)
        from hbllm_cloud.dashboard.auth import DashboardAuthMiddleware
        app.add_middleware(DashboardAuthMiddleware)

        # API security middleware (protects /v1/* with API key auth + rate limiting)
        from hbllm_cloud.api_middleware import ApiSecurityMiddleware
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
        from hbllm_cloud.knowledge.embeddings import EmbeddingsService
        from hbllm_cloud.knowledge.vector_store import VectorStore
        from hbllm_cloud.knowledge.processor import DocumentProcessor
        from hbllm_cloud.knowledge.api import create_knowledge_router
        from hbllm_cloud.usage import UsageTracker
        from hbllm_cloud.billing import BillingManager

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
        @app.get("/v1/usage/{tenant_id}")
        async def get_usage(tenant_id: str, days: int = 30):
            return usage_tracker.get_tenant_usage(tenant_id, days)

        @app.get("/v1/usage/{tenant_id}/daily")
        async def get_daily_usage(tenant_id: str, days: int = 30):
            return {"daily": usage_tracker.get_daily_usage(tenant_id, days)}

        @app.get("/v1/billing/plans")
        async def list_plans():
            return {"plans": billing.list_plans()}

        @app.get("/v1/billing/{tenant_id}")
        async def get_billing(tenant_id: str, days: int = 30):
            usage = usage_tracker.get_tenant_usage(tenant_id, days)
            cost = usage_tracker.estimate_cost(tenant_id, days)
            return {"usage": usage, "cost": cost}

        logger.info("📊 Knowledge base, usage tracking, and billing enabled")
    except ImportError as e:
        logger.info("Knowledge/billing modules not available: %s", e)

    # ── Tenant Portal (self-service UI) ──
    try:
        from hbllm_cloud.portal.routes import create_portal_router
        from fastapi.staticfiles import StaticFiles as SF2

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

        import hbllm_cloud.portal as _portal_pkg
        portal_static = Path(_portal_pkg.__file__).parent / "static"
        if portal_static.exists():
            app.mount("/portal/static", SF2(directory=str(portal_static)), name="portal-static")

        logger.info("🌐 Tenant portal enabled at /portal/")
    except ImportError as e:
        logger.info("Tenant portal not available: %s", e)

    # ── High-Value Platform Features ──
    try:
        from hbllm_cloud.workflows import WorkflowEngine
        from hbllm_cloud.extraction import DataExtractor
        from hbllm_cloud.agents import AgentRegistry
        from hbllm_cloud.multilang import MultiLanguageService

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
        async def list_workflow_templates():
            return {"templates": workflow_engine.list_templates()}

        @app.post("/v1/workflows/run")
        async def run_workflow(request: Request):
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
        async def list_extraction_schemas():
            return {"schemas": extractor.list_schemas()}

        @app.post("/v1/extract")
        async def extract_data(request: Request):
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
        async def classify_text(request: Request):
            body = await request.json()
            return await extractor.classify(
                text=body["text"],
                categories=body["categories"],
                multi_label=body.get("multi_label", False),
            )

        # ── Agent Endpoints ──
        @app.post("/v1/agents")
        async def create_agent(request: Request):
            body = await request.json()
            agent = agent_registry.create_agent(
                tenant_id=body["tenant_id"],
                name=body["name"],
                system_prompt=body["system_prompt"],
                tools=body.get("tools", []),
            )
            return {"agent_id": agent.agent_id, "name": agent.name}

        @app.get("/v1/agents/{tenant_id}")
        async def list_agents(tenant_id: str):
            agents = agent_registry.list_agents(tenant_id)
            return {"agents": [{"agent_id": a.agent_id, "name": a.name} for a in agents]}

        @app.get("/v1/agents/{tenant_id}/tools")
        async def list_agent_tools(tenant_id: str):
            return {"tools": agent_registry.list_tools(tenant_id)}

        @app.post("/v1/agents/{agent_id}/run")
        async def run_agent(agent_id: str, request: Request):
            body = await request.json()
            context = {
                "vector_store": _state.get("vector_store"),
                "embeddings": _state.get("embeddings"),
                "tenant_id": body.get("tenant_id", ""),
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

        @app.post("/v1/agents/{tenant_id}/tools")
        async def register_tool(tenant_id: str, request: Request):
            body = await request.json()
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
        async def detect_language(request: Request):
            body = await request.json()
            lang = await multilang.detect_language(body["text"])
            return {"language": lang, "name": multilang.SUPPORTED_LANGUAGES.get(lang, lang)}

        @app.post("/v1/translate")
        async def translate_text(request: Request):
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
        async def list_languages():
            return {"languages": multilang.list_languages()}

        # ── Embeddable Widget ──
        from fastapi.responses import FileResponse
        import hbllm_cloud.widget as _widget_pkg
        widget_path = Path(_widget_pkg.__file__).parent / "hbllm-widget.js"

        @app.get("/widget/hbllm-widget.js")
        async def serve_widget():
            return FileResponse(
                str(widget_path),
                media_type="application/javascript",
                headers={"Cache-Control": "public, max-age=3600", "Access-Control-Allow-Origin": "*"},
            )

        logger.info("🚀 Platform features enabled (workflows, extraction, agents, multilang, widget)")
    except ImportError as e:
        logger.info("Platform features not available: %s", e)

    # ── AGI Differentiators: Multi-Perspective Analysis + XAI ──
    try:
        from hbllm_cloud.multi_perspective import MultiPerspectiveAnalyzer
        from hbllm_cloud.xai import ExplainabilityEngine

        provider = _state.get("provider")
        mpa = MultiPerspectiveAnalyzer(provider=provider)
        xai = ExplainabilityEngine(db_path="data/xai.db")

        _state["multi_perspective"] = mpa
        _state["xai"] = xai

        # ── Multi-Perspective Analysis ──
        @app.get("/v1/analyze/lenses")
        async def list_lenses():
            return {"lenses": mpa.list_lenses()}

        @app.post("/v1/analyze")
        async def multi_perspective_analyze(request: Request):
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
                        "lens": p.lens_id, "name": p.lens_name, "icon": p.icon,
                        "node_analog": p.node_analog,
                        "analysis": p.analysis, "confidence": p.confidence,
                        "processing_time_ms": p.processing_time_ms,
                    }
                    for p in result.perspectives
                ],
                "lenses_used": result.lenses_used,
                "total_tokens": result.total_tokens,
                "total_duration_ms": result.total_duration_ms,
            }

        @app.post("/v1/analyze/quick")
        async def quick_analyze(request: Request):
            body = await request.json()
            return await mpa.quick_analyze(
                query=body["query"],
                context=body.get("context", ""),
            )

        # ── Explainable AI (XAI) ──
        @app.get("/v1/xai/{tenant_id}/traces")
        async def get_traces(tenant_id: str, limit: int = 50):
            return {"traces": xai.get_tenant_traces(tenant_id, limit)}

        @app.get("/v1/xai/trace/{trace_id}")
        async def get_trace(trace_id: str):
            trace = xai.get_trace(trace_id)
            if not trace:
                raise HTTPException(status_code=404, detail="Trace not found")
            return trace

        @app.get("/v1/xai/{tenant_id}/audit")
        async def get_audit_log(tenant_id: str, limit: int = 100):
            return {"audit_log": xai.get_audit_log(tenant_id, limit)}

        @app.get("/v1/xai/{tenant_id}/compliance")
        async def compliance_report(tenant_id: str, days: int = 30):
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
    description="REST API for the Hierarchical Brain LLM — a modular cognitive architecture with multi-tenant isolation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check server health and node count."""
    node_count = len(_state.get("nodes", []))
    mode = _state.get("mode", "unknown")
    provider = _state.get("provider")
    return HealthResponse(
        status="healthy",
        nodes_registered=node_count,
        bus_type=_state.get("bus_type", "unknown"),
        provider_mode=mode,
    )


@app.get("/metrics")
async def metrics():
    """Return real-time MessageBus performance metrics."""
    bus = _state.get("bus")
    if not bus or not hasattr(bus, "metrics"):
        return {"error": "Bus not initialized or metrics unavailable"}
    return bus.metrics.snapshot()


# ─── Provider-based Chat (lightweight) ────────────────────────────────────────

async def _chat_via_provider(request: ChatRequest) -> ChatResponse:
    """Handle chat using the LLM provider abstraction (no brain pipeline)."""
    from hbllm.serving.provider import get_provider

    correlation_id = str(uuid.uuid4())

    # Resolve provider: request override → state default
    if request.provider:
        provider = get_provider(request.provider)
    elif "provider" in _state:
        provider = _state["provider"]
    else:
        raise HTTPException(status_code=503, detail="No LLM provider configured")

    # Build messages
    messages = []
    system_content = request.system_prompt or (
        "You are HBLLM, an advanced AI assistant powered by a modular cognitive architecture. "
        "Be helpful, concise, and accurate."
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

    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": request.text})

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
    bus = _state["bus"]
    correlation_id = str(uuid.uuid4())

    response_future: asyncio.Future = asyncio.get_event_loop().create_future()

    async def output_handler(msg: Message):
        if msg.correlation_id == correlation_id and not response_future.done():
            response_future.set_result(msg)

    await bus.subscribe("sensory.output", output_handler)

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
        },
    )
    await bus.publish("memory.store", memory_msg)

    # Send to router
    query_msg = Message(
        type=MessageType.QUERY,
        source_node_id="api_server",
        tenant_id=request.tenant_id,
        session_id=request.session_id,
        topic="router.query",
        payload=QueryPayload(text=request.text).model_dump(),
        correlation_id=correlation_id,
    )
    await bus.publish("router.query", query_msg)

    try:
        result_msg = await asyncio.wait_for(response_future, timeout=30.0)
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
            },
        )
        await bus.publish("memory.store", store_msg)

        return ChatResponse(
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            correlation_id=correlation_id,
            response_text=response_text,
            source_node=source,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline timed out (30s)")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Send a message through the HBLLM system.

    Automatically routes to the full brain pipeline or external LLM provider
    based on system mode. Use 'provider' field to force a specific provider.
    """
    mode = _state.get("mode", "provider")

    # Explicit provider override always uses provider path
    if request.provider:
        return await _chat_via_provider(request)

    # Route based on mode
    if mode == "full" and _state.get("bus"):
        return await _chat_via_brain(request)
    else:
        return await _chat_via_provider(request)


@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream a response from the cognitive pipeline as Server-Sent Events (SSE).
    
    Each SSE event has the format:
        data: {"token": "...", "done": false}
        data: {"token": "", "done": true, "correlation_id": "..."}
    """
    bus = _state.get("bus")
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")
    
    correlation_id = str(uuid.uuid4())
    
    token_queue: asyncio.Queue = asyncio.Queue()
    
    async def output_handler(msg: Message):
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
        session_id=request.session_id,
        topic="memory.store",
        payload={
            "session_id": request.session_id,
            "tenant_id": request.tenant_id,
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
        session_id=request.session_id,
        topic="router.query",
        payload=QueryPayload(text=request.text).model_dump(),
        correlation_id=correlation_id,
    )
    await bus.publish("router.query", query_msg)
    
    async def event_generator():
        import json as _json
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(token_queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
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


@app.get("/v1/memory/{tenant_id}/{session_id}")
async def get_memory(tenant_id: str, session_id: str, limit: int = 20):
    """Retrieve recent conversation history for a tenant's session."""
    bus = _state.get("bus")
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")
    
    correlation_id = str(uuid.uuid4())
    response_future: asyncio.Future = asyncio.get_event_loop().create_future()
    
    async def memory_handler(msg: Message):
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
    except asyncio.TimeoutError:
        return {"session_id": session_id, "turns": []}


@app.post("/v1/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a response for RLHF / DPO continuous learning.
    
    Feedback is published to the LearnerNode which accumulates samples
    and triggers DPO training once a batch threshold is reached.
    """
    bus = _state.get("bus")
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


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
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
