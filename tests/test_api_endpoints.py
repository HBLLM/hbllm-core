"""
Tests for WebSocket + Knowledge Graph + Rules API endpoints.
"""

import pytest

from hbllm.serving.api import ChatRequest, HealthResponse, app

# ── Schema Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_request_accepts_session():
    """ChatRequest auto-generates session IDs."""
    req = ChatRequest(tenant_id="t1", text="Hello")
    assert req.session_id  # Auto-generated UUID
    assert req.model_size == "125M"
    assert req.provider is None


@pytest.mark.asyncio
async def test_health_response_provider_mode():
    """HealthResponse includes provider_mode field."""
    resp = HealthResponse(status="healthy", nodes_registered=15, bus_type="in_process", provider_mode="full")
    assert resp.provider_mode == "full"


# ── Route existence tests ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_routes_registered():
    """Verify KnowledgeGraph endpoints are registered."""
    routes = [r.path for r in app.routes]
    assert "/v1/knowledge/{entity}" in routes
    assert "/v1/knowledge/path" in routes
    assert "/v1/knowledge/subgraph/{entity}" in routes
    assert "/v1/knowledge/stats" in routes


@pytest.mark.asyncio
async def test_rules_route_registered():
    """Verify rules endpoint is registered."""
    routes = [r.path for r in app.routes]
    assert "/v1/rules" in routes


@pytest.mark.asyncio
async def test_websocket_route_registered():
    """Verify WebSocket endpoint is registered."""
    routes = [r.path for r in app.routes]
    assert "/v1/chat/ws" in routes


@pytest.mark.asyncio
async def test_chat_stream_route_registered():
    """Verify SSE stream endpoint is registered."""
    routes = [r.path for r in app.routes]
    assert "/v1/chat/stream" in routes


@pytest.mark.asyncio
async def test_memory_route_registered():
    """Verify memory endpoint is registered."""
    routes = [r.path for r in app.routes]
    assert "/v1/memory/{tenant_id}/{session_id}" in routes


@pytest.mark.asyncio
async def test_feedback_route_registered():
    """Verify feedback endpoint is registered."""
    routes = [r.path for r in app.routes]
    assert "/v1/feedback" in routes


@pytest.mark.asyncio
async def test_all_core_routes_count():
    """Verify all expected core routes exist."""
    routes = [r.path for r in app.routes]
    expected = [
        "/health",
        "/metrics",
        "/v1/chat",
        "/v1/chat/stream",
        "/v1/chat/ws",
        "/v1/memory/{tenant_id}/{session_id}",
        "/v1/feedback",
        "/v1/knowledge/{entity}",
        "/v1/knowledge/path",
        "/v1/knowledge/subgraph/{entity}",
        "/v1/knowledge/stats",
        "/v1/rules",
    ]
    for ep in expected:
        assert ep in routes, f"Missing route: {ep}"
