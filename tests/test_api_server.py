import pytest

from hbllm.serving.api import app, ChatRequest, HealthResponse


@pytest.mark.asyncio
async def test_fastapi_app_exists():
    """Verify the FastAPI app is properly configured."""
    assert app.title == "HBLLM Cognitive API"
    assert app.version == "1.0.0"


@pytest.mark.asyncio
async def test_chat_request_schema():
    """Verify the ChatRequest model validates correctly."""
    req = ChatRequest(tenant_id="tenant_1", text="Hello world")
    assert req.tenant_id == "tenant_1"
    assert req.text == "Hello world"
    assert req.session_id  # auto-generated
    assert req.model_size == "125M"


@pytest.mark.asyncio
async def test_chat_request_validation():
    """Verify ChatRequest rejects empty text."""
    with pytest.raises(Exception):
        ChatRequest(tenant_id="t", text="")


@pytest.mark.asyncio
async def test_health_response_schema():
    """Verify the HealthResponse model."""
    resp = HealthResponse(status="healthy", nodes_registered=22, bus_type="in_process")
    assert resp.status == "healthy"
    assert resp.nodes_registered == 22
