import os
import pytest
import jwt
from fastapi.testclient import TestClient
from hbllm.serving.api import app
from hbllm.security.tenant_guard import TenantContext, require_tenant, TenantIsolationError
from fastapi import Request


# ── Mock/Test Endpoints (named not starting with 'test_') ──
@app.get("/v1/mock-isolation")
@require_tenant
async def mock_isolation_endpoint(request: Request, tenant_id: str = ""):
    """A test endpoint requiring the tenant context to match the param."""
    return {"status": "success", "tenant_id": tenant_id}


@app.get("/v1/mock-isolation-error")
async def mock_isolation_error_endpoint():
    """An endpoint raising a TenantIsolationError to test global handling."""
    raise TenantIsolationError("Mock isolation violation", tenant_id="tenant_x")


@pytest.fixture
def jwt_secret():
    return "test_secret_key"


@pytest.fixture
def client(jwt_secret, monkeypatch):
    # Enable test auth middleware configuration
    monkeypatch.setenv("HBLLM_JWT_SECRET", jwt_secret)
    monkeypatch.setenv("HBLLM_ENV", "production")
    monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")

    # Set the middleware secret key directly for testing
    for middleware in app.user_middleware:
        if middleware.cls.__name__ == "JWTAuthMiddleware":
            middleware.kwargs["secret_key"] = jwt_secret

    return TestClient(app)


def test_unauthenticated_request_rejected(client):
    """Verify unauthenticated requests are rejected with 401 in production."""
    response = client.get("/v1/mock-isolation?tenant_id=tenant_A")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json()["detail"]


def test_authenticated_request_success(client, jwt_secret):
    """Verify authenticated requests pass identity and run in TenantContext."""
    token = jwt.encode(
        {"tenant_id": "tenant_A", "user_id": "user_1"}, jwt_secret, algorithm="HS256"
    )
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/v1/mock-isolation?tenant_id=tenant_A", headers=headers)
    assert response.status_code == 200
    assert response.json()["tenant_id"] == "tenant_A"


def test_cross_tenant_access_blocked_by_guard(client, jwt_secret):
    """Verify that cross-tenant mismatch triggers TenantIsolationError -> 403."""
    # Token is for tenant_A, but we request tenant_B (triggers require_tenant check)
    token = jwt.encode(
        {"tenant_id": "tenant_A", "user_id": "user_1"}, jwt_secret, algorithm="HS256"
    )
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/v1/mock-isolation?tenant_id=tenant_B", headers=headers)
    assert response.status_code == 403
    assert "Access denied" in response.json()["detail"]


def test_global_exception_handler_maps_403(client, jwt_secret):
    """Verify that TenantIsolationError is mapped to a clean 403 by the exception handler."""
    token = jwt.encode({"tenant_id": "tenant_A"}, jwt_secret, algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/v1/mock-isolation-error", headers=headers)
    assert response.status_code == 403
    assert "Access denied: Mock isolation violation" in response.json()["detail"]


def test_unauthenticated_request_fallback_in_dev(jwt_secret, monkeypatch):
    """Verify that in non-production, unauthenticated requests fall back to sovereign identity."""
    monkeypatch.setenv("HBLLM_ENV", "development")
    monkeypatch.setenv("HBLLM_JWT_SECRET", jwt_secret)
    monkeypatch.setenv("HBLLM_TENANT_GUARD_MODE", "strict")

    from hbllm.security.identity_resolver import resolve_sovereign_identity

    expected_tenant, _ = resolve_sovereign_identity()

    client_dev = TestClient(app)
    # No Auth header passed
    response = client_dev.get(f"/v1/mock-isolation?tenant_id={expected_tenant}")
    assert response.status_code == 200
    assert response.json()["tenant_id"] == expected_tenant


def test_chat_websocket_requires_auth(client):
    """Verify `/v1/chat/ws` rejects connection with 1008 if missing token."""
    with pytest.raises(Exception):
        with client.websocket_connect("/v1/chat/ws") as websocket:
            pass
