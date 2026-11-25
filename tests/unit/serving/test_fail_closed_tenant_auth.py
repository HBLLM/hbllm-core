"""Security regression tests guaranteeing fail-closed tenant authentication across HTTP & WS endpoints."""

from __future__ import annotations

import time
from types import SimpleNamespace

import jwt
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from hbllm.serving.api import _state, app
from hbllm.serving.deps import get_tenant_id as deps_get_tenant_id
from hbllm.serving.studio.helpers import get_tenant_id as studio_get_tenant_id


def _configure_jwt_secret(monkeypatch: pytest.MonkeyPatch, secret: str) -> None:
    monkeypatch.setenv("HBLLM_JWT_SECRET", secret)
    monkeypatch.setenv("HBLLM_ENV", "production")
    monkeypatch.delenv("HBLLM_UNSAFE_DEV_AUTH", raising=False)


def test_get_tenant_helpers_fail_closed() -> None:
    req = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(HTTPException) as exc1:
        deps_get_tenant_id(req)  # type: ignore[arg-type]
    assert exc1.value.status_code == 401

    with pytest.raises(HTTPException) as exc2:
        studio_get_tenant_id(req)
    assert exc2.value.status_code == 401


def test_unauthenticated_endpoints_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test_fail_closed_secret_key_32_chars"
    _configure_jwt_secret(monkeypatch, secret)
    client = TestClient(app)

    for method, path in (
        ("GET", "/metrics"),
        ("GET", "/studio/voice/speakers"),
        ("GET", "/studio/emotion/state"),
        ("GET", "/v1/devices"),
    ):
        resp = client.request(method, path)
        assert resp.status_code == 401, f"{method} {path} unexpectedly returned {resp.status_code}"


def test_audio_ws_rejects_unauthenticated_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test_fail_closed_secret_key_32_chars"
    _configure_jwt_secret(monkeypatch, secret)
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/v1/audio/ws"):
            pass
    assert exc_info.value.code == 1008


def test_device_register_ignores_client_spoofed_tenant_id(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "test_fail_closed_secret_key_32_chars"
    _configure_jwt_secret(monkeypatch, secret)

    class _FakeBridge:
        def __init__(self) -> None:
            self.registered = None

        def register_device(self, device) -> None:
            self.registered = device

    fake_bridge = _FakeBridge()
    prev = _state.get("device_bridge")
    _state["device_bridge"] = fake_bridge
    try:
        token = jwt.encode(
            {"tenant_id": "real_tenant", "exp": int(time.time()) + 3600},
            secret,
            algorithm="HS256",
        )
        client = TestClient(app)
        resp = client.post(
            "/v1/devices/presence/register",
            headers={"Authorization": f"Bearer {token}"},
            json={"device_id": "dev-1", "tenant_id": "attacker_tenant"},
        )
        assert resp.status_code == 200
        assert resp.json()["tenant_id"] == "real_tenant"
        assert fake_bridge.registered is not None
        assert fake_bridge.registered.tenant_id == "real_tenant"
    finally:
        if prev is None:
            _state.pop("device_bridge", None)
        else:
            _state["device_bridge"] = prev
