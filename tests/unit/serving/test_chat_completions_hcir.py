"""Tests proving /v1/chat/completions routes through per-tenant Pure-HCIR runtimes (C-2)."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

from hbllm.serving.api import _state, app
from hbllm.serving.cognitive_registry import CognitiveSystemRegistry


def _auth_headers(monkeypatch: pytest.MonkeyPatch, tenant_id: str) -> dict[str, str]:
    secret = "c2_chat_completions_test_secret_32_ch"
    monkeypatch.setenv("HBLLM_JWT_SECRET", secret)
    monkeypatch.setenv("HBLLM_ENV", "production")
    monkeypatch.delenv("HBLLM_UNSAFE_DEV_AUTH", raising=False)
    token = jwt.encode(
        {"tenant_id": tenant_id, "exp": int(time.time()) + 3600},
        secret,
        algorithm="HS256",
    )
    return {"Authorization": f"Bearer {token}"}


def test_chat_completions_serves_pure_hcir_and_isolates_tenants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = CognitiveSystemRegistry(base_memory_dir=tmp_path)
    prev_reg = _state.get("cognitive_registry")
    prev_sys = _state.pop("cognitive_system", None)
    _state["cognitive_registry"] = registry
    try:
        client = TestClient(app)

        # 1. Tenant Alpha asserts a spatial fact and hits Pure-HCIR directly
        resp_a = client.post(
            "/v1/chat/completions",
            headers=_auth_headers(monkeypatch, "tenant_alpha"),
            json={"messages": [{"role": "user", "content": "The red ball is on the table."}]},
        )
        assert resp_a.status_code == 200
        assert resp_a.headers.get("X-HBLLM-Engine") == "pure-hcir"
        body_a = resp_a.json()
        assert body_a["model"] == "hbllm-pure-hcir"
        assert body_a["choices"][0]["message"]["content"]

        # Verify distinct workspace tenant binding
        sys_a = registry.get_or_create("tenant_alpha")
        assert sys_a.tiered_workspace.tenant_id == "tenant_alpha"

        # 2. Tenant Alpha recalls its own fact
        recall_a = client.post(
            "/v1/chat/completions",
            headers=_auth_headers(monkeypatch, "tenant_alpha"),
            json={"messages": [{"role": "user", "content": "Where is the red ball?"}]},
        )
        assert recall_a.status_code == 200
        assert recall_a.headers.get("X-HBLLM-Engine") == "pure-hcir"
        assert "table" in recall_a.json()["choices"][0]["message"]["content"].lower()

        # 3. Tenant Beta gets an isolated runtime that does NOT see Alpha's red ball
        sys_b = registry.get_or_create("tenant_beta")
        assert sys_b is not sys_a
        assert sys_b.tiered_workspace.tenant_id == "tenant_beta"
        beta_turn = sys_b.process_dialogue("Where is the red ball?")
        assert not getattr(beta_turn, "is_success", False)
        assert "table" not in str(getattr(beta_turn, "response_text", "") or "").lower()
    finally:
        registry.clear()
        if prev_reg is None:
            _state.pop("cognitive_registry", None)
        else:
            _state["cognitive_registry"] = prev_reg
        if prev_sys is not None:
            _state["cognitive_system"] = prev_sys


def test_chat_completions_logs_warning_and_counts_exception_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    registry = CognitiveSystemRegistry(base_memory_dir=tmp_path)
    prev_reg = _state.get("cognitive_registry")
    prev_sys = _state.pop("cognitive_system", None)

    class _BrokenSystem:
        def process_dialogue(self, text: str) -> object:
            raise RuntimeError("boom")

    _state["cognitive_registry"] = registry
    _state["cognitive_system"] = _BrokenSystem()
    try:
        client = TestClient(app)
        with caplog.at_level(logging.WARNING, logger="hbllm.serving.api"):
            # No brain/provider configured -> returns 503 after fallback, but warning & counter fire
            resp = client.post(
                "/v1/chat/completions",
                headers=_auth_headers(monkeypatch, "tenant_err"),
                json={"messages": [{"role": "user", "content": "Hello"}]},
            )
            assert resp.status_code == 503
        assert registry.hcir_fallback_count == 1
        assert any("Pure-HCIR fast path failed" in rec.message for rec in caplog.records)
    finally:
        registry.clear()
        _state.pop("cognitive_system", None)
        if prev_reg is None:
            _state.pop("cognitive_registry", None)
        else:
            _state["cognitive_registry"] = prev_reg
        if prev_sys is not None:
            _state["cognitive_system"] = prev_sys
