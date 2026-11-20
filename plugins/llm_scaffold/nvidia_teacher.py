"""NVIDIA NIM / OpenAI-compatible remote teacher adapter (strictly outside hbllm/hcir/).

Lightweight HTTP client (zero torch/transformers RAM overhead) that queries
NVIDIA NIM (default ``https://integrate.api.nvidia.com/v1/chat/completions``)
using ``NVIDIA_API_KEY`` and returns structured JSON lessons for verifier-gated
distillation into HCIR.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from plugins.llm_scaffold.local_teacher import _SYSTEM_PROMPT, parse_teacher_json

logger = logging.getLogger(__name__)

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


def _load_dotenv_key(var_name: str, env_path: Path = Path(".env")) -> str:
    if not env_path.exists():
        return ""
    try:
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            k, v = stripped.split("=", 1)
            if k.strip() == var_name:
                return v.strip().strip('"').strip("'")
    except OSError:
        return ""
    return ""


class NvidiaNIMTeacher:
    """Calls an NVIDIA NIM OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 30.0,
        max_tokens: int = 320,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = (
            api_key or os.getenv("NVIDIA_API_KEY", "") or _load_dotenv_key("NVIDIA_API_KEY")
        )
        self.model = (
            model
            or os.getenv("NVIDIA_MODEL")
            or _load_dotenv_key("NVIDIA_MODEL")
            or DEFAULT_NVIDIA_MODEL
        )
        self.base_url = (
            base_url or os.getenv("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
        ).rstrip("/")
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _build_request(self, utterance: str) -> urllib.request.Request:
        if not self.api_key:
            raise RuntimeError(
                "NVIDIA_API_KEY is not set. Export NVIDIA_API_KEY='nvapi-...' before running --mode nvidia."
            )
        url = f"{self.base_url}/chat/completions"
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Unsupported NVIDIA base URL scheme: {url!r}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Utterance: {utterance}\nReturn ONLY valid JSON."},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode("utf-8")
        return urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

    def __call__(self, utterance: str) -> dict[str, Any]:
        """Query NVIDIA NIM and parse the structured JSON lesson."""
        req = self._build_request(utterance)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            logger.warning("NVIDIA NIM HTTP %s: %s", exc.code, detail)
            return {
                "response_text": "",
                "claim": "",
                "entities": [],
                "lexicon": [],
                "confidence": 0.0,
                "verified": False,
                "error": f"HTTP {exc.code}: {detail}",
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            logger.warning("NVIDIA NIM request failed: %s", exc)
            return {
                "response_text": "",
                "claim": "",
                "entities": [],
                "lexicon": [],
                "confidence": 0.0,
                "verified": False,
                "error": str(exc),
            }

        choices = data.get("choices") or []
        content = ""
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            content = str(msg.get("content", ""))
        return parse_teacher_json(content)
