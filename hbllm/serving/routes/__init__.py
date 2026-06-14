"""
HBLLM Serving Routes — modular API endpoint organization.

Modules:
  - health:  /health, /health/live, /health/ready, /metrics, /routing/stats
  - memory:  /v1/memory/*, /v1/sync/*, /v1/feedback, /v1/knowledge/*, /v1/rules
  - chat:    /v1/chat, /v1/chat/stream, /v1/chat/ws (remains in api.py)
  - completions: /v1/chat/completions (OpenAI compat, remains in api.py)
"""

from hbllm.serving.routes.health import router as health_router
from hbllm.serving.routes.memory import router as memory_router

__all__ = ["health_router", "memory_router"]
