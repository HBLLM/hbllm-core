"""
Shared helpers for Studio sub-routers.

Eliminates the repeated ``_state.get("brain"); _get_node_map(brain)``
boilerplate from every endpoint.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import HTTPException

from hbllm.serving.state import _get_node_map, _state


def get_brain() -> Any:
    """Return the brain instance or None."""
    return _state.get("brain")


def get_bus() -> Any:
    """Return the message bus from the brain, or None."""
    brain = get_brain()
    return getattr(brain, "bus", None)


def require_bus() -> Any:
    """Return the message bus or raise 503."""
    bus = get_bus()
    if not bus:
        raise HTTPException(status_code=503, detail="Brain pipeline not initialized")
    return bus


def get_node_map() -> dict[str, Any]:
    """Return the node map from the current brain."""
    brain = get_brain()
    return _get_node_map(brain)


def get_node(class_name: str) -> Any | None:
    """Lookup a node by class name from the brain."""
    return get_node_map().get(class_name)


def require_node(class_name: str) -> Any:
    """Lookup a node by class name or raise 503."""
    node = get_node(class_name)
    if not node:
        raise HTTPException(
            status_code=503,
            detail=f"{class_name} not loaded",
        )
    return node


def get_data_dir() -> str:
    """Return HBLLM_DATA_DIR (default: 'data')."""
    return os.environ.get("HBLLM_DATA_DIR", "data")


def get_tenant_id(request: Any) -> str:
    """Extract tenant_id from request state."""
    return getattr(request.state, "tenant_id", "default")


def get_user_id(request: Any) -> str:
    """Extract user_id from request state."""
    return getattr(request.state, "user_id", "default")
