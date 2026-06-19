"""
Studio Sub-Router Package.

Aggregates all studio endpoint modules into a single router that can be
mounted on the FastAPI app. New feature modules are imported here;
legacy endpoints from the original monolithic studio.py are preserved
via the ``_legacy`` import.

Usage in api.py::

    from hbllm.serving.studio import router as studio_router
    app.include_router(studio_router)
"""

from __future__ import annotations

from fastapi import APIRouter

# Import sub-routers
from hbllm.serving.studio._legacy import router as legacy_router
from hbllm.serving.studio.cognitive import router as cognitive_router
from hbllm.serving.studio.emotion import router as emotion_router
from hbllm.serving.studio.perception import router as perception_router
from hbllm.serving.studio.persona import router as persona_router

# Aggregate router
router = APIRouter()

# New modular sub-routers (take priority — they override stale legacy equivalents)
router.include_router(emotion_router, tags=["studio:emotion"])
router.include_router(persona_router, tags=["studio:persona"])
router.include_router(perception_router, tags=["studio:perception"])
router.include_router(cognitive_router, tags=["studio:cognitive"])

# Legacy endpoints (everything else: SNN, memory, learning, voice, plugins, etc.)
router.include_router(legacy_router, tags=["studio:legacy"])
