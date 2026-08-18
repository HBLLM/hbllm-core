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
from hbllm.serving.studio.hcir_epistemics import router as hcir_epistemics_router
from hbllm.serving.studio.knowledge_graph import router as knowledge_graph_router
from hbllm.serving.studio.learning import router as learning_router
from hbllm.serving.studio.memory import router as memory_router
from hbllm.serving.studio.perception import router as perception_router
from hbllm.serving.studio.persona import router as persona_router
from hbllm.serving.studio.plugins import router as plugins_router
from hbllm.serving.studio.rbac import router as rbac_router
from hbllm.serving.studio.snn import router as snn_router
from hbllm.serving.studio.voice import router as voice_router

# Aggregate router
router = APIRouter()

# New modular sub-routers (take priority — they override stale legacy equivalents)
router.include_router(emotion_router, tags=["studio:emotion"])
router.include_router(persona_router, tags=["studio:persona"])
router.include_router(perception_router, tags=["studio:perception"])
router.include_router(cognitive_router, tags=["studio:cognitive"])
router.include_router(hcir_epistemics_router, tags=["studio:hcir_epistemics"])
router.include_router(snn_router, tags=["studio:snn"])
router.include_router(memory_router, tags=["studio:memory"])
router.include_router(knowledge_graph_router, tags=["studio:knowledge_graph"])
router.include_router(learning_router, tags=["studio:learning"])
router.include_router(voice_router, tags=["studio:voice"])
router.include_router(plugins_router, tags=["studio:plugins"])
router.include_router(rbac_router, tags=["studio:rbac"])

# Legacy endpoints (remaining: swarm, temporal, synapsis, stats, memory-summary,
# knowledge-summary, lora — general dashboard aggregation endpoints)
router.include_router(legacy_router, tags=["studio:legacy"])
