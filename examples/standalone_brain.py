#!/usr/bin/env python3
"""
Standalone Pure-HCIR Core Example
=================================

Demonstrates using the HBLLM Pure-HCIR cognitive runtime without Sentra.

Usage:
    cd core
    ./.venv/bin/python examples/standalone_brain.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from hbllm.hcir.graph import ConceptNode
from hbllm.hcir.kernel.transaction_manager import (
    TransactionManager,
    TransactionOp,
    TransactionOperation,
)
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("standalone")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hbllm_standalone_") as tmp_dir:
        system = build_cognitive_system(
            CognitiveSystemConfig(
                world_id="standalone_demo",
                memory_dir=Path(tmp_dir),
                enable_grounding=False,
                enable_brain_llm=False,
                enable_autonomous_curiosity=False,
            )
        )
        try:
            node = ConceptNode(
                id="concept_hbllm",
                label="HBLLM",
                definition="A human-brain-inspired neuro-symbolic Cognitive OS over HCIR.",
                domain="architecture",
            )
            tx = TransactionManager(system.tiered_workspace.persistent)
            tx.commit_operations(
                [
                    TransactionOperation(
                        op=TransactionOp.UPSERT_NODE,
                        node_id=node.id,
                        node_data=node.model_dump(),
                    )
                ],
                author="standalone_demo",
            )

            result = system.process_dialogue("Explain HBLLM")
            logger.info("Response: %s", (result.response_text or "")[:200])
            logger.info(
                "Workspace persistent nodes: %d",
                system.tiered_workspace.persistent.graph.node_count,
            )
        finally:
            ont_store = system.services.extensions.get("ontology_store")
            if ont_store is not None and hasattr(ont_store, "close"):
                ont_store.close()


if __name__ == "__main__":
    main()
