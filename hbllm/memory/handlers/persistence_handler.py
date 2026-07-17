"""Persistence Handler for MemoryNode.

Manages clean shutdown, awaiting in-flight tasks, and persisting Semantic Memory
and Knowledge Graph instances to disk.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class PersistenceHandler:
    """Handles the persistence and shutdown of memory data sources and engines."""

    def __init__(self, node: Any) -> None:
        self.node = node

    async def shutdown(self) -> None:
        """Persist in-memory data to disk and clean up."""
        # Await any in-flight background storage tasks before persisting
        all_tasks = self.node._pending_tasks.union(self.node._improvement_tasks)
        if all_tasks:
            logger.info(
                "Awaiting %d pending background tasks before shutdown (timeout=5s)", len(all_tasks)
            )
            try:
                await asyncio.wait_for(
                    asyncio.gather(*all_tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for background tasks during MemoryNode stop")
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
                logger.error("Error during background task cleanup: %s", e)
            finally:
                self.node._pending_tasks.clear()
                self.node._improvement_tasks.clear()

        logger.info("Stopping MemoryNode — persisting semantic memory and knowledge graph")
        try:
            self.node.semantic_db.save_to_disk(self.node._persistence_dir / "semantic")
            if str(self.node.db.db_path) != ":memory:":
                for tid, kg in self.node._knowledge_graphs.items():
                    if tid == "default":
                        kg.save_to_disk(self.node._persistence_dir / "knowledge_graph.json")
                    else:
                        kg.save_to_disk(self.node._persistence_dir / f"knowledge_graph_{tid}.json")
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, ConnectionError) as e:
            logger.error("Failed to persist memory to disk: %s", e)

        # Close connections and release locks
        if hasattr(self.node.semantic_db, "close"):
            self.node.semantic_db.close()
        if hasattr(self.node.db, "close"):
            await self.node.db.close()
        if hasattr(self.node.procedural_db, "close"):
            await self.node.procedural_db.close()
        if hasattr(self.node.value_db, "close"):
            await self.node.value_db.close()
