#!/usr/bin/env python3
"""
Standalone HBLLM Core Example
==============================

Demonstrates using the HBLLM core library without Sentra. This proves
the decoupled architecture works independently.

Requires:
  - OPENAI_API_KEY env var (or replace with another provider)

Usage:
    cd core
    ./venv/bin/python examples/standalone_brain.py
"""

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("standalone")


async def main() -> None:
    from hbllm.brain.factory import BrainConfig, BrainFactory

    # ── Configuration ─────────────────────────────────────────────────
    # All subsystems are opt-in via inject_* flags.
    # Defaults are sensible — this config only overrides what's needed.
    config = BrainConfig(
        data_dir="/tmp/hbllm_standalone",
        system_prompt="You are a helpful AI assistant powered by HBLLM core.",
        inject_knowledge=True,  # Enables KnowledgeBase
        inject_persistence=True,  # Enables BrainState (SQLite KV)
        inject_awareness=True,  # Enables CognitiveAwareness (bus monitoring)
        inject_plugins=True,  # Enables PluginManager
        inject_evaluation=True,  # Enables EvaluationNode (quality feedback)
        inject_reflection=True,  # Enables ReflectionNode (batch learning)
    )

    # ── Create Brain ──────────────────────────────────────────────────
    # BrainFactory wires everything: nodes, bus, subsystems.
    brain = await BrainFactory.create(
        provider="openai/gpt-4o-mini",  # Or "anthropic", "ollama/llama3", etc.
        config=config,
    )

    logger.info("Brain created with %d nodes", len(brain.nodes))
    logger.info("Node types: %s", [type(n).__name__ for n in brain.nodes])

    # ── Subsystem Access ──────────────────────────────────────────────
    # All subsystems are available as brain attributes.
    logger.info("KnowledgeBase: %s", brain.knowledge_base)
    logger.info("BrainState: %s", brain.state)
    logger.info("Awareness: %s", brain.awareness)
    logger.info("PluginManager: %s", brain.plugin_manager)

    # ── Use BrainState (KV persistence) ───────────────────────────────
    if brain.state:
        brain.state.save("demo_key", "hello from standalone!")
        value = brain.state.load("demo_key")
        logger.info("BrainState: saved and loaded '%s'", value)

    # ── Use KnowledgeBase ─────────────────────────────────────────────
    if brain.knowledge_base:
        stats = brain.knowledge_base.get_stats()
        logger.info("KnowledgeBase stats: %s", stats)

    # ── Query the Brain ───────────────────────────────────────────────
    response = await brain.query("What is HBLLM?")
    logger.info("Response: %s", response.content[:200])

    # ── Check Awareness ───────────────────────────────────────────────
    if brain.awareness:
        snap = brain.awareness.snapshot()
        logger.info(
            "Awareness snapshot: queries=%d, sessions=%d", snap.queries_total, snap.active_sessions
        )

    # ── Shutdown ──────────────────────────────────────────────────────
    await brain.stop()
    logger.info("Brain stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
