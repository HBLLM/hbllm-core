"""
Legacy Node Creation — extracted from BrainFactory._build_brain.

Handles creating and registering the 27+ individual cognitive nodes
used in the legacy (non-composite) brain path.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from hbllm.brain.policy_engine import PolicyEngine
from hbllm.brain.provider_adapter import ProviderLLM
from hbllm.brain.skill_registry import SkillRegistry
from hbllm.brain.wiring.snn import wire_comprehension_stream, wire_expression_stream
from hbllm.brain.wiring.subsystems import _register_node
from hbllm.network.bus import MessageBus
from hbllm.network.node import Node
from hbllm.network.registry import ServiceRegistry
from hbllm.serving.provider import LLMProvider

logger = logging.getLogger(__name__)


def create_legacy_nodes(
    llm: ProviderLLM,
    llm_provider: LLMProvider,
    cfg: Any,  # BrainConfig
    policy_engine: PolicyEngine | None,
    domain_registry: Any,
    skill_registry: SkillRegistry,
    *,
    dual_router: Any | None = None,
) -> list[Node]:
    """Create and return the list of legacy cognitive nodes.

    Does NOT start them on the bus — caller must iterate and start.
    """
    from hbllm.brain.collective_node import CollectiveNode
    from hbllm.brain.critic_node import CriticNode
    from hbllm.brain.curiosity_node import CuriosityNode
    from hbllm.brain.decision_node import DecisionNode
    from hbllm.brain.experience_node import ExperienceNode
    from hbllm.brain.identity_node import IdentityNode
    from hbllm.brain.learner_node import LearnerNode
    from hbllm.brain.meta_node import MetaReasoningNode
    from hbllm.brain.planner_node import PlannerNode
    from hbllm.brain.router_node import RouterNode
    from hbllm.brain.rule_extractor import RuleExtractorNode
    from hbllm.brain.sentinel_node import SentinelNode
    from hbllm.brain.sleep_node import SleepCycleNode
    from hbllm.brain.workspace_node import WorkspaceNode
    from hbllm.brain.world_model_node import WorldModelNode
    from hbllm.memory.memory_node import MemoryNode

    # Router + Comprehension Stream
    router_node = RouterNode(node_id="router", llm=llm, domain_registry=domain_registry)
    router_node._centroids_path = Path(cfg.data_dir) / "router_centroids.json"

    if cfg.inject_comprehension:
        wire_comprehension_stream(router_node, domain_registry)

    # Decision node
    decision_node = DecisionNode(node_id="decision", llm=llm, policy_engine=policy_engine)

    # Core cognitive pipeline nodes
    nodes: list[Node] = [
        router_node,
        PlannerNode(
            node_id="planner",
            branch_factor=cfg.planner_branch_factor,
            max_depth=cfg.planner_max_depth,
            policy_engine=policy_engine,
            llm=llm,
        ),
        CriticNode(node_id="critic", llm=llm),
        decision_node,
        WorkspaceNode(node_id="workspace"),
        MemoryNode(node_id="memory", db_path=Path(cfg.data_dir) / "working_memory.db"),
        ExperienceNode(node_id="experience", llm=llm),
        MetaReasoningNode(node_id="meta"),
        RuleExtractorNode(node_id="rule_extractor"),
        CuriosityNode(node_id="curiosity"),
        CollectiveNode(node_id="collective", skill_registry=skill_registry),
        LearnerNode(node_id="learner"),
        WorldModelNode(node_id="world_model"),
        SleepCycleNode(node_id="sleep", llm=llm),
    ]

    # Wire expression-side Cognitive Stream (Layer 5)
    if cfg.inject_comprehension:
        wire_expression_stream(decision_node, router_node, llm, dual_router=dual_router)

    # Wire memory search into ComprehensionStream for per-concept retrieval
    _wire_memory_search(router_node, nodes)

    # Browser Node
    if cfg.inject_browser:
        from hbllm.actions.browser_node import BrowserNode

        nodes.append(BrowserNode(node_id="browser"))
        logger.info("BrowserNode wired (web search & scrape)")

    # Execution Node
    if cfg.inject_execution:
        from hbllm.actions.execution_node import ExecutionNode

        nodes.append(ExecutionNode(node_id="execution"))
        logger.info("ExecutionNode wired (sandboxed python execution)")

    # Sentinel (governance)
    sentinel_node = None
    if cfg.inject_sentinel and policy_engine:
        sentinel_node = SentinelNode(
            node_id="sentinel",
            policy_engine=policy_engine,
        )
        nodes.append(sentinel_node)

    # Identity
    if cfg.inject_identity:
        nodes.append(IdentityNode(node_id="identity"))

    # Host shell execution node
    if cfg.inject_shell:
        from hbllm.actions.shell_node import HostShellNode

        shell_node = HostShellNode(
            node_id="shell_executor",
            workspace_dir=None,
            require_manual_approval=cfg.require_shell_approval,
            policy_engine=policy_engine,
        )
        nodes.append(shell_node)
        logger.info("HostShellNode wired (manual approval=%s)", cfg.require_shell_approval)

    # Perception nodes
    if cfg.inject_perception:
        from hbllm.perception.audio_in_node import AudioInputNode
        from hbllm.perception.audio_out_node import AudioOutputNode
        from hbllm.perception.vision_node import VisionNode

        nodes.extend(
            [
                AudioInputNode(node_id="audio_in"),
                AudioOutputNode(node_id="audio_out"),
                VisionNode(node_id="vision"),
            ]
        )

    # Reasoning nodes
    if cfg.inject_fuzzy_logic:
        from hbllm.actions.fuzzy_node import FuzzyNode

        nodes.append(FuzzyNode(node_id="fuzzy", llm=llm))
        logger.info("FuzzyNode wired (scikit-fuzzy reasoning)")

    if cfg.inject_symbolic_logic:
        from hbllm.actions.logic_node import LogicNode

        nodes.append(LogicNode(node_id="logic", llm=llm))
        logger.info("LogicNode wired (Z3 theorem prover)")

    # Domain module nodes
    _create_domain_modules(nodes, llm, llm_provider)

    return nodes


def _wire_memory_search(router_node: Any, nodes: list[Node]) -> None:
    """Wire semantic memory search into the ComprehensionStream."""
    comp_stream = getattr(router_node, "comprehension_stream", None)
    if comp_stream is None or comp_stream.memory_search_fn is not None:
        return

    memory_node = None
    for n in nodes:
        if getattr(n, "node_id", None) == "memory":
            memory_node = n
            break

    if memory_node is not None and hasattr(memory_node, "semantic_db"):
        import asyncio

        async def _mem_search(text: str, top_k: int = 3, tenant_id: str = "default") -> list:
            return await asyncio.to_thread(
                memory_node.semantic_db.search,
                text,
                top_k=top_k,
                tenant_id=tenant_id,
            )

        comp_stream.memory_search_fn = _mem_search
        logger.info("ComprehensionStream memory_search_fn wired to MemoryNode")


def _create_domain_modules(
    nodes: list[Node],
    llm: ProviderLLM,
    llm_provider: LLMProvider,
) -> None:
    """Create DomainModuleNode instances for general/coding/math."""
    from hbllm.modules.base_module import DomainModuleNode

    if (
        type(llm_provider).__name__ == "LocalProvider"
        and getattr(llm_provider, "_model", None) is not None
    ):
        model = llm_provider._model
        tokenizer = llm_provider._tokenizer
        for domain in ["general", "coding", "math"]:
            nodes.append(
                DomainModuleNode(
                    node_id=f"domain_{domain}",
                    domain_name=domain,
                    model=model,
                    tokenizer=tokenizer,
                    lora_state_dict=None,
                )
            )
    else:
        for domain in ["general", "coding", "math"]:
            nodes.append(
                DomainModuleNode(
                    node_id=f"domain_{domain}",
                    domain_name=domain,
                    llm=llm,
                )
            )
