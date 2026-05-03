"""
SkillEngine — unified skill lifecycle management node.

Consolidates: SkillCompilerNode + SkillIntelligenceNode + SkillInductionNode
             + FailureAnalyzerNode + RuleExtractorNode

All five nodes manage the skill lifecycle: extraction → induction →
execution governance → failure analysis → rule mining.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.provider_adapter import ProviderLLM
    from hbllm.brain.skill_registry import SkillRegistry
    from hbllm.network.messages import Message

logger = logging.getLogger(__name__)


class SkillEngine(Node):
    """
    Composite node for the complete skill lifecycle.
    """

    def __init__(
        self,
        node_id: str = "skill_engine",
        *,
        llm: ProviderLLM | None = None,
        skill_registry: SkillRegistry | None = None,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.CORE,
            capabilities=[
                "skill_compilation",
                "skill_induction",
                "skill_intelligence",
                "failure_analysis",
                "rule_extraction",
            ],
        )
        self.description = "Unified skill lifecycle (compile → induce → govern → repair → rules)"
        self._llm = llm
        self._skill_registry = skill_registry

        # Sub-nodes
        self._compiler: Any = None
        self._intelligence: Any = None
        self._induction: Any = None
        self._failure_analyzer: Any = None
        self._rule_extractor: Any = None

    async def on_start(self) -> None:
        from hbllm.brain.failure_analyzer_node import FailureAnalyzerNode
        from hbllm.brain.rule_extractor import RuleExtractorNode
        from hbllm.brain.skill_compiler_node import SkillCompilerNode
        from hbllm.brain.skill_induction_node import SkillInductionNode
        from hbllm.brain.skill_intelligence_node import SkillIntelligenceNode

        self._compiler = SkillCompilerNode(
            node_id=f"{self.node_id}.compiler",
            skill_registry=self._skill_registry,
            llm=self._llm,
        )

        self._intelligence = SkillIntelligenceNode(
            node_id=f"{self.node_id}.intelligence",
            skill_registry=self._skill_registry,
        )

        self._induction = SkillInductionNode(
            node_id=f"{self.node_id}.induction",
        )

        self._failure_analyzer = FailureAnalyzerNode(
            node_id=f"{self.node_id}.failure_analyzer",
            llm=self._llm,
        )

        self._rule_extractor = RuleExtractorNode(
            node_id=f"{self.node_id}.rule_extractor",
        )

        bus = self.bus
        for sub in [
            self._compiler,
            self._intelligence,
            self._induction,
            self._failure_analyzer,
            self._rule_extractor,
        ]:
            await sub.start(bus)

        logger.info(
            "SkillEngine started with sub-nodes: compiler, intelligence, "
            "induction, failure_analyzer, rule_extractor"
        )

    async def on_stop(self) -> None:
        for sub in [
            self._compiler,
            self._intelligence,
            self._induction,
            self._failure_analyzer,
            self._rule_extractor,
        ]:
            if sub is not None:
                await sub.stop()

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def health_check(self):
        from hbllm.network.node import HealthStatus, NodeHealth

        subs = [
            self._compiler,
            self._intelligence,
            self._induction,
            self._failure_analyzer,
            self._rule_extractor,
        ]
        sub_healths = []
        for sub in subs:
            if sub is not None:
                sub_healths.append(await sub.health_check())

        statuses = [h.status for h in sub_healths]
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return NodeHealth(
            node_id=self.node_id,
            status=overall,
            uptime_seconds=self.uptime,
            capabilities_available=self.capabilities,
            message=f"Composite: {len(sub_healths)} sub-nodes",
        )

    @property
    def compiler(self):
        return self._compiler

    @property
    def intelligence(self):
        return self._intelligence

    @property
    def induction(self):
        return self._induction

    @property
    def failure_analyzer(self):
        return self._failure_analyzer

    @property
    def rule_extractor(self):
        return self._rule_extractor
