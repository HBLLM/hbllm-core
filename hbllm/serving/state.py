import logging
from typing import Any

logger = logging.getLogger(__name__)

# Shared global state across API routers
_state: dict[str, Any] = {}


def _get_node_map(brain) -> dict[str, Any]:
    """Helper to build a flat map of node class name -> node instance.
    Handles composite nodes by traversing their sub-nodes, and maps direct brain attributes.
    """
    node_map: dict[str, Any] = {}
    if not brain:
        return node_map

    # 1. Walk top-level nodes in brain.nodes
    nodes = getattr(brain, "nodes", [])
    for node in nodes:
        cls_name = type(node).__name__
        node_map[cls_name] = node

        # 2. Extract nested sub-nodes from composites
        for attr_name in [
            "_memory",
            "_experience",
            "_sleep",
            "_meta",
            "_evaluation",
            "_reflection",
            "_curiosity",
            "_collective",
            "_identity",
            "_sentinel",
            "_policy_engine",
            "_confidence_estimator",
            "_router",
            "_planner",
            "_critic",
            "_decision",
            "_revision",
            "_compiler",
            "_intelligence",
            "_induction",
            "_failure_analyzer",
            "_rule_extractor",
            "_workspace",
            "_attention",
            "_load_manager",
            "_scheduler",
            "_learner",
            "_world_model",
            "_process_reward",
            "_spawner",
        ]:
            val = getattr(node, attr_name, None)
            if val is not None:
                sub_cls_name = type(val).__name__
                node_map[sub_cls_name] = val

        for prop_name in [
            "memory",
            "experience",
            "sleep",
            "meta",
            "evaluation",
            "reflection",
            "curiosity",
            "collective",
            "identity",
            "sentinel",
            "policy_engine",
            "confidence_estimator",
            "router",
            "planner",
            "critic",
            "decision",
            "revision",
            "compiler",
            "intelligence",
            "induction",
            "failure_analyzer",
            "rule_extractor",
            "workspace",
            "attention",
            "load_manager",
            "scheduler",
            "learner",
            "world_model",
            "process_reward",
        ]:
            try:
                val = getattr(node, prop_name, None)
                if val is not None:
                    sub_cls_name = type(val).__name__
                    node_map[sub_cls_name] = val
            except Exception as e:
                logger.debug(
                    "[State] 3. Add direct brain attributes (in case some are not in brain.nodes): %s",
                    e,
                )
    for attr_name in [
        "cognitive_metrics",
        "self_model",
        "skill_registry",
        "goal_manager",
        "evaluation_node",
        "attention_manager",
        "load_manager",
        "reflection_node",
        "skill_compiler_node",
        "skill_intelligence_node",
        "failure_analyzer_node",
        "scheduler_node",
        "policy_engine",
        "sentinel",
        "revision_node",
        "tool_memory",
    ]:
        val = getattr(brain, attr_name, None)
        if val is not None:
            cls_name = type(val).__name__
            node_map[cls_name] = val

            # Alias common node class name lookups
            if attr_name == "evaluation_node":
                node_map["EvaluationNode"] = val
            elif attr_name == "reflection_node":
                node_map["ReflectionNode"] = val
            elif attr_name == "skill_compiler_node":
                node_map["SkillCompilerNode"] = val
            elif attr_name == "attention_manager":
                node_map["AttentionManager"] = val
            elif attr_name == "load_manager":
                node_map["LoadManager"] = val
            elif attr_name == "scheduler_node":
                node_map["SchedulerNode"] = val

    return node_map
