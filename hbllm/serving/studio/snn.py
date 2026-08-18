"""
Studio SNN (Spiking Neural Network) Endpoints.

Exposes the cognitive SNN subsystem status including priming potentials,
attention fatigue, reflex rules, expression/comprehension streams,
and STDP plasticity diagnostics.

Extracted from ``_legacy.py`` — see Work Stream 1.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from hbllm.serving.studio.helpers import get_brain, get_node_map

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/snn/status")
async def get_snn_status():
    from hbllm.network.metrics import MetricsCollector

    collector = MetricsCollector.get_instance()

    # Extract priming category potentials
    categories = ["physics", "math", "coding", "finance", "personal", "general"]

    # Try to extract exact real-time potentials from MemoryNode primer if available
    brain = get_brain()
    memory_node = None
    if brain:
        node_map = get_node_map()
        memory_node = node_map.get("MemoryNode")

    # If memory node is active, discover all dynamic clusters/categories
    all_cats = list(categories)
    if memory_node and hasattr(memory_node, "primer"):
        for cat in memory_node.primer.categories.keys():
            if cat not in all_cats:
                all_cats.append(cat)

    priming_potentials = {}
    for cat in all_cats:
        neuron_id = f"priming_{cat}"
        # Fall back to metrics collector value
        pot = collector._mem_gauges.get(f"snn_potential:{neuron_id}", 0.0)
        threshold = 1.0

        label = cat
        # Try to get descriptive cluster label if it's a dynamic cluster
        if cat.startswith("cluster_") and memory_node and hasattr(memory_node, "semantic_db"):
            try:
                cluster_id = int(cat.split("_")[1])
                label = memory_node.semantic_db.cluster_manager.get_cluster_label(
                    cluster_id, memory_node.semantic_db.documents
                )
            except (ValueError, IndexError):
                pass

        # If memory node is active, get precise current values
        if memory_node and hasattr(memory_node, "primer"):
            acc = memory_node.primer.categories.get(cat)
            if acc:
                pot = acc.get_potential()
                threshold = acc.neuron.config.threshold

        priming_potentials[cat] = {
            "label": label,
            "potential": pot,
            "threshold": threshold,
            "history": collector.get_snn_history(neuron_id),
        }

    # Extract attention fatigue potential
    attn_pot = collector._mem_gauges.get("snn_potential:human_attention_fatigue", 0.0)
    attn_threshold = 0.8
    refractory_time = 0.0

    attention_fatigue = {
        "potential": attn_pot,
        "threshold": attn_threshold,
        "refractory_time_remaining": refractory_time,
        "history": collector.get_snn_history("human_attention_fatigue"),
    }

    # Extract any reflex rules from metrics collector
    reflexes = {}
    for key in list(collector._mem_gauges.keys()):
        if key.startswith("snn_potential:reflex_"):
            neuron_id = key.replace("snn_potential:", "")
            reflexes[neuron_id] = {
                "potential": collector._mem_gauges[key],
                "threshold": 1.0,
                "history": collector.get_snn_history(neuron_id),
            }

    return {
        "status": "success",
        "priming_categories": priming_potentials,
        "attention_fatigue": attention_fatigue,
        "reflex_rules": reflexes,
    }


@router.post("/api/snn/stimulate")
async def stimulate_snn_neuron(request: Request):
    body = await request.json()
    category = body.get("category")
    charge = float(body.get("charge", 0.5))

    if not category:
        raise HTTPException(status_code=400, detail="Category is required")

    brain = get_brain()
    if brain:
        node_map = get_node_map()
        memory_node = node_map.get("MemoryNode")
        if memory_node and hasattr(memory_node, "primer"):
            memory_node.primer.stimulate_category(category, charge)
            # Force update metrics collector immediately
            pot = memory_node.primer.categories[category].get_potential()
            from hbllm.network.metrics import MetricsCollector

            MetricsCollector.get_instance().record_snn_potential(f"priming_{category}", pot)
            return {
                "status": "success",
                "message": f"Stimulated category {category} with {charge} charge.",
            }

    # Fallback if MemoryNode is not loaded
    from hbllm.network.metrics import MetricsCollector

    collector = MetricsCollector.get_instance()
    neuron_id = f"priming_{category}"
    cur_pot = collector._mem_gauges.get(f"snn_potential:{neuron_id}", 0.0)
    new_pot = min(1.0, cur_pot + charge)
    collector.record_snn_potential(neuron_id, new_pot)
    return {
        "status": "success",
        "message": f"Stimulated fallback metrics for category {category}.",
    }


@router.post("/api/snn/replay")
async def replay_cognitive_search(request: Request):
    body = await request.json()
    query = body.get("query")
    priming_state = body.get("priming_state", {})
    tenant_id = getattr(request.state, "tenant_id", "default")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    brain = get_brain()
    memory_node = None
    if brain:
        node_map = get_node_map()
        memory_node = node_map.get("MemoryNode")

    if memory_node and hasattr(memory_node, "semantic_db"):
        sem_db = memory_node.semantic_db
    else:
        from hbllm.memory.semantic import SemanticMemory

        sem_db = SemanticMemory()
        if not sem_db.documents:
            sem_db.store(
                "Winner content with physics mechanics and quantum physics equations.",
                metadata={"category": "physics", "usefulness_score": 0.8},
                tenant_id=tenant_id,
            )
            sem_db.store(
                "Runner content with software coding and python algorithms.",
                metadata={"category": "coding", "usefulness_score": 0.5},
                tenant_id=tenant_id,
            )

    # 1. Unprimed Search (baseline)
    unprimed_env = sem_db.search(
        query=query, top_k=5, tenant_id=tenant_id, priming_boosts=None, explain=True
    )

    # 2. Primed Search (replayed state)
    primed_env = sem_db.search(
        query=query, top_k=5, tenant_id=tenant_id, priming_boosts=priming_state, explain=True
    )

    unprimed_results = unprimed_env["results"] if isinstance(unprimed_env, dict) else unprimed_env
    primed_results = primed_env["results"] if isinstance(primed_env, dict) else primed_env

    # Compute ranking differentials
    differentials = sem_db.get_ranking_differential(primed_results)

    return {
        "status": "success",
        "unprimed": unprimed_results,
        "primed": primed_results,
        "differentials": differentials,
    }


@router.get("/api/snn/expression")
async def get_snn_expression_status():
    """Return ExpressionStream state: rendering tiers, content plans, PRM scores."""
    node_map = get_node_map()

    # Try to find ExpressionStream from DecisionNode
    decision_node = node_map.get("DecisionNode")
    expression_stream = getattr(decision_node, "expression_stream", None) if decision_node else None

    result = {
        "status": "active" if expression_stream else "not_loaded",
        "rendering_tiers": {
            "broca": {
                "label": "Broca (v4)",
                "tokens": "~80",
                "description": "SNN decides content, LLM is grammar-only",
            },
            "shallow": {
                "label": "Shallow (v3)",
                "tokens": "~300",
                "description": "SNN reasons, LLM renders text",
            },
            "deep": {
                "label": "Deep (v1-v2)",
                "tokens": "~600",
                "description": "LLM handles everything",
            },
        },
        "last_render": None,
        "content_plans": [],
        "prm_scores": [],
        "token_savings": [],
    }

    if expression_stream:
        # Extract last render info
        last_meta = getattr(expression_stream, "_last_render_metadata", None)
        if last_meta and isinstance(last_meta, dict):
            result["last_render"] = {
                "tier": last_meta.get("tier", "deep"),
                "token_count": last_meta.get("token_count", 0),
                "thought_count": last_meta.get("thought_count", 0),
                "prm_score": last_meta.get("prm_score", 0.0),
            }

        # Extract content plans from planner
        planner = getattr(expression_stream, "_content_planner", None)
        if planner:
            last_plans = getattr(planner, "_last_plans", [])
            for plan in last_plans[-5:]:
                result["content_plans"].append(
                    {
                        "content_type": getattr(plan, "content_type", "unknown"),
                        "key_points": getattr(plan, "key_points", []),
                        "emphasis": getattr(plan, "emphasis", 0.5),
                    }
                )

        # Extract PRM reward history
        prm = getattr(expression_stream, "_trained_prm", None)
        if prm:
            history = getattr(prm, "_score_history", [])
            for entry in history[-20:]:
                result["prm_scores"].append(
                    {
                        "score": entry.get("score", 0.0)
                        if isinstance(entry, dict)
                        else float(entry),
                        "timestamp": entry.get("timestamp", 0.0)
                        if isinstance(entry, dict)
                        else 0.0,
                    }
                )

    return result


@router.get("/api/snn/comprehension")
async def get_snn_comprehension_status():
    """Return ComprehensionStream state: concepts, associations, causal chains."""
    node_map = get_node_map()

    decision_node = node_map.get("DecisionNode")
    comp_stream = None
    if decision_node:
        expr_stream = getattr(decision_node, "expression_stream", None)
        if expr_stream:
            comp_stream = getattr(expr_stream, "_comprehension_stream", None)

    result = {
        "status": "active" if comp_stream else "not_loaded",
        "channels": [
            {"name": "entity", "description": "Named entities and key nouns"},
            {"name": "clause", "description": "Clause boundary detection"},
            {"name": "discourse", "description": "Discourse markers and connectives"},
            {"name": "surprise", "description": "Unexpected or novel content"},
            {"name": "constraint", "description": "Requirements and conditions"},
        ],
        "last_concepts": [],
        "last_associations": [],
        "last_causal_chains": [],
    }

    if comp_stream:
        # Extract last understanding state
        last_state = getattr(comp_stream, "_last_state", None)
        if last_state:
            for concept in getattr(last_state, "concepts", [])[:10]:
                result["last_concepts"].append(
                    {
                        "text": getattr(concept, "text", ""),
                        "domain_activation": getattr(concept, "domain_activation", {}),
                        "channel_metadata": getattr(concept, "channel_metadata", {}),
                    }
                )
            for assoc in getattr(last_state, "associations", [])[:10]:
                result["last_associations"].append(
                    {
                        "type": getattr(assoc, "association_type", ""),
                        "source_idx": getattr(assoc, "source_idx", 0),
                        "target_idx": getattr(assoc, "target_idx", 0),
                        "confidence": getattr(assoc, "confidence", 0.0),
                    }
                )
            for chain in getattr(last_state, "causal_chains", [])[:5]:
                result["last_causal_chains"].append(
                    {
                        "depth": getattr(chain, "depth", 0),
                        "probability": getattr(chain, "combined_probability", 0.0),
                        "snn_confidence": getattr(chain, "snn_confidence", 0.0),
                    }
                )

    return result


@router.get("/api/snn/plasticity")
async def get_snn_plasticity_status():
    """Return STDP plasticity stats: weight summaries, training history."""
    node_map = get_node_map()

    decision_node = node_map.get("DecisionNode")
    trained_prm = None
    if decision_node:
        expr_stream = getattr(decision_node, "expression_stream", None)
        if expr_stream:
            trained_prm = getattr(expr_stream, "_trained_prm", None)

    result = {
        "status": "active" if trained_prm else "not_loaded",
        "stdp_rule": {
            "learning_rate": 0.01,
            "time_constant": 20.0,
            "description": "Spike-Timing-Dependent Plasticity: strengthens causal (pre→post) connections",
        },
        "networks": [],
        "training_stats": None,
    }

    if trained_prm:
        # PRM network info
        prm_net = getattr(trained_prm, "_network", None)
        if prm_net:
            layers = getattr(prm_net, "layer_names", [])
            result["networks"].append(
                {
                    "name": "TrainedPRM",
                    "architecture": "6→8→4→2",
                    "layers": list(layers),
                    "step_count": getattr(prm_net, "step_count", 0),
                }
            )

        # Training collector stats
        collector = getattr(trained_prm, "_collector", None)
        if collector:
            examples = getattr(collector, "_examples", [])
            result["training_stats"] = {
                "total_examples": len(examples),
                "last_accuracy": getattr(collector, "_last_accuracy", None),
                "last_weight_delta": getattr(collector, "_last_weight_delta", None),
                "batch_threshold": 20,
                "ready_for_batch": len(examples) >= 20,
            }

    # Also check ContentPlanner network
    if decision_node:
        expr_stream = getattr(decision_node, "expression_stream", None)
        if expr_stream:
            content_planner = getattr(expr_stream, "_content_planner", None)
            if content_planner:
                cp_net = getattr(content_planner, "_network", None)
                if cp_net:
                    result["networks"].append(
                        {
                            "name": "ContentPlanner",
                            "architecture": "8→12→6→3",
                            "layers": list(getattr(cp_net, "layer_names", [])),
                            "step_count": getattr(cp_net, "step_count", 0),
                        }
                    )

    return result
