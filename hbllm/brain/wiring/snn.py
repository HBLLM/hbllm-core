"""
SNN Wiring — Comprehension & Expression stream wiring extracted from factory.py.

These functions create and wire the SNN (Spiking Neural Network) cognitive
streams into the brain's node graph:

  - wire_comprehension_stream: ComprehensionStream → RouterNode
  - wire_expression_stream:    ExpressionStream → DecisionNode
"""

from __future__ import annotations

import logging
from typing import Any

from hbllm.brain.snn.neuromodulation import NeuromodulationEngine

logger = logging.getLogger(__name__)


def wire_comprehension_stream(
    router_node: Any,
    domain_registry: Any,
    neuromodulator: NeuromodulationEngine | None = None,
) -> None:
    """Wire the Cognitive Stream comprehension pipeline into a RouterNode.

    Creates a ComprehensionStream with:
      - LexicalBuffer for subword noise absorption
      - ComprehensionEnsemble (5-channel SNN)
      - Encoder bound to router_node._encode_text (reuses ONNX session)
      - Domain centroids shared from router_node (same reference)

    The comprehension stream is stored on router_node.comprehension_stream
    and will be invoked during handle_message() for queries >= 5 words.
    """
    try:
        from hbllm.brain.snn.comprehension import (
            ComprehensionEnsemble,
            ComprehensionStream,
            LexicalBuffer,
            populate_from_registry,
        )

        # Populate technical terms from domain registry centroid texts
        if domain_registry is not None:
            populate_from_registry(domain_registry)

        lexical_buffer = LexicalBuffer()

        # Try to create STDP plasticity for the ensemble
        plastic_weights = None
        try:
            from hbllm.brain.snn.plasticity import PlasticWeightMatrix, STDPRule

            stdp_rule = STDPRule(
                learning_rate=0.01,
                time_constant=0.5,
                w_min=0.0,
                w_max=2.0,
            )
            # Create a temporary ensemble to get static weights
            _tmp = ComprehensionEnsemble(domain="general")
            plastic_weights = PlasticWeightMatrix(
                _tmp._signal_weights,
                stdp_rule,
                neuromodulator=neuromodulator,
            )
            logger.info("STDP plasticity enabled for ComprehensionEnsemble")
        except Exception as e:
            logger.debug("STDP plasticity not available (non-fatal): %s", e)

        ensemble = ComprehensionEnsemble(domain="general", plastic_weights=plastic_weights)

        # Try to create AssociationLayer for concept relationship detection
        association_layer = None
        try:
            from hbllm.brain.snn.reasoning.association import AssociationLayer

            assoc_stdp = None
            try:
                from hbllm.brain.snn.plasticity import STDPRule as _STDPRule

                assoc_stdp = _STDPRule(
                    learning_rate=0.01,
                    time_constant=0.5,
                    w_min=0.0,
                    w_max=2.0,
                )
            except Exception as e:
                logger.debug("[SNN] Node processing skipped: %s", e)

            association_layer = AssociationLayer(stdp_rule=assoc_stdp)
            logger.info("AssociationLayer wired to ComprehensionStream")
        except Exception as e:
            logger.debug("AssociationLayer not available (non-fatal): %s", e)

        # Try to create CausalReasoner for multi-hop causal reasoning
        causal_reasoner = None
        try:
            from hbllm.brain.causality.causal_graph import CausalGraph
            from hbllm.brain.snn.reasoning.reasoner import CausalReasoner
            from hbllm.brain.snn.reasoning.reasoning_network import ReasoningNetwork

            # Use data_dir from router_node if available, else default
            data_dir = getattr(router_node, "data_dir", "data")
            causal_graph = CausalGraph(data_dir=data_dir)
            reasoning_net = ReasoningNetwork(stdp_rule=assoc_stdp if assoc_stdp else None)
            causal_reasoner = CausalReasoner(
                causal_graph=causal_graph,
                reasoning_network=reasoning_net,
                max_depth=3,
                min_probability=0.3,
                top_k=5,
            )
            logger.info("CausalReasoner wired to ComprehensionStream")
        except Exception as e:
            logger.debug("CausalReasoner not available (non-fatal): %s", e)

        stream = ComprehensionStream(
            ensemble=ensemble,
            lexical_buffer=lexical_buffer,
            encoder=router_node._encode_text,
            domain_centroids=router_node.domain_centroids,
            memory_search_fn=None,  # No memory coupling in v1; wired later if needed
            association_layer=association_layer,
            causal_reasoner=causal_reasoner,
        )
        router_node.comprehension_stream = stream
        logger.info("ComprehensionStream wired to RouterNode")
    except Exception as e:
        logger.warning("Failed to wire ComprehensionStream (non-fatal): %s", e)


def wire_expression_stream(
    decision_node: Any,
    router_node: Any | None = None,
    llm: Any | None = None,
    dual_router: Any | None = None,
    neuromodulator: NeuromodulationEngine | None = None,
) -> None:
    """Wire the expression-side Cognitive Stream into a DecisionNode.

    Creates an ExpressionStream with:
      - ThoughtPlanner (symbolic outline from UnderstandingState)
      - ThoughtController (SNN-gated thought sequencer)
      - RewardEvaluator (per-fragment scoring)

    The expression stream is stored on decision_node.expression_stream
    and will be invoked during _exec_text_response() when comprehension
    data is present in the payload.
    """
    try:
        from hbllm.brain.snn.expression import (
            ExpressionStream,
            RewardEvaluator,
            ThoughtController,
            ThoughtPlanner,
        )

        planner = ThoughtPlanner(
            base_token_budget=512,
            constraint_expansion=True,
            min_salience_for_goal=0.3,
        )
        controller = ThoughtController(
            readiness_threshold=0.6,
            coherence_threshold=0.5,
            max_wait_steps=5,
        )

        # Try to create STDP plasticity for the controller
        try:
            from hbllm.brain.snn.plasticity import PlasticWeightMatrix, STDPRule

            stdp_rule = STDPRule(
                learning_rate=0.01,
                time_constant=0.5,
                w_min=0.0,
                w_max=2.0,
            )
            ctrl_plastic = PlasticWeightMatrix(
                controller._static_weights,
                stdp_rule,
                neuromodulator=neuromodulator,
            )
            controller.plastic_weights = ctrl_plastic
            logger.info("STDP plasticity enabled for ThoughtController")
        except Exception as e:
            logger.debug("STDP plasticity for controller not available: %s", e)

        # Try to bind the ONNX encoder for embedding-based scoring
        encoder = None
        if router_node is not None and hasattr(router_node, "_encode_text"):
            encoder = router_node._encode_text

        evaluator = RewardEvaluator(
            encoder=encoder,
            min_acceptable_reward=0.4,
        )

        # Try to create TrainedPRM for learnable reward scoring
        trained_prm = None
        try:
            from hbllm.brain.snn.expression.trained_prm import TrainedPRM

            prm_stdp = None
            try:
                from hbllm.brain.snn.plasticity import STDPRule as _PRMSTDPRule

                prm_stdp = _PRMSTDPRule(
                    learning_rate=0.01,
                    time_constant=0.5,
                    w_min=0.0,
                    w_max=2.0,
                )
            except Exception as e:
                logger.debug("[SNN] Plasticity update skipped: %s", e)

            trained_prm = TrainedPRM(
                reward_evaluator=evaluator,
                stdp_rule=prm_stdp,
                fallback_threshold=50,
                snn_blend_weight=0.6,
            )
            logger.info("TrainedPRM wired to ExpressionStream")
        except Exception as e:
            logger.debug("TrainedPRM not available (non-fatal): %s", e)

        # Try to create ShallowRenderer for v3 rendering mode
        shallow_renderer = None
        try:
            from hbllm.brain.snn.expression.shallow_renderer import ShallowRenderer

            shallow_renderer = ShallowRenderer(min_confidence=0.3)
            logger.info("ShallowRenderer wired to ExpressionStream")
        except Exception as e:
            logger.debug("ShallowRenderer not available (non-fatal): %s", e)

        # Try to create v4 Broca components
        content_planner = None
        broca_encoder = None
        try:
            from hbllm.brain.snn.expression.broca_encoder import BrocaEncoder
            from hbllm.brain.snn.expression.content_planner import ContentPlanner

            content_planner = ContentPlanner()
            broca_encoder = BrocaEncoder()
            logger.info("Broca's area components (v4) wired to ExpressionStream")

            # Wire PRMTrainer for batch training
            if trained_prm is not None:
                try:
                    from hbllm.brain.snn.expression.prm_trainer import PRMTrainer

                    prm_trainer = PRMTrainer(trained_prm, epochs=3, batch_size=20)
                    decision_node._prm_trainer = prm_trainer
                    logger.info("PRMTrainer wired to DecisionNode")
                except Exception as e2:
                    logger.debug("PRMTrainer not available (non-fatal): %s", e2)
        except Exception as e:
            logger.debug("Broca components not available (non-fatal): %s", e)

        # Bind LLM generate function if available
        llm_generate = None
        if llm is not None and hasattr(llm, "generate"):

            async def _generate(prompt: str) -> str:
                result = await llm.generate(prompt)
                return str(result)

            llm_generate = _generate

        stream = ExpressionStream(
            planner=planner,
            controller=controller,
            evaluator=evaluator,
            llm_generate=llm_generate,
            max_revisions=1,
            enable_gating=True,
            trained_prm=trained_prm,
            shallow_renderer=shallow_renderer,
            shallow_mode=False,  # Opt-in: brain pipeline needs LLM reasoning
            content_planner=content_planner,
            broca_encoder=broca_encoder,
            broca_mode=False,  # Opt-in: full brain pipeline needs LLM reasoning
            dual_router=dual_router,
        )

        decision_node.expression_stream = stream
        logger.info("ExpressionStream wired to DecisionNode")
    except Exception as e:
        logger.warning("Failed to wire ExpressionStream (non-fatal): %s", e)
