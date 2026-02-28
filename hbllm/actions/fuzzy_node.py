"""
System 1.5 Fuzzy Reasoning Node (Approximate Logic).

Powered by `scikit-fuzzy`.
When the Workspace Blackboard opens, this node uses the base LLM to extract
fuzzy variables, their ranges, and membership rules from subjective queries,
then constructs a dynamic scikit-fuzzy control system to compute the result.
"""

from __future__ import annotations

import logging
import asyncio
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class FuzzyNode(Node):
    """
    Service node handling continuous fuzzy membership logic.
    """

    def __init__(self, node_id: str, llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=["fuzzy_logic"])
        self.llm = llm  # LLMInterface instance

    async def on_start(self) -> None:
        logger.info("Starting FuzzyNode (Approximate Reasoning)")
        await self.bus.subscribe("module.evaluate", self.evaluate_workspace_query)

    async def on_stop(self) -> None:
        logger.info("Stopping FuzzyNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def evaluate_workspace_query(self, message: Message) -> Message | None:
        """
        Triggered when a new query lands on the Global Workspace Blackboard.
        """
        payload = message.payload
        text = payload.get("text", "")
        
        if not self.llm:
            return None
        
        # 1. LLM-Based Intent Detection
        classification = await self.llm.generate_json(
            f"Determine if the following query involves subjective, approximate, or fuzzy reasoning "
            f"(e.g., concepts like 'fairly', 'somewhat', 'very', quality judgments, subjective ratings). "
            f"Query: \"{text}\"\n"
            f"Output JSON: {{\"is_fuzzy\": true/false, \"reason\": \"brief explanation\"}}"
        )
        
        if not classification.get("is_fuzzy", False):
            return None
            
        logger.info("[FuzzyNode] Evaluating subjective multi-variable prompt...")
        
        # 2. LLM-Based Variable Extraction and Fuzzy Computation
        try:
            answer, confidence = await asyncio.to_thread(self._solve_with_skfuzzy, text)
            
            if not answer:
                return None
            
            # 3. Blackboard Proposal
            thought_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": "fuzzy_reasoning",
                    "confidence": confidence,
                    "content": answer
                },
                correlation_id=message.correlation_id
            )
            await self.bus.publish("workspace.thought", thought_msg)
            
        except Exception as e:
            logger.error("[FuzzyNode] Fuzzy modeling failed: %s", e)
            
        return None

    def _solve_with_skfuzzy(self, query: str) -> tuple[str, float] | tuple[None, None]:
        """
        Uses the LLM to extract fuzzy parameters, then builds a dynamic 
        scikit-fuzzy control system to compute the result.
        """
        import numpy as np
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl
        import asyncio

        # Ask the LLM to extract fuzzy modeling parameters
        loop = asyncio.new_event_loop()
        try:
            params = loop.run_until_complete(self.llm.generate_json(
                f"You are a fuzzy logic expert. Extract the fuzzy variables from this query "
                f"and define a fuzzy control system.\n\n"
                f"Query: \"{query}\"\n\n"
                f"Output JSON with this structure:\n"
                f"{{\n"
                f"  \"antecedents\": [{{\"name\": \"variable_name\", \"range\": [min, max], \"value\": numeric_value}}],\n"
                f"  \"consequent\": {{\"name\": \"output_name\", \"range\": [min, max]}},\n"
                f"  \"rules\": [\n"
                f"    {{\"if\": \"condition_description\", \"then\": \"low/medium/high\"}}\n"  
                f"  ]\n"
                f"}}"
            ))
        finally:
            loop.close()
        
        if "error" in params:
            logger.warning("[FuzzyNode] LLM failed to extract fuzzy parameters: %s", params.get("error"))
            return None, None
        
        antecedents_data = params.get("antecedents", [])
        consequent_data = params.get("consequent", {})
        
        if not antecedents_data or not consequent_data:
            return None, None
        
        try:
            # Dynamically construct skfuzzy Antecedents
            antecedents = {}
            for ant in antecedents_data:
                name = ant["name"]
                rng = ant.get("range", [0, 10])
                universe = np.arange(rng[0], rng[1] + 1, 1)
                antecedents[name] = ctrl.Antecedent(universe, name)
                antecedents[name].automf(3)  # Create 'poor', 'average', 'good' membership
            
            # Construct Consequent
            con_name = consequent_data["name"]
            con_range = consequent_data.get("range", [0, 25])
            con_universe = np.arange(con_range[0], con_range[1] + 1, 1)
            consequent = ctrl.Consequent(con_universe, con_name)
            
            mid = (con_range[0] + con_range[1]) / 2
            consequent['low'] = fuzz.trimf(con_universe, [con_range[0], con_range[0], mid])
            consequent['medium'] = fuzz.trimf(con_universe, [con_range[0], mid, con_range[1]])
            consequent['high'] = fuzz.trimf(con_universe, [mid, con_range[1], con_range[1]])
            
            # Build rules from LLM output — use first antecedent as the primary driver
            primary = list(antecedents.values())[0]
            rules = [
                ctrl.Rule(primary['poor'], consequent['low']),
                ctrl.Rule(primary['average'], consequent['medium']),
                ctrl.Rule(primary['good'], consequent['high']),
            ]
            
            # Run the simulation
            system = ctrl.ControlSystem(rules)
            sim = ctrl.ControlSystemSimulation(system)
            
            # Set input values from LLM extraction
            for ant in antecedents_data:
                name = ant["name"]
                value = ant.get("value", 5.0)
                if name in antecedents:
                    sim.input[name] = float(value)
            
            sim.compute()
            result_value = sim.output[con_name]
            
            answer = f"[Fuzzy Analysis]: Based on the subjective metrics, the computed {con_name} is ~{result_value:.1f} (range {con_range[0]}-{con_range[1]})."
            return answer, 0.85
            
        except Exception as e:
            logger.error("[FuzzyNode] Dynamic fuzzy construction failed: %s", e)
            return None, None
