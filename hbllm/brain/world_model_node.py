"""
System 3 World Model Node (Environment Simulation).

Before the Cognitive Workspace commits to executing physical actions 
(like running Python scripts or clicking links), it asks the World Model
to predict the outcome. If the simulation predicts failure or a safety
violation, the World Model rejects the thought back to the Blackboard 
for the Intuition Engine to rewrite.
"""

from __future__ import annotations

import ast
import logging
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class WorldModelNode(Node):
    """
    Simulation Node to predict success/failure of proposed actions before execution.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=["simulation", "ast_parsing"])
        # A list of inherently dangerous modules we want to prevent the LLM from executing
        self.dangerous_imports = {"os", "subprocess", "sys", "shutil", "socket"}

    async def on_start(self) -> None:
        logger.info("Starting WorldModelNode (System 3 Simulation Engine)")
        await self.bus.subscribe("workspace.simulate", self.simulate_action)

    async def on_stop(self) -> None:
        logger.info("Stopping WorldModelNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def simulate_action(self, message: Message) -> Message | None:
        """
        Triggered when Workspace wants to test an action before executing it.
        """
        payload = message.payload
        action_type = payload.get("action_type")
        
        if action_type == "execute_python":
            code = payload.get("content", "")
            logger.info("[WorldModel] Simulating Python AST execution...")
            prediction = self._simulate_ast(code)
            
            # Post the simulation results back to the workspace
            sim_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": "simulation_result",
                    "confidence": 1.0, # The AST validation is deterministic
                    "prediction": prediction["status"], # 'SUCCESS' or 'FAILURE'
                    "content": prediction["reason"]
                },
                correlation_id=message.correlation_id
            )
            await self.bus.publish("workspace.thought", sim_msg)
            
        return None

    def _simulate_ast(self, code: str) -> dict[str, str]:
        """
        Statistcally parse Python code to predict runtime failure or catch dangerous operations.
        (A true AGI World Model would use a massive transformer to predict the world state,
        but for this prototype, AST parsing serves the same gateway architectural purpose).
        """
        try:
            tree = ast.parse(code)
            
            # Walk the AST looking for imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] in self.dangerous_imports:
                            return {"status": "FAILURE", "reason": f"ImportError: the module '{alias.name}' is blocked by sandbox safety policies."}
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split('.')[0] in self.dangerous_imports:
                        return {"status": "FAILURE", "reason": f"ImportError: the module '{node.module}' is blocked by sandbox safety policies."}

            return {"status": "SUCCESS", "reason": "AST check passed. Code appears structurally safe."}
            
        except SyntaxError as e:
            return {"status": "FAILURE", "reason": f"SyntaxError on line {e.lineno}: {e.msg}"}
        except Exception as e:
            return {"status": "FAILURE", "reason": f"Simulation crashed: {str(e)}"}
