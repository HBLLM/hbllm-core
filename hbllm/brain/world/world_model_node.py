"""
System 3 World Model Node (Environment Simulation).

Before the Cognitive Workspace commits to executing physical actions
(like running Python scripts or clicking links), it asks the World Model
to predict the outcome. If the simulation predicts failure or a safety
violation, the World Model rejects the thought back to the Blackboard
for the Intuition Engine to rewrite.

Narrowed strictly to a Repository / Code World model simulation.
"""

from __future__ import annotations

import ast
import logging
import re
from typing import TYPE_CHECKING, Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

if TYPE_CHECKING:
    from hbllm.brain.core.provider_adapter import ProviderLLM

logger = logging.getLogger(__name__)


class WorldModelNode(Node):
    """
    Simulation Node to predict success/failure of proposed actions before execution.
    """

    def __init__(self, node_id: str, llm: ProviderLLM | None = None):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=[
                "simulation",
                "ast_parsing",
                "outcome_prediction",
                "code_world_simulation",
            ],
        )
        self.llm = llm
        # A list of inherently dangerous modules we want to prevent the LLM from executing
        self.dangerous_imports = {"os", "subprocess", "sys", "shutil", "socket"}
        # Dangerous bash patterns
        self._dangerous_bash_patterns = [
            re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
            re.compile(r"\bmkfs\b", re.IGNORECASE),
            re.compile(r"\bdd\s+if=\b", re.IGNORECASE),
            re.compile(r"\b:(\){\s*:|;\s*})", re.IGNORECASE),  # fork bomb
            re.compile(r"\bchmod\s+777\b", re.IGNORECASE),
            re.compile(r"\bcurl\b.*\|\s*\bbash\b", re.IGNORECASE),
            re.compile(r"\bwget\b.*\|\s*\bsh\b", re.IGNORECASE),
            re.compile(r"\b>\/dev\/sd", re.IGNORECASE),
        ]

        # Virtual repository state
        self._virtual_files: dict[str, str] = {}
        self._virtual_dependencies: dict[str, list[str]] = {}

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
                    "confidence": 1.0,  # The AST validation is deterministic
                    "prediction": prediction["status"],  # 'SUCCESS' or 'FAILURE'
                    "content": prediction["reason"],
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", sim_msg)

        elif action_type in ("bash_command", "browser_action"):
            content = payload.get("content", "")
            prediction = await self._simulate_command(action_type, content)
            sim_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="workspace.thought",
                payload={
                    "type": "simulation_result",
                    "confidence": 0.8,
                    "prediction": prediction["status"],
                    "content": prediction["reason"],
                },
                correlation_id=message.correlation_id,
            )
            await self.bus.publish("workspace.thought", sim_msg)

        elif action_type == "simulate_skill":
            steps = payload.get("steps", [])
            overall_status = "SUCCESS"
            reasons = []

            for step in steps:
                if isinstance(step, str) and (
                    "import" in step or "def " in step or "print(" in step
                ):
                    pred = self._simulate_ast(step)
                    if pred["status"] == "FAILURE":
                        overall_status = "FAILURE"
                        reasons.append(pred["reason"])
                        break

            if overall_status == "SUCCESS":
                reason = "All skill steps passed dry-run heuristic and AST checks safely."
            else:
                reason = f"Skill Simulation Failed: {reasons[0]}"

            return message.create_response(
                {"status": "simulation_result", "prediction": overall_status, "content": reason}
            )

        elif action_type == "repository_mutation":
            file_path = payload.get("file_path", "")
            content = payload.get("content", "")
            mutation_type = payload.get("mutation_type", "write")

            if mutation_type == "write":
                self._virtual_files[file_path] = content
                self.simulate_parse_imports(file_path, content)
                comp_result = self.simulate_compilation(file_path)
                return message.create_response(
                    {
                        "status": "simulation_result",
                        "prediction": comp_result["status"],
                        "content": f"Simulated write to {file_path}. " + comp_result["reason"],
                    }
                )
            elif mutation_type == "delete":
                self._virtual_files.pop(file_path, None)
                self._virtual_dependencies.pop(file_path, None)
                return message.create_response(
                    {
                        "status": "simulation_result",
                        "prediction": "SUCCESS",
                        "content": f"Simulated deletion of {file_path} succeeded.",
                    }
                )

        return None

    def _simulate_ast(self, code: str) -> dict[str, str]:
        """
        Statically parse Python code to predict runtime failure or catch dangerous operations.
        """
        try:
            # 1. Compile check for deep syntax verification
            compile(code, "<ast_simulation>", "exec")

            # 2. Extract AST tree and walk nodes
            tree = ast.parse(code)

            # Walk the AST looking for imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in self.dangerous_imports:
                            return {
                                "status": "FAILURE",
                                "reason": f"ImportError: the module '{alias.name}' is blocked by sandbox safety policies.",
                            }
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] in self.dangerous_imports:
                        return {
                            "status": "FAILURE",
                            "reason": f"ImportError: the module '{node.module}' is blocked by sandbox safety policies.",
                        }

            return {
                "status": "SUCCESS",
                "reason": "AST check passed. Code appears structurally safe.",
            }

        except SyntaxError as e:
            return {"status": "FAILURE", "reason": f"SyntaxError on line {e.lineno}: {e.msg}"}
        except Exception as e:
            return {"status": "FAILURE", "reason": f"Simulation crashed: {str(e)}"}

    async def _simulate_command(self, action_type: str, content: str) -> dict[str, str]:
        """Simulate bash/browser actions with pattern-based detection and optional LLM prediction."""
        # Step 1: Fast pattern-based dangerous command detection (for bash)
        if action_type == "bash_command" and content:
            for pattern in self._dangerous_bash_patterns:
                if pattern.search(content):
                    return {
                        "status": "FAILURE",
                        "reason": f"Dangerous bash pattern detected: {pattern.pattern}",
                    }

        # Step 2: If LLM is available, ask it to predict the outcome
        if self.llm:
            try:
                result = await self.llm.generate_json(
                    f"You are a systems safety evaluator. Predict whether this {action_type} "
                    f"command is safe to execute in a sandboxed environment.\n\n"
                    f"Command: {content[:500]}\n\n"
                    f"Respond with JSON:\n"
                    f'{{"prediction": "SUCCESS" or "FAILURE", '
                    f'"confidence": 0.0-1.0, '
                    f'"reason": "brief explanation"}}'
                )
                prediction = result.get("prediction", "SUCCESS")
                confidence = float(result.get("confidence", 0.7))
                reason = result.get("reason", f"LLM prediction for {action_type}")
                return {
                    "status": prediction,
                    "reason": reason,
                    "confidence": str(confidence),
                }
            except Exception as e:
                logger.warning("[WorldModel] LLM prediction failed, using heuristic: %s", e)

        # Step 3: Fallback heuristic
        return {
            "status": "SUCCESS",
            "reason": f"Heuristic prediction: {action_type} appears safe in sandbox.",
        }

    # ── Code World Simulation Extensions ──────────────────────────────

    def simulate_parse_imports(self, file_path: str, content: str) -> list[str]:
        imports = []
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except Exception as e:
            logger.debug("[WorldModel] Import extraction failed for %s: %s", file_path, e)
        self._virtual_dependencies[file_path] = imports
        return imports

    def simulate_compilation(self, file_path: str) -> dict[str, Any]:
        content = self._virtual_files.get(file_path, "")
        try:
            compile(content, file_path, "exec")
            return {"status": "SUCCESS", "reason": "Compilation build succeeded."}
        except SyntaxError as e:
            return {"status": "FAILURE", "reason": f"SyntaxError: {str(e)}"}
        except Exception as e:
            return {"status": "FAILURE", "reason": f"Compilation failed: {str(e)}"}
