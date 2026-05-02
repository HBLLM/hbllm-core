"""
Skill Induction Node — autonomous generation of new atomic capabilities.

Transforms a "Capability Gap" detected by Reflection into a valid,
sandboxed Python Tool that can be registered by the Agent.
"""

from __future__ import annotations

import ast
import logging
import time
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class SecurityInterceptor(ast.NodeVisitor):
    """
    AST-based safety checker for induced code.
    Blocks dangerous imports and system-level calls.
    """

    DANGEROUS_MODULES = {"os", "subprocess", "shutil", "socket", "sys", "pathlib", "builtins"}
    DANGEROUS_FUNCTIONS = {"exec", "eval", "getattr", "setattr", "delattr", "open"}

    def __init__(self):
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name in self.DANGEROUS_MODULES:
                self.errors.append(f"Forbidden import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module in self.DANGEROUS_MODULES:
            self.errors.append(f"Forbidden import from module: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in self.DANGEROUS_FUNCTIONS:
                self.errors.append(f"Forbidden function call: {node.func.id}")
        self.generic_visit(node)


class SkillInductionNode(Node):
    """
    Induces new Python tools to solve identified capability gaps.
    """

    def __init__(self, node_id: str, llm: Any = None) -> None:
        super().__init__(
            node_id=node_id,
            node_type=NodeType.META,
            capabilities=["skill_induction", "code_generation"],
        )
        self.llm = llm

    async def on_start(self) -> None:
        logger.info(f"Starting SkillInductionNode: {self.node_id}")
        await self.bus.subscribe("system.induction.request", self.handle_message)

    async def on_stop(self) -> None:
        pass

    async def handle_message(self, message: Message) -> Message | None:
        if message.type != MessageType.QUERY:
            return None

        gap_description = message.payload.get("gap", "")
        if not gap_description:
            return message.create_error("No gap description provided for induction.")

        logger.info(f"Inducing skill for gap: {gap_description}")

        # 1. Generate Python Tool via LLM
        # We prompt for a specific format that the @tool decorator expects
        prompt = (
            "You are the HBLLM Skill Induction Engine.\n"
            "Generate a specialized Python tool function to fill the following capability gap.\n\n"
            f"Gap: {gap_description}\n\n"
            "Constraints:\n"
            "1. Must be a single Python function.\n"
            "2. Must have a clear docstring and type hints.\n"
            "3. Must NOT use dangerous modules like os, subprocess, or sys.\n"
            '4. Return ONLY the code in a JSON block: {"name": "tool_name", "code": "def ...", "description": "..."}\n'
        )

        try:
            if not self.llm:
                return message.create_error("No LLM available for induction.")

            # Using generate_json if available, or fallback
            induction_data = await self.llm.generate_json(prompt)

            tool_name = induction_data.get("name")
            tool_code = induction_data.get("code")
            tool_desc = induction_data.get("description")

            if not tool_name or not tool_code:
                return message.create_error("LLM failed to generate valid tool data.")

            # 2. Security Validation
            interceptor = SecurityInterceptor()
            try:
                tree = ast.parse(tool_code)
                interceptor.visit(tree)
            except SyntaxError as se:
                return message.create_error(f"Induced code has syntax errors: {str(se)}")

            if interceptor.errors:
                logger.error(f"Security Policy Violation in induced skill: {interceptor.errors}")
                return message.create_error(
                    f"Security validation failed: {', '.join(interceptor.errors)}"
                )

            # 3. Success! Publish induction event
            induced_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=message.tenant_id,
                session_id=message.session_id,
                topic="system.skill.induced",
                payload={
                    "name": tool_name,
                    "code": tool_code,
                    "description": tool_desc,
                    "gap": gap_description,
                    "timestamp": time.time(),
                },
                correlation_id=message.correlation_id or message.id,
            )
            await self.bus.publish("system.skill.induced", induced_msg)

            logger.info(f"Successfully induced skill: {tool_name}")
            return message.create_response({"status": "SUCCESS", "skill_name": tool_name})

        except Exception as e:
            logger.error(f"Skill induction failed: {e}")
            return message.create_error(f"Induction pipeline error: {str(e)}")
