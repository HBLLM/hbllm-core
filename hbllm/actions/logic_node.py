"""
System 2 Symbolic Logic Node (Deductive Reasoning).

Powered by Microsoft's Z3 Theorem Prover.
When the Workspace Blackboard opens, this node looks for formal logical or 
mathematical constraints in the prompt. If found, it uses the base LLM to 
translate the text into Z3 Python solver code, executes it in a sandboxed 
namespace, and posts the verified result back to the Blackboard.
"""

from __future__ import annotations

import logging
import asyncio

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class LogicNode(Node):
    """
    Service node that represents slow, methodical 'System 2' logical deduction.
    """

    def __init__(self, node_id: str, llm=None):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE, capabilities=["symbolic_math", "theorem_proving"])
        self.llm = llm  # LLMInterface instance — injected at startup

    async def on_start(self) -> None:
        """Subscribe to the Workspace's open evaluation calls."""
        logger.info("Starting LogicNode (System 2)")
        await self.bus.subscribe("module.evaluate", self.evaluate_workspace_query)

    async def on_stop(self) -> None:
        logger.info("Stopping LogicNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def evaluate_workspace_query(self, message: Message) -> Message | None:
        """
        Triggered when a new query lands on the Global Workspace Blackboard.
        """
        payload = message.payload
        text = payload.get("text", "")
        
        # 1. Intent Detection — use LLM to determine if this is a logic problem
        if not self.llm:
            return None

        classification = await self.llm.generate_json(
            f"Classify whether the following query requires formal logical deduction, "
            f"mathematical constraint solving, or theorem proving. "
            f"Query: \"{text}\"\n"
            f"Output JSON: {{\"is_logical\": true/false, \"reason\": \"brief explanation\"}}"
        )
        
        if not classification.get("is_logical", False):
            return None
            
        logger.info("[LogicNode] Evaluating formal constraint problem...")
        
        # 2. LLM Translation: English → Z3 Python Code
        try:
            answer = await asyncio.to_thread(self._solve_with_z3, text)
            
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
                    "type": "symbolic_logic",
                    "confidence": 1.0,  # Z3 answers are unequivocally true
                    "content": answer,
                    "is_intermediate": True  # Return to Intuition Engine for conversational wrapper
                },
                correlation_id=message.correlation_id
            )
            await self.bus.publish("workspace.thought", thought_msg)
            
        except Exception as e:
            logger.error("[LogicNode] Constraint modeling failed: %s", e)
            
        return None

    def _solve_with_z3(self, query: str) -> str | None:
        """
        Uses the LLM to translate English into Z3 solver code, then executes 
        it in a restricted sandbox to obtain the proven result.
        """
        import z3
        import asyncio
        
        # Ask the LLM to write Z3 code
        loop = asyncio.new_event_loop()
        try:
            z3_code = loop.run_until_complete(self.llm.generate(
                f"You are a Z3 theorem prover expert. Translate the following problem into "
                f"Python code using the z3-solver library. Use ONLY z3 API calls. "
                f"The code must define variables, add constraints to a Solver, check satisfiability, "
                f"and store the final human-readable answer string in a variable called `result`.\n\n"
                f"Problem: \"{query}\"\n\n"
                f"Output ONLY the Python code, no explanations:\n```python\n",
                max_tokens=256,
                temperature=0.2
            ))
        finally:
            loop.close()
        
        # Clean the output — strip code fences if present
        z3_code = z3_code.strip()
        if z3_code.startswith("```"):
            z3_code = z3_code.split("\n", 1)[-1]  # Remove opening fence line
        if z3_code.endswith("```"):
            z3_code = z3_code.rsplit("```", 1)[0]
        z3_code = z3_code.strip()
        
        if not z3_code:
            return None
        
        logger.info("[LogicNode] Executing LLM-generated Z3 code:\n%s", z3_code[:200])
        
        # 3. Execute in a restricted sandbox
        sandbox_globals = {
            "__builtins__": {
                "print": print, "range": range, "len": len,
                "str": str, "int": int, "float": float, "bool": bool,
                "list": list, "dict": dict, "tuple": tuple,
                "True": True, "False": False, "None": None,
            },
            "z3": z3,
        }
        sandbox_locals: dict = {}
        
        # Inject common z3 names into the sandbox
        for name in dir(z3):
            if not name.startswith("_"):
                sandbox_globals[name] = getattr(z3, name)
        
        try:
            exec(z3_code, sandbox_globals, sandbox_locals)
            result = sandbox_locals.get("result")
            if result is not None:
                return str(result)
            else:
                logger.warning("[LogicNode] Z3 code executed but no 'result' variable found.")
                return None
        except Exception as e:
            logger.error("[LogicNode] Z3 sandbox execution failed: %s", e)
            return None
