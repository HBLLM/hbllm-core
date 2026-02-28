"""
Global Cognitive Workspace Node (The Blackboard).

Receives intents from the RouterNode. 
Instead of forwarding them strictly to one Domain Expert, it publishes a 
`workspace.update` event. All available independent modules (LLM Intuition, 
Z3 Logic, Fuzzy Engine) can read the blackboard and post `workspace.thought`
proposals back to the workspace concurrently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import asyncio
import time

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class WorkspaceNode(Node):
    """
    The central intelligence blackboard. Maintains state of current reasoning
    efforts and aggregates competing or collaborating thoughts.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER)
        
        # In-memory "Blackboard" mapping conversation correlation_ids
        # to the array of proposed thoughts and their active deadlines.
        self.blackboards: Dict[str, Dict[str, Any]] = {}

    async def on_start(self) -> None:
        """Subscribe to the blackboard topics."""
        logger.info("Starting Global WorkspaceNode")
        await self.bus.subscribe("workspace.update", self.handle_update)
        await self.bus.subscribe("workspace.thought", self.handle_thought)

    async def on_stop(self) -> None:
        logger.info("Stopping Global WorkspaceNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_update(self, message: Message) -> Message | None:
        """
        Triggered when Router identifies context and drops it on the Blackboard.
        """
        payload = message.payload
        correlation_id = message.id
        
        logger.info("Workspace received new Problem State: %s...", str(payload.get("text", ""))[:40])
        
        # Initialize a new Blackboard session for this specific User Query
        self.blackboards[correlation_id] = {
            "tenant_id": message.tenant_id,
            "session_id": message.session_id,
            "original_query": payload,
            "thoughts": [],
            "start_time": time.time(),
            # We give the cognitive modules 4 seconds to think and post proposals
            "deadline": time.time() + 4.0, 
            "resolved": False,
            "turn_count": 1,  # Track internal monologue turns
            "absolute_deadline": time.time() + 30.0,  # Hard cap: 30s max monologue time
        }
        
        # Broadcast the context to ALL subjective thinking modules simultaneously
        # (This avoids the Router playing isolated favorites).
        broadcast_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="module.evaluate",  # Intuition, Logic, and Fuzzy modules listen here
            payload=payload,
            correlation_id=correlation_id
        )
        await self.bus.publish("module.evaluate", broadcast_msg)
        
        # Spawn an async watcher to wait for the deadline or consensus
        asyncio.create_task(self._consensus_watcher(correlation_id))
        
        return None
        
    async def handle_thought(self, message: Message) -> Message | None:
        """
        Triggered when a System 1 or System 2 module posts a thought proposal
        onto the blackboard.
        """
        corr_id = message.correlation_id
        if not corr_id or corr_id not in self.blackboards:
            return None # Orphaned thought
            
        board = self.blackboards[corr_id]
        if board["resolved"]:
            return None # Too late
            
        proposal = message.payload
        thought_type = proposal.get("type", "intuition")
        confidence = proposal.get("confidence", 0.0)
        
        logger.info(
            "Workspace received a [%s] thought with %s confidence from %s", 
            thought_type, confidence, message.source_node_id
        )
        
        board["thoughts"].append({
            "node": message.source_node_id,
            "type": thought_type,
            "confidence": confidence,
            "content": proposal.get("content")
        })
        
        # Phase 9: The Internal Monologue Loop
        if thought_type == "symbolic_logic" and confidence == 1.0:
            is_intermediate = proposal.get("is_intermediate", False)
            if is_intermediate:
                logger.info("Workspace logic proof complete. Feeding back to Intuition Engine for monologue...")
                
                # We inject the logical proof into the user's prompt
                new_payload = board["original_query"].copy()
                new_payload["text"] = f"System Log: The logic engine has mathematically proven: '{proposal.get('content')}'. Based on this, formulate a final friendly response to the user."
                
                board["turn_count"] += 1
                # Extend deadline to allow Intuition node to read the proof and generate text
                board["deadline"] = time.time() + 4.0
                
                # Re-publish as context update for the Intuition Engine
                broadcast_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=board["tenant_id"],
                    session_id=board["session_id"],
                    topic="module.evaluate",
                    payload=new_payload,
                    correlation_id=corr_id
                )
                await self.bus.publish("module.evaluate", broadcast_msg)
            else:
                logger.info("Workspace immediately resolved by formal Symbolic Logic proof.")
                await self._finalize_board(corr_id)
        # Phase 9.2: World Model Simulation Loop
        if thought_type == "simulation_result":
            # The WorldModelNode finished predicting the outcome of an action
            prediction = proposal.get("prediction")
            reason = proposal.get("content")
            
            if prediction == "SUCCESS":
                logger.info("Workspace WorldModel simulation passed nicely. Executing safely.")
                # We can now safely finalize with the original thought 
                # (which was temporarily cached in `board["simulating_thought"]`)
                await self._commit_to_decision(corr_id, board["simulating_thought"])
            else:
                logger.warning("Workspace WorldModel predicted a FAILURE: %s", reason)
                # Initiate an internal monologue turn to fix the broken code
                new_payload = board["original_query"].copy()
                new_payload["text"] = f"System Log: Your previous action was simulated and failed due to: '{reason}'. Please fix the errors and try again."
                
                board["turn_count"] += 1
                board["resolved"] = False # Un-resolve it so thinking continues
                board["deadline"] = time.time() + 4.0
                
                broadcast_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=board["tenant_id"],
                    session_id=board["session_id"],
                    topic="module.evaluate",
                    payload=new_payload,
                    correlation_id=corr_id
                )
                await self.bus.publish("module.evaluate", broadcast_msg)
            return None
            
        # Phase 9.4: Active Halting from CriticNode
        if thought_type == "critique":
            status = proposal.get("status")
            if status == "FAIL":
                target_node = proposal.get("target_node")
                reason = proposal.get("reason")
                content_failed = proposal.get("original_content")
                
                # Prevent infinite backtracking loops — max 3 retries
                retry_key = f"retries_{corr_id}"
                current_retries = board.get(retry_key, 0)
                if current_retries >= 3:
                    logger.warning("Workspace max retries reached for %s, accepting as-is.", corr_id)
                    return None
                board[retry_key] = current_retries + 1
                
                logger.warning(
                    "Workspace halting %s's thought due to Critic evaluation: %s", 
                    target_node, reason
                )
                
                # We need to remove the flawed thought from the board so it isn't picked at consensus
                board["thoughts"] = [t for t in board["thoughts"] if t["content"] != content_failed]
                
                # Trigger a forced backtrack via Internal Monologue
                new_payload = board["original_query"].copy()
                new_payload["text"] = f"CRITICAL FEEDBACK: Your previous thought '{content_failed}' was evaluated by the Critic and FAILED for the following reason: '{reason}'. Please try a completely different approach to answer the user."
                
                board["turn_count"] += 1
                board["deadline"] = time.time() + 4.0 # Give them time to redo it
                
                broadcast_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=self.node_id,
                    tenant_id=board["tenant_id"],
                    session_id=board["session_id"],
                    topic="module.evaluate",
                    payload=new_payload,
                    correlation_id=corr_id
                )
                await self.bus.publish("module.evaluate", broadcast_msg)
            return None
            
        return None
        
    async def _consensus_watcher(self, corr_id: str):
        """Wait until deadline expires, then finalize. Hard 30s absolute cap."""
        board = self.blackboards.get(corr_id)
        if not board:
            return
        
        try:
            while time.time() < board["deadline"] and not board["resolved"]:
                # Enforce absolute ceiling: never wait more than 30s total
                if time.time() >= board.get("absolute_deadline", float("inf")):
                    logger.warning("Workspace absolute deadline reached for %s.", corr_id)
                    break
                await asyncio.sleep(0.1)
            
            if not board["resolved"]:
                await self._finalize_board(corr_id)
        except Exception:
            logger.exception("Error in consensus watcher for %s", corr_id)
            # Emergency cleanup — send an error response so the user isn't left hanging
            await self._send_error_fallback(
                corr_id, "An internal error occurred during reasoning. Please try again."
            )

    async def _finalize_board(self, corr_id: str):
        """
        Examine all thoughts on the blackboard, select the best approach, 
        and forward it to the Decision Engine for action generation.
        """
        board = self.blackboards.get(corr_id)
        if not board or board["resolved"]:
            return
            
        board["resolved"] = True
        
        if not board["thoughts"]:
            logger.warning("Workspace deadline expired with ZERO thoughts generated.")
            # Graceful fallback: send an honest "I couldn't reason about this" reply
            await self._send_error_fallback(
                corr_id,
                "I wasn't able to form a clear response to that. Could you rephrase your question?"
            )
            return
            
        # Select highest confidence thought
        best_thought = max(board["thoughts"], key=lambda t: t["confidence"])
        logger.info(
            "Workspace reached Consensus! Selecting %s thought from %s",
            best_thought["type"], best_thought["node"]
        )

        content = best_thought.get("content", "")
        # Very simple heuristic to check if the LLM output wants to execute python
        # (Assuming the LLM was prompted to output <execute_python> tags)
        if "execute_python" in best_thought["type"] or "<execute_python>" in str(content):
            logger.info("Workspace detected an executable action. Pushing to WorldModel for simulation...")
            board["simulating_thought"] = best_thought
            
            # Extract just the code if possible, for prototype we assume the whole content is code
            # or wrapped tightly. In a real prompt, we'd regex out the python block.
            code_to_sim = str(content)
            
            sim_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=board["tenant_id"],
                session_id=board["session_id"],
                topic="workspace.simulate",
                payload={
                    "action_type": "execute_python",
                    "content": code_to_sim
                },
                correlation_id=corr_id
            )
            await self.bus.publish("workspace.simulate", sim_msg)
            # Do NOT finalize. Wait for `handle_thought` to get the `simulation_result`.
            return

        # If it doesn't need simulation, commit immediately
        await self._commit_to_decision(corr_id, best_thought)

    async def _commit_to_decision(self, corr_id: str, best_thought: dict[str, Any]):
        """Helper to push the final approved thought to the physical Decision Engine."""
        board = self.blackboards.get(corr_id)
        if not board:
            return
        
        try:
            decision_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=board["tenant_id"],
                session_id=board["session_id"],
                topic="decision.evaluate",
                payload={
                    "original_query": board["original_query"],
                    "selected_thought": best_thought
                },
                correlation_id=corr_id
            )
            await self.bus.publish("decision.evaluate", decision_msg)
        except Exception:
            logger.exception("Error committing thought to DecisionNode for %s", corr_id)
            await self._send_error_fallback(
                corr_id, "An error occurred while processing the response. Please try again."
            )
        finally:
            # Cleanup memory regardless of success/failure
            self.blackboards.pop(corr_id, None)
    
    async def _send_error_fallback(self, corr_id: str, error_text: str):
        """
        Send a graceful error response to the user interface so they aren't 
        left waiting forever when something goes wrong.
        """
        board = self.blackboards.get(corr_id)
        try:
            msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=board["tenant_id"] if board else "unknown",
                session_id=board["session_id"] if board else "unknown",
                topic="sensory.output",
                payload={"text": error_text, "source": "workspace_fallback"},
                correlation_id=corr_id
            )
            await self.bus.publish("sensory.output", msg)
        except Exception:
            logger.exception("Failed to send error fallback for %s", corr_id)
        finally:
            self.blackboards.pop(corr_id, None)
