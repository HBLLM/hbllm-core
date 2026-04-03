"""
Global Cognitive Workspace Node (The Blackboard).

Receives intents from the RouterNode.
Instead of forwarding them strictly to one Domain Expert, it publishes a
`workspace.update` event. All available independent modules (LLM Intuition,
Z3 Logic, Fuzzy Engine) can read the blackboard and post `workspace.thought`
proposals back to the workspace concurrently.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from hbllm.network.messages import Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class WorkspaceNode(Node):
    """
    The central intelligence blackboard. Maintains state of current reasoning
    efforts and aggregates competing or collaborating thoughts.
    """

    def __init__(self, node_id: str, thinking_deadline: float = 4.0, max_concurrent_boards: int = 100, max_board_age: float = 300.0):
        super().__init__(node_id=node_id, node_type=NodeType.CORE)

        # In-memory "Blackboard" mapping conversation correlation_ids
        # to the array of proposed thoughts and their active deadlines.
        self.blackboards: dict[str, dict[str, Any]] = {}
        self._sweeper_task: asyncio.Task | None = None
        self._watcher_tasks: set[asyncio.Task] = set()
        # Max age for any blackboard entry before forced cleanup (seconds)
        self._max_board_age = max_board_age
        # Configurable thinking deadline (seconds for cognitive modules to respond)
        self._thinking_deadline = thinking_deadline
        # Max concurrent blackboards to prevent unbounded memory growth
        self._max_concurrent_boards = max_concurrent_boards

    async def on_start(self) -> None:
        """Subscribe to the blackboard topics."""
        logger.info("Starting Global WorkspaceNode")
        await self.bus.subscribe("workspace.update", self.handle_update)
        await self.bus.subscribe("workspace.thought", self.handle_thought)
        # Start periodic sweeper to clean orphaned blackboards
        self._sweeper_task = asyncio.create_task(self._periodic_sweeper())

    async def on_stop(self) -> None:
        """Gracefully shut down: cancel watchers, notify users of active boards, cancel sweeper."""
        logger.info("Stopping Global WorkspaceNode")
        # Cancel the sweeper
        if self._sweeper_task and not self._sweeper_task.done():
            self._sweeper_task.cancel()
            try:
                await self._sweeper_task
            except asyncio.CancelledError:
                pass
        # Cancel all consensus watcher tasks
        for task in self._watcher_tasks:
            if not task.done():
                task.cancel()
        if self._watcher_tasks:
            await asyncio.gather(*self._watcher_tasks, return_exceptions=True)
        self._watcher_tasks.clear()
        # Gracefully close any active blackboards
        for corr_id in list(self.blackboards.keys()):
            await self._send_error_fallback(
                corr_id, "The system is shutting down. Please try again later."
            )

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_update(self, message: Message) -> Message | None:
        """
        Triggered when Router identifies context and drops it on the Blackboard.
        """
        payload = message.payload
        correlation_id = message.correlation_id or message.id

        logger.info("Workspace received new Problem State: %s...", str(payload.get("text", ""))[:40])

        # Guard: evict oldest boards if capacity exceeded
        if len(self.blackboards) >= self._max_concurrent_boards:
            oldest_ids = sorted(
                self.blackboards,
                key=lambda cid: self.blackboards[cid].get("start_time", 0),
            )[:10]
            for cid in oldest_ids:
                logger.warning("Workspace capacity exceeded, evicting stale board: %s", cid)
                await self._send_error_fallback(cid, "System overloaded. Please try again.")

        # Initialize a new Blackboard session for this specific User Query
        self.blackboards[correlation_id] = {
            "tenant_id": message.tenant_id,
            "session_id": message.session_id,
            "original_query": payload,
            "thoughts": [],
            "start_time": time.time(),
            # Configurable deadline for cognitive modules to think and post proposals
            "deadline": time.time() + self._thinking_deadline,
            "resolved": False,
            "turn_count": 1,  # Track internal monologue turns
            "absolute_deadline": time.time() + 30.0,  # Hard cap: 30s max monologue time
        }

        # Broadcast the context to ALL subjective thinking modules simultaneously
        # (This avoids the Router playing isolated favorites).
        # Phase 11 Supplement: Explicitly flag if this query should search priority memory
        broadcast_msg = Message(
            type=MessageType.EVENT,
            source_node_id=self.node_id,
            tenant_id=message.tenant_id,
            session_id=message.session_id,
            topic="module.evaluate",  # Intuition, Logic, and Fuzzy modules listen here
            payload={**payload, "search_priority_memory": True},
            correlation_id=correlation_id
        )
        await self.bus.publish("module.evaluate", broadcast_msg)

        # Spawn an async watcher to wait for the deadline or consensus
        self._spawn_watcher(correlation_id)

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
                board["deadline"] = time.time() + self._thinking_deadline

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
                board["deadline"] = time.time() + self._thinking_deadline

                self._spawn_watcher(corr_id)

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
                new_payload["text"] = f"CONSTITUTIONAL VIOLATION: Your previous thought '{content_failed}' was reviewed by the Critic Node and violated core system principles: '{reason}'. Please revise your response to strictly comply with all safety and logic principles."

                board["turn_count"] += 1
                board["deadline"] = time.time() + self._thinking_deadline # Give them time to redo it
                board["resolved"] = False  # Make sure it's unresolved

                self._spawn_watcher(corr_id)

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

    def _spawn_watcher(self, corr_id: str) -> None:
        """Create a tracked consensus watcher task with automatic cleanup."""
        task = asyncio.create_task(self._consensus_watcher(corr_id))
        self._watcher_tasks.add(task)
        task.add_done_callback(self._watcher_tasks.discard)

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
            await self._send_error_fallback(
                corr_id,
                "I wasn't able to form a clear response to that. Could you rephrase your question?"
            )
            return

        # ── Inject memory context before consensus ──
        await self._inject_memory_context(corr_id, board)

        # Select highest confidence thought
        best_thought = max(board["thoughts"], key=lambda t: t["confidence"])
        logger.info(
            "Workspace reached Consensus! Selecting %s thought from %s",
            best_thought["type"], best_thought["node"]
        )

        content = best_thought.get("content", "")

        # Phase 12: Swarm Handoff Integration
        if best_thought["type"] == "swarm_transfer":
            target_specialty = content.strip().lower()
            logger.info("Workspace initiating Native Swarm Transfer to specialty: '%s'", target_specialty)

            # Repackage the blackboard state so the new specialist has full context
            transfer_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                tenant_id=board["tenant_id"],
                session_id=board["session_id"],
                topic="system.swarm.transfer",
                payload={
                    "target_domain": target_specialty,
                    "original_query": board["original_query"],
                    "history": board["thoughts"]
                },
                correlation_id=corr_id
            )
            await self.bus.publish("system.swarm.transfer", transfer_msg)
            # Cleanup the local blackboard as it's no longer our responsibility
            self.blackboards.pop(corr_id, None)
            return

        import re
        match = re.search(r"```python\n(.*?)```", str(content), re.DOTALL | re.IGNORECASE)

        if match:
            logger.info("Workspace detected python code in winning thought. Verifying via ExecutionNode...")
            code_to_exec = match.group(1).strip()

            exec_msg = Message(
                type=MessageType.QUERY,
                source_node_id=self.node_id,
                tenant_id=board["tenant_id"],
                session_id=board["session_id"],
                topic="action.execute_code",
                payload={"code": code_to_exec},
                correlation_id=corr_id
            )

            try:
                resp = await self.request("action.execute_code", exec_msg, timeout=6.0)
                status = resp.payload.get("status")

                if status == "SUCCESS":
                    logger.info("Workspace verification passed.")
                    best_thought["execution_output"] = resp.payload.get("output", "")
                    await self._commit_to_decision(corr_id, best_thought)
                    # Autonomous DPO signal (Success)
                    await self._emit_training_feedback(board, best_thought, rating=1)
                else:
                    err = resp.payload.get("error", "Unknown execution error")
                    logger.warning("Workspace code execution failed: %s", err)
                    # Phase 3: Autonomous DPO signal (Failure)
                    await self._emit_training_feedback(board, best_thought, rating=-1)

                    # Remove the failing thought so it isn't picked again
                    board["thoughts"] = [t for t in board["thoughts"] if t["content"] != content]

                    # Force internal monologue to fix the code
                    new_payload = board["original_query"].copy()
                    new_payload["text"] = f"CRITICAL SYSTEM ERROR: The code approach you provided failed with the following traceback:\n```\n{err}\n```\n\nPlease analyze the traceback and formulate a corrected approach."

                    board["turn_count"] += 1
                    board["resolved"] = False
                    board["deadline"] = time.time() + self._thinking_deadline

                    self._spawn_watcher(corr_id)

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
            except Exception as e:
                logger.warning("Execution verification timed out: %s. Accepting thought with warning.", e)
                best_thought["execution_warning"] = "Verification timed out"
                await self._commit_to_decision(corr_id, best_thought)
            return

        # Legacy simulation heuristic
        if "execute_python" in best_thought["type"] or "<execute_python>" in str(content):
            logger.info("Workspace detected an executable action. Pushing to WorldModel for simulation...")
            board["simulating_thought"] = best_thought

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
            board["resolved"] = False
            return

        # If it doesn't need simulation, commit immediately
        await self._commit_to_decision(corr_id, best_thought)

    async def _inject_memory_context(self, corr_id: str, board: dict[str, Any]) -> None:
        """
        Query semantic and procedural memory for relevant context before consensus.
        Inject results as supplementary thoughts on the blackboard.

        Skips memory services that have no active subscribers to avoid hangs.
        """
        query_text = board["original_query"].get("text", "")
        if not query_text:
            return

        # Check which memory services are available by inspecting bus subscriptions
        bus_subs = getattr(self.bus, '_subscriptions', {})
        has_semantic = bool(bus_subs.get("memory.search"))
        has_procedural = bool(bus_subs.get("memory.skill.find"))

        if not has_semantic and not has_procedural:
            return  # No memory services available

        async def _try_semantic():
            if not has_semantic:
                return
            try:
                mem_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    tenant_id=board["tenant_id"],
                    session_id=board["session_id"],
                    topic="memory.search",
                    payload={"query": query_text, "limit": 3},
                    correlation_id=corr_id,
                )
                resp = await asyncio.wait_for(
                    self.bus.request("memory.search", mem_msg, timeout=1.5),
                    timeout=2.0,
                )
                results = resp.payload.get("results", [])
                if results:
                    memory_context = "\n".join(
                        f"- {r.get('text', r.get('content', ''))[:200]}" for r in results[:3]
                    )
                    board["thoughts"].append({
                        "node": "memory_retrieval",
                        "type": "memory_context",
                        "confidence": 0.3,
                        "content": f"Relevant past context:\n{memory_context}",
                    })
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                logger.debug("Semantic memory retrieval failed", exc_info=True)

        async def _try_procedural():
            if not has_procedural:
                return
            try:
                skill_msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=self.node_id,
                    tenant_id=board["tenant_id"],
                    topic="memory.skill.find",
                    payload={"query": query_text, "limit": 2},
                    correlation_id=corr_id,
                )
                resp = await asyncio.wait_for(
                    self.bus.request("memory.skill.find", skill_msg, timeout=1.5),
                    timeout=2.0,
                )
                skills = resp.payload.get("skills", [])
                if skills:
                    skill_text = "\n".join(
                        f"- {s.get('name', 'unknown')}: {', '.join(s.get('steps', [])[:3])}"
                        for s in skills[:2]
                    )
                    board["thoughts"].append({
                        "node": "procedural_memory",
                        "type": "skill_context",
                        "confidence": 0.25,
                        "content": f"Applicable learned skills:\n{skill_text}",
                    })
            except (TimeoutError, asyncio.CancelledError):
                pass
            except Exception:
                logger.debug("Procedural memory retrieval failed", exc_info=True)

        # Run both in parallel with a hard 2.5s ceiling
        try:
            await asyncio.wait_for(
                asyncio.gather(_try_semantic(), _try_procedural(), return_exceptions=True),
                timeout=2.5,
            )
        except (TimeoutError, asyncio.CancelledError):
            logger.debug("Memory context injection timed out for %s", corr_id)

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

    def _compress_text(self, text: str, max_chars: int = 4000) -> str:
        """Middle-out truncation for excessively long strings to protect Learner context."""
        if len(text) <= max_chars:
            return text
        half = max_chars // 2
        head = text[:half]
        tail = text[-half:]
        omitted = len(text) - max_chars
        return f"{head}\n\n[... {omitted} characters dynamically omitted to preserve context bounds ...]\n\n{tail}"

    async def _emit_training_feedback(self, board: dict[str, Any], thought: dict[str, Any], rating: int) -> None:
        """Phase 3: Autonomous Neural-Symbolic Training Feedback."""
        try:
            prompt = board["original_query"].get("text", "")

            # Compress response to prevent DPO context window exhaustion
            raw_response = thought.get("content", "")
            response = self._compress_text(raw_response, max_chars=4000)

            # Use original_query ID or session to correlate
            message_id = board["original_query"].get("message_id", "auto_" + str(time.time()))

            feedback_msg = Message(
                type=MessageType.FEEDBACK,
                source_node_id=self.node_id,
                tenant_id=board.get("tenant_id", "default"),
                session_id=board.get("session_id", "default"),
                topic="system.feedback",
                payload={
                    "message_id": message_id,
                    "rating": rating,
                    "prompt": prompt,
                    "response": response,
                }
            )
            await self.bus.publish("system.feedback", feedback_msg)
            logger.info("Emitted autonomous training feedback (rating=%d) for auto-training loop", rating)
        except Exception as e:
            logger.warning("Failed to emit autonomous training feedback: %s", e)

    async def _periodic_sweeper(self) -> None:
        """Periodically clean up blackboard entries that have exceeded max age."""
        while True:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                now = time.time()
                stale_ids = [
                    cid for cid, board in self.blackboards.items()
                    if now - board.get("start_time", now) > self._max_board_age
                ]
                for cid in stale_ids:
                    logger.warning("Sweeper cleaning stale blackboard: %s", cid)
                    await self._send_error_fallback(
                        cid, "Request timed out. Please try again."
                    )
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in blackboard sweeper")
