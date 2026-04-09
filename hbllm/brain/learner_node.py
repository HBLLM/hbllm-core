"""
Continuous Learning Engine (Artificial Neuroplasticity).

Listens for Feedback messages on the bus. When a user provides feedback
(positive or negative) on a generation, it accumulates the sample.
During the Sleep Cycle, it performs DPO (Direct Preference Optimization)
to update active LoRA adapter weights — effectively "learning" from
the day's interactions without interrupting live service.

v2 enhancements:
- Micro-learning: single-step LoRA updates between tasks from evaluation feedback
- Knowledge distillation: captures high-confidence successful responses
- Subscribes to system.evaluation for real-time learning signals
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    import torch.nn

from hbllm.network.messages import FeedbackPayload, Message, MessageType
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class LearnerNode(Node):
    """
    Continuous Learning Engine (Artificial Neuroplasticity).

    v2: Now supports micro-learning between tasks via system.evaluation events,
    enabling real-time adaptation without waiting for the sleep cycle.
    """

    def __init__(
        self,
        node_id: str,
        model: Any = None,  # torch.nn.Module | None
        tokenizer: Any = None,
        batch_size: int = 4,
        lr: float = 1e-5,
        dpo_beta: float = 0.1,
        lora_r: int = 8,
        checkpoint_dir: str = "./checkpoints/learner",
        enable_micro_learning: bool = True,
        micro_learn_threshold: float = 0.3,
        distillation_threshold: float = 0.85,
    ) -> None:
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE)
        self.node_type = NodeType.MEMORY
        self.model = model
        self.tokenizer = tokenizer

        # Continuous Lifelong DPO: persistent queue
        self.pending_pairs: dict[str, dict[str, str | None]] = {}
        self.queue_path = "workspace/reflection/dpo_queue.json"

        os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)

        self.batch_size = batch_size
        self.lr = lr
        self.dpo_beta = dpo_beta
        self.lora_r = lora_r
        self.checkpoint_dir = checkpoint_dir
        self.training_task: asyncio.Task[None] | None = None
        self._optimizer: Any | None = None
        self._lora_injected = False
        self._training_steps = 0
        self._lock = asyncio.Lock()

        # v2: Micro-learning settings
        self.enable_micro_learning = enable_micro_learning
        self.micro_learn_threshold = micro_learn_threshold  # learn from low-scoring interactions
        self.distillation_threshold = distillation_threshold  # distill from high-scoring ones

        # v2: Micro-learning stats
        self._micro_learn_steps = 0
        self._distillation_count = 0
        self._micro_learn_queue: list[dict[str, Any]] = []
        self._distillation_bank: list[dict[str, str]] = []
        self._max_micro_queue = 50
        self._max_distillation_bank = 200

    async def on_start(self) -> None:
        """Subscribe to feedback messages, sleep triggers, and evaluation events."""
        logger.info(
            "Starting LearnerNode '%s' (micro_learning=%s)",
            self.node_id,
            self.enable_micro_learning,
        )
        await self.bus.subscribe("system.feedback", self.handle_message)
        await self.bus.subscribe("system.sleep.dpo_trigger", self.handle_sleep_trigger)

        # v2: Subscribe to evaluation events for micro-learning
        if self.enable_micro_learning:
            await self.bus.subscribe("system.evaluation", self._handle_evaluation)

    async def on_stop(self) -> None:
        """Clean up."""
        logger.info(
            "Stopping LearnerNode '%s' (steps=%d micro=%d distilled=%d)",
            self.node_id,
            self._training_steps,
            self._micro_learn_steps,
            self._distillation_count,
        )
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()
            try:
                await self.training_task
            except asyncio.CancelledError:
                pass

    async def handle_message(self, message: Message) -> Message | None:
        """Process incoming feedback."""
        if message.type != MessageType.FEEDBACK:
            return None

        try:
            payload = FeedbackPayload(**message.payload)
        except Exception as e:
            logger.error("Learner '%s' received invalid FeedbackPayload: %s", self.node_id, e)
            return message.create_error(f"Invalid FeedbackPayload: {e}")

        prompt = payload.prompt
        response = payload.response

        if not prompt or not response:
            logger.debug(
                "Learner '%s' received feedback with missing prompt/response", self.node_id
            )
            return None

        async with self._lock:
            logger.info(
                "Learner '%s' processing feedback for msg %s (rating: %d)",
                self.node_id,
                payload.message_id,
                payload.rating,
            )

            if prompt not in self.pending_pairs:
                self.pending_pairs[prompt] = {"chosen": None, "rejected": None}

            if payload.rating == 1:
                self.pending_pairs[prompt]["chosen"] = response
            elif payload.rating == -1:
                self.pending_pairs[prompt]["rejected"] = response

            pair = self.pending_pairs[prompt]
            if pair["chosen"] and pair["rejected"]:
                # Append to persistent JSON array
                queue = []
                p = Path(self.queue_path)
                if p.exists():
                    try:
                        with p.open("r") as f:
                            queue = json.load(f)
                            if not isinstance(queue, list):
                                queue = []
                    except (json.JSONDecodeError, OSError) as e:
                        logger.warning("Could not read dpo_queue.json, starting fresh: %s", e)
                        queue = []

                queue.append((prompt, pair["chosen"], pair["rejected"]))

                try:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    with p.open("w") as f:
                        json.dump(queue, f)
                    logger.info(
                        "LearnerNode queued contrastive DPO pair for prompt: '%s...'", prompt[:30]
                    )
                except OSError as e:
                    logger.error("Failed to write to dpo_queue.json: %s", e)

                del self.pending_pairs[prompt]

        return None

    async def handle_sleep_trigger(self, message: Message) -> Message | None:
        """Triggered by SleepNode when user is idle for background DPO processing."""
        if not os.path.exists(self.queue_path):
            await self._broadcast_complete()
            return None

        with open(self.queue_path) as f:
            try:
                batch = json.load(f)
            except json.JSONDecodeError:
                batch = []

        if not batch:
            await self._broadcast_complete()
            return None

        # Empty the queue so we don't double train if it crashes midway
        os.remove(self.queue_path)

        if self.training_task and not self.training_task.done():
            logger.warning("[LearnerNode] DPO already running.")
            return None

        # Cap batch to self.batch_size to prevent memory explosion
        target_batch = batch[: self.batch_size]
        re_queue = batch[self.batch_size :]

        if re_queue:
            with open(self.queue_path, "w") as f:
                json.dump(re_queue, f)

        self.training_task = asyncio.create_task(self._run_dpo_training(target_batch))
        return None

    async def _run_dpo_training(self, batch: list[tuple[str, str, str]]) -> None:
        """Execute DPO training in a background thread."""
        logger.info("LearnerNode starting DPO training on %d perfect pairs...", len(batch))

        def _train() -> None:
            import torch

            if self.model is None or self.tokenizer is None:
                logger.warning("No model/tokenizer available. Skipping DPO training.")
                return

            self._ensure_lora()
            self._ensure_optimizer()

            from hbllm.training.dpo import compute_dpo_loss, get_batch_logps

            device = next(self.model.parameters()).device
            self.model.train()

            # Build DPO pairs from feedback
            for prompt_text, chosen_text, rejected_text in batch:
                try:
                    pair = self._build_dpo_pair(prompt_text, chosen_text, rejected_text, device)
                    if pair is None:
                        continue

                    chosen_ids, rejected_ids = pair

                    # ── Reference log-probs: base model with LoRA disabled ──
                    from hbllm.modules.lora import LoRAManager

                    LoRAManager.set_active_adapter(self.model, None)

                    with torch.no_grad():
                        ref_chosen_out = self.model(chosen_ids)
                        ref_rejected_out = self.model(rejected_ids)
                        ref_chosen_logits = (
                            ref_chosen_out["logits"]
                            if isinstance(ref_chosen_out, dict)
                            else ref_chosen_out
                        )
                        ref_rejected_logits = (
                            ref_rejected_out["logits"]
                            if isinstance(ref_rejected_out, dict)
                            else ref_rejected_out
                        )
                        ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_ids)
                        ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_ids)

                    LoRAManager.set_active_adapter(self.model, "default")

                    # ── Policy log-probs: model with LoRA active ──
                    chosen_out = self.model(chosen_ids)
                    rejected_out = self.model(rejected_ids)

                    chosen_logits = (
                        chosen_out["logits"] if isinstance(chosen_out, dict) else chosen_out
                    )
                    rejected_logits = (
                        rejected_out["logits"] if isinstance(rejected_out, dict) else rejected_out
                    )

                    policy_chosen_logps = get_batch_logps(chosen_logits, chosen_ids)
                    policy_rejected_logps = get_batch_logps(rejected_logits, rejected_ids)

                    losses, chosen_rewards, rejected_rewards = compute_dpo_loss(
                        policy_chosen_logps=policy_chosen_logps,
                        policy_rejected_logps=policy_rejected_logps,
                        reference_chosen_logps=ref_chosen_logps,
                        reference_rejected_logps=ref_rejected_logps,
                        beta=self.dpo_beta,
                    )

                    loss = losses.mean()
                    loss.backward()  # type: ignore[no-untyped-call]

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    if self._optimizer:
                        self._optimizer.step()
                        self._optimizer.zero_grad()
                    self._training_steps += 1

                    reward_margin = float((chosen_rewards - rejected_rewards).mean().item())
                    logger.info(
                        "  DPO step %d: loss=%.4f reward_margin=%.4f",
                        self._training_steps,
                        float(loss.item()),
                        reward_margin,
                    )

                except Exception as e:
                    logger.warning("DPO step failed for sample: %s", e)
                    continue

            # Save LoRA adapter if we did any training
            if self._training_steps > 0 and self._lora_injected:
                self._save_adapter()

        try:
            await asyncio.to_thread(_train)
            logger.info(
                "LearnerNode DPO training complete (%d total steps). Broadcasting update...",
                self._training_steps,
            )
            await self._broadcast_complete()

        except Exception as e:
            logger.error("LearnerNode background DPO task failed: %s", e)
            await self._broadcast_complete()

    async def _broadcast_complete(self) -> None:
        update_msg = Message(
            type=MessageType.LEARNING_UPDATE,
            source_node_id=self.node_id,
            target_node_id="",
            topic="system.learning_update",
            payload={"status": "weights_updated", "steps": getattr(self, "_training_steps", 0)},
        )
        await self.bus.publish("system.learning_update", update_msg)

    # ── v2: Micro-Learning ───────────────────────────────────────────

    async def _handle_evaluation(self, message: Message) -> None:
        """
        Process evaluation events for micro-learning.

        Two paths:
        1. Low-scoring interactions (< micro_learn_threshold) → queue for
           immediate single-step correction when a better response is available
        2. High-scoring interactions (> distillation_threshold) → store in
           distillation bank for knowledge reinforcement during sleep
        """
        payload = message.payload
        overall_score = payload.get("overall_score", 0.5)
        query = payload.get("query", "")
        response = payload.get("response", "")

        if not query or not response:
            return

        # Path 1: Low-scoring → queue for micro-correction
        if overall_score < self.micro_learn_threshold:
            if len(self._micro_learn_queue) < self._max_micro_queue:
                self._micro_learn_queue.append({
                    "query": query,
                    "bad_response": response,
                    "score": overall_score,
                    "dimensions": payload.get("dimensions", {}),
                })
                logger.debug(
                    "[LearnerNode] Queued low-score interaction for micro-learning "
                    "(score=%.2f, queue=%d)",
                    overall_score,
                    len(self._micro_learn_queue),
                )

        # Path 2: High-scoring → distillation bank
        elif overall_score > self.distillation_threshold:
            if len(self._distillation_bank) < self._max_distillation_bank:
                self._distillation_bank.append({
                    "query": query,
                    "response": response,
                })
                self._distillation_count += 1
                logger.debug(
                    "[LearnerNode] Banked high-confidence response for distillation "
                    "(score=%.2f, bank=%d)",
                    overall_score,
                    len(self._distillation_bank),
                )

    async def micro_learn(
        self,
        query: str,
        bad_response: str,
        good_response: str,
    ) -> bool:
        """
        Execute a single-step LoRA update for immediate correction.

        Called when EvaluationNode detects a poor response AND a better
        alternative is available (e.g., from user correction or retry).

        Returns True if the micro-learning step was executed.
        """
        if self.model is None or self.tokenizer is None:
            logger.debug("[LearnerNode] No model available for micro-learning")
            return False

        if not self.enable_micro_learning:
            return False

        async with self._lock:
            try:
                # Queue as a DPO pair for immediate processing
                pair_key = query[:100]  # truncate for dict key
                if pair_key not in self.pending_pairs:
                    self.pending_pairs[pair_key] = {"chosen": None, "rejected": None}

                self.pending_pairs[pair_key]["chosen"] = good_response
                self.pending_pairs[pair_key]["rejected"] = bad_response

                # Persist to queue for sleep-cycle batch processing
                queue: list[Any] = []
                p = Path(self.queue_path)
                if p.exists():
                    try:
                        with p.open("r") as f:
                            queue = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        queue = []

                queue.append((query, good_response, bad_response))
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("w") as f:
                    json.dump(queue, f)

                del self.pending_pairs[pair_key]
                self._micro_learn_steps += 1

                logger.info(
                    "[LearnerNode] Micro-learn step %d: queued correction for '%s...'",
                    self._micro_learn_steps,
                    query[:40],
                )
                return True

            except Exception as e:
                logger.warning("[LearnerNode] Micro-learn failed: %s", e)
                return False

    def get_distillation_bank(self) -> list[dict[str, str]]:
        """Get the current distillation bank for sleep-cycle knowledge reinforcement."""
        return list(self._distillation_bank)

    def clear_distillation_bank(self) -> int:
        """Clear and return the count of distilled items."""
        count = len(self._distillation_bank)
        self._distillation_bank.clear()
        return count

    def get_micro_learn_queue(self) -> list[dict[str, Any]]:
        """Get queued low-scoring interactions awaiting correction."""
        return list(self._micro_learn_queue)

    def micro_learning_stats(self) -> dict[str, Any]:
        """Return micro-learning statistics."""
        return {
            "enabled": self.enable_micro_learning,
            "micro_learn_steps": self._micro_learn_steps,
            "distillation_count": self._distillation_count,
            "micro_queue_depth": len(self._micro_learn_queue),
            "distillation_bank_size": len(self._distillation_bank),
            "total_dpo_steps": self._training_steps,
            "thresholds": {
                "micro_learn": self.micro_learn_threshold,
                "distillation": self.distillation_threshold,
            },
        }

    def _build_dpo_pair(
        self,
        prompt_text: str,
        chosen_text: str,
        rejected_text: str,
        device: Any,  # torch.device
    ) -> tuple[Any, Any] | None:
        """Build a chosen/rejected tensor pair."""
        if not prompt_text or not chosen_text or not rejected_text:
            return None

        try:
            import torch

            # Tokenize combined prompt+response for chosen
            full_chosen = f"{prompt_text}\n{chosen_text}"
            c_ids = self.tokenizer.encode(full_chosen)[:512]
            chosen_tensor = torch.tensor([c_ids], dtype=torch.long, device=device)

            # Tokenize combined prompt+response for rejected
            full_rejected = f"{prompt_text}\n{rejected_text}"
            r_ids = self.tokenizer.encode(full_rejected)[:512]
            rejected_tensor = torch.tensor([r_ids], dtype=torch.long, device=device)

            return chosen_tensor, rejected_tensor
        except Exception as e:
            logger.debug("Failed to build DPO pair: %s", e)

        return None

    def _ensure_lora(self) -> None:
        """Inject LoRA if not already done."""
        if self._lora_injected:
            return
        try:
            from hbllm.modules.lora import LoRAManager

            injected = LoRAManager.inject(self.model, r=self.lora_r)
            logger.info(
                "LearnerNode injected LoRA (r=%d) into %d modules", self.lora_r, len(injected)
            )
            self._lora_injected = True
        except Exception as e:
            logger.warning("Could not inject LoRA: %s", e)

    def _ensure_optimizer(self) -> None:
        """Create optimizer targeting LoRA params only."""
        if self._optimizer is not None:
            return

        import torch

        lora_params = [
            p for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad
        ]
        if not lora_params:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]

        self._optimizer = torch.optim.AdamW(lora_params, lr=self.lr, weight_decay=0.01)

    def _save_adapter(self) -> None:
        """Save the trained LoRA adapter."""
        from pathlib import Path

        import torch

        try:
            from hbllm.modules.lora import LoRAManager

            adapter_state = LoRAManager.get_lora_state_dict(self.model)
            save_dir = Path(self.checkpoint_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "learner_lora_adapter.pt"
            torch.save(adapter_state, save_path)
            logger.info("LearnerNode saved LoRA adapter to %s", save_path)
        except Exception as e:
            logger.warning("Failed to save LoRA adapter: %s", e)
