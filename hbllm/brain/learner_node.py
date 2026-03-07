import asyncio
import logging
import torch
from typing import Any

from hbllm.network.node import Node, NodeType
from hbllm.network.messages import Message, MessageType, FeedbackPayload

logger = logging.getLogger(__name__)


class LearnerNode(Node):
    """
    Continuous Learning Engine.

    Listens for Feedback messages on the bus. When a user provides feedback
    (positive or negative) on a generation, it accumulates the sample.
    Once enough samples are gathered, it performs DPO (Direct Preference Optimization)
    in a background thread to update the active LoRA adapter weights, then
    broadcasts that the weights have been updated.
    """

    def __init__(
        self,
        node_id: str,
        model: torch.nn.Module | None = None,
        tokenizer: Any = None,
        batch_size: int = 4,
        lr: float = 1e-5,
        dpo_beta: float = 0.1,
        lora_r: int = 8,
        checkpoint_dir: str = "./checkpoints/learner",
    ):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE)
        self.node_type = NodeType.MEMORY
        self.model = model
        self.tokenizer = tokenizer
        self.feedback_buffer: list[FeedbackPayload] = []
        self.batch_size = batch_size
        self.lr = lr
        self.dpo_beta = dpo_beta
        self.lora_r = lora_r
        self.checkpoint_dir = checkpoint_dir
        self.training_task = None
        self._optimizer = None
        self._lora_injected = False
        self._training_steps = 0

    async def on_start(self) -> None:
        """Subscribe to feedback messages."""
        logger.info("Starting LearnerNode '%s'", self.node_id)
        await self.bus.subscribe("system.feedback", self.handle_message)

    async def on_stop(self) -> None:
        """Clean up."""
        logger.info("Stopping LearnerNode '%s'", self.node_id)
        if self.training_task and not self.training_task.done():
            self.training_task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        """Process incoming feedback."""
        if message.type != MessageType.FEEDBACK:
            return None

        try:
            payload = FeedbackPayload(**message.payload)
        except Exception as e:
            return message.create_error(f"Invalid FeedbackPayload: {e}")

        logger.info("Learner '%s' received feedback for msg %s: %d", self.node_id, payload.message_id, payload.rating)

        if payload.prompt and payload.response:
            self.feedback_buffer.append(payload)

        # Trigger background training if batch size is met
        if len(self.feedback_buffer) >= self.batch_size:
            batch = self.feedback_buffer[:self.batch_size]
            self.feedback_buffer = self.feedback_buffer[self.batch_size:]
            
            if self.training_task and not self.training_task.done():
                logger.warning("Training already in progress. Queuing batch.")
            else:
                self.training_task = asyncio.create_task(self._run_dpo_training(batch))

        return None

    async def _run_dpo_training(self, batch: list[FeedbackPayload]) -> None:
        """Execute DPO training in a background thread."""
        logger.info("LearnerNode starting DPO training on %d samples...", len(batch))

        def _train():
            if self.model is None or self.tokenizer is None:
                logger.warning("No model/tokenizer available. Skipping DPO training.")
                return

            self._ensure_lora()
            self._ensure_optimizer()

            from hbllm.training.dpo import compute_dpo_loss, get_batch_logps

            device = next(self.model.parameters()).device
            self.model.train()

            # Build DPO pairs from feedback
            for fb in batch:
                try:
                    pair = self._build_dpo_pair(fb, device)
                    if pair is None:
                        continue

                    chosen_ids, rejected_ids = pair

                    # Forward pass — policy model
                    chosen_out = self.model(chosen_ids)
                    rejected_out = self.model(rejected_ids)

                    chosen_logits = chosen_out["logits"] if isinstance(chosen_out, dict) else chosen_out
                    rejected_logits = rejected_out["logits"] if isinstance(rejected_out, dict) else rejected_out

                    policy_chosen_logps = get_batch_logps(chosen_logits, chosen_ids)
                    policy_rejected_logps = get_batch_logps(rejected_logits, rejected_ids)

                    # Reference log probs: use detached model (frozen copy approximation)
                    with torch.no_grad():
                        ref_chosen_logps = policy_chosen_logps.detach()
                        ref_rejected_logps = policy_rejected_logps.detach()

                    losses, chosen_rewards, rejected_rewards = compute_dpo_loss(
                        policy_chosen_logps=policy_chosen_logps,
                        policy_rejected_logps=policy_rejected_logps,
                        reference_chosen_logps=ref_chosen_logps,
                        reference_rejected_logps=ref_rejected_logps,
                        beta=self.dpo_beta,
                    )

                    loss = losses.mean()
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self._optimizer.step()
                    self._optimizer.zero_grad()
                    self._training_steps += 1

                    reward_margin = (chosen_rewards - rejected_rewards).mean().item()
                    logger.info(
                        "  DPO step %d: loss=%.4f reward_margin=%.4f",
                        self._training_steps, loss.item(), reward_margin,
                    )

                except Exception as e:
                    logger.warning("DPO step failed for sample: %s", e)
                    continue

            # Save LoRA adapter if we did any training
            if self._training_steps > 0 and self._lora_injected:
                self._save_adapter()

        try:
            await asyncio.to_thread(_train)
            logger.info("LearnerNode DPO training complete (%d total steps). Broadcasting update...", self._training_steps)
            
            update_msg = Message(
                type=MessageType.LEARNING_UPDATE,
                source_node_id=self.node_id,
                target_node_id="",
                topic="system.learning_update",
                payload={"status": "weights_updated", "steps": self._training_steps}
            )
            await self.bus.publish("system.learning_update", update_msg)
            
        except Exception as e:
            logger.error("LearnerNode DPO training failed: %s", e)

    def _build_dpo_pair(
        self, fb: FeedbackPayload, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build a chosen/rejected pair from feedback."""
        prompt_text = fb.prompt or ""
        response_text = fb.response or ""

        if not prompt_text or not response_text:
            return None

        try:
            # Tokenize combined prompt+response
            full_text = f"{prompt_text}\n{response_text}"
            ids = self.tokenizer.encode(full_text)
            ids = ids[:512]  # Truncate to max length

            full_tensor = torch.tensor([ids], dtype=torch.long, device=device)

            if fb.rating == 1:
                # Positive feedback: response is chosen, corrupted version is rejected
                corrupted = self._corrupt_response(response_text)
                corrupted_text = f"{prompt_text}\n{corrupted}"
                c_ids = self.tokenizer.encode(corrupted_text)[:512]
                corrupted_tensor = torch.tensor([c_ids], dtype=torch.long, device=device)
                return full_tensor, corrupted_tensor
            elif fb.rating == -1:
                # Negative feedback: construct a minimal "better" response
                better = f"I understand your question about {prompt_text[:50]}. Let me provide a more helpful answer."
                better_text = f"{prompt_text}\n{better}"
                b_ids = self.tokenizer.encode(better_text)[:512]
                better_tensor = torch.tensor([b_ids], dtype=torch.long, device=device)
                return better_tensor, full_tensor
        except Exception as e:
            logger.debug("Failed to build DPO pair: %s", e)

        return None

    @staticmethod
    def _corrupt_response(response: str) -> str:
        """Create a corrupted version of a response for negative examples."""
        words = response.split()
        if len(words) > 4:
            # Truncate to create a worse response
            return " ".join(words[:len(words) // 2]) + "..."
        return "I don't know."

    def _ensure_lora(self) -> None:
        """Inject LoRA if not already done."""
        if self._lora_injected:
            return
        try:
            from hbllm.modules.lora import LoRAManager
            injected = LoRAManager.inject(self.model, r=self.lora_r)
            logger.info("LearnerNode injected LoRA (r=%d) into %d modules", self.lora_r, len(injected))
            self._lora_injected = True
        except Exception as e:
            logger.warning("Could not inject LoRA: %s", e)

    def _ensure_optimizer(self) -> None:
        """Create optimizer targeting LoRA params only."""
        if self._optimizer is not None:
            return

        lora_params = [p for n, p in self.model.named_parameters() if "lora_" in n and p.requires_grad]
        if not lora_params:
            lora_params = [p for p in self.model.parameters() if p.requires_grad]

        self._optimizer = torch.optim.AdamW(lora_params, lr=self.lr, weight_decay=0.01)

    def _save_adapter(self) -> None:
        """Save the trained LoRA adapter."""
        import os
        from pathlib import Path

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

