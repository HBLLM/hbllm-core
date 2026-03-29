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
        batch_size: int = 4, # Used as limit per sleep cycle now
        lr: float = 1e-5,
        dpo_beta: float = 0.1,
        lora_r: int = 8,
        checkpoint_dir: str = "./checkpoints/learner",
    ):
        super().__init__(node_id=node_id, node_type=NodeType.DOMAIN_MODULE)
        self.node_type = NodeType.MEMORY
        self.model = model
        self.tokenizer = tokenizer
        
        # Continuous Lifelong DPO: persistent queue
        self.pending_pairs: dict[str, dict[str, str | None]] = {}
        self.queue_path = "workspace/reflection/dpo_queue.json"
        import os
        os.makedirs(os.path.dirname(self.queue_path), exist_ok=True)
        
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
        """Subscribe to feedback messages and sleep triggers."""
        logger.info("Starting LearnerNode '%s' in Continuous Overnight Mode", self.node_id)
        await self.bus.subscribe("system.feedback", self.handle_message)
        await self.bus.subscribe("system.sleep.dpo_trigger", self.handle_sleep_trigger)

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

        prompt = payload.prompt
        response = payload.response
        if prompt and response:
            if prompt not in self.pending_pairs:
                self.pending_pairs[prompt] = {"chosen": None, "rejected": None}
                
            if payload.rating == 1:
                self.pending_pairs[prompt]["chosen"] = response
            elif payload.rating == -1:
                self.pending_pairs[prompt]["rejected"] = response
                
            pair = self.pending_pairs[prompt]
            if pair["chosen"] and pair["rejected"]:
                import json, os
                
                # Append to persistent JSON array
                queue = []
                if os.path.exists(self.queue_path):
                    with open(self.queue_path, "r") as f:
                        try:
                            queue = json.load(f)
                        except json.JSONDecodeError:
                            queue = []
                            
                queue.append((prompt, pair["chosen"], pair["rejected"]))
                
                with open(self.queue_path, "w") as f:
                    json.dump(queue, f)
                    
                del self.pending_pairs[prompt]
                logger.info("LearnerNode queued contrastive DPO pair persistently for overnight training.")

        return None
        
    async def handle_sleep_trigger(self, message: Message) -> Message | None:
        """Triggered by SleepNode when user is idle for background DPO processing."""
        import json, os
        
        if not os.path.exists(self.queue_path):
            await self._broadcast_complete()
            return None
            
        with open(self.queue_path, "r") as f:
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
        target_batch = batch[:self.batch_size]
        re_queue = batch[self.batch_size:]
        
        if re_queue:
            with open(self.queue_path, "w") as f:
                json.dump(re_queue, f)
            
        self.training_task = asyncio.create_task(self._run_dpo_training(target_batch))
        return None

    async def _run_dpo_training(self, batch: list[tuple[str, str, str]]) -> None:
        """Execute DPO training in a background thread."""
        logger.info("LearnerNode starting DPO training on %d perfect pairs...", len(batch))

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
            for prompt_text, chosen_text, rejected_text in batch:
                try:
                    pair = self._build_dpo_pair(prompt_text, chosen_text, rejected_text, device)
                    if pair is None:
                        continue

                    chosen_ids, rejected_ids = pair

                    # ── Reference log-probs: base model with LoRA disabled ──
                    # Temporarily disable LoRA adapters to get the frozen
                    # reference model's log-probabilities. This ensures the
                    # DPO loss produces a non-zero gradient signal.
                    from hbllm.modules.lora import LoRAManager
                    LoRAManager.set_active_adapter(self.model, None)

                    with torch.no_grad():
                        ref_chosen_out = self.model(chosen_ids)
                        ref_rejected_out = self.model(rejected_ids)
                        ref_chosen_logits = ref_chosen_out["logits"] if isinstance(ref_chosen_out, dict) else ref_chosen_out
                        ref_rejected_logits = ref_rejected_out["logits"] if isinstance(ref_rejected_out, dict) else ref_rejected_out
                        ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_ids)
                        ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_ids)

                    LoRAManager.set_active_adapter(self.model, "default")

                    # ── Policy log-probs: model with LoRA active ──
                    # Forward pass — policy model
                    chosen_out = self.model(chosen_ids)
                    rejected_out = self.model(rejected_ids)

                    chosen_logits = chosen_out["logits"] if isinstance(chosen_out, dict) else chosen_out
                    rejected_logits = rejected_out["logits"] if isinstance(rejected_out, dict) else rejected_out

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
            await self._broadcast_complete()
            
        except Exception as e:
            logger.error("LearnerNode background DPO task failed: %s", e)
            await self._broadcast_complete()

    async def _broadcast_complete(self):
        update_msg = Message(
            type=MessageType.LEARNING_UPDATE,
            source_node_id=self.node_id,
            target_node_id="",
            topic="system.learning_update",
            payload={"status": "weights_updated", "steps": getattr(self, "_training_steps", 0)}
        )
        await self.bus.publish("system.learning_update", update_msg)

    def _build_dpo_pair(
        self, prompt_text: str, chosen_text: str, rejected_text: str, device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Build a chosen/rejected tensor pair."""
        if not prompt_text or not chosen_text or not rejected_text:
            return None

        try:
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

