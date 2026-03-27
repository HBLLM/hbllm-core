import asyncio
import copy
import logging
from typing import Any

from hbllm.network.node import Node, NodeType
from hbllm.network.messages import Message, MessageType, SpawnRequestPayload
from hbllm.modules.base_module import DomainModuleNode
from hbllm.data.synthesizer import DataSynthesizer

logger = logging.getLogger(__name__)

class SpawnerNode(Node):
    """
    Self-Expansion Engine.

    Listens for SPAWN_REQUEST events from the RouterNode. When triggered,
    it generates a synthetic dataset for the unknown topic, trains a fresh
    LoRA adapter on that data, and dynamically instantiates and registers
    a new DomainModuleNode on the live bus.
    """

    def __init__(self, node_id: str, model: Any, tokenizer: Any):
        super().__init__(node_id=node_id, node_type=NodeType.SPAWNER)
        self.model = model
        self.tokenizer = tokenizer
        self.synthesizer = DataSynthesizer(model=model, tokenizer=tokenizer)
        self.spawning_tasks = set()

    async def on_start(self) -> None:
        """Subscribe to spawn requests."""
        logger.info("Starting SpawnerNode '%s'", self.node_id)
        await self.bus.subscribe("system.spawn", self.handle_message)

    async def on_stop(self) -> None:
        """Clean up background tasks."""
        logger.info("Stopping SpawnerNode")
        for task in list(self.spawning_tasks):
            task.cancel()

    async def handle_message(self, message: Message) -> Message | None:
        """Handle incoming spawn requests."""
        if message.type != MessageType.SPAWN_REQUEST:
            return None
            
        try:
            payload = SpawnRequestPayload(**message.payload)
        except Exception as e:
            return message.create_error(f"Invalid SpawnRequestPayload: {e}")
            
        logger.info("Spawner received request to expand into domain: '%s'", payload.topic)

        # Launch the spawn process in the background so we don't block the bus
        task = asyncio.create_task(self._spawn_new_module(payload.topic))
        self.spawning_tasks.add(task)
        task.add_done_callback(self.spawning_tasks.discard)
        
        return None

    async def _spawn_new_module(self, topic: str) -> None:
        """The core expansion logic: synthesize data → train LoRA → register module."""
        domain_name = topic.replace(" ", "_").lower()
        new_node_id = f"domain_{domain_name}"
        
        logger.info("--- Self-Expansion Initiated for '%s' ---", domain_name)
        
        # 1. Generate Synthetic Data
        try:
            dataset_path = await asyncio.to_thread(
                self.synthesizer.generate_dataset,
                topic=topic,
                num_samples=10
            )
            logger.info("Generated synthetic data at: %s", dataset_path)
        except Exception as e:
            logger.error("Data synthesis failed for %s: %s", domain_name, e)
            return
            
        # 2. Train LoRA Adapter on Synthetic Data
        lora_state_dict = await self._train_lora_adapter(domain_name, dataset_path)
        
        # 3. Create and Register the New Node
        try:
            logger.info("Wiring new DomainModuleNode ('%s') to the MessageBus...", new_node_id)
            new_module = DomainModuleNode(
                node_id=new_node_id,
                domain_name=domain_name,
                model=self.model,
                tokenizer=self.tokenizer,
                lora_state_dict=lora_state_dict,
            )
            
            # Start the node — it registers itself on the bus
            await new_module.start(self.bus)
            
            logger.info("--- Self-Expansion COMPLETE for '%s' ---", domain_name)
            
            # 4. Announce completion
            completion_msg = Message(
                type=MessageType.SPAWN_COMPLETE,
                source_node_id=self.node_id,
                target_node_id="",
                topic="system.spawn.complete",
                payload={
                    "domain": domain_name,
                    "status": "active",
                    "has_lora": lora_state_dict is not None,
                }
            )
            await self.bus.publish("system.spawn.complete", completion_msg)
            
        except Exception as e:
             logger.error("Failed to spawn DomainModuleNode %s: %s", new_node_id, e)

    async def _train_lora_adapter(
        self, domain_name: str, dataset_path: str
    ) -> dict | None:
        """Train a LoRA adapter on the synthetic dataset."""
        logger.info("Training LoRA adapter for domain '%s'...", domain_name)

        def _train():
            try:
                import torch
                from hbllm.modules.lora import LoRAManager
                from hbllm.training.sft import load_sft_data, InstructionDataset, collate_sft

                # Load the synthetic dataset
                raw_data = load_sft_data(dataset_path)
                if not raw_data:
                    logger.warning("No training data loaded from %s", dataset_path)
                    return None

                dataset = InstructionDataset(raw_data, self.tokenizer, max_length=256)
                if len(dataset) == 0:
                    logger.warning("Empty dataset for domain '%s'", domain_name)
                    return None

                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=2, shuffle=True, collate_fn=collate_sft,
                )

                # Inject LoRA into a deep copy to avoid mutating the shared model
                import torch
                train_model = copy.deepcopy(self.model)
                injected = LoRAManager.inject(train_model, r=8)
                logger.info("Injected LoRA into %d modules (on isolated copy)", len(injected))

                # Optimizer targeting only LoRA parameters
                lora_params = [
                    p for n, p in train_model.named_parameters()
                    if "lora_" in n and p.requires_grad
                ]
                if not lora_params:
                    logger.warning("No LoRA parameters found. Skipping training.")
                    return None

                optimizer = torch.optim.AdamW(lora_params, lr=1e-4, weight_decay=0.01)
                device = next(train_model.parameters()).device
                train_model.train()

                max_steps = min(20, len(loader) * 2)  # Quick training for domain bootstrap
                step = 0

                for epoch in range(2):
                    for batch in loader:
                        if step >= max_steps:
                            break

                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)

                        output = train_model(input_ids)
                        logits = output["logits"] if isinstance(output, dict) else output

                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, logits.size(-1)),
                            labels[:, 1:].reshape(-1),
                            ignore_index=-100,
                        )

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        step += 1

                        if step % 5 == 0:
                            logger.info(
                                "  [Spawner] %s step %d/%d loss=%.4f",
                                domain_name, step, max_steps, loss.item(),
                            )

                # Extract LoRA state dict
                adapter_state = LoRAManager.get_lora_state_dict(train_model)

                # Save adapter
                from pathlib import Path
                save_dir = Path(f"./checkpoints/domains/{domain_name}")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / "lora_adapter.pt"
                torch.save(adapter_state, save_path)
                logger.info("Saved domain adapter: %s", save_path)

                return adapter_state

            except Exception as e:
                logger.warning("LoRA training failed for '%s': %s", domain_name, e)
                return None

        try:
            return await asyncio.to_thread(_train)
        except Exception as e:
            logger.warning("LoRA training thread failed: %s", e)
            return None

