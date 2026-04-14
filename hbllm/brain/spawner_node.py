import asyncio
import copy
import logging
import os
from typing import Any

from hbllm.data.synthesizer import DataSynthesizer
from hbllm.modules.base_module import DomainModuleNode
from hbllm.network.messages import Message, MessageType, SpawnRequestPayload
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)

# Domain complexity tiers for automatic LoRA rank selection
_HEAVY_DOMAINS: frozenset[str] = frozenset(
    {
        "medical",
        "medicine",
        "healthcare",
        "clinical",
        "diagnosis",
        "legal",
        "law",
        "compliance",
        "regulatory",
        "coding",
        "programming",
        "software",
        "engineering",
        "mathematics",
        "math",
        "statistics",
        "calculus",
        "science",
        "physics",
        "chemistry",
        "biology",
        "finance",
        "accounting",
        "trading",
        "economics",
    }
)

_MEDIUM_DOMAINS: frozenset[str] = frozenset(
    {
        "writing",
        "creative",
        "marketing",
        "education",
        "history",
        "geography",
        "philosophy",
        "psychology",
        "cooking",
        "nutrition",
        "fitness",
        "sports",
    }
)


def _classify_domain_rank(domain: str, default_rank: int = 8) -> int:
    """
    Automatically select LoRA rank based on domain complexity.

    Heavy domains (medical, legal, coding, math) get higher rank for
    deeper specialization. Light domains (style, tone) get lower rank.

    Supports hierarchical sub-domains: ``coding.python`` matches ``coding``.

    Args:
        domain: The domain name to classify.
        default_rank: Fallback rank for unrecognized domains.

    Returns:
        Recommended LoRA rank (4, 8, 16, 32, or 64).
    """
    # Split on dots for sub-domain hierarchy, then also split on underscores/spaces
    domain_lower = domain.lower()
    tokens = set()
    for part in domain_lower.replace(".", " ").replace("_", " ").split():
        tokens.add(part)

    if tokens & _HEAVY_DOMAINS:
        return 32
    if tokens & _MEDIUM_DOMAINS:
        return 16
    return default_rank


class SpawnerNode(Node):
    """
    Self-Expansion Engine.

    Listens for SPAWN_REQUEST events from the RouterNode. When triggered,
    it generates a synthetic dataset for the unknown topic, trains a fresh
    LoRA adapter on that data, and dynamically instantiates and registers
    a new DomainModuleNode on the live bus.
    """

    def __init__(
        self,
        node_id: str,
        model: Any,
        tokenizer: Any,
        adapter_registry: Any | None = None,
        default_lora_rank: int | None = None,
    ):
        super().__init__(node_id=node_id, node_type=NodeType.SPAWNER)
        self.model = model
        self.tokenizer = tokenizer
        self.synthesizer = DataSynthesizer(model=model, tokenizer=tokenizer)
        self.spawning_tasks: set[asyncio.Task[Any]] = set()
        self.adapter_registry = adapter_registry
        # Priority: constructor arg > env var > auto-classify per domain
        self.default_lora_rank = (
            default_lora_rank or int(os.environ.get("HBLLM_LORA_RANK", "0")) or None
        )

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

        # Resolve LoRA rank: request override > global default > auto-classify
        lora_rank = (
            payload.lora_rank or self.default_lora_rank or _classify_domain_rank(payload.topic)
        )
        logger.info("Selected LoRA rank r=%d for domain '%s'", lora_rank, payload.topic)

        # Launch the spawn process in the background so we don't block the bus
        task = asyncio.create_task(
            self._spawn_new_module(payload.topic, message.tenant_id, lora_rank)
        )
        self.spawning_tasks.add(task)
        task.add_done_callback(self.spawning_tasks.discard)

        return None

    async def _spawn_new_module(
        self, topic: str, tenant_id: str | None = None, lora_rank: int = 8
    ) -> None:
        """The core expansion logic: resolve adapter → (or synthesize + train) → register module."""
        # Support hierarchical sub-domains (e.g., "coding.python" stays as-is)
        domain_name = topic.replace(" ", "_").lower()
        if tenant_id and tenant_id != "system":
            domain_name = f"{tenant_id}_{domain_name}"
        new_node_id = f"domain_{domain_name.replace('.', '_')}"

        logger.info("--- Self-Expansion Initiated for '%s' ---", domain_name)

        # Step 1: Try to resolve a pre-trained adapter from the registry
        lora_state_dict = None
        if self.adapter_registry and self.adapter_registry.enabled:
            try:
                lora_state_dict = await self.adapter_registry.resolve(domain_name)
                if lora_state_dict:
                    logger.info(
                        "Resolved pre-trained adapter for '%s' from registry, skipping training",
                        domain_name,
                    )
            except Exception as e:
                logger.warning("Adapter registry lookup failed for '%s': %s", domain_name, e)

        # Step 2: Fall back to synthesize + train if no adapter was found
        if lora_state_dict is None:
            try:
                dataset_path = await asyncio.to_thread(
                    self.synthesizer.generate_dataset, topic=topic, num_samples=10
                )
                logger.info("Generated synthetic data at: %s", dataset_path)
            except Exception as e:
                logger.error("Data synthesis failed for %s: %s", domain_name, e)
                return

            lora_state_dict = await self._train_lora_adapter(
                domain_name, dataset_path, tenant_id, lora_rank
            )

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

            # 4. Announce completion with centroid_text for Router
            completion_msg = Message(
                type=MessageType.SPAWN_COMPLETE,
                source_node_id=self.node_id,
                target_node_id="",
                topic="system.spawn.complete",
                payload={
                    "domain": domain_name,
                    "status": "active",
                    "has_lora": lora_state_dict is not None,
                },
            )
            await self.bus.publish("system.spawn.complete", completion_msg)

            # 5. Notify Router to register the new domain with centroid
            centroid_text = f"Topics relating to {domain_name.replace('.', ' ').replace('_', ' ')}"
            domain_msg = Message(
                type=MessageType.EVENT,
                source_node_id=self.node_id,
                target_node_id="",
                topic="system.domain_registered",
                payload={
                    "domain": domain_name,
                    "centroid_text": centroid_text,
                },
            )
            await self.bus.publish("system.domain_registered", domain_msg)

        except Exception as e:
            logger.error("Failed to spawn DomainModuleNode %s: %s", new_node_id, e)

    async def _train_lora_adapter(
        self,
        domain_name: str,
        dataset_path: str,
        tenant_id: str | None = None,
        lora_rank: int = 8,
    ) -> dict[str, Any] | None:
        """Train a LoRA adapter on the synthetic dataset."""
        logger.info("Training LoRA adapter for domain '%s' (rank=%d)...", domain_name, lora_rank)

        def _train() -> dict[str, Any] | None:
            try:
                import torch

                from hbllm.modules.lora import LoRAManager
                from hbllm.training.sft import InstructionDataset, collate_sft, load_sft_data

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
                    dataset,
                    batch_size=2,
                    shuffle=True,
                    collate_fn=collate_sft,
                )

                # Inject LoRA directly into the shared model (memory-efficient PEFT)
                import torch

                train_model = self.model
                injected = LoRAManager.inject(train_model, r=lora_rank)
                LoRAManager.add_adapter(train_model, domain_name)
                LoRAManager.set_active_adapter(train_model, domain_name)

                logger.info(
                    "Injected LoRA (r=%d) into %d modules directly",
                    lora_rank,
                    len(injected),
                )

                # Optimizer targeting only LoRA parameters
                lora_params = [
                    p for n, p in train_model.named_parameters() if "lora_" in n and p.requires_grad
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

                        loss.backward()  # type: ignore[no-untyped-call]
                        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        step += 1

                        if step % 5 == 0:
                            logger.info(
                                "  [Spawner] %s step %d/%d loss=%.4f",
                                domain_name,
                                step,
                                max_steps,
                                loss.item(),
                            )

                # Extract LoRA state dict
                adapter_state = LoRAManager.get_lora_state_dict(train_model)

                # Save adapter with hierarchical path support
                from pathlib import Path

                save_dir = Path(f"./checkpoints/domains/{domain_name}")
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / "lora_adapter.pt"
                torch.save(adapter_state, save_path)
                logger.info("Saved domain adapter: %s", save_path)

                # Apply strictly scoped adapter immediately if using PEFT
                if tenant_id and hasattr(train_model, "add_adapter"):
                    try:
                        self.model.add_adapter(tenant_id, adapter_state)
                    except Exception as adapter_err:
                        logger.warning(
                            "Failed to inject live adapter for tenant '%s': %s",
                            tenant_id,
                            adapter_err,
                        )

                # Reset active adapter context variable
                LoRAManager.set_active_adapter(train_model, None)

                return adapter_state

            except Exception as e:
                logger.warning("LoRA training failed for '%s': %s", domain_name, e)
                return None

        try:
            return await asyncio.to_thread(_train)
        except Exception as e:
            logger.warning("LoRA training thread failed: %s", e)
            return None
