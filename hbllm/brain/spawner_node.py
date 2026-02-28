import asyncio
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

    Listens for SPAWN_REQUEST logic from the RouterNode. When triggered,
    it generates a synthetic dataset for the unknown topic, initializes 
    a fresh LoRA adapter (simulated by a brief training delay), and dynamically
    instantiates and registers a new DomainModuleNode on the live bus.
    """

    def __init__(self, node_id: str, model: Any, tokenizer: Any):
        super().__init__(node_id=node_id, node_type=NodeType.ROUTER) # Reusing router/admin type
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
        """The core expansion logic."""
        domain_name = topic.replace(" ", "_").lower()
        new_node_id = f"domain_{domain_name}"
        
        logger.info("--- Self-Expansion Initiated for '%s' ---", domain_name)
        
        # 1. Generate Synthetic Data
        try:
            # Shift to thread to prevent blocking
            dataset_path = await asyncio.to_thread(
                self.synthesizer.generate_dataset,
                topic=topic,
                num_samples=10
            )
            logger.info("Generated synthetic data at: %s", dataset_path)
        except Exception as e:
            logger.error("Data synthesis failed for %s: %s", domain_name, e)
            return
            
        # 2. Simulate LoRA Initialization and Training
        logger.info("Initializing new LoRA adapter and fine-tuning (simulated 3s)...")
        await asyncio.sleep(3.0) 
        
        # 3. Create and Register the New Node
        try:
            logger.info("Wiring new DomainModuleNode ('%s') to the MessageBus...", new_node_id)
            new_module = DomainModuleNode(
                node_id=new_node_id,
                domain_name=domain_name,
                model=self.model,
                tokenizer=self.tokenizer
                # lora_state_dict=newly_trained_adapter
            )
            
            # Since the Spawner operates on the same local event loop as the chat CLI, 
            # we can inject it into the InProcessBus. In a distributed setup, this would
            # be a container orchestration call (e.g., Kubernetes Job).
            
            # Start the node
            await new_module.start(self.bus)
            
            # The node registers itself in its `on_start()` method, but for the 
            # InProcessBus we need the ServiceRegistry to know about it. The registry 
            # listens to NODE_REGISTERED events natively on the bus, so this is handled automatically!
            
            logger.info("--- Self-Expansion COMPLETE for '%s' ---", domain_name)
            
            # 4. Announce completion
            completion_msg = Message(
                type=MessageType.SPAWN_COMPLETE,
                source_node_id=self.node_id,
                target_node_id="",
                topic="system.spawn.complete",
                payload={"domain": domain_name, "status": "active"}
            )
            await self.bus.publish("system.spawn.complete", completion_msg)
            
        except Exception as e:
             logger.error("Failed to spawn DomainModuleNode %s: %s", new_node_id, e)
