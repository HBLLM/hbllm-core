"""
Interactive API / CLI chat interface for the Modular Brain.

Boots the MessageBus (InProcessBus) and instantiates all the nodes:
Router, Memory, Planner, and multiple DomainModules. It accepts user 
input, queries the Router, awaits the response, and displays it.
"""

import argparse
import asyncio
import logging
import sys
import uuid
from typing import Any

from hbllm.brain.planner_node import PlannerNode
from hbllm.brain.router_node import RouterNode
from hbllm.memory.memory_node import MemoryNode
from hbllm.modules.base_module import DomainModuleNode
from hbllm.brain.learner_node import LearnerNode
from hbllm.brain.spawner_node import SpawnerNode
from hbllm.brain.meta_node import MetaReasoningNode
from hbllm.perception.vision_node import VisionNode
from hbllm.perception.audio_in_node import AudioInputNode
from hbllm.perception.audio_out_node import AudioOutputNode
from hbllm.actions.execution_node import ExecutionNode
from hbllm.actions.browser_node import BrowserNode
from hbllm.actions.logic_node import LogicNode
from hbllm.actions.fuzzy_node import FuzzyNode
from hbllm.brain.workspace_node import WorkspaceNode
from hbllm.brain.world_model_node import WorldModelNode
from hbllm.brain.sleep_node import SleepCycleNode
from hbllm.brain.critic_node import CriticNode
from hbllm.actions.api_node import ApiNode
from hbllm.brain.decision_node import DecisionNode
from hbllm.brain.llm_interface import LLMInterface
from hbllm.network.bus import InProcessBus
from hbllm.network.redis_bus import RedisBus
from hbllm.network.messages import Message, MessageType, QueryPayload
from hbllm.network.registry import ServiceRegistry

# We use the transformer from Phase 1
from hbllm.model.config import get_config
from hbllm.model.transformer import HBLLMForCausalLM
# And the tokenizer
from hbllm_tokenizer_rs import Vocab

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chat")
# Silence some noisy loggers
logging.getLogger("hbllm.model").setLevel(logging.WARNING)


async def async_main(args: argparse.Namespace) -> None:
    print(f"Initializing HBLLM ({args.model_size}) Brain Interface...\n")

    # 1. Initialize Network Layer
    if args.bus == "redis":
        bus = RedisBus(redis_url=args.redis_url)
    else:
        bus = InProcessBus()
        
    registry = ServiceRegistry()
    await bus.start()

    # 2. Load the Model (Shared across domain nodes)
    logger.info("Loading base transformer model...")
    config = get_config(args.model_size)
    model = HBLLMForCausalLM(config)
    
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    logger.info("Loading tokenizer...")
    try:
        vocab = Vocab.from_file("test_workspace/vocab.json")
    except Exception:
        logger.error(
            "Could not load tokenizer from 'test_workspace/vocab.json'. "
            "Run 'python hbllm/cli.py data --samples 1000 --vocab-size 32768 --work-dir ./test_workspace' "
            "to generate it first."
        )
        sys.exit(1)

    # 2b. Create the shared LLM inference interface
    llm_interface = LLMInterface(model=model, tokenizer=vocab, device=device)

    # 3. Instantiate and Start Nodes
    nodes = [
        MemoryNode(node_id="memory_01", db_path="chat_memory.db"),
        RouterNode(node_id="router_01", llm=llm_interface),
        PlannerNode(node_id="planner_01"),
        LearnerNode(node_id="learner_01"),
        SpawnerNode(node_id="spawner_01", model=model, tokenizer=vocab),
        MetaReasoningNode(node_id="meta_01"),
        VisionNode(node_id="vision_01"),
        AudioInputNode(node_id="audio_in_01", model_size="tiny"),
        AudioOutputNode(node_id="audio_out_01"),
        ExecutionNode(node_id="exec_01"),
        BrowserNode(node_id="browser_01"),
        LogicNode(node_id="logic_01", llm=llm_interface),
        FuzzyNode(node_id="fuzzy_01", llm=llm_interface),
        WorkspaceNode(node_id="workspace_01"),
        WorldModelNode(node_id="world_model_01"),
        SleepCycleNode(node_id="sleep_01", idle_timeout_seconds=20.0),
        CriticNode(node_id="critic_01", llm=llm_interface),
        DecisionNode(node_id="decision_01", llm=llm_interface),
        ApiNode(node_id="api_01", llm=llm_interface),
        
        # Domain Modules — shared base model with optional per-domain LoRA adapters
        DomainModuleNode(
            node_id="domain_general", 
            domain_name="general", 
            model=model, 
            tokenizer=vocab,
            lora_state_dict=None 
        ),
        DomainModuleNode(
            node_id="domain_coding", 
            domain_name="coding", 
            model=model, 
            tokenizer=vocab,
            lora_state_dict=None
        ),
        DomainModuleNode(
            node_id="domain_math", 
            domain_name="math", 
            model=model, 
            tokenizer=vocab,
            lora_state_dict=None
        ),
    ]

    for node in nodes:
        await registry.register(node.get_info())
        await node.start(bus)
        
    print("\n--- All Nodes Online. Type 'quit' to exit. ---\n")

    session_id = str(uuid.uuid4())
    user_node_id = "user_cli"

    # Async queue to print out-of-band system messages (like SPAWN_COMPLETE)
    system_messages = asyncio.Queue()
    
    async def spawn_complete_handler(msg: Message) -> Message | None:
        if msg.type == MessageType.SPAWN_COMPLETE:
            domain = msg.payload.get("domain", "unknown")
            await system_messages.put(f"\n[SYSTEM ALERT]: New Domain Module '{domain}' is now ONLINE and ready for queries!\n")
        return None
        
    async def system_improve_handler(msg: Message) -> Message | None:
        if msg.type == MessageType.SYSTEM_IMPROVE:
            domain = msg.payload.get("domain", "unknown")
            reasoning = msg.payload.get("reasoning", "")
            await system_messages.put(f"\n[AGI ALERT]: Meta-Reasoning detected weakness in domain '{domain.upper()}'.\nReason: {reasoning}\nTriggering offline enhancement dataset dump...\n")
        return None
    
    await bus.subscribe("system.spawn.complete", spawn_complete_handler)
    await bus.subscribe("system.improve", system_improve_handler)

    # Global Output Listener
    # Because of the new Workspace -> Decision architecture, the CLI should passively 
    # listen for `sensory.output` events instead of blocking synchronously on requests.
    async def listen_for_output():
        async def on_output(message: Message):
            if message.payload:
                text = message.payload.get("text")
                source = message.payload.get("source", "system")
                if text:
                    print(f"\n[{source.upper()}] > {text}\n> ", end="", flush=True)
            return None
        await bus.subscribe("sensory.output", on_output)
        
    asyncio.create_task(listen_for_output())

    print("\nHBLLM Global Workspace Chat CLI")
    
    # Chat Loop
    while True:
        try:
            # Check for background system alerts
            while not system_messages.empty():
                alert = await system_messages.get()
                print(alert)
                
            # We must use a thread to prevent input() from blocking the asyncio event loop!
            user_input = await asyncio.to_thread(input, "User> ")
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input.strip():
                continue

            if user_input.lower().startswith("/image"):
                parts = user_input.split(" ", 2)
                if len(parts) >= 2:
                    image_path = parts[1]
                    text_query = parts[2] if len(parts) > 2 else "Describe this image."
                    print(f"\n[Vision] Analyzing '{image_path}'...")
                    
                    try:
                        vision_msg = Message(
                            type=MessageType.QUERY,
                            source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                            topic="vision.process",
                            payload={"image_path": image_path}
                        )
                        vision_resp = await bus.request("vision.process", vision_msg, timeout=45.0)
                        
                        if vision_resp.type == MessageType.ERROR:
                            print(f"[Vision Error]: {vision_resp.payload.get('error')}\n")
                            continue
                            
                        caption = vision_resp.payload.get("text", "")
                        print(f"[Vision Extracted Semantic Caption]: {caption}\n")
                        
                        # Augment the user input seamlessly so the router and experts know what they are looking at
                        user_input = f"<image_caption>{caption}</image_caption>\n{text_query}"
                    except TimeoutError:
                        print("[Vision Error]: Timed out waiting for VisionNode.\n")
                        continue
                    except Exception as e:
                        print(f"[Vision Error]: {e}\n")
                        continue
                else:
                    print("Usage: /image <path> <optional question>")
                    continue

            if user_input.lower().startswith("/fuzzy"):
                query = user_input[6:].strip()
                if not query:
                    print("Usage: /fuzzy <service was somewhat poor but food was very delicious. tip?>")
                    continue
                    
                msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                    topic="workspace.update",
                    payload={"text": query, "intent": "fuzzy"}
                )
                await bus.publish("workspace.update", msg)
                continue

            if user_input.lower().startswith("/logic"):
                query = user_input[6:].strip()
                if not query:
                    print("Usage: /logic <The dog is twice as heavy as the cat. Together 30kg. How heavy is dog?>")
                    continue
                    
                msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                    topic="workspace.update",
                    payload={"text": query, "intent": "deduction"}
                )
                await bus.publish("workspace.update", msg)
                continue

            if user_input.lower().startswith("/fuzzy"):
                query = user_input[6:].strip()
                if not query:
                    print("Usage: /fuzzy <service was somewhat poor but food was very delicious. tip?>")
                    continue
                    
                msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                    topic="workspace.update",
                    payload={"text": query, "intent": "fuzzy"}
                )
                await bus.publish("workspace.update", msg)
                continue

            if user_input.lower().startswith("/logic"):
                query = user_input[6:].strip()
                if not query:
                    print("Usage: /logic <The dog is twice as heavy as the cat. Together 30kg. How heavy is dog?>")
                    continue
                    
                msg = Message(
                    type=MessageType.QUERY,
                    source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                    topic="workspace.update",
                    payload={"text": query, "intent": "deduction"}
                )
                await bus.publish("workspace.update", msg)
                continue

            if user_input.lower().startswith("/search"):
                query = user_input[8:].strip()
                if not query:
                    print("Usage: /search <query>")
                    continue
                
                print(f"\n[BrowserNode] Executing web search for: '{query}'...")
                try:
                    search_msg = Message(
                        type=MessageType.QUERY,
                        source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                        topic="task.execute.search",
                        payload={"query": query, "max_results": 1}
                    )
                    # We give the browser up to 15 seconds to fetch and scrape the page
                    search_resp = await bus.request("task.execute.search", search_msg, timeout=15.0)
                    
                    if search_resp.type == MessageType.ERROR:
                        print(f"[Search Error]: {search_resp.payload.get('error')}\n")
                        continue
                        
                    print(f"\n{search_resp.payload.get('text', '')}\n")
                except TimeoutError:
                    print("[Search Error]: Timed out waiting for BrowserNode.\n")
                except Exception as e:
                    print(f"[Search Error]: {e}\n")
                continue

            if user_input.lower().startswith("/exec"):
                code = user_input[5:].strip()
                if not code:
                    print("Usage: /exec <python_code>")
                    continue
                
                print("\n[ExecutionNode] Running script in sandbox...")
                try:
                    exec_msg = Message(
                        type=MessageType.QUERY,
                        source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                        topic="task.execute.python",
                        payload={"code": code}
                    )
                    exec_resp = await bus.request("task.execute.python", exec_msg, timeout=15.0)
                    
                    if exec_resp.type == MessageType.ERROR:
                        print(f"[Execution Error]: {exec_resp.payload.get('error')}\n")
                        continue
                        
                    stdout = exec_resp.payload.get("stdout", "")
                    stderr = exec_resp.payload.get("stderr", "")
                    
                    print(f"[Execution Output]\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
                    
                    # Store context seamlessly
                    user_input = f"<executed_code>{code}</executed_code>\n<stdout>{stdout}</stdout>\n<stderr>{stderr}</stderr>\nI executed the code above."
                except TimeoutError:
                    print("[Execution Error]: Timed out waiting for ExecutionNode.\n")
                    continue
                except Exception as e:
                    print(f"[Execution Error]: {e}\n")
                    continue

            if user_input.lower().startswith("/audio"):
                text = user_input[6:].strip()
                if not text:
                    print("Usage: /audio <text_to_speak>")
                    continue
                
                print(f"[AudioOutput] Synthesizing speech for: '{text}'...")
                
                # Dispatch to TTS AudioOutputNode
                audio_msg = Message(
                    type=MessageType.EVENT,
                    source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                    topic="sensory.audio.out",
                    payload={"text": text}
                )
                await bus.publish("sensory.audio.out", audio_msg)
                print("[AudioOutput] Dispatched to TTS engine.\n")
                continue

            if user_input.lower().startswith("/recall"):
                query = user_input[7:].strip()
                if not query:
                    print("Usage: /recall <search_query>")
                    continue
                
                print(f"\n[Semantic Memory] Searching vectors for: '{query}'...")
                try:
                    search_msg = Message(
                        type=MessageType.QUERY,
                        source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                        topic="memory.search",
                        payload={"query_text": query, "limit": 3}
                    )
                    search_resp = await bus.request("memory.search", search_msg, timeout=10.0)
                    
                    if search_resp.type == MessageType.ERROR:
                        print(f"[Memory Error]: {search_resp.payload.get('error')}\n")
                        continue
                        
                    results = search_resp.payload.get("results", [])
                    if not results:
                        print("[Semantic Memory] No relevant memories found.\n")
                    else:
                        print("[Semantic Memory] Top Matches:")
                        for idx, res in enumerate(results):
                            score = res.get("score", 0.0)
                            content = res.get("content", "").replace('\n', ' ')
                            print(f"  {idx+1}. (Score: {score:.2f}) {content[:100]}...")
                        print()
                except TimeoutError:
                    print("[Memory Error]: Timed out waiting for MemoryNode.\n")
                except Exception as e:
                    print(f"[Memory Error]: {e}\n")
                continue

            # 1. Store user message in memory
            store_msg = Message(
                type=MessageType.QUERY,
                source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                topic="memory.store",
                payload={"session_id": session_id, "role": "user", "content": user_input}
            )
            await bus.request("memory.store", store_msg, timeout=5.0)

            # 2. Retrieve recent history for context
            hist_msg = Message(
                type=MessageType.QUERY,
                source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                topic="memory.retrieve_recent",
                payload={"session_id": session_id, "limit": 5}
            )
            hist_resp = await bus.request("memory.retrieve_recent", hist_msg, timeout=5.0)
            history = hist_resp.payload.get("turns", [])

            # 3. Send query to Router
            # The Router will now push it to the `WorkspaceNode` instead of 
            # fulfilling it directly, so we just publish and let the async output listener handle it.
            query_msg = Message(
                type=MessageType.QUERY,
                source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                topic="router.query",
                payload=QueryPayload(text=user_input, context=history).model_dump()
            )
            
            await bus.publish("router.query", query_msg)
            
            # Note: We no longer block on `bus.request` waiting for the Router.
            # The `DecisionNode` will inherently publish to `sensory.output` when 
            # the cognitive modules reach consensus.
            
            # Optional feedback for Continuous Learning
            try:
                await asyncio.sleep(0.1) # Let logs clear
                rating_str = await asyncio.to_thread(input, "Rate this response? (+1/0/-1 or Enter to skip): ")
                rating_str = rating_str.strip()
                if rating_str and rating_str in ("+1", "1", "0", "-1"):
                    feedback_payload = {
                        "message_id": query_msg.id, # Using the original query ID since we don't have a synchronous response ID anymore
                        "rating": int(rating_str.replace("+", "")),
                        "prompt": user_input,
                        "response": "ASYNC_WORKSPACE_EVALUATION", # We don't have the text here since it was pushed to sensory.output asynchronously
                        "module_id": "global_workspace"
                    }
                    feedback_msg = Message(
                        type=MessageType.FEEDBACK,
                        source_node_id=user_node_id,
                tenant_id="local_user",
                session_id=session_id,
                        target_node_id="",
                        topic="system.feedback",
                        payload=feedback_payload
                    )
                    await bus.publish("system.feedback", feedback_msg)
            except Exception as e:
                logger.error("Feedback error: %s", e)
            
            # 4. Store assistant reply in memory
            # (In a production system, this would be moved into DecisionNode so memory is only committed to once actual action was taken)
            
        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except TimeoutError:
            print("\n[Error: System timed out waiting for a response]\n")
        except Exception as e:
            print(f"\n[Unexpected Error: {e}]\n")

    # Clean up
    for node in nodes:
        await node.stop()


def main():
    parser = argparse.ArgumentParser(description="HBLLM Distributed Brain CLI")
    parser.add_argument("--model-size", type=str, default="125m", choices=["125m", "500m", "1.5b"], help="Base model size to load")
    parser.add_argument("--bus", type=str, default="in_process", choices=["in_process", "redis"], help="Message bus backend to use")
    parser.add_argument("--redis-url", type=str, default="redis://localhost:6379", help="Redis URL (if using --bus redis)")
    args = parser.parse_args()
    
    asyncio.run(async_main(args))

if __name__ == "__main__":
    main()
