"""
Interactive CLI chat interface powered by the HCIR Cognitive OS.

Boots ``build_cognitive_system()``, ``MessageBus``, ``MemoryNode``, and optional
action/perception nodes, routing user turns deterministically through HCIR dialogue.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid

from hbllm.actions.api_node import ApiNode
from hbllm.actions.browser_node import BrowserNode
from hbllm.actions.execution_node import ExecutionNode
from hbllm.actions.logic_node import LogicNode
from hbllm.memory.memory_node import MemoryNode
from hbllm.network.bus import InProcessBus, MessageBus
from hbllm.network.messages import Message, MessageType
from hbllm.network.redis_bus import RedisBus
from hbllm.network.registry import ServiceRegistry
from hbllm.perception.audio_in_node import AudioInputNode
from hbllm.perception.audio_out_node import AudioOutputNode
from hbllm.perception.vision_node import VisionNode
from hbllm.runtime.bootstrap import CognitiveSystemConfig, build_cognitive_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chat")
logging.getLogger("hbllm.model").setLevel(logging.WARNING)


async def async_main(args: argparse.Namespace) -> None:
    print(f"Initializing HBLLM HCIR Chat Interface ({args.model_size})...\n")

    bus: MessageBus
    if args.bus == "redis":
        bus = RedisBus(redis_url=args.redis_url)
    else:
        bus = InProcessBus()

    registry = ServiceRegistry()
    await bus.start()

    cog_sys = build_cognitive_system(
        CognitiveSystemConfig(world_id="chat_cli", enable_autonomous_curiosity=True)
    )
    if cog_sys.runtime is not None:
        await cog_sys.runtime.start()

    nodes = [
        MemoryNode(node_id="memory_01", db_path="chat_memory.db"),
        VisionNode(node_id="vision_01"),
        AudioInputNode(node_id="audio_in_01", model_size="tiny"),
        AudioOutputNode(node_id="audio_out_01"),
        ExecutionNode(node_id="exec_01"),
        BrowserNode(node_id="browser_01"),
        LogicNode(node_id="logic_01"),
        ApiNode(node_id="api_01"),
    ]
    for node in nodes:
        await registry.register(node.get_info())
        await node.start(bus)

    print("\n--- HCIR Cognitive OS Online. Type 'quit' to exit. ---\n")

    session_id = str(uuid.uuid4())
    user_node_id = "user_cli"

    while True:
        try:
            user_input = await asyncio.to_thread(input, "User> ")
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input.strip():
                continue

            if user_input.lower().startswith("/search "):
                query = user_input[8:].strip()
                resp = await bus.request(
                    "task.execute.search",
                    Message(
                        type=MessageType.QUERY,
                        source_node_id=user_node_id,
                        tenant_id="local_user",
                        session_id=session_id,
                        topic="task.execute.search",
                        payload={"query": query, "max_results": 1},
                    ),
                    timeout=15.0,
                )
                print(f"\n{resp.payload.get('text', '')}\n")
                continue

            if user_input.lower().startswith("/exec "):
                code = user_input[6:].strip()
                resp = await bus.request(
                    "task.execute.python",
                    Message(
                        type=MessageType.QUERY,
                        source_node_id=user_node_id,
                        tenant_id="local_user",
                        session_id=session_id,
                        topic="task.execute.python",
                        payload={"code": code},
                    ),
                    timeout=15.0,
                )
                print(f"\nSTDOUT:\n{resp.payload.get('stdout', '')}\n")
                continue

            await bus.request(
                "memory.store",
                Message(
                    type=MessageType.QUERY,
                    source_node_id=user_node_id,
                    tenant_id="local_user",
                    session_id=session_id,
                    topic="memory.store",
                    payload={"session_id": session_id, "role": "user", "content": user_input},
                ),
                timeout=5.0,
            )

            turn_result = await asyncio.to_thread(
                cog_sys.process_dialogue, user_input, speaker_id=user_node_id
            )
            response_text = str(getattr(turn_result, "response_text", "") or "(no response)")
            print(f"\n[HCIR] > {response_text}\n")

            await bus.request(
                "memory.store",
                Message(
                    type=MessageType.QUERY,
                    source_node_id="hcir",
                    tenant_id="local_user",
                    session_id=session_id,
                    topic="memory.store",
                    payload={
                        "session_id": session_id,
                        "role": "assistant",
                        "content": response_text,
                    },
                ),
                timeout=5.0,
            )
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as exc:
            print(f"\n[Error]: {exc}\n")

    for node in nodes:
        await node.stop()
    if cog_sys.runtime is not None:
        await cog_sys.runtime.stop()
    await bus.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="HBLLM HCIR Chat CLI")
    parser.add_argument("--model-size", type=str, default="125m", choices=["125m", "500m", "1.5b"])
    parser.add_argument("--bus", type=str, default="in_process", choices=["in_process", "redis"])
    parser.add_argument(
        "--redis-url",
        type=str,
        default=os.getenv("HBLLM_REDIS_URL", "redis://localhost:6379"),
    )
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
