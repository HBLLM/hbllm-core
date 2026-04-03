import asyncio
import time
from typing import Any

from hbllm.brain.process_reward_node import ProcessRewardNode
from hbllm.network.bus import InProcessBus
from hbllm.network.messages import Message, MessageType


async def run_prm_benchmark() -> dict[str, Any]:
    bus = InProcessBus()
    await bus.start()

    prm_node = ProcessRewardNode(node_id="test_prm")
    await prm_node.start(bus)

    # Mock trained state for pure inference speed benchmarking
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForProcessReward

    config = get_config("125m")
    config.vocab_size = 65536
    config.hidden_size = 32
    config.num_layers = 2
    config.num_attention_heads = 2
    prm_node.prm_model = HBLLMForProcessReward(config)
    prm_node.prm_is_trained = True

    prompts = [
        "To sort a list of integers, I will first...",
        "I'm not exactly sure how to solve the problem, maybe...",
        "The fastest sorting algorithm in python is typically built-in TimSort. Let me demonstrate.",
        "To find the sum of all elements, we simply subtract them.",
    ]

    start_time = time.time()
    results = []

    # Run 100 evaluations to benchmark throughput
    for _ in range(25):
        for text in prompts:
            req = Message(
                type=MessageType.QUERY,
                source_node_id="benchmarker",
                topic="action.score_thought",
                payload={"content": text},
            )
            # Evaluate using the bus request
            try:
                msg_response = await bus.request("action.score_thought", req, timeout=5.0)
                if msg_response and hasattr(msg_response, "payload"):
                    results.append(msg_response.payload.get("score", 0.0))
            except Exception as e:
                print(f"Error: {e}")

    end_time = time.time()
    await prm_node.stop()
    await bus.stop()

    latency = end_time - start_time
    return {
        "throughput_evals_per_sec": len(results) / latency,
        "total_time_sec": latency,
        "total_evals": len(results),
    }


if __name__ == "__main__":
    import yaml

    results = asyncio.run(run_prm_benchmark())
    print(yaml.dump(results))
