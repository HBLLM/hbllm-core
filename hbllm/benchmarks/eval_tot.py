import asyncio
import time
from typing import Any

import yaml  # type: ignore[import-untyped]

from hbllm.brain.planner_node import PlannerNode
from hbllm.brain.process_reward_node import ProcessRewardNode
from hbllm.network.bus import InProcessBus


async def run_tot_benchmark() -> dict[str, Any]:
    bus = InProcessBus()
    await bus.start()

    # Shared nodes
    prm_node = ProcessRewardNode(node_id="prm")
    planner = PlannerNode(node_id="tot_planner")

    # Mock models for speed
    from hbllm.model.config import get_config
    from hbllm.model.transformer import HBLLMForProcessReward

    config = get_config("125m")
    config.vocab_size = 65536
    config.hidden_size = 32
    config.num_layers = 1
    config.num_attention_heads = 2
    config.num_kv_heads = 2

    prm_node.prm_model = HBLLMForProcessReward(config)
    prm_node.prm_is_trained = True

    # Needs a local provider mock for the planner
    class MockProvider:
        def __init__(self) -> None:
            self.name = "local"

        async def generate(self, messages: Any, **kwargs: Any) -> Any:
            from hbllm.serving.provider import LLMResponse

            return LLMResponse(
                content="Action: <tool_call><name>math</name><args>{'expr':'2+2'}</args></tool_call>",
                model="mock",
                usage={},
            )

    from typing import Any, cast

    planner.llm = cast(Any, MockProvider())

    await prm_node.start(bus)
    await planner.start(bus)

    # We evaluate ToT latency for a simple objective
    # (By setting max_budget low, we measure overhead of MCTS structure)
    planner.max_budget = 10  # type: ignore[attr-defined]
    planner.batch_size = 5  # type: ignore[attr-defined]

    start_time = time.time()
    task = asyncio.create_task(planner.execute_objective("Calculate the square root of 25."))  # type: ignore[attr-defined]

    # Wait for completion
    try:
        await asyncio.wait_for(task, timeout=15.0)
    except (TimeoutError, asyncio.TimeoutError):
        print("ToT execution timed out, measuring overhead to this point.")

    end_time = time.time()

    summary = planner.memory.summary()  # type: ignore[attr-defined]
    total_nodes = summary.get("total_nodes", 0)

    await planner.stop()
    await prm_node.stop()
    await bus.stop()

    latency = end_time - start_time

    return {
        "mcts_nodes_expanded": total_nodes,
        "nodes_per_sec": total_nodes / latency if latency > 0 else 0,
        "total_latency_sec": latency,
    }


if __name__ == "__main__":
    results = asyncio.run(run_tot_benchmark())
    print(yaml.dump(results))
