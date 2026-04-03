"""
Process Reward Model Node.

Serves the Process Reward Model (PRM) evaluations over the network bus.
When a reasoning step (thought) is proposed in MCTS, this node evaluates
the text and returns a continuous score [0.0, 1.0] representing the
likelihood that the step is correct or leads to a correct answer.
"""

import logging
from pathlib import Path

import torch

from hbllm.model.config import get_config
from hbllm.model.tokenizer import HBLLMTokenizer
from hbllm.model.transformer import HBLLMForProcessReward
from hbllm.network.messages import Message
from hbllm.network.node import Node, NodeType

logger = logging.getLogger(__name__)


class ProcessRewardNode(Node):
    """
    Cognitive node that runs the continuous Process Reward Model.

    Subscribes to 'action.score_thought'.
    """

    def __init__(
        self,
        node_id: str,
        checkpoint_dir: str = "cognitive_checkpoints",
        model_name: str = "125m",
        device: str = "cpu",
        llm=None,  # Fallback LLM if PRM is not fully trained
    ):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.DOMAIN_MODULE,
            capabilities=["process_reward", "thought_evaluation"]
        )
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.prm_path = self.checkpoint_dir / "prm_adapter.pt"  # Or separate PRM weights
        self.llm = llm

        # Load PRM model
        config = get_config(model_name)
        self.prm_model = HBLLMForProcessReward(config)
        self.tokenizer = HBLLMTokenizer.from_tiktoken()  # Standard dummy tokenizer for tests or load real one
        self.prm_is_trained = False

    async def on_start(self) -> None:
        """Load the PRM weights and subscribe to the evaluation topic."""
        logger.info("Starting ProcessRewardNode")

        if self.prm_path.exists():
            try:
                state_dict = torch.load(self.prm_path, map_location=self.device, weights_only=True)
                self.prm_model.load_state_dict(state_dict, strict=False)
                self.prm_is_trained = True
                logger.info(f"[ProcessRewardNode] Loaded PRM weights from {self.prm_path}")
            except Exception as e:
                logger.warning(f"[ProcessRewardNode] Failed to load PRM weights: {e}")
        else:
            logger.info(f"[ProcessRewardNode] No PRM weights found at {self.prm_path}. Will use fallback heuristic/LLM.")

        self.prm_model.to(self.device)
        self.prm_model.eval()

        await self.bus.subscribe("action.score_thought", self.handle_score_request)

    async def on_stop(self) -> None:
        logger.info("Stopping ProcessRewardNode")

    async def handle_message(self, message: Message) -> Message | None:
        return None

    async def handle_score_request(self, message: Message) -> Message | None:
        """
        Evaluate a single reasoning step and return a continuous score [0.0, 1.0].
        """
        payload = message.payload
        thought_content = payload.get("content", "")
        if not thought_content:
            return message.create_response({"score": 0.5, "source": "empty"})

        score = await self.score_thought(thought_content)

        logger.debug(f"[ProcessRewardNode] Scored thought: {score:.3f}")
        return message.create_response({
            "score": score,
            "source": "prm_network" if self.prm_is_trained else "fallback"
        })

    async def score_thought(self, content: str) -> float:
        """Run the PRM model or fallback on the thought content."""
        if not self.prm_is_trained and self.llm:
            # Fallback to LLM if PRM isn't trained yet
            try:
                result = await self.llm.generate_json(
                    f"Evaluate the logical soundness and correctness of this reasoning step. "
                    f"Output a score between 0.0 and 1.0.\n\n"
                    f"Step: {content[:300]}\n\n"
                    f"Output JSON: {{\"score\": 0.0-1.0}}"
                )
                return float(result.get("score", 0.5))
            except Exception:
                pass

        # If PRM is "trained" (or we are forcing it for tests), run the neural net
        try:
            # Tokenize
            tokens = self.tokenizer.encode(content)
            # Add batch dimension and move to device
            input_ids = torch.tensor([tokens], device=self.device)

            with torch.no_grad():
                outputs = self.prm_model(input_ids)

            # Extract score from sigmoid output
            score: float = outputs["scores"].item()
            return min(max(score, 0.0), 1.0)
        except Exception as e:
            logger.error(f"[ProcessRewardNode] PRM inference failed: {e}")
            return 0.5
