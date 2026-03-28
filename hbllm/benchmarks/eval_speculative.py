import time
import torch
import yaml
from typing import Dict, Any

from hbllm.model.config import get_config
from hbllm.model.transformer import HBLLMForCausalLM
from hbllm.serving.provider import LocalProvider

def run_speculative_benchmark() -> Dict[str, Any]:
    # 1. Setup Models
    target_config = get_config("125m")
    target_config.vocab_size = 32000
    target_config.hidden_size = 256
    target_config.num_layers = 4
    target_config.num_attention_heads = 4
    target_config.num_kv_heads = 4
    
    draft_config = get_config("125m")
    draft_config.vocab_size = 32000
    draft_config.hidden_size = 64
    draft_config.num_layers = 1
    draft_config.num_attention_heads = 2
    draft_config.num_kv_heads = 2

    torch.manual_seed(42)
    target_model = HBLLMForCausalLM(target_config)
    draft_model = HBLLMForCausalLM(draft_config)

    target_model.eval()
    draft_model.eval()

    # Create dummy tokenizer
    class MockTokenizer:
        def __init__(self):
            self.eos_id = 1
        def encode(self, text, **kwargs):
            return [10, 20, 30, 40, 50]
        def decode(self, token_ids):
            return " " + " ".join(str(x) for x in token_ids)
        def apply_chat_template(self, msgs, **kwargs):
            return msgs[0]["content"]

    tokenizer = MockTokenizer()

    input_text = "Once upon a time in a faraway land,"
    input_ids = torch.tensor([tokenizer.encode(input_text)])
    
    # Warmup
    with torch.no_grad():
        _ = target_model.generate(input_ids, max_new_tokens=10, eos_token_id=1)

    max_tokens = 50

    # 2. Benchmark Autoregressive (Vanilla)
    start_time = time.time()
    with torch.no_grad():
        baseline_output = target_model.generate(
            input_ids, max_new_tokens=max_tokens, eos_token_id=1
        )
    baseline_time = time.time() - start_time
    baseline_tokens = baseline_output.size(1) - input_ids.size(1)
    baseline_tps = baseline_tokens / baseline_time

    # 3. Benchmark Speculative Decoding (Fixed Gamma=4)
    start_time = time.time()
    with torch.no_grad():
        spec_fixed_output = target_model.generate_speculative(
            input_ids,
            draft_model=draft_model,
            max_new_tokens=max_tokens,
            eos_token_id=1,
            gamma=4,
            adaptive_gamma=False
        )
    spec_fixed_time = time.time() - start_time
    spec_fixed_tokens = spec_fixed_output.size(1) - input_ids.size(1)
    spec_fixed_tps = spec_fixed_tokens / spec_fixed_time

    # 4. Benchmark Speculative Decoding (Adaptive Gamma)
    start_time = time.time()
    with torch.no_grad():
        spec_adaptive_output = target_model.generate_speculative(
            input_ids,
            draft_model=draft_model,
            max_new_tokens=max_tokens,
            eos_token_id=1,
            gamma=4,
            adaptive_gamma=True
        )
    spec_adaptive_time = time.time() - start_time
    spec_adaptive_tokens = spec_adaptive_output.size(1) - input_ids.size(1)
    spec_adaptive_tps = spec_adaptive_tokens / spec_adaptive_time

    return {
        "autoregressive": {
            "tokens_per_sec": baseline_tps,
            "latency_sec": baseline_time,
            "tokens_generated": baseline_tokens
        },
        "speculative_fixed": {
            "tokens_per_sec": spec_fixed_tps,
            "latency_sec": spec_fixed_time,
            "tokens_generated": spec_fixed_tokens,
            "speedup": spec_fixed_tps / baseline_tps
        },
        "speculative_adaptive": {
            "tokens_per_sec": spec_adaptive_tps,
            "latency_sec": spec_adaptive_time,
            "tokens_generated": spec_adaptive_tokens,
            "speedup": spec_adaptive_tps / baseline_tps
        }
    }

if __name__ == "__main__":
    results = run_speculative_benchmark()
    print(yaml.dump(results))
