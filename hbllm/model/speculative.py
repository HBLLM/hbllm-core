"""
Speculative Decoding Engine for HBLLM.

Improves autoregressive generation latency by utilizing a small Draft Model
to propose K tokens, which are then verified in parallel by the Main Model
via Rejection Sampling to guarantee identical output distributions.
"""

import logging
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def sample(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 0.9) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples a token from logits using nucleus top-p sampling."""
    logits = logits / max(temperature, 1e-7)
    
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
    probs = F.softmax(logits, dim=-1)
    
    if temperature < 1e-3:
        return torch.argmax(probs, dim=-1, keepdim=True), probs
        
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token, probs

def speculate_step(
    main_model: torch.nn.Module, 
    draft_model: torch.nn.Module, 
    draft_input_ids: torch.Tensor, 
    main_input_ids: torch.Tensor, 
    K: int = 4, 
    temperature: float = 0.7, 
    top_p: float = 0.9
) -> torch.Tensor:
    """
    Performs one chunk step of speculative decoding.
    NOTE: KV-Cache passing is omitted here for code clarity during 
    prototype drafting, as adjusting sequence lengths of KV tensors 
    after rejection requires specialized buffer stripping.
    
    1. Draft model predicts K tokens.
    2. Main model forces all K tokens in a single parallel step.
    3. Output matches the exact target probability distribution.
    """
    draft_tokens = []
    draft_probs_list = []
    
    curr_input = draft_input_ids
    
    # 1. Autoregressive Draft Generation (Fast)
    with torch.no_grad():
        for _ in range(K):
            out = draft_model(curr_input)
            logits = out["logits"] if isinstance(out, dict) else out
            
            next_token, probs = sample(logits[:, -1, :], temperature=temperature, top_p=top_p)
            draft_tokens.append(next_token)
            draft_probs_list.append(probs)
            
            # Concat for the next draft loop (recomputing without KV cache for prototype safety)
            curr_input = torch.cat([curr_input, next_token], dim=1)
            
    draft_tensor = torch.cat(draft_tokens, dim=1)
    
    # 2. Parallel Target Generation (Slow but One-Shot)
    full_input = torch.cat([main_input_ids, draft_tensor], dim=1)
    
    with torch.no_grad():
        out_main = main_model(full_input)
        main_logits = out_main["logits"] if isinstance(out_main, dict) else out_main
        
    target_logits = main_logits[:, -(K + 1):, :] 
    
    accepted_tokens = []
    
    # 3. Acceptance/Rejection Sampling Equivalence
    for i in range(K):
        # Draft probability
        p = draft_probs_list[i][0, draft_tensor[0, i]].item()
        # Main model true probability
        q_probs_all = F.softmax(target_logits[:, i, :] / max(temperature, 1e-7), dim=-1)
        q = q_probs_all[0, draft_tensor[0, i]].item()
        
        # Speculative random gate
        r = torch.rand(1).item()
        
        if r < min(1.0, q / max(p, 1e-7)):
            # Accept draft token!
            accepted_tokens.append(draft_tensor[:, i:i+1])
        else:
            # Reject! Resample from corrected distribution bounds to maintain mathematical equivalency.
            diff_probs = torch.clamp(q_probs_all - draft_probs_list[i], min=0.0)
            if diff_probs.sum() > 0:
                diff_probs = diff_probs / diff_probs.sum(dim=-1, keepdim=True)
                resampled_token = torch.multinomial(diff_probs, 1)
            else:
                # Fallback in extreme floating point precision collapse
                resampled_token = torch.argmax(q_probs_all, dim=-1, keepdim=True)
                
            accepted_tokens.append(resampled_token)
            break
            
    else:
        # If all K tokens perfectly align, we extract one bonus token from the main_logits matrix!
        bonus_token, _ = sample(target_logits[:, -1, :], temperature=temperature, top_p=top_p)
        accepted_tokens.append(bonus_token)
        
    return torch.cat(accepted_tokens, dim=1)
