"""
Speculative Decoding Engine for HBLLM.

Improves autoregressive generation latency by utilizing a small Draft Model
to propose K tokens, which are then verified in parallel by the Main Model
via Rejection Sampling to guarantee identical output distributions.

This implementation uses KV caching for both draft and main models to avoid
redundant recomputation of previous positions.
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Samples a token from logits using nucleus top-p sampling."""
    logits = logits / max(temperature, 1e-7)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift to keep the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)

    if temperature < 1e-3:
        return torch.argmax(probs, dim=-1, keepdim=True), probs

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token, probs


def _model_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past_key_values: list | None = None,
) -> tuple[torch.Tensor, list | None]:
    """
    Unified model forward that handles both dict and tuple return formats.

    Returns:
        (logits, past_key_values) — logits are [batch, seq, vocab],
        past_key_values may be None if the model doesn't support caching.
    """
    kwargs = {}
    if past_key_values is not None:
        kwargs["past_key_values"] = past_key_values
    kwargs["use_cache"] = True

    out = model(input_ids, **kwargs)

    if isinstance(out, dict):
        logits = out["logits"]
        pkv = out.get("past_key_values")
    elif isinstance(out, tuple):
        logits = out[0]
        pkv = out[1] if len(out) > 1 else None
    else:
        logits = out
        pkv = None

    return logits, pkv


def speculate_step(
    main_model: torch.nn.Module,
    draft_model: torch.nn.Module,
    draft_input_ids: torch.Tensor,
    main_input_ids: torch.Tensor,
    K: int = 4,
    temperature: float = 0.7,
    top_p: float = 0.9,
    draft_past_kv: list | None = None,
    main_past_kv: list | None = None,
) -> tuple[torch.Tensor, list | None, list | None]:
    """
    Performs one chunk step of speculative decoding with KV cache support.

    1. Draft model autoregressively proposes K tokens (using its KV cache).
    2. Main model verifies all K tokens in a single parallel forward pass.
    3. Rejection sampling ensures the output matches the exact target distribution.

    Args:
        main_model: The large target model.
        draft_model: The small draft/proposal model.
        draft_input_ids: Input token IDs for the draft model (just the new tokens
            if using KV cache, or full sequence if not).
        main_input_ids: Input token IDs for the main model (just the new tokens
            if using KV cache, or full sequence if not).
        K: Number of speculative tokens to draft.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        draft_past_kv: KV cache from the draft model's previous call.
        main_past_kv: KV cache from the main model's previous call.

    Returns:
        (accepted_tokens, updated_draft_past_kv, updated_main_past_kv)
    """
    draft_tokens = []
    draft_probs_list = []

    curr_input = draft_input_ids
    curr_draft_kv = draft_past_kv

    # ─── 1. Autoregressive Draft Generation (Fast) ────────────────────────
    with torch.no_grad():
        for _ in range(K):
            logits, curr_draft_kv = _model_forward(
                draft_model, curr_input, curr_draft_kv
            )

            next_token, probs = sample(
                logits[:, -1, :], temperature=temperature, top_p=top_p
            )
            draft_tokens.append(next_token)
            draft_probs_list.append(probs)

            # Next iteration only needs the new token (KV cache has the rest)
            curr_input = next_token

    draft_tensor = torch.cat(draft_tokens, dim=1)  # [batch, K]

    # ─── 2. Parallel Target Verification (One-Shot) ───────────────────────
    # Feed the original input + all draft tokens to the main model at once.
    # If we have a main_past_kv, we only need to feed new tokens.
    verify_input = torch.cat([main_input_ids, draft_tensor], dim=1)

    with torch.no_grad():
        main_logits, updated_main_kv = _model_forward(
            main_model, verify_input, main_past_kv
        )

    # Extract logits for the K draft positions + 1 bonus position
    target_logits = main_logits[:, -(K + 1):, :]

    # ─── 3. Acceptance / Rejection Sampling ───────────────────────────────
    accepted_tokens = []
    accepted_count = 0

    for i in range(K):
        # Draft probability for the proposed token
        p = draft_probs_list[i][0, draft_tensor[0, i]].item()

        # Main model true probability for the same token
        q_probs_all = F.softmax(
            target_logits[:, i, :] / max(temperature, 1e-7), dim=-1
        )
        q = q_probs_all[0, draft_tensor[0, i]].item()

        # Speculative random gate
        r = torch.rand(1).item()

        if r < min(1.0, q / max(p, 1e-7)):
            # Accept draft token
            accepted_tokens.append(draft_tensor[:, i : i + 1])
            accepted_count += 1
        else:
            # Reject — resample from corrected distribution to maintain
            # mathematical equivalence with the target distribution.
            diff_probs = torch.clamp(q_probs_all - draft_probs_list[i], min=0.0)
            if diff_probs.sum() > 0:
                diff_probs = diff_probs / diff_probs.sum(dim=-1, keepdim=True)
                resampled_token = torch.multinomial(diff_probs, 1)
            else:
                # Fallback for floating point precision collapse
                resampled_token = torch.argmax(q_probs_all, dim=-1, keepdim=True)

            accepted_tokens.append(resampled_token)
            accepted_count += 1
            break
    else:
        # All K tokens accepted — we earn one bonus token from the main model
        bonus_token, _ = sample(
            target_logits[:, -1, :], temperature=temperature, top_p=top_p
        )
        accepted_tokens.append(bonus_token)
        accepted_count += 1

    # ─── 4. Trim KV Caches to Match Accepted Length ───────────────────────
    # The draft KV cache currently contains all K speculative positions.
    # If we rejected early, we need to trim it to match only accepted tokens.
    if curr_draft_kv is not None and accepted_count < K:
        # Trim: we accepted `accepted_count` of the `K` draft tokens,
        # so we need to remove the last (K - accepted_count) positions.
        trim_amount = K - accepted_count
        trimmed_draft_kv = []
        for layer_kv in curr_draft_kv:
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                k, v = layer_kv
                trimmed_draft_kv.append((
                    k[:, :, :-trim_amount, :],
                    v[:, :, :-trim_amount, :],
                ))
            else:
                trimmed_draft_kv.append(layer_kv)
        curr_draft_kv = trimmed_draft_kv

    # Similarly trim the main model KV cache
    if updated_main_kv is not None and accepted_count < K:
        trim_amount = K - accepted_count
        trimmed_main_kv = []
        for layer_kv in updated_main_kv:
            if isinstance(layer_kv, (tuple, list)) and len(layer_kv) == 2:
                k, v = layer_kv
                trimmed_main_kv.append((
                    k[:, :, :-trim_amount, :],
                    v[:, :, :-trim_amount, :],
                ))
            else:
                trimmed_main_kv.append(layer_kv)
        updated_main_kv = trimmed_main_kv

    result_tokens = torch.cat(accepted_tokens, dim=1)

    logger.debug(
        "Speculative step: proposed %d, accepted %d (%.0f%% hit rate)",
        K, accepted_count, 100 * accepted_count / K,
    )

    return result_tokens, curr_draft_kv, updated_main_kv
