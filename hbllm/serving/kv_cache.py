"""
Enterprise-Grade KV Cache with Sliding Window Attention & 8-bit Key Quantization.

Features:
  - Pre-allocated O(1) buffer for autoregressive inference
  - Sliding Window Attention (SWA) for constant-VRAM long-context generation
  - Attention Sinks: preserves the first N tokens (critical for generation quality)
  - Optional 8-bit key quantization to further reduce memory footprint
"""


import torch


class KVCache:
    """
    Pre-allocated KV Cache with Sliding Window + Attention Sink support.

    When ``sliding_window`` is set, the cache acts as a rolling buffer:
    - The first ``attention_sinks`` positions are pinned (never evicted).
    - The remaining ``sliding_window - attention_sinks`` positions hold the
      most recent tokens and are shifted as new tokens arrive.
    - This guarantees O(1) memory regardless of total sequence length.

    When ``sliding_window`` is None, the cache behaves as a standard
    fixed-size buffer and raises ValueError if the sequence exceeds
    ``max_seq_len``.
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device('cpu'),
        quantize_k: bool = False,
        sliding_window: int | None = None,
        attention_sinks: int = 4,
        cpu_offload: bool = False,
    ):
        self.max_seq_len = max_seq_len
        self.device = device
        self.quantize_k = quantize_k
        self.dtype = dtype
        self.cpu_offload = cpu_offload

        # Sliding Window configuration
        self.sliding_window = sliding_window
        self.attention_sinks = attention_sinks if sliding_window else 0

        # Effective buffer size: use sliding_window if set, else max_seq_len
        self.buffer_size = sliding_window if sliding_window else max_seq_len

        # Validate constraints
        if self.sliding_window and self.attention_sinks >= self.sliding_window:
            raise ValueError(
                f"attention_sinks ({self.attention_sinks}) must be less than "
                f"sliding_window ({self.sliding_window})"
            )

        # Pre-allocate buffers: [batch_size, num_kv_heads, buffer_size, head_dim]
        shape = (batch_size, num_kv_heads, self.buffer_size, head_dim)

        if quantize_k:
            self.key_cache = torch.zeros(shape, dtype=torch.int8, device=device)
            self.key_scales = torch.zeros(
                (batch_size, num_kv_heads, self.buffer_size, 1),
                dtype=dtype, device=device,
            )
        else:
            self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
            self.key_scales = None

        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)

        # CPU Offload Tensors for holding evicted history beyond the VRAM sliding window
        if self.cpu_offload:
            cpu_shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
            if quantize_k:
                self.cpu_key_cache = torch.empty(cpu_shape, dtype=torch.int8, pin_memory=True)
                self.cpu_key_scales = torch.empty(
                    (batch_size, num_kv_heads, max_seq_len, 1), dtype=dtype, pin_memory=True
                )
            else:
                self.cpu_key_cache = torch.empty(cpu_shape, dtype=dtype, pin_memory=True)
                self.cpu_key_scales = None

            self.cpu_value_cache = torch.empty(cpu_shape, dtype=dtype, pin_memory=True)

        # Tracks current filled length in the buffer
        self.seq_len = 0

        # Tracks the total number of tokens that have passed through the cache
        # (may exceed buffer_size when sliding)
        self._total_tokens_seen = 0

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new keys/values and return the full cached history.

        When the sliding window is active and the buffer would overflow, this
        method evicts old tokens while preserving attention sinks.

        Args:
            keys:   [batch, num_kv_heads, new_len, head_dim]
            values: [batch, num_kv_heads, new_len, head_dim]
            seq_offset: Starting position in the logical sequence.

        Returns:
            (cached_keys, cached_values) sliced to current seq_len.
        """
        new_len = keys.shape[2]
        self._total_tokens_seen = max(self._total_tokens_seen, seq_offset + new_len)

        if self.sliding_window is not None:
            return self._update_sliding(keys, values, seq_offset, new_len)
        else:
            return self._update_static(keys, values, seq_offset, new_len)

    def _update_static(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_offset: int,
        new_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard fixed-buffer update without sliding window."""
        end_offset = seq_offset + new_len

        if end_offset > self.max_seq_len:
            raise ValueError(
                f"KV Cache exceeded: {end_offset} > {self.max_seq_len}. "
                f"Consider enabling sliding_window in ModelConfig."
            )

        self._write_to_cache(keys, values, seq_offset, end_offset)
        self.seq_len = max(self.seq_len, end_offset)

        return self._read_cache()

    def _update_sliding(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_offset: int,
        new_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sliding window update with attention sink preservation."""
        window = self.sliding_window
        sinks = self.attention_sinks

        # Phase 1: Determine if we need to evict
        projected_len = self.seq_len + new_len

        if projected_len <= window:
            # Buffer still has room — standard write
            write_start = self.seq_len
            write_end = self.seq_len + new_len
            self._write_to_cache(keys, values, write_start, write_end)
            self.seq_len = write_end
        else:
            # Buffer overflows — shift to maintain sinks + recent tokens
            recent_capacity = window - sinks

            if self.seq_len >= window:
                # Cache is already full: shift existing recent tokens left by new_len
                # Keep sinks[0:sinks] untouched
                # Shift recent region: move [sinks + new_len : window] → [sinks : window - new_len]
                shift_amount = min(new_len, recent_capacity)

                # Offload evicted tokens to CPU gracefully via PCIe
                if self.cpu_offload:
                    self._offload_evicted_to_cpu(sinks, shift_amount, new_len, window)

                if shift_amount < recent_capacity:
                    # Copy the surviving recent tokens
                    src_start = sinks + shift_amount
                    src_end = window
                    dst_start = sinks
                    dst_end = window - shift_amount

                    self.key_cache[:, :, dst_start:dst_end, :] = \
                        self.key_cache[:, :, src_start:src_end, :].clone()
                    self.value_cache[:, :, dst_start:dst_end, :] = \
                        self.value_cache[:, :, src_start:src_end, :].clone()

                    if self.quantize_k and self.key_scales is not None:
                        self.key_scales[:, :, dst_start:dst_end, :] = \
                            self.key_scales[:, :, src_start:src_end, :].clone()

                # Write new tokens at the end of the buffer
                write_start = window - shift_amount
                write_end = window
                actual_new = min(new_len, shift_amount)
                self._write_to_cache(
                    keys[:, :, -actual_new:, :],
                    values[:, :, -actual_new:, :],
                    write_start, write_end,
                )
                self.seq_len = window

            else:
                # Cache is partially filled but new tokens push it over
                # First fill remaining space
                remaining = window - self.seq_len
                if remaining > 0 and remaining < new_len:
                    # Write what fits directly
                    self._write_to_cache(
                        keys[:, :, :remaining, :],
                        values[:, :, :remaining, :],
                        self.seq_len,
                        window,
                    )
                    self.seq_len = window

                    # Now handle the overflow portion
                    overflow_keys = keys[:, :, remaining:, :]
                    overflow_values = values[:, :, remaining:, :]
                    overflow_len = overflow_keys.shape[2]
                    shift_amount = min(overflow_len, recent_capacity)

                    # Offload evicted tokens
                    if self.cpu_offload:
                        self._offload_evicted_to_cpu(sinks, shift_amount, overflow_len, window)

                    if shift_amount < recent_capacity:
                        src_start = sinks + shift_amount
                        src_end = window
                        dst_start = sinks
                        dst_end = window - shift_amount

                        self.key_cache[:, :, dst_start:dst_end, :] = \
                            self.key_cache[:, :, src_start:src_end, :].clone()
                        self.value_cache[:, :, dst_start:dst_end, :] = \
                            self.value_cache[:, :, src_start:src_end, :].clone()

                        if self.quantize_k and self.key_scales is not None:
                            self.key_scales[:, :, dst_start:dst_end, :] = \
                                self.key_scales[:, :, src_start:src_end, :].clone()

                    write_start = window - shift_amount
                    actual_new = min(overflow_len, shift_amount)
                    self._write_to_cache(
                        overflow_keys[:, :, -actual_new:, :],
                        overflow_values[:, :, -actual_new:, :],
                        write_start, window,
                    )
                else:
                    # new_len fits exactly or the initial fill is sufficient
                    self._write_to_cache(keys, values, self.seq_len, self.seq_len + new_len)
                    self.seq_len = self.seq_len + new_len

        return self._read_cache()

    def _write_to_cache(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        """Write keys/values into the buffer at [start:end], handling quantization."""
        if self.quantize_k:
            scale = keys.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5) / 127.0
            q_keys = (keys / scale).round().to(torch.int8)
            self.key_cache[:, :, start:end, :] = q_keys
            self.key_scales[:, :, start:end, :] = scale
        else:
            self.key_cache[:, :, start:end, :] = keys.to(self.dtype)

        self.value_cache[:, :, start:end, :] = values.to(self.dtype)

    def _read_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the currently valid portion of the cache, dequantizing if needed."""
        if self.quantize_k:
            k_slice = self.key_cache[:, :, :self.seq_len, :].to(self.dtype)
            s_slice = self.key_scales[:, :, :self.seq_len, :]
            final_keys = k_slice * s_slice
        else:
            final_keys = self.key_cache[:, :, :self.seq_len, :]

        return (final_keys, self.value_cache[:, :, :self.seq_len, :])

    @property
    def total_tokens_seen(self) -> int:
        """Total number of tokens processed (may exceed buffer_size)."""
        return self._total_tokens_seen

    def _offload_evicted_to_cpu(self, sinks: int, shift_amount: int, new_len: int, window: int) -> None:
        """Asynchronously stream evicted tokens to pinned CPU memory."""
        start_logical = self._total_tokens_seen - new_len - (window - sinks)
        if start_logical < 0 or start_logical >= self.max_seq_len:
            return

        end_logical = min(start_logical + shift_amount, self.max_seq_len)
        actual_shift = end_logical - start_logical

        evicted_k = self.key_cache[:, :, sinks : sinks + actual_shift, :]
        evicted_v = self.value_cache[:, :, sinks : sinks + actual_shift, :]

        # Non-blocking copy into pinned memory
        self.cpu_key_cache[:, :, start_logical:end_logical, :].copy_(evicted_k, non_blocking=True)
        self.cpu_value_cache[:, :, start_logical:end_logical, :].copy_(evicted_v, non_blocking=True)

        if self.quantize_k and self.cpu_key_scales is not None and self.key_scales is not None:
            evicted_scales = self.key_scales[:, :, sinks : sinks + actual_shift, :]
            self.cpu_key_scales[:, :, start_logical:end_logical, :].copy_(evicted_scales, non_blocking=True)

    def get_full_cache_cpu(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the fully assembled KV history sequence located on the CPU."""
        if not getattr(self, "cpu_offload", False):
            raise ValueError("cpu_offload must be enabled to retrieve the full cache.")

        active_len = self.seq_len
        sinks = self.attention_sinks
        total_tokens = min(self._total_tokens_seen, self.max_seq_len)

        # 1. Sync Attention Sinks to the beginning of the CPU cache
        if sinks > 0:
            self.cpu_key_cache[:, :, :sinks, :].copy_(self.key_cache[:, :, :sinks, :], non_blocking=True)
            self.cpu_value_cache[:, :, :sinks, :].copy_(self.value_cache[:, :, :sinks, :], non_blocking=True)
            if self.quantize_k and self.cpu_key_scales is not None and self.key_scales is not None:
                self.cpu_key_scales[:, :, :sinks, :].copy_(self.key_scales[:, :, :sinks, :], non_blocking=True)

        # 2. Sync the current active rolling window
        if active_len > sinks:
            active_start_logical = total_tokens - (active_len - sinks)
            self.cpu_key_cache[:, :, active_start_logical:total_tokens, :].copy_(
                self.key_cache[:, :, sinks:active_len, :], non_blocking=True
            )
            self.cpu_value_cache[:, :, active_start_logical:total_tokens, :].copy_(
                self.value_cache[:, :, sinks:active_len, :], non_blocking=True
            )
            if self.quantize_k and self.cpu_key_scales is not None and self.key_scales is not None:
                self.cpu_key_scales[:, :, active_start_logical:total_tokens, :].copy_(
                    self.key_scales[:, :, sinks:active_len, :], non_blocking=True
                )

        # Ensure all async copies to CPU finish before providing CPU tensor access
        if self.device != torch.device("cpu"):
            torch.cuda.current_stream(self.device).synchronize() if self.device.type == "cuda" else None

        return (
            self.cpu_key_cache[:, :, :total_tokens, :],
            self.cpu_value_cache[:, :, :total_tokens, :]
        )
