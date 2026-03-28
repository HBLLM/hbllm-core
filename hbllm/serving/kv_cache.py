import torch
from typing import Tuple

class KVCache:
    """
    Holds pre-allocated Key and Value tensors for autoregressive decoding.
    
    Prevents O(N) memory reallocation on every step by pre-allocating the max
    sequence length budget and performing in-place slicing updates.
    """
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device('cpu'),
    ):
        self.max_seq_len = max_seq_len
        self.device = device
        
        # [batch_size, num_kv_heads, max_seq_len, head_dim]
        shape = (batch_size, num_kv_heads, max_seq_len, head_dim)
        
        self.key_cache = torch.zeros(shape, dtype=dtype, device=device)
        self.value_cache = torch.zeros(shape, dtype=dtype, device=device)
        
        # Tracks current filled length
        self.seq_len = 0

    def update(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        seq_offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update the cache with new incoming keys/values.
        
        Args:
            keys: [batch, num_kv_heads, seq_len, head_dim]
            values: [batch, num_kv_heads, seq_len, head_dim]
            seq_offset: The sliding offset in the max_seq_len buffer
            
        Returns:
            The complete unpadded slice of historical keys and values up to the new seq_len.
        """
        new_len = keys.shape[2]
        end_offset = seq_offset + new_len
        
        if end_offset > self.max_seq_len:
            raise ValueError(
                f"KV Cache exceeded: requested length {end_offset} "
                f"> max_seq_len {self.max_seq_len}"
            )
            
        # In-place write
        self.key_cache[:, :, seq_offset:end_offset, :] = keys
        self.value_cache[:, :, seq_offset:end_offset, :] = values
        
        self.seq_len = max(self.seq_len, end_offset)
        
        # Return full history up to seq_len
        return (
            self.key_cache[:, :, :self.seq_len, :],
            self.value_cache[:, :, :self.seq_len, :]
        )
