"""Rotary Position Embedding (RoPE) implementation.

Reference:
    https://github.com/LTH14/JiT/blob/main/util/model_util.py
"""

import torch
import torch.nn as nn
from einops import rearrange


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryEmbeddingFast(nn.Module):
    """Fast 1D Rotary Position Embedding for autoregressive generation.

    Precomputes all frequency matrices for efficient forward passes.
    Supports flexible position ordering through the rope_order parameter,
    enabling out-of-order token generation while maintaining correct spatial
    position information via fast table lookup.

    Args:
        dim: Embedding dimension (typically head_dim)
        max_seq_len: Maximum sequence length for precomputation (default: 4096)
        theta: Base for frequency computation (default: 10000)
    """

    def __init__(self, dim, max_seq_len=4096, theta=10000):
        super().__init__()

        # Compute frequency bases: 1 / (theta^(2i/d))
        # Each base corresponds to a dimension pair
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # Generate position indices
        t = torch.arange(max_seq_len).float()

        # Outer product: (max_seq_len, dim//2)
        freqs = torch.einsum("..., f -> ... f", t, freqs)

        # Duplicate each frequency twice: (max_seq_len, dim)
        # This matches the standard RoPE pattern where each dimension pair shares the same frequency
        freqs = freqs.repeat_interleave(2, dim=-1)

        # Precompute cos and sin for numerical stability
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, t, rope_order=None):
        """Apply rotary position embedding to input tensor.

        Args:
            t: Input tensor of shape (B, num_heads, seq_len, head_dim)
            rope_order: Optional tensor of position indices for each token
                       - Training: (B, N) specifying position for each token
                       - Sampling with KV cache: (B, 1) for the current token
                       - If None, uses sequential positions [0, 1, 2, ...]

        Returns:
            Tensor with rotary position embeddings applied
        """
        # Convert to fp32 for numerical stability
        input_dtype = t.dtype
        t = t.float()

        # Determine sequence length from input tensor
        # t shape: (B, num_heads, N, head_dim)
        N = t.shape[2]

        # Select appropriate RoPE frequencies based on rope_order
        if rope_order is not None:
            # rope_order specifies position indices for each token
            # rope_order shape:
            #   - Training: (B, N) where N is full sequence length
            #   - Sampling with KV cache: (B, 1) for the current token
            # Advanced indexing: freqs_cos[rope_order] outputs (B, N, head_dim) or (B, 1, head_dim)
            # Expand to (B, 1, N, head_dim) or (B, 1, 1, head_dim) for broadcasting
            rope_freqs_cos = self.freqs_cos[rope_order].unsqueeze(1)
            rope_freqs_sin = self.freqs_sin[rope_order].unsqueeze(1)
        else:
            # Standard mode: sequential positions from 0
            # freqs_cos[:N] outputs (N, head_dim)
            # Expand to (1, 1, N, head_dim) for broadcasting
            rope_freqs_cos = self.freqs_cos[:N].unsqueeze(0).unsqueeze(0)
            rope_freqs_sin = self.freqs_sin[:N].unsqueeze(0).unsqueeze(0)

        # Compute RoPE in fp32 and convert back to original dtype
        output = t * rope_freqs_cos + rotate_half(t) * rope_freqs_sin
        return output.to(input_dtype)
