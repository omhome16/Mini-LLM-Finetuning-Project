import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math

@dataclass
class ModelArgs:
    """
    Configuration for the custom Transformer model, mirroring Pythia-70M.
    """
    dim: int = 512  # Embedding dimension
    n_layers: int = 6  # Number of decoder blocks
    n_heads: int = 8  # Number of attention heads
    vocab_size: int = 50304  # Pythia-70m vocabulary size
    norm_eps: float = 1e-5  # Epsilon for RMSNorm
    rope_theta: float = 10000.0  # Theta for RoPE

    max_batch_size: int = 4  # Max batch size for training on 4GB VRAM
    max_seq_len: int = 512  # Max sequence length

    epochs: int = 3  # Total training epochs
    log_interval: int = 10  # Interval to print logs
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class RMSNorm(nn.Module):
    """
    Implements RMSNorm.
    Pythia use this instead of LayerNorm.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size (int): The dimension of the input tensor.
            eps (float): Epsilon for avoiding dividing by zero.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        rms = x.norm(2, dim=-1, keepdim=True) * (x.shape[-1] ** -0.5)
        return self.weight * (x / (rms + self.eps))


class RoPE(nn.Module):
    """
    Implements Rotary Positional Embeddings (RoPE).
    """

    def __init__(self, head_dim: int, rope_theta: float = 10000.0):
        """
        Args:
            head_dim (int): The dimension of each attention head.
            rope_theta (float): The base frequency for rotation.
        """
        super().__init__()
        self.head_dim = head_dim
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates the tensor by half its dimension."""
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Applies RoPE to the input tensor."""
        # Get positions and frequencies
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim/2)

        # Interleave frequencies
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1).repeat(1, x.size(1), 1, 1)  # (B, H, S, D_h)
        sin = emb.sin().unsqueeze(1).repeat(1, x.size(1), 1, 1)  # (B, H, S, D_h)

        # Apply rotation
        x_rotated = self.rotate_half(x)
        return (x * cos) + (x_rotated * sin)


